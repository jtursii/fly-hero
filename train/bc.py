"""Behavior cloning training loop (PLAN Phase 3 task 6 / Phase 3b), model-
agnostic over ConnectomeBrainPolicy and GRUBaseline (both expose
init_state/reset_stream/step_frame -- see flyhero/brain/policy.py and
flyhero/brain/baseline_gru.py).

Burn-in (no_grad) then a gradient window, BCE loss on frets + strum
(pos_weight estimated from the actual note-aware sampler, not a fixed
guess), AdamW with separate LRs for readout vs. brain/CNN+GRU params,
grad-clip, optional per-frame gradient checkpointing (D26), periodic val
eval on a fixed note-dense excerpt per val song, curriculum advancement
(Easy->Medium->Hard->Expert at val hit_rate>=0.8, 20% of clips from earlier
difficulties once advanced), and resumability (--resume, checkpoint every N
steps or M seconds, plus on SIGTERM).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.tensorboard import SummaryWriter

from flyhero.brain.baseline_gru import GRUBaseline
from flyhero.brain.policy import ConnectomeBrainPolicy
from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta
from flyhero.brain.readout import Readout
from flyhero.game.clip_sampler import ClipSampler, list_available_song_ids
from flyhero.game.decoder import decode_trace, frets_to_held_mask
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.common import build_eval_clip, build_val_set, estimate_strum_pos_weight, make_training_clip
from train.watchdog import GradWatchdog, reconstruct_from_metrics

_stop_requested = False
_stop_signal_name: str | None = None


def _handle_stop_signal(signum, frame) -> None:
    global _stop_requested, _stop_signal_name
    _stop_requested = True
    _stop_signal_name = signal.Signals(signum).name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bc.yaml")
    p.add_argument("--game-config", default="configs/game.yaml")
    p.add_argument("--brain-config", default="configs/brain.yaml")
    p.add_argument("--model", choices=["connectome", "gru"], default="connectome")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--resume", default=None,
        help="a checkpoint file path, or the literal 'latest' combined with --run-dir to resolve "
        "<run-dir>/checkpoint_latest.pt",
    )
    p.add_argument("--run-dir", default=None, help="required when --resume latest is used")
    p.add_argument(
        "--init-from", default=None,
        help="seed model/optimizer/step/difficulty/pos_weight from this checkpoint file, but start a "
        "brand-new run dir (unlike --resume, which continues writing into the checkpoint's own run "
        "dir) -- e.g. forking a new run from a backup checkpoint while leaving the original run dir "
        "untouched. Mutually exclusive with --resume.",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--fixed-difficulty", default=None, help="disable curriculum, train/eval only this difficulty")
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument(
        "--debug-inject-rollback-at-step", type=int, default=None,
        help="TEST ONLY (D33): force one watchdog rollback (reason=injected_test) right after this step, to "
        "verify watchdog state survives stop/resume. Never use on a real run.",
    )
    args = p.parse_args()
    if args.resume == "latest":
        if not args.run_dir:
            p.error("--resume latest requires --run-dir <path>")
        args.resume = str(Path(args.run_dir) / "checkpoint_latest.pt")
    if args.resume and args.init_from:
        p.error("--resume and --init-from are mutually exclusive")
    return args


def existing_run_dirs(run_name: str, runs_root: Path = Path("runs")) -> list[Path]:
    """Run dirs named exactly <YYYYMMDD_HHMMSS>_<run_name> (not prefixes of
    longer names, e.g. bc_full_real vs bc_full_real_v2)."""
    import re

    pat = re.compile(r"^\d{8}_\d{6}_" + re.escape(run_name) + r"$")
    return sorted(d for d in runs_root.glob(f"*_{run_name}") if d.is_dir() and pat.match(d.name))


def build_policy(model_kind: str, cfg_bc: dict, cfg_game: dict, cfg_brain: dict, photo_map, device, dtype):
    if model_kind == "gru":
        model = GRUBaseline(frame_size=cfg_game["frame_size"]).to(device=device, dtype=dtype)
        return model

    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"], cfg_brain.get("graph_file", "graph.npz"))
    brain = ConnectomeBrain(graph, cfg_brain, device=device, dtype=dtype)
    readout = Readout(brain.n_dn, cfg_brain["readout"]["n_actions"]).to(device=device, dtype=dtype)
    retina_cfg = cfg_game["retina"]
    retina = TorchRetina.build(
        photo_map, cfg_game["frame_size"], retina_cfg["gain"], retina_cfg["ema_alpha"],
        retina_cfg["blur_sigma"], device,
    )
    return ConnectomeBrainPolicy(brain, retina, readout, cfg_brain["substeps_per_frame"])


def choose_difficulty(rng: np.random.Generator, difficulties: list[str], difficulty_idx: int, current_frac: float) -> str:
    if difficulty_idx == 0:
        return difficulties[0]
    if rng.random() < current_frac:
        return difficulties[difficulty_idx]
    earlier = difficulties[:difficulty_idx]
    return earlier[rng.integers(len(earlier))]


def run_clip(
    policy, env: VecRhythmEnv, clips, burn_in_frames: int, train_frames: int, device, dtype,
    with_grad: bool, gradient_checkpointing: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Runs burn_in_frames (always no_grad) then train_frames (with_grad
    controls whether they build a graph). Returns (final_state,
    logits_stack[B, train_frames, 6], max_abs_state_over_whole_clip) -- the
    last is the raw carried state's max magnitude (for ConnectomeBrain, this
    is literally max|v|; for GRUBaseline, the hidden state's max magnitude),
    tracked over both burn-in and the gradient window."""
    batch = len(clips)
    env.reset(clips)
    policy.reset_stream(batch)
    state = policy.init_state(batch, device, dtype)
    max_abs_state = 0.0

    with torch.no_grad():
        for _ in range(burn_in_frames):
            obs, _, _, _ = env.step(np.zeros((batch, 6), dtype=bool))
            frame_t = torch.from_numpy(obs["frame"]).to(device)
            state, _ = policy.step_frame(state, frame_t)
            cur = float(state.abs().max().item()); max_abs_state = cur if (cur != cur or cur > max_abs_state) else max_abs_state
    # state is already un-tracked (built under no_grad) -- passed into the
    # gradient window as a plain numeric starting point, like i_photo. We
    # never need gradients w.r.t. this specific carried-over value, only
    # w.r.t. the parameters used to process it forward from here, which
    # accumulate requires_grad naturally through the ops below.
    state = state.detach()

    logits_list = []
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        for _ in range(train_frames):
            obs, _, _, _ = env.step(np.zeros((batch, 6), dtype=bool))
            frame_t = torch.from_numpy(obs["frame"]).to(device)
            if with_grad and gradient_checkpointing:
                state, logits = grad_checkpoint(policy.step_frame, state, frame_t, use_reentrant=False)
            else:
                state, logits = policy.step_frame(state, frame_t)
            logits_list.append(logits)
            cur = float(state.detach().abs().max().item()); max_abs_state = cur if (cur != cur or cur > max_abs_state) else max_abs_state
    logits_stack = torch.stack(logits_list, dim=1)  # [B, T, 6]
    return state, logits_stack, max_abs_state


def score_eval_entry(
    policy, env: VecRhythmEnv, entry, burn_in_s: float, fps: int, hit_window_s: float, device, dtype,
) -> dict:
    """Runs one val entry (burn-in + its fixed excerpt) through the policy
    and scores it -- the single source of truth for "what is this
    checkpoint's hit_rate on this song excerpt", used by both
    evaluate_difficulty (aggregate eval, below) and scripts/
    render_gameplay_video.py (so a rendered clip's hit_rate is
    *structurally* guaranteed to match eval's number for the same
    checkpoint/song/excerpt, not just carefully kept in sync by hand).
    Returns metrics plus the raw probs/frets/strum/excerpt_notes/clip/
    burn_in_actual a caller can use for rendering."""
    clip, burn_in_actual = build_eval_clip(entry, burn_in_s)
    burn_in_frames = round(burn_in_actual * fps)
    excerpt_frames = round(entry.excerpt_s * fps)
    _, logits_stack, _ = run_clip(
        policy, env, [clip], burn_in_frames, excerpt_frames, device, dtype,
        with_grad=False, gradient_checkpointing=False,
    )
    probs = torch.sigmoid(logits_stack[0]).cpu().numpy()  # [T, 6]
    frets, strum = decode_trace(probs)
    held_mask = frets_to_held_mask(frets)

    # score against the notes actually inside the excerpt window
    # (frame-index-aligned with the excerpt, not the burn-in prefix).
    excerpt_notes = clip.notes[clip.notes["time_s"] >= burn_in_actual].copy()
    excerpt_notes["time_s"] -= burn_in_actual
    metrics, events = score_playthrough(excerpt_notes, held_mask, strum, fps, hit_window_s)
    return dict(
        metrics=metrics, events=events, probs=probs, frets=frets, strum=strum, held_mask=held_mask,
        excerpt_notes=excerpt_notes, clip=clip, burn_in_actual=burn_in_actual,
        burn_in_frames=burn_in_frames, excerpt_frames=excerpt_frames,
    )


def evaluate_difficulty(
    policy, env: VecRhythmEnv, val_entries, burn_in_s: float, fps: int, hit_window_s: float, device, dtype,
) -> dict:
    hit_rates, overstrums = [], []
    for entry in val_entries:
        result = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)
        hit_rates.append(result["metrics"]["hit_rate"])
        overstrums.append(result["metrics"]["overstrums_per_min"])
    return dict(
        hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
        overstrums_per_min=float(np.mean(overstrums)) if overstrums else 0.0,
        n_songs=len(val_entries),
    )


def apply_gradient_step(
    policy, optimizer, loss: torch.Tensor, grad_clip_norm: float, grad_outlier_clip: float, watchdog: GradWatchdog,
) -> dict:
    """Backward pass + NaN/outlier-safe gradient sanitization (D28, D30) +
    per-parameter-group clipping + the GradWatchdog skip decision --
    extracted from the training loop so it's unit-testable without any
    connectome/game data (tests/test_watchdog.py). Mutates policy's
    gradients, and (unless the step is skipped) its parameters via
    optimizer.step(). Assumes the fixed 2-group [readout, brain] optimizer
    layout built in main()/tests.

    D30 (docs/DECISIONS.md): offline diagnosis (scripts/
    diagnose_grad_explosion.py) found the known-unstable
    photoreceptor<->lamina pathway (R7, R8, L1, L5, one 2-neuron type;
    D28) passes through a *finite-but-astronomically-large*
    (1e5-1e14+) gradient regime before it literally overflows to inf --
    torch.nan_to_num alone doesn't catch that, and a single such entry
    dominates clip_grad_norm_'s combined norm, crushing every other
    parameter's real gradient to near zero on that step. Fixed here with:
    (1) an element-wise magnitude clamp in addition to non-finite zeroing,
    (2) separate clip_grad_norm_ calls for the readout vs. brain parameter
    groups (a brain-side blowup can no longer starve the readout), and
    (3) skipping the optimizer step entirely (via `watchdog`) when
    sanitization touched far more than the known pathway's entries, or the
    post-sanitize brain grad_norm is way above the recent rolling median."""
    is_nan_loss = bool(torch.isnan(loss).item())
    if is_nan_loss:
        optimizer.zero_grad()
        should_skip, skip_reason = watchdog.record_step(float("nan"), 0)
        return dict(
            is_nan_loss=True, n_bad_elems=0, grad_norm_brain=float("nan"), grad_norm_readout=float("nan"),
            should_skip=should_skip, skip_reason=skip_reason,
        )

    optimizer.zero_grad()
    loss.backward()
    readout_params = list(policy.readout_parameters())
    brain_params = list(policy.non_readout_parameters())
    n_bad_elems = 0
    for p in readout_params + brain_params:
        if p.grad is None:
            continue
        bad = ~torch.isfinite(p.grad)
        n_bad = int(bad.sum().item())
        if n_bad:
            n_bad_elems += n_bad
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        outlier = p.grad.abs() > grad_outlier_clip
        n_outlier = int(outlier.sum().item())
        if n_outlier:
            n_bad_elems += n_outlier
            p.grad = torch.clamp(p.grad, min=-grad_outlier_clip, max=grad_outlier_clip)

    grad_norm_readout = torch.nn.utils.clip_grad_norm_(readout_params, grad_clip_norm)
    grad_norm_brain = torch.nn.utils.clip_grad_norm_(brain_params, grad_clip_norm)
    should_skip, skip_reason = watchdog.record_step(float(grad_norm_brain.item()), n_bad_elems)
    if should_skip:
        optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()
    return dict(
        is_nan_loss=False, n_bad_elems=n_bad_elems,
        grad_norm_brain=float(grad_norm_brain.item()), grad_norm_readout=float(grad_norm_readout.item()),
        should_skip=should_skip, skip_reason=skip_reason,
    )


def perform_rollback(policy, optimizer: torch.optim.Optimizer, healthy_checkpoint_path: Path, watchdog: GradWatchdog, device) -> torch.optim.Optimizer:
    """Restores policy weights from healthy_checkpoint_path, halves the
    brain parameter group's LR, and rebuilds the optimizer from scratch
    (D30: Adam's moments may already be contaminated by the unhealthy
    stretch, so they're not carried across a rollback). Returns the new
    optimizer -- callers must rebind their local `optimizer` to it."""
    ckpt = torch.load(healthy_checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["model_state"])
    lr_readout = optimizer.param_groups[0]["lr"]
    lr_brain = optimizer.param_groups[1]["lr"] / 2.0
    new_optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.readout_parameters()), "lr": lr_readout},
            {"params": list(policy.non_readout_parameters()), "lr": lr_brain},
        ]
    )
    watchdog.note_rollback()
    return new_optimizer


def write_stopped_file(run_dir: Path, reason: str, step: int) -> None:
    (run_dir / "STOPPED.txt").write_text(f"stopped at step {step}: {reason}\n")


def main() -> None:
    args = parse_args()
    if args.init_from:
        # D33 guard: re-forking an existing run name (e.g. bc_full_real_v2
        # from backups/checkpoint_step13000.pt again) would silently throw
        # away everything that run trained past the fork point. Continuing a
        # run is --resume latest --run-dir <dir>.
        run_name_guard = args.run_name or f"bc_{args.model}"
        clash = existing_run_dirs(run_name_guard)
        if clash:
            raise SystemExit(
                f"--init-from refused: run name {run_name_guard!r} already exists ({clash[-1]}). "
                f"To continue it use: --resume latest --run-dir {clash[-1]}"
            )
    cfg_bc = load_config(args.config)
    cfg_game = load_config(args.game_config)
    cfg_brain = load_config(args.brain_config) if args.model == "connectome" else {}

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = getattr(torch, args.dtype)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)  # Ctrl+C behaves identically to SIGTERM: checkpoint, then exit

    seed = cfg_bc["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    processed_dir = Path(cfg_bc["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    graph_meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, graph_meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)

    policy = build_policy(args.model, cfg_bc, cfg_game, cfg_brain, photo_map, device, dtype)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.readout_parameters()), "lr": cfg_bc["lr_readout"]},
            {"params": list(policy.non_readout_parameters()), "lr": cfg_bc["lr_brain"]},
        ]
    )
    # D30/D31 (docs/DECISIONS.md): overnight-stability watchdog + the
    # magnitude bound used by apply_gradient_step's outlier sanitization.
    # D33: watchdog state is saved in every checkpoint and restored on
    # --resume (rollback count, baseline, best hit_rate); the halved brain
    # LR rides along in the optimizer state. Thresholds can be overridden
    # via an optional `watchdog:` config section (tests use a tiny window).
    watchdog = GradWatchdog(**cfg_bc.get("watchdog", {}))
    grad_outlier_clip = float(cfg_bc.get("grad_outlier_clip", 10000.0))

    difficulties = [args.fixed_difficulty] if args.fixed_difficulty else cfg_bc["curriculum"]["difficulties"]
    splits = json.loads((processed_dir / "splits.json").read_text())
    burn_in_s, train_window_s = cfg_bc["burn_in_s"], cfg_bc["train_window_s"]
    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    batch_size = cfg_bc["batch_size"]
    gradient_checkpointing = (
        args.gradient_checkpointing if args.gradient_checkpointing is not None else cfg_bc["gradient_checkpointing"]
    )

    samplers: dict[str, ClipSampler] = {}

    def get_sampler(difficulty: str) -> ClipSampler:
        if difficulty not in samplers:
            samplers[difficulty] = ClipSampler(
                processed_dir, splits["train"], difficulty, burn_in_s, train_window_s, fps,
                cfg_game["chord_merge_min_gap_s"], cfg_bc["frac_clips_with_notes"],
            )
        return samplers[difficulty]

    val_set = build_val_set(
        processed_dir, difficulties, cfg_bc["eval"]["n_val_songs"], cfg_bc["eval"]["excerpt_s"],
        cfg_game["chord_merge_min_gap_s"],
    )

    step = 0
    difficulty_idx = 0
    pos_weight = 1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt["step"]
        difficulty_idx = ckpt["difficulty_idx"]
        pos_weight = ckpt["pos_weight"]
        rng.bit_generator.state = ckpt["numpy_rng_state"]
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        print(f"resumed from {args.resume} at step={step} difficulty={difficulties[difficulty_idx]}")
        if ckpt.get("watchdog_state") is not None:
            watchdog.load_state_dict(ckpt["watchdog_state"])
            src = "checkpoint"
        else:
            # pre-D33 checkpoint: recover what metrics.jsonl records.
            rec = reconstruct_from_metrics(Path(args.resume).resolve().parent / "metrics.jsonl", step)
            watchdog.best_hit_rate = rec["best_hit_rate"]
            watchdog.rollback_count = rec["rollback_count"]
            src = "metrics.jsonl (pre-D33 checkpoint; rolling window re-established from scratch)"
        print(f"[watchdog] restored from {src}: rollback_count={watchdog.rollback_count} "
              f"healthy_baseline={watchdog.healthy_baseline} window_len={len(watchdog._recent)} "
              f"best_hit_rate={watchdog.best_hit_rate} lr_brain={optimizer.param_groups[1]['lr']:.3e}")
    elif args.init_from:
        # D30: fork a brand-new run from an arbitrary (e.g. backup)
        # checkpoint, unlike --resume which continues writing into the
        # checkpoint's own run dir. Optimizer state is intentionally NOT
        # loaded here (see main()'s run-dir setup below) -- D30's diagnosis
        # found the live gradient, not accumulated Adam state, was
        # contaminated, but a fresh optimizer is also simply the right
        # choice for "start a new run."
        ckpt = torch.load(args.init_from, map_location=device)
        policy.load_state_dict(ckpt["model_state"])
        step = ckpt["step"]
        difficulty_idx = ckpt["difficulty_idx"]
        pos_weight = ckpt["pos_weight"]
        print(f"initialized from {args.init_from} at step={step} difficulty={difficulties[difficulty_idx]} "
              f"pos_weight={pos_weight:.3f} (fresh optimizer, fresh run dir)")
        # D33: a fork keeps the lineage's best hit_rate (the eval-drop rule
        # compares against it); rollback count and grad-norm baseline start
        # fresh, since the fork has a fresh optimizer.
        if ckpt.get("watchdog_state") is not None:
            watchdog.best_hit_rate = dict(ckpt["watchdog_state"]["best_hit_rate"])
        else:
            watchdog.best_hit_rate = reconstruct_from_metrics(
                Path(args.init_from).resolve().parent / "metrics.jsonl", step,
            )["best_hit_rate"]
        print(f"[watchdog] fork inherits best_hit_rate={watchdog.best_hit_rate}")
    else:
        pos_weight = estimate_strum_pos_weight(
            get_sampler(difficulties[difficulty_idx]), fps, hit_window_s, train_window_s, n_samples=200, rng=rng,
        )
        print(f"initial strum pos_weight ({difficulties[difficulty_idx]}): {pos_weight:.3f}")

    if args.resume:
        # Continue writing into the same run dir (metrics.jsonl, tb/) rather
        # than starting a fresh one, so a resume doesn't split one run's
        # history across two directories.
        run_dir = Path(args.resume).resolve().parent
    else:
        # Fresh run dir -- either a genuinely new run, or a --init-from fork
        # (which needs its own dir precisely so the source checkpoint's run
        # dir stays untouched).
        run_name = args.run_name or f"bc_{args.model}"
        run_dir = make_run_dir(run_name, args.config, seed)
        for extra_cfg in (args.game_config, args.brain_config) if args.model == "connectome" else (args.game_config,):
            import shutil

            shutil.copy(extra_cfg, run_dir / Path(extra_cfg).name)
        (run_dir / "model_kind.txt").write_text(args.model + "\n")
        if args.init_from:
            (run_dir / "init_from.txt").write_text(f"{args.init_from}\n")
    metrics_path = run_dir / "metrics.jsonl"
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(f"run dir: {run_dir}")

    burn_in_frames = round(burn_in_s * fps)
    train_frames = round(train_window_s * fps)

    start_time = time.time()
    last_checkpoint_time = start_time
    step_times: list[float] = []
    nan_steps = 0
    window_steps = 0

    def do_checkpoint(reason: str) -> None:
        nonlocal last_checkpoint_time
        path = run_dir / f"checkpoint_step{step}.pt"
        from train.common import save_checkpoint

        save_checkpoint(
            path, policy, optimizer, step, difficulty_idx, pos_weight, rng, extra={"reason": reason},
            watchdog_state=watchdog.state_dict(),
        )
        latest = run_dir / "checkpoint_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
        last_checkpoint_time = time.time()
        print(f"checkpoint saved: {path} (reason={reason})")

    try:
        while True:
            if args.max_steps is not None and step >= args.max_steps:
                break
            if args.max_time_s is not None and (time.time() - start_time) >= args.max_time_s:
                break
            if _stop_requested:
                do_checkpoint(_stop_signal_name or "stop_signal")
                break

            t0 = time.time()
            current_frac = 1.0 if args.fixed_difficulty else (1.0 - cfg_bc["curriculum"]["earlier_frac"])
            clips, fret_bits_list, strum_list = [], [], []
            for _ in range(batch_size):
                d = choose_difficulty(rng, difficulties, difficulty_idx, current_frac)
                clip, fret_bits, strum = make_training_clip(
                    get_sampler(d), burn_in_s, train_window_s, fps, hit_window_s, rng,
                )
                clips.append(clip)
                fret_bits_list.append(fret_bits)
                strum_list.append(strum)

            fret_bits_t = torch.as_tensor(np.stack(fret_bits_list), device=device, dtype=dtype)
            strum_t = torch.as_tensor(np.stack(strum_list), device=device, dtype=dtype)

            _, logits_stack, max_abs_state = run_clip(
                policy, env, clips, burn_in_frames, train_frames, device, dtype,
                with_grad=True, gradient_checkpointing=gradient_checkpointing,
            )
            fret_loss = F.binary_cross_entropy_with_logits(logits_stack[..., :5], fret_bits_t)
            strum_loss = F.binary_cross_entropy_with_logits(
                logits_stack[..., 5], strum_t, pos_weight=torch.tensor(pos_weight, device=device, dtype=dtype),
            )
            loss = fret_loss + strum_loss

            # NaN/outlier-safe gradient sanitization + per-group clipping +
            # the skip decision (D28, strengthened by D30 -- see
            # apply_gradient_step's docstring for the full mechanism).
            step_result = apply_gradient_step(policy, optimizer, loss, cfg_bc["grad_clip_norm"], grad_outlier_clip, watchdog)
            is_nan_loss = step_result["is_nan_loss"]
            is_nan = step_result["n_bad_elems"] > 0 or is_nan_loss  # "step needed sanitization", distinct from "step skipped"
            if is_nan:
                nan_steps += 1

            step_dt = time.time() - t0
            step_times.append(step_dt)
            window_steps += 1

            step += 1

            # Sustained-instability rollback check (overnight safeguards,
            # D30/D31): every step, independent of the periodic-checkpoint/eval
            # cadence below, since a runaway rolling median shouldn't have to
            # wait for the next eval to be caught.
            if watchdog.should_rollback_from_steps():
                healthy_path = run_dir / "checkpoint_last_healthy.pt"
                if healthy_path.exists():
                    optimizer = perform_rollback(policy, optimizer, healthy_path, watchdog, device)
                    print(f"[watchdog] ROLLBACK #{watchdog.rollback_count} (sustained_grad_elevation): "
                          f"restored {healthy_path}, brain lr -> {optimizer.param_groups[1]['lr']:.2e}, optimizer moments reset")
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(dict(
                            step=step, time=time.time(), rollback=True, rollback_n=watchdog.rollback_count,
                            rollback_reason="sustained_grad_elevation", restored_from=str(healthy_path),
                            new_lr_brain=optimizer.param_groups[1]["lr"],
                        )) + "\n")
                    if watchdog.exhausted():
                        do_checkpoint("rollback_exhausted")
                        write_stopped_file(run_dir, "3 rollbacks exhausted (last reason=sustained_grad_elevation)", step)
                        break
                else:
                    print("[watchdog] sustained elevation detected but no checkpoint_last_healthy.pt yet -- continuing")

            if args.debug_inject_rollback_at_step is not None and step == args.debug_inject_rollback_at_step:
                healthy_path = run_dir / "checkpoint_last_healthy.pt"
                if not healthy_path.exists():
                    do_checkpoint("pre_injected_rollback")
                    healthy_path.symlink_to(f"checkpoint_step{step}.pt")
                optimizer = perform_rollback(policy, optimizer, healthy_path, watchdog, device)
                print(f"[watchdog] ROLLBACK #{watchdog.rollback_count} (injected_test): restored {healthy_path}, "
                      f"brain lr -> {optimizer.param_groups[1]['lr']:.2e}")
                with metrics_path.open("a") as f:
                    f.write(json.dumps(dict(
                        step=step, time=time.time(), rollback=True, rollback_n=watchdog.rollback_count,
                        rollback_reason="injected_test", restored_from=str(healthy_path),
                        new_lr_brain=optimizer.param_groups[1]["lr"],
                    )) + "\n")

            if step % 10 == 0:
                steps_per_s = len(step_times) / sum(step_times) if step_times else 0.0
                record = dict(
                    step=step, time=time.time(), difficulty=difficulties[difficulty_idx],
                    loss=float(loss.item()) if not is_nan_loss else float("nan"),
                    fret_loss=float(fret_loss.item()) if not is_nan_loss else float("nan"),
                    strum_loss=float(strum_loss.item()) if not is_nan_loss else float("nan"),
                    grad_norm=step_result["grad_norm_brain"], grad_norm_readout=step_result["grad_norm_readout"],
                    max_abs_state=max_abs_state, n_bad_elems=step_result["n_bad_elems"],
                    skip=step_result["should_skip"], skip_reason=step_result["skip_reason"],
                    nan_frac=nan_steps / window_steps, steps_per_s=steps_per_s, pos_weight=pos_weight,
                    rolling_median_grad_norm=watchdog.rolling_median(),
                    watchdog_skip_rate=watchdog.skip_count / max(watchdog.total_steps, 1),
                    rollback_count=watchdog.rollback_count,
                )
                with metrics_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                for k, v in record.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(k, v, step)
                step_times = step_times[-50:]

            if step % cfg_bc["checkpoint_interval_steps"] == 0 or (time.time() - last_checkpoint_time) >= cfg_bc["checkpoint_interval_s"]:
                do_checkpoint("interval")
                if watchdog.is_healthy_now():
                    healthy_path = run_dir / "checkpoint_last_healthy.pt"
                    if healthy_path.exists() or healthy_path.is_symlink():
                        healthy_path.unlink()
                    healthy_path.symlink_to(f"checkpoint_step{step}.pt")

            if step % cfg_bc["eval"]["interval_steps"] == 0:
                eval_t0 = time.time()
                current_difficulty = difficulties[difficulty_idx]
                result = evaluate_difficulty(
                    policy, env, val_set[current_difficulty], burn_in_s, fps, hit_window_s, device, dtype,
                )
                eval_wall_s = time.time() - eval_t0
                record = dict(
                    step=step, time=time.time(), eval_difficulty=current_difficulty, eval_hit_rate=result["hit_rate"],
                    eval_overstrums_per_min=result["overstrums_per_min"], eval_n_songs=result["n_songs"],
                    eval_wall_s=eval_wall_s,
                )
                with metrics_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                tb_writer.add_scalar("eval_hit_rate", result["hit_rate"], step)
                tb_writer.add_scalar("eval_overstrums_per_min", result["overstrums_per_min"], step)
                print(
                    f"[eval step={step}] difficulty={current_difficulty} hit_rate={result['hit_rate']:.4f} "
                    f"overstrums/min={result['overstrums_per_min']:.2f} wall={eval_wall_s:.1f}s"
                )

                # Overnight safeguards (D30/D31): track the best checkpoint at
                # this difficulty, and roll back if hit_rate has cratered.
                prev_best = watchdog.best_hit_rate.get(current_difficulty)
                should_rollback_eval = watchdog.record_eval(current_difficulty, result["hit_rate"])
                is_new_best = prev_best is None or result["hit_rate"] > prev_best
                if is_new_best:
                    best_path, ckpt_path = run_dir / "checkpoint_best.pt", run_dir / f"checkpoint_step{step}.pt"
                    if ckpt_path.exists():
                        if best_path.exists() or best_path.is_symlink():
                            best_path.unlink()
                        best_path.symlink_to(ckpt_path.name)
                if should_rollback_eval:
                    healthy_path = run_dir / "checkpoint_last_healthy.pt"
                    if healthy_path.exists():
                        optimizer = perform_rollback(policy, optimizer, healthy_path, watchdog, device)
                        print(f"[watchdog] ROLLBACK #{watchdog.rollback_count} (eval_hit_rate_drop): "
                              f"restored {healthy_path}, brain lr -> {optimizer.param_groups[1]['lr']:.2e}, optimizer moments reset")
                        with metrics_path.open("a") as f:
                            f.write(json.dumps(dict(
                                step=step, time=time.time(), rollback=True, rollback_n=watchdog.rollback_count,
                                rollback_reason="eval_hit_rate_drop", restored_from=str(healthy_path),
                                new_lr_brain=optimizer.param_groups[1]["lr"],
                            )) + "\n")
                        if watchdog.exhausted():
                            do_checkpoint("rollback_exhausted")
                            write_stopped_file(run_dir, "3 rollbacks exhausted (last reason=eval_hit_rate_drop)", step)
                            break
                    else:
                        print("[watchdog] eval hit_rate drop detected but no checkpoint_last_healthy.pt yet -- continuing")

                if not args.fixed_difficulty and result["hit_rate"] >= cfg_bc["curriculum"]["advance_hit_rate"] and difficulty_idx < len(difficulties) - 1:
                    difficulty_idx += 1
                    new_difficulty = difficulties[difficulty_idx]
                    pos_weight = estimate_strum_pos_weight(
                        get_sampler(new_difficulty), fps, hit_window_s, train_window_s, n_samples=200, rng=rng,
                    )
                    print(f"[curriculum] advanced to {new_difficulty}, new strum pos_weight={pos_weight:.3f}")
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(dict(step=step, time=time.time(), curriculum_advanced_to=new_difficulty, new_pos_weight=pos_weight)) + "\n")

    except Exception as exc:
        print(f"[watchdog] CRASH: {exc!r}")
        try:
            do_checkpoint("crash")
        except Exception as ckpt_exc:
            print(f"[watchdog] checkpoint-on-crash also failed: {ckpt_exc!r}")
        write_stopped_file(run_dir, f"crash: {exc!r}", step)
        raise
    do_checkpoint("final")
    tb_writer.close()
    print(f"done. total steps={step}, elapsed={time.time() - start_time:.1f}s, run dir={run_dir}")


if __name__ == "__main__":
    main()
