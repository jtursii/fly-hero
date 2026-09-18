"""Behavior cloning training loop (PLAN Phase 3 task 6 / Phase 3b), model-
agnostic over ConnectomeBrainPolicy and GRUBaseline (both expose
init_state/reset_stream/step_frame -- see flyhero/brain/policy.py and
flyhero/brain/baseline_gru.py).

Burn-in (no_grad) then a gradient window, BCE loss on frets + strum
(pos_weight estimated from the actual note-aware sampler, not a fixed
guess), AdamW with separate LRs for readout vs. brain/CNN+GRU params,
grad-clip, optional per-frame gradient checkpointing (D26), periodic val
eval on a fixed note-dense excerpt per val song, curriculum advancement
(Easy->Medium->Hard->Expert when the 3-eval moving average of val hit_rate
reaches advance_hit_rate (D36), 20% of clips from earlier difficulties once
advanced), an optional cosine LR decay (D36), and resumability (--resume,
checkpoint every N steps or M seconds, plus on SIGTERM).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
from flyhero.game.decoder import FRET_THRESHOLD, STRUM_THRESHOLD, decode_trace, frets_to_held_mask
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from flyhero.utils.run import git_sha, make_run_dir
from train.common import build_eval_clip, build_val_set, estimate_strum_pos_weight, make_training_clip
from train.watchdog import GradWatchdog, load_metrics_lineage, reconstruct_from_metrics

_stop_requested = False
_stop_signal_name: str | None = None


def _handle_stop_signal(signum, frame) -> None:
    global _stop_requested, _stop_signal_name
    _stop_requested = True
    _stop_signal_name = signal.Signals(signum).name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", default=None,
        help="default: on --resume, the config recorded in <run-dir>/train_config.txt (else configs/bc.yaml)",
    )
    p.add_argument("--game-config", default="configs/game.yaml")
    p.add_argument("--brain-config", default="configs/brain.yaml")
    p.add_argument("--model", choices=["connectome", "gru"], default="connectome")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--resume", default=None,
        help="a checkpoint file path, or the literal 'latest' / 'best' combined with --run-dir to resolve "
        "<run-dir>/checkpoint_latest.pt / checkpoint_best.pt",
    )
    p.add_argument("--run-dir", default=None, help="required with --resume latest/best")
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
    p.add_argument(
        "--set-lr-brain", type=float, default=None,
        help="ONE-SHOT, with --resume: replace the brain group's base LR (the value the LR schedule scales and "
        "rollbacks halve). Saved in the optimizer state, so don't repeat it on later resumes.",
    )
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument(
        "--debug-inject-rollback-at-step", type=int, default=None,
        help="TEST ONLY (D33): force one watchdog rollback (reason=injected_test) right after this step, to "
        "verify watchdog state survives stop/resume. Never use on a real run.",
    )
    args = p.parse_args()
    if args.resume in ("latest", "best"):
        if not args.run_dir:
            p.error(f"--resume {args.resume} requires --run-dir <path>")
        args.resume = str(Path(args.run_dir) / f"checkpoint_{args.resume}.pt")
    if args.resume and args.init_from:
        p.error("--resume and --init-from are mutually exclusive")
    if args.set_lr_brain is not None and not args.resume:
        p.error("--set-lr-brain only applies with --resume")
    if args.config is None:
        recorded = Path(args.resume).resolve().parent / "train_config.txt" if args.resume else None
        args.config = recorded.read_text().strip() if recorded and recorded.exists() else "configs/bc.yaml"
    return args


def lr_factor(step: int, schedule: dict | None) -> float:
    """D36: cosine decay from 1 at schedule.start_step to final_frac at
    start_step + duration_steps, flat outside that range. No schedule -> 1."""
    if not schedule:
        return 1.0
    t = (step - schedule["start_step"]) / schedule["duration_steps"]
    t = min(max(t, 0.0), 1.0)
    final = schedule["final_frac"]
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * t))


def apply_lr_schedule(optimizer: torch.optim.Optimizer, step: int, schedule: dict | None) -> None:
    """lr = base_lr * lr_factor(step) for every param group. base_lr lives in
    the param group itself, so it's saved/restored with the optimizer state."""
    f = lr_factor(step, schedule)
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = group["base_lr"] * f


def should_advance_curriculum(watchdog: GradWatchdog, difficulty: str, advance_hit_rate: float) -> bool:
    """D36: advance when the eval_ma_window-eval moving average at the current
    difficulty reaches advance_hit_rate (not a single eval)."""
    ma = watchdog.eval_ma(difficulty)
    return ma is not None and ma >= advance_hit_rate


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
    with_grad: bool, gradient_checkpointing: bool, tbptt_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Runs burn_in_frames (always no_grad) then train_frames (with_grad
    controls whether they build a graph). tbptt_frames (D40): if set, the
    carried state is detached every tbptt_frames frames of the gradient
    window, so no gradient flows back further than that (truncated BPTT);
    the forward pass and the loss are unchanged. Returns (final_state,
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
        for i in range(train_frames):
            if tbptt_frames and i > 0 and i % tbptt_frames == 0:
                state = state.detach()
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
    from eval.compare import CATEGORIES, MISS_MODES, analyze_excerpt  # lazy: eval.compare imports this module

    hit_rates, overstrums = [], []
    by_cat = {c: dict(n=0, hits=0, miss_modes=dict.fromkeys(MISS_MODES, 0)) for c in CATEGORIES}
    for entry in val_entries:
        result = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)
        hit_rates.append(result["metrics"]["hit_rate"])
        overstrums.append(result["metrics"]["overstrums_per_min"])
        # D39's failure-mode breakdown at the default decoder thresholds,
        # summed over excerpts (D40's per-eval single-note miss counters).
        a = analyze_excerpt(result["probs"], result["excerpt_notes"], fps, hit_window_s, STRUM_THRESHOLD, FRET_THRESHOLD)
        for c, d in a["by_category"].items():
            by_cat[c]["n"] += d["n"]
            by_cat[c]["hits"] += d["hits"]
            for m, k in d["miss_modes"].items():
                by_cat[c]["miss_modes"][m] += k
    return dict(
        hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
        overstrums_per_min=float(np.mean(overstrums)) if overstrums else 0.0,
        n_songs=len(val_entries), by_category=by_cat,
    )


def apply_gradient_step(
    policy, optimizer, loss: torch.Tensor, grad_clip_norm: float, grad_outlier_clip: float, watchdog: GradWatchdog,
    probe_type_idx: torch.Tensor | None = None,
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
    post-sanitize brain grad_norm is way above the recent rolling median.

    probe_type_idx (D40): cell-type indices whose *raw* (pre-sanitization)
    tau/bias gradient magnitude is returned as `probe_raw_grad` [n] (max of
    |d tau|, |d bias| per type; inf if either is non-finite) -- the direct
    "does this type get a finite gradient" readout for R7/R8/L1/L5/AN_multi_1."""
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
    probe_raw_grad = None
    if probe_type_idx is not None:
        brain = policy.brain
        g = torch.stack([
            (brain.tau.grad if brain.tau.grad is not None else torch.zeros_like(brain.tau))[probe_type_idx],
            (brain.bias.grad if brain.bias.grad is not None else torch.zeros_like(brain.bias))[probe_type_idx],
        ]).detach().cpu().double().abs()  # [2, n]
        g[~torch.isfinite(g)] = float("inf")
        probe_raw_grad = g.max(dim=0).values.tolist()
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
        should_skip=should_skip, skip_reason=skip_reason, probe_raw_grad=probe_raw_grad,
    )


# D28/D30's unstable photoreceptor<->lamina pathway: never one finite
# gradient under full-window BPTT (all-zero Adam moments in v2, D32).
WATCHED_TYPES = ("R7", "R8", "L1", "L5", "AN_multi_1")


def brain_drift_stats(policy, ref: dict, optimizer: torch.optim.Optimizer, watched: dict[str, int]) -> dict:
    """D40 pass criteria, logged with each eval: how far the brain has moved
    from `ref` (the fork's init checkpoint: brain.g/tau/bias tensors), and
    whether the watched types have any Adam moment / have left their tau."""
    brain = policy.brain
    with torch.no_grad():
        sg, sg0 = F.softplus(brain.g), F.softplus(ref["g"])  # [n_gain_groups]
        out = dict(
            drift_g_rel=float((sg - sg0).norm() / sg0.norm()),
            drift_g_max_rel=float(((sg - sg0).abs() / sg0).max()),
            drift_tau_rel=float((brain.tau - ref["tau"]).norm() / ref["tau"].norm()),
            drift_bias_rel=float((brain.bias - ref["bias"]).norm() / ref["bias"].norm()),
        )
        idx = torch.tensor(list(watched.values()), device=brain.tau.device)
        moments = []
        for p in (brain.tau, brain.bias):
            st = optimizer.state.get(p, {})
            m = st.get("exp_avg")
            moments.append(m[idx].abs() if m is not None else torch.zeros(len(idx), device=idx.device))
        m = torch.stack(moments).max(dim=0).values.tolist()  # [n_watched]
        tau = brain.tau[idx].tolist()
        for (name, _), mi, ti in zip(watched.items(), m, tau):
            out[f"watch_{name}_adam_m"] = mi
            out[f"watch_{name}_tau"] = ti
    return out


def perform_rollback(policy, optimizer: torch.optim.Optimizer, healthy_checkpoint_path: Path, watchdog: GradWatchdog, device) -> torch.optim.Optimizer:
    """Restores policy weights from healthy_checkpoint_path, halves the
    brain parameter group's LR (its base_lr, which the D36 schedule scales),
    and rebuilds the optimizer from scratch
    (D30: Adam's moments may already be contaminated by the unhealthy
    stretch, so they're not carried across a rollback). Returns the new
    optimizer -- callers must rebind their local `optimizer` to it."""
    ckpt = torch.load(healthy_checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["model_state"])
    g_readout, g_brain = optimizer.param_groups[0], optimizer.param_groups[1]
    base_readout = g_readout.get("base_lr", g_readout["lr"])
    base_brain = g_brain.get("base_lr", g_brain["lr"]) / 2.0
    new_optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.readout_parameters()), "lr": g_readout["lr"], "base_lr": base_readout},
            {"params": list(policy.non_readout_parameters()), "lr": g_brain["lr"] / 2.0, "base_lr": base_brain},
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
            {"params": list(policy.readout_parameters()), "lr": cfg_bc["lr_readout"], "base_lr": cfg_bc["lr_readout"]},
            {"params": list(policy.non_readout_parameters()), "lr": cfg_bc["lr_brain"], "base_lr": cfg_bc["lr_brain"]},
        ]
    )
    lr_schedule = cfg_bc.get("lr_schedule")
    # D30/D31 (docs/DECISIONS.md): overnight-stability watchdog + the
    # magnitude bound used by apply_gradient_step's outlier sanitization.
    # D33: watchdog state is saved in every checkpoint and restored on
    # --resume (rollback count, baseline, best hit_rate); the halved brain
    # LR rides along in the optimizer state. Thresholds can be overridden
    # via an optional `watchdog:` config section (tests use a tiny window).
    watchdog_cfg = dict(cfg_bc.get("watchdog", {}))
    watchdog = GradWatchdog(**watchdog_cfg)
    eval_kwargs = {k: v for k, v in watchdog_cfg.items() if k in ("eval_ma_window", "rollback_eval_drop")}
    grad_outlier_clip = float(cfg_bc.get("grad_outlier_clip", 10000.0))

    difficulties = [args.fixed_difficulty] if args.fixed_difficulty else cfg_bc["curriculum"]["difficulties"]
    splits = json.loads((processed_dir / "splits.json").read_text())
    burn_in_s, train_window_s = cfg_bc["burn_in_s"], cfg_bc["train_window_s"]
    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    batch_size = cfg_bc["batch_size"]
    gradient_checkpointing = (
        args.gradient_checkpointing if args.gradient_checkpointing is not None else cfg_bc["gradient_checkpointing"]
    )
    # D40 (bc_trunc_bptt): both default to the pre-D40 behavior when absent.
    tbptt_frames = cfg_bc.get("tbptt_frames")
    fret_label_onset = cfg_bc.get("fret_label_onset", "segment")
    print(f"tbptt_frames={tbptt_frames} fret_label_onset={fret_label_onset}")

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
        for group in optimizer.param_groups:  # pre-D36 checkpoints have no base_lr
            group.setdefault("base_lr", group["lr"])
        if args.set_lr_brain is not None:
            print(f"[lr] brain base_lr {optimizer.param_groups[1]['base_lr']:.3e} -> {args.set_lr_brain:.3e} (--set-lr-brain)")
            optimizer.param_groups[1]["base_lr"] = args.set_lr_brain
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
            rec = reconstruct_from_metrics(Path(args.resume).resolve().parent / "metrics.jsonl", step, **eval_kwargs)
            watchdog.best_hit_rate = rec["best_hit_rate"]
            watchdog.rollback_count = rec["rollback_count"]
            watchdog.eval_history = rec["eval_history"]
            watchdog.best_eval_ma = rec["best_eval_ma"]
            src = "metrics.jsonl (pre-D33 checkpoint; rolling window re-established from scratch)"
        print(f"[watchdog] restored from {src}: rollback_count={watchdog.rollback_count} "
              f"healthy_baseline={watchdog.healthy_baseline} window_len={len(watchdog._recent)} "
              f"best_hit_rate={watchdog.best_hit_rate} eval_history={watchdog.eval_history} "
              f"best_eval_ma={watchdog.best_eval_ma} base_lr_brain={optimizer.param_groups[1]['base_lr']:.3e} "
              f"base_lr_readout={optimizer.param_groups[0]['base_lr']:.3e}")
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
                Path(args.init_from).resolve().parent / "metrics.jsonl", step, **eval_kwargs,
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
            shutil.copy(extra_cfg, run_dir / Path(extra_cfg).name)
        (run_dir / "model_kind.txt").write_text(args.model + "\n")
        if args.init_from:
            (run_dir / "init_from.txt").write_text(f"{args.init_from}\n")
    metrics_path = run_dir / "metrics.jsonl"
    # Later bare `--resume` calls (fly.sh resume / guard) reuse this config.
    (run_dir / "train_config.txt").write_text(f"{args.config}\n")
    if args.resume:
        # Snapshot of what this resume ran with (the run dir's bc.yaml is the
        # launch-time copy).
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(args.config, run_dir / f"resume_{stamp}_step{step}_{Path(args.config).name}")
        resolved = Path(args.resume).resolve()
        latest = run_dir / "checkpoint_latest.pt"
        latest_step = max([step] + [r.get("step", 0) for r in load_metrics_lineage(metrics_path)])
        if latest.exists() and latest.resolve().stem.startswith("checkpoint_step"):
            latest_step = max(latest_step, int(latest.resolve().stem.removeprefix("checkpoint_step")))
        with metrics_path.open("a") as f:
            f.write(json.dumps(dict(
                step=step, time=time.time(), resumed_from_step=step, resumed_from=resolved.name,
                config=args.config, git_sha=git_sha(), set_lr_brain=args.set_lr_brain,
                discarded_after_step=latest_step if latest_step > step else None,
            )) + "\n")
        if not (latest.exists() and latest.resolve() == resolved):
            # Resuming from an older checkpoint (e.g. --resume best): later
            # steps are an abandoned branch; a crash auto-resume before the
            # next checkpoint must come back here, not to that branch.
            print(f"[resume] checkpoint_latest.pt -> {resolved.name} "
                  f"(steps {step + 1}-{latest_step} abandoned; files kept)")
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(resolved.name)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(f"run dir: {run_dir}")

    # D40: drift reference = the fork's init checkpoint (recorded in
    # init_from.txt, so a --resume of the fork keeps the same reference),
    # plus the watched unstable types' gradient probe.
    drift_ref, watched, probe_idx = None, {}, None
    if args.model == "connectome":
        type_names = graph_meta["type_names"]
        watched = {n: type_names.index(n) for n in WATCHED_TYPES if n in type_names}
        probe_idx = torch.tensor(list(watched.values()), device=device)
        ref_file = run_dir / "init_from.txt"
        if ref_file.exists():
            ref_state = torch.load(ref_file.read_text().strip(), map_location=device)["model_state"]
            drift_ref = {k: ref_state[f"brain.{k}"].to(device=device, dtype=dtype) for k in ("g", "tau", "bias")}
            print(f"[drift] reference: {ref_file.read_text().strip()}")
    probe_finite = dict.fromkeys(watched, 0)
    probe_steps = 0

    burn_in_frames = round(burn_in_s * fps)
    train_frames = round(train_window_s * fps)

    start_time = time.time()
    last_checkpoint_time = start_time
    step_times: list[float] = []
    nan_steps = 0
    window_steps = 0

    saved_step = None

    def do_checkpoint(reason: str) -> None:
        nonlocal last_checkpoint_time, saved_step
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
        saved_step = step
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
            apply_lr_schedule(optimizer, step, lr_schedule)
            current_frac = 1.0 if args.fixed_difficulty else (1.0 - cfg_bc["curriculum"]["earlier_frac"])
            clips, fret_bits_list, strum_list = [], [], []
            for _ in range(batch_size):
                d = choose_difficulty(rng, difficulties, difficulty_idx, current_frac)
                clip, fret_bits, strum = make_training_clip(
                    get_sampler(d), burn_in_s, train_window_s, fps, hit_window_s, rng, fret_label_onset,
                )
                clips.append(clip)
                fret_bits_list.append(fret_bits)
                strum_list.append(strum)

            fret_bits_t = torch.as_tensor(np.stack(fret_bits_list), device=device, dtype=dtype)
            strum_t = torch.as_tensor(np.stack(strum_list), device=device, dtype=dtype)

            _, logits_stack, max_abs_state = run_clip(
                policy, env, clips, burn_in_frames, train_frames, device, dtype,
                with_grad=True, gradient_checkpointing=gradient_checkpointing, tbptt_frames=tbptt_frames,
            )
            fret_loss = F.binary_cross_entropy_with_logits(logits_stack[..., :5], fret_bits_t)
            strum_loss = F.binary_cross_entropy_with_logits(
                logits_stack[..., 5], strum_t, pos_weight=torch.tensor(pos_weight, device=device, dtype=dtype),
            )
            loss = fret_loss + strum_loss

            # NaN/outlier-safe gradient sanitization + per-group clipping +
            # the skip decision (D28, strengthened by D30 -- see
            # apply_gradient_step's docstring for the full mechanism).
            step_result = apply_gradient_step(
                policy, optimizer, loss, cfg_bc["grad_clip_norm"], grad_outlier_clip, watchdog, probe_idx,
            )
            if step_result.get("probe_raw_grad") is not None:
                probe_steps += 1
                for name, gmax in zip(watched, step_result["probe_raw_grad"]):
                    probe_finite[name] += int(0.0 < gmax < float("inf"))
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
                    lr_readout=optimizer.param_groups[0]["lr"], lr_brain=optimizer.param_groups[1]["lr"],
                    watchdog_baseline=watchdog.healthy_baseline, long_median_grad_norm=watchdog.long_median(),
                    tbptt_frames=tbptt_frames,
                )
                if step_result.get("probe_raw_grad") is not None:
                    for name, gmax in zip(watched, step_result["probe_raw_grad"]):
                        record[f"watch_{name}_raw_grad"] = gmax if gmax < float("inf") else "inf"
                        record[f"watch_{name}_finite_frac"] = probe_finite[name] / max(probe_steps, 1)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                for k, v in record.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(k, v, step)
                step_times = step_times[-50:]

            if step % cfg_bc["eval"]["interval_steps"] == 0:
                eval_t0 = time.time()
                current_difficulty = difficulties[difficulty_idx]
                result = evaluate_difficulty(
                    policy, env, val_set[current_difficulty], burn_in_s, fps, hit_window_s, device, dtype,
                )
                eval_wall_s = time.time() - eval_t0
                # Overnight safeguards (D30/D31/D35): track the best checkpoint
                # at this difficulty, and roll back if the eval moving average
                # has fallen.
                prev_best = watchdog.best_hit_rate.get(current_difficulty)
                should_rollback_eval = watchdog.record_eval(current_difficulty, result["hit_rate"])
                eval_ma = watchdog.eval_ma(current_difficulty)
                record = dict(
                    step=step, time=time.time(), eval_difficulty=current_difficulty, eval_hit_rate=result["hit_rate"],
                    eval_overstrums_per_min=result["overstrums_per_min"], eval_n_songs=result["n_songs"],
                    eval_wall_s=eval_wall_s, eval_ma=eval_ma,
                    best_eval_ma=watchdog.best_eval_ma.get(current_difficulty),
                )
                for cat, d in result["by_category"].items():
                    if d["n"]:
                        record[f"eval_{cat}_hit_rate"] = d["hits"] / d["n"]
                        for m in ("extra_fret", "no_strum", "wrong_lane", "missing_fret"):
                            record[f"eval_{cat}_{m}"] = d["miss_modes"][m]
                if drift_ref is not None:
                    record.update(brain_drift_stats(policy, drift_ref, optimizer, watched))
                with metrics_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                tb_writer.add_scalar("eval_hit_rate", result["hit_rate"], step)
                tb_writer.add_scalar("eval_overstrums_per_min", result["overstrums_per_min"], step)
                if eval_ma is not None:
                    tb_writer.add_scalar("eval_hit_rate_ma", eval_ma, step)
                print(
                    f"[eval step={step}] difficulty={current_difficulty} hit_rate={result['hit_rate']:.4f} "
                    f"ma={eval_ma if eval_ma is None else round(eval_ma, 4)} "
                    f"overstrums/min={result['overstrums_per_min']:.2f} wall={eval_wall_s:.1f}s"
                )
                is_new_best = prev_best is None or result["hit_rate"] > prev_best
                if is_new_best:
                    should_rollback_eval = False  # never roll back on the best single eval so far
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

                if (
                    not args.fixed_difficulty and not should_rollback_eval
                    and should_advance_curriculum(watchdog, current_difficulty, cfg_bc["curriculum"]["advance_hit_rate"])
                    and difficulty_idx < len(difficulties) - 1
                ):
                    difficulty_idx += 1
                    new_difficulty = difficulties[difficulty_idx]
                    pos_weight = estimate_strum_pos_weight(
                        get_sampler(new_difficulty), fps, hit_window_s, train_window_s, n_samples=200, rng=rng,
                    )
                    print(f"[curriculum] advanced to {new_difficulty}, new strum pos_weight={pos_weight:.3f}")
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(dict(step=step, time=time.time(), curriculum_advanced_to=new_difficulty, new_pos_weight=pos_weight)) + "\n")

                if is_new_best:
                    # Save the evaluated weights now, with this eval (and any
                    # curriculum advance it caused) in the saved state; the
                    # interval block below won't save this step again.
                    do_checkpoint("interval_best" if step % cfg_bc["checkpoint_interval_steps"] == 0 else "eval_best")
                    best_path = run_dir / "checkpoint_best.pt"
                    if best_path.exists() or best_path.is_symlink():
                        best_path.unlink()
                    best_path.symlink_to(f"checkpoint_step{step}.pt")

            # After the eval, so a checkpoint at an eval step carries that
            # eval in its watchdog state (a resume from it -- e.g. --resume
            # best -- must not treat the next eval as a new best).
            if step % cfg_bc["checkpoint_interval_steps"] == 0 or (time.time() - last_checkpoint_time) >= cfg_bc["checkpoint_interval_s"]:
                if saved_step != step:
                    do_checkpoint("interval")
                if watchdog.is_healthy_now():
                    healthy_path = run_dir / "checkpoint_last_healthy.pt"
                    if healthy_path.exists() or healthy_path.is_symlink():
                        healthy_path.unlink()
                    healthy_path.symlink_to(f"checkpoint_step{step}.pt")

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
