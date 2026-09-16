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

_stop_requested = False


def _handle_sigterm(signum, frame) -> None:
    global _stop_requested
    _stop_requested = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bc.yaml")
    p.add_argument("--game-config", default="configs/game.yaml")
    p.add_argument("--brain-config", default="configs/brain.yaml")
    p.add_argument("--model", choices=["connectome", "gru"], default="connectome")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--fixed-difficulty", default=None, help="disable curriculum, train/eval only this difficulty")
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    return p.parse_args()


def build_policy(model_kind: str, cfg_bc: dict, cfg_game: dict, cfg_brain: dict, photo_map, device, dtype):
    if model_kind == "gru":
        model = GRUBaseline(frame_size=cfg_game["frame_size"]).to(device=device, dtype=dtype)
        return model

    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])
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


def evaluate_difficulty(
    policy, env: VecRhythmEnv, val_entries, burn_in_s: float, fps: int, hit_window_s: float, device, dtype,
) -> dict:
    hit_rates, overstrums = [], []
    for entry in val_entries:
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
        metrics, _ = score_playthrough(excerpt_notes, held_mask, strum, fps, hit_window_s)
        hit_rates.append(metrics["hit_rate"])
        overstrums.append(metrics["overstrums_per_min"])
    return dict(
        hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
        overstrums_per_min=float(np.mean(overstrums)) if overstrums else 0.0,
        n_songs=len(val_entries),
    )


def main() -> None:
    args = parse_args()
    cfg_bc = load_config(args.config)
    cfg_game = load_config(args.game_config)
    cfg_brain = load_config(args.brain_config) if args.model == "connectome" else {}

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = getattr(torch, args.dtype)
    signal.signal(signal.SIGTERM, _handle_sigterm)

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
        run_name = args.run_name or f"bc_{args.model}"
        run_dir = make_run_dir(run_name, args.config, seed)
        for extra_cfg in (args.game_config, args.brain_config) if args.model == "connectome" else (args.game_config,):
            import shutil

            shutil.copy(extra_cfg, run_dir / Path(extra_cfg).name)
        (run_dir / "model_kind.txt").write_text(args.model + "\n")
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

        save_checkpoint(path, policy, optimizer, step, difficulty_idx, pos_weight, rng, extra={"reason": reason})
        latest = run_dir / "checkpoint_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
        last_checkpoint_time = time.time()
        print(f"checkpoint saved: {path} (reason={reason})")

    while True:
        if args.max_steps is not None and step >= args.max_steps:
            break
        if args.max_time_s is not None and (time.time() - start_time) >= args.max_time_s:
            break
        if _stop_requested:
            do_checkpoint("sigterm")
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

        # NaN-safe gradient sanitization (user decision, docs/PROGRESS.md's
        # 2026-09-16 smoke-test entry): backward() through the real
        # connectome reliably produces NaN gradients for a handful of
        # shared-per-cell-type parameters (found: R7, R8 photoreceptors, L1,
        # L5, one 2-neuron type) -- a gradient-explosion-through-time effect
        # in the 240-substep BPTT unroll, not a forward-pass bug (forward
        # activity stays bounded throughout). Rather than skip the whole
        # step (which was previously happening on ~100% of steps, i.e. zero
        # training progress -- clip_grad_norm_'s combined norm propagates
        # one NaN parameter into every parameter's effective gradient), NaN/
        # Inf entries are zeroed in place right after backward(), before
        # clipping: the ~9,023 unaffected cell types still get a real
        # update every step; the ~5 affected types simply see no gradient on
        # a step where their contribution was unstable (equivalent to "no
        # signal this step" for just those parameters, not a corrupted
        # update) and stay near their current value until a step where their
        # gradient happens to be finite.
        is_nan_loss = bool(torch.isnan(loss).item())
        if not is_nan_loss:
            optimizer.zero_grad()
            loss.backward()
            n_grad_elems, n_bad_elems = 0, 0
            for p in list(policy.readout_parameters()) + list(policy.non_readout_parameters()):
                if p.grad is None:
                    continue
                bad = ~torch.isfinite(p.grad)
                n_bad = int(bad.sum().item())
                if n_bad:
                    n_bad_elems += n_bad
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                n_grad_elems += p.grad.numel()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(policy.readout_parameters()) + list(policy.non_readout_parameters()), cfg_bc["grad_clip_norm"],
            )
            optimizer.step()
            optimizer.zero_grad()
            is_nan = n_bad_elems > 0  # "step needed sanitization", not "step skipped" -- every step now updates
        else:
            is_nan = True
            grad_norm = torch.tensor(float("nan"))
        if is_nan:
            nan_steps += 1

        step_dt = time.time() - t0
        step_times.append(step_dt)
        window_steps += 1

        step += 1
        if step % 10 == 0:
            steps_per_s = len(step_times) / sum(step_times) if step_times else 0.0
            record = dict(
                step=step, difficulty=difficulties[difficulty_idx],
                loss=float(loss.item()) if not is_nan_loss else float("nan"),
                fret_loss=float(fret_loss.item()) if not is_nan_loss else float("nan"),
                strum_loss=float(strum_loss.item()) if not is_nan_loss else float("nan"),
                grad_norm=float(grad_norm.item()), max_abs_state=max_abs_state,
                nan_frac=nan_steps / window_steps, steps_per_s=steps_per_s, pos_weight=pos_weight,
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(k, v, step)
            step_times = step_times[-50:]

        if step % cfg_bc["checkpoint_interval_steps"] == 0 or (time.time() - last_checkpoint_time) >= cfg_bc["checkpoint_interval_s"]:
            do_checkpoint("interval")

        if step % cfg_bc["eval"]["interval_steps"] == 0:
            eval_t0 = time.time()
            current_difficulty = difficulties[difficulty_idx]
            result = evaluate_difficulty(
                policy, env, val_set[current_difficulty], burn_in_s, fps, hit_window_s, device, dtype,
            )
            eval_wall_s = time.time() - eval_t0
            record = dict(
                step=step, eval_difficulty=current_difficulty, eval_hit_rate=result["hit_rate"],
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

            if not args.fixed_difficulty and result["hit_rate"] >= cfg_bc["curriculum"]["advance_hit_rate"] and difficulty_idx < len(difficulties) - 1:
                difficulty_idx += 1
                new_difficulty = difficulties[difficulty_idx]
                pos_weight = estimate_strum_pos_weight(
                    get_sampler(new_difficulty), fps, hit_window_s, train_window_s, n_samples=200, rng=rng,
                )
                print(f"[curriculum] advanced to {new_difficulty}, new strum pos_weight={pos_weight:.3f}")
                with metrics_path.open("a") as f:
                    f.write(json.dumps(dict(step=step, curriculum_advanced_to=new_difficulty, new_pos_weight=pos_weight)) + "\n")

    do_checkpoint("final")
    tb_writer.close()
    print(f"done. total steps={step}, elapsed={time.time() - start_time:.1f}s, run dir={run_dir}")


if __name__ == "__main__":
    main()
