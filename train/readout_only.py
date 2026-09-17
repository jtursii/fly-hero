"""Readout-only control (PLAN Phase 4's overnight frozen-brain control,
accelerated per the user's Phase 3b decision): the frozen, untrained
connectome brain's DN rates are recorded once (no_grad) over a fixed cache
of 4,000 note-aware training clips plus full-length val-song traces, then
only the readout is trained against the cached rates -- same labels, loss,
and decoder as the full model (invariant 2 unaffected: still reads DN rates
only). No brain forward/backward is needed per readout-training step, so
many epochs run in the time bc.py would need for a handful of real steps --
fast enough to run today rather than overnight.

Two phases, both run by default (--skip-cache-if-exists to reuse a prior
cache):
  1. cache: for each difficulty, sample N_CLIPS_PER_DIFFICULTY clips via the
     same note-aware ClipSampler bc.py uses, run the frozen brain (no_grad),
     and save DN-rate traces + labels to data/processed/dn_cache/. Also
     records each val song's full-length DN trace (not just a 60s excerpt --
     "exactly like the full model" refers to the scoring pipeline: decode +
     rules.score_playthrough, not the excerpt-vs-full-song choice, which
     differs from bc.py's in-training eval only because it's now cheap).
  2. train: fit a fresh Readout on the cached rates with the same
     curriculum/mixing/pos_weight/BCE-loss mechanics as bc.py, evaluating on
     the cached full-song val traces every eval_interval_steps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta
from flyhero.brain.readout import Readout
from flyhero.game.clip_sampler import ClipSampler, list_available_song_ids
from flyhero.game.decoder import decode_trace, frets_to_held_mask
from flyhero.game.labels import compute_frame_labels
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import load_song_merged, slice_song
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.common import unpack_fret_targets

N_CLIPS_TOTAL = 4000
CACHE_BATCH_SIZE = 32  # no_grad-only forward pass -- no activations retained for backward,
# so this can be much larger than bc.py's memory-constrained fwd+bwd batch_size (D26's
# batch=4 was chosen for training, not for this one-time no-grad recording pass).
CACHE_ROOT = Path("data/processed/dn_cache")


def _clip_cache_dir(difficulty: str) -> Path:
    return CACHE_ROOT / "clips" / difficulty


def _val_cache_dir(difficulty: str) -> Path:
    return CACHE_ROOT / "val" / difficulty


def record_dn_rates(
    brain: ConnectomeBrain, retina: TorchRetina, env: VecRhythmEnv, clips, n_frames_each: list[int], substeps: int,
) -> list[np.ndarray]:
    """clips: list of Song, each with its own duration; n_frames_each: frame
    count per clip (may differ -- used for val songs of varying length, so
    this does NOT batch across clips of different lengths; batching only
    happens in build_cache where all training clips share one duration)."""
    out = []
    for clip, n_frames in zip(clips, n_frames_each):
        env.reset([clip])
        retina.reset()
        v = brain.init_state(1)
        rates = np.zeros((n_frames, brain.n_dn), dtype=np.float32)
        with torch.no_grad():
            for f in range(n_frames):
                obs, _, _, _ = env.step(np.zeros((1, 6), dtype=bool))
                frame_t = torch.from_numpy(obs["frame"]).to(brain.device)
                i_photo = retina.step(frame_t).to(dtype=brain.dtype)
                v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=substeps)
                rates[f] = dn_rate_mean[0].cpu().numpy()
        out.append(rates)
    return out


def record_dn_rates_batched(
    brain: ConnectomeBrain, retina: TorchRetina, env: VecRhythmEnv, clips, n_frames: int, substeps: int,
) -> np.ndarray:
    """All clips share n_frames (training-clip cache path: fixed
    burn_in_s+train_window_s). Returns dn_rates[B, n_frames, n_dn]."""
    batch = len(clips)
    env.reset(clips)
    retina.reset()
    v = brain.init_state(batch)
    rates = np.zeros((batch, n_frames, brain.n_dn), dtype=np.float32)
    with torch.no_grad():
        for f in range(n_frames):
            obs, _, _, _ = env.step(np.zeros((batch, 6), dtype=bool))
            frame_t = torch.from_numpy(obs["frame"]).to(brain.device)
            i_photo = retina.step(frame_t).to(dtype=brain.dtype)
            v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=substeps)
            rates[:, f, :] = dn_rate_mean.cpu().numpy()
    return rates


def build_cache(cfg_bc: dict, cfg_game: dict, cfg_brain: dict, brain, retina, env, rng: np.random.Generator) -> None:
    processed_dir = Path(cfg_bc["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())
    burn_in_s, train_window_s = cfg_bc["burn_in_s"], cfg_bc["train_window_s"]
    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    substeps = cfg_brain["substeps_per_frame"]
    difficulties = cfg_bc["curriculum"]["difficulties"]
    n_per_difficulty = N_CLIPS_TOTAL // len(difficulties)
    burn_in_frames, train_frames = round(burn_in_s * fps), round(train_window_s * fps)
    total_frames = burn_in_frames + train_frames
    batch = CACHE_BATCH_SIZE

    for difficulty in difficulties:
        out_dir = _clip_cache_dir(difficulty)
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(list(out_dir.glob("*.npz"))) >= n_per_difficulty:
            print(f"[cache] {difficulty}: already have >= {n_per_difficulty} clips, skipping")
            continue

        sampler = ClipSampler(
            processed_dir, splits["train"], difficulty, burn_in_s, train_window_s, fps,
            cfg_game["chord_merge_min_gap_s"], cfg_bc["frac_clips_with_notes"],
        )
        t0 = time.time()
        n_written = 0
        while n_written < n_per_difficulty:
            n_this_batch = min(batch, n_per_difficulty - n_written)
            clips, fret_bits_list, strum_list = [], [], []
            for _ in range(n_this_batch):
                song, start_s, _ = sampler.sample(rng)
                window_start = start_s - burn_in_s
                clip = slice_song(song, window_start, burn_in_s + train_window_s)
                grad_only = slice_song(clip, burn_in_s, train_window_s)
                fret_target, strum = compute_frame_labels(grad_only.notes, train_window_s, fps, hit_window_s)
                clips.append(clip)
                fret_bits_list.append(unpack_fret_targets(fret_target))
                strum_list.append(strum.astype(np.float32))

            dn_rates = record_dn_rates_batched(brain, retina, env, clips, total_frames, substeps)
            for i in range(n_this_batch):
                np.savez(
                    out_dir / f"clip_{n_written + i:05d}.npz",
                    dn_rates_grad=dn_rates[i, burn_in_frames:, :],
                    fret_bits=fret_bits_list[i],
                    strum=strum_list[i],
                )
            n_written += n_this_batch
        print(f"[cache] {difficulty}: wrote {n_written} clips in {time.time() - t0:.1f}s")

    for difficulty in difficulties:
        out_dir = _val_cache_dir(difficulty)
        out_dir.mkdir(parents=True, exist_ok=True)
        val_ids = sorted(splits["val"])
        available = list_available_song_ids(processed_dir, val_ids, difficulty)[: cfg_bc["eval"]["n_val_songs"]]
        # Per-song skip (not per-difficulty): robust to a prior interrupted
        # run that finished some songs but not all -- caught in review
        # before this went further (a killed diagnostic run left exactly
        # this partial state for Medium: 1/10 songs cached, valid but
        # incomplete, which the old "any files -> skip" check would have
        # silently frozen at 1 val song forever).
        already = {p.stem for p in out_dir.glob("*.npz")}
        todo = [sid for sid in available if sid not in already]
        if not todo:
            print(f"[val cache] {difficulty}: already have all {len(available)} val songs, skipping")
            continue
        t0 = time.time()
        for sid in todo:
            path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
            song = load_song_merged(path, cfg_game["chord_merge_min_gap_s"])
            n_frames = max(int(round(song.duration_s * fps)), 1)
            (rates,) = record_dn_rates(brain, retina, env, [song], [n_frames], substeps=cfg_brain["substeps_per_frame"])
            np.savez(out_dir / f"{sid}.npz", dn_rates=rates, duration_s=song.duration_s)
        print(f"[val cache] {difficulty}: wrote {len(todo)} full-length val traces ({len(available)} total) in {time.time() - t0:.1f}s")


class CachedClipPool:
    def __init__(self, difficulty: str):
        self.paths = sorted(_clip_cache_dir(difficulty).glob("*.npz"))
        if not self.paths:
            raise RuntimeError(f"no cached clips for difficulty={difficulty} -- run build_cache first")

    def sample(self, rng: np.random.Generator) -> dict:
        path = self.paths[rng.integers(len(self.paths))]
        return dict(np.load(path))


def evaluate_readout_full_songs(readout: Readout, difficulty: str, fps: int, hit_window_s: float, cfg_game: dict, cfg_bc: dict) -> dict:
    processed_dir = Path(cfg_bc["processed_dir"])
    val_dir = _val_cache_dir(difficulty)
    splits = json.loads((processed_dir / "splits.json").read_text())
    val_ids = sorted(splits["val"])
    available = list_available_song_ids(processed_dir, val_ids, difficulty)[: cfg_bc["eval"]["n_val_songs"]]

    hit_rates, overstrums = [], []
    for sid in available:
        cache_path = val_dir / f"{sid}.npz"
        if not cache_path.exists():
            continue
        data = np.load(cache_path)
        dn_rates = torch.from_numpy(data["dn_rates"])  # [T, n_dn]
        with torch.no_grad():
            logits = readout(dn_rates)
        probs = torch.sigmoid(logits).numpy()
        frets, strum = decode_trace(probs)
        held_mask = frets_to_held_mask(frets)

        song_path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
        song = load_song_merged(song_path, cfg_game["chord_merge_min_gap_s"])
        metrics, _ = score_playthrough(song.notes, held_mask, strum, fps, hit_window_s)
        hit_rates.append(metrics["hit_rate"])
        overstrums.append(metrics["overstrums_per_min"])

    return dict(
        hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
        overstrums_per_min=float(np.mean(overstrums)) if overstrums else 0.0,
        n_songs=len(hit_rates),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bc.yaml")
    p.add_argument("--game-config", default="configs/game.yaml")
    p.add_argument("--brain-config", default="configs/brain.yaml")
    p.add_argument("--run-name", default="readout_only")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    cfg_bc = load_config(args.config)
    cfg_game = load_config(args.game_config)
    cfg_brain = load_config(args.brain_config)
    seed = cfg_bc["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    processed_dir = Path(cfg_bc["processed_dir"])
    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"], cfg_brain.get("graph_file", "graph.npz"))
    photo_map = build_photoreceptor_map(dict(np.load(processed_dir / "graph.npz")), meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)

    # No-grad-only forward pass -- Gate G3's chosen mps/float32 is a large
    # speedup over CPU here too (no backward pass to justify CPU's simplicity).
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype = torch.float32
    brain = ConnectomeBrain(graph, cfg_brain, device=device, dtype=dtype)
    retina_cfg = cfg_game["retina"]
    retina = TorchRetina.build(
        photo_map, cfg_game["frame_size"], retina_cfg["gain"], retina_cfg["ema_alpha"], retina_cfg["blur_sigma"], device,
    )

    if args.rebuild_cache:
        import shutil

        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    print("=== building/checking DN-rate cache ===")
    build_cache(cfg_bc, cfg_game, cfg_brain, brain, retina, env, rng)

    print("\n=== fitting readout on cached rates ===")
    readout = Readout(brain.n_dn, cfg_brain["readout"]["n_actions"])
    optimizer = torch.optim.AdamW(readout.parameters(), lr=cfg_bc["lr_readout"])

    difficulties = cfg_bc["curriculum"]["difficulties"]
    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    batch_size = cfg_bc["batch_size"]

    pools = {d: CachedClipPool(d) for d in difficulties}

    def estimate_pos_weight_from_cache(difficulty: str, n_samples: int = 200) -> float:
        n_pos, n_neg = 0, 0
        for _ in range(n_samples):
            data = pools[difficulty].sample(rng)
            n_pos += int(data["strum"].sum())
            n_neg += int(len(data["strum"]) - data["strum"].sum())
        return float(n_neg / n_pos) if n_pos > 0 else 1.0

    difficulty_idx = 0
    pos_weight = estimate_pos_weight_from_cache(difficulties[difficulty_idx])
    print(f"initial strum pos_weight ({difficulties[difficulty_idx]}): {pos_weight:.3f}")

    run_dir = make_run_dir(args.run_name, args.config, seed)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"run dir: {run_dir}")

    start_time = time.time()
    for step in range(1, args.max_steps + 1):
        current_frac = 1.0 - cfg_bc["curriculum"]["earlier_frac"]
        fret_bits_batch, strum_batch, dn_rates_batch = [], [], []
        for _ in range(batch_size):
            if difficulty_idx == 0 or rng.random() < current_frac:
                d = difficulties[difficulty_idx]
            else:
                d = difficulties[: difficulty_idx][rng.integers(difficulty_idx)]
            data = pools[d].sample(rng)
            dn_rates_batch.append(data["dn_rates_grad"])
            fret_bits_batch.append(data["fret_bits"])
            strum_batch.append(data["strum"])

        dn_rates_t = torch.as_tensor(np.stack(dn_rates_batch), dtype=torch.float32)
        fret_bits_t = torch.as_tensor(np.stack(fret_bits_batch), dtype=torch.float32)
        strum_t = torch.as_tensor(np.stack(strum_batch), dtype=torch.float32)

        logits = readout(dn_rates_t)
        fret_loss = F.binary_cross_entropy_with_logits(logits[..., :5], fret_bits_t)
        strum_loss = F.binary_cross_entropy_with_logits(
            logits[..., 5], strum_t, pos_weight=torch.tensor(pos_weight),
        )
        loss = fret_loss + strum_loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(readout.parameters(), cfg_bc["grad_clip_norm"])
        optimizer.step()

        if step % 50 == 0:
            record = dict(
                step=step, difficulty=difficulties[difficulty_idx], loss=float(loss.item()),
                grad_norm=float(grad_norm.item()), pos_weight=pos_weight,
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

        if step % cfg_bc["eval"]["interval_steps"] == 0 or step == args.max_steps:
            current_difficulty = difficulties[difficulty_idx]
            result = evaluate_readout_full_songs(readout, current_difficulty, fps, hit_window_s, cfg_game, cfg_bc)
            print(
                f"[eval step={step}] difficulty={current_difficulty} hit_rate={result['hit_rate']:.4f} "
                f"overstrums/min={result['overstrums_per_min']:.2f} n_songs={result['n_songs']}"
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(dict(step=step, eval_difficulty=current_difficulty, **result)) + "\n")

            if result["hit_rate"] >= cfg_bc["curriculum"]["advance_hit_rate"] and difficulty_idx < len(difficulties) - 1:
                difficulty_idx += 1
                pos_weight = estimate_pos_weight_from_cache(difficulties[difficulty_idx])
                print(f"[curriculum] advanced to {difficulties[difficulty_idx]}, new pos_weight={pos_weight:.3f}")

    torch.save(readout.state_dict(), run_dir / "readout_final.pt")
    print(f"done. elapsed={time.time() - start_time:.1f}s, run dir={run_dir}")


if __name__ == "__main__":
    main()
