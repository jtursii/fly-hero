"""Phase 6 task 1: record the showcase checkpoint playing the showcase songs.

Runs the *same* inference path as eval (env.step -> policy.step_frame ->
decode_trace -> rules.score_playthrough, invariants 1/2/4), with the frame
loop mirroring eval.evaluate.evaluate_songs' single-song case exactly (blank
pre_roll_s lead-in, post_roll_s tail so the last note's hit window closes,
probabilities taken over the song's own clock). The only addition is
`policy.step_frame_recorded`, which hands back the per-frame intermediates
the web viewer needs without changing the computation -- tests assert the
recorded logits are bit-identical to train.bc.run_clip's and that the
recorded hit_rate equals eval's for the same song.

Per 60 Hz frame it records: whole-population neuron rates (mean over the
frame's substeps), retina input current, logits, decoder probabilities;
then decodes the whole trace into actions and scores it into events.
Activity is mean-pooled to 20 fps and quantized to 4 bits *here* (the
export format, D44), so the intermediate is ~800 MB/song rather than ~9 GB.

Output per song: runs/<run>/<song_id>.npz plus record.json.

Usage:
  uv run python -m export.record --config configs/export.yaml \\
      --checkpoint runs/20260916_222606_bc_full_real_v2/checkpoint_step42000.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval.evaluate import resolve_run_config, score_song, with_pre_roll
from flyhero.game.decoder import decode_trace, frets_to_held_mask
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import Song, load_song_merged, slice_song
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.bc import build_policy

ACTIVITY_LEVELS = 16  # 4-bit


def quantize_4bit(rates: np.ndarray) -> np.ndarray:
    """rates in [0, 1] -> uint8 levels 0..15 (the brain's rate variable is
    already clamped to [0,1] by the model, so no rescaling is needed)."""
    return np.clip(np.rint(rates * (ACTIVITY_LEVELS - 1)), 0, ACTIVITY_LEVELS - 1).astype(np.uint8)


def choose_retina_channels(photo_map, n_channels: int) -> np.ndarray:
    """Indices (into i_photo's photoreceptor axis) of the retina points the
    site displays: evenly spaced over the photoreceptors that are both
    included and have a valid display position, so the inset samples the
    whole eye rather than one corner. Deterministic -- a display choice,
    not a model one; the brain still receives all 11,118."""
    valid = np.flatnonzero(photo_map.included_mask & ~np.isnan(photo_map.image_pos[:, 0]))
    if len(valid) <= n_channels:
        return valid
    return valid[np.linspace(0, len(valid) - 1, n_channels).round().astype(np.int64)]


def scope_channels(meta: dict) -> list[str]:
    """Oscilloscope channel names (D44, extended by D53): whole-brain and DN
    mean rate, mean retina drive, one mean-rate trace per super_class, then
    the excitatory and inhibitory population means the site plots."""
    return (["all", "dn", "retina_drive"]
            + [f"sc:{n}" for n in meta["super_class_names"]] + ["exc", "inh"])


def neuron_signs(graph: dict) -> np.ndarray:
    """+1 excitatory / -1 inhibitory per neuron, 0 where it has no outgoing
    edge. FlyWire's sign is a property of the presynaptic neuron's predicted
    transmitter, so every edge leaving a neuron carries the same sign -- this
    asserts that rather than assuming it, and reads it off the unfiltered
    edge list because min_syn gates edges, never a neuron's transmitter."""
    n = len(graph["type_id"])
    total = np.zeros(n, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    np.add.at(total, graph["pre"], graph["sign"].astype(np.int64))
    np.add.at(count, graph["pre"], 1)
    if not np.all(np.abs(total) == count):
        raise ValueError("a neuron's outgoing edges disagree in sign")
    return np.sign(total).astype(np.int8)


def record_song(
    policy, env: VecRhythmEnv, song: Song, cfg: dict, cfg_game: dict, device, dtype,
    super_class_id: np.ndarray, dn_idx: np.ndarray, activity_fps: int,
    retina_idx: np.ndarray, neuron_sign: np.ndarray, log=print,
) -> dict:
    """One song, start to finish. Returns every array the exporter needs."""
    fps = cfg_game["fps"]
    pre_roll_s, post_roll_s = cfg["pre_roll_s"], cfg["post_roll_s"]
    clip = with_pre_roll(song, pre_roll_s, post_roll_s)
    pre_frames = round(pre_roll_s * fps)
    song_frames = round((clip.duration_s - pre_roll_s) * fps)

    n_nodes = len(super_class_id)
    activity_step = fps // activity_fps
    n_activity = int(np.ceil(song_frames / activity_step))
    # Pooled to activity_fps and 4-bit quantized as we go: the full 60 Hz
    # float population trace for a 6:30 song would be 139,241 x 23,376 x 4 B.
    activity = np.zeros((n_activity, n_nodes), dtype=np.uint8)
    n_sc = int(super_class_id.max()) + 1
    scope = np.zeros((song_frames, 3 + n_sc + 2), dtype=np.float32)  # +2: exc, inh
    # Retina is stored at activity_fps on the displayed channels only: the
    # full 60 Hz x 11,118 stream is ~1 GB/song and the site shows a subsample.
    retina = np.zeros((n_activity, len(retina_idx)), dtype=np.float32)
    logits_all = np.zeros((song_frames, 6), dtype=np.float32)

    sc_masks = [torch.from_numpy(super_class_id == c).to(device) for c in range(n_sc)]
    exc_t = torch.from_numpy(neuron_sign > 0).to(device)
    inh_t = torch.from_numpy(neuron_sign < 0).to(device)
    dn_t = torch.from_numpy(dn_idx).to(device)
    retina_t = torch.from_numpy(retina_idx).to(device)

    env.reset([clip])
    policy.reset_stream(1)
    state = policy.init_state(1, device, dtype)
    t0 = time.time()
    pool_buf = np.zeros((activity_step, n_nodes), dtype=np.float32)
    retina_buf = np.zeros((activity_step, len(retina_idx)), dtype=np.float32)
    with torch.no_grad():
        for _ in range(pre_frames):  # blank lead-in, same as eval's burn-in
            obs, _, _, _ = env.step(np.zeros((1, 6), dtype=bool))
            state, _ = policy.step_frame(state, torch.from_numpy(obs["frame"]).to(device))
        for f in range(song_frames):
            obs, _, _, _ = env.step(np.zeros((1, 6), dtype=bool))
            state, logits, i_photo, rates = policy.step_frame_recorded(
                state, torch.from_numpy(obs["frame"]).to(device)
            )
            r = rates[0]
            logits_all[f] = logits[0].float().cpu().numpy()
            scope[f, 0] = float(r.mean())
            scope[f, 1] = float(r[dn_t].mean())
            scope[f, 2] = float(i_photo[0].abs().mean())
            for c, mask in enumerate(sc_masks):
                scope[f, 3 + c] = float(r[mask].mean())
            scope[f, 3 + n_sc] = float(r[exc_t].mean())
            scope[f, 4 + n_sc] = float(r[inh_t].mean())
            pool_buf[f % activity_step] = r.float().cpu().numpy()
            retina_buf[f % activity_step] = i_photo[0, retina_t].float().cpu().numpy()
            if f % activity_step == activity_step - 1 or f == song_frames - 1:
                used = (f % activity_step) + 1
                activity[f // activity_step] = quantize_4bit(pool_buf[:used].mean(axis=0))
                retina[f // activity_step] = retina_buf[:used].mean(axis=0)
            if (f + 1) % 3000 == 0:
                log(f"    frame {f + 1}/{song_frames}, wall {time.time() - t0:.0f}s")

    probs = 1.0 / (1.0 + np.exp(-logits_all.astype(np.float64)))  # [F, 6]
    frets, strum = decode_trace(probs)
    held_mask = frets_to_held_mask(frets)
    metrics, events = score_playthrough(song.notes, held_mask, strum, fps, cfg_game["hit_window_s"])
    # Same call eval/evaluate.py scores with, so hit_rate is comparable by
    # construction rather than by convention.
    scored = score_song(song, probs, fps, cfg_game["hit_window_s"])

    actions = (frets.astype(np.uint8) << np.arange(5, dtype=np.uint8)).sum(axis=1).astype(np.uint8)
    actions |= (strum.astype(np.uint8) << 5)

    log(f"    hit_rate {metrics['hit_rate']:.4f} ({metrics['n_hits']}/{metrics['n_notes']}), "
        f"overstrums/min {metrics['overstrums_per_min']:.1f}, wall {time.time() - t0:.0f}s")
    return dict(
        song=song, metrics=metrics, scored=scored, events=events, activity=activity,
        retina=retina, scope=scope, probs=probs.astype(np.float32), actions=actions,
        frames=song_frames, activity_frames=n_activity,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/export.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--run-name", default="record")
    p.add_argument("--max-frames", type=int, default=None, help="debug only: cap frames per song")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(cfg["seed"])
    run_dir = make_run_dir(args.run_name, args.config, cfg["seed"])
    print(f"run dir: {run_dir}", flush=True)

    ckpt_file = Path(args.checkpoint).resolve()
    src_run = ckpt_file.parent
    cfg_game = load_config(resolve_run_config(src_run, "game.yaml"))
    cfg_brain = load_config(resolve_run_config(src_run, "brain.yaml"))
    processed_dir = Path(cfg["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)
    policy = build_policy("connectome", {}, cfg_game, cfg_brain, photo_map, device, dtype)
    ckpt = torch.load(ckpt_file, map_location=device)
    policy.load_state_dict(ckpt["model_state"])
    step = int(ckpt["step"])
    print(f"{ckpt_file} (step {step}, {device})", flush=True)

    splits = json.loads((processed_dir / "splits.json").read_text())
    split_of = {sid: name for name, ids in splits.items() for sid in ids}
    super_class_id = graph_np["super_class_id"].astype(np.int64)
    dn_idx = graph_np["dn_idx"].astype(np.int64)
    neuron_sign = neuron_signs(graph_np)
    retina_idx = choose_retina_channels(photo_map, cfg["retina_channels"])
    retina_display_pos = photo_map.image_pos[retina_idx]  # [C, 2] in frame pixels

    out_songs = []
    for entry in cfg["showcase_songs"]:
        sid, difficulty = entry["song_id"], entry["difficulty"]
        song = load_song_merged(processed_dir / "songs" / f"{sid}_{difficulty}.npz", cfg_game["chord_merge_min_gap_s"])
        if args.max_frames:
            song = slice_song(song, 0.0, args.max_frames / cfg_game["fps"])
        split = split_of[sid]
        print(f"  {song.artist} - {song.title} [{difficulty}, {split}, {song.duration_s:.0f}s]", flush=True)
        rec = record_song(
            policy, env, song, cfg, cfg_game, device, dtype, super_class_id, dn_idx,
            cfg["activity_fps"], retina_idx, neuron_sign,
        )
        np.savez(
            run_dir / f"{sid}.npz", activity=rec["activity"], retina=rec["retina"],
            scope=rec["scope"], probs=rec["probs"], actions=rec["actions"],
            # The exact notes that were scored -- the manifest must describe
            # this playthrough, not a re-load of the song from disk.
            notes=song.notes,
            event_frame=np.array([e.frame for e in rec["events"]], dtype=np.int32),
            event_note_idx=np.array([e.note_idx for e in rec["events"]], dtype=np.int32),
            event_kind=np.array([e.kind for e in rec["events"]]),
        )
        out_songs.append(dict(
            song_id=sid, difficulty=difficulty, split=split,
            seen_label="never seen" if split == "test" else "practiced",
            title=song.title, artist=song.artist, charter=song.charter,
            duration_s=song.duration_s, frames=rec["frames"], activity_frames=rec["activity_frames"],
            chart_offset_s=song.chart_offset_s, ini_delay_s=song.ini_delay_s,
            metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in rec["metrics"].items()},
            n_events=len(rec["events"]),
        ))
        (run_dir / "record.json").write_text(json.dumps(dict(
            checkpoint=str(ckpt_file), step=step, device=str(device), fps=cfg_game["fps"],
            activity_fps=cfg["activity_fps"], scope_channels=scope_channels(meta),
            retina_channel_idx=retina_idx.tolist(),
            retina_display_pos=retina_display_pos.astype(float).tolist(),
            songs=out_songs,
        ), indent=1))
    print(f"wrote {run_dir / 'record.json'}", flush=True)


if __name__ == "__main__":
    main()
