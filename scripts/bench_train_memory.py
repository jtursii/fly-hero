"""Phase 3b diagnostic: real-config memory benchmark under the actual
training-loop shape (1s burn-in under no_grad + 2s gradient window, forward+
backward), not bench_brain.py's uniform-grad-throughout 1s measurement.
Compares batch=8 with/without per-frame gradient checkpointing, and batch=4
without checkpointing. Device/dtype fixed to Gate G3's chosen mps/float32.

Reuses bench_brain.py's pipeline construction (_load_real_songs, TorchRetina,
VecRhythmEnv) and its subprocess-isolation pattern for peak-memory
measurement (D23: process/driver peak readings are contaminated by earlier
cells in the same process, so every measured cell runs in its own
subprocess). Since bc.py's real loss doesn't exist yet, a logits.sum() proxy
loss is used to measure memory/time -- matching what bench_brain.py already
does for the same reason.

This is a diagnostic script (Phase 3b plan, part 1a): it prints a table and
picks the fastest config <=40GiB peak, but does not write configs/brain.yaml
or DECISIONS.md -- that happens only once the user approves promoting the
result to a committed decision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta
from flyhero.brain.readout import Readout
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import load_song_merged
from flyhero.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_G3_MAX_PEAK_BYTES = 40 * (1024**3)
DEVICE_NAME = "mps"
DTYPE = torch.float32


def _peak_mem_sample(device: torch.device) -> int:
    if device.type == "mps":
        torch.mps.synchronize()
        return int(torch.mps.driver_allocated_memory())
    import resource

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def _load_real_songs(cfg_game: dict, batch: int, min_frames: int) -> list:
    processed_dir = Path(cfg_game["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())
    songs = []
    for sid in splits["train"]:
        p = processed_dir / "songs" / f"{sid}_Expert.npz"
        if p.exists():
            song = load_song_merged(p, cfg_game["chord_merge_min_gap_s"])
            if song.duration_s * cfg_game["fps"] >= min_frames:
                songs.append(song)
        if len(songs) >= batch:
            break
    return songs


def _measure_config(batch: int, checkpoint: bool) -> dict:
    """Runs entirely within the calling process -- always invoked as an
    isolated subprocess so peak-memory sampling isn't polluted by earlier
    cells (D23)."""
    cfg_brain = load_config("configs/brain.yaml")
    cfg_game = load_config("configs/game.yaml")
    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])

    device = torch.device(DEVICE_NAME)
    processed_dir = Path(cfg_game["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])

    burn_in_s = 1.0
    train_s = 2.0
    fps = cfg_game["fps"]
    n_burn_in = round(burn_in_s * fps)
    n_train = round(train_s * fps)
    n_frames = n_burn_in + n_train

    songs = _load_real_songs(cfg_game, batch, min_frames=n_frames)
    if len(songs) < batch:
        raise RuntimeError(f"need {batch} train-split songs with >= {n_frames} frames, found {len(songs)}")

    env = VecRhythmEnv(cfg_game, photo_map)
    env.reset(songs)
    zero_actions = np.zeros((batch, 6), dtype=bool)
    frames_list = []
    for _ in range(n_frames):
        obs, _, _, _ = env.step(zero_actions)
        frames_list.append(obs["frame"])
    frames_np = np.stack(frames_list)  # [n_frames, batch, H, W], rendered once, not timed
    frames_t_all = torch.from_numpy(frames_np).to(device)

    brain = ConnectomeBrain(graph, cfg_brain, device=device, dtype=DTYPE)
    readout = Readout(brain.n_dn, cfg_brain["readout"]["n_actions"]).to(device=device, dtype=DTYPE)
    retina_cfg = cfg_game["retina"]
    retina = TorchRetina.build(
        photo_map, cfg_game["frame_size"], retina_cfg["gain"], retina_cfg["ema_alpha"],
        retina_cfg["blur_sigma"], device,
    )
    params = list(brain.parameters()) + list(readout.parameters())
    substeps = cfg_brain["substeps_per_frame"]

    def frame_step_ckpt(v, i_photo):
        return torch.utils.checkpoint.checkpoint(
            brain.frame_step, v, i_photo, substeps, use_reentrant=False
        )

    def run(track_memory: bool) -> int:
        retina.reset()
        v = brain.init_state(batch)
        peak = _peak_mem_sample(device) if track_memory else 0

        with torch.no_grad():
            for f in range(n_burn_in):
                i_photo = retina.step(frames_t_all[f]).to(dtype=DTYPE)
                v, _ = brain.frame_step(v, i_photo, n_substeps=substeps)
        v = v.detach()
        if track_memory:
            peak = max(peak, _peak_mem_sample(device))

        logits = None
        for f in range(n_burn_in, n_frames):
            i_photo = retina.step(frames_t_all[f]).to(dtype=DTYPE)
            if checkpoint:
                v, dn_rate_mean = frame_step_ckpt(v, i_photo)
            else:
                v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=substeps)
            logits = readout(dn_rate_mean)
            if track_memory:
                peak = max(peak, _peak_mem_sample(device))
        logits.sum().backward()
        if track_memory:
            peak = max(peak, _peak_mem_sample(device))
        return peak

    run(track_memory=False)  # warm-up
    for p in params:
        p.grad = None
    _empty_cache(device)

    _sync(device)
    t0 = time.perf_counter()
    run(track_memory=False)
    _sync(device)
    elapsed = time.perf_counter() - t0
    for p in params:
        p.grad = None
    _empty_cache(device)

    peak = run(track_memory=True)
    return {"elapsed": elapsed, "peak": peak}


def _launch_subprocess(batch: int, checkpoint: bool) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "worker", str(batch), "1" if checkpoint else "0"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": "."},
    )
    if result.returncode != 0:
        return {"error": (result.stderr.strip() or result.stdout.strip())[-3000:]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as e:
        return {"error": f"could not parse worker output ({e}); stdout={result.stdout[-500:]}"}


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**3):.2f} GiB"


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available -- this benchmark is fixed to mps/float32 (Gate G3's chosen config)")

    configs = [
        ("batch=8, no checkpoint", 8, False),
        ("batch=8, checkpoint", 8, True),
        ("batch=4, no checkpoint", 4, False),
    ]

    rows = []
    for label, batch, ckpt in configs:
        print(f"running: {label} ...", flush=True)
        res = _launch_subprocess(batch, ckpt)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            rows.append(dict(label=label, batch=batch, checkpoint=ckpt, error=res["error"]))
            continue
        peak_ok = res["peak"] <= GATE_G3_MAX_PEAK_BYTES
        print(f"  elapsed={res['elapsed']:.3f}s peak={_fmt_bytes(res['peak'])} (<=40GiB: {peak_ok})")
        rows.append(dict(label=label, batch=batch, checkpoint=ckpt, **res, peak_ok=peak_ok))

    print("\n| config | elapsed (s) | peak | <=40GiB |")
    print("|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['label']} | ERROR | | |")
        else:
            print(f"| {r['label']} | {r['elapsed']:.3f} | {_fmt_bytes(r['peak'])} | {r['peak_ok']} |")

    ok_rows = [r for r in rows if "error" not in r and r["peak_ok"]]
    if ok_rows:
        fastest = min(ok_rows, key=lambda r: r["elapsed"])
        print(f"\nFastest config <=40GiB: {fastest['label']} ({fastest['elapsed']:.3f}s, {_fmt_bytes(fastest['peak'])})")
    else:
        print("\nNo config met the <=40GiB peak bound.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _, _, batch_s, ckpt_s = sys.argv
        out = _measure_config(int(batch_s), ckpt_s == "1")
        print(json.dumps(out))
    else:
        main()
