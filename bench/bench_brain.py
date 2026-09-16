"""Gate G3 benchmark (PLAN Phase 3 task 3): sweep device x dtype x batch x
min_syn for the brain alone, then measure the full per-frame step (retina +
brain substeps + readout) at batch=8. Writes docs/bench.md and prints the
chosen device/dtype against Gate G3's threshold (fwd+bwd for 1 game-second
at batch 8 in <=3s wall-clock, <=40GB peak).

Every timed/memory-measured cell runs in its OWN subprocess (this script
re-invokes itself as a worker -- see _cli_worker). This was not the original
design and was added after the first run showed why it's necessary (D23):
process peak RSS (resource.getrusage) is a lifetime high-water mark that
never resets within a process, and MPS's caching allocator doesn't return
memory to the driver until torch.mps.empty_cache() is called -- both mean a
single long-running process's "peak" reading after the first large cell just
reflects that earlier cell's footprint, not the current one's. The first run
of this script showed exactly that: CPU peak pinned at the same value for
every cell after the first big one, and MPS backward times/memory wildly
inconsistent between cells using the same shapes. Subprocess isolation is
the only fully correct fix for the OS-level RSS issue.

Peak-memory methodology: MPS polls torch.mps.driver_allocated_memory() every
substep (no native peak-tracking API exists for MPS); CPU reports process
peak RSS (already bytes on macOS/Darwin). Not directly comparable to each
other -- see D23 in docs/DECISIONS.md.

Follows bench/bench_env.py's convention otherwise: no argparse, load config
+ real data, warm-up before timing.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flyhero.brain.rate_model import ConnectomeBrain, filter_by_min_syn, load_graph_and_meta
from flyhero.brain.readout import Readout
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import load_song_merged
from flyhero.utils.config import load_config

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "float64": torch.float64}
GATE_G3_MAX_FWD_BWD_S = 3.0
GATE_G3_MAX_PEAK_BYTES = 40 * (1024**3)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _peak_mem_sample(device: torch.device) -> int:
    if device.type == "mps":
        torch.mps.synchronize()
        return int(torch.mps.driver_allocated_memory())
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS (Darwin), KB on Linux.
    return int(ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def _zero_grads(params) -> None:
    for p in params:
        p.grad = None


def _run_brain_substeps(
    brain: ConnectomeBrain, batch: int, n_substeps: int, backward: bool, track_memory: bool
) -> int:
    device = brain.device
    v = brain.init_state(batch)
    i_photo = torch.rand(batch, brain.n_input, device=device, dtype=brain.dtype)
    peak = _peak_mem_sample(device) if track_memory else 0
    ctx = torch.enable_grad() if backward else torch.no_grad()
    with ctx:
        for _ in range(n_substeps):
            v, r = brain.step(v, i_photo)
            if track_memory:
                peak = max(peak, _peak_mem_sample(device))
        if backward:
            v.sum().backward()
            if track_memory:
                peak = max(peak, _peak_mem_sample(device))
    return peak


def _measure_brain_cell(
    device_name: str, dtype_name: str, batch: int, min_syn: int, backward: bool,
    game_seconds: float, n_warmup: int = 2,
) -> dict:
    """Runs entirely within the calling process -- always invoked as an
    isolated subprocess (see _launch_subprocess) so peak-memory sampling
    isn't polluted by earlier cells."""
    cfg_brain = load_config("configs/brain.yaml")
    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])
    base_min_syn = int(meta["min_syn"])
    graph_variant = graph if min_syn == base_min_syn else filter_by_min_syn(graph, min_syn)

    device = torch.device(device_name)
    dtype = DTYPE_MAP[dtype_name]
    brain = ConnectomeBrain(graph_variant, cfg_brain, device=device, dtype=dtype)
    n_substeps = round(game_seconds / brain.dt)

    for _ in range(n_warmup):
        _run_brain_substeps(brain, batch, n_substeps, backward, track_memory=False)
        _zero_grads(brain.parameters())
    _empty_cache(device)

    _sync(device)
    t0 = time.perf_counter()
    _run_brain_substeps(brain, batch, n_substeps, backward, track_memory=False)
    _sync(device)
    elapsed = time.perf_counter() - t0
    _zero_grads(brain.parameters())
    _empty_cache(device)

    peak = _run_brain_substeps(brain, batch, n_substeps, backward, track_memory=True)
    return {"elapsed": elapsed, "peak": peak}


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


def _measure_full_pipeline_cell(
    device_name: str, dtype_name: str, batch: int, backward: bool, n_frames: int,
    n_warmup_frames: int = 5,
) -> dict:
    """Runs entirely within the calling process -- always invoked as an
    isolated subprocess (see _launch_subprocess)."""
    cfg_brain = load_config("configs/brain.yaml")
    cfg_game = load_config("configs/game.yaml")
    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])

    device = torch.device(device_name)
    dtype = DTYPE_MAP[dtype_name]

    processed_dir = Path(cfg_game["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    songs = _load_real_songs(cfg_game, batch, min_frames=n_frames)
    if len(songs) < batch:
        raise RuntimeError(
            f"need {batch} train-split songs with >= {n_frames} frames, found {len(songs)}"
        )

    env = VecRhythmEnv(cfg_game, photo_map)
    env.reset(songs)
    zero_actions = np.zeros((batch, 6), dtype=bool)
    frames_list = []
    for _ in range(n_frames):
        obs, _, _, _ = env.step(zero_actions)
        frames_list.append(obs["frame"])
    frames_np = np.stack(frames_list)  # [n_frames, batch, H, W], rendered once, not timed
    frames_t_all = torch.from_numpy(frames_np).to(device)

    brain = ConnectomeBrain(graph, cfg_brain, device=device, dtype=dtype)
    readout = Readout(brain.n_dn, cfg_brain["readout"]["n_actions"]).to(device=device, dtype=dtype)
    retina_cfg = cfg_game["retina"]
    retina = TorchRetina.build(
        photo_map, cfg_game["frame_size"], retina_cfg["gain"], retina_cfg["ema_alpha"],
        retina_cfg["blur_sigma"], device,
    )
    params = list(brain.parameters()) + list(readout.parameters())

    def run(track_memory: bool) -> int:
        retina.reset()
        v = brain.init_state(batch)
        peak = _peak_mem_sample(device) if track_memory else 0
        logits = None
        ctx = torch.enable_grad() if backward else torch.no_grad()
        with ctx:
            for f in range(n_frames):
                # retina.step() always returns float32 (matches sim.py's
                # actual hot-path dtype, see retina_torch.py) -- cast to the
                # brain's dtype so e.g. a float16 brain doesn't hit a dtype
                # mismatch in index_add_ (found by the first run of this
                # script: "self (Half) and source (Float)").
                i_photo = retina.step(frames_t_all[f]).to(dtype=dtype)
                v, dn_rate_mean = brain.frame_step(
                    v, i_photo, n_substeps=cfg_brain["substeps_per_frame"]
                )
                logits = readout(dn_rate_mean)
                if track_memory:
                    peak = max(peak, _peak_mem_sample(device))
            if backward:
                logits.sum().backward()
                if track_memory:
                    peak = max(peak, _peak_mem_sample(device))
        return peak

    run(track_memory=False)  # warm-up
    _zero_grads(params)
    _empty_cache(device)

    _sync(device)
    t0 = time.perf_counter()
    run(track_memory=False)
    _sync(device)
    elapsed = time.perf_counter() - t0
    _zero_grads(params)
    _empty_cache(device)

    peak = run(track_memory=True)
    return {"elapsed": elapsed, "peak": peak}


def _launch_subprocess(args: list[str]) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": "."},
    )
    if result.returncode != 0:
        return {"error": (result.stderr.strip() or result.stdout.strip())[-2000:]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as e:
        return {"error": f"could not parse worker output ({e}); stdout={result.stdout[-500:]}"}


def _cli_worker() -> None:
    kind = sys.argv[1]
    if kind == "grid":
        _, _, device_name, dtype_name, batch_s, min_syn_s, backward_s, game_seconds_s = sys.argv
        result = _measure_brain_cell(
            device_name, dtype_name, int(batch_s), int(min_syn_s), backward_s == "1",
            float(game_seconds_s),
        )
    elif kind == "full":
        _, _, device_name, dtype_name, batch_s, backward_s, n_frames_s = sys.argv
        result = _measure_full_pipeline_cell(
            device_name, dtype_name, int(batch_s), backward_s == "1", int(n_frames_s)
        )
    else:
        raise ValueError(f"unknown worker kind: {kind}")
    print(json.dumps(result))


def brain_only_sweep(meta: dict, cfg_brain: dict) -> list[dict]:
    bench_cfg = cfg_brain["bench"]
    game_seconds = bench_cfg["game_seconds"]
    results: list[dict] = []

    for device_name in bench_cfg["devices"]:
        if device_name == "mps" and not torch.backends.mps.is_available():
            continue
        for dtype_name in bench_cfg["dtypes"]:
            for min_syn in bench_cfg["min_syn_values"]:
                for batch in bench_cfg["batches"]:
                    row = dict(device=device_name, dtype=dtype_name, batch=batch, min_syn=min_syn)
                    fwd = _launch_subprocess(
                        ["grid", device_name, dtype_name, str(batch), str(min_syn), "0", str(game_seconds)]
                    )
                    if "error" in fwd:
                        row["fwd_error"] = fwd["error"]
                    else:
                        row["fwd_time_s"] = fwd["elapsed"]
                        row["fwd_mem_bytes"] = fwd["peak"]
                    bwd = _launch_subprocess(
                        ["grid", device_name, dtype_name, str(batch), str(min_syn), "1", str(game_seconds)]
                    )
                    if "error" in bwd:
                        row["bwd_error"] = bwd["error"]
                    else:
                        row["fwd_bwd_time_s"] = bwd["elapsed"]
                        row["fwd_bwd_mem_bytes"] = bwd["peak"]
                    print(
                        f"[grid] device={device_name} dtype={dtype_name} batch={batch} "
                        f"min_syn={min_syn} fwd={row.get('fwd_time_s')} "
                        f"fwd_bwd={row.get('fwd_bwd_time_s')}"
                    )
                    results.append(row)
    return results


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**3):.2f} GiB"


def write_bench_md(
    grid_results: list[dict], full_pipeline_results: list[dict], chosen: dict | None
) -> None:
    lines = ["# Gate G3 benchmark (Phase 3a)\n"]
    lines.append(
        "Every cell below ran in its own subprocess, so peak-memory readings are isolated "
        "per cell (see this file's module docstring for why: process peak RSS never resets "
        "within a process, and MPS's caching allocator doesn't return memory to the driver "
        "without an explicit `empty_cache()` -- both make in-process, cross-cell peak sampling "
        "unreliable after the first large cell). MPS peak = `torch.mps.driver_allocated_memory()` "
        "polled every substep; CPU peak = process peak RSS (`ru_maxrss`, bytes on macOS). Not "
        "directly comparable to each other -- see D23 in docs/DECISIONS.md.\n"
    )

    lines.append("\n## Brain-only grid (device x dtype x batch x min_syn, 1 game-second)\n")
    lines.append(
        "| device | dtype | batch | min_syn | fwd (s) | fwd peak | fwd+bwd (s) | fwd+bwd peak |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in grid_results:
        fwd = f"{row['fwd_time_s']:.3f}" if "fwd_time_s" in row else f"error: {row.get('fwd_error')}"
        fwd_mem = _fmt_bytes(row["fwd_mem_bytes"]) if "fwd_mem_bytes" in row else ""
        bwd = (
            f"{row['fwd_bwd_time_s']:.3f}"
            if "fwd_bwd_time_s" in row
            else f"error: {row.get('bwd_error')}"
        )
        bwd_mem = _fmt_bytes(row["fwd_bwd_mem_bytes"]) if "fwd_bwd_mem_bytes" in row else ""
        lines.append(
            f"| {row['device']} | {row['dtype']} | {row['batch']} | {row['min_syn']} "
            f"| {fwd} | {fwd_mem} | {bwd} | {bwd_mem} |"
        )

    lines.append(
        "\n## Full pipeline (retina + brain substeps + readout), batch=8, min_syn=5, 1 game-second\n"
    )
    lines.append("| device | dtype | fwd (s) | fwd peak | fwd+bwd (s) | fwd+bwd peak | Gate G3 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in full_pipeline_results:
        if "error" in row:
            lines.append(f"| {row['device']} | {row['dtype']} | unsupported: {row['error']} | | | | FAIL |")
            continue
        gate_pass = (
            row["fwd_bwd_time_s"] <= GATE_G3_MAX_FWD_BWD_S
            and row["fwd_bwd_mem_bytes"] <= GATE_G3_MAX_PEAK_BYTES
        )
        lines.append(
            f"| {row['device']} | {row['dtype']} | {row['fwd_time_s']:.3f} "
            f"| {_fmt_bytes(row['fwd_mem_bytes'])} | {row['fwd_bwd_time_s']:.3f} "
            f"| {_fmt_bytes(row['fwd_bwd_mem_bytes'])} | {'PASS' if gate_pass else 'FAIL'} |"
        )

    lines.append("\n## Chosen device\n")
    if chosen is None:
        lines.append(
            f"**No config in the full-pipeline sweep met Gate G3** "
            f"(fwd+bwd <= {GATE_G3_MAX_FWD_BWD_S}s and peak <= {GATE_G3_MAX_PEAK_BYTES / (1024**3):.0f}GB "
            "at batch=8, min_syn=5). See docs/PLAN.md's Phase 3 fallbacks."
        )
    else:
        lines.append(
            f"**{chosen['device']} / {chosen['dtype']}**: fwd+bwd = {chosen['fwd_bwd_time_s']:.3f}s, "
            f"peak = {_fmt_bytes(chosen['fwd_bwd_mem_bytes'])} (threshold: "
            f"<={GATE_G3_MAX_FWD_BWD_S}s, <={GATE_G3_MAX_PEAK_BYTES / (1024**3):.0f}GB)."
        )

    Path("docs/bench.md").write_text("\n".join(lines) + "\n")


def pick_device(full_pipeline_results: list[dict]) -> dict | None:
    priority = [("mps", "float32"), ("mps", "float16"), ("cpu", "float32"), ("cpu", "float16")]
    by_key = {(r["device"], r["dtype"]): r for r in full_pipeline_results}
    for key in priority:
        row = by_key.get(key)
        if row is None or "error" in row:
            continue
        if (
            row["fwd_bwd_time_s"] <= GATE_G3_MAX_FWD_BWD_S
            and row["fwd_bwd_mem_bytes"] <= GATE_G3_MAX_PEAK_BYTES
        ):
            return row
    return None


def main() -> None:
    cfg_brain = load_config("configs/brain.yaml")
    cfg_game = load_config("configs/game.yaml")
    _, meta = load_graph_and_meta(cfg_brain["processed_dir"])

    print("=== brain-only grid sweep ===")
    grid_results = brain_only_sweep(meta, cfg_brain)

    print("\n=== full pipeline (retina + brain + readout), batch=8, min_syn=5 ===")
    full_pipeline_batch = cfg_brain["bench"]["full_pipeline_batch"]
    n_frames = round(cfg_brain["bench"]["game_seconds"] * cfg_game["fps"])
    full_pipeline_results = []
    for device_name in cfg_brain["bench"]["devices"]:
        if device_name == "mps" and not torch.backends.mps.is_available():
            continue
        for dtype_name in cfg_brain["bench"]["dtypes"]:
            row = dict(device=device_name, dtype=dtype_name)
            fwd = _launch_subprocess(
                ["full", device_name, dtype_name, str(full_pipeline_batch), "0", str(n_frames)]
            )
            bwd = _launch_subprocess(
                ["full", device_name, dtype_name, str(full_pipeline_batch), "1", str(n_frames)]
            )
            if "error" in fwd:
                row["error"] = fwd["error"]
                print(f"[full] device={device_name} dtype={dtype_name} FAILED (fwd): {fwd['error']}")
            elif "error" in bwd:
                row["error"] = bwd["error"]
                print(f"[full] device={device_name} dtype={dtype_name} FAILED (bwd): {bwd['error']}")
            else:
                row.update(
                    fwd_time_s=fwd["elapsed"], fwd_mem_bytes=fwd["peak"],
                    fwd_bwd_time_s=bwd["elapsed"], fwd_bwd_mem_bytes=bwd["peak"],
                )
                print(
                    f"[full] device={device_name} dtype={dtype_name} fwd={fwd['elapsed']:.3f}s "
                    f"fwd_bwd={bwd['elapsed']:.3f}s peak={_fmt_bytes(bwd['peak'])}"
                )
            full_pipeline_results.append(row)

    chosen = pick_device(full_pipeline_results)
    write_bench_md(grid_results, full_pipeline_results, chosen)

    if chosen is None:
        print(
            f"\nGate G3: NO config met the threshold "
            f"(fwd+bwd <= {GATE_G3_MAX_FWD_BWD_S}s, peak <= "
            f"{GATE_G3_MAX_PEAK_BYTES / (1024**3):.0f}GB) at batch=8, min_syn=5."
        )
    else:
        print(
            f"\nGate G3: chosen {chosen['device']}/{chosen['dtype']} -- "
            f"fwd+bwd={chosen['fwd_bwd_time_s']:.3f}s, peak={_fmt_bytes(chosen['fwd_bwd_mem_bytes'])}"
        )
    print("wrote docs/bench.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("grid", "full"):
        _cli_worker()
    else:
        main()
