"""gain0 stability sweep (PLAN Phase 3 task 4 + D24's added checks): for each
candidate gain0 in configs/brain.yaml's stability_sweep, run a fresh
untrained ConnectomeBrain for 10s of a real train-split Medium song (never
test-split, per invariant 5) and report:

  - fraction of neurons with r > active_r_threshold (last 1s, steady-state)
  - max |v| over the whole run
  - a NaN check over the whole run
  - signal reach: fraction of DNs whose activity differs (above
    reach_threshold) between this run and a "blank highway" run (same
    rendered background/lanes/strikeline, zero notes -- D24) of the same
    song/duration
  - median input->DN latency (ms), over DNs that cleared reach_threshold, of
    the first frame their real-vs-blank difference exceeds it

Hard stop (invariant 7): if no gain0 achieves DN reach > reach_frac_min
without saturation (active fraction exceeding active_frac_max), this script
stops and reports rather than silently picking a fallback -- there is no
fallback list for stability in docs/PLAN.md, unlike the throughput gate, so
that's a genuine open question for the user for stopped runs.

No gradients are needed here (this only characterizes untrained dynamics),
so everything runs under torch.no_grad() on CPU for simplicity/determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import NOTE_DTYPE, Song, load_song_merged
from flyhero.utils.config import load_config


def _pick_train_medium_song(cfg_game: dict, min_duration_s: float, min_notes_in_window: int = 10) -> Song:
    """A train-split Medium song long enough for the sweep AND with enough
    notes inside the first min_duration_s seconds -- otherwise the "real"
    condition could look nearly identical to the blank highway simply
    because the song hasn't started yet, weakening the signal-reach check
    for reasons unrelated to gain0."""
    processed_dir = Path(cfg_game["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())
    for sid in splits["train"]:
        p = processed_dir / "songs" / f"{sid}_Medium.npz"
        if p.exists():
            song = load_song_merged(p, cfg_game["chord_merge_min_gap_s"])
            if song.duration_s < min_duration_s:
                continue
            n_notes_in_window = int((song.notes["time_s"] < min_duration_s).sum())
            if n_notes_in_window >= min_notes_in_window:
                return song
    raise RuntimeError(
        f"no train-split Medium song >= {min_duration_s}s with >= {min_notes_in_window} "
        "notes in that window found"
    )


def _capped(song: Song, duration_s: float, blank: bool) -> Song:
    return Song(
        song_id=song.song_id,
        title=song.title,
        artist=song.artist,
        charter=song.charter,
        difficulty=song.difficulty,
        notes=np.zeros(0, dtype=NOTE_DTYPE) if blank else song.notes,
        duration_s=duration_s,
        chart_offset_s=song.chart_offset_s,
        ini_delay_s=song.ini_delay_s,
    )


def _run_condition(
    brain: ConnectomeBrain,
    retina: TorchRetina,
    env: VecRhythmEnv,
    song: Song,
    n_frames: int,
    substeps_per_frame: int,
    active_r_threshold: float,
) -> dict:
    env.reset([song])
    retina.reset()
    v = brain.init_state(1)
    zero_actions = np.zeros((1, 6), dtype=bool)

    max_abs_v = 0.0
    nan_found = False
    frac_active_history: list[float] = []
    dn_trace = torch.zeros(n_frames, brain.n_dn)

    with torch.no_grad():
        for f in range(n_frames):
            obs, _, _, _ = env.step(zero_actions)
            frame_t = torch.from_numpy(obs["frame"])
            i_photo = retina.step(frame_t)
            v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=substeps_per_frame)
            r = torch.clamp(torch.relu(v), 0.0, 1.0)
            if torch.isnan(v).any() or torch.isnan(r).any():
                nan_found = True
            max_abs_v = max(max_abs_v, v.abs().max().item())
            frac_active_history.append((r > active_r_threshold).float().mean().item())
            dn_trace[f] = dn_rate_mean[0]

    return dict(
        max_abs_v=max_abs_v,
        nan_found=nan_found,
        frac_active_history=frac_active_history,
        dn_trace=dn_trace,
    )


def _signal_reach(
    real_dn_trace: torch.Tensor, blank_dn_trace: torch.Tensor, reach_threshold: float, fps: float
) -> tuple[float, float]:
    diff = (real_dn_trace - blank_dn_trace).abs()  # [n_frames, n_dn]
    max_diff = diff.max(dim=0).values
    reached = max_diff > reach_threshold
    reach_frac = float(reached.float().mean().item())
    if not reached.any():
        return reach_frac, float("nan")

    latencies_ms = []
    for d in reached.nonzero(as_tuple=True)[0].tolist():
        first_frame = int((diff[:, d] > reach_threshold).nonzero(as_tuple=True)[0][0])
        latencies_ms.append(first_frame / fps * 1000.0)
    return reach_frac, float(np.median(latencies_ms))


def main() -> None:
    cfg_brain = load_config("configs/brain.yaml")
    cfg_game = load_config("configs/game.yaml")
    sweep_cfg = cfg_brain["stability_sweep"]

    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])
    photo_map = build_photoreceptor_map(graph, meta["type_names"])

    song = _pick_train_medium_song(cfg_game, sweep_cfg["song_duration_s"])
    print(f"song: {song.artist} - {song.title} ({song.difficulty}, train split)")

    duration_s = sweep_cfg["song_duration_s"]
    n_frames = round(duration_s * cfg_game["fps"])
    real_song = _capped(song, duration_s, blank=False)
    blank_song = _capped(song, duration_s, blank=True)

    env = VecRhythmEnv(cfg_game, photo_map)
    retina_cfg = cfg_game["retina"]

    rows = []
    for gain0 in sweep_cfg["gain0_values"]:
        cfg = dict(cfg_brain)
        cfg["gain0"] = gain0
        brain = ConnectomeBrain(graph, cfg, device="cpu", dtype=torch.float32)
        retina = TorchRetina.build(
            photo_map, cfg_game["frame_size"], retina_cfg["gain"], retina_cfg["ema_alpha"],
            retina_cfg["blur_sigma"], device="cpu",
        )

        real_stats = _run_condition(
            brain, retina, env, real_song, n_frames, cfg_brain["substeps_per_frame"],
            sweep_cfg["active_r_threshold"],
        )
        blank_stats = _run_condition(
            brain, retina, env, blank_song, n_frames, cfg_brain["substeps_per_frame"],
            sweep_cfg["active_r_threshold"],
        )

        last_1s = max(int(round(cfg_game["fps"])), 1)
        active_frac = float(np.mean(real_stats["frac_active_history"][-last_1s:]))
        reach_frac, median_latency_ms = _signal_reach(
            real_stats["dn_trace"], blank_stats["dn_trace"], sweep_cfg["reach_threshold"],
            cfg_game["fps"],
        )
        saturated = active_frac > sweep_cfg["active_frac_max"]
        nan_found = real_stats["nan_found"] or blank_stats["nan_found"]
        passes = (reach_frac > sweep_cfg["reach_frac_min"]) and not saturated and not nan_found

        row = dict(
            gain0=gain0,
            active_frac=active_frac,
            max_abs_v=max(real_stats["max_abs_v"], blank_stats["max_abs_v"]),
            nan_found=nan_found,
            reach_frac=reach_frac,
            median_latency_ms=median_latency_ms,
            saturated=saturated,
            passes=passes,
        )
        rows.append(row)
        print(
            f"gain0={gain0}: active_frac={active_frac:.4f} max|v|={row['max_abs_v']:.4f} "
            f"nan={nan_found} reach_frac={reach_frac:.4f} "
            f"median_latency_ms={median_latency_ms:.2f} saturated={saturated} pass={passes}"
        )

    any_pass = any(r["passes"] for r in rows)

    lines = ["# gain0 stability sweep (Phase 3a, D24)\n"]
    lines.append(f"Song: {song.artist} - {song.title} ({song.difficulty}, train split), {duration_s}s.\n")
    lines.append(
        "\"Blank highway\" = same rendered background/lanes/strikeline, zero notes (D24). "
        f"active_r_threshold={sweep_cfg['active_r_threshold']}, "
        f"reach_threshold={sweep_cfg['reach_threshold']}, "
        f"active_frac_max (saturation bar)={sweep_cfg['active_frac_max']}, "
        f"reach_frac_min (hard-stop bar)={sweep_cfg['reach_frac_min']}.\n"
    )
    lines.append(
        "\n| gain0 | active_frac (last 1s) | max\\|v\\| | NaN | reach_frac | "
        "median latency (ms) | saturated | pass |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['gain0']} | {r['active_frac']:.4f} | {r['max_abs_v']:.4f} | {r['nan_found']} "
            f"| {r['reach_frac']:.4f} | {r['median_latency_ms']:.2f} | {r['saturated']} "
            f"| {r['passes']} |"
        )

    if any_pass:
        lines.append(
            f"\n**At least one gain0 clears reach_frac > {sweep_cfg['reach_frac_min']} without "
            "saturation or NaN.**"
        )
    else:
        lines.append(
            f"\n**STOPPED (invariant 7): no gain0 achieves DN reach > "
            f"{sweep_cfg['reach_frac_min']} without saturation.** No fallback list exists for "
            "stability in docs/PLAN.md (unlike the throughput gate) -- this needs the user's "
            "input: widen the gain0 search range, adjust tau_init_s, or revisit dt/substep count."
        )
    Path("docs/gain0_stability.md").write_text("\n".join(lines) + "\n")
    print("\nwrote docs/gain0_stability.md")

    if not any_pass:
        print(
            "\nSTOPPED (invariant 7): no gain0 in the sweep achieves DN reach > "
            f"{sweep_cfg['reach_frac_min']} without saturation. See docs/gain0_stability.md."
        )


if __name__ == "__main__":
    main()
