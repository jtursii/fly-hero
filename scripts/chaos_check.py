"""Phase 3b diagnostic: chaos check at gain0=10, bias_init=0.005 (current
configs/brain.yaml values) -- reliability and perturbation-sensitivity, on
top of Phase 3a's existing gain0/bias_init stability sweep (docs/
gain0_stability.md), which this script reads rather than re-runs.

(a) Reliability: the same 5s train-split Medium segment is run twice, each
preceded by a different 2s burn-in on a different train-split song (so v
starts each run from a different nonzero recurrent state, not both from
v=0). The retina is reset fresh at the start of the shared 5s segment in
both runs (same visual input stream in both runs) -- isolating "does a
different v history change subsequent behavior" from "did the two runs also
see different pixels." Reports per-DN Pearson correlation over the last 3s
of the shared segment, and the mean.

(b) Perturbation: a 3s blank-highway run (D24: rendered background/lanes/
strikeline, zero notes) from v=0, with vs without small Gaussian noise added
to the photoreceptor input current every frame (std = 1% of a typical
photoreceptor-current magnitude, measured from a real train-split song's
retina output rather than hardcoded). Reports relative DN divergence
(||dn_noisy - dn_clean|| / ||dn_clean||) at the final frame.

(c) Reads docs/gain0_stability.md's existing per-gain0 pass/fail column
(already produced by scripts/gain0_sweep.py in Phase 3a) rather than
re-sweeping.

No fixed numeric gate was given for the perturbation divergence -- this
script reports the raw number and flags it as an open question rather than
silently pass/fail it (see the Phase 3b plan's "ambiguities" list).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.retina_torch import TorchRetina
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import NOTE_DTYPE, Song, load_song_merged
from flyhero.utils.config import load_config

RELIABILITY_CORR_MIN = 0.5


def _train_medium_songs(cfg_game: dict, min_duration_s: float, min_notes_in_window: int = 10) -> list[Song]:
    processed_dir = Path(cfg_game["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())
    out = []
    for sid in splits["train"]:
        p = processed_dir / "songs" / f"{sid}_Medium.npz"
        if not p.exists():
            continue
        song = load_song_merged(p, cfg_game["chord_merge_min_gap_s"])
        if song.duration_s < min_duration_s:
            continue
        n_notes = int((song.notes["time_s"] < min_duration_s).sum())
        if n_notes >= min_notes_in_window:
            out.append(song)
    return out


def _capped(song: Song, duration_s: float, blank: bool) -> Song:
    return Song(
        song_id=song.song_id, title=song.title, artist=song.artist, charter=song.charter,
        difficulty=song.difficulty,
        notes=np.zeros(0, dtype=NOTE_DTYPE) if blank else song.notes,
        duration_s=duration_s, chart_offset_s=song.chart_offset_s, ini_delay_s=song.ini_delay_s,
    )


def _run_frames(
    env: VecRhythmEnv, retina: TorchRetina, brain: ConnectomeBrain, song: Song, n_frames: int,
    substeps: int, v: torch.Tensor, reset_retina: bool, noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs n_frames of `song` starting from state `v` (carried in, not
    reinitialized). Returns (v_final, dn_trace[n_frames, n_dn])."""
    env.reset([song])
    if reset_retina:
        retina.reset()
    zero_actions = np.zeros((1, 6), dtype=bool)
    dn_trace = torch.zeros(n_frames, brain.n_dn)
    with torch.no_grad():
        for f in range(n_frames):
            obs, _, _, _ = env.step(zero_actions)
            frame_t = torch.from_numpy(obs["frame"])
            i_photo = retina.step(frame_t)
            if noise_std > 0.0:
                i_photo = i_photo + torch.randn_like(i_photo) * noise_std
            v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=substeps)
            dn_trace[f] = dn_rate_mean[0]
    return v, dn_trace


def check_reliability(
    cfg_brain: dict, cfg_game: dict, graph: dict, photo_map, target_song: Song,
    burn_in_songs: list[Song],
) -> dict:
    fps = cfg_game["fps"]
    substeps = cfg_brain["substeps_per_frame"]
    n_burn_in = round(2.0 * fps)
    n_target = round(5.0 * fps)
    n_last_3s = round(3.0 * fps)
    target_5s = _capped(target_song, 5.0, blank=False)

    traces = []
    for burn_song in burn_in_songs:
        brain = ConnectomeBrain(graph, cfg_brain, device="cpu", dtype=torch.float32)
        retina = TorchRetina.build(
            photo_map, cfg_game["frame_size"], cfg_game["retina"]["gain"],
            cfg_game["retina"]["ema_alpha"], cfg_game["retina"]["blur_sigma"], device="cpu",
        )
        env = VecRhythmEnv(cfg_game, photo_map)

        burn_capped = _capped(burn_song, 2.0, blank=False)
        v0 = brain.init_state(1)
        v_after_burn_in, _ = _run_frames(
            env, retina, brain, burn_capped, n_burn_in, substeps, v0, reset_retina=True,
        )

        _, dn_trace = _run_frames(
            env, retina, brain, target_5s, n_target, substeps, v_after_burn_in, reset_retina=True,
        )
        traces.append(dn_trace)

    trace_a, trace_b = traces

    # Bug check (user-requested): if the burn-in state weren't actually
    # carried into the target window, frame 0's DN readout would be
    # identical between the two runs (both would just be "brain freshly
    # seeing the target song's first frame") -- max|delta DN| at frame 0
    # must be nonzero, and should decay over the following frames as the
    # burn-in-state difference washes out (see the module docstring's
    # tau-based explanation for why the last-3s traces end up bit-identical).
    n_first_60 = min(60, trace_a.shape[0])
    per_frame_max_diff = (trace_a[:n_first_60] - trace_b[:n_first_60]).abs().max(dim=1).values.tolist()

    last_a = trace_a[-n_last_3s:].numpy()  # [n_last_3s, n_dn]
    last_b = trace_b[-n_last_3s:].numpy()

    n_dn = last_a.shape[1]
    corrs = np.full(n_dn, np.nan)
    for d in range(n_dn):
        a, b = last_a[:, d], last_b[:, d]
        if a.std() > 1e-9 and b.std() > 1e-9:
            corrs[d] = np.corrcoef(a, b)[0, 1]
    mean_corr = float(np.nanmean(corrs))
    frac_nan = float(np.isnan(corrs).mean())
    valid = corrs[~np.isnan(corrs)]

    return dict(
        target_song=f"{target_song.artist} - {target_song.title}",
        burn_in_songs=[f"{s.artist} - {s.title}" for s in burn_in_songs],
        mean_corr_last_3s=mean_corr,
        frac_dn_constant=frac_nan,  # DNs with ~zero variance in one/both runs -- corr undefined
        n_dn=n_dn,
        n_valid_dn=int(len(valid)),
        min_corr=float(valid.min()) if len(valid) else float("nan"),
        median_corr=float(np.median(valid)) if len(valid) else float("nan"),
        max_abs_diff_trace_a_b=float(np.abs(last_a - last_b).max()),
        max_val_trace_a=float(np.abs(last_a).max()),
        per_frame_max_diff_first_60=per_frame_max_diff,
    )


def _measure_typical_input_magnitude(
    cfg_game: dict, photo_map, song: Song, duration_s: float,
) -> float:
    """Samples over `duration_s` (not just the first few seconds) since a
    song's intro is commonly silent/note-free for several seconds (lookahead
    is only 1s, and the first note can be much later -- e.g. real songs in
    this library have first notes past 20s, see docs/PROGRESS.md's Phase 2
    timing spot check) -- a short early-only window can measure near-zero
    magnitude simply because no notes are on screen yet, not because the
    signal is genuinely small."""
    retina = TorchRetina.build(
        photo_map, cfg_game["frame_size"], cfg_game["retina"]["gain"],
        cfg_game["retina"]["ema_alpha"], cfg_game["retina"]["blur_sigma"], device="cpu",
    )
    env = VecRhythmEnv(cfg_game, photo_map)
    env.reset([song])
    zero_actions = np.zeros((1, 6), dtype=bool)
    n_frames = round(duration_s * cfg_game["fps"])
    mags = []
    with torch.no_grad():
        for _ in range(n_frames):
            obs, _, _, _ = env.step(zero_actions)
            i_photo = retina.step(torch.from_numpy(obs["frame"]))
            nonzero = i_photo[i_photo.abs() > 1e-12]
            if nonzero.numel() > 0:
                mags.append(nonzero.abs().mean().item())
    return float(np.mean(mags)) if mags else 0.0


def check_perturbation(
    cfg_brain: dict, cfg_game: dict, graph: dict, photo_map, reference_song: Song,
) -> dict:
    fps = cfg_game["fps"]
    substeps = cfg_brain["substeps_per_frame"]
    n_frames = round(3.0 * fps)

    typical_mag = _measure_typical_input_magnitude(
        cfg_game, photo_map, reference_song, duration_s=min(reference_song.duration_s, 20.0)
    )
    noise_std = 0.01 * typical_mag

    blank_song = _capped(reference_song, 3.0, blank=True)

    def run(noise: float) -> torch.Tensor:
        brain = ConnectomeBrain(graph, cfg_brain, device="cpu", dtype=torch.float32)
        retina = TorchRetina.build(
            photo_map, cfg_game["frame_size"], cfg_game["retina"]["gain"],
            cfg_game["retina"]["ema_alpha"], cfg_game["retina"]["blur_sigma"], device="cpu",
        )
        env = VecRhythmEnv(cfg_game, photo_map)
        v0 = brain.init_state(1)
        torch.manual_seed(0)
        _, dn_trace = _run_frames(
            env, retina, brain, blank_song, n_frames, substeps, v0, reset_retina=True, noise_std=noise,
        )
        return dn_trace

    dn_clean = run(0.0)
    dn_noisy = run(noise_std)

    diff_norm = (dn_noisy[-1] - dn_clean[-1]).norm().item()
    clean_norm = dn_clean[-1].norm().item()
    rel_divergence = diff_norm / clean_norm if clean_norm > 1e-12 else float("nan")

    return dict(
        typical_input_magnitude=typical_mag,
        noise_std=noise_std,
        clean_dn_norm_at_3s=clean_norm,
        diff_norm_at_3s=diff_norm,
        relative_divergence_at_3s=rel_divergence,
    )


def read_gain0_sweep_table() -> list[dict]:
    """Parses docs/gain0_stability.md's markdown table (written by Phase 3a's
    scripts/gain0_sweep.py) rather than re-running the sweep."""
    text = Path("docs/gain0_stability.md").read_text()
    rows = []
    for line in text.splitlines():
        m = re.match(r"\|\s*([\d.]+)\s*\|.*\|\s*(True|False)\s*\|\s*$", line)
        if m:
            rows.append(dict(gain0=float(m.group(1)), passes=m.group(2) == "True"))
    return rows


def main() -> None:
    cfg_brain = load_config("configs/brain.yaml")
    cfg_game = load_config("configs/game.yaml")
    graph, meta = load_graph_and_meta(cfg_brain["processed_dir"])
    photo_map = build_photoreceptor_map(graph, meta["type_names"])

    print(f"gain0={cfg_brain['gain0']} bias_init={cfg_brain['bias_init']}")

    songs = _train_medium_songs(cfg_game, min_duration_s=7.0, min_notes_in_window=10)
    if len(songs) < 3:
        raise RuntimeError(f"need >=3 distinct train-split Medium songs >=7s, found {len(songs)}")
    target_song = songs[0]
    burn_in_songs = [songs[1], songs[2]]

    print("\n=== (a) reliability ===")
    reliability = check_reliability(cfg_brain, cfg_game, graph, photo_map, target_song, burn_in_songs)
    diffs = reliability["per_frame_max_diff_first_60"]
    print(f"per-frame max|delta DN|, first {len(diffs)} frames of the target window:")
    print("  " + ", ".join(f"{d:.4g}" for d in diffs))
    if diffs[0] == 0.0:
        raise RuntimeError(
            "BUG CHECK FAILED: max|delta DN| at frame 0 is exactly 0 -- the burn-in state is not "
            "being carried into the target window (both runs saw an identical fresh state), not a "
            "real reliability result. Stopping per the user's explicit instruction."
        )
    decay_frame = next((i for i, d in enumerate(diffs) if d < 1e-6), None)
    print(f"frame 0: {diffs[0]:.6g} (nonzero, as expected -- burn-in state is carried)")
    print(f"decays below 1e-6 at frame: {decay_frame}")
    reliability_summary = {k: v for k, v in reliability.items() if k != "per_frame_max_diff_first_60"}
    print(json.dumps(reliability_summary, indent=2))
    reliability_ok = reliability["mean_corr_last_3s"] >= RELIABILITY_CORR_MIN

    print("\n=== (b) perturbation ===")
    perturbation = check_perturbation(cfg_brain, cfg_game, graph, photo_map, target_song)
    print(json.dumps(perturbation, indent=2))

    print("\n=== (c) gain0 sweep (from docs/gain0_stability.md) ===")
    sweep_rows = read_gain0_sweep_table()
    for r in sweep_rows:
        print(f"  gain0={r['gain0']}: {'PASS' if r['passes'] else 'FAIL'}")

    print("\n=== summary ===")
    print(f"reliability mean_corr_last_3s={reliability['mean_corr_last_3s']:.4f} (threshold {RELIABILITY_CORR_MIN}): {'OK' if reliability_ok else 'FAIL -- STOP per invariant 7'}")
    print(f"perturbation relative_divergence_at_3s={perturbation['relative_divergence_at_3s']:.4f} (no fixed gate given -- flagged as an ambiguity)")

    out = dict(reliability=reliability, perturbation=perturbation, gain0_sweep=sweep_rows)
    Path("docs/chaos_check.md").write_text(
        "# Chaos check (Phase 3b diagnostic)\n\n"
        f"gain0={cfg_brain['gain0']}, bias_init={cfg_brain['bias_init']}\n\n"
        "## (a) Reliability\n"
        f"Target song: {reliability['target_song']}\n\n"
        f"Burn-in songs: {reliability['burn_in_songs']}\n\n"
        f"Mean per-DN Pearson correlation over the last 3s: **{reliability['mean_corr_last_3s']:.4f}** "
        f"(threshold {RELIABILITY_CORR_MIN}, {reliability['n_dn']} DNs, "
        f"{reliability['frac_dn_constant']:.4f} fraction with near-zero variance in a run, excluded from the mean)\n\n"
        f"Bug check: per-frame max\\|delta DN\\| at frame 0 = **{diffs[0]:.6g}** (nonzero -- burn-in "
        f"state confirmed carried into the target window), decays below 1e-6 at frame **{decay_frame}**. "
        f"First {len(diffs)} frames: {', '.join(f'{d:.4g}' for d in diffs)}\n\n"
        "## (b) Perturbation\n"
        f"Typical photoreceptor input magnitude (measured on {reliability['target_song']}): "
        f"{perturbation['typical_input_magnitude']:.6f}; noise std (1%): {perturbation['noise_std']:.6f}\n\n"
        f"Relative DN divergence at 3s: **{perturbation['relative_divergence_at_3s']:.4f}** "
        "(no fixed numeric gate given by the user for this check)\n\n"
        "## (c) gain0 sweep (existing, from Phase 3a)\n"
        + "\n".join(f"- gain0={r['gain0']}: {'PASS' if r['passes'] else 'FAIL'}" for r in sweep_rows)
        + "\n"
    )
    print("\nwrote docs/chaos_check.md")


if __name__ == "__main__":
    main()
