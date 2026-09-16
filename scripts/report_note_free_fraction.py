"""Reports the library-wide (train split, all difficulties) fraction of each
song's valid gradient-window start times that fall in the 0-note bucket, and
the median across songs -- per the user's explicit request when reviewing
Phase 3b's note-aware clip sampler (flyhero/game/clip_sampler.py). Writes
docs/note_free_fraction.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from flyhero.game.clip_sampler import note_free_fraction_report
from flyhero.utils.config import load_config


def main() -> None:
    cfg_bc = load_config("configs/bc.yaml")
    cfg_game = load_config("configs/game.yaml")
    processed_dir = Path(cfg_bc["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())

    report = note_free_fraction_report(
        processed_dir, splits["train"], cfg_bc["curriculum"]["difficulties"],
        cfg_bc["burn_in_s"], cfg_bc["train_window_s"], cfg_game["fps"],
        cfg_game["chord_merge_min_gap_s"],
    )

    print(f"n (song, difficulty) pairs: {report['n_song_difficulty_pairs']}")
    print(f"median note-free fraction: {report['median_note_free_frac']:.4f}")
    print(f"mean note-free fraction:   {report['mean_note_free_frac']:.4f}")

    lines = ["# Note-free fraction report (Phase 3b)\n"]
    lines.append(
        f"For every train-split (song, difficulty) pair, the fraction of valid "
        f"{cfg_bc['train_window_s']}s gradient-window start times (burn_in="
        f"{cfg_bc['burn_in_s']}s, fps={cfg_game['fps']}) with 0 notes in the window, "
        f"restricted to windows with either 0 or >={cfg_bc['min_notes_with']} notes "
        "(matching what the sampler actually draws from; 1-note windows are excluded "
        "from the ratio).\n"
    )
    lines.append(f"\n**n = {report['n_song_difficulty_pairs']} (song, difficulty) pairs**")
    lines.append(f"\n**Library-wide median note-free fraction: {report['median_note_free_frac']:.4f}**")
    lines.append(f"\nMean: {report['mean_note_free_frac']:.4f}\n")

    by_difficulty: dict[str, list[float]] = {}
    for row in report["per_song"]:
        by_difficulty.setdefault(row["difficulty"], []).append(row["note_free_frac"])
    lines.append("\n| difficulty | n songs | median note-free frac |")
    lines.append("|---|---|---|")
    import numpy as np

    for diff, fracs in by_difficulty.items():
        lines.append(f"| {diff} | {len(fracs)} | {np.median(fracs):.4f} |")

    Path("docs/note_free_fraction.md").write_text("\n".join(lines) + "\n")
    print("\nwrote docs/note_free_fraction.md")


if __name__ == "__main__":
    main()
