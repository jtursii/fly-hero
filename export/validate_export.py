"""Phase 6 task 3 / Gate G6: validate what export_web.py wrote.

Checks, in order:
  1. Re-score: decode actions.bin back into (held frets, strum) per frame and
     re-score it against manifest.json's notes with rules.score_playthrough
     (invariant 4's one implementation). The resulting hit_rate must equal
     the manifest's exactly -- this is the check that the *exported* bytes,
     not just the in-memory recording, reproduce the reported number.
  2. Size budgets (D45: <= 21 MB per song, <= 60 MB total under data/).
  3. Audio placement: the only audio under web/ may be the D37 showcase
     MP3s, one per exported song, and nothing else.
  4. Every file the manifests reference exists and has the length the
     manifest's frame counts imply, and the shared brain files match
     brain.json's neuron and flow-edge counts.

Usage: uv run python -m export.validate_export --config configs/export.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyhero.game.rules import score_playthrough
from flyhero.game.song import NOTE_DTYPE
from flyhero.utils.config import load_config

MB = 1024 * 1024
PER_SONG_BUDGET_MB = 21.0  # D45 (was PLAN's 15.0)
TOTAL_BUDGET_MB = 60.0
AUDIO_SUFFIXES = {".mp3", ".ogg", ".wav", ".opus", ".m4a", ".flac", ".aac", ".wma"}


def unpack_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint8 [F] bitmask -> (held_mask[F] int64 over the 5 fret bits,
    strum[F] bool). The inverse of export_web.py's packing."""
    held = (actions & 0b11111).astype(np.int64)
    strum = ((actions >> 5) & 1).astype(bool)
    return held, strum


def notes_from_manifest(rows: list[list]) -> np.ndarray:
    notes = np.zeros(len(rows), dtype=NOTE_DTYPE)
    for i, (t, lane, sustain) in enumerate(rows):
        notes[i]["time_s"] = t
        notes[i]["lane_mask"] = lane
        notes[i]["sustain_s"] = sustain
    return notes


def validate_song(song_dir: Path, cfg_game: dict, failures: list[str]) -> float:
    manifest = json.loads((song_dir / "manifest.json").read_text())
    sid = manifest["song_id"]
    actions = np.fromfile(song_dir / "actions.bin", dtype=np.uint8)
    if len(actions) != manifest["frames"]:
        failures.append(f"{sid}: actions.bin has {len(actions)} frames, manifest says {manifest['frames']}")
        return 0.0

    held, strum = unpack_actions(actions)
    notes = notes_from_manifest(manifest["notes"])
    metrics, _ = score_playthrough(notes, held, strum, manifest["fps"], cfg_game["hit_window_s"])

    for key in ("hit_rate", "n_hits", "n_misses", "n_overstrums", "n_notes"):
        got, want = metrics[key], manifest[key]
        if abs(float(got) - float(want)) > 1e-9:
            failures.append(f"{sid}: re-scored {key} = {got} but manifest says {want}")

    n_act = manifest["activity_frames"]
    expect = {
        "activity.bin": n_act * (manifest["activity_slots"] // 2),
        "retina.bin": n_act * manifest["retina_channels"],
        "probs.bin": manifest["frames"] * 6,
    }
    for name, want_bytes in expect.items():
        got_bytes = (song_dir / name).stat().st_size
        if got_bytes != want_bytes:
            failures.append(f"{sid}: {name} is {got_bytes} bytes, expected {want_bytes}")

    size_mb = sum(f.stat().st_size for f in song_dir.iterdir()) / MB
    if size_mb > PER_SONG_BUDGET_MB:
        failures.append(f"{sid}: {size_mb:.1f} MB exceeds the {PER_SONG_BUDGET_MB:.0f} MB per-song budget")
    print(f"  {manifest['artist']} - {manifest['title']} [{manifest['seen_label']}]: "
          f"re-scored hit_rate {metrics['hit_rate']:.4f} == manifest, {size_mb:.1f} MB")
    return size_mb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/export.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg_game = load_config("configs/game.yaml")
    web_dir = Path(cfg["web_dir"])
    data_dir = web_dir / "data"
    failures: list[str] = []

    brain = json.loads((data_dir / "brain.json").read_text())
    n = brain["n_neurons"]
    want_flow = brain["flow_edges"]["count"] * 3 * 4
    for name, want in (("positions.bin", n * 3 * 4), ("classes.bin", n), ("pos_source.bin", n),
                       ("flow_edges.bin", want_flow)):
        got = (data_dir / name).stat().st_size
        if got != want:
            failures.append(f"{name} is {got} bytes, expected {want}")

    song_dirs = sorted((data_dir / "songs").iterdir())
    print(f"{len(song_dirs)} songs under {data_dir}")
    for song_dir in song_dirs:
        validate_song(song_dir, cfg_game, failures)

    total_mb = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file()) / MB
    if total_mb > TOTAL_BUDGET_MB:
        failures.append(f"data/ total {total_mb:.1f} MB exceeds the {TOTAL_BUDGET_MB:.0f} MB budget")

    # D37: the only audio allowed under web/ is one showcase MP3 per song.
    expected_audio = {web_dir / "audio" / f"{d.name}.mp3" for d in song_dirs}
    found_audio = {f for f in web_dir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES}
    for extra in sorted(found_audio - expected_audio):
        failures.append(f"unexpected audio under web/: {extra} (D37 allows only the showcase MP3s)")
    missing_audio = sorted(p for p in expected_audio if not p.exists())
    audio_mb = sum(f.stat().st_size for f in found_audio) / MB

    print(f"data/ total {total_mb:.1f} MB (budget {TOTAL_BUDGET_MB:.0f}); "
          f"audio {audio_mb:.1f} MB in {len(found_audio)} file(s)"
          + (f"; missing audio: {[p.name for p in missing_audio]}" if missing_audio else ""))

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nG6 validation PASSED")


if __name__ == "__main__":
    main()
