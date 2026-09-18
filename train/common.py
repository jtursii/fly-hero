"""Shared helpers for train/bc.py and train/readout_only.py (PLAN Phase 3b):
label unpacking, note-dense evaluation excerpts, a fixed val set, and
strum pos_weight estimation from the actual sampler rather than a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flyhero.game.clip_sampler import ClipSampler
from flyhero.game.labels import compute_frame_labels
from flyhero.game.song import Song, load_song_merged, slice_song


def unpack_fret_targets(fret_target_uint8: np.ndarray) -> np.ndarray:
    """fret_target[T] uint8 lane-mask -> [T, 5] float32 0/1."""
    bits = np.zeros((len(fret_target_uint8), 5), dtype=np.float32)
    for lane in range(5):
        bits[:, lane] = (fret_target_uint8 >> lane) & 1
    return bits


def find_note_dense_window(notes_time_s: np.ndarray, duration_s: float, window_s: float, step_s: float = 1.0) -> float:
    """Deterministic (no randomness): the start time in [0, duration_s -
    window_s] maximizing the number of notes with time_s in
    [start, start+window_s). Ties broken by the earliest start (np.argmax's
    default), so the result is stable across calls/runs."""
    max_start = duration_s - window_s
    if max_start <= 0:
        return 0.0
    starts = np.arange(0.0, max_start, step_s)
    if len(starts) == 0:
        return 0.0
    notes_sorted = np.sort(notes_time_s.astype(np.float64))
    lo = np.searchsorted(notes_sorted, starts, side="left")
    hi = np.searchsorted(notes_sorted, starts + window_s, side="left")
    counts = hi - lo
    return float(starts[int(np.argmax(counts))])


@dataclass
class ValSongEntry:
    song: Song
    excerpt_start_s: float
    excerpt_s: float


def build_val_set(
    processed_dir: Path, difficulties: list[str], n_songs: int, excerpt_s: float,
    chord_merge_min_gap_s: float,
) -> dict[str, list[ValSongEntry]]:
    """A fixed (deterministic) set of up to n_songs val-split songs per
    difficulty, each with a fixed note-dense excerpt_s window -- same songs,
    same excerpt, every call (user requirement: "same excerpts every time").
    Sorted song_id order (not random) is what makes this deterministic."""
    splits = json.loads((processed_dir / "splits.json").read_text())
    val_ids = sorted(splits["val"])

    out: dict[str, list[ValSongEntry]] = {}
    for difficulty in difficulties:
        entries = []
        for sid in val_ids:
            path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
            if not path.exists():
                continue
            song = load_song_merged(path, chord_merge_min_gap_s)
            if song.duration_s < excerpt_s:
                continue
            excerpt_start = find_note_dense_window(song.notes["time_s"], song.duration_s, excerpt_s)
            entries.append(ValSongEntry(song=song, excerpt_start_s=excerpt_start, excerpt_s=excerpt_s))
            if len(entries) >= n_songs:
                break
        out[difficulty] = entries
    return out


def build_eval_clip(entry: ValSongEntry, burn_in_s: float) -> tuple[Song, float]:
    """(clip_song, burn_in_actual_s): clip_song covers
    [excerpt_start-burn_in_actual, excerpt_start+excerpt_s), where
    burn_in_actual = min(burn_in_s, excerpt_start_s) -- less burn-in only if
    the fixed excerpt starts too close to the song's own beginning."""
    burn_in_actual = min(burn_in_s, entry.excerpt_start_s)
    window_start = entry.excerpt_start_s - burn_in_actual
    duration = burn_in_actual + entry.excerpt_s
    clip = slice_song(entry.song, window_start, duration)
    return clip, burn_in_actual


def estimate_strum_pos_weight(
    sampler: ClipSampler, fps: int, hit_window_s: float, train_window_s: float, n_samples: int, rng: np.random.Generator,
) -> float:
    """Draws n_samples clips from `sampler` (the note-aware 90/10 sampler
    actually used for training) and computes pos_weight = n_negative /
    n_positive over the strum target across those clips' gradient windows --
    the real positive-frame rate under this sampler, not a fixed guess."""
    n_pos = 0
    n_neg = 0
    for _ in range(n_samples):
        song, start_s, _ = sampler.sample(rng)
        clip = slice_song(song, start_s, train_window_s)
        _, strum = compute_frame_labels(clip.notes, train_window_s, fps, hit_window_s)
        n_pos += int(strum.sum())
        n_neg += int(len(strum) - strum.sum())
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def make_training_clip(
    sampler: ClipSampler, burn_in_s: float, train_window_s: float, fps: int, hit_window_s: float,
    rng: np.random.Generator, fret_onset: str = "segment",
) -> tuple[Song, np.ndarray, np.ndarray]:
    """Draws one (song, start_time) from `sampler` and returns
    (burn_in_plus_window_song, fret_bits[T,5], strum[T]) where T =
    round(train_window_s * fps) covers only the gradient window (burn-in
    frames carry no loss)."""
    song, start_s, _ = sampler.sample(rng)
    window_start = start_s - burn_in_s
    clip = slice_song(song, window_start, burn_in_s + train_window_s)
    grad_only = slice_song(clip, burn_in_s, train_window_s)
    fret_target, strum = compute_frame_labels(grad_only.notes, train_window_s, fps, hit_window_s, fret_onset)
    return clip, unpack_fret_targets(fret_target), strum.astype(np.float32)


def save_checkpoint(
    path: Path, model, optimizer, step: int, difficulty_idx: int, pos_weight: float,
    numpy_rng: np.random.Generator, extra: dict, watchdog_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "difficulty_idx": difficulty_idx,
            "pos_weight": pos_weight,
            "numpy_rng_state": numpy_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "extra": extra,
            "watchdog_state": watchdog_state,
        },
        path,
    )
