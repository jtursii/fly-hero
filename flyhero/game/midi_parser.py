"""Parser for Clone Hero .mid files (PLAN Phase 2 task 3).

Real game-ripped .mid files in this library commonly have out-of-range MIDI
data bytes inside proprietary SysEx payloads (never inside note_on/note_off
messages -- verified in library/scan.py's check_clip_corruption_note_impact);
mido.MidiFile(..., clip=True) is required to parse them at all.

v1 rule simplifications: star power, solo markers, forcing markers, and
SysEx open/tap markers are ignored (not decoded, only counted). Fret note
numbers per difficulty reuse flyhero.library.scan.NOTE_RANGES, the same
windows used at Phase-0 inventory time. .mid notes never carry a forced/tap
flag (that requires decoding the ignored SysEx), so ParsedDifficulty.flags is
always 0 here, and there is no "open note" concept to drop -- it's simply
never decoded.

The whole PART GUITAR track is scanned once (not once per difficulty): a
note number belongs to at most one difficulty's range, so a single pass can
build all four difficulties' note arrays together. `ignored_counts`
(sysex/out-of-range-note counts) is therefore file-scoped, not genuinely
per-difficulty -- the same dict is attached to every returned
ParsedDifficulty for a uniform shape with chart_parser.py's per-section
counts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mido

from flyhero.game.song import ParsedDifficulty, build_notes_array
from flyhero.game.tempo import build_tempo_map
from flyhero.library.scan import NOTE_RANGES

TRACK_NAME = "PART GUITAR"


def _find_guitar_track(midi: mido.MidiFile):
    for track in midi.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "track_name":
                if msg.name.strip() == TRACK_NAME:
                    return track
                break
    return None


def _collect_tempo_events(midi: mido.MidiFile) -> tuple[list[tuple[int, float]], bool]:
    """(events, warn) where warn is True if any set_tempo was found outside
    track 0 -- real files are supposed to keep tempo on the conductor track,
    but that's a convention, not something mido enforces."""
    events: list[tuple[int, float]] = []
    warn = False
    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.is_meta and msg.type == "set_tempo":
                events.append((abs_tick, mido.tempo2bpm(msg.tempo)))
                if track_idx != 0:
                    warn = True
    return events, warn


def _parse_guitar_track(
    track,
    note_ranges: dict[str, range],
    tick_to_seconds,
    resolution: int,
    sustain_min_beats: float,
) -> dict[str, ParsedDifficulty]:
    ignored_counts = {"ignored_notes": 0, "sysex_ignored": 0}
    open_starts: dict[str, dict[int, int]] = {d: {} for d in note_ranges}
    events_by_tick: dict[str, dict[int, dict[int, int]]] = {
        d: defaultdict(dict) for d in note_ranges
    }

    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == "sysex":
            ignored_counts["sysex_ignored"] += 1
            continue
        if msg.type not in ("note_on", "note_off"):
            continue

        matched_diff = next((d for d, rng in note_ranges.items() if msg.note in rng), None)
        if matched_diff is None:
            ignored_counts["ignored_notes"] += 1
            continue

        fret = msg.note - note_ranges[matched_diff].start
        is_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        starts = open_starts[matched_diff]
        if is_off:
            start_tick = starts.pop(fret, None)
            if start_tick is None:
                continue  # stray note_off with no matching note_on
            events_by_tick[matched_diff][start_tick][fret] = abs_tick - start_tick
        else:
            starts[fret] = abs_tick

    result: dict[str, ParsedDifficulty] = {}
    for diff in note_ranges:
        rows: list[tuple[float, int, float, int]] = []
        total_note_count = 0
        last_end_tick = 0
        for tick in sorted(events_by_tick[diff]):
            frets = events_by_tick[diff][tick]
            lane_mask = 0
            max_end_tick = tick
            for fret, length_ticks in frets.items():
                lane_mask |= 1 << fret
                max_end_tick = max(max_end_tick, tick + length_ticks)

            time_s = tick_to_seconds(tick)
            sustain_s = 0.0
            if max_end_tick > tick:
                beats = (max_end_tick - tick) / resolution
                if beats >= sustain_min_beats:
                    sustain_s = tick_to_seconds(max_end_tick) - time_s

            rows.append((time_s, lane_mask, sustain_s, 0))
            total_note_count += 1
            last_end_tick = max(last_end_tick, max_end_tick)

        notes = build_notes_array(rows)
        duration_s = tick_to_seconds(last_end_tick) if rows else 0.0
        result[diff] = ParsedDifficulty(
            notes=notes,
            duration_s=duration_s,
            open_note_count=0,
            total_note_count=total_note_count,
            ignored_counts=dict(ignored_counts),
        )
    return result


def parse_midi(
    path: Path, sustain_min_beats: float = 0.5
) -> tuple[dict[str, ParsedDifficulty], float]:
    """Returns (per-difficulty results, chart_offset_s). .mid files have no
    Offset concept, so the second element is always 0.0 (kept for a uniform
    return shape with parse_chart)."""
    midi = mido.MidiFile(path, clip=True)
    resolution = midi.ticks_per_beat

    tempo_events, warn_outside_track0 = _collect_tempo_events(midi)
    if warn_outside_track0:
        print(f"WARNING: set_tempo found outside track 0 in {path}")
    tick_to_seconds = build_tempo_map(tempo_events, resolution)

    guitar_track = _find_guitar_track(midi)
    if guitar_track is None:
        return {}, 0.0

    all_diffs = _parse_guitar_track(
        guitar_track, NOTE_RANGES, tick_to_seconds, resolution, sustain_min_beats
    )
    result = {d: pd for d, pd in all_diffs.items() if pd.total_note_count > 0}
    return result, 0.0
