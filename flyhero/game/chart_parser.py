"""Parser for Clone Hero/Moonscraper .chart files (PLAN Phase 2 task 2).

A .chart file is a set of `[SectionName] { ... }` blocks. `[Song]` and
`[SyncTrack]` are `key = value` / `tick = EVENT ...` respectively; each of
`[EasySingle]`/`[MediumSingle]`/`[HardSingle]`/`[ExpertSingle]` holds `N`
(note) and `S` (star power) events keyed by tick.

v1 rule simplifications (PLAN Phase 2 task 5): HOPO (`N 5`, forced) and tap
(`N 6`) notes are stored as flags but strummed like any other note (rules.py,
not here, decides how they're scored). Open notes (`N 7`) are dropped
entirely -- verified against real charts that an open-note tick never also
carries a 0-4 fret event, so "drop the tick" and "drop the open flag" are the
same thing. Star power (`S`) events are ignored and counted, symmetric with
midi_parser.py's handling of .mid star power.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from flyhero.game.song import FLAG_FORCED, FLAG_TAP, ParsedDifficulty, build_notes_array
from flyhero.game.tempo import TickToSeconds, build_tempo_map
from flyhero.utils.text import read_text_lenient

DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")

FRET_FORCED = 5
FRET_TAP = 6
FRET_OPEN = 7

_SECTION_RE = re.compile(r"\[(\w+)\]\s*\{(.*?)\}", re.DOTALL)
_LINE_RE = re.compile(r"^\s*(\S+)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _parse_sections(text: str) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    for m in _SECTION_RE.finditer(text):
        name = m.group(1)
        lines = [(lm.group(1), lm.group(2)) for lm in _LINE_RE.finditer(m.group(2))]
        sections[name] = lines
    return sections


def _parse_single_section(
    lines: list[tuple[str, str]],
    tick_to_seconds: TickToSeconds,
    resolution: int,
    sustain_min_beats: float,
) -> ParsedDifficulty:
    events_by_tick: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for tick_str, rest in lines:
        parts = rest.split()
        kind = parts[0]
        if kind == "N":
            events_by_tick[int(tick_str)].append(("N", int(parts[1]), int(parts[2])))
        elif kind == "S":
            events_by_tick[int(tick_str)].append(("S", int(parts[1]), int(parts[2])))
        # other event kinds (e.g. "E" text events) are irrelevant, ignored.

    ignored_counts = {"star_power_events": 0}
    rows: list[tuple[float, int, float, int]] = []
    open_note_count = 0
    total_note_count = 0
    last_end_tick = 0

    for tick in sorted(events_by_tick):
        items = events_by_tick[tick]
        ignored_counts["star_power_events"] += sum(1 for it in items if it[0] == "S")

        frets = {fret: length for kind, fret, length in items if kind == "N"}
        fret_bits_present = [f for f in range(5) if f in frets]

        if not fret_bits_present:
            if FRET_OPEN in frets:
                open_note_count += 1
            continue  # pure open note, or a stray forced/tap marker: dropped

        lane_mask = 0
        max_end_tick = tick
        for f in fret_bits_present:
            lane_mask |= 1 << f
            end_tick = tick + frets[f]
            max_end_tick = max(max_end_tick, end_tick)

        flags = 0
        if FRET_FORCED in frets:
            flags |= FLAG_FORCED
        if FRET_TAP in frets:
            flags |= FLAG_TAP

        time_s = tick_to_seconds(tick)
        sustain_s = 0.0
        if max_end_tick > tick:
            beats = (max_end_tick - tick) / resolution
            if beats >= sustain_min_beats:
                sustain_s = tick_to_seconds(max_end_tick) - time_s

        rows.append((time_s, lane_mask, sustain_s, flags))
        total_note_count += 1
        last_end_tick = max(last_end_tick, max_end_tick)

    notes = build_notes_array(rows)
    duration_s = tick_to_seconds(last_end_tick) if rows else 0.0
    return ParsedDifficulty(
        notes=notes,
        duration_s=duration_s,
        open_note_count=open_note_count,
        total_note_count=total_note_count,
        ignored_counts=ignored_counts,
    )


def parse_chart(
    path: Path, sustain_min_beats: float = 0.5
) -> tuple[dict[str, ParsedDifficulty], float]:
    """Returns (per-difficulty results, chart Offset in seconds). Offset is
    recorded on the resulting Song by ingest.py but not used for training."""
    text = read_text_lenient(path)
    sections = _parse_sections(text)

    song_meta = dict(sections.get("Song", []))
    resolution = int(song_meta.get("Resolution", "192"))
    offset_s = float(song_meta.get("Offset", "0"))

    tempo_events = []
    for tick_str, rest in sections.get("SyncTrack", []):
        parts = rest.split()
        if parts[0] == "B":
            tempo_events.append((int(tick_str), int(parts[1]) / 1000.0))

    tick_to_seconds = build_tempo_map(tempo_events, resolution)

    result: dict[str, ParsedDifficulty] = {}
    for diff in DIFFICULTIES:
        section_name = f"{diff}Single"
        if section_name in sections:
            result[diff] = _parse_single_section(
                sections[section_name], tick_to_seconds, resolution, sustain_min_beats
            )

    return result, offset_s
