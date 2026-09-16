"""Tick->seconds conversion shared by chart_parser.py and midi_parser.py.
Both formats reduce to the same shape: a list of (tick, bpm) tempo-change
events plus a ticks-per-quarter-note resolution."""

from __future__ import annotations

from bisect import bisect_right
from typing import Callable

TickToSeconds = Callable[[int], float]


def build_tempo_map(tempo_events: list[tuple[int, float]], resolution: int) -> TickToSeconds:
    """Return a tick->seconds function. `tempo_events` need not include tick
    0 or be sorted; a default 120 BPM is assumed before the first event."""
    events: dict[int, float] = {0: 120.0}
    for tick, bpm in tempo_events:
        events[tick] = bpm
    ticks = sorted(events)
    bpms = [events[t] for t in ticks]

    seconds_at = [0.0]
    for i in range(1, len(ticks)):
        delta_ticks = ticks[i] - ticks[i - 1]
        seconds_at.append(
            seconds_at[-1] + delta_ticks * (60.0 / bpms[i - 1]) / resolution
        )

    def tick_to_seconds(t: int) -> float:
        idx = max(bisect_right(ticks, t) - 1, 0)
        delta = t - ticks[idx]
        return seconds_at[idx] + delta * (60.0 / bpms[idx]) / resolution

    return tick_to_seconds
