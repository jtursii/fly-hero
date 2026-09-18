"""The single scoring implementation (invariant 4): used by training,
evaluation, export, Gate G2's scripted-player check, and env.py's per-step
reward. PLAN Phase 2 task 6.

- Hit: a strum frame while the held fret set exactly equals a note's
  lane_mask, within +-hit_window_s. When more than one open, unhit,
  lane-matching note's window contains the strum (common in dense charts --
  measured >=90% of adjacent-note gaps overlap at hit_window_s=0.07 in this
  library's densest song), the strum counts for the *earliest* such note.
  Each note can be hit at most once.
- Miss: resolved lazily, when a note's window closes (time > note.time_s +
  hit_window_s) without it having been hit -- not the instant a strum fails
  to match it, since a later strum within the still-open window could still
  hit it.
- Overstrum: a strum frame with no open, lane-matching, unhit note in its
  window.
- Sustain credit: frames after a hit, while held frets still cover the
  note's lane_mask, up to time_s + sustain_s.

RuleEngine is stateful and single-song, stepped one frame at a time -- this
is what env.py runs per batch element, and what score_playthrough() below
wraps for a whole pre-recorded action sequence (Gate G2's scripted player,
tests, later phases' evaluate.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyhero.game.labels import compute_frame_labels


@dataclass
class NoteEvent:
    frame: int
    note_idx: int  # -1 for an overstrum (no note involved)
    kind: str  # "hit" | "miss" | "overstrum"


class OpenNoteCapacityError(RuntimeError):
    pass


@dataclass
class RuleEngine:
    notes: np.ndarray  # NOTE_DTYPE, sorted by time_s, already load-time-merged
    fps: int
    hit_window_s: float
    max_open: int = 8

    _next_idx: int = field(default=0, init=False)
    _open: list[int] = field(default_factory=list, init=False)
    hit_mask: np.ndarray = field(init=False)
    missed_mask: np.ndarray = field(init=False)
    n_hits: int = field(default=0, init=False)
    n_misses: int = field(default=0, init=False)
    n_overstrums: int = field(default=0, init=False)
    sustain_frames_hit: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n = len(self.notes)
        self.hit_mask = np.zeros(n, dtype=bool)
        self.missed_mask = np.zeros(n, dtype=bool)

    def step(self, frame_idx: int, held_mask: int, strum: bool) -> list[NoteEvent]:
        t = frame_idx / self.fps
        events: list[NoteEvent] = []

        while self._next_idx < len(self.notes) and (
            self.notes["time_s"][self._next_idx] - self.hit_window_s <= t
        ):
            self._open.append(self._next_idx)
            self._next_idx += 1
        if len(self._open) > self.max_open:
            raise OpenNoteCapacityError(
                f"{len(self._open)} simultaneously open notes exceeds max_open={self.max_open} "
                f"at t={t:.4f}s -- raise max_open (see ingest.py's max-simultaneous-open-windows "
                "report, configs/game.yaml's max_open_notes_capacity)."
            )

        if strum:
            candidates = [
                idx
                for idx in self._open
                if not self.hit_mask[idx]
                and int(self.notes["lane_mask"][idx]) == held_mask
                and abs(t - float(self.notes["time_s"][idx])) <= self.hit_window_s
            ]
            if candidates:
                idx = min(candidates, key=lambda i: float(self.notes["time_s"][i]))
                self.hit_mask[idx] = True
                self.n_hits += 1
                events.append(NoteEvent(frame_idx, idx, "hit"))
            else:
                self.n_overstrums += 1
                events.append(NoteEvent(frame_idx, -1, "overstrum"))

        for idx in self._open:
            if self.hit_mask[idx]:
                note_end = float(self.notes["time_s"][idx]) + float(self.notes["sustain_s"][idx])
                lane = int(self.notes["lane_mask"][idx])
                if (
                    float(self.notes["time_s"][idx]) <= t <= note_end
                    and (held_mask & lane) == lane
                ):
                    self.sustain_frames_hit += 1

        still_open = []
        for idx in self._open:
            window_end = float(self.notes["time_s"][idx]) + self.hit_window_s
            note_end = float(self.notes["time_s"][idx]) + float(self.notes["sustain_s"][idx])
            resolved_end = max(window_end, note_end) if self.hit_mask[idx] else window_end
            if t > resolved_end:
                if not self.hit_mask[idx] and not self.missed_mask[idx]:
                    self.missed_mask[idx] = True
                    self.n_misses += 1
                    events.append(NoteEvent(frame_idx, idx, "miss"))
                continue
            still_open.append(idx)
        self._open = still_open

        return events

    def finalize(self, total_frames: int) -> dict:
        """Call after the last step (or after fast-forwarding time past the
        song's end) to resolve any still-open notes as misses and compute
        aggregate metrics."""
        # flush remaining open notes as misses if their window has closed by
        # the end of the recording.
        end_t = total_frames / self.fps
        for idx in list(self._open):
            if not self.hit_mask[idx] and not self.missed_mask[idx]:
                self.missed_mask[idx] = True
                self.n_misses += 1
        self._open = []

        n_notes = len(self.notes)
        hit_rate = self.n_hits / n_notes if n_notes > 0 else 0.0
        minutes = (total_frames / self.fps) / 60.0
        overstrums_per_min = self.n_overstrums / minutes if minutes > 0 else 0.0
        total_sustain_frames = int(np.round(self.notes["sustain_s"] * self.fps).sum())
        sustain_frac = (
            self.sustain_frames_hit / total_sustain_frames if total_sustain_frames > 0 else 1.0
        )

        return {
            "hit_rate": hit_rate,
            "overstrums_per_min": overstrums_per_min,
            "sustain_frac": sustain_frac,
            "n_notes": n_notes,
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "n_overstrums": self.n_overstrums,
        }


def score_playthrough(
    notes: np.ndarray,
    held_mask_seq: np.ndarray,
    strum_seq: np.ndarray,
    fps: int,
    hit_window_s: float,
    max_open: int = 8,
) -> tuple[dict, list[NoteEvent]]:
    """Score a full, pre-recorded playthrough (held_mask_seq[F] int, per-frame
    lane bitmask; strum_seq[F] bool) against `notes`. Used by Gate G2's
    scripted-player check and tests; env.py calls RuleEngine.step() directly
    per batch element instead, but it's the same class either way."""
    engine = RuleEngine(notes=notes, fps=fps, hit_window_s=hit_window_s, max_open=max_open)
    all_events: list[NoteEvent] = []
    for f in range(len(held_mask_seq)):
        all_events.extend(engine.step(f, int(held_mask_seq[f]), bool(strum_seq[f])))
    metrics = engine.finalize(len(held_mask_seq))
    return metrics, all_events


def max_simultaneous_open_windows(notes: np.ndarray, hit_window_s: float) -> int:
    """Sweep-line peak concurrency of [time_s - hit_window_s, time_s +
    hit_window_s] intervals. Used by ingest.py to report the real library-
    wide worst case, which sets RuleEngine's/env.py's max_open capacity
    rather than a guessed constant."""
    n = len(notes)
    if n == 0:
        return 0
    starts = notes["time_s"].astype(np.float64) - hit_window_s
    ends = notes["time_s"].astype(np.float64) + hit_window_s
    events = [(s, 1) for s in starts] + [(e, -1) for e in ends]
    # Opens before closes at an exact tie -> a conservative (not
    # underestimated) peak.
    events.sort(key=lambda x: (x[0], -x[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def scripted_perfect_actions(
    notes: np.ndarray, duration_s: float, fps: int, hit_window_s: float, fret_onset: str = "segment",
    fret_onset_k: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A scripted "perfect player" that plays exactly the labels.py targets:
    held frets = fret_target, strum = strum label. This is deliberate, not a
    separate hand-rolled hold-window -- an independent per-note hold window
    (e.g. "hold from strum_frame-1 through sustain end") would OR together
    when two notes' windows overlap in a dense trill, producing a held mask
    that matches *neither* note's lane_mask exactly and turning would-be hits
    into overstrums. Reusing labels.py's segment-based targets is what makes
    them correct for a perfect player in the first place (see labels.py's
    docstring on the max(sustain_s, hit_window_s) hold-duration floor), and
    it's the direct check that the label design is actually playable."""
    fret_target, strum = compute_frame_labels(notes, duration_s, fps, hit_window_s, fret_onset, fret_onset_k)
    return fret_target.astype(np.int64), strum.astype(bool)
