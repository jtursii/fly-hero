"""Phase 2 tests: song.py round-trip, chart_parser.py, midi_parser.py, and a
.chart<->.mid parity check using hand-built fixtures (PLAN Phase 2 task 11,
planned alongside tasks 1-3)."""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pytest

from flyhero.game.chart_parser import parse_chart
from flyhero.game.midi_parser import parse_midi
from flyhero.game.song import Song, load_song, save_song
from flyhero.library.scan import NOTE_RANGES
from flyhero.utils.config import load_config

PATHS_CONFIG = Path("configs/paths.yaml")
REAL_CLIP_MID_REL_PATH = "Guitar Hero - Metallica/Metallica - No Leaf Clover/notes.mid"


def _real_library_root() -> Path | None:
    if not PATHS_CONFIG.exists():
        return None
    cfg = load_config(PATHS_CONFIG)
    root = Path(cfg["song_library"]).expanduser()
    return root if root.is_dir() else None


def test_save_load_song_round_trip(tmp_path):
    notes = np.zeros(2, dtype=[("time_s", "f4"), ("lane_mask", "u1"), ("sustain_s", "f4"), ("flags", "u1")])
    notes[0] = (1.0, 0b00011, 0.2, 1)
    notes[1] = (1.8, 0b00100, 0.0, 0)
    song = Song(
        song_id="abc123",
        title="Test Song",
        artist="Test Artist",
        charter="Test Charter",
        difficulty="Expert",
        notes=notes,
        duration_s=2.0,
        chart_offset_s=0.5,
        ini_delay_s=-0.1,
    )
    path = tmp_path / "abc123_Expert.npz"
    save_song(song, path)
    loaded = load_song(path)

    assert loaded.song_id == "abc123"
    assert loaded.title == "Test Song"
    assert loaded.artist == "Test Artist"
    assert loaded.charter == "Test Charter"
    assert loaded.difficulty == "Expert"
    assert loaded.duration_s == pytest.approx(2.0)
    assert loaded.chart_offset_s == pytest.approx(0.5)
    assert loaded.ini_delay_s == pytest.approx(-0.1)
    np.testing.assert_array_equal(loaded.notes, notes)


CHART_FIXTURE = """[Song]
{
\tResolution = 192
\tOffset = 0
}
[SyncTrack]
{
\t0 = B 120000
\t384 = B 150000
}
[ExpertSingle]
{
\t576 = N 0 96
\t576 = N 1 96
\t768 = N 2 0
}
"""


def _build_equivalent_midi(path: Path) -> None:
    """Same two tempo changes and the same chord + single note as
    CHART_FIXTURE, but at ticks_per_beat=480 vs the chart's Resolution=192
    (a 2.5x scale-up of every tick position), to prove the tick->seconds
    conversion is resolution-independent."""
    conductor = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120.0), time=0),
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(150.0), time=960),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    guitar = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="PART GUITAR", time=0),
            mido.Message("note_on", note=96, velocity=100, time=1440),
            mido.Message("note_on", note=97, velocity=100, time=0),
            mido.Message("note_off", note=96, velocity=0, time=240),
            mido.Message("note_off", note=97, velocity=0, time=0),
            mido.Message("note_on", note=98, velocity=100, time=240),
            mido.Message("note_off", note=98, velocity=0, time=0),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(conductor)
    midi.tracks.append(guitar)
    midi.save(path)


def test_chart_and_midi_parse_to_the_same_song(tmp_path):
    chart_path = tmp_path / "notes.chart"
    chart_path.write_text(CHART_FIXTURE)
    mid_path = tmp_path / "notes.mid"
    _build_equivalent_midi(mid_path)

    chart_diffs, chart_offset_s = parse_chart(chart_path)
    mid_diffs, mid_offset_s = parse_midi(mid_path)

    assert chart_offset_s == pytest.approx(0.0)
    assert mid_offset_s == pytest.approx(0.0)  # .mid has no Offset concept

    assert set(chart_diffs) == {"Expert"}
    assert set(mid_diffs) == {"Expert"}

    chart_notes = chart_diffs["Expert"].notes
    mid_notes = mid_diffs["Expert"].notes

    assert len(chart_notes) == len(mid_notes) == 2

    np.testing.assert_allclose(chart_notes["time_s"], [1.4, 1.8], atol=1e-4)
    np.testing.assert_allclose(mid_notes["time_s"], [1.4, 1.8], atol=1e-4)

    np.testing.assert_array_equal(chart_notes["lane_mask"], [0b011, 0b100])
    np.testing.assert_array_equal(mid_notes["lane_mask"], [0b011, 0b100])

    # sustain of 0.5 beat at 150 BPM = 0.2s; exactly at the default
    # sustain_min_beats=0.5 threshold, so it must NOT be zeroed out.
    np.testing.assert_allclose(chart_notes["sustain_s"], [0.2, 0.0], atol=1e-4)
    np.testing.assert_allclose(mid_notes["sustain_s"], [0.2, 0.0], atol=1e-4)

    np.testing.assert_array_equal(chart_notes["flags"], [0, 0])
    np.testing.assert_array_equal(mid_notes["flags"], [0, 0])


OPEN_NOTE_CHART_FIXTURE = """[Song]
{
\tResolution = 192
}
[SyncTrack]
{
\t0 = B 120000
}
[ExpertSingle]
{
\t192 = N 0 0
\t384 = N 7 0
}
"""


def test_chart_open_note_is_dropped_and_counted(tmp_path):
    path = tmp_path / "notes.chart"
    path.write_text(OPEN_NOTE_CHART_FIXTURE)

    diffs, _ = parse_chart(path)
    pd = diffs["Expert"]

    assert len(pd.notes) == 1  # the open note is not in the array at all
    assert pd.notes["lane_mask"][0] == 0b00001
    assert pd.open_note_count == 1
    assert pd.total_note_count == 1


def test_midi_velocity_zero_note_on_treated_as_note_off(tmp_path):
    """A note_on with velocity=0 is a note-off per the MIDI spec's running-
    status convention; game-ripped .mid files use this instead of an
    explicit note_off. Fret 0 (note 96) held for 240 ticks at the default
    120 BPM / ticks_per_beat=480 -> 0.5 beat -> exactly sustain_min_beats."""
    guitar = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="PART GUITAR", time=0),
            mido.Message("note_on", note=96, velocity=100, time=0),
            mido.Message("note_on", note=96, velocity=0, time=240),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(guitar)
    path = tmp_path / "notes.mid"
    midi.save(path)

    diffs, _ = parse_midi(path)
    pd = diffs["Expert"]

    assert len(pd.notes) == 1
    assert pd.notes["time_s"][0] == pytest.approx(0.0)
    assert pd.notes["lane_mask"][0] == 0b00001
    assert pd.notes["sustain_s"][0] == pytest.approx(0.25, abs=1e-4)  # 0.5 beat @ 120 BPM


def _real_library_or_skip():
    root = _real_library_root()
    if root is None:
        pytest.skip("requires the real song library from configs/paths.yaml")
    p = root / REAL_CLIP_MID_REL_PATH
    if not p.exists():
        pytest.skip(f"fixture song not present in this library: {REAL_CLIP_MID_REL_PATH}")
    return p


def test_parse_real_clip_required_midi_file():
    """Regression test on one of the 123 real .mid files that need
    mido's clip=True to parse at all (found during Phase-0 inventory).
    Confirms parse_midi handles a real multi-track, corrupted-SysEx file:
    all four difficulties present, and every decoded note falls inside its
    difficulty's valid fret range (the forcing/tap/overdrive marker notes
    just outside each range must be excluded, not decoded as frets)."""
    path = _real_library_or_skip()

    diffs, offset_s = parse_midi(path)

    assert offset_s == pytest.approx(0.0)
    assert set(diffs) == {"Easy", "Medium", "Hard", "Expert"}

    for diff, pd in diffs.items():
        assert pd.total_note_count > 0
        assert pd.open_note_count == 0  # .mid opens are never decoded, v1
        # lane_mask must only use bits 0-4 (frets 0-4) -- nothing from the
        # marker notes (forced/tap/overdrive) just above each fret window.
        assert np.all(pd.notes["lane_mask"] > 0)
        assert np.all(pd.notes["lane_mask"] < (1 << 5))

    # ignored_counts is file-scoped (the whole PART GUITAR track is scanned
    # once), so every difficulty carries an identical copy -- check that,
    # then read the counts once. Real file has 2 sysex messages in PART
    # GUITAR and several hundred forcing/overdrive marker notes outside the
    # valid ranges (verified by direct inspection while planning this test).
    all_ignored_counts = [pd.ignored_counts for pd in diffs.values()]
    assert all(c == all_ignored_counts[0] for c in all_ignored_counts)
    assert all_ignored_counts[0]["ignored_notes"] > 0
    assert all_ignored_counts[0]["sysex_ignored"] == 2
