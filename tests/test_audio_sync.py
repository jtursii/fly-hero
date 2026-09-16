"""Regression test for D29: scripts/render_gameplay_video.py's audio-mux
offset must be computed from the excerpt's own start song-time (the video's
real frame-0 instant), not `excerpt_start_s - burn_in` -- burn-in frames are
simulated for warm-up but never written into the rendered video, so using
the burn-in-inclusive start desynced audio by exactly `burn_in_s` (found on
"Paramore - Misery Business", which sounded off by roughly 1-2 beats even
though its chart Offset and song.ini delay are both 0 -- the bug was in how
the excerpt window was combined with those values, not the values
themselves). "Metallica - One" (Offset=delay=0) could never have caught a
sign/order bug in combining Offset+delay+excerpt-start, since every term is
zero there.

Exercises the real pipeline end to end: a synthetic click track (one click
per synthetic chart note) is muxed via flyhero.game.sim's actual
compute_audio_start_offset_s + mux_audio_into_video (real ffmpeg call), then
the output audio is decoded back to PCM samples and each click's detected
time is checked against where its note should land in the rendered video's
own timeline, for a song with nonzero chart Offset AND nonzero song.ini
delay simultaneously (so a sign error in combining either term, or in
combining them with the excerpt start, would fail this test).
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from flyhero.game.sim import compute_audio_start_offset_s, generate_click_track_wav, mux_audio_into_video


def _make_silent_video(path: Path, duration_s: float, fps: int = 10, size: int = 16) -> None:
    writer = imageio.get_writer(str(path), fps=fps)
    try:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(int(round(duration_s * fps))):
            writer.append_data(frame)
    finally:
        writer.close()


def _extract_click_onsets_s(video_path: Path, sample_rate: int = 44100, threshold: float = 0.15) -> list[float]:
    """Decodes video_path's audio track to mono PCM and returns the start
    time of each contiguous above-threshold run (a click's attack, not its
    decay tail)."""
    wav_path = video_path.with_suffix(".extracted.wav")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate), str(wav_path)],
        check=True, capture_output=True, text=True,
    )
    with wave.open(str(wav_path), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    above = np.abs(samples) > threshold
    rising = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1
    if above.size and above[0]:
        rising = np.concatenate([[0], rising])
    times = [float(i) / sample_rate for i in rising]

    # The click waveform is an oscillating (not monotonic) burst, so it
    # crosses the threshold many times per click -- merge onsets within
    # 20ms of the previous one into a single event (the click's first
    # crossing, i.e. its attack).
    merged: list[float] = []
    for t in times:
        if not merged or t - merged[-1] > 0.02:
            merged.append(t)
    return merged


def test_click_track_lands_on_note_times_with_nonzero_offset_and_delay(tmp_path: Path):
    chart_offset_s = 0.7
    ini_delay_s = 0.3
    video_start_song_time_s = 5.0  # e.g. an excerpt whose own first frame is song time 5.0s
    note_times_s = np.array([5.5, 6.5, 7.5, 8.5])  # song-timeline (chart) note times

    click_path = tmp_path / "clicks.wav"
    generate_click_track_wav(note_times_s, chart_offset_s, ini_delay_s, click_path)

    video_path = tmp_path / "silent.mp4"
    _make_silent_video(video_path, duration_s=6.0)

    offset_s = compute_audio_start_offset_s(chart_offset_s, ini_delay_s, video_start_song_time_s)
    out_path = tmp_path / "out.mp4"
    mux_audio_into_video(video_path, [click_path], offset_s, out_path)

    detected = sorted(_extract_click_onsets_s(out_path))
    expected = sorted(t - video_start_song_time_s for t in note_times_s)

    assert len(detected) == len(expected), f"detected {detected}, expected {expected}"
    for d, e in zip(detected, expected):
        assert abs(d - e) < 0.08, f"click at {d:.3f}s, expected {e:.3f}s (detected={detected}, expected={expected})"


def test_burn_in_inclusive_start_would_have_desynced_by_burn_in_s(tmp_path: Path):
    """Guards the specific regression: passing excerpt_start - burn_in (the
    old, buggy quantity) instead of excerpt_start into
    compute_audio_start_offset_s shifts every click by exactly burn_in_s --
    confirms the fix is the actual excerpt-start-only quantity, not an
    accidental cancellation."""
    chart_offset_s, ini_delay_s = 0.0, 0.0
    excerpt_start_s = 6.0
    burn_in_s = 1.0
    note_times_s = np.array([6.5])

    correct_offset = compute_audio_start_offset_s(chart_offset_s, ini_delay_s, excerpt_start_s)
    buggy_offset = compute_audio_start_offset_s(chart_offset_s, ini_delay_s, excerpt_start_s - burn_in_s)
    assert buggy_offset - correct_offset == burn_in_s
