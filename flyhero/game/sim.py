"""Vectorized rhythm-game simulator + debug video CLI (PLAN Phase 2 task 10).

VecRhythmEnv batches B songs; each step() advances every env by one 60Hz
frame. Internally it runs one rules.RuleEngine per batch element (see
rules.py's docstring) rather than a numpy-vectorized note-matcher: with
<=max_open_notes_capacity concurrently open notes per env this is cheap.
The retina blur/sample IS batched across envs (see _obs()) -- that was the
throughput-critical path for Gate G2's >=5x-real-time requirement (D21,
amended from the original >=10x; measured 6.8x at batch 64 on CPU after
that optimization work, accepted rather than pursuing a torch/GPU rewrite
now -- see bench_env.py and docs/PROGRESS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from flyhero.game.render import render_frame
from flyhero.game.retina import (
    PhotoreceptorMap,
    _nearest_rc,
    bilinear_sample,
    build_photoreceptor_map,
    gaussian_blur,
    gaussian_blur_batch,
)
from flyhero.game.rules import RuleEngine, score_playthrough, scripted_perfect_actions
from flyhero.game.song import Song, load_song_merged
from flyhero.utils.config import load_config


class VecRhythmEnv:
    """actions[B,6]: columns 0-4 are held frets (green..orange), column 5 is
    strum, for the *current* frame. reward/done/info follow gymnasium's
    vectorized-env conventions loosely (this project doesn't wrap gymnasium's
    VectorEnv base class -- PLAN only asks for reset/step with this shape)."""

    def __init__(self, cfg: dict, photo_map: PhotoreceptorMap):
        self.fps = cfg["fps"]
        self.hit_window_s = cfg["hit_window_s"]
        self.lookahead_s = cfg["lookahead_s"]
        self.frame_size = cfg["frame_size"]
        self.max_open = cfg["max_open_notes_capacity"]
        self.reward_cfg = cfg["reward"]
        self.retina_cfg = cfg["retina"]
        self.photo_map = photo_map

        # photo_map.image_pos never changes across an env's lifetime, but
        # _obs() runs every step. Two measured wins: (1) precompute the
        # nearest-pixel indices once instead of every frame; (2) a dense
        # gather-then-multiply-by-mask (excluded photoreceptors point at a
        # dummy pixel and get zeroed by the mask) beats a sparse boolean-
        # indexed assignment by ~35% -- both were real chunks of Gate G2's
        # throughput benchmark, not just constant-factor noise.
        valid = photo_map.included_mask & ~np.isnan(photo_map.image_pos[:, 0])
        n_photo = len(photo_map.photoreceptor_idx)
        self._sample_r = np.zeros(n_photo, dtype=np.int64)
        self._sample_c = np.zeros(n_photo, dtype=np.int64)
        r_valid, c_valid = _nearest_rc(photo_map.image_pos[valid], self.frame_size, self.frame_size)
        self._sample_r[valid] = r_valid
        self._sample_c[valid] = c_valid
        self._sample_mask = valid.astype(np.float32)

    def reset(self, songs: list[Song]) -> dict:
        self.songs = songs
        self.batch = len(songs)
        self.frame_idx = np.zeros(self.batch, dtype=np.int64)
        self.n_frames = np.array(
            [max(int(round(s.duration_s * self.fps)), 1) for s in songs], dtype=np.int64
        )
        self.engines = [
            RuleEngine(s.notes, self.fps, self.hit_window_s, self.max_open) for s in songs
        ]
        self.ema: np.ndarray | None = None
        self.done = np.zeros(self.batch, dtype=bool)
        self.metrics: list[dict | None] = [None] * self.batch
        return self._obs()

    def _obs(self) -> dict:
        frames = np.zeros((self.batch, self.frame_size, self.frame_size), dtype=np.uint8)
        for b in range(self.batch):
            t = self.frame_idx[b] / self.fps
            frames[b] = render_frame(self.songs[b].notes, t, self.lookahead_s, self.frame_size)

        # Batched across envs (not a per-env Python loop), float32 throughout,
        # precomputed sample indices (not photo_map.image_pos re-parsed every
        # frame): this is the path Gate G2's >=10x-real-time throughput check
        # measures.
        blurred = gaussian_blur_batch(
            frames.astype(np.float32) / 255.0, sigma=self.retina_cfg["blur_sigma"]
        )
        intensity = blurred[:, self._sample_r, self._sample_c] * self._sample_mask
        if self.ema is None:
            self.ema = intensity.copy()
        alpha = self.retina_cfg["ema_alpha"]
        current = self.retina_cfg["gain"] * (intensity - self.ema)
        self.ema = alpha * intensity + (1 - alpha) * self.ema
        current[:, ~self.photo_map.included_mask] = 0.0

        return {"frame": frames, "retina": current.astype(np.float32)}

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.batch, dtype=np.float32)
        infos: list[dict] = []
        for b in range(self.batch):
            if self.done[b]:
                infos.append({"events": []})
                continue

            held_mask = 0
            for lane in range(5):
                if actions[b, lane]:
                    held_mask |= 1 << lane
            strum = bool(actions[b, 5])

            prev_sustain = self.engines[b].sustain_frames_hit
            events = self.engines[b].step(int(self.frame_idx[b]), held_mask, strum)
            sustain_delta = self.engines[b].sustain_frames_hit - prev_sustain

            r = sustain_delta * self.reward_cfg["sustain_frame"]
            for e in events:
                r += self.reward_cfg.get(e.kind, 0.0)
            rewards[b] = r
            infos.append({"events": events})

            self.frame_idx[b] += 1
            if self.frame_idx[b] >= self.n_frames[b]:
                self.done[b] = True
                self.metrics[b] = self.engines[b].finalize(int(self.n_frames[b]))

        return self._obs(), rewards, self.done.copy(), infos


def render_debug_video(
    song: Song, cfg: dict, photo_map: PhotoreceptorMap, out_path: Path
) -> dict:
    """3 panels side by side: the note highway, fret targets (bottom half)
    vs the scripted player's actions (top half) with hit/miss/overstrum
    event markers, and the retina input drawn at each photoreceptor's own
    image position -- this panel is also the visual check of retina
    orientation."""
    import imageio.v2 as imageio

    from flyhero.game.labels import compute_frame_labels

    fps = cfg["fps"]
    hit_window_s = cfg["hit_window_s"]
    lookahead_s = cfg["lookahead_s"]
    frame_size = cfg["frame_size"]

    fret_target, _strum = compute_frame_labels(song.notes, song.duration_s, fps, hit_window_s)
    held_seq, strum_seq = scripted_perfect_actions(song.notes, song.duration_s, fps, hit_window_s)
    metrics, events = score_playthrough(
        song.notes, held_seq, strum_seq, fps, hit_window_s, cfg["max_open_notes_capacity"]
    )
    event_by_frame: dict[int, list] = {}
    for e in events:
        event_by_frame.setdefault(e.frame, []).append(e)

    lane_w = frame_size // 5
    event_color = {"hit": 255, "miss": 60, "overstrum": 180}

    rows = (photo_map.image_pos[:, 0] * (frame_size - 1))
    cols = (photo_map.image_pos[:, 1] * (frame_size - 1))
    valid_pos = photo_map.included_mask & ~np.isnan(photo_map.image_pos[:, 0])
    rows_i = np.clip(np.nan_to_num(rows), 0, frame_size - 1).astype(np.int64)
    cols_i = np.clip(np.nan_to_num(cols), 0, frame_size - 1).astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps)
    try:
        n_frames = len(held_seq)
        for f in range(n_frames):
            t = f / fps
            highway = render_frame(song.notes, t, lookahead_s, frame_size)

            panel2 = np.full((frame_size, frame_size), 20, dtype=np.uint8)
            for lane in range(5):
                c0, c1 = lane * lane_w, (lane + 1) * lane_w
                if fret_target[f] & (1 << lane):
                    panel2[frame_size // 2 :, c0:c1] = 120
                if held_seq[f] & (1 << lane):
                    panel2[: frame_size // 2, c0:c1] = 200
            for e in event_by_frame.get(f, []):
                panel2[0:4, :] = event_color[e.kind]

            blurred = gaussian_blur(highway.astype(np.float64) / 255.0, sigma=cfg["retina"]["blur_sigma"])
            intensity = bilinear_sample(blurred, photo_map.image_pos)
            panel3 = np.full((frame_size, frame_size), 10, dtype=np.uint8)
            vals = np.clip(intensity * 255, 0, 255).astype(np.uint8)
            panel3[rows_i[valid_pos], cols_i[valid_pos]] = vals[valid_pos]

            combined = np.concatenate([highway, panel2, panel3], axis=1)
            rgb = np.stack([combined] * 3, axis=-1)
            writer.append_data(rgb)
    finally:
        writer.close()

    return metrics


AUDIO_FILENAMES = ("song.mp3", "song.ogg", "song.opus", "song.wav")


def find_song_audio(folder: Path) -> Path | None:
    for name in AUDIO_FILENAMES:
        p = folder / name
        if p.exists():
            return p
    return None


def mux_audio_into_video(
    video_path: Path, audio_path: Path, audio_start_offset_s: float, out_path: Path
) -> None:
    """Mux `audio_path` into `video_path`'s silent track. Per D22, this only
    ever writes under a gitignored path (e.g. media/) -- it copies audio out
    of the (still read-only) song library, never into it, and never into
    web/.

    Sync convention: chart Offset / song.ini delay both describe how far the
    chart's note timeline is shifted relative to the raw audio (positive =
    notes pushed later relative to the audio); to sync, the audio must start
    playing `-(chart_offset_s + ini_delay_s)` seconds into the video
    timeline -- negative means the audio starts before video t=0, handled by
    trimming that much off the front of the audio file instead of shifting
    video frames (which can't start before 0)."""
    import subprocess

    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_start_offset_s >= 0:
        audio_input_args = ["-itsoffset", f"{audio_start_offset_s:.6f}", "-i", str(audio_path)]
    else:
        audio_input_args = ["-ss", f"{-audio_start_offset_s:.6f}", "-i", str(audio_path)]

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        *audio_input_args,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--video", nargs=2, metavar=("SONG_ID", "DIFFICULTY"))
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--mux-audio-from",
        default=None,
        help="song folder rel_path (relative to song_library) to find audio in and mux into --out",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["processed_dir"])

    if args.video:
        song_id, difficulty = args.video
        graph = np.load(processed_dir / "graph.npz")
        import json

        meta = json.load(open(processed_dir / "graph_meta.json"))
        photo_map = build_photoreceptor_map(graph, meta["type_names"])
        song = load_song_merged(
            processed_dir / "songs" / f"{song_id}_{difficulty}.npz", cfg["chord_merge_min_gap_s"]
        )
        out_path = Path(args.out) if args.out else Path(f"runs/debug_video_{song_id}_{difficulty}.mp4")

        video_path = out_path
        if args.mux_audio_from:
            video_path = out_path.parent / f"_silent_{out_path.name}"
        metrics = render_debug_video(song, cfg, photo_map, video_path)

        if args.mux_audio_from:
            paths_cfg = load_config(args.paths_config)
            song_root = Path(paths_cfg["song_library"]).expanduser() / args.mux_audio_from
            audio_path = find_song_audio(song_root)
            if audio_path is None:
                parser.error(f"no audio file found under {song_root}")
            offset_s = -(song.chart_offset_s + song.ini_delay_s)
            mux_audio_into_video(video_path, audio_path, offset_s, out_path)
            video_path.unlink()
            print(f"Muxed audio from {audio_path} (offset {offset_s:+.3f}s)")

        print(f"Wrote {out_path}")
        print(metrics)
        return 0

    parser.error("nothing to do -- pass --video SONG_ID DIFFICULTY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
