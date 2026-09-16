"""fly.sh video: renders a gameplay video of a checkpoint's current skill --
inference (no_grad) through the exact same retina -> brain/GRU -> readout ->
decoder -> rules.py path bc.py's eval uses (invariants 1 and 4), so the
rendered clip's hit_rate is guaranteed to equal eval's number for that same
checkpoint/song/excerpt (train.bc.score_eval_entry is the single shared
implementation both call -- see its docstring).

Panels (2x2 grid): highway with the model's own frets/strums + hit/miss/
overstrum markers; fret targets vs. the model's actions (legend drawn in
that panel's quadrant); retina input at each photoreceptor's image position
(--retina-panel current|raw: signed adaptation-filtered current, a diverging
colormap -- white=positive, blue=negative, black=zero -- or the raw sampled
intensity before that filter, same colormap since it's always >=0); a
DN-activity (or GRU hidden-state) raster, z-scored per neuron over the whole
clip. A text overlay bar shows run/step/difficulty/song/live hits/accuracy/
overstrums.

Song/difficulty selection: default is the first of the fixed eval songs
(train.common.build_val_set) at the checkpoint's current curriculum
difficulty -- the same 60s excerpt eval scores. An explicit song name is
fuzzy-matched (flyhero/game/song_search.py) against val+test (never train
unless --allow-train), listing matches if ambiguous.

Audio: reuses flyhero/game/sim.py's stem-selection/muxing and
compute_audio_start_offset_s (D22/D29's Offset+delay sync convention),
passing the excerpt's own start time as the song-time instant that the
rendered video's frame 0 corresponds to (burn-in frames are simulated for
warm-up but never written into the video -- see main() below).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from flyhero.game.retina import PhotoreceptorMap, build_photoreceptor_map
from flyhero.game.sim import (
    VecRhythmEnv,
    compute_audio_start_offset_s,
    generate_click_track_wav,
    mux_audio_into_video,
    select_audio_stems,
)
from flyhero.game.song import load_song_merged
from flyhero.game.song_search import search_songs
from flyhero.utils.config import load_config
from train.bc import build_policy, score_eval_entry
from train.common import ValSongEntry, build_val_set, find_note_dense_window

FRAME_SIZE_DISPLAY_SCALE = 6  # each frame_size x frame_size panel -> a 2x2 grid, upscaled for visibility
OVERLAY_BAR_H = 28


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir, e.g. runs/20260916_123616_bc_full_real")
    p.add_argument("song", nargs="?", default=None)
    p.add_argument("difficulty", nargs="?", default=None)
    p.add_argument("--ckpt", default=None, help="override checkpoint path (default: <run>/checkpoint_latest.pt)")
    p.add_argument("--allow-train", action="store_true")
    p.add_argument(
        "--full-song", action="store_true",
        help="render/score the entire song instead of the fixed 60s note-dense excerpt (slow: no cap on frame count)",
    )
    p.add_argument(
        "--audio", choices=["full", "guitar", "clicks"], default="full",
        help="full: every stem, mixed equally. guitar: guitar stem only. "
        "clicks: full mix + a synthesized click overlaid at each chart note's time (sync debugging).",
    )
    p.add_argument(
        "--retina-panel", choices=["current", "raw"], default="current",
        help="current: signed adaptation-filtered current fed to the brain (default). "
        "raw: sampled intensity before the EMA adaptation filter.",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default=None)
    return p.parse_args()


def resolve_checkpoint(run_dir: Path, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)
    latest = run_dir / "checkpoint_latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"no checkpoint_latest.pt in {run_dir} and --ckpt not given")
    return (run_dir / latest.readlink()).resolve() if latest.is_symlink() else latest


def resolve_song_entry(
    processed_dir: Path, cfg_bc: dict, cfg_game: dict, difficulty: str, song_query: str | None, allow_train: bool,
    full_song: bool = False,
) -> ValSongEntry:
    excerpt_s = cfg_bc["eval"]["excerpt_s"]
    if song_query is None:
        val_set = build_val_set(
            processed_dir, [difficulty], cfg_bc["eval"]["n_val_songs"], excerpt_s, cfg_game["chord_merge_min_gap_s"],
        )
        entries = val_set[difficulty]
        if not entries:
            raise RuntimeError(f"no eval songs available at difficulty={difficulty}")
        return entries[0]

    matches = search_songs(processed_dir, song_query, difficulty, allow_train)
    if not matches:
        raise SystemExit(
            f"no {'val/test' if not allow_train else 'val/test/train'} song matching {song_query!r} "
            f"at difficulty={difficulty}"
        )
    if len(matches) > 1:
        lines = "\n".join(f"  - {m.display_name} ({m.split}, id={m.song_id})" for m in matches)
        raise SystemExit(f"ambiguous song query {song_query!r} -- matches:\n{lines}")
    m = matches[0]
    song = load_song_merged(
        processed_dir / "songs" / f"{m.song_id}_{difficulty}.npz", cfg_game["chord_merge_min_gap_s"],
    )
    if full_song:
        return ValSongEntry(song=song, excerpt_start_s=0.0, excerpt_s=song.duration_s)
    if song.duration_s < excerpt_s:
        raise SystemExit(f"{m.display_name} is shorter ({song.duration_s:.1f}s) than the eval excerpt ({excerpt_s}s)")
    excerpt_start = find_note_dense_window(song.notes["time_s"], song.duration_s, excerpt_s)
    return ValSongEntry(song=song, excerpt_start_s=excerpt_start, excerpt_s=excerpt_s)


def capture_visualization_frames(
    policy, env: VecRhythmEnv, photo_map: PhotoreceptorMap, clip, burn_in_frames: int, excerpt_frames: int,
    frame_size: int, device, dtype, is_connectome: bool,
) -> dict:
    """A second, independent pass over the same clip (fresh policy state --
    doesn't touch or depend on score_eval_entry's own run) purely to collect
    per-frame visuals: the rendered highway frame, retina input, and DN/
    hidden-state activity. Deterministic + no_grad, so it reproduces the
    same decode as score_eval_entry; used only for rendering, not scoring."""
    batch = 1
    env.reset([clip])
    policy.reset_stream(batch)
    state = policy.init_state(batch, device, dtype)

    highway_frames, retina_frames, retina_raw_frames, activity_frames = [], [], [], []
    total_frames = burn_in_frames + excerpt_frames
    with torch.no_grad():
        for f in range(total_frames):
            obs, _, _, _ = env.step(np.zeros((1, 6), dtype=bool))
            frame_uint8 = obs["frame"][0]
            frame_t = torch.from_numpy(obs["frame"]).to(device)
            if is_connectome:
                i_photo = policy.retina.step(frame_t).to(dtype=policy.brain.dtype)
                state, dn_rate_mean = policy.brain.frame_step(state, i_photo, n_substeps=policy.substeps_per_frame)
                if f >= burn_in_frames:
                    retina_frames.append(i_photo[0].cpu().numpy())
                    retina_raw_frames.append(policy.retina.last_intensity[0].cpu().numpy())
                    r_dn = torch.clamp(torch.relu(state[:, policy.brain.dn_idx]), 0.0, 1.0)
                    activity_frames.append(r_dn[0].cpu().numpy())
            else:
                state, _ = policy.step_frame(state, frame_t)
                if f >= burn_in_frames:
                    activity_frames.append(state[0].cpu().numpy())
            if f >= burn_in_frames:
                highway_frames.append(frame_uint8)

    return dict(
        highway_frames=np.stack(highway_frames),
        retina_frames=np.stack(retina_frames) if retina_frames else None,
        retina_raw_frames=np.stack(retina_raw_frames) if retina_raw_frames else None,
        activity_frames=np.stack(activity_frames),
    )


RETINA_RESTING_GRAY = 76  # ~30% of 255; resting-state brightness for covered photoreceptor pixels


def _diverging_colormap(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """values: [...] signed floats -> RGB uint8 [..., 3]. Positive -> white
    (scaled by magnitude), negative -> blue (scaled by magnitude), 0 ->
    resting gray. `valid` marks pixels covered by a photoreceptor's
    footprint; pixels outside it stay black regardless of `values`. Scaled
    by the array's own max absolute value. Used both for the signed retina
    current and (where it's always >=0, so only the positive branch is
    exercised, ramping resting gray -> white) the raw sampled intensity."""
    scale = np.abs(values).max()
    scale = scale if scale > 1e-9 else 1.0
    norm = np.clip(values / scale, -1.0, 1.0)
    t_pos = np.clip(norm, 0.0, 1.0)
    t_neg = np.clip(-norm, 0.0, 1.0)
    gray_channel = np.clip(RETINA_RESTING_GRAY + t_pos * (255 - RETINA_RESTING_GRAY) - t_neg * RETINA_RESTING_GRAY, 0, 255)
    blue_channel = np.clip(RETINA_RESTING_GRAY + (t_pos + t_neg) * (255 - RETINA_RESTING_GRAY), 0, 255)
    rgb = np.zeros(values.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.where(valid, gray_channel, 0)
    rgb[..., 1] = np.where(valid, gray_channel, 0)
    rgb[..., 2] = np.where(valid, blue_channel, 0)
    return rgb


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _retina_panel(retina_signal: np.ndarray, photo_map: PhotoreceptorMap, frame_size: int) -> np.ndarray:
    """retina_signal: [n_photo] signed current or (always >=0) raw sampled
    intensity -- caller (main()) picks which via --retina-panel. Returns an
    RGB [frame_size, frame_size, 3] panel: each photoreceptor's footprint is
    drawn at resting gray (visible even with zero signal, so both eye
    silhouettes always show), with signal placed at its image position on
    top via the diverging colormap; pixels with no photoreceptor coverage
    stay black."""
    rows_i = np.clip(np.nan_to_num(photo_map.image_pos[:, 0] * (frame_size - 1)), 0, frame_size - 1).astype(np.int64)
    cols_i = np.clip(np.nan_to_num(photo_map.image_pos[:, 1] * (frame_size - 1)), 0, frame_size - 1).astype(np.int64)
    valid = photo_map.included_mask & ~np.isnan(photo_map.image_pos[:, 0])
    grid = np.zeros((frame_size, frame_size), dtype=np.float64)
    covered = np.zeros((frame_size, frame_size), dtype=bool)
    grid[rows_i[valid], cols_i[valid]] = retina_signal[valid]
    covered[rows_i[valid], cols_i[valid]] = True
    return _diverging_colormap(grid, covered)


def zscore_activity_per_neuron(activity_frames: np.ndarray) -> np.ndarray:
    """activity_frames: [T, n_units] -> per-neuron z-score over the whole
    clip (mean/std computed over axis 0), so each unit's trace is judged
    against its own baseline rather than the population's raw magnitude."""
    mean = activity_frames.mean(axis=0, keepdims=True)
    std = activity_frames.std(axis=0, keepdims=True)
    std = np.where(std > 1e-9, std, 1.0)
    return (activity_frames - mean) / std


def _activity_raster_panel(
    activity_z: np.ndarray, frame: int, frame_size: int, window: int = 128, clip_std: float = 3.0,
) -> np.ndarray:
    """activity_z: [T, n_units], already z-scored per neuron over the whole
    clip (zscore_activity_per_neuron). Shows a scrolling raster (units on
    the vertical axis, recent time on the horizontal axis) of the last
    `window` frames up to `frame`, resized to frame_size x frame_size.
    Magnitude clipped at `clip_std` standard deviations (a fixed scale
    across the whole clip, not renormalized per window) and rendered as
    grayscale-in-RGB for consistency with the other (now RGB) panels."""
    lo = max(0, frame - window + 1)
    hist = activity_z[lo : frame + 1]  # [<=window, n_units]
    if hist.shape[0] < window:
        pad = np.zeros((window - hist.shape[0], hist.shape[1]), dtype=hist.dtype)
        hist = np.concatenate([pad, hist], axis=0)
    mag = np.clip(np.abs(hist), 0, clip_std)
    img = np.clip((mag / clip_std) * 255, 0, 255).astype(np.uint8).T  # [n_units, window]
    gray = np.asarray(Image.fromarray(img).resize((frame_size, frame_size), Image.NEAREST))
    return _gray_to_rgb(gray)


def render_video_frames(
    highway_model: np.ndarray, held_mask: np.ndarray, strum: np.ndarray, events, fret_target: np.ndarray,
    retina_frames: np.ndarray | None, activity_z: np.ndarray, photo_map: PhotoreceptorMap, frame_size: int,
    fps: int, overlay_text_lines: list[str],
) -> list[np.ndarray]:
    n_frames = highway_model.shape[0]
    event_color = {"hit": 255, "miss": 60, "overstrum": 180}
    event_by_frame: dict[int, list] = {}
    for e in events:
        event_by_frame.setdefault(e.frame, []).append(e)

    lane_w = frame_size // 5
    n_hits_so_far = n_misses_so_far = n_overstrums_so_far = 0
    out_frames = []
    for f in range(n_frames):
        highway = _gray_to_rgb(highway_model[f])

        panel2 = np.full((frame_size, frame_size), 20, dtype=np.uint8)
        for lane in range(5):
            c0, c1 = lane * lane_w, (lane + 1) * lane_w
            if int(fret_target[f]) & (1 << lane):
                panel2[frame_size // 2 :, c0:c1] = 120
            if int(held_mask[f]) & (1 << lane):
                panel2[: frame_size // 2, c0:c1] = 200
        for e in event_by_frame.get(f, []):
            panel2[0:4, :] = event_color[e.kind]
            if e.kind == "hit":
                n_hits_so_far += 1
            elif e.kind == "miss":
                n_misses_so_far += 1
            elif e.kind == "overstrum":
                n_overstrums_so_far += 1
        panel2 = _gray_to_rgb(panel2)

        panel3 = (
            _retina_panel(retina_frames[f], photo_map, frame_size)
            if retina_frames is not None
            else _gray_to_rgb(np.full((frame_size, frame_size), 10, dtype=np.uint8))
        )
        panel4 = _activity_raster_panel(activity_z, f, frame_size)

        top = np.concatenate([highway, panel2], axis=1)
        bottom = np.concatenate([panel3, panel4], axis=1)
        grid = np.concatenate([top, bottom], axis=0)  # [2*frame_size, 2*frame_size, 3]

        big = Image.fromarray(grid, mode="RGB").resize(
            (grid.shape[1] * FRAME_SIZE_DISPLAY_SCALE, grid.shape[0] * FRAME_SIZE_DISPLAY_SCALE), Image.NEAREST
        )
        canvas = Image.new("RGB", (big.width, big.height + OVERLAY_BAR_H), (0, 0, 0))
        canvas.paste(big, (0, OVERLAY_BAR_H))
        draw = ImageDraw.Draw(canvas)

        resolved = n_hits_so_far + n_misses_so_far
        acc = (n_hits_so_far / resolved) if resolved > 0 else 0.0
        text = (
            " | ".join(overlay_text_lines)
            + f" | hits: {n_hits_so_far} | acc: {acc * 100:.1f}% | overstrums: {n_overstrums_so_far}"
        )
        draw.text((4, 6), text, fill=(255, 255, 255))

        # Legend for the targets-vs-actions panel (top-right quadrant of the
        # 2x2 grid): panel2 draws target in the bottom half, the model's own
        # held frets in the top half (overwritten by a 4px event-color bar
        # at the very top on hit/miss/overstrum frames).
        legend_x = big.width // 2 + 4
        legend_y = OVERLAY_BAR_H + 4
        draw.text((legend_x, legend_y), "top: actual  bottom: target", fill=(255, 255, 255))
        draw.text((legend_x, legend_y + 10), "bar: hit=white miss=dark overstrum=light", fill=(255, 255, 255))

        out_frames.append(np.asarray(canvas))
    return out_frames


def main() -> None:
    args = parse_args()
    t_start = time.time()
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"no such run dir: {run_dir}")

    model_kind = (run_dir / "model_kind.txt").read_text().strip()
    cfg_bc = load_config(run_dir / "bc.yaml")
    cfg_game = load_config(run_dir / "game.yaml")
    cfg_brain = load_config(run_dir / "brain.yaml") if model_kind == "connectome" else {}

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = torch.float32

    ckpt_path = resolve_checkpoint(run_dir, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    difficulties = cfg_bc["curriculum"]["difficulties"]
    difficulty_idx = ckpt["difficulty_idx"]
    checkpoint_difficulty = difficulties[difficulty_idx] if difficulty_idx < len(difficulties) else difficulties[-1]
    difficulty = args.difficulty or checkpoint_difficulty

    processed_dir = Path(cfg_bc["processed_dir"])
    graph_np = dict(np.load(processed_dir / "graph.npz"))
    graph_meta_path = processed_dir / "graph_meta.json"
    graph_meta = json.loads(graph_meta_path.read_text())
    photo_map = build_photoreceptor_map(graph_np, graph_meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)

    policy = build_policy(model_kind, cfg_bc, cfg_game, cfg_brain, photo_map, device, dtype)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    entry = resolve_song_entry(processed_dir, cfg_bc, cfg_game, difficulty, args.song, args.allow_train, args.full_song)
    if args.full_song and args.song is None:
        entry = ValSongEntry(song=entry.song, excerpt_start_s=0.0, excerpt_s=entry.song.duration_s)
    print(f"song: {entry.song.artist} - {entry.song.title} (difficulty={difficulty}, excerpt starts at {entry.excerpt_start_s:.1f}s)")

    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    burn_in_s = cfg_bc["burn_in_s"]
    result = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)
    print(f"hit_rate: {result['metrics']['hit_rate']:.4f}  overstrums/min: {result['metrics']['overstrums_per_min']:.1f}")

    vis = capture_visualization_frames(
        policy, env, photo_map, result["clip"], result["burn_in_frames"], result["excerpt_frames"],
        cfg_game["frame_size"], device, dtype, is_connectome=(model_kind == "connectome"),
    )

    from flyhero.game.labels import compute_frame_labels

    fret_target, _ = compute_frame_labels(result["excerpt_notes"], entry.excerpt_s, fps, hit_window_s)

    retina_display = vis["retina_raw_frames"] if args.retina_panel == "raw" else vis["retina_frames"]
    activity_z = zscore_activity_per_neuron(vis["activity_frames"])

    run_name = run_dir.name
    step = ckpt["step"]
    song_slug = "".join(c if c.isalnum() else "_" for c in entry.song.title)[:40]
    frames = render_video_frames(
        vis["highway_frames"], result["held_mask"], result["strum"], result["events"], fret_target,
        retina_display, activity_z, photo_map, cfg_game["frame_size"], fps,
        overlay_text_lines=[run_name, f"step {step}", difficulty, f"{entry.song.artist} - {entry.song.title}"],
    )

    out_path = Path(args.out) if args.out else Path(f"media/videos/{run_name}_step{step}_{song_slug}_{difficulty}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    silent_path = out_path.parent / f"_silent_{out_path.name}"
    writer = imageio.get_writer(str(silent_path), fps=fps)
    try:
        for fr in frames:
            writer.append_data(fr)
    finally:
        writer.close()

    paths_cfg = load_config("configs/paths.yaml")
    library_root = Path(paths_cfg["song_library"]).expanduser()
    song_id_cache = processed_dir / "song_id_index.json"

    # rel_path lookup: ingest.py doesn't persist rel_path in the per-song
    # npz (only song_id, a one-way hash of it -- D17), so finding this
    # song's audio files means reversing that hash via a cached library
    # index (flyhero.library.scan.build_song_id_index), keyed by the exact
    # song_id (robust to duplicate/near-duplicate titles, unlike matching
    # on artist/title text).
    from flyhero.library.scan import find_song_folder_by_id

    song_root = find_song_folder_by_id(library_root, song_id_cache, entry.song.song_id)
    if song_root is not None:
        stem_mix = "full" if args.audio == "clicks" else args.audio
        audio_paths = select_audio_stems(song_root, stem_mix)
        click_path = None
        if args.audio == "clicks":
            click_path = out_path.parent / f"_clicks_{out_path.stem}.wav"
            generate_click_track_wav(
                entry.song.notes["time_s"], entry.song.chart_offset_s, entry.song.ini_delay_s, click_path,
            )
            audio_paths = audio_paths + [click_path]
        if audio_paths:
            # video frame 0 == song time entry.excerpt_start_s -- NOT
            # excerpt_start_s - burn_in, since burn-in frames are simulated
            # for warm-up but never written into the video (see
            # capture_visualization_frames: highway_frames only appends for
            # f >= burn_in_frames). Using the burn-in-inclusive start here
            # previously desynced audio by exactly burn_in_s (D29).
            offset_s = compute_audio_start_offset_s(
                entry.song.chart_offset_s, entry.song.ini_delay_s, entry.excerpt_start_s,
            )
            mux_audio_into_video(silent_path, audio_paths, offset_s, out_path)
            silent_path.unlink()
            if click_path is not None:
                click_path.unlink()
        else:
            shutil.move(str(silent_path), str(out_path))
    else:
        print(f"WARNING: could not find {entry.song.artist} - {entry.song.title} in the song library; no audio muxed")
        shutil.move(str(silent_path), str(out_path))

    elapsed = time.time() - t_start
    print(f"wrote {out_path}")
    print(f"elapsed: {elapsed:.1f}s" + ("  (>3min -- try --device cpu)" if elapsed > 180 else ""))
    subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
