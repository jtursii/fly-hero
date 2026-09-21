"""Phase 6 task 2: turn an export/record.py run into web/public/ assets.

Writes, under `web_dir`:
  data/brain.json      neuron count, the K activity slots and what they are,
                       super_class names, retina display coordinates,
                       FlyWire attribution, format notes.
  data/positions.bin   Float32 [N, 3], centered on the centroid and scaled to
                       unit radius (FlyWire nm coordinates, D15's per-neuron
                       soma-else-anchor resolution).
  data/classes.bin     Uint8 [N] super_class_id.
  data/pos_source.bin  Uint8 [N] D15 provenance: 0 soma, 1 anchor, 2 none.
  data/flow_edges.bin  Int32 [E, 3], real edges for the site's signal lines
                       (presynaptic slot, postsynaptic neuron, sign).
  data/songs/<id>/manifest.json, activity.bin, retina.bin, actions.bin,
                  probs.bin, scope.bin, events.json
  audio/<id>.mp3       128 kbps full mix, D37 (showcase songs only, private
                       repo only -- revisit before any public release).

Formats (all little-endian, which every platform this ships to is):
  activity.bin  Uint8 [F20, K/2], 4-bit: slot 2i in the low nibble of byte i,
                slot 2i+1 in the high nibble. Level L means rate L/15.
  retina.bin    Uint8 [F20, C], scaled by manifest's retina_scale.
  actions.bin   Uint8 [F60] bitmask: bits 0-4 frets (green..orange), bit 5 strum.
  probs.bin     Uint8 [F60, 6] decoder probabilities x 255 (5 frets + strum),
                the raw model outputs the actions were decoded from.
  scope.bin     Float32 [F60, S] oscilloscope traces, channels in brain.json.

Usage:
  uv run python -m export.export_web --config configs/export.yaml --record-dir runs/<record run>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from flyhero.library.scan import find_song_folder_by_id
from flyhero.utils.config import load_config

ATTRIBUTION = {
    "connectome": "FlyWire FAFB v783",
    "citations": [
        "Dorkenwald et al. 2024, Nature (FlyWire whole-brain connectome)",
        "Schlegel et al. 2024, Nature (FlyWire cell-type annotations)",
    ],
    "license": "CC BY 4.0",
}


def pack_4bit(levels: np.ndarray) -> np.ndarray:
    """levels[F, K] uint8 in 0..15 -> packed[F, K/2] uint8, slot 2i in the
    low nibble. K must be even."""
    lo = levels[:, 0::2]
    hi = levels[:, 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def choose_slots(activity_by_song: list[np.ndarray], dn_idx: np.ndarray, input_idx: np.ndarray, k: int) -> dict:
    """The K neurons the site animates, the same set for every song (PLAN).

    PLAN's "input + DNs + top 15k by variance" doesn't fit K=4096 (D44):
    inputs alone are 11,118. Allocation instead, in priority order:
      1. every DN (1,303) -- invariant 2 makes them the neurons the actions
         actually come from, so they're the ones worth watching;
      2. an evenly spaced 1,024-photoreceptor sample of the input layer, so
         the visual drive is visible without spending 11k slots on it;
      3. the rest filled by variance across all showcase songs (excluding
         slots already taken), i.e. the neurons that actually move.
    """
    n_nodes = activity_by_song[0].shape[1]
    taken = np.zeros(n_nodes, dtype=bool)

    slots = list(dn_idx)
    taken[dn_idx] = True

    n_photo = min(1024, k - len(slots), len(input_idx))
    photo_sample = input_idx[np.linspace(0, len(input_idx) - 1, n_photo).round().astype(np.int64)]
    photo_sample = np.unique(photo_sample)
    photo_sample = photo_sample[~taken[photo_sample]]
    slots.extend(photo_sample.tolist())
    taken[photo_sample] = True

    # Streaming variance over the concatenated 20 Hz activity of every song
    # (levels 0..15; ranking is scale-free so the quantization is harmless).
    total_n = sum(len(a) for a in activity_by_song)
    s1 = np.zeros(n_nodes, dtype=np.float64)
    s2 = np.zeros(n_nodes, dtype=np.float64)
    for a in activity_by_song:
        af = a.astype(np.float64)
        s1 += af.sum(axis=0)
        s2 += (af * af).sum(axis=0)
    var = s2 / total_n - (s1 / total_n) ** 2
    var[taken] = -np.inf
    n_fill = k - len(slots)
    fill = np.argpartition(-var, n_fill)[:n_fill]
    fill = fill[np.argsort(-var[fill])]
    slots.extend(fill.tolist())

    slot_idx = np.asarray(slots, dtype=np.int64)
    assert len(slot_idx) == k and len(np.unique(slot_idx)) == k, (len(slot_idx), k)
    kinds = ["dn"] * len(dn_idx) + ["input"] * len(photo_sample) + ["variance"] * n_fill
    return dict(slot_idx=slot_idx, kinds=kinds, n_dn=len(dn_idx), n_input=len(photo_sample), n_variance=n_fill)


#: super_class names grouped into the three stages the site draws signal
#: lines between. Everything not listed is "central".
FLOW_OPTIC = ("optic", "visual_projection", "visual_centrifugal", "sensory")
FLOW_DN = ("descending",)
#: How many edges of each (from-stage, to-stage) kind to keep, highest
#: syn_count first. The three forward-pathway kinds are uncapped (there are
#: only 747 of them); the intra-stage ones are capped so the lines stay
#: legible and the file stays small.
FLOW_BUDGET = {(0, 1): None, (1, 2): None, (0, 2): None, (0, 0): 1000, (1, 1): 400}


def flow_edges(graph, slot_idx: np.ndarray, super_class_names: list[str]) -> np.ndarray:
    """Real connectome edges for the site's signal lines -> Int32 [E, 3] of
    (presynaptic slot, postsynaptic neuron index, sign).

    Every edge here is a real FlyWire edge with its real sign; nothing is
    synthesised. The selection rule is *presynaptic neuron is one of the K
    recorded slots*, because the site colours a line by the recorded
    activity of the neuron the signal leaves from -- the postsynaptic
    neuron need not be recorded, and usually isn't (only 198 edges have
    both ends recorded, and none of those run optic -> central).

    Kept, highest syn_count first: every optic->central, central->DN and
    optic->DN edge, plus FLOW_BUDGET's sample of the intra-optic and
    intra-central ones so the optic lobes are not bare.
    """
    sc = graph["super_class_id"]
    name_to_id = {n: i for i, n in enumerate(super_class_names)}
    stage = np.ones(len(sc), dtype=np.int8)  # 1 = central
    stage[np.isin(sc, [name_to_id[n] for n in FLOW_OPTIC if n in name_to_id])] = 0
    stage[np.isin(sc, [name_to_id[n] for n in FLOW_DN if n in name_to_id])] = 2

    slot_of = np.full(len(sc), -1, dtype=np.int64)
    slot_of[slot_idx] = np.arange(len(slot_idx))
    pre, post = graph["pre"], graph["post"]
    keep = slot_of[pre] >= 0
    pre, post = pre[keep], post[keep]
    syn, sign = graph["syn_count"][keep], graph["sign"][keep]
    kinds = list(zip(stage[pre].tolist(), stage[post].tolist()))
    kinds = np.array([k[0] * 3 + k[1] for k in kinds], dtype=np.int64)

    rows = []
    for (a, b), budget in FLOW_BUDGET.items():
        idx = np.flatnonzero(kinds == a * 3 + b)
        # Deterministic: strongest connections first, ties broken by edge id.
        idx = idx[np.lexsort((idx, -syn[idx]))]
        if budget is not None:
            idx = idx[:budget]
        rows.append(np.stack([slot_of[pre[idx]], post[idx], sign[idx]], axis=1))
    return np.concatenate(rows).astype(np.int32)


def normalize_positions(pos_nm: np.ndarray) -> np.ndarray:
    """Centered on the centroid, scaled so the farthest neuron sits at
    radius 1 -- the site's camera framing assumes this."""
    pos = pos_nm.astype(np.float64)
    pos = pos - pos.mean(axis=0)
    radius = np.linalg.norm(pos, axis=1).max()
    return (pos / radius).astype(np.float32)


def notes_json(notes: np.ndarray) -> list[list]:
    """[time_s, lane_mask, sustain_s] per note -- the highway's note source
    (the site never sees chart files, D9)."""
    return [[round(float(n["time_s"]), 4), int(n["lane_mask"]), round(float(n["sustain_s"]), 4)] for n in notes]


def encode_audio(src_paths: list[Path], out_path: Path, bitrate: str) -> None:
    """Mix the song folder's stems to one 128 kbps MP3 (D37). Copies audio
    out of the read-only library; never writes back into it."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y"]
    for p in src_paths:
        cmd += ["-i", str(p)]
    if len(src_paths) > 1:
        cmd += ["-filter_complex", f"amix=inputs={len(src_paths)}:duration=longest:normalize=0"]
    cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate, "-ac", "2", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/export.yaml")
    p.add_argument("--record-dir", required=True)
    p.add_argument("--skip-audio", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    record_dir = Path(args.record_dir)
    rec_meta = json.loads((record_dir / "record.json").read_text())
    processed_dir = Path(cfg["processed_dir"])
    web_dir = Path(cfg["web_dir"])
    data_dir = web_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    graph = np.load(processed_dir / "graph.npz")
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    n_nodes = len(graph["type_id"])

    recs = {s["song_id"]: np.load(record_dir / f"{s['song_id']}.npz", allow_pickle=False)
            for s in rec_meta["songs"]}
    slots = choose_slots(
        [recs[s["song_id"]]["activity"] for s in rec_meta["songs"]],
        graph["dn_idx"].astype(np.int64), graph["input_idx_photoreceptor"].astype(np.int64),
        cfg["activity_slots"],
    )
    slot_idx = slots["slot_idx"]
    print(f"slots: {len(slot_idx)} = {slots['n_dn']} DN + {slots['n_input']} input + {slots['n_variance']} variance")

    normalize_positions(graph["pos_nm"]).tofile(data_dir / "positions.bin")
    graph["super_class_id"].astype(np.uint8).tofile(data_dir / "classes.bin")
    graph["pos_source_id"].astype(np.uint8).tofile(data_dir / "pos_source.bin")
    flow = flow_edges(graph, slot_idx, meta["super_class_names"])
    flow.tofile(data_dir / "flow_edges.bin")
    print(f"flow edges: {len(flow)} real connections from recorded neurons")

    songs_out = []
    for s in rec_meta["songs"]:
        sid = s["song_id"]
        rec = recs[sid]
        song_dir = data_dir / "songs" / sid
        song_dir.mkdir(parents=True, exist_ok=True)

        pack_4bit(rec["activity"][:, slot_idx]).tofile(song_dir / "activity.bin")
        retina = rec["retina"]
        retina_scale = float(np.percentile(np.abs(retina), 99.5)) or 1.0
        np.clip(np.rint(np.abs(retina) / retina_scale * 255), 0, 255).astype(np.uint8).tofile(song_dir / "retina.bin")
        rec["actions"].astype(np.uint8).tofile(song_dir / "actions.bin")
        np.clip(np.rint(rec["probs"] * 255), 0, 255).astype(np.uint8).tofile(song_dir / "probs.bin")
        rec["scope"].astype(np.float32).tofile(song_dir / "scope.bin")

        events = [dict(frame=int(f), note_idx=int(n), kind=str(k)) for f, n, k in
                  zip(rec["event_frame"], rec["event_note_idx"], rec["event_kind"])]
        (song_dir / "events.json").write_text(json.dumps(events, separators=(",", ":")))

        manifest = dict(
            song_id=sid, title=s["title"], artist=s["artist"], charter=s["charter"],
            difficulty=s["difficulty"], split=s["split"], seen_label=s["seen_label"],
            fps=rec_meta["fps"], activity_fps=rec_meta["activity_fps"],
            frames=s["frames"], activity_frames=s["activity_frames"],
            duration_s=round(s["duration_s"], 3),
            # D22/D29's one sync convention: audio-file time 0 lands at this song time.
            audio_offset_s=round(-(s["chart_offset_s"] + s["ini_delay_s"]), 4),
            audio=f"audio/{sid}.mp3",
            # Not rounded: validate_export.py re-scores the exported actions and
            # compares exactly, so this must be the full-precision value.
            hit_rate=float(s["metrics"]["hit_rate"]),
            n_notes=int(s["metrics"]["n_notes"]), n_hits=int(s["metrics"]["n_hits"]),
            n_misses=int(s["metrics"]["n_misses"]), n_overstrums=int(s["metrics"]["n_overstrums"]),
            overstrums_per_min=round(float(s["metrics"]["overstrums_per_min"]), 3),
            retina_scale=retina_scale, retina_channels=int(retina.shape[1]),
            activity_slots=int(len(slot_idx)), notes=notes_json(rec["notes"]),
        )
        (song_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

        if not args.skip_audio:
            from flyhero.game.sim import select_audio_stems

            library_root = Path(load_config("configs/paths.yaml")["song_library"]).expanduser()
            folder = find_song_folder_by_id(library_root, processed_dir / "song_id_index.json", sid)
            if folder is None:
                print(f"  WARNING: no library folder for {sid}; no audio exported")
            else:
                encode_audio(select_audio_stems(folder, "full"), web_dir / "audio" / f"{sid}.mp3",
                             cfg["audio_bitrate"])
        songs_out.append(dict(song_id=sid, title=s["title"], artist=s["artist"],
                              seen_label=s["seen_label"], hit_rate=manifest["hit_rate"],
                              duration_s=manifest["duration_s"]))
        print(f"  {s['artist']} - {s['title']}: {s['frames']} frames, hit_rate {manifest['hit_rate']:.4f}")

    brain = dict(
        n_neurons=int(n_nodes), activity_slots=int(len(slot_idx)),
        slot_neuron_idx=slot_idx.tolist(), slot_kind=slots["kinds"],
        slot_allocation=dict(dn=slots["n_dn"], input=slots["n_input"], variance=slots["n_variance"]),
        super_class_names=meta["super_class_names"],
        scope_channels=rec_meta["scope_channels"],
        retina_display_pos=rec_meta["retina_display_pos"],
        frame_size=load_config("configs/game.yaml")["frame_size"],
        pos_source_legend={"0": "soma", "1": "anchor", "2": "none"},
        pos_source_counts=meta["pos_source_counts"],
        checkpoint_step=rec_meta["step"],
        activity_format="uint8 [frames, slots/2], 4-bit: slot 2i low nibble, 2i+1 high nibble, level/15 = rate",
        flow_edges=dict(
            count=int(len(flow)),
            format="int32 [E, 3]: presynaptic slot, postsynaptic neuron index, sign (+1 excitatory, -1 inhibitory)",
            selection="real FlyWire edges whose presynaptic neuron is a recorded slot; "
                      "every optic->central, central->DN and optic->DN edge, plus the "
                      f"{FLOW_BUDGET[(0, 0)]} / {FLOW_BUDGET[(1, 1)]} strongest intra-optic / intra-central ones",
        ),
        attribution=ATTRIBUTION, songs=songs_out,
    )
    (data_dir / "brain.json").write_text(json.dumps(brain, separators=(",", ":")))
    print(f"wrote {data_dir}")


if __name__ == "__main__":
    main()
