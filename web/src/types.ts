/** Shapes of the Phase 6 export (`export/export_web.py`). Keep in sync with
 *  that module's docstring -- it is the format's source of truth. */

/** `[time_s, lane_mask, sustain_s]`. lane_mask bits 0..4 = green..orange. */
export type Note = [number, number, number];

export interface SongSummary {
  song_id: string;
  title: string;
  artist: string;
  seen_label: string;
  hit_rate: number;
  duration_s: number;
}

export interface Brain {
  n_neurons: number;
  activity_slots: number;
  slot_neuron_idx: number[];
  slot_kind: ("dn" | "input" | "variance")[];
  slot_allocation: { dn: number; input: number; variance: number };
  super_class_names: string[];
  scope_channels: string[];
  retina_display_pos: [number, number][];
  frame_size: number;
  pos_source_legend: Record<string, string>;
  pos_source_counts: Record<string, number>;
  checkpoint_step: number;
  activity_format: string;
  flow_edges: { count: number; format: string; selection: string };
  attribution: { connectome: string; citations: string[]; license: string };
  songs: SongSummary[];
}

export interface Manifest {
  song_id: string;
  title: string;
  artist: string;
  charter: string;
  difficulty: string;
  split: string;
  seen_label: string;
  fps: number;
  activity_fps: number;
  frames: number;
  activity_frames: number;
  duration_s: number;
  /** Audio-file time 0 lands at this song time (D22/D29's one convention). */
  audio_offset_s: number;
  audio: string;
  hit_rate: number;
  n_notes: number;
  n_hits: number;
  n_misses: number;
  n_overstrums: number;
  overstrums_per_min: number;
  retina_scale: number;
  retina_channels: number;
  activity_slots: number;
  notes: Note[];
}

/** From `rules.score_playthrough` at record time. `note_idx` is -1 for an
 *  overstrum. The site replays these; it never re-scores (invariant 4). */
export interface GameEvent {
  frame: number;
  note_idx: number;
  kind: "hit" | "miss" | "overstrum";
}

/** Whole-brain assets, fetched once on page load. */
export interface BrainAssets {
  brain: Brain;
  /** Float32 [N, 3], centred on the centroid, scaled to unit radius. */
  positions: Float32Array;
  /** Uint8 [N] super_class id. */
  classes: Uint8Array;
  /** Uint8 [N] D15 provenance: 0 soma, 1 anchor, 2 none. */
  posSource: Uint8Array;
  /** Int32 [E, 3]: presynaptic slot, postsynaptic neuron index, edge sign.
   *  Real FlyWire edges -- the brain panel's signal lines. */
  flowEdges: Int32Array;
}

/** The light per-song files: everything the highway and transport need. */
export interface SongLight {
  manifest: Manifest;
  events: GameEvent[];
  /** Uint8 [F60] bitmask: bits 0-4 frets (green..orange), bit 5 strum. */
  actions: Uint8Array;
  /** Uint8 [F60, 6] decoder probabilities x 255 (5 frets + strum). */
  probs: Uint8Array;
}

/** The heavy per-song files: fetched only once a song is selected (D45),
 *  and only by the panels that need them (the brain, today). */
export interface SongHeavy {
  /** Uint8 [F20, K/2], 4-bit: slot 2i low nibble, 2i+1 high nibble. */
  activity: Uint8Array;
  /** Uint8 [F20, C] scaled by manifest.retina_scale. */
  retina: Uint8Array;
  /** Float32 [F60, S], channels named by brain.scope_channels. */
  scope: Float32Array;
}

/** Per-note replay state, derived once from events.json when a song loads. */
export interface NoteState {
  /** Song time of the hit event, or -1. */
  hitAt: number;
  /** Song time of the miss event (when the hit window closed), or -1. */
  missAt: number;
}
