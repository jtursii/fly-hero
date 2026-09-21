/** Loaders for the Phase 6 export.
 *
 *  Split in two on purpose (PLAN Phase 7 task 8, D45): the whole-brain assets
 *  are ~2 MB and load once at boot, while a single song's data is up to
 *  21 MB, so it is fetched only when that song is selected -- and the heavy
 *  part of it (activity/retina/scope) only when a panel actually needs it.
 */

import type {
  Brain,
  BrainAssets,
  GameEvent,
  Manifest,
  NoteState,
  SongHeavy,
  SongLight,
} from "./types.ts";

const DATA = "data";

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function fetchFloat32(url: string): Promise<Float32Array> {
  const bytes = await fetchBytes(url);
  // .bin files are little-endian and 4-byte aligned by construction, so the
  // buffer can be reinterpreted rather than copied.
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function fetchInt32(url: string): Promise<Int32Array> {
  const bytes = await fetchBytes(url);
  return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Everything needed before a song is picked. */
export async function loadBrainAssets(): Promise<BrainAssets> {
  const brain = await fetchJson<Brain>(`${DATA}/brain.json`);
  const [positions, classes, posSource, flowEdges] = await Promise.all([
    fetchFloat32(`${DATA}/positions.bin`),
    fetchBytes(`${DATA}/classes.bin`),
    fetchBytes(`${DATA}/pos_source.bin`),
    fetchInt32(`${DATA}/flow_edges.bin`),
  ]);
  const n = brain.n_neurons;
  if (positions.length !== n * 3) throw new Error(`positions.bin: ${positions.length} != ${n * 3}`);
  if (classes.length !== n) throw new Error(`classes.bin: ${classes.length} != ${n}`);
  if (posSource.length !== n) throw new Error(`pos_source.bin: ${posSource.length} != ${n}`);
  const wantEdges = brain.flow_edges.count * 3;
  if (flowEdges.length !== wantEdges) {
    throw new Error(`flow_edges.bin: ${flowEdges.length} != ${wantEdges}`);
  }
  return { brain, positions, classes, posSource, flowEdges };
}

/** The per-song files the highway, transport and stat strip need (~0.5 MB). */
export async function loadSongLight(songId: string): Promise<SongLight> {
  const dir = `${DATA}/songs/${songId}`;
  const [manifest, events, actions, probs] = await Promise.all([
    fetchJson<Manifest>(`${dir}/manifest.json`),
    fetchJson<GameEvent[]>(`${dir}/events.json`),
    fetchBytes(`${dir}/actions.bin`),
    fetchBytes(`${dir}/probs.bin`),
  ]);
  if (actions.length !== manifest.frames) {
    throw new Error(`actions.bin: ${actions.length} != ${manifest.frames} frames`);
  }
  if (probs.length !== manifest.frames * 6) {
    throw new Error(`probs.bin: ${probs.length} != ${manifest.frames * 6}`);
  }
  return { manifest, events, actions, probs };
}

/** The recording itself (up to 21 MB). Fetched lazily, after the light
 *  files, so the transport and highway start without waiting on it (D45). */
export async function loadSongHeavy(songId: string, manifest: Manifest): Promise<SongHeavy> {
  const dir = `${DATA}/songs/${songId}`;
  const [activity, retina, scope] = await Promise.all([
    fetchBytes(`${dir}/activity.bin`),
    fetchBytes(`${dir}/retina.bin`),
    fetchFloat32(`${dir}/scope.bin`),
  ]);
  const want = manifest.activity_frames * (manifest.activity_slots / 2);
  if (activity.length !== want) throw new Error(`activity.bin: ${activity.length} != ${want}`);
  return { activity, retina, scope };
}

/** Collapse events.json into one record per note, in note order.
 *
 *  rules.py emits at most one hit or one miss per note, so this is a
 *  regrouping of what the scorer already decided -- not a re-score. */
export function buildNoteStates(manifest: Manifest, events: GameEvent[]): NoteState[] {
  const states: NoteState[] = manifest.notes.map(() => ({ hitAt: -1, missAt: -1 }));
  for (const e of events) {
    if (e.note_idx < 0 || e.note_idx >= states.length) continue; // overstrum
    const t = e.frame / manifest.fps;
    if (e.kind === "hit") states[e.note_idx].hitAt = t;
    else if (e.kind === "miss") states[e.note_idx].missAt = t;
  }
  return states;
}

/** Overstrum event times in song seconds, ascending. */
export function overstrumTimes(manifest: Manifest, events: GameEvent[]): Float64Array {
  const ts = events.filter((e) => e.kind === "overstrum").map((e) => e.frame / manifest.fps);
  ts.sort((a, b) => a - b);
  return Float64Array.from(ts);
}
