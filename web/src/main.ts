/** App wiring: one clock, one animation frame, every panel reading the same
 *  playhead (PLAN Phase 7 task 7).
 *
 *  Load order follows D45 / PLAN task 8: the whole-brain assets (~2 MB) come
 *  down at boot, and a song's own data (up to 21 MB) only once it is picked.
 */

import "./style.css";
import { MasterClock, type Rate } from "./clock.ts";
import {
  buildNoteStates, loadBrainAssets, loadSongHeavy, loadSongLight, overstrumTimes,
} from "./data.ts";
import { Highway } from "./highway.ts";
import { BrainView } from "./brain.ts";
import { RetinaEncoder, RetinaView } from "./retina.ts";
import { DecisionView } from "./decision.ts";
import { ScopeView } from "./scope.ts";
import type { BrainAssets, GameEvent, Manifest, SongLight } from "./types.ts";

const RATES: Rate[] = [0.25, 1, 2];

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing from index.html`);
  return el as T;
};

const el = {
  ckpt: $("ckpt"),
  credit: $("credit"),
  brainCanvas: $<HTMLCanvasElement>("brain-canvas"),
  eyesCanvas: $<HTMLCanvasElement>("eyes-canvas"),
  dnCanvas: $<HTMLCanvasElement>("dn-canvas"),
  scopeCanvas: $<HTMLCanvasElement>("scope-canvas"),
  aboutOpen: $<HTMLButtonElement>("about-open"),
  aboutClose: $<HTMLButtonElement>("about-close"),
  about: $("about"),
  bActive: $("b-active"),
  bActiveK: $("b-active-k"),
  bMean: $("b-mean"),
  bFps: $("b-fps"),
  crtTitle: $("crt-title"),
  highway: $<HTMLCanvasElement>("highway"),
  led: $("crt-led"),
  songs: $("songs"),
  play: $<HTMLButtonElement>("play"),
  rates: $("rates"),
  scrubber: $<HTMLInputElement>("scrubber"),
  time: $("time"),
  counter: $("counter"),
  sync: $("sync"),
  audio: $<HTMLAudioElement>("audio"),
  sHit: $("s-hit"),
  sNotes: $("s-notes"),
  sOver: $("s-over"),
  sDiff: $("s-diff"),
};

const clock = new MasterClock();
const highway = new Highway(el.highway);
let brainView: BrainView | null = null;
let retinaView: RetinaView | null = null;
const decisionView = new DecisionView(el.dnCanvas);
const scopeView = new ScopeView(el.scopeCanvas);
let assets: BrainAssets | null = null;
let song: SongLight | null = null;
let currentId: string | null = null;
let loading = false;
/** Rolling mean of the whole frame (every panel), shown in the brain panel. */
let frameMs = 0;
let lastFrameAt = 0;
let scrubbing = false;
/** Event times in song seconds, ascending, for the live counters. */
let eventTimes = { hit: new Float64Array(0), miss: new Float64Array(0), over: new Float64Array(0) };

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
}

/** Count of ascending times <= t (upper bound), by binary search. */
function countUpTo(times: Float64Array, t: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function kindTimes(manifest: Manifest, events: GameEvent[], kind: GameEvent["kind"]): Float64Array {
  const ts = events.filter((e) => e.kind === kind).map((e) => e.frame / manifest.fps);
  ts.sort((a, b) => a - b);
  return Float64Array.from(ts);
}

// --- song selection -------------------------------------------------------

async function selectSong(id: string): Promise<void> {
  if (loading || id === currentId) return;
  loading = true;
  currentId = id;
  clock.pause();
  highway.clearSong();
  brainView?.setSong(null, null);
  retinaView?.setSong(null);
  decisionView.setSong(null);
  scopeView.setSong(null);
  song = null;
  syncSongButtons(true);

  try {
    const light = await loadSongLight(id);
    const { manifest, events, actions } = light;
    const states = buildNoteStates(manifest, events);
    song = light;
    eventTimes = {
      hit: kindTimes(manifest, events, "hit"),
      miss: kindTimes(manifest, events, "miss"),
      over: kindTimes(manifest, events, "overstrum"),
    };
    highway.setSong(manifest, states, actions, overstrumTimes(manifest, events));

    // The recording runs one post-roll second past the song, so the clock's
    // end is the frame count, not manifest.duration_s.
    const endS = manifest.frames / manifest.fps;
    el.audio.src = manifest.audio;
    el.audio.load();
    clock.setSong(el.audio, manifest.audio_offset_s, endS);
    el.scrubber.max = endS.toFixed(3);
    el.scrubber.value = "0";
    el.scrubber.disabled = false;
    el.play.disabled = false;

    // The retina is derived from the manifest's notes, so the panel is live
    // as soon as the light files land -- it never waits on the recording.
    retinaView?.setSong(manifest);
    // probs.bin and actions.bin are light files, so the decision panel is
    // live with the transport; only the oscilloscope waits on the recording.
    decisionView.setSong(manifest, light.probs, actions);
    el.crtTitle.textContent =
      `${manifest.artist} — ${manifest.title} · ${manifest.difficulty}`.toUpperCase();
    el.sHit.textContent = (manifest.hit_rate * 100).toFixed(1) + "%";
    el.sNotes.textContent = `${manifest.n_hits} / ${manifest.n_notes}`;
    el.sOver.textContent = manifest.overstrums_per_min.toFixed(1);
    el.sDiff.textContent = manifest.difficulty;
  } catch (err) {
    currentId = null;
    console.error(err);
    el.counter.textContent = `load failed: ${String(err)}`;
  } finally {
    loading = false;
    syncSongButtons(false);
  }

  // D45 / PLAN task 8: the recording itself (up to 21 MB) comes down only
  // now, and only after the light files, so the transport is usable first.
  if (song && brainView && assets && currentId === id) {
    el.bActive.textContent = "···";
    try {
      const heavy = await loadSongHeavy(id, song.manifest);
      if (currentId !== id) return; // the user moved on while it downloaded
      const scopeAll = Math.max(0, assets.brain.scope_channels.indexOf("all"));
      brainView.setSong(heavy, song.manifest, scopeAll);
      scopeView.setSong(song.manifest, heavy.scope, assets.brain.scope_channels);
    } catch (err) {
      console.error(err);
      el.bActive.textContent = "—";
    }
  }
}

function syncSongButtons(busy: boolean): void {
  for (const b of el.songs.querySelectorAll<HTMLButtonElement>(".song-btn")) {
    b.classList.toggle("active", b.dataset.id === currentId);
    b.disabled = busy;
  }
}

function buildSongButtons(a: BrainAssets): void {
  for (const s of a.brain.songs) {
    const b = document.createElement("button");
    b.className = "song-btn";
    b.dataset.id = s.song_id;
    // D52: no "practiced" badge -- every showcase song is one, so the label
    // distinguished nothing. `seen_label` stays in the manifests.
    b.innerHTML = `<span class="t">${s.title}</span><span class="a">${s.artist}</span>`;
    b.addEventListener("click", () => void selectSong(s.song_id));
    el.songs.appendChild(b);
  }
}

function buildRateButtons(): void {
  for (const r of RATES) {
    const b = document.createElement("button");
    b.className = "rate-btn" + (r === 1 ? " active" : "");
    b.textContent = `${r}×`;
    b.addEventListener("click", () => {
      clock.setRate(r);
      for (const other of el.rates.querySelectorAll(".rate-btn")) other.classList.remove("active");
      b.classList.add("active");
    });
    el.rates.appendChild(b);
  }
}

// --- frame loop -----------------------------------------------------------

function frame(nowMs: number): void {
  if (lastFrameAt) frameMs += ((nowMs - lastFrameAt) - frameMs) * 0.05;
  lastFrameAt = nowMs;
  clock.tick(nowMs);
  const t = clock.time;

  highway.draw(t);
  brainView?.render(t);
  retinaView?.render(t);
  decisionView.draw(t);
  scopeView.draw(t);

  syncPlayButton();
  el.led.classList.toggle("on", clock.isPlaying);

  if (song) {
    if (!scrubbing) el.scrubber.value = t.toFixed(3);
    el.time.textContent = `${fmtTime(t)} / ${fmtTime(clock.duration)}`;
    const hits = countUpTo(eventTimes.hit, t);
    const misses = countUpTo(eventTimes.miss, t);
    const overs = countUpTo(eventTimes.over, t);
    el.counter.textContent = `${hits} hit · ${misses} miss · ${overs} overstrum`;
    if (brainView) {
      el.bActive.textContent = brainView.stats.active.toLocaleString();
      el.bMean.textContent = brainView.stats.meanActivation.toFixed(4);
      el.bFps.textContent = `${frameMs.toFixed(1)} ms/frame`;
    }
    const ms = clock.drift * 1000;
    el.sync.textContent = `sync ${ms >= 0 ? "+" : ""}${ms.toFixed(0)} ms`;
    el.sync.classList.toggle("bad", Math.abs(ms) > 40);
  }
  requestAnimationFrame(frame);
}

// --- boot -----------------------------------------------------------------

/** The glyph is also set every frame, but a click can land while the frame
 *  loop is stalled (a song's 11-20 MB recording downloading, say), and a
 *  button that does not react reads as a missed click. */
function syncPlayButton(): void {
  el.play.innerHTML = clock.isPlaying ? "&#10073;&#10073;" : "&#9654;";
}

function toggleTransport(): void {
  clock.toggle();
  syncPlayButton();
}

function wireTransport(): void {
  el.play.addEventListener("click", toggleTransport);

  el.scrubber.addEventListener("pointerdown", () => (scrubbing = true));
  const endScrub = () => {
    if (!scrubbing) return;
    scrubbing = false;
    clock.seek(parseFloat(el.scrubber.value));
  };
  el.scrubber.addEventListener("pointerup", endScrub);
  el.scrubber.addEventListener("pointercancel", endScrub);
  // Dragging seeks live so the highway tracks the thumb; `change` catches
  // keyboard use of the slider, where no pointer events fire.
  el.scrubber.addEventListener("input", () => {
    if (scrubbing) clock.seek(parseFloat(el.scrubber.value));
  });
  el.scrubber.addEventListener("change", () => clock.seek(parseFloat(el.scrubber.value)));

  el.audio.addEventListener("loadedmetadata", () => clock.noteAudioDuration(el.audio.duration));

  const setAbout = (open: boolean) => {
    el.about.hidden = !open;
    // A modal over a playing replay is a reading surface, not a pause.
    el.aboutOpen.setAttribute("aria-expanded", String(open));
  };
  el.aboutOpen.addEventListener("click", () => setAbout(true));
  el.aboutClose.addEventListener("click", () => setAbout(false));
  el.about.addEventListener("click", (e) => {
    if (e.target === el.about) setAbout(false); // click-away on the backdrop
  });

  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.code === "Escape") {
      setAbout(false);
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      if (!el.play.disabled) toggleTransport();
    } else if (e.code === "ArrowLeft") {
      clock.seek(clock.time - 5);
    } else if (e.code === "ArrowRight") {
      clock.seek(clock.time + 5);
    }
  });

  const onResize = () => {
    highway.resize();
    brainView?.resize();
    retinaView?.resize();
    decisionView.resize();
    scopeView.resize();
  };
  window.addEventListener("resize", onResize);
  // The CRT and the brain canvas are both sized by the grid, so watch the
  // panels rather than only the window.
  const ro = new ResizeObserver(onResize);
  ro.observe(el.highway.parentElement!);
  ro.observe(el.brainCanvas.parentElement!);
  ro.observe(el.eyesCanvas.parentElement!);
  ro.observe(el.dnCanvas.parentElement!);
  ro.observe(el.scopeCanvas.parentElement!);
}

async function boot(): Promise<void> {
  wireTransport();
  buildRateButtons();
  highway.draw(0);

  assets = await loadBrainAssets();
  const b = assets.brain;
  el.ckpt.textContent = b.checkpoint_step.toLocaleString();
  el.credit.textContent =
    `${b.attribution.connectome} · ${b.n_neurons.toLocaleString()} neurons · ` +
    `${b.flow_edges.count.toLocaleString()} signal lines drawn`;
  el.bActiveK.textContent =
    `active of ${b.activity_slots.toLocaleString()} recorded (${b.slot_allocation.dn} DN, ` +
    `${b.slot_allocation.input} photoreceptors)`;
  buildSongButtons(assets);
  // ?nobrain=1 skips the WebGL panel. scripts/check_sync.mjs uses it so the
  // clock is measured on its own: headless Chromium software-renders 139k
  // points at well under 60 fps, which would swamp the numbers.
  // ?nobrain=1 skips the two panels that draw every frame (the WebGL
  // connectome and the 2D eyes). scripts/check_sync.mjs uses it so the clock
  // is measured on its own: headless Chromium software-renders both far
  // under 60 fps, which would swamp the numbers.
  if (!new URLSearchParams(location.search).has("nobrain")) {
    retinaView = new RetinaView(el.eyesCanvas, b.retina_display_pos);
    try {
      brainView = new BrainView(el.brainCanvas, assets);
    } catch (err) {
      console.warn("brain placeholder unavailable (no WebGL?)", err);
    }
  } else {
    el.brainCanvas.style.display = "none";
  }
  requestAnimationFrame(frame);
}

/** Debug handle. `scripts/check_sync.mjs` drives the transport through this
 *  and reads the clock-vs-audio drift; it is also handy in the console. */
declare global {
  interface Window {
    flyhero: {
      clock: MasterClock;
      audio: HTMLAudioElement;
      selectSong: (id: string) => Promise<void>;
      getBrain: () => BrainView | null;
      getRetina: () => RetinaView | null;
      makeRetinaEncoder: (pos: [number, number][]) => RetinaEncoder;
      getDecision: () => DecisionView;
      getScope: () => ScopeView;
      getSong: () => SongLight | null;
      getAssets: () => BrainAssets | null;
    };
  }
}
window.flyhero = {
  clock,
  audio: el.audio,
  selectSong,
  getBrain: () => brainView,
  getRetina: () => retinaView,
  // scripts/check_retina.mjs builds its own encoder with this and compares
  // it against retina.bin; the panel's own encoder is left alone.
  makeRetinaEncoder: (pos: [number, number][]) => new RetinaEncoder(pos),
  getDecision: () => decisionView,
  getScope: () => scopeView,
  getSong: () => song,
  getAssets: () => assets,
};

void boot();
