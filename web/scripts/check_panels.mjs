/** Checks the Session 4 panels against the files they claim to show, and
 *  measures the frame time with every panel live.
 *
 *  The decision bars and the oscilloscope are read back through
 *  `window.flyhero.getDecision()/getScope()`, and compared against
 *  `probs.bin`, `actions.bin` and `scope.bin` re-fetched and re-indexed here
 *  -- a second implementation of the mapping, so a swapped channel, a
 *  swapped bit or an off-by-one frame fails instead of looking plausible.
 *
 *  It also asserts the decoder's own rule over the whole song: every fret
 *  bit in actions.bin is exactly `prob > 0.5`, and every strum frame is a
 *  local maximum above 0.5 (decoder.py's plateau tiebreak included).
 *
 *  The strum rule is checked to within one quantization step, and that is a
 *  fact about the export, not a loosened test: decode_strum ran on float64
 *  probabilities, while probs.bin stores them as bytes, so two neighbours
 *  less than 1/255 apart tie once quantized and their peak is no longer
 *  recoverable from the shipped file (20 of What I've Done's 397 strums;
 *  against the recorder's own float probabilities there are 0 exceptions).
 *  It is the reason the panel reads fired-ness from actions.bin instead of
 *  re-deciding it from the bars -- the bars alone cannot always tell.
 *
 *  Usage:  npm run dev  &&  node scripts/check_panels.mjs
 */

import { chromium } from "playwright";

const URL = process.argv[2] ?? "http://localhost:5173/";
const SONG = 0;
/** Times to compare, including ones between frames so the interpolation is
 *  exercised, and one revisited after a scrub. */
const TIMES = [12.0, 12.008, 60.5, 109.03, 109.0371, 150.25];
const FPS_SECONDS = 8;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (ok, msg) => {
  if (!ok) failures++;
  console.log(`${ok ? "  ok  " : "  FAIL"} ${msg}`);
};

// Headed: software rendering the connectome panel starves the compositor
// and rAF stops firing (same reason check_brain.mjs is headed).
const browser = await chromium.launch({ headless: false, args: ["--mute-audio"] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
page.on("console", (m) => {
  if (m.type() === "error") console.log(`  console error: ${m.text()}`);
});
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });

const assets = await page.evaluate(() => window.flyhero.getAssets().brain);
const song = assets.songs[SONG];
console.log(`${song.title} — ${song.artist}`);
await page.evaluate((id) => window.flyhero.selectSong(id), song.song_id);
await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
// The oscilloscope needs the heavy files; they land a moment after the rest.
await page.waitForFunction(() => !Number.isNaN(window.flyhero.getScope().sample(0).exc), { timeout: 60000 });

console.log("\nchannels");
check(assets.scope_channels.includes("exc") && assets.scope_channels.includes("inh"),
  `scope.bin carries exc/inh (${assets.scope_channels.length} channels)`);

// --- the panels against the raw files -------------------------------------

const res = await page.evaluate(async ({ songId, times, channels }) => {
  const dir = `data/songs/${songId}`;
  const [probsBuf, actionsBuf, scopeBuf, manifest] = await Promise.all([
    fetch(`${dir}/probs.bin`).then((r) => r.arrayBuffer()),
    fetch(`${dir}/actions.bin`).then((r) => r.arrayBuffer()),
    fetch(`${dir}/scope.bin`).then((r) => r.arrayBuffer()),
    fetch(`${dir}/manifest.json`).then((r) => r.json()),
  ]);
  const probs = new Uint8Array(probsBuf);
  const actions = new Uint8Array(actionsBuf);
  const scope = new Float32Array(scopeBuf);
  const fps = manifest.fps;
  const nCh = channels.length;
  const excCh = channels.indexOf("exc");
  const inhCh = channels.indexOf("inh");
  // Row order on the panel is strum first, then the five frets.
  const rowChannel = [5, 0, 1, 2, 3, 4];

  const dn = window.flyhero.getDecision();
  const sc = window.flyhero.getScope();
  const rows = [];
  for (const t of times) {
    const x = t * fps;
    const f0 = Math.floor(x);
    const f1 = Math.min(f0 + 1, manifest.frames - 1);
    const w = x - f0;
    const shown = dn.sample(t);
    const want = rowChannel.map((c) => {
      const a = probs[f0 * 6 + c] / 255;
      const b = probs[f1 * 6 + c] / 255;
      return a + (b - a) * w;
    });
    const frame = Math.round(x);
    const firedWant = [5, 0, 1, 2, 3, 4].map((bit) => (actions[frame] >> bit) & 1);
    const s = sc.sample(t);
    rows.push({
      t,
      frame,
      probErr: Math.max(...shown.map((r, i) => Math.abs(r.prob - want[i]))),
      // Frets must match the bit exactly; the strum marker decays, so only
      // its frame-0 value is compared as a flag.
      fretMismatch: shown.slice(1).filter((r, i) => (r.fired > 0 ? 1 : 0) !== firedWant[i + 1]).length,
      strumShown: shown[0].fired,
      strumWant: firedWant[0],
      scopeErr: Math.max(
        Math.abs(s.exc - scope[frame * nCh + excCh]),
        Math.abs(s.inh - scope[frame * nCh + inhCh]),
      ),
      exc: s.exc,
      inh: s.inh,
    });
  }

  // decoder.py's rule over the whole recording.
  let fretBad = 0;
  let strumBad = 0;
  let strumTied = 0;
  let strums = 0;
  for (let f = 0; f < manifest.frames; f++) {
    for (let lane = 0; lane < 5; lane++) {
      const p = probs[f * 6 + lane] / 255;
      const bit = (actions[f] >> lane) & 1;
      // probs.bin is quantized to 1/255, so only values that round onto the
      // threshold itself are ambiguous; those are skipped, not excused.
      if (Math.abs(p - 0.5) < 1 / 255) continue;
      if ((p > 0.5 ? 1 : 0) !== bit) fretBad++;
    }
    const s = (actions[f] >> 5) & 1;
    if (s) {
      strums++;
      const p = probs[f * 6 + 5] / 255;
      const left = f > 0 ? probs[(f - 1) * 6 + 5] / 255 : -1;
      const right = f < manifest.frames - 1 ? probs[(f + 1) * 6 + 5] / 255 : -1;
      if (!(p > 0.5)) strumBad++;
      else if (!(p > left && p >= right)) {
        // Tied with a neighbour only after quantization; anything beyond one
        // step is a genuine mismatch and still fails.
        if (p >= left - 1 / 255 && p >= right - 1 / 255) strumTied++;
        else strumBad++;
      }
    }
  }
  return { rows, fretBad, strumBad, strumTied, strums, frames: manifest.frames, yMax: sc.sample(0).yMax };
}, { songId: song.song_id, times: TIMES, channels: assets.scope_channels });

console.log("\ndecision bars vs probs.bin / actions.bin");
for (const r of res.rows) {
  check(r.probErr < 1e-9, `t=${r.t}s (frame ${r.frame}): max |prob shown - file| = ${r.probErr.toExponential(1)}`);
  check(r.fretMismatch === 0, `t=${r.t}s: 5 fret lit-flags match actions.bin`);
}
check(res.rows.every((r) => r.strumWant === 0 || r.strumShown > 0),
  "a strum frame always shows the strum marker lit");
check(res.fretBad === 0,
  `every fret bit over ${res.frames} frames is exactly prob > 0.50 (${res.fretBad} exceptions)`);
check(res.strumBad === 0,
  `all ${res.strums} strums are above 0.50 and a local maximum (${res.strumBad} exceptions)`);
console.log(`  note  ${res.strumTied} of ${res.strums} strums tie a neighbour once probs.bin is`
  + ` quantized to bytes, so their peak is not recoverable from the file`);

console.log("\noscilloscope vs scope.bin");
for (const r of res.rows) {
  check(r.scopeErr < 1e-9, `t=${r.t}s: exc ${r.exc.toFixed(5)} / inh ${r.inh.toFixed(5)}, err ${r.scopeErr.toExponential(1)}`);
}
check(res.yMax > 0, `y-axis fixed per song at ${res.yMax.toFixed(4)} mean rate`);

// --- scrubbing ------------------------------------------------------------

const scrub = await page.evaluate(async (t) => {
  const dn = window.flyhero.getDecision();
  const sc = window.flyhero.getScope();
  const before = { dn: dn.sample(t).map((r) => r.prob), sc: sc.sample(t) };
  window.flyhero.clock.seek(240);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  window.flyhero.clock.seek(t);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const after = { dn: dn.sample(t).map((r) => r.prob), sc: sc.sample(t) };
  return { before, after };
}, TIMES[3]);
check(JSON.stringify(scrub.before) === JSON.stringify(scrub.after),
  "both panels read identically after scrubbing away and back");

// --- frame time, everything live ------------------------------------------

console.log(`\nframe time (${FPS_SECONDS}s playing, all panels live)`);
await page.evaluate(() => {
  window.flyhero.clock.seek(100);
  window.flyhero.clock.play();
  window.__f = [];
  const tick = (t) => {
    if (window.__last) window.__f.push(t - window.__last);
    window.__last = t;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
await sleep(FPS_SECONDS * 1000);
const fps = await page.evaluate(() => {
  const d = window.__f.slice(10).sort((a, b) => a - b);
  const q = (p) => d[Math.min(d.length - 1, Math.floor(d.length * p))];
  return { n: d.length, p50: q(0.5), p95: q(0.95), max: d[d.length - 1] };
});
console.log(`  n=${fps.n}  median ${fps.p50.toFixed(2)} ms  p95 ${fps.p95.toFixed(2)} ms  max ${fps.max.toFixed(2)} ms`);
check(fps.p95 <= 16.7, `p95 frame time within the 16.7 ms 60 fps budget`);
await page.evaluate(() => window.flyhero.clock.pause());

// --- screenshots ----------------------------------------------------------

await page.evaluate(() => window.flyhero.clock.seek(109.03));
await sleep(600);
await page.screenshot({ path: "scripts/_shot_panels_full.png" });
for (const [sel, name] of [["#dn-panel", "decision"], ["#scope-panel", "scope"]]) {
  const box = await page.locator(sel).boundingBox();
  await page.screenshot({ path: `scripts/_shot_${name}.png`, clip: box });
}
await page.click("#about-open");
await sleep(300);
await page.screenshot({ path: "scripts/_shot_about.png" });
check(await page.locator("#about").isVisible(), "About modal opens");
await page.keyboard.press("Escape");
await sleep(200);
check(await page.locator("#about").isHidden(), "Escape closes it");

await browser.close();
console.log(failures ? `\n${failures} CHECK(S) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
