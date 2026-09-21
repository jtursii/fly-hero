/** Drives the running dev server in headless Chromium and measures whether
 *  the master clock stays locked to the audio file.
 *
 *  What this proves: the highway puts a note on the strikeline exactly when
 *  the playhead reaches that note's time (u = 0 in highway.ts, by
 *  construction), so "notes land on the audio" reduces to "the playhead
 *  equals audio time plus the manifest's offset". That difference is sampled
 *  here at 1x, 0.25x and 2x, from the start and after scrubbing.
 *
 *  The bar is one animation frame's worth of media time (16.7 ms at 1x,
 *  33.3 ms at 2x): a visual clock cannot resolve finer than the frame it
 *  draws on, and `audio.currentTime` itself jitters at roughly that scale, so
 *  an error below it is neither visible nor measurable. Reported alongside is
 *  the settle time after a scrub, which must be under half a second.
 *
 *  Headless Chromium software-renders at ~30 fps, so these are pessimistic;
 *  it is also why nothing here is a frame-rate measurement.
 *
 *  Usage:  npm run dev            # in another shell
 *          node scripts/check_sync.mjs [url]
 */

import { chromium } from "playwright";

const URL = (process.argv[2] ?? "http://localhost:5173/") + "?nobrain=1";
const FRAME_MS = 1000 / 60;
/** Steady-state budget: one animation frame of media time at the given rate. */
const budgetFor = (rate) => FRAME_MS * rate;
/** A scrub may take this long to re-lock before it counts as a failure. */
const SETTLE_BUDGET_MS = 500;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  args: [
    "--autoplay-policy=no-user-gesture-required",
    "--mute-audio",
    "--use-gl=angle",
    "--use-angle=swiftshader",
  ],
});
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });

const problems = [];
page.on("console", (m) => {
  // SwiftShader emits GL performance warnings that say nothing about the app.
  if (m.type() === "error") problems.push(`console error: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("requestfailed", (r) => {
  // Switching songs aborts the previous mp3's in-flight load; that is the
  // media element doing the right thing, not a broken request.
  const aborted = r.failure()?.errorText === "net::ERR_ABORTED";
  if (!(aborted && r.url().endsWith(".mp3"))) {
    problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`);
  }
});

await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });

const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);
console.log(`songs: ${songs.map((s) => `${s.title} (${s.seen_label})`).join(", ")}`);

async function measure(label, { seekTo, rate, seconds }) {
  const res = await page.evaluate(
    async ({ seekTo, rate, seconds }) => {
      const { clock, audio, getSong } = window.flyhero;
      const offset = getSong().manifest.audio_offset_s;
      clock.setRate(rate);
      clock.seek(seekTo);
      await clock.play();
      const samples = []; // [ms since play, raw drift ms]
      let stalls = 0;
      let prevWall = performance.now();
      const t0 = prevWall;
      while (performance.now() - t0 < seconds * 1000) {
        await new Promise((r) => requestAnimationFrame(r));
        const now = performance.now();
        const gap = now - prevWall;
        prevWall = now;
        if (audio.paused || audio.seeking) continue;
        const ct = audio.currentTime;
        // A stalled animation frame means the clock was not ticked; the
        // recovery that follows is the renderer's fault, not the clock's.
        // (Headless SwiftShader stalls on the WebGL panel; real GPUs do not.)
        if (gap > 40) {
          stalls++;
          continue;
        }
        samples.push([now - t0, (clock.time - (ct + offset)) * 1000]);
      }
      clock.pause();
      return { samples, stalls, frames: samples.length + stalls, audioEnd: audio.currentTime };
    },
    { seekTo, rate, seconds },
  );

  const { samples, stalls } = res;
  if (samples.length < 30) throw new Error(`${label}: only ${samples.length} samples`);
  const budget = budgetFor(rate);

  // Settle time: the last moment at which the drift was still outside budget.
  let settleMs = 0;
  for (const [ms, d] of samples) if (Math.abs(d) > budget) settleMs = ms;
  const steady = samples.filter(([ms]) => ms > settleMs).map(([, d]) => d);
  const steadyMax = steady.length ? Math.max(...steady.map(Math.abs)) : Infinity;
  const steadyMean = steady.length ? steady.reduce((a, b) => a + b, 0) / steady.length : NaN;
  const advanced = res.audioEnd - Math.max(0, seekTo);

  const ok =
    settleMs <= SETTLE_BUDGET_MS &&
    steadyMax <= budget &&
    steady.length > 20 &&
    advanced > seconds * rate * 0.5;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${label.padEnd(26)} ` +
      `settle ${String(Math.round(settleMs)).padStart(4)} ms | ` +
      `steady mean ${steadyMean.toFixed(1).padStart(5)} ms, |max| ${steadyMax.toFixed(1).padStart(5)} ms ` +
      `(budget ${budget.toFixed(1)} ms = 1 frame of media time) | ` +
      `${(res.frames / seconds).toFixed(0)} fps, ${stalls} stalled | ` +
      `audio +${advanced.toFixed(2)} s`,
  );
  return ok;
}

let allOk = true;
for (const song of songs) {
  await page.evaluate((id) => window.flyhero.selectSong(id), song.song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await page.waitForFunction(() => window.flyhero.audio.readyState >= 2, { timeout: 60000 });
  console.log(`\n--- ${song.title} ---`);
  allOk = (await measure("1x from 0", { seekTo: 0, rate: 1, seconds: 7 })) && allOk;
  allOk = (await measure("1x, scrub to 90 s", { seekTo: 90, rate: 1, seconds: 7 })) && allOk;
  allOk = (await measure("0.25x, scrub to 45 s", { seekTo: 45, rate: 0.25, seconds: 6 })) && allOk;
  allOk = (await measure("2x, scrub to 120 s", { seekTo: 120, rate: 2, seconds: 6 })) && allOk;
}

// Frames with the playhead parked on dense stretches, for eyeballing.
await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
await page.waitForFunction(() => window.flyhero.getSong() !== null);
for (const [t, name] of [[63.5, "_shot_full.png"], [96.2, "_shot_crt.png"]]) {
  await page.evaluate((x) => window.flyhero.clock.seek(x), t);
  await sleep(500);
  await page.screenshot({ path: `scripts/${name}` });
}

console.log(`\nconsole/network problems: ${problems.length}`);
for (const p of problems.slice(0, 20)) console.log("  " + p);

await browser.close();
if (!allOk || problems.length) process.exit(1);
console.log("\nALL SYNC CHECKS PASS");
