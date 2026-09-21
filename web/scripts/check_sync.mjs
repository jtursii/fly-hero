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

// --- play/pause: one click, one flip, clock and audio in the same state ---
//
// Real clicks on the button, so main.ts's handler is in the path too. The
// hazard being checked is a click landing while an earlier audio.play()
// promise is still in flight.
async function transportChecks() {
  await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await page.waitForFunction(() => window.flyhero.audio.readyState >= 2, { timeout: 60000 });
  await page.evaluate(() => window.flyhero.clock.seek(30));

  const state = () => page.evaluate(() => ({
    playing: window.flyhero.clock.isPlaying,
    paused: window.flyhero.audio.paused,
    t: window.flyhero.clock.time,
  }));
  let ok = true;
  const say = (good, msg) => { console.log(`${good ? "PASS " : "FAIL "} ${msg}`); if (!good) ok = false; };

  // 1. Every single click toggles, and both sides agree afterwards.
  for (let i = 0; i < 6; i++) {
    const before = await state();
    await page.click("#play");
    await sleep(320);
    const after = await state();
    if (after.playing === before.playing) say(false, `click ${i + 1} did not toggle (still ${after.playing ? "playing" : "paused"})`);
    if (after.playing === after.paused) say(false, `click ${i + 1}: clock ${after.playing ? "playing" : "paused"} but audio ${after.paused ? "paused" : "playing"}`);
  }
  say(ok, "6 single clicks: each toggled, clock and audio agreed every time");

  // 2. Rapid clicks, faster than an audio play() promise settles.
  for (const n of [2, 3, 8, 9]) {
    const before = await state();
    for (let i = 0; i < n; i++) { await page.click("#play", { delay: 0 }); await sleep(25); }
    await sleep(700);
    const after = await state();
    const want = n % 2 === 0 ? before.playing : !before.playing;
    say(after.playing === want && after.playing !== after.paused,
        `${n} rapid clicks -> ${after.playing ? "playing" : "paused"} (expected ${want ? "playing" : "paused"}), ` +
        `audio ${after.paused ? "paused" : "playing"}`);
  }

  // 3. Play on a finished song restarts it rather than doing nothing.
  await page.evaluate(() => {
    window.flyhero.clock.pause();
    window.flyhero.clock.seek(window.flyhero.clock.duration);
  });
  await sleep(200);
  await page.click("#play");
  await sleep(700);
  const restarted = await state();
  say(restarted.playing && !restarted.paused && restarted.t < 5,
      `play at the end restarts (playing ${restarted.playing}, t ${restarted.t.toFixed(2)} s)`);
  await page.evaluate(() => { window.flyhero.clock.pause(); window.flyhero.clock.seek(30); });
  await sleep(200);

  // 4. After all that, playing still advances both the clock and the audio.
  const before = await page.evaluate(() => ({ t: window.flyhero.clock.time, a: window.flyhero.audio.currentTime }));
  if (!(await state()).playing) await page.click("#play");
  await sleep(1200);
  const after = await page.evaluate(() => ({ t: window.flyhero.clock.time, a: window.flyhero.audio.currentTime }));
  say(after.t - before.t > 0.8 && after.a - before.a > 0.8,
      `still running after the hammering: clock +${(after.t - before.t).toFixed(2)} s, audio +${(after.a - before.a).toFixed(2)} s`);
  await page.evaluate(() => window.flyhero.clock.pause());
  return ok;
}

console.log("\n--- transport ---");
allOk = (await transportChecks()) && allOk;

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
