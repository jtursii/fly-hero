/** Asserts the browser-derived retina matches the recording.
 *
 *  The site does not ship the retina: `src/retina.ts` re-derives each
 *  photoreceptor's current from the manifest's notes (site plan §2 item 1).
 *  That is only legitimate if it reproduces what `flyhero/game/retina.py`
 *  computed at record time, so this compares it against `retina.bin`.
 *
 *  What is being compared, exactly:
 *    - `retina.bin` frame k is |mean of the signed current over 60 Hz frames
 *      3k, 3k+1, 3k+2|, quantized to uint8 with `manifest.retina_scale` as
 *      full scale (export_web.py takes the absolute value, so the sign is
 *      not recoverable from the recording -- only magnitude is compared).
 *    - `export/record.py` consumes the observation *after* stepping the env,
 *      so recorded 60 Hz frame f is the highway rendered at song time
 *      (f + 1) / 60. The check asserts that convention by also scoring the
 *      f/60 alignment and requiring it to be worse.
 *
 *  Usage:  npm run dev  &&  node scripts/check_retina.mjs [url]
 */
import { chromium } from "playwright";

const fail = [];
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fail.push(msg); };

const browser = await chromium.launch({ headless: false, args: ["--mute-audio"] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1050 } });
await page.goto(process.argv[2] ?? "http://localhost:5174/", { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);

for (const song of songs) {
  await page.evaluate((id) => window.flyhero.selectSong(id), song.song_id);
  await page.waitForFunction((id) => window.flyhero.getSong()?.manifest.song_id === id,
                             song.song_id, { timeout: 60000 });

  const out = await page.evaluate(async (id) => {
    const { manifest } = window.flyhero.getSong();
    const pos = window.flyhero.getAssets().brain.retina_display_pos;
    const bytes = new Uint8Array(await (await fetch(`data/songs/${id}/retina.bin`)).arrayBuffer());
    const channels = manifest.retina_channels;
    const enc = window.flyhero.makeRetinaEncoder(pos);

    // Activity frames spread across the song, skipping the first (the EMA
    // is still seeded there) and the last (partial pooling window).
    const frames = [40, 200, 800, 1500, 2400].filter((k) => k < manifest.activity_frames - 1);
    const score = (shift) => {
      let sumAbs = 0, maxAbs = 0, n = 0, peak = 0, saturated = 0;
      for (const k of frames) {
        // Mean of the three 60 Hz frames the recorder pooled, then |.|.
        const acc = new Float64Array(channels);
        for (let j = 0; j < 3; j++) {
          const f = 3 * k + j;
          enc.seekTo(manifest.notes, f, (g) => (g + shift) / 60);
          for (let i = 0; i < channels; i++) acc[i] += enc.current[i] / 3;
        }
        for (let i = 0; i < channels; i++) {
          const byte = bytes[k * channels + i];
          const want = (byte / 255) * manifest.retina_scale;
          const got = Math.abs(acc[i]);
          const err = Math.abs(got - want);
          // retina_scale is the recording's p99.5, so export_web.py clips the
          // tail: a 255 byte means ">= full scale" and cannot be compared.
          if (byte === 255) { saturated++; continue; }
          sumAbs += err; maxAbs = Math.max(maxAbs, err); n++;
          peak = Math.max(peak, want, got);
        }
      }
      return { mean: sumAbs / n, max: maxAbs, n, peak, saturated };
    };
    return { scale: manifest.retina_scale, frames, shifted: score(1), unshifted: score(0) };
  }, song.song_id);

  // One uint8 step of retina.bin is scale/255; the browser also blurs in
  // float64 where the recorder used float32, so exact equality is not the
  // bar -- reproducing the recording to inside its own quantization is.
  const step = out.scale / 255;
  const { shifted, unshifted, scale } = out;
  console.log(`  ${song.title}: ${shifted.n} samples over ${out.frames.length} frames ` +
              `(${shifted.saturated} clipped at full scale, excluded), ` +
              `full scale ${scale.toFixed(4)}, one uint8 step ${step.toFixed(5)}`);
  console.log(`     mean |error| ${shifted.mean.toFixed(6)} (${(shifted.mean / step).toFixed(3)} steps), ` +
              `max ${shifted.max.toFixed(6)} (${(shifted.max / step).toFixed(2)} steps), ` +
              `peak current seen ${shifted.peak.toFixed(4)}`);
  check(shifted.mean < 0.5 * step, `${song.title}: mean error under half a uint8 step`);
  check(shifted.max < 2 * step, `${song.title}: worst channel within 2 uint8 steps`);
  check(shifted.mean * 4 < unshifted.mean,
        `${song.title}: the recorder's (f+1)/60 alignment beats f/60 ` +
        `(${shifted.mean.toFixed(6)} vs ${unshifted.mean.toFixed(6)})`);
}

// The panel itself must follow the master clock, and be live before the
// heavy files land (it only needs the manifest).
await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
await page.waitForFunction((id) => window.flyhero.getSong()?.manifest.song_id === id,
                           songs[0].song_id, { timeout: 60000 });
// Wait until the page is quiescent: the brain panel's 11-20 MB recording
// download stalls animation frames, and a canvas read during that catches a
// drawing from before the seek.
await page.waitForFunction(() => window.flyhero.getBrain()?.stats.active > 0, { timeout: 60000 });
await page.waitForTimeout(500);

// Times with notes actually on screen: a quiet stretch renders an empty
// highway, where every current is exactly 0 and the panel cannot change.
const noteTimes = await page.evaluate(() => {
  const n = window.flyhero.getSong().manifest.notes;
  return [n[Math.floor(n.length * 0.4)][0], n[Math.floor(n.length * 0.6)][0]];
});
// The panel's own currents, not its pixels: Chromium starts a 2D canvas in
// software and promotes it to GPU rendering after a few frames, which
// shifts arc antialiasing by ~0.7% -- a rasterizer difference, not a state
// one (verified: identical currents, identical canvas size, different
// pixels). Pixels are only asked the coarse question of whether anything
// is drawn.
const at = async (t) => {
  await page.evaluate((s) => window.flyhero.clock.seek(s), t);
  await page.waitForTimeout(260);
  return page.evaluate(() => {
    const cur = Array.from(window.flyhero.getRetina().currents);
    const c = document.getElementById("eyes-canvas");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let lit = 0, sum = 0;
    for (let i = 0; i < d.length; i += 4) { sum += d[i] + d[i + 1] + d[i + 2]; if (d[i] > 200) lit++; }
    return { cur, lit, sum, nonzero: cur.filter((x) => x !== 0).length };
  });
};
const a = await at(noteTimes[0]);
const b = await at(noteTimes[1]);
const a2 = await at(noteTimes[0]);
const same = (x, y) => x.cur.every((v, i) => v === y.cur[i]);
check(!same(a, b), `the eyes change with the playhead (${a.nonzero} vs ${b.nonzero} live receptors)`);
check(same(a, a2), "scrubbing back to the same time restores identical currents");
check(a.nonzero > 0 && a.lit > 0, `notes on screen light photoreceptors (${a.lit} lit pixels)`);

// A quiet stretch: every current is exactly 0, and the eye must still be
// visible rather than an empty panel.
const quiet = await at(1.0);
check(quiet.nonzero === 0, "a blank highway leaves every photoreceptor at rest");
check(quiet.sum > 0, "the eye outline and lattice are drawn with nothing on screen");

await browser.close();
if (fail.length) { console.error(`\n${fail.length} check(s) failed`); process.exit(1); }
console.log("\nall retina checks passed");
