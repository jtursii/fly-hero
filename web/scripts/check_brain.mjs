/** Asserts the connectome panel's interactions: slow auto-rotate when idle,
 *  paused while dragging and back a few seconds after release, a zoom that
 *  clamps at both ends, and activity that follows the master clock.
 *
 *  Usage:  npm run dev  &&  node scripts/check_brain.mjs [url]
 */
import { chromium } from "playwright";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const state = (page) => page.evaluate(() => window.flyhero.getBrain().cameraState);
const fail = [];
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fail.push(msg); };

// Headed, on the real GPU: software rendering 139k points + 2.1k lines
// starves the compositor and rAF stops firing, which is an artefact of
// swiftshader rather than anything the panel does.
const browser = await chromium.launch({ headless: false, args: ["--mute-audio"] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1050 } });
await page.goto(process.argv[2] ?? "http://localhost:5174/", { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);
await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
await page.waitForFunction(() => window.flyhero.getBrain().stats.active > 0, { timeout: 60000 });

// Auto-rotate when idle.
const a0 = await state(page);
await sleep(1200);
const a1 = await state(page);
check(Math.abs(a1.azimuth - a0.azimuth) > 0.005, `idle auto-rotate moved the camera (${(a1.azimuth - a0.azimuth).toFixed(3)} rad)`);

// Dragging rotates it and pauses the spin.
const box = await page.locator("#brain-canvas").boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2, { steps: 12 });
const dragging = await state(page);
check(!dragging.autoRotate, "auto-rotate pauses while dragging");
check(Math.abs(dragging.azimuth - a1.azimuth) > 0.15, "dragging rotated the brain");
await page.mouse.up();
await sleep(1000);
check(!(await state(page)).autoRotate, "still paused 1 s after release");
await sleep(2600);
check((await state(page)).autoRotate, "auto-rotate resumes a few seconds after release");

// Zoom clamps at both ends.
for (let i = 0; i < 40; i++) await page.mouse.wheel(0, -120);
const near = await state(page);
for (let i = 0; i < 80; i++) await page.mouse.wheel(0, 120);
const far = await state(page);
check(near.distance >= 1.09 && near.distance <= 1.2, `zoom-in clamped at ${near.distance.toFixed(2)}`);
check(far.distance >= 3.1 && far.distance <= 3.25, `zoom-out clamped at ${far.distance.toFixed(2)}`);

// Activity follows the master clock, including a backwards scrub.
const activeAt = async (t) => {
  await page.evaluate((s) => window.flyhero.clock.seek(s), t);
  await sleep(250);
  return page.evaluate(() => window.flyhero.getBrain().stats);
};
const times = [20, 40, 60, 80, 100, 120];
const seen = [];
for (const t of times) seen.push(await activeAt(t));
const s1 = seen[1];
const s1b = await activeAt(40);
// The active count drifts slowly (1,774-2,075 over a song), so one pair of
// times can legitimately tie; six cannot.
check(new Set(seen.map((s) => s.active)).size >= 3,
      `activity tracks the playhead (active: ${seen.map((s) => s.active).join(", ")})`);
check(new Set(seen.map((s) => s.meanActivation)).size === times.length,
      "mean activation differs at every sampled time");
check(s1.active === s1b.active && s1.meanActivation === s1b.meanActivation,
      "scrubbing back to the same time restores the same activity");
check(s1.meanActivation > 0 && s1.meanActivation < 1, `mean activation in range (${s1.meanActivation.toFixed(4)})`);

// Frame time, with the clock running (PLAN task 8's 60 fps target).
await page.evaluate(() => { window.flyhero.clock.seek(100); window.flyhero.clock.toggle(); });
await sleep(2000);
const d = await page.evaluate(() => new Promise((res) => {
  const ts = []; let n = 0;
  const tick = (t) => { ts.push(t); if (++n < 240) requestAnimationFrame(tick); else {
    const gaps = ts.slice(1).map((t, i) => t - ts[i]).sort((a, b) => a - b);
    res({ median: gaps[gaps.length >> 1], p95: gaps[Math.floor(gaps.length * 0.95)] });
  }};
  requestAnimationFrame(tick);
}));
console.log(`     frame time: median ${d.median.toFixed(1)} ms, p95 ${d.p95.toFixed(1)} ms ` +
            `(vsync-capped; 60 fps needs <= 16.7)`);
check(d.p95 <= 16.7, "p95 frame time is inside the 60 fps budget");

await browser.close();
if (fail.length) { console.error(`\n${fail.length} check(s) failed`); process.exit(1); }
console.log("\nall brain-panel checks passed");
