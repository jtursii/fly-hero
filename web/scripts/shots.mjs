/** Screenshots the site at hand-picked dense moments, for eyeballing the
 *  highway. Not a test -- `check_sync.mjs` is the one that asserts.
 *
 *  Each moment is a few milliseconds after a hit, so the burst, the lit
 *  frets and the strikeline flash are all in frame.
 *
 *  Usage:  npm run dev  &&  node scripts/shots.mjs
 */

import { chromium } from "playwright";

const URL = process.argv[2] ?? "http://localhost:5173/";
const SHOTS = [
  { song: 0, t: 109.03, name: "highway_dense" }, // What I've Done, 7 notes in view
  { song: 1, t: 289.9, name: "highway_sustains" }, // Nothing Else Matters, 3 sustains
  { song: 2, t: 247.25, name: "highway_yellow" },
  // A miss and an overstrum with no hit within 0.6 s, so each is the only
  // thing happening at the strikeline.
  { song: 1, t: 95.82, name: "miss" },
  { song: 1, t: 296.3, name: "overstrum" },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  args: ["--mute-audio", "--use-gl=angle", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);

for (const shot of SHOTS) {
  await page.evaluate((id) => window.flyhero.selectSong(id), songs[shot.song].song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await page.evaluate((t) => window.flyhero.clock.seek(t), shot.t);
  await sleep(700);
  await page.screenshot({ path: `scripts/_shot_${shot.name}.png` });
  await page.screenshot({
    path: `scripts/_shot_${shot.name}_crt.png`,
    clip: { x: 852, y: 82, width: 812, height: 800 },
  });
  console.log(`${shot.name}: ${songs[shot.song].title} @ ${shot.t}s`);
}
await browser.close();
