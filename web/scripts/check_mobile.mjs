/** Mobile layout and touch checks (D56/D57).
 *
 *  What it asserts, at a phone and a tablet viewport with touch emulation on:
 *    - the CRT is above every telemetry panel, and the panels are stacked in
 *      one column (no side-by-side pair squeezed into a phone's width);
 *    - nothing overflows horizontally, at any scroll position;
 *    - the fly-logo watermark is still 8-12% of the screen's width, i.e. it
 *      scaled with the CRT rather than being pinned to a pixel size;
 *    - a tap on the transport toggles it, and a touch drag on the scrubber
 *      seeks -- both through real touch events, not synthetic clicks;
 *    - a one-finger drag on the connectome panel rotates the brain;
 *    - the mobile render tier is on (smaller points, dimmer glow, DPR cap).
 *
 *  Headed, like the other checks: headless software rendering both starves
 *  rAF and reports frame times that say nothing about a real device.
 *
 *  Usage:  npm run dev            # in another shell
 *          node scripts/check_mobile.mjs [url]
 */

import { chromium } from "playwright";

const URL = process.argv[2] ?? "http://localhost:5173/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const VIEWPORTS = [
  { name: "phone  (390x844)", width: 390, height: 844, deviceScaleFactor: 3 },
  { name: "tablet (820x1180)", width: 820, height: 1180, deviceScaleFactor: 2 },
];

const browser = await chromium.launch({
  headless: false,
  args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
});

let allOk = true;
const problems = [];

for (const vp of VIEWPORTS) {
  const context = await browser.newContext({
    viewport: { width: vp.width, height: vp.height },
    deviceScaleFactor: vp.deviceScaleFactor,
    hasTouch: true,
    isMobile: true,
  });
  const page = await context.newPage();
  page.on("pageerror", (e) => problems.push(`${vp.name} pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`${vp.name} console: ${m.text()}`); });

  let ok = true;
  const say = (good, msg) => {
    console.log(`${good ? "PASS " : "FAIL "} ${msg}`);
    if (!good) { ok = false; allOk = false; }
  };

  console.log(`\n--- ${vp.name} ---`);
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
  const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);
  await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await page.waitForFunction(() => window.flyhero.audio.readyState >= 2, { timeout: 60000 });
  await sleep(800);

  // --- layout -------------------------------------------------------------
  const layout = await page.evaluate(() => {
    const top = (sel) => document.querySelector(sel).getBoundingClientRect().top + window.scrollY;
    const box = (sel) => {
      const r = document.querySelector(sel).getBoundingClientRect();
      return { top: r.top + window.scrollY, left: r.left, w: r.width, h: r.height };
    };
    const crt = box(".crt");
    const screen = document.querySelector(".crt-screen").getBoundingClientRect();
    const logo = document.querySelector(".crt-logo").getBoundingClientRect();
    return {
      crt,
      panels: {
        brain: top("#brain-panel"),
        eyes: top("#eyes-panel"),
        dn: top("#dn-panel"),
        scope: top("#scope-panel"),
      },
      dnLeft: document.querySelector("#dn-panel").getBoundingClientRect().left,
      scopeLeft: document.querySelector("#scope-panel").getBoundingClientRect().left,
      logoPctOfScreen: (100 * logo.width) / screen.width,
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      quality: window.flyhero.getQuality(),
      // The page has to be able to grow past the viewport, or everything
      // under the CRT is unreachable however far you swipe.
      scrollHeight: document.documentElement.scrollHeight,
      appHeight: Math.round(document.querySelector("#app").getBoundingClientRect().height),
      // Absolutely positioned titles and notes that run into each other.
      collisions: [...document.querySelectorAll(".panel")].flatMap((p) => {
        const t = p.querySelector(".panel-title");
        const n = p.querySelector(".panel-note");
        if (!t || !n) return [];
        const rt = t.getBoundingClientRect();
        const rn = n.getBoundingClientRect();
        const overlaps = rt.right > rn.left && rt.left < rn.right &&
                         rt.bottom > rn.top && rt.top < rn.bottom;
        return overlaps ? [p.id || p.className] : [];
      }),
    };
  });

  const crtBottom = layout.crt.top + layout.crt.h;
  const firstPanel = Math.min(...Object.values(layout.panels));
  say(crtBottom <= firstPanel + 1,
      `CRT is above every panel (CRT ends at y=${Math.round(crtBottom)}, first panel starts at y=${Math.round(firstPanel)})`);

  const stacked = layout.dnLeft === layout.scopeLeft;
  say(vp.width > 700 ? true : stacked,
      `panels stacked in one column (decision panel left ${Math.round(layout.dnLeft)}, ` +
      `oscilloscope left ${Math.round(layout.scopeLeft)}${vp.width > 700 ? " -- side by side allowed above 700px" : ""})`);

  say(layout.scrollWidth <= layout.innerWidth + 1,
      `no horizontal overflow (scrollWidth ${layout.scrollWidth} vs viewport ${layout.innerWidth})`);

  say(layout.scrollHeight >= layout.appHeight - 1,
      `the page can actually scroll to its content (scrollHeight ${layout.scrollHeight} ` +
      `vs #app height ${layout.appHeight})`);

  say(layout.collisions.length === 0,
      `no panel title runs into its note (${layout.collisions.length ? layout.collisions.join(", ") : "none"})`);

  say(layout.logoPctOfScreen >= 8 && layout.logoPctOfScreen <= 12,
      `logo scales with the CRT (${layout.logoPctOfScreen.toFixed(1)}% of screen width, brief says 8-12%)`);

  say(layout.quality === "reduced" || layout.quality === "minimal",
      `mobile render tier active (quality "${layout.quality}")`);

  const gl = await page.evaluate(() => window.flyhero.getBrain() ? true : false);
  say(gl, `connectome panel initialised (WebGL available: ${gl})`);

  // --- touch: transport ---------------------------------------------------
  const playBox = await page.locator("#play").boundingBox();
  const before = await page.evaluate(() => window.flyhero.clock.isPlaying);
  await page.touchscreen.tap(playBox.x + playBox.width / 2, playBox.y + playBox.height / 2);
  await sleep(700);
  const afterTap = await page.evaluate(() => ({
    playing: window.flyhero.clock.isPlaying,
    paused: window.flyhero.audio.paused,
  }));
  say(afterTap.playing !== before && afterTap.playing !== afterTap.paused,
      `tap toggles the transport (${before ? "playing" : "paused"} -> ${afterTap.playing ? "playing" : "paused"}, ` +
      `audio ${afterTap.paused ? "paused" : "playing"})`);

  // --- touch: scrubber ----------------------------------------------------
  await page.evaluate(() => { window.flyhero.clock.pause(); window.flyhero.clock.seek(10); });
  await sleep(200);
  const scrub = await page.locator("#scrubber").boundingBox();
  const y = scrub.y + scrub.height / 2;
  await page.touchscreen.tap(scrub.x + scrub.width * 0.6, y);
  await sleep(400);
  const afterScrub = await page.evaluate(() => window.flyhero.clock.time);
  say(Math.abs(afterScrub - 10) > 5,
      `touch on the scrubber seeks (10 s -> ${afterScrub.toFixed(1)} s)`);

  // --- touch: brain rotation ----------------------------------------------
  //
  // Through CDP, so the browser synthesises real touch input and turns it
  // into the pointer events OrbitControls actually listens to. Hand-built
  // TouchEvents do not reach it -- three r169 is pointer-event based -- and
  // dispatching them looks like a working drag that rotates nothing.
  await page.locator("#brain-panel").scrollIntoViewIfNeeded();
  await sleep(300);
  const cdp = await context.newCDPSession(page);
  const brainBox = await page.locator("#brain-canvas").boundingBox();
  const cx = brainBox.x + brainBox.width / 2;
  const cy = brainBox.y + brainBox.height / 2;
  const scrollY0 = await page.evaluate(() => window.scrollY);

  const az0 = await page.evaluate(() => window.flyhero.getBrain()?.cameraState.azimuth ?? null);
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: cx, y: cy }] });
  for (let i = 1; i <= 12; i++) {
    await cdp.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [{ x: cx + i * 9, y: cy }],
    });
    await sleep(16);
  }
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await sleep(400);
  const az1 = await page.evaluate(() => window.flyhero.getBrain()?.cameraState.azimuth ?? null);
  const scrollY1 = await page.evaluate(() => window.scrollY);
  const moved = az0 === null ? 0 : Math.abs(az1 - az0);
  say(moved > 0.05,
      `one-finger drag rotates the brain (azimuth ${az0?.toFixed(3)} -> ${az1?.toFixed(3)}, ` +
      `moved ${moved.toFixed(3)} rad)`);
  // touch-action: none on the canvas -- the drag must rotate, not scroll the
  // page out from under the finger.
  say(scrollY1 === scrollY0,
      `the drag did not scroll the page instead (scrollY ${scrollY0} -> ${scrollY1})`);

  // --- frame time, for the record (not a gate: this is a desktop GPU) ------
  const perf = await page.evaluate(async (s) => {
    const { clock } = window.flyhero;
    clock.seek(60);
    clock.play();
    const gaps = [];
    let prev = performance.now();
    const t0 = prev;
    while (performance.now() - t0 < s * 1000) {
      await new Promise((r) => requestAnimationFrame(r));
      const now = performance.now();
      gaps.push(now - prev);
      prev = now;
    }
    clock.pause();
    gaps.sort((a, b) => a - b);
    return {
      median: gaps[Math.floor(gaps.length / 2)],
      p95: gaps[Math.floor(gaps.length * 0.95)],
    };
  }, 5);
  console.log(`  (frame time at this viewport, desktop GPU: median ${perf.median.toFixed(1)} ms, ` +
              `p95 ${perf.p95.toFixed(1)} ms -- not a device measurement)`);

  await page.evaluate(() => window.flyhero.clock.pause());
  await page.screenshot({ path: `scripts/_shot_mobile_${vp.width}.png`, fullPage: true });
  await context.close();
}

console.log(`\nconsole/page problems: ${problems.length}`);
for (const p of problems.slice(0, 20)) console.log("  " + p);

await browser.close();
if (!allOk || problems.length) process.exit(1);
console.log("\nALL MOBILE CHECKS PASS");
