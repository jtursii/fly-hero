/** Desktop layout check (Session 6).
 *
 *  Session 5 (D57) enlarged and tilted the CRT. A 3D-transformed element's
 *  *painted* box is not its layout box, and the tilted tube ended up drawn
 *  over the transport row below it: the play button, the scrubber and the
 *  song picker were all still in the DOM, still "visible", and completely
 *  unclickable, because the CRT's transformed box sat on top of them. That
 *  is the whole of the reported "the pause button does nothing" -- the
 *  button was never reached by the click at all.
 *
 *  So this check does not ask whether a control is on screen. It asks the
 *  only question that matters for a control: **if a real mouse clicked the
 *  middle of it, would the click land on it?** -- which is
 *  `document.elementFromPoint` at its centre, the same hit test the browser
 *  runs for a click. A decorative layer painted over a button fails here
 *  even though every rectangle looks right.
 *
 *  Asserted at each desktop viewport:
 *    - the page fits the viewport with no scrolling in either axis;
 *    - every control (play/pause, the three speeds, the scrubber, the volume
 *      slider and its mute button, every song button, About) is fully inside
 *      the viewport;
 *    - `elementFromPoint` at each control's centre is that control or a
 *      child of it -- nothing is painted over it;
 *    - the CRT does not overlap the transport or the stat strip;
 *    - pause survives repeated real mid-song clicks, and spacebar toggles;
 *    - the volume slider reaches the audio element, mute round-trips to the
 *      same level, and the level survives a reload.
 *
 *  Headed, like the other checks: a headless viewport is not the layout a
 *  real Chrome window has, and this file is about a real Chrome window.
 *
 *  Usage:  npm run dev            # in another shell
 *          node scripts/check_layout.mjs [url]
 */

import { chromium } from "playwright";

const URL = process.argv[2] ?? "http://localhost:5173/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** The desktop sizes the fix is required to hold at: three MacBook widths
 *  and 1080p. These are CSS px -- the browser window is larger. */
const VIEWPORTS = [
  { width: 1440, height: 900 },
  { width: 1512, height: 982 },
  { width: 1728, height: 1117 },
  { width: 1920, height: 1080 },
];

const browser = await chromium.launch({
  headless: false,
  args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
});

let allOk = true;
const problems = [];

for (const vp of VIEWPORTS) {
  const name = `${vp.width}x${vp.height}`;
  const context = await browser.newContext({ viewport: vp, deviceScaleFactor: 2 });
  const page = await context.newPage();
  page.on("pageerror", (e) => problems.push(`${name} pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`${name} console: ${m.text()}`); });

  let ok = true;
  const say = (good, msg) => {
    console.log(`${good ? "PASS " : "FAIL "} ${msg}`);
    if (!good) { ok = false; allOk = false; }
  };

  console.log(`\n--- ${name} ---`);
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
  // A song has to be loaded or the transport is disabled and the song
  // buttons are the only live controls -- that is not the layout we ship.
  const songs = await page.evaluate(() => window.flyhero.getAssets().brain.songs);
  await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await sleep(600);

  // --- no page scroll -----------------------------------------------------
  const scroll = await page.evaluate(() => ({
    h: document.documentElement.scrollHeight,
    w: document.documentElement.scrollWidth,
    ih: window.innerHeight,
    iw: window.innerWidth,
  }));
  // 1 px of slack for sub-pixel layout rounding; anything real is far larger.
  say(scroll.h <= scroll.ih + 1,
      `page fits vertically (scrollHeight ${scroll.h} <= viewport ${scroll.ih})`);
  say(scroll.w <= scroll.iw + 1,
      `page fits horizontally (scrollWidth ${scroll.w} <= viewport ${scroll.iw})`);

  // --- every control: on screen, and actually hittable ---------------------
  const controls = await page.evaluate(() => {
    const out = [];
    const add = (label, node) => {
      const r = node.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const hit = document.elementFromPoint(cx, cy);
      // Descendants count: a click on a song button's title span is a click
      // on the song button. Anything else at that point is painted over it.
      const reached = hit !== null && node.contains(hit);
      const describe = (e) => {
        if (!e) return "nothing (outside the viewport)";
        const cls = e.className && typeof e.className === "string"
          ? "." + e.className.trim().split(/\s+/).join(".") : "";
        return `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""}${cls}`;
      };
      out.push({
        label,
        rect: { top: r.top, left: r.left, bottom: r.bottom, right: r.right, w: r.width, h: r.height },
        centre: { x: cx, y: cy },
        reached,
        hit: describe(hit),
        inView: r.top >= 0 && r.left >= 0
          && r.bottom <= window.innerHeight && r.right <= window.innerWidth
          && r.width > 0 && r.height > 0,
      });
    };
    add("play/pause", document.getElementById("play"));
    document.querySelectorAll(".rate-btn").forEach((b, i) => add(`speed ${b.textContent} (#${i})`, b));
    add("scrubber", document.getElementById("scrubber"));
    add("volume", document.getElementById("volume"));
    add("mute", document.getElementById("mute"));
    document.querySelectorAll(".song-btn").forEach((b, i) =>
      add(`song "${b.querySelector(".t").textContent}" (#${i})`, b));
    add("About", document.getElementById("about-open"));
    return out;
  });

  for (const c of controls) {
    const r = c.rect;
    say(c.inView,
        `${c.label} fully in the viewport ` +
        `(${r.left.toFixed(0)},${r.top.toFixed(0)} ${r.w.toFixed(0)}x${r.h.toFixed(0)})`);
    say(c.reached,
        `${c.label} is hittable at its centre ` +
        `(${c.centre.x.toFixed(0)},${c.centre.y.toFixed(0)} -> ${c.hit})`);
  }

  // --- the CRT keeps to its own space -------------------------------------
  // Painted box, not layout box: `getBoundingClientRect` on a 3D-transformed
  // element returns the transformed rectangle, which is exactly the one that
  // was covering the transport.
  const overlap = await page.evaluate(() => {
    const r = (sel) => document.querySelector(sel).getBoundingClientRect();
    const crt = r(".crt");
    const hits = (o) => !(crt.right <= o.left || crt.left >= o.right
                       || crt.bottom <= o.top || crt.top >= o.bottom);
    return {
      crt: { top: crt.top, bottom: crt.bottom, h: crt.height, w: crt.width },
      transport: hits(r(".transport")),
      stats: hits(r(".stats")),
      transportTop: r(".transport").top,
    };
  });
  say(!overlap.transport,
      `CRT does not overlap the transport (CRT bottom ${overlap.crt.bottom.toFixed(1)}, ` +
      `transport top ${overlap.transportTop.toFixed(1)})`);
  say(!overlap.stats, `CRT does not overlap the stat strip`);
  console.log(`  (CRT painted box ${overlap.crt.w.toFixed(0)}x${overlap.crt.h.toFixed(0)})`);

  // --- pause, driven the way a person drives it ---------------------------
  // Real mouse clicks at the button's centre, never `clock.toggle()`: the
  // reported bug was entirely in the path between the mouse and the button,
  // and a test that calls the toggle function skips exactly the broken part.
  await page.waitForFunction(() => window.flyhero.audio.readyState >= 2, { timeout: 60000 });

  /** One real click at the centre of #play, with no auto-scroll or
   *  actionability retry that could paper over an obscured button. */
  const clickPlay = async () => {
    const c = await page.evaluate(() => {
      const r = document.getElementById("play").getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
    await page.mouse.click(c.x, c.y);
    await sleep(250);
    return page.evaluate(() => {
      const d = window.flyhero.clock.debug();
      return { want: d.want, playing: d.playing, audioPaused: d.audioPaused, t: d.clockTime };
    });
  };

  // Start mid-song, where a pause has real audio to stop.
  await page.evaluate(() => window.flyhero.clock.seek(30));
  await sleep(200);

  const states = [];
  for (let i = 0; i < 6; i++) {
    states.push(await clickPlay());
    // Let the song actually run between presses: a pause that only looks
    // right the instant after the click is not a pause.
    await sleep(700);
  }
  // Clicks alternate play/pause/play/... starting from stopped.
  const wantSeq = states.map((s) => s.want);
  say(wantSeq.every((w, i) => w === (i % 2 === 0)),
      `6 real clicks alternate play/pause (want ${wantSeq.map((w) => (w ? "1" : "0")).join("")})`);
  say(states.every((s) => s.playing === s.want),
      `the clock follows the button every time`);
  say(states.every((s) => s.audioPaused === !s.want),
      `the audio element follows the button every time ` +
      `(audioPaused ${states.map((s) => (s.audioPaused ? "1" : "0")).join("")})`);

  // A pause has to actually hold. The clock must not creep while stopped.
  const pausedAt = await page.evaluate(() => window.flyhero.clock.debug().clockTime);
  await sleep(1200);
  const stillAt = await page.evaluate(() => {
    const d = window.flyhero.clock.debug();
    return { t: d.clockTime, audioPaused: d.audioPaused, playing: d.playing };
  });
  say(!stillAt.playing && Math.abs(stillAt.t - pausedAt) < 0.01,
      `the playhead holds while paused (${pausedAt.toFixed(3)} -> ${stillAt.t.toFixed(3)} s ` +
      `over 1.2 s)`);
  say(stillAt.audioPaused, `the audio stays stopped while paused`);

  // --- volume --------------------------------------------------------------
  // Dragged, not assigned: the slider has to reach the audio element.
  const volBox = await page.evaluate(() => {
    const r = document.getElementById("volume").getBoundingClientRect();
    return { x0: r.left + r.width * 0.9, x1: r.left + r.width * 0.25, y: r.top + r.height / 2 };
  });
  await page.mouse.move(volBox.x0, volBox.y);
  await page.mouse.down();
  await page.mouse.move(volBox.x1, volBox.y, { steps: 8 });
  await page.mouse.up();
  await sleep(200);
  const vol = await page.evaluate(() => ({
    slider: parseFloat(document.getElementById("volume").value),
    audio: window.flyhero.audio.volume,
    muted: window.flyhero.audio.muted,
  }));
  say(vol.audio < 0.5 && Math.abs(vol.audio - vol.slider) < 1e-6 && !vol.muted,
      `dragging the volume slider sets the audio element ` +
      `(slider ${vol.slider.toFixed(2)}, audio.volume ${vol.audio.toFixed(2)})`);

  // Mute is a toggle that must come back to the same level, not to zero.
  await page.click("#mute");
  await sleep(150);
  const muted = await page.evaluate(() => ({ m: window.flyhero.audio.muted, v: window.flyhero.audio.volume }));
  await page.click("#mute");
  await sleep(150);
  const unmuted = await page.evaluate(() => ({ m: window.flyhero.audio.muted, v: window.flyhero.audio.volume }));
  say(muted.m && !unmuted.m && Math.abs(unmuted.v - vol.slider) < 1e-6,
      `mute toggles and restores the level (${vol.slider.toFixed(2)} -> muted -> ` +
      `${unmuted.v.toFixed(2)})`);

  // It has to survive a reload, which is the only reason to store it at all.
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForFunction(() => document.querySelectorAll(".song-btn").length > 0, { timeout: 30000 });
  const restored = await page.evaluate(() => window.flyhero.audio.volume);
  say(Math.abs(restored - vol.slider) < 1e-6,
      `the level survives a reload (${restored.toFixed(2)})`);
  // Back to a loaded song for the keyboard section below.
  await page.evaluate((id) => window.flyhero.selectSong(id), songs[0].song_id);
  await page.waitForFunction(() => window.flyhero.getSong() !== null, { timeout: 30000 });
  await page.click("#play");
  await sleep(250);

  // --- spacebar ------------------------------------------------------------
  // Pressed straight after a click, so #play still has focus: Space on a
  // focused button is also a click, and the two used to cancel out.
  const beforeSpace = await page.evaluate(() => ({
    want: window.flyhero.clock.debug().want,
    focus: document.activeElement?.id ?? "",
  }));
  await page.keyboard.press("Space");
  await sleep(250);
  const afterSpace = await page.evaluate(() => window.flyhero.clock.debug().want);
  say(afterSpace !== beforeSpace.want,
      `spacebar toggles once with #play focused (focus "${beforeSpace.focus}", ` +
      `want ${beforeSpace.want} -> ${afterSpace})`);

  // And from the scrubber, where focus lands after a scrub.
  await page.focus("#scrubber");
  const beforeSpace2 = await page.evaluate(() => window.flyhero.clock.debug().want);
  await page.keyboard.press("Space");
  await sleep(250);
  const afterSpace2 = await page.evaluate(() => window.flyhero.clock.debug().want);
  say(afterSpace2 !== beforeSpace2,
      `spacebar toggles with the scrubber focused (want ${beforeSpace2} -> ${afterSpace2})`);
  const scrolled = await page.evaluate(() => window.scrollY);
  say(scrolled === 0, `spacebar did not scroll the page (scrollY ${scrolled})`);

  await page.evaluate(() => window.flyhero.clock.pause());

  if (!ok) await page.screenshot({ path: `scripts/_shot_layout_fail_${vp.width}.png` });
  await context.close();
}

console.log(`\nconsole/page problems: ${problems.length}`);
for (const p of problems.slice(0, 20)) console.log("  " + p);

await browser.close();
if (!allOk) process.exit(1);
console.log("\nALL LAYOUT CHECKS PASS");
