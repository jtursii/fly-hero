/** The note highway, drawn on a 2D canvas inside the CRT.
 *
 *  Everything here is a pure function of the master playhead: given a song
 *  time it draws the frame, with no animation state carried between frames.
 *  That is what makes scrubbing exact -- a burst, a flash or a dimming note
 *  is derived from the event's own time and the playhead, so jumping the
 *  clock backwards redraws the past correctly instead of replaying it.
 *
 *  It never decides anything about the game. Hits, misses and overstrums come
 *  from `events.json`, which `rules.py` produced at record time (invariant 4),
 *  and the lit frets come straight out of `actions.bin`.
 */

import type { Manifest, NoteState } from "./types.ts";

/** Seconds of chart between the strikeline and the horizon. */
const LEAD_S = 1.6;
/** Foreshortening: a note at the horizon is drawn 1/(1+PERSP) of near size. */
const PERSP = 2.6;
/** How far past the strikeline (in LEAD_S units) a missed note keeps going
 *  before it is culled -- 0.30 s, matching MISS_FADE_S so it fades out just
 *  as it reaches the bottom edge. */
const PAST_U = -0.1875;
/** Shape of the run-out below the strikeline: leaves the line at the speed
 *  the projective mapping gave it, then decelerates toward the bottom. */
const PAST_EASE = 1.8;

const BURST_S = 0.42;
const HIT_FLASH_S = 0.14;
const OVERSTRUM_FLASH_S = 0.2;
const MISS_FADE_S = 0.3;

export const LANE_COLORS = ["#21d95a", "#ff3b30", "#ffd60a", "#2f9dff", "#ff8c1a"];
const LANE_DIM = ["#0f5c28", "#6b1a16", "#6e5c05", "#14456e", "#6e3c0b"];

/** Deterministic [0,1) from an integer seed -- keeps bursts identical across
 *  redraws of the same instant, so scrubbing does not reshuffle them. */
function hash01(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export class Highway {
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;

  private manifest: Manifest | null = null;
  private notes: [number, number, number][] = [];
  private states: NoteState[] = [];
  private actions = new Uint8Array(0);
  private overstrums = new Float64Array(0);
  /** Hit times, ascending, for the strikeline flash lookup. */
  private hitTimes = new Float64Array(0);

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.resize();
  }

  setSong(
    manifest: Manifest,
    states: NoteState[],
    actions: Uint8Array,
    overstrums: Float64Array,
  ): void {
    this.manifest = manifest;
    this.notes = manifest.notes;
    this.states = states;
    this.actions = actions;
    this.overstrums = overstrums;
    const hits = states.filter((s) => s.hitAt >= 0).map((s) => s.hitAt);
    hits.sort((a, b) => a - b);
    this.hitTimes = Float64Array.from(hits);
  }

  clearSong(): void {
    this.manifest = null;
    this.notes = [];
    this.states = [];
    this.actions = new Uint8Array(0);
    this.overstrums = new Float64Array(0);
    this.hitTimes = new Float64Array(0);
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, Math.round(rect.width));
    this.h = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // --- geometry -----------------------------------------------------------

  private get horizonY(): number {
    return this.h * 0.08;
  }
  private get strikeY(): number {
    return this.h * 0.74;
  }
  private get laneW(): number {
    return this.w * 0.158;
  }

  /** Near-to-far size factor for a note `u` lead-times from the strikeline. */
  private scale(u: number): number {
    if (u >= 0) return 1 / (1 + PERSP * u);
    // Only a little wider below the line: the lane spread uses this too, and
    // at the projective rate lane 0 would leave the canvas entirely.
    return 1 + 0.18 * this.pastP(u);
  }

  /** Progress 0..1 through the run-out below the strikeline. */
  private pastP(u: number): number {
    return Math.min(-u, -PAST_U) / -PAST_U;
  }

  private yOf(u: number): number {
    if (u < 0) {
      // Continuing the projective mapping here would put the note below the
      // camera plane -- a note 0.1 s past the line lands off the bottom of
      // the canvas, which is why a miss used to look like the note simply
      // vanishing. The run-out gets its own eased mapping into the strip
      // between the strikeline and the bottom edge instead.
      const p = 1 - Math.pow(1 - this.pastP(u), PAST_EASE);
      return this.strikeY + (this.h - this.strikeY) * p;
    }
    const far = 1 / (1 + PERSP);
    const s = this.scale(u);
    return this.strikeY - (this.strikeY - this.horizonY) * ((1 - s) / (1 - far));
  }

  private xOf(lane: number, u: number): number {
    return this.w / 2 + (lane - 2) * this.laneW * this.scale(u);
  }

  // --- drawing ------------------------------------------------------------

  draw(t: number): void {
    const { ctx, w, h } = this;
    ctx.fillStyle = "#04060a";
    ctx.fillRect(0, 0, w, h);
    if (!this.manifest) {
      this.drawIdle();
      return;
    }

    const fps = this.manifest.fps;
    const frame = Math.min(Math.max(Math.round(t * fps), 0), this.actions.length - 1);
    const held = this.actions.length ? this.actions[frame] : 0;

    this.drawSurface();
    this.drawSustains(t, held);
    this.drawNotes(t);
    this.drawStrikeline(t);
    this.drawFrets(held);
    this.drawBursts(t);
    this.drawVignette();
  }

  private drawIdle(): void {
    const { ctx, w, h } = this;
    this.drawSurface();
    ctx.save();
    ctx.fillStyle = "rgba(180,220,255,0.35)";
    ctx.font = `600 ${Math.round(h * 0.035)}px ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.fillText("SELECT A SONG", w / 2, h * 0.45);
    ctx.restore();
    this.drawVignette();
  }

  /** The trapezoid, its lane dividers, and the glow along the horizon. */
  private drawSurface(): void {
    const { ctx, w } = this;
    const yTop = this.horizonY;
    const yBot = this.h;
    const uBot = PAST_U;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(this.xOf(-0.5, 1), yTop);
    ctx.lineTo(this.xOf(4.5, 1), yTop);
    ctx.lineTo(this.xOf(4.5, uBot), yBot);
    ctx.lineTo(this.xOf(-0.5, uBot), yBot);
    ctx.closePath();
    const g = ctx.createLinearGradient(0, yTop, 0, yBot);
    g.addColorStop(0, "#0a1426");
    g.addColorStop(0.55, "#070d18");
    g.addColorStop(1, "#05080f");
    ctx.fillStyle = g;
    ctx.fill();

    ctx.clip();
    for (let i = -1; i <= 5; i++) {
      const edge = i - 0.5;
      const outer = i < 0 || i > 4;
      ctx.beginPath();
      ctx.moveTo(this.xOf(edge, 1), yTop);
      ctx.lineTo(this.xOf(edge, uBot), yBot);
      ctx.strokeStyle = outer ? "rgba(110,190,255,0.30)" : "rgba(120,170,230,0.12)";
      ctx.lineWidth = outer ? 2 : 1;
      ctx.stroke();
    }
    ctx.restore();

    // Horizon haze: the highway fading into the back of the tube.
    const hg = ctx.createLinearGradient(0, yTop - this.h * 0.04, 0, yTop + this.h * 0.12);
    hg.addColorStop(0, "rgba(60,140,220,0.22)");
    hg.addColorStop(1, "rgba(60,140,220,0)");
    ctx.fillStyle = hg;
    ctx.fillRect(0, yTop - this.h * 0.04, w, this.h * 0.16);
  }

  /** Sustain tails. After the note is hit, the tail burns for as long as
   *  `actions.bin` says that fret is still down -- a readout of the recorded
   *  action, not a judgement about whether the sustain "counted". */
  private drawSustains(t: number, held: number): void {
    for (let i = 0; i < this.notes.length; i++) {
      const [time, mask, sustain] = this.notes[i];
      if (sustain <= 0) continue;
      const uStart = (time - t) / LEAD_S;
      const uEnd = (time + sustain - t) / LEAD_S;
      if (uEnd < PAST_U || uStart > 1.05) continue;
      const st = this.states[i];
      const hit = st.hitAt >= 0 && t >= st.hitAt;
      const dead = st.missAt >= 0 && t >= st.missAt;

      // Once the head is at the line the tail is being consumed: only the
      // part still above the strikeline is left.
      const uLo = Math.max(uStart, hit || dead ? 0 : PAST_U);
      const uHi = Math.min(uEnd, 1);
      if (uHi <= uLo) continue;

      for (let lane = 0; lane < 5; lane++) {
        if (!(mask & (1 << lane))) continue;
        const burning = hit && (held & (1 << lane)) !== 0;
        this.drawTail(lane, uLo, uHi, burning ? 1 : dead ? 0 : 0.45, t, i);
      }
    }
  }

  private drawTail(
    lane: number,
    uLo: number,
    uHi: number,
    heat: number,
    t: number,
    seed: number,
  ): void {
    const { ctx } = this;
    const halfLo = this.laneW * this.scale(uLo) * 0.16;
    const halfHi = this.laneW * this.scale(uHi) * 0.16;
    const xLo = this.xOf(lane, uLo);
    const xHi = this.xOf(lane, uHi);
    const yLo = this.yOf(uLo);
    const yHi = this.yOf(uHi);

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(xLo - halfLo, yLo);
    ctx.lineTo(xHi - halfHi, yHi);
    ctx.lineTo(xHi + halfHi, yHi);
    ctx.lineTo(xLo + halfLo, yLo);
    ctx.closePath();
    if (heat >= 1) {
      // Flicker is a function of the playhead, so it is the same on redraw.
      const flick = 0.75 + 0.25 * Math.sin(t * 47 + seed);
      const g = ctx.createLinearGradient(0, yLo, 0, yHi);
      g.addColorStop(0, `rgba(255,255,255,${0.95 * flick})`);
      g.addColorStop(0.35, LANE_COLORS[lane]);
      g.addColorStop(1, LANE_DIM[lane]);
      ctx.fillStyle = g;
      ctx.shadowColor = LANE_COLORS[lane];
      ctx.shadowBlur = 18 * flick;
    } else {
      ctx.fillStyle = heat > 0 ? LANE_DIM[lane] : "rgba(40,44,52,0.55)";
    }
    ctx.fill();
    ctx.restore();
  }

  private drawNotes(t: number): void {
    for (let i = 0; i < this.notes.length; i++) {
      const [time, mask] = this.notes[i];
      const st = this.states[i];
      // A hit note is gone the instant rules.py consumed it; the burst takes
      // over from there.
      if (st.hitAt >= 0 && t >= st.hitAt) continue;
      let u = (time - t) / LEAD_S;
      if (u > 1.02) continue;
      let alpha = 1;
      let missed = false;
      if (st.missAt >= 0 && t >= st.missAt) {
        const age = t - st.missAt;
        if (age > MISS_FADE_S) continue;
        missed = true;
        alpha = 1 - age / MISS_FADE_S;
      }
      if (u < PAST_U) continue;
      u = Math.max(u, PAST_U);
      for (let lane = 0; lane < 5; lane++) {
        if (mask & (1 << lane)) this.drawGem(lane, u, missed, alpha);
      }
    }
  }

  private drawGem(lane: number, u: number, missed: boolean, alpha: number): void {
    const { ctx } = this;
    const s = this.scale(u);
    const x = this.xOf(lane, u);
    const y = this.yOf(u);
    const rx = this.laneW * s * 0.40;
    const ry = rx * 0.46;

    ctx.save();
    ctx.globalAlpha = alpha;
    if (!missed) {
      const halo = ctx.createRadialGradient(x, y, 0, x, y, rx * 2.1);
      halo.addColorStop(0, LANE_COLORS[lane] + "70");
      halo.addColorStop(1, LANE_COLORS[lane] + "00");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.ellipse(x, y, rx * 2.1, ry * 2.4, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    if (missed) {
      // Grey, lit only by its own rim: it has to stay legible against the
      // near-black stretch below the strikeline, or the miss reads as the
      // note simply vanishing.
      const g = ctx.createLinearGradient(x, y - ry, x, y + ry);
      g.addColorStop(0, "#3a414f");
      g.addColorStop(1, "#171b23");
      ctx.fillStyle = g;
      ctx.fill();
      ctx.strokeStyle = LANE_DIM[lane];
      ctx.lineWidth = Math.max(1, rx * 0.16);
      ctx.stroke();
      ctx.strokeStyle = "rgba(170,182,200,0.5)";
    } else {
      const g = ctx.createLinearGradient(x, y - ry, x, y + ry);
      g.addColorStop(0, "#ffffff");
      g.addColorStop(0.35, LANE_COLORS[lane]);
      g.addColorStop(1, LANE_DIM[lane]);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
    }
    ctx.lineWidth = Math.max(1, rx * 0.11);
    ctx.stroke();
    ctx.restore();
  }

  /** How strongly an event list fires at `t`: 1 at the event, 0 a window later. */
  private flashAt(times: Float64Array, t: number, windowS: number): number {
    let best = 0;
    for (let i = times.length - 1; i >= 0; i--) {
      const age = t - times[i];
      if (age < 0) continue;
      if (age > windowS) break;
      best = Math.max(best, 1 - age / windowS);
    }
    return best;
  }

  private drawStrikeline(t: number): void {
    const { ctx } = this;
    const y = this.strikeY;
    const xL = this.xOf(-0.5, 0);
    const xR = this.xOf(4.5, 0);
    const hit = this.flashAt(this.hitTimes, t, HIT_FLASH_S);
    const over = this.flashAt(this.overstrums, t, OVERSTRUM_FLASH_S);

    ctx.save();
    if (over > 0) {
      // Overstrum: the fly strummed with nothing to strum. Red, brief.
      const g = ctx.createLinearGradient(0, y - 26, 0, y + 26);
      g.addColorStop(0, `rgba(255,40,40,0)`);
      g.addColorStop(0.5, `rgba(255,50,50,${0.5 * over})`);
      g.addColorStop(1, `rgba(255,40,40,0)`);
      ctx.fillStyle = g;
      ctx.fillRect(xL - 20, y - 26, xR - xL + 40, 52);
    }
    if (hit > 0) {
      const g = ctx.createLinearGradient(0, y - 18, 0, y + 18);
      g.addColorStop(0, "rgba(200,240,255,0)");
      g.addColorStop(0.5, `rgba(220,245,255,${0.55 * hit})`);
      g.addColorStop(1, "rgba(200,240,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(xL - 12, y - 18, xR - xL + 24, 36);
    }
    ctx.beginPath();
    ctx.moveTo(xL, y);
    ctx.lineTo(xR, y);
    ctx.lineWidth = 3 + 3 * Math.max(hit, over);
    ctx.strokeStyle = over > 0.05 ? `rgba(255,90,80,0.95)` : `rgba(190,230,255,${0.7 + 0.3 * hit})`;
    ctx.shadowColor = over > 0.05 ? "#ff3b30" : "#8fd4ff";
    ctx.shadowBlur = 12 + 22 * Math.max(hit, over);
    ctx.stroke();
    ctx.restore();
  }

  /** The five fret buttons, lit while `actions.bin` holds that fret. */
  private drawFrets(held: number): void {
    const { ctx } = this;
    const y = this.strikeY;
    const r = this.laneW * 0.34;
    for (let lane = 0; lane < 5; lane++) {
      const x = this.xOf(lane, 0);
      const lit = (held & (1 << lane)) !== 0;
      ctx.save();
      if (lit) {
        const halo = ctx.createRadialGradient(x, y, 0, x, y, r * 2.2);
        halo.addColorStop(0, LANE_COLORS[lane] + "5e");
        halo.addColorStop(1, LANE_COLORS[lane] + "00");
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.ellipse(x, y, r * 2.2, r * 1.3, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.ellipse(x, y, r * (lit ? 1.06 : 1), r * (lit ? 0.5 : 0.46), 0, 0, Math.PI * 2);
      if (lit) {
        const g = ctx.createRadialGradient(x, y - r * 0.2, 0, x, y, r);
        g.addColorStop(0, "#ffffff");
        g.addColorStop(0.5, LANE_COLORS[lane]);
        g.addColorStop(1, LANE_DIM[lane]);
        ctx.fillStyle = g;
        ctx.shadowColor = LANE_COLORS[lane];
        ctx.shadowBlur = 24;
      } else {
        const g = ctx.createRadialGradient(x, y - r * 0.2, 0, x, y, r);
        g.addColorStop(0, LANE_DIM[lane]);
        g.addColorStop(1, "rgba(8,11,17,0.92)");
        ctx.fillStyle = g;
      }
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = Math.max(1.5, r * 0.1);
      ctx.strokeStyle = lit ? "rgba(255,255,255,0.9)" : LANE_DIM[lane];
      ctx.stroke();
      ctx.restore();
    }
  }

  /** Flame bursts at the fret of every note hit in the last BURST_S. */
  private drawBursts(t: number): void {
    const { ctx } = this;
    const y = this.strikeY;
    const r = this.laneW * 0.34;
    for (let i = 0; i < this.notes.length; i++) {
      const st = this.states[i];
      if (st.hitAt < 0) continue;
      const age = t - st.hitAt;
      if (age < 0 || age > BURST_S) continue;
      const a = age / BURST_S;
      const mask = this.notes[i][1];
      for (let lane = 0; lane < 5; lane++) {
        if (!(mask & (1 << lane))) continue;
        const x = this.xOf(lane, 0);
        ctx.save();
        ctx.globalCompositeOperation = "lighter";

        const fade = (1 - a) * (1 - a);
        const rad = r * (0.45 + 1.25 * a);
        const core = ctx.createRadialGradient(x, y, 0, x, y, rad);
        core.addColorStop(0, `rgba(255,255,255,${0.85 * fade})`);
        core.addColorStop(0.3, LANE_COLORS[lane] + hex2(0.55 * fade));
        core.addColorStop(0.65, `rgba(255,150,40,${0.22 * fade})`);
        core.addColorStop(1, "rgba(255,80,0,0)");
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.ellipse(x, y, rad, rad * 0.85, 0, 0, Math.PI * 2);
        ctx.fill();

        // Embers, thrown up the highway and out to the sides.
        const n = 14;
        for (let k = 0; k < n; k++) {
          const seed = i * 31 + lane * 7 + k;
          const ang = -Math.PI / 2 + (hash01(seed) - 0.5) * 2.0;
          const spd = r * (1.4 + 2.6 * hash01(seed + 101));
          const px = x + Math.cos(ang) * spd * a;
          const py = y + Math.sin(ang) * spd * a * 0.85;
          const pr = r * 0.13 * (1 - a) * (0.5 + hash01(seed + 202));
          if (pr <= 0.2) continue;
          const eg = ctx.createRadialGradient(px, py, 0, px, py, pr * 2.4);
          eg.addColorStop(0, `rgba(255,240,200,${0.9 * fade})`);
          eg.addColorStop(0.5, `rgba(255,150,40,${0.55 * fade})`);
          eg.addColorStop(1, "rgba(255,60,0,0)");
          ctx.fillStyle = eg;
          ctx.beginPath();
          ctx.arc(px, py, pr * 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }
  }

  /** In-canvas vignette; the bezel, scanlines and curvature are CSS. */
  private drawVignette(): void {
    const { ctx, w, h } = this;
    const g = ctx.createRadialGradient(w / 2, h * 0.52, h * 0.25, w / 2, h * 0.52, h * 0.95);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0,0.75)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }
}

/** 0..1 alpha as a 2-digit hex suffix for `#rrggbb` colors. */
function hex2(a: number): string {
  const v = Math.round(Math.min(Math.max(a, 0), 1) * 255);
  return v.toString(16).padStart(2, "0");
}
