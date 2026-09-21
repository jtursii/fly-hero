/** The fly's-eye panel (PLAN Phase 7 task 5, site plan §2 item 1).
 *
 *  The retina is *derived in the browser*, not shipped: a photoreceptor's
 *  current is a pure function of the rendered 64x64 highway frame, which is
 *  a pure function of the manifest's notes and the playhead. This file is a
 *  port of the Python that produced the recording, and `scripts/
 *  check_retina.mjs` holds it to that by comparing against `retina.bin`:
 *
 *    flyhero/game/render.py   render_frame    -> renderFrame64
 *    flyhero/game/retina.py   gaussian_blur   -> blur64 (sigma 1, 5 taps)
 *                             bilinear_sample -> nearest pixel per receptor
 *    flyhero/game/sim.py      _obs            -> RetinaEncoder.step
 *
 *  current = gain * (intensity - ema), then ema = a * intensity + (1-a) * ema
 *  -- the EMA is read before it is updated, and gain/alpha/sigma come from
 *  configs/game.yaml (4.0 / 0.2 / 1.0).
 */

import type { Manifest, Note } from "./types.ts";

export const FRAME = 64;
const LOOKAHEAD_S = 1.0;
const GAIN = 4.0;
const EMA_ALPHA = 0.2;
/** render.py's greys. */
const BACKGROUND = 20;
const LANE_LINE = 60;
const STRIKELINE = 130;
const NOTE_TAIL = 160;
const NOTE_HEAD = 255;
const STRIKELINE_ROW = 54; // int(round(0.85 * 63))
/** int(round(x)) for np.linspace(0, 64, 6). */
const LANE_EDGES = [0, 13, 26, 38, 51, 64];
/** exp(-ax^2 / 2) for ax = -2..2, normalised (sigma 1, kernel_size 5). */
const KERNEL = (() => {
  const k = [-2, -1, 0, 1, 2].map((a) => Math.exp(-(a * a) / 2));
  const s = k.reduce((a, b) => a + b, 0);
  return k.map((v) => v / s);
})();
/** Frames of lead-in after a seek. The EMA's memory decays by 0.8 per
 *  frame, so 40 frames leaves 0.8^40 = 1.3e-4 of any wrong starting state
 *  -- far below `retina.bin`'s own 1/255 quantization. */
const BURN_FRAMES = 40;

/** Python's round() and numpy's are half-to-even; JavaScript's is half-up,
 *  and render.py rounds row positions that can land exactly on .5. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const clampInt = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

/** render.py's `render_frame`, for `frame_size = 64`. */
export function renderFrame64(notes: Note[], t: number, out: Uint8Array): void {
  out.fill(BACKGROUND);
  for (let i = 1; i < LANE_EDGES.length - 1; i++) {
    const c = LANE_EDGES[i];
    for (let r = 0; r < FRAME; r++) out[r * FRAME + c] = LANE_LINE;
  }
  for (let c = 0; c < FRAME; c++) out[STRIKELINE_ROW * FRAME + c] = STRIKELINE;

  const rowForTime = (noteTime: number) => (1 - (noteTime - t) / LOOKAHEAD_S) * STRIKELINE_ROW;
  for (const [noteTime, laneMask, sustain] of notes) {
    if (noteTime < t - 0.5 || noteTime > t + LOOKAHEAD_S) continue;
    const headRow = rowForTime(noteTime);
    const tailRow = sustain > 0 ? rowForTime(noteTime + sustain) : headRow;
    for (let lane = 0; lane < 5; lane++) {
      if (!(laneMask & (1 << lane))) continue;
      const c0 = Math.max(LANE_EDGES[lane], 0);
      const c1 = Math.min(LANE_EDGES[lane + 1], FRAME);
      const r0 = clampInt(roundHalfEven(Math.min(tailRow, headRow)), 0, FRAME - 1);
      const r1 = clampInt(roundHalfEven(Math.max(tailRow, headRow)), 0, FRAME - 1);
      if (sustain > 0 && r1 > r0) {
        for (let r = r0; r < r1; r++) {
          for (let c = c0; c < c1; c++) {
            const i = r * FRAME + c;
            if (out[i] < NOTE_TAIL) out[i] = NOTE_TAIL;
          }
        }
      }
      const headR = roundHalfEven(headRow);
      if (headR >= 0 && headR < FRAME) {
        const lo = Math.max(headR - 1, 0);
        const hi = Math.min(headR + 1, FRAME);
        for (let r = lo; r < hi; r++) {
          for (let c = c0; c < c1; c++) out[r * FRAME + c] = NOTE_HEAD;
        }
      }
    }
  }
}

/** retina.py's `gaussian_blur` on `frame / 255`: separable, edge-padded. */
export function blur64(frame: Uint8Array, rowBuf: Float32Array, out: Float32Array): void {
  for (let r = 0; r < FRAME; r++) {
    for (let c = 0; c < FRAME; c++) {
      let acc = 0;
      for (let i = 0; i < 5; i++) {
        acc += KERNEL[i] * frame[r * FRAME + clampInt(c + i - 2, 0, FRAME - 1)];
      }
      rowBuf[r * FRAME + c] = acc / 255;
    }
  }
  for (let r = 0; r < FRAME; r++) {
    for (let c = 0; c < FRAME; c++) {
      let acc = 0;
      for (let i = 0; i < 5; i++) {
        acc += KERNEL[i] * rowBuf[clampInt(r + i - 2, 0, FRAME - 1) * FRAME + c];
      }
      out[r * FRAME + c] = acc;
    }
  }
}

/** One photoreceptor population, stepped one 60 Hz frame at a time. */
export class RetinaEncoder {
  /** Signed current per displayed photoreceptor, in model units. */
  readonly current: Float32Array;
  private ema: Float32Array;
  private intensity: Float32Array;
  private pixel: Int32Array;
  private frame = new Uint8Array(FRAME * FRAME);
  private rowBuf = new Float32Array(FRAME * FRAME);
  private blurred = new Float32Array(FRAME * FRAME);
  private started = false;

  /** `displayPos` is brain.json's `retina_display_pos`: [vertical, horizontal]
   *  in [0,1], the real PCA image position of each shown photoreceptor. */
  constructor(displayPos: [number, number][]) {
    const n = displayPos.length;
    this.current = new Float32Array(n);
    this.ema = new Float32Array(n);
    this.intensity = new Float32Array(n);
    this.pixel = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      // retina.py's _nearest_rc: round to the nearest pixel, then clip.
      const r = clampInt(roundHalfEven(displayPos[i][0] * (FRAME - 1)), 0, FRAME - 1);
      const c = clampInt(roundHalfEven(displayPos[i][1] * (FRAME - 1)), 0, FRAME - 1);
      this.pixel[i] = r * FRAME + c;
    }
  }

  reset(): void {
    this.started = false;
    this.current.fill(0);
  }

  /** Render the highway at song time `t`, sample it, and advance the EMA --
   *  exactly sim.py's `_obs` for one frame. */
  step(notes: Note[], t: number): void {
    renderFrame64(notes, t, this.frame);
    blur64(this.frame, this.rowBuf, this.blurred);
    for (let i = 0; i < this.pixel.length; i++) this.intensity[i] = this.blurred[this.pixel[i]];
    if (!this.started) {
      // sim.py seeds the EMA with the first frame it sees, so current is 0.
      this.ema.set(this.intensity);
      this.started = true;
    }
    for (let i = 0; i < this.current.length; i++) {
      this.current[i] = GAIN * (this.intensity[i] - this.ema[i]);
      this.ema[i] = EMA_ALPHA * this.intensity[i] + (1 - EMA_ALPHA) * this.ema[i];
    }
  }

  /** Currents at 60 Hz frame `f` (song time f/60) from a cold start, with
   *  `BURN_FRAMES` of lead-in. Used after a scrub, and by the parity check. */
  seekTo(notes: Note[], f: number, frameTime: (f: number) => number): void {
    this.reset();
    for (let g = Math.max(0, f - BURN_FRAMES); g <= f; g++) this.step(notes, frameTime(g));
  }
}

/** Display geometry: real image positions mapped into an eye-shaped outline.
 *
 *  `place_on_screen` (retina.py) already splits the two eyes -- horizontal
 *  < 0.5 is the fly's left eye, >= 0.5 the right -- with the frontal field
 *  at the shared midline and the periphery at the outer edge. The PCA
 *  normalisation fills a unit square, so the square is mapped onto a disk
 *  (the standard elliptical grid mapping) to give each eye a real outline;
 *  that is a display choice, and the relative layout inside it is the real
 *  `image_pos`, never a decorative hex grid.
 */
interface EyeDot {
  eye: 0 | 1;
  x: number; // [-1, 1] within its own eye
  y: number;
}

function layout(displayPos: [number, number][]): EyeDot[] {
  return displayPos.map(([v, h]) => {
    const eye: 0 | 1 = h < 0.5 ? 0 : 1;
    // u: 0 at the frontal midline, 1 at the periphery, spread across the
    // whole eye. The left eye is the mirror image of the right, as the
    // anatomy is: each eye's frontal field sits on its nasal side.
    const u = eye === 1 ? (h - 0.5) * 2 : (0.5 - h) * 2;
    const sx = eye === 1 ? u * 2 - 1 : 1 - u * 2;
    const sy = v * 2 - 1; // 0 = dorsal = image top
    // Square -> disk, so the lattice fills an eye rather than a box.
    return { eye, x: sx * Math.sqrt(1 - (sy * sy) / 2), y: sy * Math.sqrt(1 - (sx * sx) / 2) };
  });
}

const REST = [66, 78, 100] as const; // dim grey: at rest
const POSITIVE = [255, 255, 255] as const; // light arriving
const NEGATIVE = [92, 140, 255] as const; // light leaving

export class RetinaView {
  private ctx: CanvasRenderingContext2D;
  private dots: EyeDot[];
  private encoder: RetinaEncoder;
  private notes: Note[] = [];
  /** p99.5 of |current| from the recording: the panel's honest full scale. */
  private scale = 1;
  private lastFrame = -1;
  private ready = false;

  constructor(canvas: HTMLCanvasElement, displayPos: [number, number][]) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context for the retina panel");
    this.ctx = ctx;
    this.dots = layout(displayPos);
    this.encoder = new RetinaEncoder(displayPos);
    this.resize();
  }

  /** The panel's current photoreceptor currents, for scripts/check_retina.mjs. */
  get currents(): Float32Array {
    return this.encoder.current;
  }

  setSong(manifest: Manifest | null): void {
    this.notes = manifest ? manifest.notes : [];
    this.scale = manifest ? manifest.retina_scale : 1;
    this.ready = manifest !== null;
    this.lastFrame = -1;
    this.encoder.reset();
  }

  resize(): void {
    const c = this.ctx.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.max(1, Math.round(c.clientWidth * dpr));
    c.height = Math.max(1, Math.round(c.clientHeight * dpr));
  }

  /** Song time of 60 Hz display frame `f`. */
  private static frameTime = (f: number) => f / 60;

  /** `t` is the master playhead: the panel is a pure function of it, so
   *  scrubbing and rate changes need no special handling. */
  render(t: number): void {
    if (this.ready) {
      const f = Math.max(0, Math.round(t * 60));
      if (f !== this.lastFrame) {
        const gap = f - this.lastFrame;
        if (this.lastFrame >= 0 && gap > 0 && gap <= BURN_FRAMES) {
          // Playing (or fast-forwarding): step every frame the EMA missed.
          for (let g = this.lastFrame + 1; g <= f; g++) this.encoder.step(this.notes, g / 60);
        } else {
          this.encoder.seekTo(this.notes, f, RetinaView.frameTime);
        }
        this.lastFrame = f;
      }
    }
    this.draw();
  }

  private draw(): void {
    const { ctx } = this;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Two eyes side by side, each tilted outward.
    const pad = h * 0.045;
    const eyeH = h - 2 * pad;
    const eyeW = Math.min(w * 0.46, eyeH * 0.86);
    const ry = eyeH / 2;
    const rx = eyeW / 2;
    const cy = h / 2;
    const dotR = Math.max(1, rx * 0.032);

    for (const eye of [0, 1] as const) {
      const cx = eye === 0 ? w * 0.5 - w * 0.245 : w * 0.5 + w * 0.245;
      const tilt = (eye === 0 ? -1 : 1) * 0.16; // leaning away from the midline
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(tilt);

      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      // Visible with no notes on screen: a faint fill and a rim, always drawn.
      ctx.fillStyle = "rgba(10, 15, 24, 0.85)";
      ctx.fill();
      ctx.strokeStyle = "rgba(120, 150, 190, 0.32)";
      ctx.lineWidth = Math.max(1, rx * 0.012);
      ctx.stroke();

      ctx.save();
      ctx.clip();
      for (let i = 0; i < this.dots.length; i++) {
        const d = this.dots[i];
        if (d.eye !== eye) continue;
        const cur = this.encoder.current[i] / this.scale;
        const m = Math.min(Math.abs(cur), 1);
        const target = cur >= 0 ? POSITIVE : NEGATIVE;
        const r = REST[0] + (target[0] - REST[0]) * m;
        const g = REST[1] + (target[1] - REST[1]) * m;
        const b = REST[2] + (target[2] - REST[2]) * m;
        ctx.beginPath();
        ctx.arc(d.x * rx * 0.94, d.y * ry * 0.94, dotR * (1 + 0.5 * m), 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
        ctx.globalAlpha = 0.62 + 0.38 * m;
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Vignette: darker toward the rim, so it reads as looking out.
      const vg = ctx.createRadialGradient(0, 0, rx * 0.25, 0, 0, rx);
      vg.addColorStop(0, "rgba(0, 0, 0, 0)");
      vg.addColorStop(0.72, "rgba(0, 0, 0, 0.18)");
      vg.addColorStop(1, "rgba(0, 0, 0, 0.72)");
      ctx.fillStyle = vg;
      ctx.fillRect(-rx, -ry, rx * 2, ry * 2);
      ctx.restore();
      ctx.restore();
    }
  }

}
