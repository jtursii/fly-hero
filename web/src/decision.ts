/** The decision panel: the six numbers the brain actually outputs.
 *
 *  These are the readout's probabilities from `probs.bin` -- one per fret
 *  plus strum -- drawn as bars in the lane colors, interpolated between the
 *  two 60 Hz frames that bracket the master playhead. The tick on each track
 *  is the decoder's threshold (`flyhero/game/decoder.py`), so a bar past the
 *  tick is the reason the fret went down; a bar short of it is the reason it
 *  did not.
 *
 *  Whether an output actually *fired* is never inferred from the bar: it is
 *  read from `actions.bin`, the same bitmask the CRT's fret lights use. For
 *  the frets the two always agree, but strum is deliberately different --
 *  the decoder fires it on a strict local maximum above the threshold, so a
 *  bar can sit well past the tick with no strum, and the panel would be
 *  lying if it drew the threshold as the whole story. Hence the "peak" note
 *  on the strum row.
 *
 *  Like every other panel this is a pure function of the playhead, so
 *  scrubbing lands on the right frame instead of replaying toward it.
 */

import { LANE_COLORS } from "./highway.ts";
import type { Manifest } from "./types.ts";

/** Strum is not a lane, so it gets the UI accent rather than a lane color. */
const STRUM_COLOR = "#6fd3ff";
const ROW_COLORS = [STRUM_COLOR, ...LANE_COLORS];
const ROW_LABELS = ["STRUM", "GREEN", "RED", "YELLOW", "BLUE", "ORANGE"];
/** Column of probs.bin per row: strum is channel 5, frets are 0..4. */
const ROW_CHANNEL = [5, 0, 1, 2, 3, 4];
/** Bit of actions.bin per row (bit 5 strum, bits 0..4 frets). */
const ROW_BIT = [5, 0, 1, 2, 3, 4];

/** `decoder.FRET_THRESHOLD` / `decoder.STRUM_THRESHOLD`. Both are 0.5 (D27);
 *  kept as one constant because the export ships one decoder's output. */
const THRESHOLD = 0.5;
/** A strum is one frame long. Holding its marker lit this long makes it
 *  visible at 60 fps without inventing any duration for the event itself. */
const FIRE_HOLD_S = 0.13;

export class DecisionView {
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private manifest: Manifest | null = null;
  private probs = new Uint8Array(0);
  private actions = new Uint8Array(0);

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.resize();
  }

  setSong(manifest: Manifest | null, probs?: Uint8Array, actions?: Uint8Array): void {
    this.manifest = manifest;
    this.probs = probs ?? new Uint8Array(0);
    this.actions = actions ?? new Uint8Array(0);
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

  /** Probability of `channel` at song time `t`, linearly interpolated
   *  between the frames on either side of the playhead. */
  private probAt(channel: number, t: number): number {
    const m = this.manifest;
    if (!m || !this.probs.length) return 0;
    const last = m.frames - 1;
    const x = Math.min(Math.max(t * m.fps, 0), last);
    const f0 = Math.floor(x);
    const f1 = Math.min(f0 + 1, last);
    const a = this.probs[f0 * 6 + channel] / 255;
    const b = this.probs[f1 * 6 + channel] / 255;
    return a + (b - a) * (x - f0);
  }

  /** 0..1: how recently this output fired, from `actions.bin` alone. Frets
   *  read the playhead's own frame (the CRT lights them the same way); a
   *  strum decays over FIRE_HOLD_S because it lasts a single frame. */
  private firedAt(row: number, t: number): number {
    const m = this.manifest;
    if (!m || !this.actions.length) return 0;
    const bit = ROW_BIT[row];
    const frame = Math.min(Math.max(Math.round(t * m.fps), 0), this.actions.length - 1);
    if (row > 0) return (this.actions[frame] >> bit) & 1;
    const back = Math.max(0, frame - Math.ceil(FIRE_HOLD_S * m.fps));
    for (let f = frame; f >= back; f--) {
      if ((this.actions[f] >> bit) & 1) return 1 - (frame - f) / (FIRE_HOLD_S * m.fps);
    }
    return 0;
  }

  /** The six rows at song time `t`, in row order (strum first). Exposed for
   *  `scripts/check_panels.mjs`, which checks them against probs.bin and
   *  actions.bin rather than trusting the drawing. */
  sample(t: number): { label: string; prob: number; fired: number }[] {
    return ROW_LABELS.map((label, row) => ({
      label,
      prob: this.probAt(ROW_CHANNEL[row], t),
      fired: this.firedAt(row, t),
    }));
  }

  draw(t: number): void {
    const ctx = this.ctx;
    ctx.fillStyle = "#070a11";
    ctx.fillRect(0, 0, this.w, this.h);
    if (!this.manifest) return;

    const labelW = 52;
    const valueW = 34;
    const x0 = labelW + 6;
    const x1 = this.w - valueW - 4;
    const trackW = Math.max(10, x1 - x0);
    const rowH = this.h / ROW_LABELS.length;
    const barH = Math.max(5, Math.min(11, rowH * 0.46));
    ctx.textBaseline = "middle";

    for (let row = 0; row < ROW_LABELS.length; row++) {
      const cy = rowH * (row + 0.5);
      const p = this.probAt(ROW_CHANNEL[row], t);
      const fired = this.firedAt(row, t);
      const color = ROW_COLORS[row];
      const y = Math.round(cy - barH / 2);

      ctx.font = "700 9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = fired > 0 ? color : "#54637d";
      ctx.globalAlpha = fired > 0 ? 0.55 + 0.45 * fired : 1;
      ctx.fillText(ROW_LABELS[row], 2, cy);
      ctx.globalAlpha = 1;

      // Track, then the probability itself.
      ctx.fillStyle = "#121a27";
      ctx.fillRect(x0, y, trackW, barH);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.34 + 0.66 * Math.min(1, p / THRESHOLD);
      ctx.fillRect(x0, y, Math.max(1, trackW * p), barH);
      ctx.globalAlpha = 1;

      // A fired output gets a lit cap and a halo -- the bar alone never says
      // it fired, because for strum the bar alone does not decide it.
      if (fired > 0) {
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 9 * fired;
        ctx.fillStyle = color;
        ctx.fillRect(x0, y, Math.max(2, trackW * p), barH);
        ctx.restore();
        ctx.fillStyle = "#e8f3ff";
        ctx.globalAlpha = fired;
        ctx.fillRect(x0 + Math.max(2, trackW * p) - 2, y, 2, barH);
        ctx.globalAlpha = 1;
      }

      // The decoder's threshold.
      const tx = Math.round(x0 + trackW * THRESHOLD);
      ctx.fillStyle = p > THRESHOLD ? "#cfe0f5" : "#5b6b85";
      ctx.fillRect(tx, y - 2, 1, barH + 4);

      ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.fillStyle = p > THRESHOLD ? "#cfe0f5" : "#6d7f9b";
      ctx.fillText(p.toFixed(2), this.w - 3, cy);
    }
  }
}
