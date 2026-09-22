/** The rate oscilloscope: a 20 s window onto two whole-brain traces.
 *
 *  `scope.bin` carries a mean rate per 60 Hz frame over every one of the
 *  139,241 neurons, split by the sign of the neuron's transmitter (D53):
 *  83,793 excitatory and 45,550 inhibitory. Those are the two traces, in the
 *  connectome panel's own red and gold, so the same colors mean the same
 *  thing in both panels.
 *
 *  The window ends at the playhead and is read straight out of the file by
 *  frame index, so it is a pure function of the master clock -- there is no
 *  ring buffer to go stale, and scrubbing redraws the 20 s leading up to
 *  wherever the playhead landed rather than the 20 s that were on screen.
 *
 *  The vertical scale is fixed per song -- the range both traces cover over
 *  the whole recording -- rather than fitted to the window, so a quiet
 *  passage looks quiet instead of being stretched to fill the panel. The
 *  axis does not start at zero (these means never go near it, and a
 *  zero-based axis would draw both traces as two flat lines), so both ends
 *  of the range are printed on the panel.
 */

import type { Manifest } from "./types.ts";

const WINDOW_S = 20;
/** The connectome panel's signal-line colors (`brain.ts`). */
export const EXC_COLOR = "#ff7a5c";
export const INH_COLOR = "#ffc64d";

export class ScopeView {
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private manifest: Manifest | null = null;
  private scope: Float32Array | null = null;
  private channels = 0;
  private excCh = -1;
  private inhCh = -1;
  private yMin = 0;
  private yMax = 1;
  /** Set when the export predates D53 and has no exc/inh channels: the panel
   *  says so rather than substituting some other trace. */
  missing = false;

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.resize();
  }

  setSong(manifest: Manifest | null, scope?: Float32Array, channelNames?: string[]): void {
    this.manifest = manifest;
    this.scope = scope ?? null;
    this.excCh = channelNames?.indexOf("exc") ?? -1;
    this.inhCh = channelNames?.indexOf("inh") ?? -1;
    this.channels = channelNames?.length ?? 0;
    this.missing = !!manifest && !!scope && (this.excCh < 0 || this.inhCh < 0);
    this.yMin = 0;
    this.yMax = 1;
    if (!manifest || !scope || this.missing) return;
    let max = -Infinity;
    let min = Infinity;
    for (let f = 0; f < manifest.frames; f++) {
      const e = scope[f * this.channels + this.excCh];
      const i = scope[f * this.channels + this.inhCh];
      if (e > max) max = e;
      if (i > max) max = i;
      if (e < min) min = e;
      if (i < min) min = i;
    }
    const pad = Math.max((max - min) * 0.08, 1e-4);
    this.yMin = min - pad;
    this.yMax = max + pad;
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

  private at(ch: number, frame: number): number {
    return this.scope![frame * this.channels + ch];
  }

  /** One trace over the window, one line segment per pixel column. With 60
   *  samples/s against ~300 columns each column covers ~4 frames, so it is
   *  drawn from the column's mean -- the peaks stay honest in the reported
   *  maximum rather than in a line that would alias. */
  private trace(ch: number, f0: number, f1: number, y0: number, hh: number, color: string): void {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let px = 0; px <= this.w; px++) {
      const a = f0 + ((f1 - f0) * px) / this.w;
      const b = f0 + ((f1 - f0) * (px + 1)) / this.w;
      const lo = Math.max(0, Math.floor(a));
      const hi = Math.min(this.manifest!.frames - 1, Math.max(lo, Math.ceil(b) - 1));
      let sum = 0;
      let n = 0;
      for (let f = lo; f <= hi; f++) {
        sum += this.at(ch, f);
        n++;
      }
      const v = n ? sum / n : 0;
      const y = y0 + hh - ((v - this.yMin) / (this.yMax - this.yMin)) * hh;
      if (px === 0) ctx.moveTo(px, y);
      else ctx.lineTo(px, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }

  /** The two traces' values at the playhead, for `check_panels.mjs`. */
  sample(t: number): { exc: number; inh: number; yMin: number; yMax: number; frame: number } {
    const m = this.manifest;
    if (!m || !this.scope || this.missing) {
      return { exc: NaN, inh: NaN, yMin: this.yMin, yMax: this.yMax, frame: -1 };
    }
    const frame = Math.min(Math.max(Math.round(t * m.fps), 0), m.frames - 1);
    return {
      exc: this.at(this.excCh, frame),
      inh: this.at(this.inhCh, frame),
      yMin: this.yMin,
      yMax: this.yMax,
      frame,
    };
  }

  draw(t: number): void {
    const ctx = this.ctx;
    ctx.fillStyle = "#070a11";
    ctx.fillRect(0, 0, this.w, this.h);
    const m = this.manifest;
    if (!m || !this.scope) return;

    const pad = 12;
    const y0 = 4;
    const hh = Math.max(10, this.h - y0 - pad);

    ctx.strokeStyle = "#141c2a";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Math.round(y0 + (hh * i) / 4) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.w, y);
      ctx.stroke();
    }

    if (this.missing) {
      ctx.fillStyle = "#5c6d8a";
      ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("exc/inh traces not in this export", this.w / 2, y0 + hh / 2);
      return;
    }

    // The window ends at the playhead and is clamped to the recording, so
    // the first 20 s show the song arriving rather than a blank canvas.
    const last = m.frames - 1;
    const f1 = Math.min(Math.max(t * m.fps, 0), last);
    const f0 = Math.max(0, f1 - WINDOW_S * m.fps);
    if (f1 - f0 > 1) {
      this.trace(this.inhCh, f0, f1, y0, hh, INH_COLOR);
      this.trace(this.excCh, f0, f1, y0, hh, EXC_COLOR);
    }

    // Playhead edge: the newest sample is the right-hand edge.
    ctx.fillStyle = "rgba(207, 224, 245, 0.35)";
    ctx.fillRect(this.w - 1, y0, 1, hh);

    const frame = Math.min(Math.max(Math.round(t * m.fps), 0), last);
    ctx.font = "8.5px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = "#43526b";
    ctx.fillText(`${this.yMin.toFixed(3)} – ${this.yMax.toFixed(3)} mean rate, all 139,241`, 1, this.h - 2);
    ctx.textAlign = "right";
    ctx.fillStyle = EXC_COLOR;
    ctx.fillText(this.at(this.excCh, frame).toFixed(4), this.w - 44, this.h - 2);
    ctx.fillStyle = INH_COLOR;
    ctx.fillText(this.at(this.inhCh, frame).toFixed(4), this.w - 2, this.h - 2);
  }
}
