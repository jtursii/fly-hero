/** The single master clock. One playhead in song seconds drives every panel
 *  (PLAN Phase 7 task 7); nothing animates off its own timer.
 *
 *  Song time is the recording's timebase: frame f of actions.bin / probs.bin
 *  is song time f / fps, and manifest.notes carry times on the same clock
 *  (export/record.py scores "frame 0 = song time 0"). The audio file is
 *  placed on it by the manifest's one offset: audio time 0 lands at song time
 *  `audio_offset_s`, so audioTime = songTime - audio_offset_s.
 *
 *  The audio element is the authority, but `currentTime` moves in steps of one
 *  audio render quantum (~23 ms here), so reading it raw makes the highway
 *  judder and biases any correction by half a step. So: the clock free-runs on
 *  wall time, the audio's position is interpolated between its steps, and the
 *  clock is pulled back toward it every frame -- eased when the disagreement
 *  is small, snapped when it is large or when the audio has just been seeked.
 *
 *  `scripts/check_sync.mjs` measures the result in a real browser.
 */

/** Past this much disagreement with the audio, jump rather than ease. Well
 *  under one 60 Hz frame, so the jump itself is never visible. */
const SNAP_S = 0.025;
/** Fraction of the remaining error taken back per frame when easing. */
const EASE = 0.12;

export type Rate = 0.25 | 1 | 2;

export class MasterClock {
  private t = 0;
  private wallMs = 0;
  private playing = false;
  private audio: HTMLAudioElement | null = null;
  private audioOffset = 0;
  private audioDuration = 0;
  /** Set when the audio element has just landed on a new position, so the
   *  next usable reading is taken as truth instead of eased toward. */
  private resync = false;
  private onSeeked: (() => void) | null = null;
  /** `audio.currentTime` only moves in steps of one audio render quantum, so
   *  reading it raw makes the clock chase a staircase. We remember the last
   *  value and when it appeared, and interpolate between steps. */
  private lastCT = -1;
  private lastCTWallMs = 0;
  /** False until `currentTime` has been seen to move since the last play or
   *  seek. An element that has been told to play sits at its start position
   *  for a moment first, and the clock must wait there with it rather than
   *  running ahead of silence. */
  private started = false;
  /** What the user asked for, set synchronously by play/pause/toggle. The
   *  audio element is then reconciled to it asynchronously -- `play()`
   *  returns a promise, and a click arriving while one is in flight must
   *  not leave the clock and the audio in different states. */
  private want = false;
  /** True while `reconcile` is running, so there is only ever one in flight
   *  and the newest `want` is the one that wins. */
  private syncing = false;

  /** End of the recording in song seconds (includes the 1 s post-roll). */
  duration = 0;
  rate: Rate = 1;
  /** audio time minus clock time, in seconds -- the sync readout. */
  drift = 0;

  /** Point the clock at a song. Resets to 0 and leaves playback stopped. */
  setSong(audio: HTMLAudioElement, audioOffsetS: number, durationS: number): void {
    this.pause();
    if (this.audio && this.onSeeked) this.audio.removeEventListener("seeked", this.onSeeked);
    this.onSeeked = () => {
      this.resync = true;
      this.lastCT = -1;
      this.started = false;
    };
    audio.addEventListener("seeked", this.onSeeked);
    this.audio = audio;
    this.audioOffset = audioOffsetS;
    this.audioDuration = Number.isFinite(audio.duration) ? audio.duration : 0;
    this.duration = durationS;
    this.drift = 0;
    this.seek(0);
  }

  /** The audio element's own duration, once its metadata has arrived. */
  noteAudioDuration(d: number): void {
    if (Number.isFinite(d)) this.audioDuration = d;
  }

  get time(): number {
    return this.t;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Whether the audio can currently carry the clock (it runs out before the
   *  post-roll does, and it is silent while seeking or stalled). */
  private audioUsable(): boolean {
    const a = this.audio;
    if (!a || a.seeking || a.readyState < 1 /* HAVE_METADATA */) return false;
    const at = this.t - this.audioOffset;
    return at >= 0 && (this.audioDuration === 0 || at < this.audioDuration - 0.05);
  }

  play(): void {
    this.setPlaying(true);
  }

  pause(): void {
    this.setPlaying(false);
  }

  /** One click, one flip -- always, however fast they arrive. */
  toggle(): void {
    this.setPlaying(!this.want);
  }

  private setPlaying(want: boolean): void {
    // Pressing play on a finished song restarts it. Without this the clock
    // is already at the end, tick() stops it again immediately, and the
    // click looks like it did nothing.
    if (want && this.duration > 0 && this.t >= this.duration - 0.05) this.seek(0);
    this.want = want;
    this.playing = want;
    if (want) {
      this.wallMs = performance.now();
      this.lastCT = -1;
      this.started = false;
    }
    void this.reconcile();
  }

  /** Bring the audio element to `want`. Single-flight: a call made while one
   *  is running is a no-op, because the running loop re-reads `want` and so
   *  already acts on the newest request. */
  private async reconcile(): Promise<void> {
    const a = this.audio;
    if (!a || this.syncing) return;
    this.syncing = true;
    try {
      // Bounded: each pass either settles or observes a newer `want`.
      for (let guard = 0; guard < 8; guard++) {
        const want = this.want;
        if (want === !a.paused) break;
        if (!want) {
          a.pause();
          continue;
        }
        a.playbackRate = this.rate;
        const at = Math.max(0, this.t - this.audioOffset);
        if (Math.abs(a.currentTime - at) > 0.05) a.currentTime = at;
        try {
          await a.play();
        } catch {
          // Either a newer pause() aborted this play (then `want` is already
          // false and the next pass settles), or the browser refused it. A
          // refusal must stop the clock too: a running clock over silent
          // audio is exactly the disagreement this method exists to prevent.
          if (this.want === want) {
            this.want = false;
            this.playing = false;
            break;
          }
        }
      }
    } finally {
      this.syncing = false;
    }
  }

  setRate(rate: Rate): void {
    this.rate = rate;
    if (this.audio) this.audio.playbackRate = rate;
  }

  /** Move the playhead, taking the audio with it. */
  seek(t: number): void {
    this.t = Math.min(Math.max(t, 0), this.duration);
    this.wallMs = performance.now();
    this.drift = 0;
    this.lastCT = -1;
    this.started = false;
    const a = this.audio;
    if (a) {
      const at = this.t - this.audioOffset;
      // Clamp into the file: song time can sit before or after the audio.
      a.currentTime = Math.min(Math.max(at, 0), this.audioDuration || Math.max(at, 0));
    }
  }

  /** Advance to wall-clock `nowMs` and re-lock to the audio. Call once per
   *  animation frame, before anything reads `time`. */
  tick(nowMs: number): void {
    if (!this.playing) {
      this.wallMs = nowMs;
      return;
    }
    const dt = Math.min((nowMs - this.wallMs) / 1000, 0.25) * this.rate;
    this.wallMs = nowMs;
    this.t += dt;

    // The element can stop on its own -- an aborted play, a stall, a lost
    // decoder -- and the clock must not keep running over silence.
    if (this.audioUsable() && this.audio!.paused && !this.syncing) void this.reconcile();

    if (this.audioUsable() && !this.audio!.paused) {
      const ct = this.audio!.currentTime;
      if (ct !== this.lastCT) {
        if (this.lastCT >= 0) this.started = true;
        this.lastCT = ct;
        this.lastCTWallMs = nowMs;
      }
      // Only interpolate once the audio is demonstrably running; before that
      // its reported position is the truth and must not be extrapolated.
      const est = this.started
        ? this.lastCT + ((nowMs - this.lastCTWallMs) / 1000) * this.rate
        : this.lastCT;
      const audioSongTime = est + this.audioOffset;
      this.drift = audioSongTime - this.t;
      if (!this.started || this.resync || Math.abs(this.drift) > SNAP_S) {
        // Audio not yet rolling, first reading after a seek, or a gross
        // disagreement: take the audio's word for it rather than easing.
        this.t = audioSongTime;
        this.resync = false;
        this.drift = 0;
      } else {
        this.t += this.drift * EASE;
      }
    } else {
      this.drift = 0;
    }

    if (this.t >= this.duration) {
      this.t = this.duration;
      this.pause();
    }
  }
}
