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
 *  Pausing is the one place that does not ease at all. `pause()` stops the
 *  element synchronously and parks the playhead on `audio.currentTime`
 *  exactly; `play()` picks it up from there. See D55 -- easing across a
 *  pause is what made resuming jump.
 *
 *  `scripts/check_sync.mjs` measures the result in a real browser.
 */

/** Past this much disagreement with the audio, jump rather than ease. Well
 *  under one 60 Hz frame, so the jump itself is never visible. */
const SNAP_S = 0.025;
/** Fraction of the remaining error taken back per frame when easing. */
const EASE = 0.12;
/** On resume, re-seat the element if it is further than this from the frozen
 *  playhead. Normally 0: `pause()` froze the clock on the element's own
 *  position, so there is nothing to seat. It only fires when the freeze
 *  could not read the audio (song time outside the file, mid-seek). */
const RESUME_SEEK_S = 0.01;

export type Rate = 0.25 | 1 | 2;

/** What `?debug=1` puts on screen, and what `scripts/check_sync.mjs` reads. */
export interface ClockDebug {
  want: boolean;
  playing: boolean;
  audioPaused: boolean;
  audioCurrentTime: number;
  clockTime: number;
  /** Measured d(playhead)/d(wall clock) over the last ~0.5 s. */
  advanceRate: number;
  advancing: boolean;
  drift: number;
  syncing: boolean;
  /** Where the last pause parked the playhead, in song seconds. */
  frozenAt: number | null;
}

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
  /** Where the last pause parked the playhead, or null if it could not read
   *  the audio. Cleared on the next play or seek. Debug readout, and the
   *  value `reconcile` re-seats the element to on resume. */
  private frozenAt: number | null = null;
  /** Set by `play()`: hold the playhead where the pause left it until the
   *  element is demonstrably rolling again. See `tick`. */
  private holdForAudio = false;
  /** Playhead and wall time at the last advance-rate sample, for the debug
   *  overlay's "is the clock actually moving" line. */
  private rateT = 0;
  private rateWallMs = 0;
  private advanceRate = 0;

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
    this.lastCT = -1;
    this.started = false;
    if (want) {
      this.wallMs = performance.now();
      this.frozenAt = null;
      this.holdForAudio = true;
    } else {
      this.holdForAudio = false;
      this.freeze();
    }
    void this.reconcile();
  }

  /** Stop the audio and park the playhead on its position, both right now.
   *
   *  D55: this used to be left to `reconcile`, which returns early while an
   *  `audio.play()` promise is in flight -- so a pause landing in that
   *  window never reached the element and the music played on with the
   *  picture frozen. `HTMLMediaElement.pause()` is synchronous and aborts a
   *  pending `play()` (rejecting it, which `reconcile` already catches), so
   *  there is no reason to defer it.
   *
   *  The playhead is then parked on `audio.currentTime` rather than left
   *  wherever the free-running interpolation had reached, because that
   *  interpolation leads or lags the element by up to one render quantum.
   *  Freezing on the interpolated value and resuming from the element's is
   *  what produced the jump on resume. */
  private freeze(): void {
    const a = this.audio;
    if (!a) return;
    a.pause();
    // Read the element only after it has stopped, and only when it is
    // actually carrying the clock: outside the file's span (or mid-seek)
    // `currentTime` says nothing about where the song is.
    if (this.audioUsable()) {
      const ct = a.currentTime;
      // Chrome resumes a paused element at the next container packet
      // boundary, not where it stopped -- measured at 53-68 ms ahead on
      // these MP3s, which the clock then faithfully followed, and that was
      // the visible jump on resume. Assigning `currentTime` its own value
      // while paused resets that: the skip drops to under a millisecond.
      // It is a seek to where the element already is, so it makes no sound,
      // and it happens while stopped, where a glitch could not be heard
      // anyway. (`fastSeek` does not fix it; measured too.)
      a.currentTime = ct;
      this.t = Math.min(Math.max(ct + this.audioOffset, 0), this.duration);
      this.frozenAt = this.t;
    } else {
      this.frozenAt = null;
    }
    this.drift = 0;
    this.advanceRate = 0;
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
        // Normally a no-op on resume: `freeze()` parked the playhead on this
        // element's own position, so `at` is already where it sits. It fires
        // after a scrub, or when the freeze could not read the audio.
        if (Math.abs(a.currentTime - at) > RESUME_SEEK_S) a.currentTime = at;
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
    // A scrub replaces the pause's parked position with a chosen one.
    this.frozenAt = null;
    this.rateWallMs = 0;
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
    this.sampleAdvanceRate(nowMs);
    if (!this.playing) {
      this.wallMs = nowMs;
      // Last line of defence for D55: whatever happened to the pause, a
      // stopped clock must never sit over playing audio. Cheap (a property
      // read) and it makes the failure self-correcting instead of silent.
      // The element is brought back to the playhead rather than the
      // playhead to the element: the user paused at `t`, so `t` is right
      // and the audio's overrun is the error.
      const a = this.audio;
      if (a && !a.paused) {
        a.pause();
        const at = this.t - this.audioOffset;
        if (at >= 0 && Math.abs(a.currentTime - at) > RESUME_SEEK_S) a.currentTime = at;
      }
      return;
    }
    const dt = Math.min((nowMs - this.wallMs) / 1000, 0.25) * this.rate;
    this.wallMs = nowMs;
    // An element told to play does not start instantly -- it stays paused for
    // a few tens of milliseconds while the decoder spins up. Advancing the
    // playhead through that window is exactly the jump D55 is about (53 ms
    // measured), so on resume the clock waits where the pause left it until
    // the audio is demonstrably rolling. Only when the audio *could* be
    // carrying the clock: outside the file's span there is nothing to wait
    // for and the post-roll has to keep running.
    const waiting = this.holdForAudio && !this.started && this.audioUsable();
    if (!waiting) this.t += dt;
    if (this.started) this.holdForAudio = false;

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

  /** d(playhead)/d(wall) over ~0.5 s windows, so the debug overlay can say
   *  whether the clock is moving rather than leaving it to be inferred from
   *  a number that is changing too slowly to see. */
  private sampleAdvanceRate(nowMs: number): void {
    if (!this.rateWallMs) {
      this.rateWallMs = nowMs;
      this.rateT = this.t;
      return;
    }
    const dw = (nowMs - this.rateWallMs) / 1000;
    if (dw < 0.5) return;
    this.advanceRate = (this.t - this.rateT) / dw;
    this.rateWallMs = nowMs;
    this.rateT = this.t;
  }

  /** Everything `?debug=1` shows. Reading it has no side effects. */
  debug(): ClockDebug {
    const a = this.audio;
    return {
      want: this.want,
      playing: this.playing,
      audioPaused: a ? a.paused : true,
      audioCurrentTime: a ? a.currentTime : NaN,
      clockTime: this.t,
      advanceRate: this.advanceRate,
      advancing: Math.abs(this.advanceRate) > 0.02,
      drift: this.drift,
      syncing: this.syncing,
      frozenAt: this.frozenAt,
    };
  }
}
