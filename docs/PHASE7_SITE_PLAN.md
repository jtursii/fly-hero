# Phase 7 — Observation-only replay site (plan against the mockup)

Status: **plan only, nothing built.** Written 2026-09-17, revised 2026-09-17
with the user's decisions (below). Training untouched throughout.

> **Sessioning.** Session 1 = **Phase 6** (`export/record.py`,
> `export_web.py`, `validate_export.py`) — nothing to show a browser exists
> before this. Session 2+ = **Phase 7** (the site itself). No shareable page;
> this doc is the plan of record.

---

## 0. Decisions made this session (recorded in DECISIONS.md as D37/D38)

| # | Decision |
|---|---|
| Streak tile | **Cut.** `rules.py` is not touched. Stat strip becomes: hit rate, notes hit, **overstrums/min**, difficulty, training step + time. |
| Showcase songs | Test split only (invariant 5), **≤330 s**, rule applied before looking at hit rates. Candidate list below — user picks 3–5. |
| Song audio | **Shipped.** Full-mix audio for the chosen showcase songs is committed to this private repo (D37, amends D9 and invariant 8). Size reported in §2. |
| History rewrite | **Deferred.** Private repo, so the home-path leak is a non-issue for now. Noted in PROGRESS.md as required before any public release, with the `git filter-repo --replace-text` plan kept ready (§7). |
| Fret lights | Move from the CRT to the **controller in the fly's hands**. The screen shows only notes and hit/miss flashes. |
| Fly asset | **Stylized low-poly, authored in the pose**, primitives fallback behind a 2-hour box. No license hunting. |
| Retina | **Derived in the browser**, not shipped. Approved. |
| Wording | "mean activation" not "firing rate"; "soma-or-anchor position" stated accurately; training time = the **lineage** number (15h53m), labeled as lineage. |
| G4 / showcase checkpoint | **Training continues.** A checkpoint is only test-evaluated when it passes the *existing* val-side selection rule (3-eval MA sets a new Medium best, or a curriculum advance — D36). The showcase checkpoint is whichever last passed that when Phase 6 recording happens; never chosen by comparing test scores after the fact (D38). |

---

## 1. Showcase song candidates

Test split, Medium difficulty, ≤330 s, note count > 0 (one 0-note chart,
`64b8eb4a884ab963` "Demon" — Trivium, is a known ingest-issue song already
excluded from eval means; dropped here for the same reason). **22 qualify.**
Pick 3–5. Sorted by duration (shortest first — cheapest on the budget):

| Title — Artist | Duration | Notes | Data MB | Audio MB (128k) |
|---|---:|---:|---:|---:|
| Mauvais Garçon — Naast | 165 s | 453 | 6.98 | 2.64 |
| Paranoid — Black Sabbath (Steve Ouimette) | 169 s | 395 | 7.14 | 2.70 |
| Ace of Spades — Motörhead | 173 s | 452 | 7.33 | 2.77 |
| Hit Me With Your Best Shot — Pat Benatar (WaveGroup) | 177 s | 401 | 7.48 | 2.83 |
| Pretty Vacant — The Sex Pistols | 203 s | 477 | 8.57 | 3.25 |
| Guitar Battle vs. Tom Morello — Tom Morello | 208 s | 228 | 8.77 | 3.33 |
| The Wind Cries Mary — The Jimi Hendrix Experience | 208 s | 384 | 8.78 | 3.33 |
| Re-Education (Through Labor) — Rise Against | 225 s | 603 | 9.50 | 3.61 |
| The Joker — Steve Miller Band | 230 s | 428 | 9.70 | 3.69 |
| Slash Guitar Battle (Co-op) — Slash | 240 s | 426 | 10.10 | 3.84 |
| Slash Guitar Battle — Slash | 240 s | 623 | 10.10 | 3.84 |
| Stricken — Disturbed | 250 s | 903 | 10.50 | 3.99 |
| Armed and Ready — Michael Schenker Group | 263 s | 722 | 11.07 | 4.21 |
| Vinternoll2 — Kent | 265 s | 456 | 11.12 | 4.23 |
| Barracuda — Heart (WaveGroup) | 265 s | 646 | 11.15 | 4.24 |
| Sunshine of Your Love — Cream (WaveGroup) | 267 s | 501 | 11.24 | 4.28 |
| The Number of the Beast — Iron Maiden | 268 s | 671 | 11.27 | 4.29 |
| Fuel — Metallica | 270 s | 640 | 11.35 | 4.32 |
| Beat It — Michael Jackson | 278 s | 506 | 11.70 | 4.45 |
| The Boys Are Back In Town — Thin Lizzy | 294 s | 560 | 12.34 | 4.70 |
| Turn the Page (Live) — Bob Seger and the Silver Bullet Band | 300 s | 687 | 12.59 | 4.79 |
| L'Via L'Viaquez — The Mars Volta | 314 s | 381 | 13.17 | 5.02 |

Two of these ("Slash Guitar Battle" and its co-op version) are the same
song twice — worth avoiding both unless you want the contrast.

**Budget check, picking from the shortest end (worst case, avoid this only if you want the longer songs specifically):**

| Pick | Data (incl. shared 2.8 MB) | Audio | Combined |
|---|---:|---:|---:|
| shortest 3 | 24.2 MB | 8.1 MB | 32.4 MB |
| shortest 4 | 31.7 MB | 10.9 MB | 42.7 MB |
| shortest 5 | 40.3 MB | 14.2 MB | 54.5 MB |

The **≤60 MB budget in Phase 6's plan is for exported *data* only**
(activity/retina/actions/probabilities/events) and was scoped for "2–3
replays." Picking 5 songs pushes data alone close to that cap even at the
shortest end (40.3 MB); picking 5 of the *longer* candidates would exceed it
(5 × ~13 MB + 2.8 ≈ 68 MB). **If you pick 5, prefer ones toward the shorter
end of the list, or I'll drop K from 4096 to ~3000 slots to make room** —
your call once you've picked titles. Audio is a separate line (committed to
git, not part of the 60 MB data budget) and is nowhere near any real limit
(GitHub only soft-warns per-file at 50 MB; these are 2.6–5 MB each) — no risk
of "blowing the transfer budget" at 3–5 songs of any length on this list.

---

## 2. Data export format and size budget (updated)

Unchanged from the original plan except: audio added as a new,
separately-budgeted category (not counted against the 60 MB data cap), and
the size table now reflects `≤330 s` showcase songs (§1) rather than a
generic "300 s song" example.

### Design decisions that drive the data budget

1. **Retina is derived in the browser, not shipped** — approved this
   session. Pure function of the 64×64 frame, which is a pure function of
   manifest notes + playhead. Shipping it would cost ~59 MB/song at 20 fps;
   deriving it costs 0 bytes. Needs a parity test (§6, S5).
2. **Activity cannot be derived** — it is the recording. Gets the budget.
3. **Scoring is never reimplemented in JS** (invariant 4). `rules.py`
   produces hit/miss/overstrum events and metrics at record time; the
   browser only replays what `rules.py` already decided.
4. **Whole-population scalars (active count, mean activation, mean
   excitatory/inhibitory) are precomputed over all 139,241 neurons** — the
   browser only receives K slots and cannot compute these without
   extrapolating.

### Per song, `web/public/data/songs/<id>/`

| File | Layout | 300 s example |
|---|---|---:|
| `activity.bin` | uint4 `[F20, K]`, K=4096, 20 fps, 2 slots/byte | 12.29 MB |
| `probs.bin` | uint8 `[F60, 6]` — 5 fret + strum probabilities | 0.108 MB |
| `actions.bin` | uint8 `[F60]` lane bitmask + strum bit | 0.018 MB |
| `population.bin` | `[F60]` × (active uint16, mean_activation u8, meanE u8, meanI u8) | 0.090 MB |
| `events.json` | hits / misses / overstrums from `rules.py` | ~0.04 MB |
| `manifest.json` | title, artist, charter, difficulty, notes, offsets, metrics | ~0.06 MB |
| `audio.mp3` | **new** — 128 kbps CBR full mix (D37) | ~0.016 MB/s |

K=4096 = all 1,303 DNs + a fixed stratified sample across `super_class` +
top-variance fill — the same slot set for every song.

### Shared, once

`positions.bin` (uint16 `[139241,3]`, quantized), `classes.bin` (uint8
super_class), `signs.bin` (int8, 0=unknown — 9,898 neurons with no outgoing
edge), `slots.bin` (uint32 `[4096]` slot→node), `retina_pos.bin` (uint16
`[11108,2]` photoreceptor image coords), `flow_edges.bin` (~4096 sampled
edges), `silhouette.png` (soma-density backdrop), `brain.json`, `fly.glb`.
≈ 2.8 MB total, unchanged from the original plan.

### Precomputed vs. derived — unchanged

**Precomputed:** activity, readout probabilities, decoded actions,
`rules.py` events/metrics, population scalars, note charts, positions,
classes, signs, photoreceptor coordinates, flow edges, audio.
**Derived in browser:** the 64×64 highway frame, all retina currents, note
highway visuals, fret/strum lighting (on the controller — see decision
above), the 20 s oscilloscope ring buffer, flow-line brightness, running
hit/notes-hit counters (replaying `events.json`, never re-scoring).

---

## 3. The 3D fly

> **SUPERSEDED by D57 (2026-09-21, user decision): the 3D fly is cut
> entirely.** No `fly.glb`, no stool, no controller, no task S7. The CRT
> took the stage area instead and is now the hero of the page. §5 item 12
> ("the fly, stool, controller are illustration") is void with it; the
> About modal disclaims the CRT instead. Kept below as written.

**Decided: stylized low-poly, authored in the pose, in Blender → glTF.**
Fallback to three.js primitives behind a hard 2-hour box if the Blender work
stalls. No CC-model licensing search — not worth the risk given almost none
are rigged and this asset needs to be posed sitting on a stool with forelegs
on a controller.

**Poly budget:** fly ≤ 25k triangles, stool + controller ≤ 3k, `fly.glb`
≤ 1.5 MB Draco-compressed. One `fly.glb` holding fly + stool + controller +
CRT shell, screen as a named mesh (`CRT_Screen`) whose material is swapped
for the render-target texture at load. **Fret lights live on the
controller** (small emissive quads per lane, driven by `actions.bin`) —
not on the CRT.

Stool, controller, and CRT shell are authored geometry (boxes, cylinders, a
bevelled cube) — no sourcing needed. CRT curved glass is a shader effect in
UV space (§4), not curved geometry.

---

## 4. Rendering approach — unchanged from the original plan

Target 60 fps desktop, mobile fallback. One master playhead in seconds
drives every panel.

- **Brain:** one `THREE.Points` (139,241 verts), `aSlot` attribute, two
  `DataTexture`s (frame *n*/*n+1*) lerped in the vertex shader for size +
  brightness. Additive blending, base color by `super_class`. Silhouette =
  precomputed `silhouette.png` density plane (generated from real
  positions). Flow lines = `LineSegments` over sampled edges, red/gold from
  `signs.bin`, alpha from endpoint activity. Fixed camera, slow drift.
- **CRT + highway:** offscreen orthographic scene → `WebGLRenderTarget`
  (1024×1024); the CRT screen shader samples it with barrel distortion,
  scanlines, vignette, fresnel glow. No curved geometry.
- **Retina lattice:** two `THREE.Points` groups at real `image_pos`, tilted
  outward, hex sprite alpha texture, radial vignette. Currents computed on
  GPU: render the 64×64 frame to a small RT, blur σ=1, sample at
  photoreceptor UVs, ping-pong EMA texture (α=0.2), ×gain 4.0. `+ → white`,
  `− → blue`, `0 → dim gray`.
- **Oscilloscope:** plain 2D canvas, 1,200-sample ring buffer (20 s × 60
  fps) from `population.bin`.
- **Audio:** `<audio>`/Web Audio element per song, `audio.mp3`, synced to
  the master playhead via the manifest's audio offset (same mechanism D9
  specified for the local-file picker, just pointed at a shipped file
  instead of a user-provided one). Starts only after a user click (browser
  autoplay policy, also matches D9's intent).
- **Mobile:** ~40k-point subsample, no bloom, flat textured CRT quad,
  ~800-cell retina lattice/eye, DPR cap 1.5, 30 fps cap, lazy-load one song.

---

## 5. Honesty audit — resolved items

1. **Streak tile — cut.** No `rules.py` change, no fabricated number.
2. **Song choice — fixed.** Showcase songs come only from the candidate list
   in §1 (test split, ≤330 s, picked before looking at hit rates).
3. **Overstrums — added to the stat strip.** The weak number (40/min on
   Medium, from `eval.json`) is shown next to hit rate, not hidden.
4. **Colored highway notes** — kept as an audience convenience; the retina
   panel (grayscale contrast) is the fly's actual view, and copy will say so.
5. **Fret lights — moved to the controller**, where they are honestly an
   output, not implied visual feedback.
6. **"mean activation," not "firing rate"** — dimensionless rate-model
   units, not spikes/second.
7. **"2,700,429 connections"** — credit line states this is edges at
   min_syn≥5, aggregated over neuropils, not FlyWire's raw synapse count.
8. **Position caveat** — 21,141 of 139,241 neurons (15%) use an anchor point
   rather than a true soma. Recommendation (open, §8 item 12): render these
   points slightly dimmer (the data — `pos_source_id` — is already
   exported, so this is a free, honest visual cue) plus a footnote in the
   About modal, rather than a footnote alone.
9. **Training time — use the lineage number, 15h53m, labeled "training
   time (this checkpoint's lineage)."**
10. **Retina lattice** — real `image_pos` coordinates, not a decorative
    regular hex grid.
11. **"11,108 photoreceptors driven"**, not 11,118 — 10 are always exactly
    0 and excluded, stated plainly in the credit line.
12. **The fly, stool, controller are illustration** — About copy states
    this; only the brain/retina/DN/oscilloscope panels are recorded data.
13. **Sustain weakness (`sustain_frac` 0.348 on Medium)** — not claimed
    anywhere in the mockup; no action needed unless you want it surfaced
    too (not currently planned — flag if you want it added).

---

## 6. Task breakdown

| # | Session | Work | Est. |
|---|---|---|---|
| S0 | prereq | Pick 3–5 songs from §1; confirm K/budget if 5 chosen | your call |
| S1 | **Phase 6** | `export/record.py` (record from whichever checkpoint currently passes D38's rule), `export_web.py` (incl. audio transcode to 128k MP3), `validate_export.py` (budget + no-leak asserts, §7); population scalars over all 139,241; `rules.py` re-score parity | 5.5 h |
| S2 | Phase 7 | Vite + TS + three scaffold, loaders, master clock, transport, stat strip (incl. overstrums/min) | 3 h |
| S3 | Phase 7 | Connectome panel: points, DataTexture lerp, silhouette, flow lines | 4 h |
| S4 | Phase 7 | CRT: 3D shell, RT highway, glass/scanline/vignette/glow shader | 4 h |
| S5 | Phase 7 | Retina lattice + GPU encoder **+ parity test vs. a Python reference trace** | 3 h |
| S6 | Phase 7 | DN decision bars + oscilloscope | 1.5 h |
| S7 | Phase 7 | Fly asset (§3), fret lights on controller | 3–8 h |
| S8 | Phase 7 | Audio sync, About modal, honesty copy, limitations, mobile fallback, 60 fps pass | 3.5 h |
| S9 | Phase 7 | Deploy, G7 verification | 1.5 h |

**≈ 29–34 h, 8–10 sessions.** S2–S6 don't depend on S7; the fly can be a
placeholder box until then.

---

## 7. Before-public checklist

### The path leak — status

Deferred per this session's decision (repo is private). Confirmed extent,
kept for when it matters:
- **Working tree:** `docs/library_report.md:3`, `docs/PROGRESS.md`.
- **History:** 2 commits (`1a9db62` near the root, `13e8ac0`) — a rewrite
  touches nearly all 37 commits and all 4 tags, and needs a force-push
  (already pushed to `github.com/jtursii/fly-hero`).
- `git-filter-repo` not installed (`brew install git-filter-repo`).

**Recorded rewrite plan (run only before any public release):**
```bash
git clone --mirror . ../fly-hero-backup.git   # backup first
brew install git-filter-repo
printf 'literal:/Users/joeyt/Music/Clone Hero/songs==><library root>\n' > /tmp/redact.txt
printf 'literal:/Users/joeyt==><home>\n' >> /tmp/redact.txt
git filter-repo --replace-text /tmp/redact.txt
git log --all -S'/Users/joeyt' --oneline      # must be empty
git push --force --all && git push --force --tags
```
Also fix `docs/library_report.md`/`docs/PROGRESS.md` in the working tree and
make `library/scan.py` emit a redacted root going forward, before that push.
**Action taken this session:** noted as an open issue in `docs/PROGRESS.md`
with this plan, per the user's instruction — no rewrite performed.

### No audio, no charts — updated for D37

`validate_export.py` (S1) must fail the build on:
- **any audio file under `web/` that is not one of the approved showcase
  songs' `audio.mp3`** (D37's exception is narrow — stems, non-showcase
  songs, or any file outside `web/public/audio/<showcase-id>.mp3` still fail)
- any `*.chart *.mid *.midi *.ini *.sng` under `web/`
- `web/public/data` total > 60 MB, or any song's data dir > 15 MB (audio is
  tracked and reported separately, not counted against this cap — §2)
- any absolute path (`/Users/`) inside any shipped JSON — `eval.json`
  contains absolute checkpoint paths today; the manifest writer must not
  copy them through
- D9's general rule (no audio, synthesized sounds) stays default for any
  *future* export outside this private-repo exception

---

## 8. Remaining ambiguities — my recommendation, one line each

1. ~~G4 checkpoint~~ — **resolved** (D38): never chosen by test score.
2. ~~Streak tile~~ — **resolved**: cut.
3. ~~Overstrums~~ — **resolved**: added.
4. ~~Fly asset~~ — **resolved**: stylized low-poly.
5. ~~Fret lights~~ — **resolved**: on the controller.
6. ~~Showcase songs~~ — **resolved**: candidate list in §1, awaiting your pick.
7. **Difficulty: Medium only, or mixed?** Recommend **Medium only** — it's
   the difficulty G4 actually measures, so every stat on the strip means the
   same thing across all showcase songs; mixing difficulties would need a
   per-song caveat on every number.
8. ~~Training-time label~~ — **resolved**: lineage number, labeled.
9. **D9 confirmation** — superseded by D37 for this private repo; D9 itself
   is unchanged as the public-release default.
10. ~~Retina derivation~~ — **resolved**: approved.
11. ~~History rewrite~~ — **resolved**: deferred, noted in PROGRESS.md.
12. **Anchor-position caveat: footnote only, or a visual cue?** Recommend a
    **visual cue** (anchor-positioned points rendered ~40% dimmer) plus a
    short footnote — `pos_source_id` is already in the exported data, so the
    dimming is free and more honest than prose alone.
13. **New, from this pass — audio format for the two `.ogg` sources**:
    re-encode to MP3 128k like the rest (recommended, for one consistent
    container/bitrate and zero cross-browser codec risk), or keep them as
    Opus/OGG and only standardize the `.mp3` sources? Recommend re-encoding
    everything to MP3 128k CBR — small quality loss on the two `.ogg`
    sources, in exchange for predictable size and universal `<audio>`
    support (some older Safari/iOS versions handle Opus-in-OGG poorly).
