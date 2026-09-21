# Fly Brain Hero — 1-Week Build Plan

**Outcome by Day 7:** A connectome-constrained fly brain model trained on the user's Clone Hero library plays held-out songs in a headless simulator. A Vercel-hosted three.js site replays 2–3 of them, showing brain activity in 3D next to the note highway.

**Scope for one week**
- In: song library ingest → connectome graph → simulator → brain → behavior cloning → one control → export → site.
- Out: PPO, the real Clone Hero game, spiking models, BANC, extensive ablations. ES fine-tuning is a stretch goal only.

**Plan change (2026-09-16, D34):** the shuffled-connectome control (`bc_full_shuffled`) is **deferred until after the demo site ships**. Current priority: (1) train the real brain toward Hard (`bc_full_real_v2`), (2) Phases 6–7. Phase 5's `docs/results.md` compares real brain vs GRU vs readout-only for now; the shuffled row is added when the control runs (setup pinned in D34).

**Time-critical path:** overnight training on Day 3 → 4 and Day 4 → 5. If earlier phases slip, cut features, not overnight runs.

Each phase has tasks, an acceptance **gate** (numbers go in PROGRESS.md), and **fallbacks**. Don't start the next phase until the gate passes or the user approves a fallback.

---

## Repo layout
```
fly-brain-hero/
├── CLAUDE.md
├── docs/{PLAN,PROGRESS,DECISIONS}.md
├── configs/                 # YAML configs; paths.yaml is gitignored
├── data/                    # gitignored: raw/, processed/
├── runs/                    # gitignored
├── scripts/status.sh
├── flyhero/
│   ├── utils/               # config.py run.py — shared config loader + run-dir helper
│   ├── library/             # scan.py
│   ├── connectome/          # download.py build_graph.py inspect.py shuffle.py
│   ├── game/                # song.py chart_parser.py midi_parser.py ingest.py rules.py labels.py render.py retina.py sim.py env.py
│   ├── brain/               # rate_model.py readout.py baseline_gru.py
│   ├── train/               # bc.py es.py (stretch)
│   ├── eval/                # evaluate.py
│   └── export/              # record.py export_web.py validate_export.py
├── bench/bench_brain.py
├── tests/fixtures/          # tiny hand-written charts only
└── web/                     # Vite + TS + three.js
```

---

## Phase 0 — Scaffold + library inventory (Day 1 morning, ~1.5 h)
**Tasks**
1. `uv init` with Python 3.12. Dependencies: torch, numpy, polars, pyarrow, pyyaml, mido, gymnasium, tensorboard, pytest, imageio[ffmpeg], tqdm.
2. Create the repo layout, `.gitignore` (data/, runs/, configs/paths.yaml, web/node_modules, web/dist), a config loader, a run-dir helper (timestamp, config copy, seed, git SHA), `scripts/status.sh`, and a smoke test.
3. Ask the user for the song library path. Write it to `configs/paths.yaml` as `song_library`.
4. `library/scan.py` (read-only walk) → `docs/library_report.md`, reporting:
   - Number of song folders.
   - Counts of `notes.chart`, `notes.mid`, `song.ini`, and `.sng` files.
   - Songs with no chart file.
   - Total size of chart files vs audio files.

**Gate G0**
- Tests pass.
- `torch.backends.mps.is_available()` recorded.
- Library report exists and ≥ 100 song folders have a `.chart` or `.mid` file.

**Fallback:** If most songs are `.sng` packages, **stop and ask the user**. Options: re-export or convert them to folder format, or add an `.sng` reader as an extra task.

---

## Phase 1 — Connectome graph (Day 1)
**Tasks**
1. `connectome/download.py`:
   - From Zenodo record 10676866: `proofread_connections_783.feather` (~850 MB), `proofread_root_ids_783.npy`, and the v783 neurotransmitter predictions.
   - From github.com/flyconnectome/flywire_annotations: the neuron annotations.
   - Skip files that already exist; verify file sizes.
2. `connectome/build_graph.py` → `data/processed/graph.npz`:
   - Sum synapse counts per (pre, post) across neuropil rows. Drop pairs below `min_syn` (default 5). Keep proofread neurons only. Map root_id → dense index.
   - `type_id`: `cell_type`, else `cell_class`, else a singleton type. Store the name lookup.
   - `sign` per edge from the presynaptic transmitter (D4). Report unknowns.
   - `pair_id` per edge, indexing unique (pre_type, post_type) pairs.
   - `inv_norm[post]` = 1 / total incoming synapses after thresholding.
   - Index sets: `input_idx` (visual input neurons, see step 3), `dn_idx` (`super_class == "descending"`), `t4t5_idx`.
   - `pos` per neuron (soma position if present; **record the units**) and a `super_class` id.
3. `connectome/inspect.py` → `docs/graph_report.md`:
   - Neuron count, edge count, type-pair count, DN count.
   - Photoreceptor coverage (R1–6, R7, R8) per side. **Choose the input layer:** photoreceptors if coverage is reasonable, otherwise lamina L1–L3. Record the choice in DECISIONS.md.
   - Whether column assignments exist.
4. `connectome/shuffle.py`: degree-preserving shuffle that swaps postsynaptic targets while keeping in-degree, out-degree, sign, and count. Writes `graph_shuffled.npz`.
5. Tests: index round-trip; shuffle preserves degrees.

**Gate G1**
- Proofread neurons ≈ 138.6k.
- Edge, type-pair, DN, and input counts recorded.
- Input neurons ≥ 500 and DNs ≥ 100.
- Shuffle test passes.

**Fallbacks**
- Type pairs > 300k → share gain by (pre_type, post_super_class).
- Unusable input layer → medulla columnar inputs (e.g. Mi1, Tm3).

---

## Phase 2 — Song ingest + game simulator (Day 2)
Design: a top-down highway with 5 vertical lanes. Notes scroll downward at constant speed to a strikeline near the bottom. Guitar parts only. The game runs at 60 Hz. The simulator is vectorized over a batch of songs.

**Tasks**
1. `game/song.py`: `Song(song_id, title, artist, charter, difficulty, notes, duration_s)`.
   - `notes` is an array of `(time_s, lane_mask: 5 bits, sustain_s, flags)`.
   - `song_id` = stable hash of the folder path relative to the library.
2. `game/chart_parser.py` for `.chart`:
   - `[Song]` `Resolution` (and `Offset`, recorded but not used for training).
   - `[SyncTrack]` tempo events `B` (BPM×1000). Convert ticks → seconds through the tempo map.
   - Sections `[EasySingle]`, `[MediumSingle]`, `[HardSingle]`, `[ExpertSingle]`.
   - `N 0–4` = frets. Same tick = chord. Sustain = note length.
   - `N 5` forced, `N 6` tap, `N 7` open → stored as flags.
3. `game/midi_parser.py` for `.mid` (mido):
   - Track `PART GUITAR`. Ticks per beat from the header; tempo map from set_tempo events.
   - Fret note numbers: Easy 60–64, Medium 72–76, Hard 84–88, Expert 96–100.
   - Same tick = chord.
   - Sustain only if length ≥ `sustain_min_beats` (config, default 0.5); otherwise 0.
   - Ignore star power, solo, forcing markers, and SysEx open/tap markers for v1. Record counts of what was ignored.
4. `game/ingest.py`:
   - Parse every song folder: prefer `notes.chart` when both files exist. Read `song.ini` for name/artist/charter.
   - Cache each (song, difficulty) to `data/processed/songs/<song_id>_<diff>.npz`.
   - Failures are logged to `docs/library_report.md`, never crash the run.
   - Dedupe by (artist, title).
   - Write `data/processed/splits.json`: split **by song_id** 80/10/10, seed 0 (invariant 5).
5. **v1 rule simplifications** (document them in the About modal later):
   - HOPO and tap notes are treated as strummed.
   - Open notes are dropped. Report the per-song open-note fraction; exclude songs with > 10% open notes.
6. `game/rules.py` (the only scoring implementation):
   - **Hit:** a strum frame while the held fret set equals the lane mask, within ±`hit_window_s` (config, default 0.07).
   - **Miss:** the note leaves the window without a hit.
   - **Overstrum:** a strum with no note in the window.
   - **Sustain credit:** held frames after a hit.
   - Metrics: `hit_rate`, `overstrums_per_min`, `sustain_frac`.
7. `game/labels.py`, per-frame BC targets:
   - `fret_k = 1` from `note_time − hit_window_s` until the note ends (including sustain).
   - `strum = 1` on the frame nearest `note_time`.
8. `game/render.py`: grayscale `[64, 64]` frame. Dark background, bright notes, dim lane lines, strikeline row. Lookahead ≈ 1.0 s (config), constant regardless of song tempo.
9. `game/retina.py`: frame → input currents at `input_idx`. **Time-box: 2 h.**
   - If column coordinates exist, use them. Otherwise project input-neuron positions onto their 2D principal plane per side, normalize to [0,1]², and sample the frame with a small Gaussian blur.
   - Both sides see the same screen, mirrored for the left.
   - Current = `gain * (intensity − EMA(intensity))`.
10. `game/sim.py` + `game/env.py`: `VecRhythmEnv(batch)`.
    - `reset(song_list)`; `step(actions[B,6]) → obs, reward, done, info`.
    - Reward: +1 hit, −1 miss, −0.3 overstrum, +0.01 per sustain frame.
    - Debug CLI `--video <song_id> <diff>` writes an mp4 of the highway, retina overlay, and events for the scripted player.
11. Tests (fixtures are tiny hand-written files in `tests/fixtures/`):
    - A `.chart` with two tempo changes and a chord.
    - A `.mid` built in the test with mido containing the same notes; it must parse to the **same** `Song` as the `.chart`.
    - Hit/miss/overstrum edge cases.
    - Split has no song in two splits.

**Gate G2**
- ≥ 90% of chart-bearing songs parse. ≥ 150 train songs. Counts per difficulty recorded.
- Scripted perfect player: `hit_rate = 1.000` with 0 overstrums on 20 val songs at Expert.
- Random player `hit_rate` recorded.
- Env at batch 64 runs ≥ 5× real time on CPU (amended from ≥10×, D21).
- **Human check:** the debug video of a song the user knows looks right.

**Fallback:** if a format parses poorly, drop that format for v1 as long as ≥ 150 train songs remain. Record which format was dropped.

---

## Phase 3 — Brain model + sanity baseline (Day 3; first overnight run)
**Tasks**
1. `brain/rate_model.py`, `ConnectomeBrain(graph, cfg)`:
   ```
   r      = clamp(relu(v), 0, 1)                                        # [B, N]
   w      = sign * softplus(g[gain_group_id]) * count * inv_norm[post]  # [E]
   syn    = zeros_like(v).index_add_(1, post, r[:, pre] * w)            # [B, N]
   v_next = v + (dt / tau[type_id]) * (-v + syn + b[type_id] + I_in)
   ```
   - `dt = 1/120` s (2 substeps per 60 Hz frame).
   - `tau` clamped ≥ 20 ms, initialized to 50 ms.
   - `softplus(g)` initialized ≈ `gain0` (config). `b` initialized to a small positive tonic baseline (`bias_init` config, D25 -- amended from this plan's original `b = 0`: with every photoreceptor's outgoing synapse forced inhibitory (D12) and no tonic activity for that inhibition to modulate via disinhibition, `b = 0` made it impossible for any non-photoreceptor neuron to ever reach `r > 0` at any `gain0`, proven and tested up to `gain0 = 10,000` -- see `docs/gain0_stability.md`).
   - `I_in` is nonzero only at `input_idx`.
   - **Gain sharing (D16, config `gain_sharing: fine | coarse | hybrid`, default `hybrid`)**: Phase 1's Gate G1 found 410,768 realized (pre_type, post_type) pairs, over the plan's 300k threshold. `gain_group_id` per edge is derived from `graph.npz`'s `pair_id`/`type_id`/`super_class_id` (no graph rebuild needed):
     - `fine`: `gain_group_id = pair_id` (410,768 params).
     - `coarse`: `gain_group_id` keyed by `(pre_type, post_super_class)` (22,212 params). Used for the Phase 5 ES stretch goal.
     - `hybrid` (default): a pair keeps its own `pair_id` if it has ≥ `k` edges, else it's remapped to the `coarse` bucket for its `(pre_type, post_super_class)`. `k` defaults to 2 (the largest threshold with fine-pair edge coverage ≥ 90%, per `docs/graph_report.md`), giving 244,824 fine + 18,925 coarse = 263,749 params.
2. `brain/readout.py`: `Linear(len(dn_idx), 6)` on DN rates averaged over each frame's substeps → logits (5 frets + strum).
3. `bench/bench_brain.py`:
   - Sweep device × dtype × batch {1, 8, 32} × `min_syn` {5, 10}.
   - Measure forward and forward+backward time for 1 game-second, and peak memory.
   - Write `docs/bench.md` and pick a device.
4. Stability sweep over `gain0` on 10 s of a real Medium song, untrained:
   - Fraction of neurons with r > 0.01 between 1% and 30%.
   - No NaN.
   - DN activity varies with input.
   - Record input→DN latency.
5. `brain/baseline_gru.py`: CNN on the rendered frame → GRU → 6 logits, trained by the same `bc.py`. This validates labels and the loop, not the fly.
6. `train/bc.py`:
   - Sample random (song, start time) windows from the **train** split at the current curriculum difficulty.
   - Burn-in 1 s under `no_grad`, then train on a 2 s window.
   - Loss: BCE on frets + BCE on strum with `pos_weight` (config, ≈ 20).
   - AdamW with separate learning rates for readout and brain parameters. Grad-clip 1.0. Optional per-frame gradient checkpointing.
   - Every K steps: evaluate on 10 **val** songs with `rules.py`, log, and checkpoint.
   - **Curriculum:** Easy → Medium → Hard → Expert. Advance when val `hit_rate` ≥ 0.8 at the current difficulty.

**Gate G3**
- Benchmark recorded.
- Chosen config does fwd+bwd for 1 game-second at batch 8 in ≤ 3 s wall-clock and ≤ 40 GB peak.
- Untrained activity is stable.
- GRU baseline reaches ≥ 0.95 `hit_rate` on Medium val songs within 1 h.

**Fallbacks** (apply in order until the speed gate passes; record each)
1. `min_syn = 10`
2. float16 where stable
3. Prune to neurons within 4 hops downstream of `input_idx` AND 4 hops upstream of `dn_idx`
4. Batch 4, window 1.5 s
5. Decision rate 30 Hz (brain still at dt = 1/120)

If the GRU fails G3, fix labels, rules, or the loop before touching the brain.

**End of Day 3:** launch the overnight run `bc_readout_only` (brain frozen; the reservoir control), after a 5-minute smoke test of the full model.

---

## Phase 4 — Train the connectome (Day 4; second overnight run)
**Tasks**
1. Review `bc_readout_only`. Record val `hit_rate` per difficulty.
2. Train the full model (readout + `tau` + `b` + `g`), initialized from the readout-only checkpoint. Run in the background; monitor via `scripts/status.sh`.
3. Write `eval/evaluate.py`: runs a checkpoint on **all test-split songs** at every difficulty and writes `eval.json` with per-song and aggregate metrics.
4. **Evening:** launch the overnight runs `bc_full_real` (best config) and `bc_full_shuffled` (identical config, `graph_shuffled.npz`). Run them sequentially if memory is tight. **D34: `bc_full_shuffled` deferred until after Phase 7** — run it later with the pinned setup in DECISIONS.md D34 (only change: `graph_file: "graph_shuffled.npz"` in `configs/brain.yaml`).

**Gate G4:** full real model has test `hit_rate` ≥ 0.70 on Medium by Day 5 morning.

**Fallbacks** (at most two on Day 4, one change at a time)
- Raise `pos_weight`
- Burn-in 2 s
- Lookahead 1.5 s
- Nudge `gain0` up
- **Needs approval (invariant 2):** readout from DNs + their direct presynaptic partners

If the gate still fails, showcase at Easy and document it honestly.

---

## Phase 5 — Evaluate + control + stretch (Day 5)
**Tasks**
1. Run `evaluate.py` on readout-only, full real, and the GRU baseline (D34: full shuffled deferred; add its row after Phase 7). Write `docs/results.md` with a table (test `hit_rate` and overstrums/min per difficulty) plus two sentences of honest interpretation.
2. Choose 3 **test-split** showcase songs (never trained on) at the highest difficulty where the model's `hit_rate` ≥ 0.70. Prefer songs the user likes and that are ≤ 4 min. **The user picks the final set.**
3. **Stretch (only if on schedule): ES fine-tune** in `train/es.py`.
   - Mirrored-sampling OpenAI-ES over brain parameters + readout. Population 32, σ = 0.02.
   - Gain sharing forced to `coarse` (D16) regardless of what BC trained with — fewer params (22,212) to keep the population/fitness-evaluation budget tractable.
   - Fitness = full-song score from `rules.py` on train songs.
   - Time-box to 4 h. Keep the result only if test `hit_rate` improves.

**Gate G5:** `docs/results.md` has the real / GRU / readout-only rows (the shuffled row follows after Phase 7, D34); showcase songs are recorded in PROGRESS.md.

---

## Phase 6 — Record + export replays (Day 6)
**Tasks**
1. `export/record.py`: run the showcase checkpoint on each showcase song. Per 60 Hz frame, record:
   - Neuron rates
   - Retina input
   - Logits
   - Actions
   - Events
2. `export/export_web.py` → `web/public/data/`:
   - `brain.json`: neuron count, exported neuron slots, super_class names, retina display coordinates, FlyWire attribution.
   - `positions.bin`: Float32 `[N, 3]`, centered, unit radius, all neurons.
   - `classes.bin`: Uint8 `[N]`.
   - Per song, `songs/<id>/`:
     - `manifest.json`: title, artist, charter, difficulty, fps, frames, notes (time, lanes, sustain), `hit_rate`, audio offset from `song.ini`/`.chart`.
     - `activity.bin`: Uint8 `[F, K]` at 20 fps.
     - `retina.bin`: Uint8 `[F, C]` at 20 fps.
     - `actions.bin`: Uint8 `[F60]` bitmask.
     - `events.json`.
   - K = input neurons + DNs + the top 15k neurons by variance across all showcase songs (the same set for every song).
   - **Size budget:** ≤ 15 MB per song, ≤ 60 MB total.
   - **No audio and no original chart files are exported (D9).**
   - Optional, only if < 1 h: decimated neuropil mesh via navis/flybrains → `neuropils.glb`.
3. `export/validate_export.py`:
   - Re-score exported actions against manifest notes with `rules.py`; they must match the recorded `hit_rate`.
   - Assert size budgets.
   - Assert no audio files exist under `web/`.

**Gate G6:** validation passes; total size recorded.

---

## Phase 7 — three.js site + deploy (Day 7)
**After Phase 7 ships (D34):** run the deferred `bc_full_shuffled` control with the pinned setup, add its row to `docs/results.md`.

**Before starting:** confirm D9 with the user.

**Tasks**
1. `web/`: Vite vanilla-ts, `three`, `tone`.
2. **Layout:**
   - Left (~65%): 3D brain.
   - Right top: note highway canvas.
   - Right bottom: fly's-eye retina inset.
   - Bottom bar: song picker, play/pause, scrubber, live hit counter, "About".
3. **Brain:**
   - One `THREE.Points` for all neurons, dim base color by super_class.
   - Exported neurons get an `aSlot` attribute.
   - Current and next activity frames live in two `DataTexture`s. The vertex shader interpolates between them and sets size and brightness.
   - Additive blending, `UnrealBloomPass`, `OrbitControls` with slow auto-rotate.
   - Toggle to highlight inputs and DNs.
4. **Highway:** 2D canvas driven by manifest notes at the playhead. Frets light from `actions.bin`; hits and misses flash.
5. **Retina inset:** points at display coordinates, brightness from `retina.bin`.
6. **Audio (per D9):**
   - Tone.js `PluckSynth` with one pitch per lane, played on hit events. A muted thud on misses. Starts only after a user click.
   - Optional, time-box 1 h: a "Load your own audio file" picker. The chosen file plays locally in the browser, synced using the manifest's audio offset, and is never uploaded.
7. **Master clock:** a single playhead in seconds drives every view; scrubbing seeks all of them.
8. **Performance:** fetch each song's `data/songs/<id>/` **only when that song is selected** (D45 makes this load-bearing, not an optimization: per-song size is up to 21 MB, so initial load must pull only `brain.json` + `positions.bin`/`classes.bin`/`pos_source.bin`). 60 fps on an M-series laptop. On mobile, reduce point size and disable bloom.
9. **About modal:**
   - Plain-language explanation.
   - FlyWire citation (Dorkenwald et al. 2024; Schlegel et al. 2024; CC BY 4.0) and charter credits.
   - Results table.
   - Limitations: abstract rate model, learned readout, no neuromodulation or gap junctions, simulated game, HOPO/taps treated as strums, open notes removed.
   - Not affiliated with Clone Hero or Guitar Hero.
10. **Deploy:**
    - Push to GitHub.
    - Import into Vercel with root `web`, preset Vite, output `dist`.
    - `vercel.json` with long cache headers for `/data/*` and `/audio/*` (written in Phase 6).
    - **Verify compression on the first deploy** (D45): `curl -sI -H 'Accept-Encoding: br, gzip' <url>/data/songs/<id>/activity.bin | grep -i content-encoding`. Vercel auto-compresses by content type and `application/octet-stream` may not be included; compression cannot be forced from `vercel.json` (a `Content-Encoding` header without a compressed body would be a lie). If the header is absent, ship pre-compressed `.gz` siblings and decompress with `DecompressionStream('gzip')` in the client. Gzipped, the largest song is 10.9 MB vs 20.4 MB raw.

**Gate G7**
- Production URL loads with no console errors.
- All showcase songs play with brain, highway, and retina in sync.
- Scrubbing works.
- 60 fps on desktop.
- URL recorded in PROGRESS.md.

---

## Day-by-day
| Day | Phases | Overnight |
|---|---|---|
| 1 | 0, 1 | — |
| 2 | 2 | (optional) full library ingest if slow |
| 3 | 3 | `bc_readout_only` |
| 4 | 4 | `bc_full_real` (`bc_full_shuffled` deferred until after Day 7, D34) |
| 5 | 5 (+ ES stretch) | — |
| 6 | 6 | — |
| 7 | 7 | — |
