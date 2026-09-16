# Fly Brain Hero

A rate-based model of the adult fruit fly brain, wired from the FlyWire v783 connectome, learns to play Clone Hero–style rhythm-game charts in a headless simulator, seeing the game only through a biologically-oriented retina encoder built from real photoreceptor anatomy. Training data is the user's own local Clone Hero song library. The end goal is a three.js site replaying a handful of showcase songs with the brain's activity visualized in 3D next to the note highway.

## Status

Phases 0–2 are done. See [`docs/PROGRESS.md`](docs/PROGRESS.md) for full gate numbers, deviations, and repro commands; locked design decisions are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

| Phase | Gate | One-line result |
|---|---|---|
| 0 — Library inventory | G0 | 652 song folders found (≥100 required); no `.sng` packages, no fallback needed. |
| 1 — Connectome graph | G1 | 139,241 proofread neurons, 2.7M edges, 11,118-neuron photoreceptor input layer, 1,303 descending neurons. |
| 2 — Song ingest + game sim | G2 | 568 songs cached (453/55/60 train/val/test); scripted-perfect-player `hit_rate` 1.000 on 25 Expert songs (20 random + the 5 densest); env throughput 6.8× real-time at batch 64 (≥5×, amended from ≥10×, D21); retina retinotopy validated (matched-pair 1.63 / null-baseline 22.64 / superposition 1.437, all against required thresholds). |

Currently starting Phase 3 (brain model + sanity baseline).

## Setup

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/), Python 3.12, `ffmpeg` (for debug video rendering), Node.js (for the `web/` site, later phases).

```bash
uv sync
```

You'll need your own local Clone Hero song library (this repo never ships song charts or audio — see [Data and credits](#data-and-credits)). Create `configs/paths.yaml` (gitignored) pointing at it:

```yaml
song_library: "/path/to/your/Clone Hero/songs"
```

Then, in order:

```bash
# Connectome: download FlyWire v783 data and build the trainable graph
uv run python -m flyhero.connectome.download --config configs/connectome.yaml
uv run python -m flyhero.connectome.build_graph --config configs/connectome.yaml
uv run python -m flyhero.connectome.inspect --config configs/connectome.yaml   # -> docs/graph_report.md
uv run python -m flyhero.connectome.shuffle --config configs/connectome.yaml   # degree-preserving control

# Song library: scan, then ingest into per-song/difficulty caches + splits
uv run python -m flyhero.library.scan --config configs/paths.yaml             # -> docs/library_report.md
uv run python -m flyhero.game.ingest --config configs/game.yaml

# Fast tests (must pass before every commit)
uv run pytest -q
```

### Rendering a debug video

`sim.py`'s `--video` CLI renders a 3-panel debug video (note highway | fret targets vs. scripted-player actions with hit/miss/overstrum markers | retina input at each photoreceptor's image position) for one cached `(song_id, difficulty)`, optionally muxing in the song's own audio:

```bash
PYTHONPATH=. uv run python -m flyhero.game.sim \
  --config configs/game.yaml \
  --video <song_id> <difficulty> \
  --out media/debug/example.mp4 \
  --mux-audio-from "<song folder rel_path under song_library>" \
  --audio-mix guitar   # default: guitar stem only. "full": every stem present, mixed equally.
```

`--audio-mix guitar` picks the `guitar.*` stem by filename (falling back to `song.*` with a printed warning if no guitar stem exists — never a silent substitution of some other stem). Rendered videos are written under `media/` or `runs/`, both gitignored — audio is never committed and never placed under `web/` (D22).

## Repo layout

```
flyhero/
├── connectome/   # download, build_graph, inspect, shuffle (Phase 1)
├── library/      # read-only song-library scan (Phase 0)
├── game/         # song model, .chart/.mid parsers, ingest, rules, labels,
│                 # render, retina, sim/env (Phase 2)
└── utils/        # config loader, run-dir helper, text decoding
configs/          # YAML configs; paths.yaml is gitignored
docs/             # PLAN.md, PROGRESS.md, DECISIONS.md, graph_report.md, library_report.md
bench/, scripts/  # throughput benchmarks, gate-check scripts
tests/            # pytest suite (tiny hand-built fixtures + skip-if-absent real-data checks)
web/              # three.js replay site (Phase 7, not started)
```

- [`docs/PLAN.md`](docs/PLAN.md) — the full phase-by-phase build plan, gates, and fallbacks.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — append-only log of locked design decisions.
- [`docs/graph_report.md`](docs/graph_report.md) — connectome graph statistics (Phase 1).
- [`docs/library_report.md`](docs/library_report.md) — song library inventory and ingest report (Phases 0 & 2).

## Data and credits

The connectome is [FlyWire](https://flywire.ai/) FAFB v783: Dorkenwald et al. 2024, *Nature*; Schlegel et al. 2024, *Nature*. Licensed CC BY 4.0.

Song charts and audio are **not included** in this repository — users supply their own local Clone Hero library, which is treated as read-only and never committed.

## Limitations

The brain is an abstract per-cell-type rate model (no spiking, no neuromodulation, no gap junctions), not a full biophysical simulation. Actions come from a linear readout trained by behavior cloning, not the fly's own motor circuits. The "game" is a simplified rhythm-game simulator, not real Clone Hero. HOPO/tap notes are treated as ordinary strums and open notes are dropped entirely (v1 simplifications, see `docs/PLAN.md`).
