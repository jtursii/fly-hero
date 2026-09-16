# Fly Brain Hero

A rate-based model of the adult fruit fly brain, wired from the FlyWire v783 connectome, learns to play Clone Hero–style charts in a headless simulator. Training data is the user's own local Clone Hero song library. At the end, 2–3 replays are exported to a three.js site on Vercel.

- **Plan:** `docs/PLAN.md`
- **Current state:** `docs/PROGRESS.md`
- **Locked decisions:** `docs/DECISIONS.md`

At the start of every session, read `docs/PROGRESS.md` and only the current phase of `docs/PLAN.md`.

## Invariants (never violate; ask before bending any of them)
1. The brain sees the game **only** through the retina encoder, which injects current into the designated visual input neurons. No chart data, labels, lane ids, or note timings go anywhere else in the brain or readout.
2. Actions come **only** from a linear readout (weights + bias) of descending-neuron (DN) activity. Any change to the readout's inputs requires the user's approval and an entry in DECISIONS.md.
3. Connectome topology, synapse counts, and edge signs are fixed. Trainable: per-cell-type `tau` and `bias`, per type-pair gain `g` (softplus, so always positive), and the readout.
4. Game rules and scoring have one implementation (`flyhero/game/rules.py`), used for training, evaluation, and export.
5. Train/val/test splits are **by song**. All difficulties of a song live in one split. Evaluation and showcase songs come from the test split.
6. Controls (shuffled connectome, readout-only) use the identical code path, config, and training budget as the real model.
7. Never fake, hand-edit, or silently "fix" results, and never weaken, skip, or delete tests. If an acceptance gate fails: stop, record it in PROGRESS.md, propose a fallback from the plan, and ask.
8. The song library is **read-only**: never write, move, rename, or delete anything in it. Song audio may be copied out of it into `media/` for local use (e.g. muxing into a debug video); `media/` is gitignored, audio is never committed to git, and it never goes under `web/` (Phase 7 still follows D9 unless the user changes it) (D22).

## Paths
- Song library: `song_library` in `configs/paths.yaml` (gitignored; created in Phase 0 from the path the user gives).
- `data/` (raw + processed), `runs/`, and `media/` are gitignored.

## Stack
- Python 3.12 via `uv`; PyTorch (MPS or CPU, chosen by benchmark); mido; polars/pyarrow; gymnasium; pytest; TensorBoard.
- Web: `web/` with Vite + TypeScript + three.js + Tone.js, deployed to Vercel.
- Hardware: M5 Max, 64 GB unified memory. Keep training peak memory ≤ 40 GB.

## Commands
- `uv run pytest -q` — fast tests; must pass before every commit.
- `uv run python -m flyhero.<package>.<module> --config configs/<name>.yaml` — every entry point takes a config.
- `scripts/status.sh <run_dir>` — ≤15-line run summary. Use this; never tail raw logs into context.
- `cd web && npm run dev` / `npm run build`

## Conventions
- Every run writes to `runs/<timestamp>_<name>/` with a copy of its config, the seed, and the git SHA.
- Jobs longer than 2 minutes run in the background via `nohup`, logging to their run dir.
- Never re-download or re-parse data that already exists in `data/processed/`.
- Small commits, one per task. Tag each finished phase `phase-N-done`.
- Type hints on public functions; comment tensor shapes, e.g. `# [B, N]`.
- Push to origin after every commit that completes a task.

## End of every task
Update `docs/PROGRESS.md` with what was done, gate numbers, deviations, open issues, and repro commands. Append new decisions to `docs/DECISIONS.md`. Commit.

## Compaction
When compacting, preserve: current phase and task, gate results with numbers, files modified, running jobs with commands and run dirs, and unresolved failures.
