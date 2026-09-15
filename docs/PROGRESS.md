# Progress

## Current phase
Phase 0 — done. Ready for Phase 1.

## Gate results
| Gate | Status | Numbers | Date |
|---|---|---|---|
| G0 | PASS | `uv run pytest -q`: 3 passed. `torch.backends.mps.is_available()` = True. Song folders: 652 (≥100 required). notes.chart: 306, notes.mid: 347, .sng: 0. | 2026-09-15 |

## Running jobs
(none)

## Log
<!-- newest first: date — task — outcome — deviations — repro command -->
- 2026-09-15 — Phase 0 scaffold + library inventory — PASS (G0) —
  - Deviations from PLAN.md: added `flyhero/utils/` (config.py, run.py) to the repo layout — not in the original diagram (logged as D10 in DECISIONS.md). `scan.py` walks the library recursively (it is not flat: songs sit one level under 5 setlist folders) and skips hidden/temp directories (found 1: `Guitar Hero - Metallica/.tmp.drivedownload`, an empty Google-Drive-sync artifact).
  - Cloud-storage check: 0 zero-byte files, 0 read errors in a 40-file content-read sample — the library is fully local, not cloud placeholders. No abort needed.
  - Library counts: 652 song folders (song.ini), 306 with notes.chart, 347 with notes.mid, 0 `.sng`, 1 folder with both chart and mid, 0 folders with neither.
  - Guitar-part availability: of 306 `.chart` files, all 306 have `[ExpertSingle]`, but only 8 each have `[EasySingle]`/`[MediumSingle]`/`[HardSingle]` (most custom charts in this library are Expert-only). Of 346 `.mid`-only songs, all 346 have a `PART GUITAR` track (required `mido.MidiFile(..., clip=True)` — 122 of them have out-of-range MIDI data bytes that mido rejects by default; this is a known quirk of game-ripped `.mid` files, not corruption). Overall: 652/652 songs have a detected guitar part in at least one difficulty.
  - Size breakdown: chart-related files (.chart/.mid/.ini) 49.7 MB, audio (.ogg/.mp3/.opus) 10.6 GB, video backgrounds (.webm) 446.7 MB.
  - Repro: `uv run pytest -q`; `uv run python -m flyhero.library.scan --config configs/paths.yaml` → `docs/library_report.md`.

## Open issues
- Most `.chart` files in this library only chart Expert difficulty (8/306 have Easy/Medium/Hard). The curriculum in Phase 3 (Easy → Medium → Hard → Expert) will have far fewer Easy/Medium/Hard training examples from `.chart` files specifically — worth revisiting when Phase 2/3 sizes the curriculum, though `.mid`-only songs may still cover other difficulties (not checked here, since that requires a full note-range parse, out of scope for Phase 0).
