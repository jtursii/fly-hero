# Progress

## Current phase
Phase 0 — done. Ready for Phase 1.

## Gate results
| Gate | Status | Numbers | Date |
|---|---|---|---|
| G0 | PASS | `uv run pytest -q`: 3 passed. `torch.backends.mps.is_available()` = True. Song folders: 652 (≥100 required). notes.chart: 306, notes.mid: 347, .sng: 0. | 2026-09-15 |
| Native-difficulty check (pre-Phase-1) | PASS, no plan change | Songs with native Easy AND Medium: 293 (≥150 threshold). | 2026-09-15 |

## Running jobs
(none)

## Log
<!-- newest first: date — task — outcome — deviations — repro command -->
- 2026-09-15 — Pre-Phase-1: per-difficulty note-range check on `.mid`-only songs — decision rule satisfied, no plan change —
  - Extended `flyhero/library/scan.py` to fully parse `PART GUITAR` note ranges (Easy 60-64, Medium 72-76, Hard 84-88, Expert 96-100) for the 346 `.mid`-only songs (`mid_guitar_difficulties`), and combined with the existing `.chart`-section counts into per-difficulty **song** counts (not file counts): Easy 294, Medium 294, Hard 294, Expert 652.
  - Decision rule (user-specified): amend Phase 2 to add `game/reduce.py` (derive Easy/Medium/Hard from Expert) only if fewer than 150 songs have **native** Easy AND Medium. Actual: **293** songs have both natively — rule satisfied, **no PLAN.md change**. (Native Easy/Medium coverage in this library comes almost entirely from `.mid` files — only 8 of 306 `.chart` files have non-Expert sections — but that's more than enough real lower-difficulty data for the Phase 3 curriculum.)
  - Clip-corruption impact check (`check_clip_corruption_note_impact`): of 347 `.mid` files, 123 need `clip=True` to parse at all (122 `.mid`-only + the 1 song with both `.chart` and `.mid`). Traced each failure to its exact origin in mido's parser: **all 123 fail inside `read_sysex`, 0 inside `read_message`** — the out-of-range bytes are confined to proprietary SysEx payloads (e.g. Harmonix chart metadata), never inside a `note_on`/`note_off` message. So **0 clamped values land in the 60-100 guitar note range**; `clip=True` has no effect on note/difficulty detection accuracy for this library.
  - Repro: `uv run pytest -q`; `uv run python -m flyhero.library.scan --config configs/paths.yaml` → `docs/library_report.md`.
- 2026-09-15 — Phase 0 scaffold + library inventory — PASS (G0) —
  - Deviations from PLAN.md: added `flyhero/utils/` (config.py, run.py) to the repo layout — not in the original diagram (logged as D10 in DECISIONS.md). `scan.py` walks the library recursively (it is not flat: songs sit one level under 5 setlist folders) and skips hidden/temp directories (found 1: `Guitar Hero - Metallica/.tmp.drivedownload`, an empty Google-Drive-sync artifact).
  - Cloud-storage check: 0 zero-byte files, 0 read errors in a 40-file content-read sample — the library is fully local, not cloud placeholders. No abort needed.
  - Library counts: 652 song folders (song.ini), 306 with notes.chart, 347 with notes.mid, 0 `.sng`, 1 folder with both chart and mid, 0 folders with neither.
  - Guitar-part availability: of 306 `.chart` files, all 306 have `[ExpertSingle]`, but only 8 each have `[EasySingle]`/`[MediumSingle]`/`[HardSingle]` (most custom charts in this library are Expert-only). Of 346 `.mid`-only songs, all 346 have a `PART GUITAR` track (required `mido.MidiFile(..., clip=True)` — 122 of them have out-of-range MIDI data bytes that mido rejects by default; this is a known quirk of game-ripped `.mid` files, not corruption). Overall: 652/652 songs have a detected guitar part in at least one difficulty.
  - Size breakdown: chart-related files (.chart/.mid/.ini) 49.7 MB, audio (.ogg/.mp3/.opus) 10.6 GB, video backgrounds (.webm) 446.7 MB.
  - Repro: `uv run pytest -q`; `uv run python -m flyhero.library.scan --config configs/paths.yaml` → `docs/library_report.md`.

## Open issues
(none — the `.chart`-only Expert skew noted after Phase 0 was resolved by the native-difficulty check above: 293 songs have native Easy+Medium once `.mid` files are included, well over the 150 threshold, so no curriculum-data risk.)
