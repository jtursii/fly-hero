# Song library report

Library root: `/Users/joeyt/Music/Clone Hero/songs`

Song folders found: **652**

Hidden/temp directories skipped: **1** (e.g. Guitar Hero - Metallica/.tmp.drivedownload)


## File-type counts (per song folder, not per file)

- song.ini: 652
- notes.chart: 306
- notes.mid: 347
- .sng: 0

## Cloud / local-file status

- Chart/mid files checked (size via stat): 653
- Zero-byte files: 0 (0.0%)
- Content-read sample: 40 files, 0 read errors (0.0%)
- Cloud-placeholder pattern suspected: **False**

## Guitar-part availability

Folders with **both** notes.chart and notes.mid (chart preferred at ingest): 1
Folders with **neither** notes.chart nor notes.mid: 0

### From .chart section headers (306 files)
- Songs with at least one [*Single] section: 306
  - EasySingle: 8
  - MediumSingle: 8
  - HardSingle: 8
  - ExpertSingle: 306

### From .mid note ranges, songs with .mid but no .chart (346 files)
- Songs with a `PART GUITAR` track: 346
- Songs with at least one note in a difficulty's fret range: 346
  - Easy (notes in range): 286
  - Medium (notes in range): 286
  - Hard (notes in range): 286
  - Expert (notes in range): 346
- Parse errors: 0

### Combined per-difficulty song counts (native chart section, or .mid notes in the difficulty's fret range when there's no .chart -- songs, not files)
- Easy: 294
- Medium: 294
- Hard: 294
- Expert: 652

**Overall songs with a detected guitar part: 652 / 652**

**Songs with native Easy AND Medium: 293** (decision-rule threshold: 150 -- see docs/DECISIONS.md)

### Clip-corruption note impact (123 .mid files needed `clip=True` to parse at all)
- Out-of-range byte inside a SysEx payload (never a note number): 123
- Out-of-range byte inside a note_on/note_off (or other channel) message: 0
- Other/unresolved: 0
- Of the files needing clipping, **0 had a clamped value land in the 60-100 guitar note range** -- every clamp happens inside proprietary SysEx metadata (e.g. Harmonix chart extensions), never inside a note_on/note_off message. clip=True therefore has no effect on note/difficulty detection accuracy for this library.

## Size breakdown

- Chart-related files (.chart/.mid/.ini): 49.7 MB
- Audio files (.ogg/.mp3/.opus): 10607.4 MB
- Video background files (.webm): 446.7 MB
