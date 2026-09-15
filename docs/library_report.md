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

### From .mid track names, songs with .mid but no .chart (346 files)
- Songs with a `PART GUITAR` track: 346
- Parse errors: 0
- Per-difficulty breakdown not available for .mid-only songs without a full note-range parse (out of scope for this scan).

**Overall songs with a detected guitar part: 652 / 652**

## Size breakdown

- Chart-related files (.chart/.mid/.ini): 49.7 MB
- Audio files (.ogg/.mp3/.opus): 10607.4 MB
- Video background files (.webm): 446.7 MB
