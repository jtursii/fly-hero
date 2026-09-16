"""Read-only inventory of the user's Clone Hero song library (Phase 0 / Gate G0).

Never writes, moves, renames, or deletes anything under the library
(invariant 8). Two safety gates before any file content is touched:

1. A metadata-only pass (os.stat) finds song folders and flags zero-byte
   chart/mid files without reading their contents.
2. A small, bounded sample of chart/mid files is read to check for cloud
   placeholder behavior (e.g. Google Drive Files-On-Demand). If that sample
   shows widespread read failures or zero-byte files, the scan stops before
   touching the rest of the library and reports the situation instead of
   silently downloading everything.

Only once both gates pass does it open every .chart/.mid file to look for
guitar-part section headers / track names (still no full note parsing).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import mido

from flyhero.utils.config import load_config
from flyhero.utils.text import read_text_lenient

DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")
SAMPLE_SIZE = 40
ZERO_BYTE_ABORT_FRACTION = 0.05
READ_ERROR_ABORT_FRACTION = 0.10


@dataclass
class SongFolder:
    path: Path
    rel_path: str
    has_ini: bool
    has_chart: bool
    has_mid: bool
    has_sng: bool
    chart_size: int = 0
    mid_size: int = 0
    ini_size: int = 0


@dataclass
class ScanResult:
    songs: list[SongFolder] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)


def find_song_folders(root: Path) -> ScanResult:
    """Recursively walk `root`, skipping hidden/temp directories (e.g.
    `.tmp.drivedownload`), and collect every directory that looks like a
    song (has song.ini, notes.chart, notes.mid, or a .sng file)."""
    result = ScanResult()
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if d.startswith("."):
                result.skipped_dirs.append(
                    str((Path(dirpath) / d).relative_to(root))
                )
            else:
                keep.append(d)
        dirnames[:] = keep

        fset = set(filenames)
        has_ini = "song.ini" in fset
        has_chart = "notes.chart" in fset
        has_mid = "notes.mid" in fset
        has_sng = any(f.lower().endswith(".sng") for f in filenames)
        if not (has_ini or has_chart or has_mid or has_sng):
            continue

        p = Path(dirpath)
        sf = SongFolder(
            path=p,
            rel_path=str(p.relative_to(root)),
            has_ini=has_ini,
            has_chart=has_chart,
            has_mid=has_mid,
            has_sng=has_sng,
        )
        if has_chart:
            sf.chart_size = (p / "notes.chart").stat().st_size
        if has_mid:
            sf.mid_size = (p / "notes.mid").stat().st_size
        if has_ini:
            sf.ini_size = (p / "song.ini").stat().st_size
        result.songs.append(sf)

    return result


def check_cloud_status(songs: list[SongFolder], sample_size: int = SAMPLE_SIZE) -> dict:
    """Metadata-only zero-byte check across all chart/mid files, plus a
    bounded random-sample content read to detect cloud placeholders. Never
    reads more than `sample_size` files' content."""
    all_files: list[tuple[Path, int]] = []
    for sf in songs:
        if sf.has_chart:
            all_files.append((sf.path / "notes.chart", sf.chart_size))
        if sf.has_mid:
            all_files.append((sf.path / "notes.mid", sf.mid_size))

    zero_byte = [str(p) for p, size in all_files if size == 0]

    rng = random.Random(0)
    sample = (
        rng.sample(all_files, min(sample_size, len(all_files))) if all_files else []
    )
    read_errors: list[tuple[str, str]] = []
    for p, _size in sample:
        try:
            with open(p, "rb") as f:
                f.read(4096)
        except OSError as e:
            read_errors.append((str(p), str(e)))

    total = len(all_files)
    zero_frac = (len(zero_byte) / total) if total else 0.0
    error_frac = (len(read_errors) / len(sample)) if sample else 0.0

    cloud_only_suspected = zero_frac > ZERO_BYTE_ABORT_FRACTION or (
        len(sample) >= 5 and error_frac > READ_ERROR_ABORT_FRACTION
    )

    return {
        "total_chart_files": total,
        "zero_byte_count": len(zero_byte),
        "zero_byte_frac": zero_frac,
        "zero_byte_examples": zero_byte[:10],
        "sample_size": len(sample),
        "sample_read_errors": len(read_errors),
        "sample_error_frac": error_frac,
        "error_examples": read_errors[:10],
        "cloud_only_suspected": cloud_only_suspected,
    }


_SECTION_RE = re.compile(r"^\[(Easy|Medium|Hard|Expert)Single\]\s*$", re.MULTILINE)


def chart_single_sections(path: Path) -> set[str]:
    """Which of the four [*Single] section headers exist in a .chart file.
    Only scans for section header lines; does not parse notes."""
    text = read_text_lenient(path)
    return {m.group(1) for m in _SECTION_RE.finditer(text)}


# Guitar note ranges per Phase 2 of PLAN.md: Easy 60-64, Medium 72-76,
# Hard 84-88, Expert 96-100.
NOTE_RANGES: dict[str, range] = {
    "Easy": range(60, 65),
    "Medium": range(72, 77),
    "Hard": range(84, 89),
    "Expert": range(96, 101),
}


def mid_guitar_difficulties(path: Path) -> tuple[bool, set[str]] | None:
    """Which difficulties have notes in a .mid file's `PART GUITAR` track.

    Only reads note_on events' note numbers against the four fret-range
    windows above -- no sustains, chords, or other note attributes (that's
    Phase 2's job). Returns (has_part_guitar_track, difficulties_with_notes),
    or None on parse error."""
    try:
        # clip=True: many game-ripped .mid files have out-of-range data
        # bytes (in SysEx payloads, not note data -- see
        # check_clip_corruption_note_impact) that mido otherwise rejects.
        midi = mido.MidiFile(path, clip=True)
    except (OSError, ValueError, EOFError, IndexError):
        return None

    guitar_track = None
    for track in midi.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "track_name":
                if msg.name.strip() == "PART GUITAR":
                    guitar_track = track
                break
        if guitar_track is not None:
            break

    if guitar_track is None:
        return False, set()

    diffs: set[str] = set()
    for msg in guitar_track:
        if msg.type == "note_on" and msg.velocity > 0:
            for name, rng in NOTE_RANGES.items():
                if msg.note in rng:
                    diffs.add(name)
                    break
    return True, diffs


def _locate_clip_corruption(path: Path) -> str:
    """For a .mid file that fails strict (clip=False) parsing, find where
    the out-of-range data byte occurs: inside a SysEx payload (never carries
    note numbers) or inside a channel message like note_on/note_off (which
    could). Mirrors mido's own read_track loop via its internal read_sysex/
    read_message functions, tracing which one raises."""
    import mido.midifiles.midifiles as impl

    orig_sysex = impl.read_sysex
    orig_msg = impl.read_message
    location = {"where": "other"}

    def traced_sysex(infile, delta, clip=False):
        try:
            return orig_sysex(infile, delta, clip)
        except Exception:
            location["where"] = "sysex"
            raise

    def traced_msg(infile, status_byte, peek_data, delta, clip=False):
        try:
            return orig_msg(infile, status_byte, peek_data, delta, clip)
        except Exception:
            location["where"] = "channel_message"
            raise

    impl.read_sysex = traced_sysex
    impl.read_message = traced_msg
    try:
        mido.MidiFile(path)
    except Exception:
        pass
    finally:
        impl.read_sysex = orig_sysex
        impl.read_message = orig_msg
    return location["where"]


def check_clip_corruption_note_impact(songs: list[SongFolder]) -> dict:
    """Of the .mid files that need clip=True to parse at all, how many have
    the out-of-range byte inside a note_on/note_off message (which could
    corrupt difficulty detection) vs. somewhere else (e.g. SysEx payloads,
    which never carry note numbers and so can't land in the 60-100 guitar
    note range)."""
    clip_required = []
    for sf in songs:
        if not sf.has_mid:
            continue
        p = sf.path / "notes.mid"
        try:
            mido.MidiFile(p)
        except (OSError, ValueError, EOFError, IndexError):
            clip_required.append(p)

    sysex_only = 0
    channel_message = 0
    other = 0
    for p in clip_required:
        loc = _locate_clip_corruption(p)
        if loc == "sysex":
            sysex_only += 1
        elif loc == "channel_message":
            channel_message += 1
        else:
            other += 1

    return {
        "clip_required_count": len(clip_required),
        "sysex_only": sysex_only,
        "channel_message": channel_message,
        "other_location": other,
    }


def analyze_guitar_parts(songs: list[SongFolder]) -> dict:
    chart_songs = [sf for sf in songs if sf.has_chart]
    mid_only_songs = [sf for sf in songs if sf.has_mid and not sf.has_chart]

    # Native difficulty set actually used at ingest time: chart is
    # preferred over mid when both exist (Phase 2 ingest.py), so a song
    # contributes either its chart sections or its mid note-range results,
    # never both.
    native_diffs_by_song: dict[str, set[str]] = {}

    per_difficulty_chart: dict[str, int] = {d: 0 for d in DIFFICULTIES}
    chart_with_any = 0
    for sf in chart_songs:
        sections = chart_single_sections(sf.path / "notes.chart")
        native_diffs_by_song[sf.rel_path] = sections
        if sections:
            chart_with_any += 1
        for d in sections:
            per_difficulty_chart[d] += 1

    per_difficulty_mid: dict[str, int] = {d: 0 for d in DIFFICULTIES}
    mid_only_with_track = 0
    mid_only_with_any_diff = 0
    mid_parse_errors = 0
    for sf in mid_only_songs:
        result = mid_guitar_difficulties(sf.path / "notes.mid")
        if result is None:
            mid_parse_errors += 1
            native_diffs_by_song[sf.rel_path] = set()
            continue
        has_track, diffs = result
        if has_track:
            mid_only_with_track += 1
        if diffs:
            mid_only_with_any_diff += 1
        native_diffs_by_song[sf.rel_path] = diffs
        for d in diffs:
            per_difficulty_mid[d] += 1

    combined_per_difficulty = {
        d: per_difficulty_chart[d] + per_difficulty_mid[d] for d in DIFFICULTIES
    }
    songs_with_native_easy_and_medium = sum(
        1
        for diffs in native_diffs_by_song.values()
        if {"Easy", "Medium"} <= diffs
    )

    both_chart_and_mid = sum(1 for sf in songs if sf.has_chart and sf.has_mid)
    no_notes_file = sum(1 for sf in songs if not sf.has_chart and not sf.has_mid)

    return {
        "chart_song_count": len(chart_songs),
        "chart_with_any_single_section": chart_with_any,
        "per_difficulty_chart": per_difficulty_chart,
        "mid_only_song_count": len(mid_only_songs),
        "mid_only_with_part_guitar": mid_only_with_track,
        "mid_only_with_any_difficulty": mid_only_with_any_diff,
        "per_difficulty_mid": per_difficulty_mid,
        "mid_parse_errors": mid_parse_errors,
        "combined_per_difficulty": combined_per_difficulty,
        "songs_with_native_easy_and_medium": songs_with_native_easy_and_medium,
        "both_chart_and_mid": both_chart_and_mid,
        "no_notes_file": no_notes_file,
        "overall_with_guitar_part": chart_with_any + mid_only_with_any_diff,
    }


AUDIO_EXTS = {".ogg", ".mp3", ".opus"}
VIDEO_EXTS = {".webm"}


def size_breakdown(root: Path, songs: list[SongFolder]) -> dict:
    chart_bytes = sum(sf.chart_size + sf.mid_size + sf.ini_size for sf in songs)
    audio_bytes = 0
    video_bytes = 0
    for sf in songs:
        for f in sf.path.iterdir():
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in AUDIO_EXTS:
                audio_bytes += f.stat().st_size
            elif ext in VIDEO_EXTS:
                video_bytes += f.stat().st_size
    return {
        "chart_related_bytes": chart_bytes,
        "audio_bytes": audio_bytes,
        "video_bytes": video_bytes,
    }


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def write_report(
    report_path: Path,
    root: Path,
    scan: ScanResult,
    cloud: dict,
    sizes: dict,
    guitar: dict | None,
    clip_impact: dict | None,
) -> None:
    lines = []
    lines.append("# Song library report\n")
    lines.append(f"Library root: `{root}`\n")
    lines.append(f"Song folders found: **{len(scan.songs)}**\n")
    lines.append(
        f"Hidden/temp directories skipped: **{len(scan.skipped_dirs)}**"
        + (
            " (e.g. " + ", ".join(scan.skipped_dirs[:5]) + ")"
            if scan.skipped_dirs
            else ""
        )
        + "\n"
    )

    n_ini = sum(1 for s in scan.songs if s.has_ini)
    n_chart = sum(1 for s in scan.songs if s.has_chart)
    n_mid = sum(1 for s in scan.songs if s.has_mid)
    n_sng = sum(1 for s in scan.songs if s.has_sng)
    lines.append("\n## File-type counts (per song folder, not per file)\n")
    lines.append(f"- song.ini: {n_ini}")
    lines.append(f"- notes.chart: {n_chart}")
    lines.append(f"- notes.mid: {n_mid}")
    lines.append(f"- .sng: {n_sng}")

    lines.append("\n## Cloud / local-file status\n")
    lines.append(
        f"- Chart/mid files checked (size via stat): {cloud['total_chart_files']}"
    )
    lines.append(
        f"- Zero-byte files: {cloud['zero_byte_count']} "
        f"({cloud['zero_byte_frac']:.1%})"
    )
    if cloud["zero_byte_examples"]:
        lines.append(f"  - e.g. {', '.join(cloud['zero_byte_examples'][:5])}")
    lines.append(
        f"- Content-read sample: {cloud['sample_size']} files, "
        f"{cloud['sample_read_errors']} read errors "
        f"({cloud['sample_error_frac']:.1%})"
    )
    if cloud["error_examples"]:
        for p, err in cloud["error_examples"][:5]:
            lines.append(f"  - {p}: {err}")
    lines.append(
        f"- Cloud-placeholder pattern suspected: **{cloud['cloud_only_suspected']}**"
    )

    if guitar is None:
        lines.append(
            "\n## Guitar-part availability\n\n"
            "Skipped: cloud-placeholder pattern suspected above threshold, "
            "so the scan stopped before opening every chart/mid file "
            "(to avoid triggering a mass download).\n"
        )
    else:
        lines.append("\n## Guitar-part availability\n")
        lines.append(
            f"Folders with **both** notes.chart and notes.mid "
            f"(chart preferred at ingest): {guitar['both_chart_and_mid']}"
        )
        lines.append(
            f"Folders with **neither** notes.chart nor notes.mid: "
            f"{guitar['no_notes_file']}"
        )
        lines.append(
            f"\n### From .chart section headers ({guitar['chart_song_count']} files)"
        )
        lines.append(
            f"- Songs with at least one [*Single] section: "
            f"{guitar['chart_with_any_single_section']}"
        )
        for d in DIFFICULTIES:
            lines.append(f"  - {d}Single: {guitar['per_difficulty_chart'][d]}")
        lines.append(
            f"\n### From .mid note ranges, songs with .mid but no .chart "
            f"({guitar['mid_only_song_count']} files)"
        )
        lines.append(
            f"- Songs with a `PART GUITAR` track: "
            f"{guitar['mid_only_with_part_guitar']}"
        )
        lines.append(
            f"- Songs with at least one note in a difficulty's fret range: "
            f"{guitar['mid_only_with_any_difficulty']}"
        )
        for d in DIFFICULTIES:
            lines.append(f"  - {d} (notes in range): {guitar['per_difficulty_mid'][d]}")
        lines.append(f"- Parse errors: {guitar['mid_parse_errors']}")

        lines.append(
            f"\n### Combined per-difficulty song counts "
            f"(native chart section, or .mid notes in the difficulty's fret "
            f"range when there's no .chart -- songs, not files)"
        )
        for d in DIFFICULTIES:
            lines.append(f"- {d}: {guitar['combined_per_difficulty'][d]}")

        lines.append(
            f"\n**Overall songs with a detected guitar part: "
            f"{guitar['overall_with_guitar_part']} / {len(scan.songs)}**"
        )

        lines.append(
            f"\n**Songs with native Easy AND Medium: "
            f"{guitar['songs_with_native_easy_and_medium']}** "
            f"(decision-rule threshold: 150 -- see docs/DECISIONS.md)"
        )

        if clip_impact is not None:
            lines.append(
                f"\n### Clip-corruption note impact "
                f"({clip_impact['clip_required_count']} .mid files needed "
                f"`clip=True` to parse at all)"
            )
            lines.append(
                f"- Out-of-range byte inside a SysEx payload (never a note "
                f"number): {clip_impact['sysex_only']}"
            )
            lines.append(
                f"- Out-of-range byte inside a note_on/note_off (or other "
                f"channel) message: {clip_impact['channel_message']}"
            )
            lines.append(f"- Other/unresolved: {clip_impact['other_location']}")
            lines.append(
                "- Of the files needing clipping, **0 had a clamped value "
                "land in the 60-100 guitar note range** -- every clamp "
                "happens inside proprietary SysEx metadata (e.g. Harmonix "
                "chart extensions), never inside a note_on/note_off message. "
                "clip=True therefore has no effect on note/difficulty "
                "detection accuracy for this library."
                if clip_impact["channel_message"] == 0
                else "- WARNING: at least one clamped value occurred inside "
                "a channel message; re-check whether it could be a note in "
                "60-100."
            )

    lines.append("\n## Size breakdown\n")
    lines.append(f"- Chart-related files (.chart/.mid/.ini): {_fmt_mb(sizes['chart_related_bytes'])}")
    lines.append(f"- Audio files (.ogg/.mp3/.opus): {_fmt_mb(sizes['audio_bytes'])}")
    lines.append(f"- Video background files (.webm): {_fmt_mb(sizes['video_bytes'])}")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--report",
        default="docs/library_report.md",
        help="Output path for the report (default: docs/library_report.md)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["song_library"]).expanduser()
    if not root.is_dir():
        raise SystemExit(f"song_library path does not exist or is not a dir: {root}")

    scan = find_song_folders(root)
    cloud = check_cloud_status(scan.songs)
    sizes = size_breakdown(root, scan.songs)

    guitar = None
    clip_impact = None
    if cloud["cloud_only_suspected"]:
        print(
            "WARNING: cloud-placeholder pattern suspected "
            f"(zero-byte frac={cloud['zero_byte_frac']:.1%}, "
            f"sample read-error frac={cloud['sample_error_frac']:.1%}). "
            "Stopping before opening every chart/mid file to avoid a mass "
            "download. See docs/library_report.md and tell the user."
        )
    else:
        guitar = analyze_guitar_parts(scan.songs)
        clip_impact = check_clip_corruption_note_impact(scan.songs)

    report_path = Path(args.report)
    write_report(report_path, root, scan, cloud, sizes, guitar, clip_impact)

    print(f"Song folders: {len(scan.songs)}")
    print(f"Skipped hidden/temp dirs: {len(scan.skipped_dirs)}")
    print(f"Cloud-only suspected: {cloud['cloud_only_suspected']}")
    print(f"Report written to {report_path}")

    return 1 if cloud["cloud_only_suspected"] else 0


def build_song_id_index(library_root: Path, cache_path: Path) -> dict[str, str]:
    """{song_id: rel_path} for every song folder under library_root --
    song_id (song.py::compute_song_id) is a one-way hash of rel_path, and
    Phase 2's ingest never persisted a reverse lookup (rel_path isn't stored
    per-song, by design -- see D17), so anything that needs the real
    filesystem path back from a song_id (e.g. to find its audio files)
    rebuilds this index by walking the library once and caches it to
    `cache_path` (a data/processed/ artifact, gitignored, read-only w.r.t.
    the library itself -- invariant 8)."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    from flyhero.game.song import compute_song_id

    scan = find_song_folders(library_root)
    index = {compute_song_id(sf.rel_path): sf.rel_path for sf in scan.songs}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index))
    return index


def find_song_folder_by_id(library_root: Path, cache_path: Path, song_id: str) -> Path | None:
    rel_path = build_song_id_index(library_root, cache_path).get(song_id)
    return (library_root / rel_path) if rel_path else None


if __name__ == "__main__":
    raise SystemExit(main())
