"""Ingest the song library into per-(song, difficulty) cached arrays and a
train/val/test split (PLAN Phase 2 task 4).

Order of operations (each step's output feeds the next):
1. Scan folders (library/scan.py) and read song.ini metadata for each.
2. Parse every folder's chart/mid (prefer .chart when both exist). Failures
   are caught and logged, never crash the run (invariant-adjacent PLAN rule).
3. Apply the open-note exclusion (task 5): a difficulty with > 10% open notes
   is dropped; a song's other difficulties are unaffected.
4. Resolve exact (artist, title) duplicates (same raw key): keep the member
   with the most surviving difficulties, tie-broken by Expert note count,
   then by rel_path; log what was dropped and why.
5. Group survivors by *normalized* (artist, title) (D-open-issue): any group
   with more than one raw key left is a near-duplicate group, reported for
   manual review but never merged or dropped.
6. Stratify by whether *any* group member has native Easy+Medium, shuffle
   deterministically per stratum (seed from config), split 80/10/10 by
   *group* so every member of a (near-)duplicate group lands in the same
   split -- otherwise two charts of the same underlying song could leak
   across train/test.
7. Cache surviving (song_id, difficulty) pairs and write splits.json.
"""

from __future__ import annotations

import argparse
import configparser
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flyhero.game.chart_parser import parse_chart
from flyhero.game.midi_parser import parse_midi
from flyhero.game.song import ParsedDifficulty, Song, compute_song_id, save_song
from flyhero.library.scan import find_song_folders
from flyhero.utils.config import load_config
from flyhero.utils.text import read_text_lenient

DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")

_RICHTEXT_RE = re.compile(r"<[^>]*>")
_BRACKET_RE = re.compile(r"\(.*?\)|\[.*?\]")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def _strip_richtext(s: str) -> str:
    return _RICHTEXT_RE.sub("", s)


def raw_key(s: str) -> str:
    return _strip_richtext(s).strip().lower()


def normalize_key(s: str) -> str:
    s = _strip_richtext(s).lower()
    s = _BRACKET_RE.sub("", s)
    s = _NONALNUM_RE.sub(" ", s).strip()
    return s


def _parse_ms_to_s(v: str) -> float:
    try:
        return float(v) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def read_song_ini(path: Path) -> dict[str, Any]:
    """Read song.ini's [song] section (matched case-insensitively -- real
    files use both [song] and [Song]). Strips Unity rich-text tags
    (`<color=#rrggbb>...</color>`) from text fields; real charter fields are
    wrapped in these. Returns {} if the file can't be parsed at all."""
    text = read_text_lenient(path)
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read_string(text)
    except configparser.Error:
        return {}
    section = next((s for s in cp.sections() if s.lower() == "song"), None)
    if section is None:
        return {}
    d = cp[section]
    return {
        "artist": _strip_richtext(d.get("artist", "")).strip(),
        "name": _strip_richtext(d.get("name", "")).strip(),
        "charter": _strip_richtext(d.get("charter", d.get("frets", ""))).strip(),
        "song_length_s": max(_parse_ms_to_s(d.get("song_length", "0")), 0.0),
        "delay_s": _parse_ms_to_s(d.get("delay", "0")),
    }


def _fallback_artist_title(rel_path: str) -> tuple[str, str]:
    folder_name = rel_path.rsplit("/", 1)[-1]
    if " - " in folder_name:
        artist, _, title = folder_name.partition(" - ")
        return artist.strip(), title.strip()
    return "", folder_name.strip()


@dataclass
class SongCandidate:
    rel_path: str
    path: Path
    has_chart: bool
    has_mid: bool
    artist: str
    title: str
    charter: str
    song_length_s: float
    delay_s: float
    song_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.song_id = compute_song_id(self.rel_path)


@dataclass
class ParseOutcome:
    candidate: SongCandidate
    difficulties: dict[str, ParsedDifficulty]
    chart_offset_s: float
    error: str | None = None


def collect_candidates(root: Path) -> list[SongCandidate]:
    scan = find_song_folders(root)
    candidates = []
    for sf in scan.songs:
        if not (sf.has_chart or sf.has_mid):
            continue
        ini = read_song_ini(sf.path / "song.ini") if sf.has_ini else {}
        fallback_artist, fallback_title = _fallback_artist_title(sf.rel_path)
        candidates.append(
            SongCandidate(
                rel_path=sf.rel_path,
                path=sf.path,
                has_chart=sf.has_chart,
                has_mid=sf.has_mid,
                artist=ini.get("artist") or fallback_artist,
                title=ini.get("name") or fallback_title,
                charter=ini.get("charter", ""),
                song_length_s=ini.get("song_length_s", 0.0),
                delay_s=ini.get("delay_s", 0.0),
            )
        )
    return candidates


def parse_song_folder(cand: SongCandidate, sustain_min_beats: float) -> ParseOutcome:
    try:
        if cand.has_chart:
            diffs, offset_s = parse_chart(cand.path / "notes.chart", sustain_min_beats)
        else:
            diffs, offset_s = parse_midi(cand.path / "notes.mid", sustain_min_beats)
        return ParseOutcome(cand, diffs, offset_s, None)
    except Exception as e:  # noqa: BLE001 -- must never crash the ingest run
        return ParseOutcome(cand, {}, 0.0, f"{type(e).__name__}: {e}")


def apply_open_note_filter(
    outcome: ParseOutcome, max_frac: float
) -> tuple[dict[str, ParsedDifficulty], list[dict]]:
    kept: dict[str, ParsedDifficulty] = {}
    excluded: list[dict] = []
    for diff, pd in outcome.difficulties.items():
        denom = pd.open_note_count + pd.total_note_count
        frac = (pd.open_note_count / denom) if denom > 0 else 0.0
        if frac > max_frac:
            excluded.append(
                {"rel_path": outcome.candidate.rel_path, "difficulty": diff, "open_frac": frac}
            )
        else:
            kept[diff] = pd
    return kept, excluded


def compute_duration_s(song_length_s: float, kept: dict[str, ParsedDifficulty]) -> float:
    if song_length_s > 0:
        return song_length_s
    if kept:
        return max(pd.duration_s for pd in kept.values())
    return 0.0


def resolve_exact_duplicates(
    valid_outcomes: list[ParseOutcome], kept_map: dict[str, dict[str, ParsedDifficulty]]
) -> tuple[list[ParseOutcome], list[dict]]:
    groups: dict[tuple[str, str], list[ParseOutcome]] = defaultdict(list)
    for o in valid_outcomes:
        groups[(raw_key(o.candidate.artist), raw_key(o.candidate.title))].append(o)

    survivors: list[ParseOutcome] = []
    drop_log: list[dict] = []
    for members in groups.values():
        if len(members) == 1:
            survivors.append(members[0])
            continue

        def sort_key(o: ParseOutcome) -> tuple[int, int, str]:
            kept = kept_map[o.candidate.rel_path]
            expert_notes = kept["Expert"].total_note_count if "Expert" in kept else 0
            return (-len(kept), -expert_notes, o.candidate.rel_path)

        ordered = sorted(members, key=sort_key)
        winner = ordered[0]
        survivors.append(winner)
        winner_kept = kept_map[winner.candidate.rel_path]
        for loser in ordered[1:]:
            loser_kept = kept_map[loser.candidate.rel_path]
            drop_log.append(
                {
                    "dropped": loser.candidate.rel_path,
                    "kept": winner.candidate.rel_path,
                    "dropped_n_difficulties": len(loser_kept),
                    "kept_n_difficulties": len(winner_kept),
                    "dropped_expert_notes": loser_kept["Expert"].total_note_count
                    if "Expert" in loser_kept
                    else 0,
                    "kept_expert_notes": winner_kept["Expert"].total_note_count
                    if "Expert" in winner_kept
                    else 0,
                }
            )
    return survivors, drop_log


def group_by_normalized_key(
    survivors: list[ParseOutcome], kept_map: dict[str, dict[str, ParsedDifficulty]]
) -> tuple[list[dict], list[dict]]:
    usable = [o for o in survivors if kept_map[o.candidate.rel_path]]

    norm_groups: dict[tuple[str, str], list[ParseOutcome]] = defaultdict(list)
    for o in usable:
        key = (normalize_key(o.candidate.artist), normalize_key(o.candidate.title))
        norm_groups[key].append(o)

    groups = []
    near_dup_log = []
    for key, members in norm_groups.items():
        raw_keys = {(raw_key(o.candidate.artist), raw_key(o.candidate.title)) for o in members}
        has_native_easy_medium = any(
            {"Easy", "Medium"} <= set(kept_map[o.candidate.rel_path]) for o in members
        )
        groups.append(
            {
                "key": key,
                "members": members,
                "stratum": "native_easy_medium" if has_native_easy_medium else "expert_only",
            }
        )
        if len(raw_keys) > 1:
            near_dup_log.append(
                {
                    "normalized_key": f"{key[0]} / {key[1]}",
                    "members": [o.candidate.rel_path for o in members],
                }
            )
    return groups, near_dup_log


def stratified_group_split(
    groups: list[dict], split_cfg: dict
) -> dict[str, str]:
    """Returns {song_id: split_name}. Splitting happens at the group level so
    every member (a normalized-duplicate group's song_ids) shares one split."""
    assignment: dict[str, str] = {}
    for stratum in ("native_easy_medium", "expert_only"):
        stratum_groups = sorted(
            (g for g in groups if g["stratum"] == stratum), key=lambda g: g["key"]
        )
        rng = random.Random(split_cfg["seed"])
        rng.shuffle(stratum_groups)

        n = len(stratum_groups)
        n_train = int(n * split_cfg["train"])
        n_val = int(n * split_cfg["val"])
        for i, g in enumerate(stratum_groups):
            split_name = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for o in g["members"]:
                assignment[o.candidate.song_id] = split_name
    return assignment


def write_cached_songs(
    usable: list[ParseOutcome],
    kept_map: dict[str, dict[str, ParsedDifficulty]],
    songs_dir: Path,
) -> None:
    songs_dir.mkdir(parents=True, exist_ok=True)
    for o in usable:
        cand = o.candidate
        kept = kept_map[cand.rel_path]
        duration_s = compute_duration_s(cand.song_length_s, kept)
        for diff, pd in kept.items():
            song = Song(
                song_id=cand.song_id,
                title=cand.title,
                artist=cand.artist,
                charter=cand.charter,
                difficulty=diff,
                notes=pd.notes,
                duration_s=duration_s,
                chart_offset_s=o.chart_offset_s,
                ini_delay_s=cand.delay_s,
            )
            save_song(song, songs_dir / f"{cand.song_id}_{diff}.npz")


def compute_split_stratum_difficulty_counts(
    groups: list[dict],
    assignment: dict[str, str],
    kept_map: dict[str, dict[str, ParsedDifficulty]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for g in groups:
        for o in g["members"]:
            split_name = assignment[o.candidate.song_id]
            for diff in kept_map[o.candidate.rel_path]:
                counts[f"{split_name}/{g['stratum']}/{diff}"] += 1
    return dict(sorted(counts.items()))


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def render_ingest_report_section(
    n_candidates: int,
    parse_failures: list[ParseOutcome],
    open_exclusions: list[dict],
    exact_dup_log: list[dict],
    near_dup_log: list[dict],
    split_counts: dict[str, int],
) -> str:
    lines = ["## Ingest results (Phase 2)\n"]
    lines.append(f"Song folders considered: **{n_candidates}**\n")

    lines.append(f"\n### Parse failures ({len(parse_failures)})\n")
    for o in parse_failures[:50]:
        lines.append(f"- `{o.candidate.rel_path}`: {o.error}")
    if len(parse_failures) > 50:
        lines.append(f"- ... and {len(parse_failures) - 50} more")

    lines.append(f"\n### Open-note exclusions ({len(open_exclusions)})\n")
    for e in open_exclusions:
        lines.append(
            f"- `{e['rel_path']}` [{e['difficulty']}]: {_fmt_pct(e['open_frac'])} open notes"
        )

    lines.append(f"\n### Exact-duplicate drops ({len(exact_dup_log)})\n")
    for d in exact_dup_log:
        lines.append(
            f"- dropped `{d['dropped']}` (kept `{d['kept']}`): "
            f"{d['dropped_n_difficulties']} vs {d['kept_n_difficulties']} surviving difficulties, "
            f"{d['dropped_expert_notes']} vs {d['kept_expert_notes']} Expert notes"
        )

    lines.append(f"\n### Near-duplicate groups, reported only ({len(near_dup_log)})\n")
    for g in near_dup_log:
        lines.append(f"- `{g['normalized_key']}`: {', '.join(g['members'])}")

    lines.append("\n### Split x stratum x difficulty song counts\n")
    for key, count in split_counts.items():
        lines.append(f"- {key}: {count}")

    return "\n".join(lines) + "\n"


def append_ingest_section(report_path: Path, section_text: str) -> None:
    marker = "## Ingest results (Phase 2)"
    existing = report_path.read_text() if report_path.exists() else ""
    idx = existing.find(marker)
    if idx != -1:
        existing = existing[:idx]
    existing = existing.rstrip("\n") + "\n\n" if existing.strip() else ""
    report_path.write_text(existing + section_text)


def run_ingest(
    root: Path, cfg: dict, processed_dir: Path, report_path: Path
) -> dict:
    sustain_min_beats = cfg["sustain_min_beats"]
    open_note_exclude_frac = cfg["open_note_exclude_frac"]
    split_cfg = cfg["split"]

    candidates = collect_candidates(root)
    outcomes = [parse_song_folder(c, sustain_min_beats) for c in candidates]

    parse_failures = [o for o in outcomes if o.error is not None]
    valid_outcomes = [o for o in outcomes if o.error is None]

    kept_map: dict[str, dict[str, ParsedDifficulty]] = {}
    open_exclusions: list[dict] = []
    for o in valid_outcomes:
        kept, excl = apply_open_note_filter(o, open_note_exclude_frac)
        kept_map[o.candidate.rel_path] = kept
        open_exclusions.extend(excl)

    survivors, exact_dup_log = resolve_exact_duplicates(valid_outcomes, kept_map)
    groups, near_dup_log = group_by_normalized_key(survivors, kept_map)
    assignment = stratified_group_split(groups, split_cfg)

    usable = [o for o in survivors if kept_map[o.candidate.rel_path]]
    write_cached_songs(usable, kept_map, processed_dir / "songs")

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for song_id, split_name in assignment.items():
        splits[split_name].append(song_id)
    for v in splits.values():
        v.sort()
    (processed_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")

    split_counts = compute_split_stratum_difficulty_counts(groups, assignment, kept_map)

    section = render_ingest_report_section(
        len(candidates), parse_failures, open_exclusions, exact_dup_log, near_dup_log, split_counts
    )
    append_ingest_section(report_path, section)

    return {
        "n_candidates": len(candidates),
        "n_parse_failures": len(parse_failures),
        "parse_failures": [
            {"rel_path": o.candidate.rel_path, "error": o.error} for o in parse_failures
        ],
        "open_exclusions": open_exclusions,
        "exact_dup_log": exact_dup_log,
        "near_dup_log": near_dup_log,
        "split_counts": split_counts,
        "n_songs_written": len(usable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--report", default="docs/library_report.md")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_config(args.paths_config)
    root = Path(paths_cfg["song_library"]).expanduser()
    if not root.is_dir():
        raise SystemExit(f"song_library path does not exist or is not a dir: {root}")

    processed_dir = Path(cfg["processed_dir"])
    summary = run_ingest(root, cfg, processed_dir, Path(args.report))

    print(f"Song folders considered: {summary['n_candidates']}")
    print(f"Parse failures: {summary['n_parse_failures']}")
    print(f"Open-note exclusions: {len(summary['open_exclusions'])}")
    print(f"Exact-duplicate drops: {len(summary['exact_dup_log'])}")
    print(f"Near-duplicate groups (reported only): {len(summary['near_dup_log'])}")
    print(f"Songs written: {summary['n_songs_written']}")
    print(f"Report appended to {args.report}")

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
