"""Phase 2 tests: ingest.py's song.ini decoding, exact/near-duplicate
resolution, open-note exclusion, and the group-level stratified split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flyhero.game.ingest import (
    ParseOutcome,
    SongCandidate,
    apply_open_note_filter,
    group_by_normalized_key,
    read_song_ini,
    resolve_exact_duplicates,
    run_ingest,
    stratified_group_split,
)
from flyhero.game.song import NOTE_DTYPE, ParsedDifficulty, load_song

SPLIT_CFG = {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 0}


def _pd(total_note_count: int = 1, open_note_count: int = 0, duration_s: float = 1.0) -> ParsedDifficulty:
    return ParsedDifficulty(
        notes=np.zeros(0, dtype=NOTE_DTYPE),
        duration_s=duration_s,
        open_note_count=open_note_count,
        total_note_count=total_note_count,
        ignored_counts={},
    )


def _candidate(rel_path: str, artist: str, title: str) -> SongCandidate:
    return SongCandidate(
        rel_path=rel_path,
        path=Path(rel_path),
        has_chart=True,
        has_mid=False,
        artist=artist,
        title=title,
        charter="",
        song_length_s=0.0,
        delay_s=0.0,
    )


def _outcome(cand: SongCandidate, diffs: dict[str, ParsedDifficulty]) -> ParseOutcome:
    return ParseOutcome(candidate=cand, difficulties=diffs, chart_offset_s=0.0, error=None)


# ---------------------------------------------------------------------------
# song.ini decoding
# ---------------------------------------------------------------------------


def test_read_song_ini_handles_utf16(tmp_path):
    p = tmp_path / "song.ini"
    p.write_bytes("[song]\nartist = Test Artist\nname = Test Title\n".encode("utf-16"))
    result = read_song_ini(p)
    assert result["artist"] == "Test Artist"
    assert result["name"] == "Test Title"


def test_read_song_ini_handles_utf8_bom(tmp_path):
    p = tmp_path / "song.ini"
    p.write_bytes(b"\xef\xbb\xbf[song]\nartist = BOM Artist\nname = BOM Title\n")
    result = read_song_ini(p)
    assert result["artist"] == "BOM Artist"
    assert result["name"] == "BOM Title"


def test_read_song_ini_strips_richtext_and_falls_back_to_frets(tmp_path):
    p = tmp_path / "song.ini"
    p.write_text(
        "[Song]\nartist = A\nname = B\ncharter = <color=#ffa500>Neversoft</color>\n"
        "song_length = 12345\ndelay = -50\n"
    )
    result = read_song_ini(p)
    assert result["charter"] == "Neversoft"
    assert result["song_length_s"] == pytest.approx(12.345)
    assert result["delay_s"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# exact-duplicate tie-break
# ---------------------------------------------------------------------------


def test_exact_duplicate_prefers_more_surviving_difficulties():
    cand_a = _candidate("dup_a", "DupArtist", "DupTitle")
    cand_b = _candidate("dup_b", "DupArtist", "DupTitle")
    outcome_a = _outcome(cand_a, {"Expert": _pd(10), "Easy": _pd(5)})
    outcome_b = _outcome(cand_b, {"Expert": _pd(3), "Medium": _pd(5), "Easy": _pd(5)})
    kept_map = {"dup_a": outcome_a.difficulties, "dup_b": outcome_b.difficulties}

    survivors, drop_log = resolve_exact_duplicates([outcome_a, outcome_b], kept_map)

    assert [o.candidate.rel_path for o in survivors] == ["dup_b"]
    assert drop_log[0]["dropped"] == "dup_a"
    assert drop_log[0]["kept"] == "dup_b"


def test_exact_duplicate_tie_broken_by_expert_note_count():
    cand_c = _candidate("dup_c", "DupArtist2", "DupTitle2")
    cand_d = _candidate("dup_d", "DupArtist2", "DupTitle2")
    outcome_c = _outcome(cand_c, {"Expert": _pd(20)})
    outcome_d = _outcome(cand_d, {"Expert": _pd(5)})
    kept_map = {"dup_c": outcome_c.difficulties, "dup_d": outcome_d.difficulties}

    survivors, _ = resolve_exact_duplicates([outcome_c, outcome_d], kept_map)

    assert [o.candidate.rel_path for o in survivors] == ["dup_c"]


def test_exact_duplicate_tie_broken_by_rel_path():
    cand_e = _candidate("dup_e", "DupArtist3", "DupTitle3")
    cand_f = _candidate("dup_f", "DupArtist3", "DupTitle3")
    outcome_e = _outcome(cand_e, {"Expert": _pd(10)})
    outcome_f = _outcome(cand_f, {"Expert": _pd(10)})
    kept_map = {"dup_e": outcome_e.difficulties, "dup_f": outcome_f.difficulties}

    survivors, _ = resolve_exact_duplicates([outcome_e, outcome_f], kept_map)

    assert [o.candidate.rel_path for o in survivors] == ["dup_e"]


# ---------------------------------------------------------------------------
# open-note exclusion
# ---------------------------------------------------------------------------


def test_open_note_filter_excludes_only_the_offending_difficulty():
    cand = _candidate("open_test", "OpenArtist", "OpenTitle")
    outcome = _outcome(
        cand,
        {
            "Expert": _pd(total_note_count=2, open_note_count=9),  # 81.8% open
            "Easy": _pd(total_note_count=5, open_note_count=0),
        },
    )

    kept, excluded = apply_open_note_filter(outcome, max_frac=0.10)

    assert set(kept) == {"Easy"}
    assert len(excluded) == 1
    assert excluded[0]["difficulty"] == "Expert"
    assert excluded[0]["open_frac"] == pytest.approx(9 / 11)


# ---------------------------------------------------------------------------
# near-duplicate grouping + group-level stratified split
# ---------------------------------------------------------------------------


def _filler_outcomes(n: int, native_easy_medium: bool, prefix: str) -> list[ParseOutcome]:
    outcomes = []
    for i in range(n):
        cand = _candidate(f"{prefix}_{i}", f"Artist_{prefix}_{i}", f"Title_{prefix}_{i}")
        diffs = {"Expert": _pd(10)}
        if native_easy_medium:
            diffs["Easy"] = _pd(5)
            diffs["Medium"] = _pd(5)
        outcomes.append(_outcome(cand, diffs))
    return outcomes


def test_near_duplicate_group_reported_and_never_split_across():
    fillers_expert_only = _filler_outcomes(20, native_easy_medium=False, prefix="ex")
    fillers_native = _filler_outcomes(20, native_easy_medium=True, prefix="nat")

    cand_p1 = _candidate("pretender_a", "Foo Fighters", "The Pretender")
    cand_p2 = _candidate("pretender_b", "Foo Fighters", "The Pretender (Co-op)")
    outcome_p1 = _outcome(cand_p1, {"Expert": _pd(10)})
    outcome_p2 = _outcome(cand_p2, {"Expert": _pd(10)})

    all_outcomes = fillers_expert_only + fillers_native + [outcome_p1, outcome_p2]
    kept_map = {o.candidate.rel_path: o.difficulties for o in all_outcomes}

    groups, near_dup_log = group_by_normalized_key(all_outcomes, kept_map)

    assert len(near_dup_log) == 1
    assert set(near_dup_log[0]["members"]) == {"pretender_a", "pretender_b"}

    assignment = stratified_group_split(groups, SPLIT_CFG)

    assert len(assignment) == 42
    assert assignment[cand_p1.song_id] == assignment[cand_p2.song_id]
    assert set(assignment.values()) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# end-to-end integration on a small synthetic library
# ---------------------------------------------------------------------------


def _chart_text(sections: dict[str, list[tuple[int, int]]]) -> str:
    """sections: difficulty -> list of (fret_or_7_for_open, sustain_ticks)."""
    lines = ["[Song]\n{\n\tResolution = 192\n}\n[SyncTrack]\n{\n\t0 = B 120000\n}\n"]
    for diff, notes in sections.items():
        lines.append(f"[{diff}Single]\n{{\n")
        for i, (fret, sustain) in enumerate(notes):
            tick = 192 * (i + 1)
            lines.append(f"\t{tick} = N {fret} {sustain}\n")
        lines.append("}\n")
    return "".join(lines)


def _write_song(root: Path, rel_path: str, artist: str, title: str, sections: dict) -> None:
    d = root / rel_path
    d.mkdir(parents=True, exist_ok=True)
    (d / "song.ini").write_text(
        f"[song]\nartist = {artist}\nname = {title}\ncharter = Someone\nsong_length = 10000\n"
    )
    (d / "notes.chart").write_text(_chart_text(sections))


def test_run_ingest_end_to_end(tmp_path):
    library_root = tmp_path / "library"

    for i in range(6):
        _write_song(
            library_root, f"expert_only_{i}", f"EOArtist{i}", f"EOTitle{i}",
            {"Expert": [(i % 5, 0) for i in range(10)]},
        )
    for i in range(6):
        _write_song(
            library_root, f"native_{i}", f"NatArtist{i}", f"NatTitle{i}",
            {
                "Easy": [(i % 5, 0) for i in range(5)],
                "Medium": [(i % 5, 0) for i in range(5)],
                "Expert": [(i % 5, 0) for i in range(10)],
            },
        )

    # exact duplicate: same (artist, title), different folder/difficulty coverage.
    _write_song(
        library_root, "dup_winner", "DupArtist", "DupTitle",
        {"Expert": [(0, 0)] * 10, "Easy": [(0, 0)] * 5},
    )
    _write_song(
        library_root, "dup_loser", "DupArtist", "DupTitle",
        {"Expert": [(0, 0)] * 3},
    )

    # near duplicate: normalizes to the same key, both must survive and land
    # in the same split.
    _write_song(library_root, "near_a", "Foo Fighters", "The Pretender", {"Expert": [(0, 0)] * 10})
    _write_song(
        library_root, "near_b", "Foo Fighters", "The Pretender (Co-op)",
        {"Expert": [(0, 0)] * 10},
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    report_path = tmp_path / "library_report.md"

    cfg = {
        "processed_dir": str(processed_dir),
        "sustain_min_beats": 0.5,
        "open_note_exclude_frac": 0.10,
        "split": SPLIT_CFG,
    }

    summary = run_ingest(library_root, cfg, processed_dir, report_path)

    assert summary["n_parse_failures"] == 0
    assert len(summary["exact_dup_log"]) == 1
    assert summary["exact_dup_log"][0]["dropped"] == "dup_loser"
    assert len(summary["near_dup_log"]) == 1

    splits = json.loads((processed_dir / "splits.json").read_text())
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids))  # no song in two splits

    from flyhero.game.song import compute_song_id

    near_a_id = compute_song_id("near_a")
    near_b_id = compute_song_id("near_b")

    def split_of(song_id):
        for name, ids in splits.items():
            if song_id in ids:
                return name
        return None

    assert split_of(near_a_id) == split_of(near_b_id)

    dup_loser_id = compute_song_id("dup_loser")
    assert dup_loser_id not in all_ids
    assert not (processed_dir / "songs" / f"{dup_loser_id}_Expert.npz").exists()

    dup_winner_id = compute_song_id("dup_winner")
    winner_song = load_song(processed_dir / "songs" / f"{dup_winner_id}_Expert.npz")
    assert winner_song.artist == "DupArtist"

    report_text = report_path.read_text()
    assert "## Ingest results (Phase 2)" in report_text
    assert "Near-duplicate groups" in report_text
