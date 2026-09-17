"""Paired val-set comparison of two checkpoints, broken down by note
category and failure mode (the "eval instrument", 2026-09-17).

Why this exists: the in-training eval (`train/bc.py`, D27) scores 10 fixed
val songs and reports one absolute hit_rate. On Medium that number's
eval-to-eval swing is ~0.10 while the real paired difference between two
checkpoints is ~0.02 -- so absolute numbers cannot resolve whether a change
helped. This script:

  1. uses all val excerpts at a difficulty (30 requested -> 28 exist with
     Medium >= 60s), not the first 10;
  2. reports **paired** per-excerpt differences against a baseline
     checkpoint (mean +- SE over excerpts), which removes the
     song-difficulty variance that dominates the absolute number;
  3. breaks every condition down by note category (single / 2-note adjacent
     / 2-note non-adjacent / 3+ / open) and, for missed notes, by failure
     mode (no strum in window / nothing held / missing a fret / extra fret /
     wrong lane);
  4. takes --strum-threshold and --fret-threshold as lists, so the same run
     doubles as a decoder sweep: inference runs **once** per checkpoint and
     every threshold cell re-decodes the cached probabilities.

Inference goes through `train.bc.score_eval_entry` -- the same function the
in-training eval and scripts/render_gameplay_video.py use -- so at the
default 0.5/0.5 thresholds this reproduces the in-training eval number by
construction. Scoring is `rules.score_playthrough` (invariant 4) and
decoding is `decoder.decode_trace` (invariant 1: model outputs only).

Defaults to CPU so it never contends with a training run on MPS; the
CPU-vs-MPS difference on these excerpts is ~0.01 hit_rate (documented in
PROGRESS.md's 2026-09-17 overnight entry), which is why --device mps exists
for reproducing an in-training number exactly.

Usage:
  # paired comparison of a candidate against the current best
  uv run python -m eval.compare --config configs/compare.yaml \\
      --checkpoint runs/<run>/checkpoint_step44000.pt \\
      --baseline runs/<run>/checkpoint_best.pt

  # decoder sweep on one checkpoint (baseline = that checkpoint at 0.5/0.5)
  uv run python -m eval.compare --config configs/compare.yaml \\
      --checkpoint runs/<run>/checkpoint_best.pt \\
      --strum-threshold 0.3 0.5 0.7 --fret-threshold 0.5 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flyhero.game.decoder import FRET_THRESHOLD, STRUM_THRESHOLD, decode_trace, frets_to_held_mask
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.bc import build_policy, score_eval_entry
from train.common import build_val_set

# Note categories partition every note, so the per-category counts sum to
# n_notes. "open" notes survive ingest in songs under the >10% exclusion
# rule (D17), so they get their own bucket rather than being folded in.
CATEGORIES = ("single", "chord2_adj", "chord2_nonadj", "chord3plus", "open")

# Failure modes for a note that was not hit, judged at the decoded strum
# frame nearest the note's time within +-hit_window_s.
MISS_MODES = ("no_strum", "nothing_held", "missing_fret", "extra_fret", "wrong_lane", "strum_consumed", "other")


def popcount(mask: int) -> int:
    return int(mask).bit_count()


def note_category(lane_mask: np.ndarray) -> np.ndarray:
    """lane_mask[N] uint8 lane bitmask -> category[N] str, one of
    CATEGORIES. A 2-note chord is "adjacent" when its two lanes are
    neighbouring frets (index difference 1), which is the distinction the
    chord diagnosis found mattered: adjacent 2-note chords scored 0.315 and
    non-adjacent 0.000 at step 16000."""
    m = np.asarray(lane_mask).astype(np.uint8)
    n_lanes = np.unpackbits(m[:, None], axis=1).sum(axis=1)

    lowest = np.full(len(m), -1, dtype=np.int64)
    highest = np.full(len(m), -1, dtype=np.int64)
    for lane in range(5):
        bit = ((m >> lane) & 1).astype(bool)
        lowest = np.where(bit & (lowest < 0), lane, lowest)
        highest = np.where(bit, lane, highest)
    adjacent = (highest - lowest) == 1

    out = np.full(len(m), "chord3plus", dtype="<U13")
    out[n_lanes == 0] = "open"
    out[n_lanes == 1] = "single"
    out[(n_lanes == 2) & adjacent] = "chord2_adj"
    out[(n_lanes == 2) & ~adjacent] = "chord2_nonadj"
    return out


def classify_miss(
    note_lane_mask: int, note_time_s: float, strum_frames: np.ndarray, held_mask: np.ndarray,
    fps: int, hit_window_s: float,
) -> str:
    """Why this note was not hit. Looks at the decoded strum frame nearest
    the note's own time within +-hit_window_s (ties -> the earlier frame)
    and compares the frets held on that frame against the note's lanes.

    "strum_consumed" means the frets were exactly right and a strum was in
    the window, but rules.py awarded that strum to an earlier open note --
    the one failure mode that is a consequence of chart density rather than
    of the model's output.
    """
    if len(strum_frames) == 0:
        return "no_strum"
    in_window = np.abs(strum_frames / fps - note_time_s) <= hit_window_s
    if not in_window.any():
        return "no_strum"
    candidates = strum_frames[in_window]  # ascending, so argmin breaks ties toward the earlier frame
    nearest = int(candidates[np.argmin(np.abs(candidates / fps - note_time_s))])

    held = int(held_mask[nearest])
    lane = int(note_lane_mask)
    if held == lane:
        return "strum_consumed"
    if held == 0:
        return "nothing_held"
    missing = lane & ~held
    extra = held & ~lane
    if missing and not extra:
        return "missing_fret"
    if extra and not missing:
        return "extra_fret"
    if popcount(held) == popcount(lane):
        return "wrong_lane"
    return "other"


def analyze_excerpt(
    probs: np.ndarray, notes: np.ndarray, fps: int, hit_window_s: float,
    strum_threshold: float, fret_threshold: float,
) -> dict:
    """Decode one excerpt's probabilities at the given thresholds, score it,
    and break the result down by note category and failure mode.

    probs: [T, 6] sigmoid-applied (5 frets + strum). notes: the excerpt's own
    notes, already re-based so note time 0 == probs frame 0.
    """
    frets, strum = decode_trace(probs, strum_threshold, fret_threshold)  # [T,5] bool, [T] bool
    held_mask = frets_to_held_mask(frets)  # [T] int64
    metrics, events = score_playthrough(notes, held_mask, strum, fps, hit_window_s)

    hit = np.zeros(len(notes), dtype=bool)
    for e in events:
        if e.kind == "hit":
            hit[e.note_idx] = True

    strum_frames = np.flatnonzero(strum)
    categories = note_category(notes["lane_mask"])
    by_category = {}
    for cat in CATEGORIES:
        sel = np.flatnonzero(categories == cat)
        modes = dict.fromkeys(MISS_MODES, 0)
        for i in sel:
            if hit[i]:
                continue
            mode = classify_miss(
                int(notes["lane_mask"][i]), float(notes["time_s"][i]), strum_frames, held_mask,
                fps, hit_window_s,
            )
            modes[mode] += 1
        by_category[cat] = dict(n=len(sel), hits=int(hit[sel].sum()), miss_modes=modes)

    minutes = len(probs) / fps / 60.0
    # The playthrough score rules.py's reward weights imply (configs/game.yaml:
    # hit +1, miss -1, overstrum -0.3) -- the objective the 2026-09-17
    # decoder sweep ranked threshold cells by.
    score = metrics["n_hits"] - metrics["n_misses"] - 0.3 * metrics["n_overstrums"]
    return dict(
        hit_rate=metrics["hit_rate"], score=float(score), minutes=minutes,
        n_notes=metrics["n_notes"], n_hits=metrics["n_hits"], n_misses=metrics["n_misses"],
        n_overstrums=metrics["n_overstrums"], overstrums_per_min=metrics["overstrums_per_min"],
        sustain_frac=metrics["sustain_frac"], by_category=by_category,
    )


def paired_diff(values: list[float], baseline: list[float]) -> dict:
    """mean +- SE of the per-excerpt difference. Paired, because the
    song-to-song spread (SE ~0.03 absolute) is far larger than the
    difference between two checkpoints (~0.02)."""
    d = np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    n = len(d)
    if n == 0:
        return dict(n=0, mean=None, se=None)
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return dict(n=n, mean=float(d.mean()), se=se)


def aggregate_condition(per_excerpt: list[dict], baseline: list[dict] | None) -> dict:
    """Pooled totals over excerpts, plus paired differences against
    `baseline` (the same excerpts, in the same order) when given."""
    n_notes = sum(r["n_notes"] for r in per_excerpt)
    minutes = sum(r["minutes"] for r in per_excerpt)
    out = dict(
        n_excerpts=len(per_excerpt),
        hit_rate_mean=float(np.mean([r["hit_rate"] for r in per_excerpt])) if per_excerpt else 0.0,
        hit_rate_se=(
            float(np.std([r["hit_rate"] for r in per_excerpt], ddof=1) / np.sqrt(len(per_excerpt)))
            if len(per_excerpt) > 1 else 0.0
        ),
        hit_rate_pooled=sum(r["n_hits"] for r in per_excerpt) / n_notes if n_notes else 0.0,
        score_mean=float(np.mean([r["score"] for r in per_excerpt])) if per_excerpt else 0.0,
        score_total=float(sum(r["score"] for r in per_excerpt)),
        overstrums_per_min_pooled=sum(r["n_overstrums"] for r in per_excerpt) / minutes if minutes else 0.0,
        n_notes=n_notes, minutes=minutes,
    )

    categories = {}
    for cat in CATEGORIES:
        n = sum(r["by_category"][cat]["n"] for r in per_excerpt)
        hits = sum(r["by_category"][cat]["hits"] for r in per_excerpt)
        entry = dict(
            n=n, hits=hits, hit_rate_pooled=hits / n if n else None,
            miss_modes={
                mode: sum(r["by_category"][cat]["miss_modes"][mode] for r in per_excerpt)
                for mode in MISS_MODES
            },
        )
        if baseline is not None:
            # Paired only over excerpts that actually contain this category
            # (a non-adjacent 2-note chord appears in a minority of songs),
            # so the difference isn't diluted by excerpts where it is
            # undefined on both sides.
            # A category's note count is identical in both conditions (same
            # songs, same notes -- only the decoding differs), so this index
            # set is valid for the baseline too and neither side divides by 0.
            idx = [i for i, r in enumerate(per_excerpt) if r["by_category"][cat]["n"] > 0]

            def cat_rate(rows: list[dict], i: int, cat: str = cat) -> float:
                e = rows[i]["by_category"][cat]
                return e["hits"] / e["n"]

            entry["diff"] = paired_diff(
                [cat_rate(per_excerpt, i) for i in idx], [cat_rate(baseline, i) for i in idx]
            )
            entry["baseline_hit_rate_pooled"] = (
                sum(baseline[i]["by_category"][cat]["hits"] for i in idx)
                / sum(baseline[i]["by_category"][cat]["n"] for i in idx)
            ) if idx else None
            entry["miss_modes_diff"] = {
                mode: entry["miss_modes"][mode] - sum(r["by_category"][cat]["miss_modes"][mode] for r in baseline)
                for mode in MISS_MODES
            }
        categories[cat] = entry
    out["by_category"] = categories

    if baseline is not None:
        out["hit_rate_diff"] = paired_diff(
            [r["hit_rate"] for r in per_excerpt], [r["hit_rate"] for r in baseline]
        )
        out["score_diff"] = paired_diff(
            [r["score"] for r in per_excerpt], [r["score"] for r in baseline]
        )
        out["overstrums_per_min_diff"] = paired_diff(
            [r["overstrums_per_min"] for r in per_excerpt], [r["overstrums_per_min"] for r in baseline]
        )
        better = sum(1 for a, b in zip(per_excerpt, baseline) if a["hit_rate"] > b["hit_rate"])
        out["excerpts_better"] = better
    return out


def run_inference(
    ckpt_path: Path, val_entries: list, cfg: dict, device, dtype, log=print,
) -> tuple[list[np.ndarray], list[np.ndarray], int, list[float]]:
    """(probs per excerpt [T,6], excerpt notes per excerpt, checkpoint step,
    score_eval_entry's own per-excerpt hit_rate).

    Uses train.bc.score_eval_entry, the same inference path as the
    in-training eval, and keeps only its raw probabilities -- every
    threshold cell then re-decodes these without re-running the model. The
    returned hit_rates are score_eval_entry's own (default-threshold)
    numbers, used by check_reproduces_in_training_eval below.
    """
    src_run = ckpt_path.parent
    cfg_game = load_config(resolve_run_config(src_run, "game.yaml"))
    model_kind = (src_run / "model_kind.txt").read_text().strip() if (src_run / "model_kind.txt").exists() else "connectome"
    cfg_brain = load_config(resolve_run_config(src_run, "brain.yaml")) if model_kind == "connectome" else {}

    processed_dir = Path(cfg["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)
    policy = build_policy(model_kind, {}, cfg_game, cfg_brain, photo_map, device, dtype)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model_state"])
    step = int(ckpt["step"])
    log(f"  {ckpt_path} (step {step}, {model_kind}, {device})")

    probs_list, notes_list, eval_hit_rates = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, entry in enumerate(val_entries):
            result = score_eval_entry(
                policy, env, entry, cfg["burn_in_s"], cfg_game["fps"], cfg_game["hit_window_s"], device, dtype,
            )
            probs_list.append(result["probs"].astype(np.float32))  # [T, 6]
            notes_list.append(result["excerpt_notes"])
            eval_hit_rates.append(float(result["metrics"]["hit_rate"]))
            if (i + 1) % 5 == 0 or i + 1 == len(val_entries):
                log(f"    {i + 1}/{len(val_entries)} excerpts, wall {time.time() - t0:.0f}s")
    return probs_list, notes_list, step, eval_hit_rates


def check_reproduces_in_training_eval(
    per_excerpt: list[dict], eval_hit_rates: list[float], n_in_training: int, log=print,
) -> dict:
    """This script's default-threshold numbers must equal what
    train/bc.py's own eval computes on the same probabilities, since both go
    through decode_trace -> rules.score_playthrough. Any difference means
    the two paths have drifted and every logged eval number has become
    incomparable -- so it is checked on every run, not assumed.

    Also reports the mean over the first `n_in_training` excerpts, which are
    exactly the in-training eval's song set (build_val_set is deterministic
    and takes the first n of the same sorted list), so the run's logged
    hit_rate history can be compared against this script's numbers.
    """
    ours = np.array([r["hit_rate"] for r in per_excerpt])
    theirs = np.array(eval_hit_rates)
    max_diff = float(np.abs(ours - theirs).max()) if len(ours) else 0.0
    if max_diff > 1e-12:
        raise AssertionError(
            f"compare.py's default-threshold hit_rate disagrees with train.bc.score_eval_entry's "
            f"by up to {max_diff:.3e} -- the decode/score paths have drifted"
        )
    head = float(ours[:n_in_training].mean()) if len(ours) else 0.0
    log(f"reproduction check: matches train.bc.score_eval_entry exactly on all {len(ours)} excerpts; "
        f"first {min(n_in_training, len(ours))} excerpts (the in-training eval's set) mean "
        f"hit_rate = {head:.4f}")
    return dict(max_abs_diff_vs_score_eval_entry=max_diff,
                n_in_training_eval=min(n_in_training, len(ours)),
                in_training_eval_subset_hit_rate=head)


def resolve_run_config(run_dir: Path, name: str) -> Path:
    """The config copy the run trained with, else the repo default (same
    rule eval/evaluate.py uses)."""
    p = run_dir / name
    return p if p.exists() else Path("configs") / name


def fmt_diff(d: dict | None, width: int = 16, scale: float = 1.0, places: int = 4) -> str:
    if not d or d.get("mean") is None:
        return "".rjust(width)
    return f"{d['mean'] * scale:+.{places}f}±{d['se'] * scale:.{places}f}".rjust(width)


def print_report(out: dict, log=print) -> None:
    """One compact table per section; everything is also in compare.json."""
    b = out["baseline"]
    log("")
    log(f"=== compare: {out['difficulty']}, {out['n_excerpts']} val excerpts x {out['excerpt_s']:.0f}s "
        f"({out['device']}) ===")
    log(f"baseline: {b['path']} step {b['step']} @ strum {b['strum_threshold']:.2f} / fret {b['fret_threshold']:.2f}")
    log("")
    log(f"{'condition':<34}{'hit_rate':>9}{'Δ hit_rate (paired)':>21}{'score':>10}"
        f"{'Δ score':>19}{'ovr/min':>9}{'>base':>7}")
    for c in out["conditions"]:
        agg = c["aggregate"]
        name = f"step{c['step']} s{c['strum_threshold']:.2f} f{c['fret_threshold']:.2f}"
        log(f"{name:<34}{agg['hit_rate_mean']:>9.4f}{fmt_diff(agg.get('hit_rate_diff'), 21)}"
            f"{agg['score_mean']:>10.1f}{fmt_diff(agg.get('score_diff'), 19, places=1)}"
            f"{agg['overstrums_per_min_pooled']:>9.1f}"
            f"{str(agg.get('excerpts_better', '')) + '/' + str(agg['n_excerpts']):>7}")

    for c in out["conditions"]:
        if not c["show_breakdown"]:
            continue
        agg = c["aggregate"]
        log("")
        log(f"--- step{c['step']} s{c['strum_threshold']:.2f} f{c['fret_threshold']:.2f}: "
            f"by note category (pooled hit_rate; Δ paired over excerpts containing the category) ---")
        log(f"{'category':<15}{'n':>7}{'base':>8}{'cond':>8}{'Δ (paired)':>20}{'n exc':>7}")
        for cat in CATEGORIES:
            e = agg["by_category"][cat]
            if e["n"] == 0:
                continue
            base = e.get("baseline_hit_rate_pooled")
            base_s = f"{base:.3f}" if base is not None else "-"
            log(f"{cat:<15}{e['n']:>7}{base_s:>8}{e['hit_rate_pooled']:>8.3f}"
                f"{fmt_diff(e.get('diff'), 20, places=3)}{e.get('diff', {}).get('n', ''):>7}")

        log("")
        log(f"--- step{c['step']} s{c['strum_threshold']:.2f} f{c['fret_threshold']:.2f}: "
            f"misses by failure mode (count; Δ vs baseline) ---")
        header = f"{'category':<15}{'misses':>8}" + "".join(f"{m:>15}" for m in MISS_MODES)
        log(header)
        for cat in CATEGORIES:
            e = agg["by_category"][cat]
            if e["n"] == 0:
                continue
            misses = e["n"] - e["hits"]
            row = f"{cat:<15}{misses:>8}"
            for m in MISS_MODES:
                cell = str(e["miss_modes"][m])
                if "miss_modes_diff" in e:
                    cell += f" ({e['miss_modes_diff'][m]:+d})"
                row += f"{cell:>15}"
            log(row)
        ovr = f"{agg['overstrums_per_min_pooled']:.1f}/min"
        if agg.get("overstrums_per_min_diff"):
            ovr += f"  Δ {fmt_diff(agg['overstrums_per_min_diff'], 0, places=1)}"
        log(f"{'overstrums':<15}{'':>8}{ovr}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/compare.yaml")
    p.add_argument("--checkpoint", required=True, help="the checkpoint under test")
    p.add_argument("--baseline", default=None,
                   help="reference checkpoint for the paired differences "
                        "(default: checkpoint_best.pt in --checkpoint's run dir)")
    p.add_argument("--difficulty", default=None)
    p.add_argument("--n-songs", type=int, default=None,
                   help="max val songs (default from config: 30, which yields 28 on Medium). "
                        "Use 10 to reproduce the in-training eval's song set.")
    p.add_argument("--excerpt-s", type=float, default=None)
    p.add_argument("--strum-threshold", type=float, nargs="+", default=[STRUM_THRESHOLD])
    p.add_argument("--fret-threshold", type=float, nargs="+", default=[FRET_THRESHOLD])
    p.add_argument("--baseline-strum-threshold", type=float, default=STRUM_THRESHOLD)
    p.add_argument("--baseline-fret-threshold", type=float, default=FRET_THRESHOLD)
    p.add_argument("--device", default="cpu", help="cpu (default; leaves MPS free for a training run) or mps")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--run-name", default="compare")
    args = p.parse_args()

    cfg = load_config(args.config)
    difficulty = args.difficulty or cfg["difficulty"]
    n_songs = args.n_songs if args.n_songs is not None else cfg["n_songs"]
    excerpt_s = args.excerpt_s if args.excerpt_s is not None else cfg["excerpt_s"]
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(cfg["seed"])

    ckpt_path = Path(args.checkpoint).resolve()
    baseline_path = Path(args.baseline).resolve() if args.baseline else (ckpt_path.parent / "checkpoint_best.pt").resolve()
    run_dir = make_run_dir(args.run_name, args.config, cfg["seed"])
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run dir: {run_dir}")

    cfg_game = load_config(resolve_run_config(ckpt_path.parent, "game.yaml"))
    fps, hit_window_s = cfg_game["fps"], cfg_game["hit_window_s"]
    val = build_val_set(
        Path(cfg["processed_dir"]), [difficulty], n_songs, excerpt_s, cfg_game["chord_merge_min_gap_s"],
    )
    entries = val[difficulty]
    log(f"{difficulty}: {len(entries)} val excerpts (requested up to {n_songs}) x {excerpt_s:.0f}s")

    # Inference once per distinct checkpoint; every threshold cell reuses these.
    log("inference:")
    probs, notes, step, eval_hr = run_inference(ckpt_path, entries, cfg, device, dtype, log)
    if baseline_path == ckpt_path:
        base_probs, base_notes, base_step, base_eval_hr = probs, notes, step, eval_hr
    else:
        base_probs, base_notes, base_step, base_eval_hr = run_inference(
            baseline_path, entries, cfg, device, dtype, log
        )

    base_per_excerpt = [
        analyze_excerpt(pr, nt, fps, hit_window_s, args.baseline_strum_threshold, args.baseline_fret_threshold)
        for pr, nt in zip(base_probs, base_notes)
    ]
    base_agg = aggregate_condition(base_per_excerpt, None)

    repro = None
    if (args.baseline_strum_threshold, args.baseline_fret_threshold) == (STRUM_THRESHOLD, FRET_THRESHOLD):
        repro = check_reproduces_in_training_eval(base_per_excerpt, base_eval_hr, cfg["n_in_training_eval_songs"], log)

    conditions = []
    for st in args.strum_threshold:
        for ft in args.fret_threshold:
            per_excerpt = [analyze_excerpt(pr, nt, fps, hit_window_s, st, ft) for pr, nt in zip(probs, notes)]
            conditions.append(dict(
                path=str(ckpt_path), step=step, strum_threshold=st, fret_threshold=ft,
                aggregate=aggregate_condition(per_excerpt, base_per_excerpt),
                per_excerpt=[dict(song_id=e.song.song_id, title=e.song.title, **r)
                             for e, r in zip(entries, per_excerpt)],
            ))

    # Breakdowns for every condition when there are few; otherwise the
    # baseline-equivalent cell and the best-scoring cell, so a wide sweep
    # stays readable (the JSON always has all of them).
    if len(conditions) <= 3:
        for c in conditions:
            c["show_breakdown"] = True
    else:
        best = max(range(len(conditions)), key=lambda i: conditions[i]["aggregate"]["score_mean"])
        for i, c in enumerate(conditions):
            c["show_breakdown"] = i == best or (
                c["strum_threshold"] == args.baseline_strum_threshold
                and c["fret_threshold"] == args.baseline_fret_threshold
            )

    out = dict(
        config=cfg, difficulty=difficulty, n_excerpts=len(entries), excerpt_s=excerpt_s,
        device=str(device), dtype=args.dtype, reproduction_check=repro,
        songs=[dict(song_id=e.song.song_id, title=e.song.title, artist=e.song.artist,
                    excerpt_start_s=e.excerpt_start_s) for e in entries],
        baseline=dict(
            path=str(baseline_path), step=base_step,
            strum_threshold=args.baseline_strum_threshold, fret_threshold=args.baseline_fret_threshold,
            aggregate=base_agg,
            per_excerpt=[dict(song_id=e.song.song_id, title=e.song.title, **r)
                         for e, r in zip(entries, base_per_excerpt)],
        ),
        conditions=conditions,
    )
    print_report(out, log)
    (run_dir / "compare.json").write_text(json.dumps(out, indent=1))
    (run_dir / "compare.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {run_dir / 'compare.json'}")


if __name__ == "__main__":
    main()
