"""Build the trainable connectome graph from the FlyWire v783 connections
table and the flywire_annotations neuron table.

Connectome topology, synapse counts, and edge signs are fixed at this step
(invariant 3) — nothing here is trained. Writes `data/processed/graph.npz`
(numeric arrays) and `data/processed/graph_meta.json` (name lookups and the
provenance counts required by D11-D15), which `connectome/inspect.py` turns
into `docs/graph_report.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from flyhero.utils.config import load_config

NT_COLUMNS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
NT_NAMES = ["gaba", "ach", "glut", "oct", "ser", "da"]

# `top_nt` in the annotations TSV spells these out in full (verified against
# the live v3.1.0 file); the connections table's per-edge columns use the
# short codes above. Normalize to the short codes so both paths land in the
# same NT_SIGN vocabulary.
NT_NAME_ALIASES = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
    "octopamine": "oct",
    "serotonin": "ser",
    "dopamine": "da",
}

# D4 (ACh/GABA/Glu) + D13 (dopamine/serotonin/octopamine -> +1 for v1).
NT_SIGN = {
    "ach": 1,
    "gaba": -1,
    "glut": -1,
    "oct": 1,
    "ser": 1,
    "da": 1,
    "his": -1,  # D12: histamine override, not a predicted class
}

TYPE_LEVELS = ["cell_type", "hemibrain_type", "cell_class"]  # D14, then singleton
VOXEL_TO_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)


def load_root_ids(path: Path) -> np.ndarray:
    return np.load(path).astype(np.uint64)


def load_annotations(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", infer_schema_length=None)


def load_connections(path: Path) -> pl.DataFrame:
    return pl.read_ipc(path)


def assign_neuron_nt(
    annotations: pl.DataFrame,
    connections: pl.DataFrame,
    nt_conf_threshold: float,
    photoreceptor_cell_types: list[str],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return a per-neuron DataFrame with columns root_id, nt_name, nt_sign,
    nt_source ("predicted" | "fallback" | "histamine_override" | "default"),
    plus the D11-D13 report counts."""
    raw_top_nt = pl.col("top_nt")
    normalized_top_nt = (
        pl.when(raw_top_nt.is_null() | (raw_top_nt == ""))
        .then(None)
        .otherwise(raw_top_nt.replace(NT_NAME_ALIASES))
        .alias("top_nt")
    )
    ann = annotations.select(
        "root_id",
        normalized_top_nt,
        "top_nt_conf",
        pl.col("cell_type").alias("_cell_type"),
    )

    needs_fallback = (pl.col("top_nt").is_null()) | (
        pl.col("top_nt_conf").fill_null(0.0) < nt_conf_threshold
    )
    predicted = ann.with_columns(needs_fallback.alias("_needs_fallback"))

    # Fallback: synapse-weighted mean of each neuron's outgoing per-edge NT
    # columns (D11), computed once for every neuron regardless of whether it
    # ends up used, so we can report totals cleanly.
    weighted = connections.select(
        pl.col("pre_pt_root_id").alias("root_id"),
        *[(pl.col(c) * pl.col("syn_count")).alias(c) for c in NT_COLUMNS],
        "syn_count",
    )
    fallback_scores = weighted.group_by("root_id").agg(
        [(pl.col(c).sum() / pl.col("syn_count").sum()).alias(c) for c in NT_COLUMNS]
    )
    fallback_best = fallback_scores.with_columns(
        pl.concat_list(NT_COLUMNS).list.arg_max().alias("_argmax")
    ).with_columns(
        pl.Series("nt_fallback", [NT_NAMES[i] for i in fallback_scores.select(
            pl.concat_list(NT_COLUMNS).list.arg_max()
        ).to_series().to_list()])
    )
    fallback_lookup = fallback_best.select("root_id", "nt_fallback")

    merged = predicted.join(fallback_lookup, on="root_id", how="left")

    n_predicted = merged.filter(~pl.col("_needs_fallback")).height
    n_fallback_used = merged.filter(
        pl.col("_needs_fallback") & pl.col("nt_fallback").is_not_null()
    ).height
    n_no_data = merged.filter(
        pl.col("_needs_fallback") & pl.col("nt_fallback").is_null()
    ).height

    nt_name = (
        pl.when(~pl.col("_needs_fallback"))
        .then(pl.col("top_nt"))
        .when(pl.col("nt_fallback").is_not_null())
        .then(pl.col("nt_fallback"))
        .otherwise(pl.lit("ach"))  # default per D13-adjacent convention: excitatory
        .alias("nt_name")
    )
    nt_source = (
        pl.when(~pl.col("_needs_fallback"))
        .then(pl.lit("predicted"))
        .when(pl.col("nt_fallback").is_not_null())
        .then(pl.lit("fallback"))
        .otherwise(pl.lit("default"))
        .alias("nt_source")
    )
    result = merged.with_columns(nt_name, nt_source)

    # D12: histamine override for photoreceptors, regardless of prediction.
    is_photoreceptor = pl.col("_cell_type").is_in(photoreceptor_cell_types)
    n_photoreceptor = result.filter(is_photoreceptor).height
    original_nt_for_photoreceptors = (
        result.filter(is_photoreceptor)
        .group_by("nt_name")
        .agg(pl.len().alias("count"))
        .to_dicts()
    )
    result = result.with_columns(
        pl.when(is_photoreceptor).then(pl.lit("his")).otherwise(pl.col("nt_name")).alias("nt_name"),
        pl.when(is_photoreceptor)
        .then(pl.lit("histamine_override"))
        .otherwise(pl.col("nt_source"))
        .alias("nt_source"),
    )

    result = result.with_columns(
        pl.col("nt_name").replace_strict(NT_SIGN, default=1).alias("nt_sign")
    )

    per_transmitter_neurons = (
        result.group_by("nt_name").agg(pl.len().alias("neuron_count")).to_dicts()
    )

    report = {
        "nt_conf_threshold": nt_conf_threshold,
        "n_predicted": n_predicted,
        "n_fallback_used": n_fallback_used,
        "n_no_data_default_ach": n_no_data,
        "n_histamine_override": n_photoreceptor,
        "histamine_override_original_top_nt": original_nt_for_photoreceptors,
        "per_transmitter_neuron_counts": per_transmitter_neurons,
    }
    return result.select("root_id", "nt_name", "nt_sign", "nt_source"), report


def assign_type_id(annotations: pl.DataFrame) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """D14 fallback chain: cell_type -> hemibrain_type -> cell_class ->
    singleton. Returns (type_id per row, type_names lookup, level counts)."""
    chosen = pl.lit(None, dtype=pl.Utf8)
    level_col = pl.lit(None, dtype=pl.Utf8)
    for level in TYPE_LEVELS:
        col = pl.col(level).cast(pl.Utf8)
        is_valid = col.is_not_null() & (col.str.len_chars() > 0)
        chosen = pl.when(chosen.is_not_null()).then(chosen).when(is_valid).then(col).otherwise(chosen)
        level_col = (
            pl.when(level_col.is_not_null())
            .then(level_col)
            .when(is_valid)
            .then(pl.lit(level))
            .otherwise(level_col)
        )

    df = annotations.with_columns(chosen.alias("_type_name"), level_col.alias("_type_level"))
    singleton_name = pl.lit("singleton:") + pl.col("root_id").cast(pl.Utf8)
    df = df.with_columns(
        pl.when(pl.col("_type_name").is_not_null())
        .then(pl.col("_type_name"))
        .otherwise(singleton_name)
        .alias("_type_name"),
        pl.when(pl.col("_type_level").is_not_null())
        .then(pl.col("_type_level"))
        .otherwise(pl.lit("singleton"))
        .alias("_type_level"),
    )

    level_counts = (
        df.group_by("_type_level").agg(pl.len().alias("count")).to_dicts()
    )

    names = df.select("_type_name").to_series().to_list()
    unique_names = sorted(set(names))
    name_to_id = {name: i for i, name in enumerate(unique_names)}
    type_id = np.array([name_to_id[n] for n in names], dtype=np.int64)

    level_count_map = {row["_type_level"]: row["count"] for row in level_counts}
    return type_id, unique_names, level_count_map


def resolve_positions(annotations: pl.DataFrame, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron position, preferring soma_x/y/z and falling back to the
    pos_x/y/z anchor point where soma is null (e.g. photoreceptors, whose
    somata sit in the retina outside the FAFB volume). A neuron-level
    fallback, not a column-level one: soma_x/y/z exists as a column for
    every neuron, but is null row-by-row for many of them."""
    has_soma_cols = all(c in annotations.columns for c in ("soma_x", "soma_y", "soma_z"))
    has_pos_cols = all(c in annotations.columns for c in ("pos_x", "pos_y", "pos_z"))

    result = np.full((n_nodes, 3), np.nan, dtype=np.float64)
    source = np.full(n_nodes, "none", dtype=object)

    if has_soma_cols:
        soma_arr = annotations.select(["soma_x", "soma_y", "soma_z"]).to_numpy().astype(np.float64)
        soma_valid = ~np.isnan(soma_arr).any(axis=1)
        result[soma_valid] = soma_arr[soma_valid]
        source[soma_valid] = "soma"

    if has_pos_cols:
        pos_arr = annotations.select(["pos_x", "pos_y", "pos_z"]).to_numpy().astype(np.float64)
        pos_valid = ~np.isnan(pos_arr).any(axis=1)
        needs_pos = (source == "none") & pos_valid
        result[needs_pos] = pos_arr[needs_pos]
        source[needs_pos] = "anchor"

    return result, source


def build_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    min_syn = cfg["min_syn"]
    nt_conf_threshold = cfg["nt_conf_threshold"]
    input_layer_cfg = cfg["input_layer"]
    t4t5_cell_types = cfg["t4t5_cell_types"]

    root_ids = load_root_ids(raw_dir / cfg["sources"]["root_ids"]["filename"])
    annotations_raw = load_annotations(raw_dir / cfg["sources"]["annotations"]["filename"])
    connections_raw = load_connections(raw_dir / cfg["sources"]["connections"]["filename"])

    proofread_set = pl.DataFrame({"root_id": root_ids.astype(np.int64)})
    annotations = annotations_raw.join(proofread_set, on="root_id", how="inner").unique(
        subset=["root_id"], keep="first"
    )

    n_proofread_raw = len(root_ids)
    n_proofread_annotated = annotations.height

    node_ids = annotations.select("root_id").to_series().to_numpy().astype(np.int64)
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    annotations = annotations[order.tolist()]
    id_to_index = {int(rid): i for i, rid in enumerate(node_ids)}

    connections = connections_raw.filter(
        pl.col("pre_pt_root_id").is_in(node_ids.tolist())
        & pl.col("post_pt_root_id").is_in(node_ids.tolist())
    )

    nt_per_neuron, nt_report = assign_neuron_nt(
        annotations, connections, nt_conf_threshold, input_layer_cfg["photoreceptor_cell_types"]
    )
    sign_lookup = dict(
        zip(
            nt_per_neuron.select("root_id").to_series().to_list(),
            nt_per_neuron.select("nt_sign").to_series().to_list(),
        )
    )

    aggregated = connections.group_by(["pre_pt_root_id", "post_pt_root_id"]).agg(
        pl.col("syn_count").sum().alias("syn_count")
    )
    aggregated = aggregated.filter(pl.col("syn_count") >= min_syn)

    pre_ids = aggregated.select("pre_pt_root_id").to_series().to_list()
    post_ids = aggregated.select("post_pt_root_id").to_series().to_list()
    syn_count = np.array(aggregated.select("syn_count").to_series().to_list(), dtype=np.int64)

    pre = np.array([id_to_index[p] for p in pre_ids], dtype=np.int64)
    post = np.array([id_to_index[p] for p in post_ids], dtype=np.int64)
    sign = np.array([sign_lookup.get(p, 1) for p in pre_ids], dtype=np.int8)

    nt_name_lookup = dict(
        zip(
            nt_per_neuron.select("root_id").to_series().to_list(),
            nt_per_neuron.select("nt_name").to_series().to_list(),
        )
    )
    edge_nt_names = [nt_name_lookup.get(p, "ach") for p in pre_ids]
    per_transmitter_edge_counts: dict[str, int] = {}
    for name in edge_nt_names:
        per_transmitter_edge_counts[name] = per_transmitter_edge_counts.get(name, 0) + 1
    nt_report["per_transmitter_edge_counts"] = [
        {"nt_name": k, "edge_count": v} for k, v in sorted(per_transmitter_edge_counts.items())
    ]

    type_id, type_names, type_level_counts = assign_type_id(annotations)

    pair_keys = list(zip(type_id[pre].tolist(), type_id[post].tolist()))
    unique_pairs = sorted(set(pair_keys))
    pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
    pair_id = np.array([pair_to_id[k] for k in pair_keys], dtype=np.int64)

    n_nodes = len(node_ids)
    inv_norm = np.zeros(n_nodes, dtype=np.float32)
    incoming_totals = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(incoming_totals, post, syn_count)
    nonzero = incoming_totals > 0
    inv_norm[nonzero] = 1.0 / incoming_totals[nonzero]

    super_class_vals = annotations.select("super_class").to_series().fill_null("unknown").to_list()
    unique_super_classes = sorted(set(super_class_vals))
    super_class_to_id = {name: i for i, name in enumerate(unique_super_classes)}
    super_class_id = np.array([super_class_to_id[v] for v in super_class_vals], dtype=np.int64)

    dn_idx = np.nonzero(super_class_id == super_class_to_id.get("descending", -1))[0].astype(np.int64)

    cell_types = annotations.select("cell_type").to_series().fill_null("").to_list()
    photoreceptor_set = set(input_layer_cfg["photoreceptor_cell_types"])
    lamina_set = set(input_layer_cfg["fallback_cell_types"])
    input_idx_photoreceptor = np.array(
        [i for i, ct in enumerate(cell_types) if ct in photoreceptor_set], dtype=np.int64
    )
    input_idx_lamina = np.array(
        [i for i, ct in enumerate(cell_types) if ct in lamina_set], dtype=np.int64
    )
    t4t5_set = set(t4t5_cell_types)
    t4t5_idx = np.array([i for i, ct in enumerate(cell_types) if ct in t4t5_set], dtype=np.int64)

    pos_voxel, pos_source = resolve_positions(annotations, n_nodes)
    pos_nm = pos_voxel * VOXEL_TO_NM  # D15

    pos_source_counts = {name: int((pos_source == name).sum()) for name in ("soma", "anchor", "none")}
    photoreceptor_mask = np.isin(np.arange(n_nodes), input_idx_photoreceptor)
    pos_source_counts_photoreceptor = {
        name: int(((pos_source == name) & photoreceptor_mask).sum()) for name in ("soma", "anchor", "none")
    }
    pos_source_code = {"soma": 0, "anchor": 1, "none": 2}
    pos_source_id = np.array([pos_source_code[s] for s in pos_source], dtype=np.int8)

    photoreceptor_side_counts: dict[str, int] = {}
    sides = (
        annotations.select("side").to_series().fill_null("na").to_list()
        if "side" in annotations.columns
        else ["na"] * n_nodes
    )
    for ct, side in zip(cell_types, sides):
        if ct in photoreceptor_set:
            key = f"{ct}_{side}"
            photoreceptor_side_counts[key] = photoreceptor_side_counts.get(key, 0) + 1

    # Not topology (invariant 3 only fixes topology/synapse counts/edge
    # signs) -- per-node metadata used by Phase 2's retina.py to split
    # photoreceptors into left/right eye groups for the visual-field
    # projection (the raw annotations have no column/ommatidia ID, per Phase
    # 2's retina design).
    side_code = {"left": 0, "right": 1, "center": 2, "na": 3}
    side_id = np.array([side_code.get(s, 3) for s in sides], dtype=np.int8)

    pair_counts = np.bincount(pair_id)
    n_edges_total = len(pre)
    edge_sign_report = {
        "n_edges": int(n_edges_total),
        "n_sign_negative": int((sign == -1).sum()),
        "n_sign_positive": int((sign == 1).sum()),
    }
    pair_size_distribution = []
    for thresh in (1, 2, 5, 10, 20):
        fine_mask = pair_counts >= thresh
        pair_size_distribution.append(
            {
                "min_edges": thresh,
                "n_pairs": int(fine_mask.sum()),
                "n_edges_covered": int(pair_counts[fine_mask].sum()),
            }
        )

    # D16: exact (not just the 5 reported thresholds) search for the largest
    # k with fine-pair edge coverage >= 90%, plus the hybrid parameter count
    # at that k (fine pairs keep their own gain; the rest share a coarse
    # (pre_type, post_super_class) bucket).
    max_k_candidate = int(pair_counts.max())
    best_k, best_coverage = 1, 1.0
    for k in range(1, max_k_candidate + 1):
        coverage = pair_counts[pair_counts >= k].sum() / n_edges_total
        if coverage >= 0.90:
            best_k, best_coverage = k, float(coverage)
        else:
            break
    fine_mask_at_k = pair_counts >= best_k
    n_fine_pairs_at_k = int(fine_mask_at_k.sum())
    edge_is_fine_at_k = fine_mask_at_k[pair_id]
    leftover_edges = np.nonzero(~edge_is_fine_at_k)[0]
    coarse_keys = set(
        zip(type_id[pre[leftover_edges]].tolist(), super_class_id[post[leftover_edges]].tolist())
    )
    hybrid_gain_sharing = {
        "k": best_k,
        "edge_coverage_at_k": best_coverage,
        "edge_coverage_at_k_plus_1": (
            float(pair_counts[pair_counts >= best_k + 1].sum() / n_edges_total)
            if best_k + 1 <= max_k_candidate
            else 0.0
        ),
        "n_fine_pairs": n_fine_pairs_at_k,
        "n_coarse_buckets": len(coarse_keys),
        "n_total_gain_params": n_fine_pairs_at_k + len(coarse_keys),
        "n_pure_fine_params": int(len(unique_pairs)),
        "n_pure_coarse_params": len(
            set(zip(type_id[pre].tolist(), super_class_id[post].tolist()))
        ),
    }

    np.savez(
        processed_dir / "graph.npz",
        root_id=node_ids.astype(np.uint64),
        pre=pre,
        post=post,
        syn_count=syn_count,
        sign=sign,
        pair_id=pair_id,
        type_id=type_id,
        super_class_id=super_class_id,
        inv_norm=inv_norm,
        dn_idx=dn_idx,
        input_idx_photoreceptor=input_idx_photoreceptor,
        input_idx_lamina=input_idx_lamina,
        t4t5_idx=t4t5_idx,
        pos_voxel=pos_voxel.astype(np.float32),
        pos_nm=pos_nm.astype(np.float32),
        pos_source_id=pos_source_id,  # 0=soma, 1=anchor, 2=none
        side_id=side_id,  # 0=left, 1=right, 2=center, 3=na
    )

    meta = {
        "n_proofread_raw": n_proofread_raw,
        "n_proofread_annotated": n_proofread_annotated,
        "n_nodes": n_nodes,
        "n_edges": int(len(pre)),
        "n_type_pairs": len(unique_pairs),
        "n_dn": int(len(dn_idx)),
        "n_input_photoreceptor": int(len(input_idx_photoreceptor)),
        "n_input_lamina": int(len(input_idx_lamina)),
        "n_t4t5": int(len(t4t5_idx)),
        "type_names": type_names,
        "type_level_counts": type_level_counts,
        "super_class_names": unique_super_classes,
        "min_syn": min_syn,
        "nt_report": nt_report,
        "pos_units": {
            "pos_voxel": "raw FlyWire soma/anchor voxel coordinates, resolution 4x4x40 nm/voxel",
            "pos_nm": "pos_voxel scaled by (4, 4, 40) to nanometers",
        },
        "pos_source_counts": pos_source_counts,
        "pos_source_counts_photoreceptor": pos_source_counts_photoreceptor,
        "photoreceptor_side_counts": photoreceptor_side_counts,
        "side_counts": {name: int((side_id == code).sum()) for name, code in side_code.items()},
        "edge_sign_report": edge_sign_report,
        "pair_size_distribution": pair_size_distribution,
        "hybrid_gain_sharing": hybrid_gain_sharing,
    }
    with (processed_dir / "graph_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    meta = build_graph(cfg)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("type_names",)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
