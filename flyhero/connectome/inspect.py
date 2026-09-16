"""Render `data/processed/graph.npz` + `graph_meta.json` into
`docs/graph_report.md`: the human-readable record of Gate G1 numbers and the
D11-D15 provenance counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from flyhero.connectome.build_graph import NT_SIGN
from flyhero.utils.config import load_config


def choose_input_layer(meta: dict[str, Any]) -> tuple[str, int]:
    """PLAN Phase 1 step 3: photoreceptors if coverage is reasonable,
    otherwise lamina L1-L3. 'Reasonable' = the input-neuron floor for Gate
    G1 (>= 500)."""
    if meta["n_input_photoreceptor"] >= 500:
        return "photoreceptor", meta["n_input_photoreceptor"]
    return "lamina", meta["n_input_lamina"]


def render_report(meta: dict[str, Any]) -> str:
    input_layer, n_input = choose_input_layer(meta)
    nt = meta["nt_report"]

    lines = [
        "# Connectome Graph Report (Phase 1 / Gate G1)",
        "",
        "## Neuron and edge counts",
        f"- Proofread root IDs (raw, `proofread_root_ids_783.npy`): {meta['n_proofread_raw']:,}",
        f"- Proofread neurons with an annotation row (final node set): {meta['n_proofread_annotated']:,}",
        f"- Nodes in graph.npz: {meta['n_nodes']:,}",
        f"- Edges (after `min_syn={meta['min_syn']}` threshold, summed across neuropils): {meta['n_edges']:,}",
        f"- Type pairs: {meta['n_type_pairs']:,}",
        f"- Descending neurons (DN): {meta['n_dn']:,}",
        "",
        "## Input layer choice",
        f"- Photoreceptor-matched neurons (R1-6/R7/R8, both sides): {meta['n_input_photoreceptor']:,}",
        f"- Lamina-matched neurons (L1-L3, both sides): {meta['n_input_lamina']:,}",
        f"- **Chosen input layer: {input_layer}** ({n_input:,} neurons; gate floor is 500).",
        f"- T4/T5 neurons matched: {meta['n_t4t5']:,}",
        "",
        "## Cell-type assignment (D14: cell_type -> hemibrain_type -> cell_class -> singleton)",
    ]
    for level in ["cell_type", "hemibrain_type", "cell_class", "singleton"]:
        count = meta["type_level_counts"].get(level, 0)
        lines.append(f"- {level}: {count:,} neurons")
    lines += [
        f"- Total distinct types: {len(meta['type_names']):,}",
        "",
        "## Neurotransmitter / edge sign assignment (D11-D13)",
        f"- Confidence threshold for using the predicted `top_nt`: {nt['nt_conf_threshold']}",
        f"- Predicted `top_nt` used directly: {nt['n_predicted']:,} neurons",
        f"- Fallback (synapse-weighted mean of outgoing edges) used: {nt['n_fallback_used']:,} neurons",
        f"- No data available for either predicted or fallback NT (defaulted to ACh/+1): {nt['n_no_data_default_ach']:,} neurons",
        f"- Histamine override applied (photoreceptors, D12): {nt['n_histamine_override']:,} neurons",
        "  - What `top_nt` originally predicted for these neurons before the override:",
    ]
    for row in nt["histamine_override_original_top_nt"]:
        lines.append(f"    - {row['nt_name']}: {row['count']:,}")
    lines += ["", "- Per-transmitter neuron counts (post-override, sign in parentheses):"]
    for row in nt["per_transmitter_neuron_counts"]:
        sign = NT_SIGN.get(row["nt_name"], 1)
        lines.append(f"  - {row['nt_name']} ({sign:+d}): {row['neuron_count']:,}")
    lines += ["", "- Per-transmitter edge counts (D13):"]
    for row in nt.get("per_transmitter_edge_counts", []):
        sign = NT_SIGN.get(row["nt_name"], 1)
        lines.append(f"  - {row['nt_name']} ({sign:+d}): {row['edge_count']:,}")
    lines += [
        "",
        "## Positions (D15)",
        f"- `pos_voxel`: {meta['pos_units']['pos_voxel']}",
        f"- `pos_nm`: {meta['pos_units']['pos_nm']}",
        "",
        "## Gate G1 checklist",
        f"- Proofread neurons ~138.6k? Actual final node count: {meta['n_proofread_annotated']:,}",
        f"- Input neurons >= 500: {'PASS' if n_input >= 500 else 'FAIL'} ({n_input:,})",
        f"- DNs >= 100: {'PASS' if meta['n_dn'] >= 100 else 'FAIL'} ({meta['n_dn']:,})",
        f"- Type pairs > 300k fallback needed?: {'YES - apply fallback' if meta['n_type_pairs'] > 300_000 else 'no'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="docs/graph_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["processed_dir"])
    with (processed_dir / "graph_meta.json").open() as f:
        meta = json.load(f)

    report = render_report(meta)
    Path(args.out).write_text(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
