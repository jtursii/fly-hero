"""Phase 1 tests: graph build round-trip and degree-preserving shuffle.

Uses tiny hand-built connections/annotations tables (no real download needed)
so these run fast and don't depend on the ~900 MB Zenodo/GitHub sources.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flyhero.connectome.build_graph import assign_neuron_nt, assign_type_id, build_graph
from flyhero.connectome.shuffle import degree_preserving_shuffle

ROOT_IDS = [1, 2, 3, 4, 5, 6]


def _write_fixtures(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    root_ids = np.array(ROOT_IDS, dtype=np.uint64)
    np.save(raw_dir / "root_ids.npy", root_ids)

    annotations = pl.DataFrame(
        {
            "root_id": ROOT_IDS,
            "cell_type": ["R1-6", "T4a", "PN1", None, None, "R7"],
            "hemibrain_type": [None, None, None, "HB1", None, None],
            "cell_class": [None, None, None, None, "some_class", None],
            "super_class": ["sensory", "optic", "central", "descending", "central", "sensory"],
            "top_nt": ["ach", "ach", "gaba", "ach", None, "glut"],
            "top_nt_conf": [0.9, 0.9, 0.9, 0.9, None, 0.9],
            "soma_x": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "soma_y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "soma_z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    annotations.write_csv(raw_dir / "annotations.tsv", separator="\t")

    # neuron 5 has no top_nt/top_nt_conf -> must fall back to its outgoing
    # edges (all glut) weighted by syn_count.
    connections = pl.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 3, 5, 5],
            "post_pt_root_id": [2, 2, 3, 4, 6, 6],
            "neuropil": ["AL_L", "AL_R", "LAL_L", "CRE_L", "AL_L", "AL_R"],
            "syn_count": [3, 4, 10, 2, 8, 2],
            "gaba_avg": [0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
            "ach_avg": [0.7, 0.7, 0.7, 0.7, 0.05, 0.05],
            "glut_avg": [0.1, 0.1, 0.1, 0.1, 0.8, 0.8],
            "oct_avg": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            "ser_avg": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            "da_avg": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        }
    )
    connections.write_ipc(raw_dir / "connections.feather")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    cfg = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "min_syn": 5,
        "nt_conf_threshold": 0.5,
        "sources": {
            "root_ids": {"filename": "root_ids.npy"},
            "annotations": {"filename": "annotations.tsv"},
            "connections": {"filename": "connections.feather"},
        },
        "input_layer": {
            "photoreceptor_cell_types": ["R1-6", "R7", "R8"],
            "fallback_cell_types": ["L1", "L2", "L3"],
        },
        "t4t5_cell_types": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
    }
    return cfg, processed_dir


def test_build_graph_index_round_trip(tmp_path):
    cfg, processed_dir = _write_fixtures(tmp_path)

    meta = build_graph(cfg)

    graph = np.load(processed_dir / "graph.npz")

    assert meta["n_nodes"] == 6
    assert list(graph["root_id"]) == ROOT_IDS

    # (1,2) pair: neuropils AL_L(3) + AL_R(4) = 7 >= min_syn=5, survives.
    # (2,3): 10 >= 5, survives. (3,4): 2 < 5, dropped. (5,6): 8+2=10, survives.
    root_to_idx = {rid: i for i, rid in enumerate(graph["root_id"])}
    edges = set(zip(graph["pre"].tolist(), graph["post"].tolist()))
    assert (root_to_idx[1], root_to_idx[2]) in edges
    assert (root_to_idx[2], root_to_idx[3]) in edges
    assert (root_to_idx[5], root_to_idx[6]) in edges
    assert (root_to_idx[3], root_to_idx[4]) not in edges

    # Neuron 2's predicted NT is ach -> sign +1 on its outgoing edge.
    idx_2 = root_to_idx[2]
    idx_3 = root_to_idx[3]
    e = [
        (p, q, s)
        for p, q, s in zip(graph["pre"].tolist(), graph["post"].tolist(), graph["sign"].tolist())
        if p == idx_2 and q == idx_3
    ]
    assert e and e[0][2] == 1

    # Neuron 1 (R1-6) gets the histamine override -> sign -1, despite ach prediction.
    idx_1 = root_to_idx[1]
    e = [
        s
        for p, q, s in zip(graph["pre"].tolist(), graph["post"].tolist(), graph["sign"].tolist())
        if p == idx_1
    ]
    assert all(s == -1 for s in e)

    # Neuron 5 has no top_nt -> falls back to its outgoing edges, which are
    # dominated by glut_avg (0.8) -> glut -> sign -1.
    idx_5 = root_to_idx[5]
    e = [
        s
        for p, q, s in zip(graph["pre"].tolist(), graph["post"].tolist(), graph["sign"].tolist())
        if p == idx_5
    ]
    assert e and all(s == -1 for s in e)

    assert meta["n_input_photoreceptor"] == 2  # R1-6 and R7
    assert meta["n_dn"] == 1
    assert meta["nt_report"]["n_fallback_used"] == 1
    assert meta["nt_report"]["n_histamine_override"] == 2

    # D14 type fallback: neuron 4 has no cell_type but has hemibrain_type.
    idx_4 = root_to_idx[4]
    type_names = meta["type_names"]
    assert type_names[graph["type_id"][idx_4]] == "HB1"
    # neuron 5 has no cell_type/hemibrain_type but does have cell_class -> falls
    # through to the third level of the chain.
    idx_5_type = type_names[graph["type_id"][idx_5]]
    assert idx_5_type == "some_class"

    # D15: position stored in both voxel and nm, nm = voxel * (4,4,40).
    np.testing.assert_allclose(
        graph["pos_nm"][idx_1], graph["pos_voxel"][idx_1] * np.array([4, 4, 40]), rtol=1e-5
    )


def test_assign_type_id_falls_back_to_singleton():
    annotations = pl.DataFrame(
        {
            "root_id": [1, 2],
            "cell_type": ["PN1", None],
            "hemibrain_type": [None, None],
            "cell_class": [None, None],
        }
    )

    type_id, type_names, level_counts = assign_type_id(annotations)

    assert type_names[type_id[0]] == "PN1"
    assert type_names[type_id[1]] == "singleton:2"
    assert level_counts["cell_type"] == 1
    assert level_counts["singleton"] == 1


def test_top_nt_full_word_spellings_normalize_to_correct_sign():
    # The real annotations TSV spells top_nt out in full ("glutamate", not
    # "glut"); this must still resolve to sign -1, not fall through to a
    # default +1.
    annotations = pl.DataFrame(
        {
            "root_id": [1, 2],
            "cell_type": ["PN1", "PN2"],
            "top_nt": ["glutamate", "acetylcholine"],
            "top_nt_conf": [0.95, 0.95],
        }
    )
    connections = pl.DataFrame(
        {
            "pre_pt_root_id": [],
            "post_pt_root_id": [],
            "syn_count": [],
            "gaba_avg": [],
            "ach_avg": [],
            "glut_avg": [],
            "oct_avg": [],
            "ser_avg": [],
            "da_avg": [],
        },
        schema={
            "pre_pt_root_id": pl.Int64,
            "post_pt_root_id": pl.Int64,
            "syn_count": pl.Int64,
            "gaba_avg": pl.Float64,
            "ach_avg": pl.Float64,
            "glut_avg": pl.Float64,
            "oct_avg": pl.Float64,
            "ser_avg": pl.Float64,
            "da_avg": pl.Float64,
        },
    )

    result, report = assign_neuron_nt(annotations, connections, 0.5, [])

    signs = dict(zip(result["root_id"].to_list(), result["nt_sign"].to_list()))
    assert signs[1] == -1  # glutamate -> glut -> -1
    assert signs[2] == 1  # acetylcholine -> ach -> +1
    assert report["n_predicted"] == 2


def test_degree_preserving_shuffle_preserves_degrees():
    rng = np.random.default_rng(0)
    pre = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    post = np.array([1, 2, 2, 3, 3, 0, 0, 1], dtype=np.int64)

    shuffled_post = degree_preserving_shuffle(pre, post, rng, n_swaps_factor=50)

    def degrees(pre_arr, post_arr):
        n = max(pre_arr.max(), post_arr.max()) + 1
        out_deg = np.bincount(pre_arr, minlength=n)
        in_deg = np.bincount(post_arr, minlength=n)
        return out_deg, in_deg

    out_before, in_before = degrees(pre, post)
    out_after, in_after = degrees(pre, shuffled_post)

    np.testing.assert_array_equal(out_before, out_after)
    np.testing.assert_array_equal(in_before, in_after)
    # No duplicate edges introduced, no self-loops.
    edges = list(zip(pre.tolist(), shuffled_post.tolist()))
    assert len(edges) == len(set(edges))
    assert all(p != q for p, q in edges)


def test_shuffle_actually_changes_targets():
    rng = np.random.default_rng(1)
    n = 30
    pre = np.repeat(np.arange(n), 3)
    post = (pre + rng.integers(1, n, size=len(pre))) % n

    shuffled_post = degree_preserving_shuffle(pre, post, rng, n_swaps_factor=20)

    assert not np.array_equal(post, shuffled_post)
