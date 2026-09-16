"""Phase 3a tests: gain-group derivation (D16) and ConnectomeBrain's forward/
backward correctness.

Uses a tiny hand-built graph (no real data needed) for the fast tests, same
idiom as tests/test_connectome.py, plus real-data regression/alignment tests
gated behind a skipif on data/processed/graph.npz's existence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.func import functional_call

from flyhero.brain.rate_model import ConnectomeBrain, compute_gain_group_id
from flyhero.brain.readout import Readout

REAL_GRAPH_PATH = Path("data/processed/graph.npz")
REAL_GRAPH_META_PATH = Path("data/processed/graph_meta.json")

_BRAIN_CFG = dict(
    dt=1 / 120,
    tau_min_s=0.02,
    tau_init_s=0.05,
    bias_init=0.2,
    gain_sharing="hybrid",
    gain_sharing_k=2,
    gain0=1.0,
)


def _tiny_graph() -> dict:
    """12 nodes, 3 type_ids, 3 super_classes (0=sensory/input, 1=central,
    2=descending). Edges are chosen so that under k=2: pairs (0,1) and (1,2)
    each have 3 edges (stay fine), while pairs (2,2), (0,2), (1,1) each have
    exactly 1 edge (must fall back to their coarse (pre_type,
    post_super_class) bucket) -- exercising the full hybrid path."""
    n_nodes = 12
    type_id = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
    super_class_id = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)

    pre = np.array([0, 1, 2, 3, 4, 6, 2, 3, 5], dtype=np.int64)
    post = np.array([3, 4, 5, 6, 7, 9, 10, 4, 8], dtype=np.int64)
    sign = np.array([1, 1, 1, 1, -1, -1, 1, -1, 1], dtype=np.int8)
    syn_count = np.array([5, 3, 4, 6, 2, 2, 3, 2, 1], dtype=np.int64)

    # Mirrors build_graph.py's pair_id construction: enumerate unique
    # (type_id[pre], type_id[post]) tuples, sorted, densely numbered.
    pair_keys = list(zip(type_id[pre].tolist(), type_id[post].tolist()))
    unique_pairs = sorted(set(pair_keys))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    pair_id = np.array([pair_to_id[k] for k in pair_keys], dtype=np.int64)

    incoming = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(incoming, post, syn_count)
    inv_norm = np.zeros(n_nodes, dtype=np.float32)
    nonzero = incoming > 0
    inv_norm[nonzero] = 1.0 / incoming[nonzero]

    dn_idx = np.array([9, 10, 11], dtype=np.int64)
    input_idx_photoreceptor = np.array([0, 1, 2], dtype=np.int64)

    return dict(
        pre=pre,
        post=post,
        sign=sign,
        syn_count=syn_count,
        pair_id=pair_id,
        type_id=type_id,
        super_class_id=super_class_id,
        inv_norm=inv_norm,
        dn_idx=dn_idx,
        input_idx_photoreceptor=input_idx_photoreceptor,
    )


def test_compute_gain_group_id_hybrid_k2_on_tiny_graph():
    graph = _tiny_graph()
    gain_group_id, n_groups = compute_gain_group_id(
        graph["pair_id"],
        graph["type_id"],
        graph["pre"],
        graph["post"],
        graph["super_class_id"],
        scheme="hybrid",
        k=2,
    )
    # edges 0,1,2 are pair (0,1) (3 edges) -- fine, share one group.
    assert gain_group_id[0] == gain_group_id[1] == gain_group_id[2]
    # edges 3,4,8 are pair (1,2) (3 edges) -- fine, share a different group.
    assert gain_group_id[3] == gain_group_id[4] == gain_group_id[8]
    assert gain_group_id[0] != gain_group_id[3]
    # edges 5 (pair (2,2)), 6 (pair (0,2)), 7 (pair (1,1)) each have exactly
    # 1 edge -- each falls back to its own distinct coarse bucket.
    coarse_groups = {int(gain_group_id[5]), int(gain_group_id[6]), int(gain_group_id[7])}
    assert len(coarse_groups) == 3
    assert coarse_groups.isdisjoint({int(gain_group_id[0]), int(gain_group_id[3])})
    assert n_groups == 5  # 2 fine + 3 coarse


def test_compute_gain_group_id_k_threshold_is_inclusive():
    # 2 nodes, 1 pair with exactly 2 edges (must stay fine at k=2) and 1
    # pair with 1 edge (must merge to coarse).
    type_id = np.array([0, 1], dtype=np.int64)
    super_class_id = np.array([5, 7], dtype=np.int64)
    pre = np.array([0, 0, 1], dtype=np.int64)
    post = np.array([1, 1, 0], dtype=np.int64)
    pair_id = np.array([0, 0, 1], dtype=np.int64)

    gain_group_id, n_groups = compute_gain_group_id(
        pair_id, type_id, pre, post, super_class_id, scheme="hybrid", k=2
    )
    assert gain_group_id[0] == gain_group_id[1]
    assert gain_group_id[2] != gain_group_id[0]
    assert n_groups == 2


def test_compute_gain_group_id_fine_and_coarse_schemes():
    graph = _tiny_graph()
    fine_ids, n_fine = compute_gain_group_id(
        graph["pair_id"], graph["type_id"], graph["pre"], graph["post"],
        graph["super_class_id"], scheme="fine", k=2,
    )
    np.testing.assert_array_equal(fine_ids, graph["pair_id"])
    assert n_fine == len(set(graph["pair_id"].tolist()))

    coarse_ids, n_coarse = compute_gain_group_id(
        graph["pair_id"], graph["type_id"], graph["pre"], graph["post"],
        graph["super_class_id"], scheme="coarse", k=2,
    )
    # edges 0 and 6 both have pre type_id 0, but different post super_class
    # (edge0 post=3 -> central=1, edge6 post=10 -> descending=2) -> distinct.
    assert coarse_ids[0] != coarse_ids[6]
    assert n_coarse == len(set(zip(graph["type_id"][graph["pre"]].tolist(),
                                    graph["super_class_id"][graph["post"]].tolist())))


@pytest.mark.skipif(
    not (REAL_GRAPH_PATH.exists() and REAL_GRAPH_META_PATH.exists()),
    reason="requires the real graph.npz/graph_meta.json built from FlyWire v783 data",
)
def test_gain_group_id_hybrid_k2_matches_real_graph_report():
    graph = dict(np.load(REAL_GRAPH_PATH))
    meta = json.loads(REAL_GRAPH_META_PATH.read_text())
    expected = meta["hybrid_gain_sharing"]
    assert expected["k"] == 2

    pair_counts = np.bincount(graph["pair_id"])
    n_fine_independent = int((pair_counts >= 2).sum())
    assert n_fine_independent == expected["n_fine_pairs"]

    _, n_groups = compute_gain_group_id(
        graph["pair_id"],
        graph["type_id"],
        graph["pre"],
        graph["post"],
        graph["super_class_id"],
        scheme="hybrid",
        k=2,
    )
    assert n_groups == expected["n_total_gain_params"]
    assert n_groups - n_fine_independent == expected["n_coarse_buckets"]


@pytest.mark.skipif(
    not (REAL_GRAPH_PATH.exists() and REAL_GRAPH_META_PATH.exists()),
    reason="requires the real graph.npz/graph_meta.json built from FlyWire v783 data",
)
def test_photoreceptor_map_aligned_with_graph_input_idx():
    from flyhero.game.retina import build_photoreceptor_map

    graph = dict(np.load(REAL_GRAPH_PATH))
    meta = json.loads(REAL_GRAPH_META_PATH.read_text())
    photo_map = build_photoreceptor_map(graph, meta["type_names"])
    # The brain scatters photoreceptor currents into node-index space using
    # graph["input_idx_photoreceptor"] directly (rate_model.py); this must
    # stay index-aligned with the retina's own photoreceptor_idx ordering.
    np.testing.assert_array_equal(photo_map.photoreceptor_idx, graph["input_idx_photoreceptor"])


def test_connectome_brain_step_and_frame_step_shapes():
    graph = _tiny_graph()
    brain = ConnectomeBrain(graph, _BRAIN_CFG, device="cpu", dtype=torch.float32)
    batch = 4
    v = brain.init_state(batch)
    i_photo = torch.rand(batch, brain.n_input)

    v2, r = brain.step(v, i_photo)
    assert v2.shape == (batch, brain.n_nodes)
    assert r.shape == (batch, brain.n_nodes)
    assert torch.isfinite(v2).all()
    assert torch.isfinite(r).all()

    v3, dn_mean = brain.frame_step(v, i_photo, n_substeps=2)
    assert v3.shape == (batch, brain.n_nodes)
    assert dn_mean.shape == (batch, brain.n_dn)
    assert torch.isfinite(dn_mean).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available on this machine"
)
def test_mps_forward_backward_matches_cpu():
    """Correctness test: on the tiny graph, MPS forward+backward must match
    CPU (activations and gradients) within tolerance. index_add_ along a
    non-leading dim is the main MPS-coverage risk here (torch 2.14) -- if
    this fails with an 'not implemented for MPS' error, that's the intended
    signal to fall back to scatter_add_ or a [N,B]-shaped state layout."""
    graph = _tiny_graph()
    n_dn = len(graph["dn_idx"])
    n_input = len(graph["input_idx_photoreceptor"])
    batch = 2

    torch.manual_seed(0)
    i_photo_cpu = torch.rand(batch, n_input, dtype=torch.float32)

    readout_cpu = Readout(n_dn, n_actions=6)
    readout_state = copy.deepcopy(readout_cpu.state_dict())

    def run(device: str) -> dict:
        brain = ConnectomeBrain(graph, _BRAIN_CFG, device=device, dtype=torch.float32)
        readout = Readout(n_dn, n_actions=6)
        readout.load_state_dict(readout_state)
        readout = readout.to(device)

        v = brain.init_state(batch)
        i_photo = i_photo_cpu.to(device)
        dn_rate_mean = None
        for _ in range(3):
            v, dn_rate_mean = brain.frame_step(v, i_photo, n_substeps=2)
        logits = readout(dn_rate_mean)
        loss = logits.sum()
        loss.backward()

        return {
            "v": v.detach().cpu(),
            "dn_rate_mean": dn_rate_mean.detach().cpu(),
            "logits": logits.detach().cpu(),
            "tau_grad": brain.tau.grad.detach().cpu(),
            "bias_grad": brain.bias.grad.detach().cpu(),
            "g_grad": brain.g.grad.detach().cpu(),
            "readout_w_grad": readout.linear.weight.grad.detach().cpu(),
            "readout_b_grad": readout.linear.bias.grad.detach().cpu(),
        }

    cpu_out = run("cpu")
    mps_out = run("mps")

    for key in cpu_out:
        torch.testing.assert_close(cpu_out[key], mps_out[key], atol=1e-4, rtol=1e-3)


def test_gradcheck_float64_cpu():
    """Gradients of a couple of substeps w.r.t. (tau, bias, g, i_photo) pass
    torch.autograd.gradcheck in float64 on CPU. Uses torch.func.
    functional_call (via ConnectomeBrain.forward's step() alias) to drive
    the real step() implementation with substituted parameter tensors,
    rather than duplicating its math in a standalone function."""
    graph = _tiny_graph()
    brain = ConnectomeBrain(graph, _BRAIN_CFG, device="cpu", dtype=torch.float64)
    batch = 2

    torch.manual_seed(0)
    i_photo = torch.rand(batch, brain.n_input, dtype=torch.float64, requires_grad=True)
    tau = brain.tau.detach().clone().requires_grad_(True)
    bias = brain.bias.detach().clone().requires_grad_(True)
    g = brain.g.detach().clone().requires_grad_(True)

    def loss_fn(tau, bias, g, i_photo):
        # Starting exactly at v=0 leaves several nodes sitting precisely on
        # relu's kink for the first couple of substeps (no synaptic input
        # has propagated to them yet), which makes gradcheck's central
        # difference straddle a non-smooth point and spuriously fail. A
        # small positive offset moves every node off that kink without
        # otherwise changing what's being checked (this is a test-only
        # workaround for ReLU non-smoothness, not a change to init_state).
        v = torch.full((batch, brain.n_nodes), 1e-3, dtype=torch.float64)
        params = {"tau": tau, "bias": bias, "g": g}
        for _ in range(2):  # gradcheck is expensive; a couple of substeps is enough
            v, _ = functional_call(brain, params, (v, i_photo))
        return v[:, brain.dn_idx].sum()

    assert torch.autograd.gradcheck(loss_fn, (tau, bias, g, i_photo), eps=1e-6, atol=1e-4)
