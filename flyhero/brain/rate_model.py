"""Rate-based connectome brain (PLAN Phase 3 task 1).

    r      = clamp(relu(v), 0, 1)                                        # [B, N]
    w      = sign * softplus(g[gain_group_id]) * count * inv_norm[post]  # [E]
    syn    = zeros_like(v).index_add_(1, post, r[:, pre] * w)            # [B, N]
    v_next = v + (dt / tau[type_id]) * (-v + syn + b[type_id] + I_in)

Invariant 3: connectome topology, synapse counts, and edge signs are fixed
(registered as buffers below, never trained). Trainable: per-cell-type `tau`
and `bias` (grouped by `type_id`), per-gain-group `g` (D16), and the readout
(readout.py) -- nothing else.

`graph.npz` carries `pair_id`/`type_id`/`super_class_id` but not a
`gain_group_id` array -- D16 says deriving the gain-sharing scheme is this
module's job, not the connectome build's, so changing `gain_sharing`/`k`
never requires rebuilding the graph. `compute_gain_group_id` below mirrors
`flyhero/connectome/build_graph.py`'s hybrid-gain-sharing report (lines
384-436) exactly, so it reproduces the same fine/coarse counts on the real
graph (see the real-data regression test in tests/test_rate_model.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def softplus_inverse(y: float) -> float:
    """x such that softplus(x) = y, i.e. log(exp(y) - 1)."""
    return math.log(math.expm1(y))


def load_graph_and_meta(processed_dir: str | Path) -> tuple[dict, dict]:
    processed_dir = Path(processed_dir)
    graph = dict(np.load(processed_dir / "graph.npz"))
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    return graph, meta


def filter_by_min_syn(graph: dict, min_syn: int) -> dict:
    """Filter graph.npz's edges down to syn_count >= min_syn, recomputing
    inv_norm/pair_id on the reduced edge set. Exactly equivalent to an
    independent connectome rebuild at this min_syn (verified: min_syn only
    gates the per-(pre,post) synapse-sum threshold in build_graph.py; NT
    sign, type_id, super_class_id, dn_idx, input_idx_* are node-level and
    computed independently of min_syn) -- so this avoids a slow rebuild for
    bench/bench_brain.py's min_syn sweep axis. Mirrors
    build_graph.py:323-333's pair_id/inv_norm computation exactly."""
    mask = graph["syn_count"] >= min_syn
    pre = graph["pre"][mask]
    post = graph["post"][mask]
    sign = graph["sign"][mask]
    syn_count = graph["syn_count"][mask]
    type_id = graph["type_id"]

    pair_keys = list(zip(type_id[pre].tolist(), type_id[post].tolist()))
    unique_pairs = sorted(set(pair_keys))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    pair_id = np.array([pair_to_id[k] for k in pair_keys], dtype=np.int64)

    n_nodes = len(type_id)
    incoming_totals = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(incoming_totals, post, syn_count)
    inv_norm = np.zeros(n_nodes, dtype=np.float32)
    nonzero = incoming_totals > 0
    inv_norm[nonzero] = 1.0 / incoming_totals[nonzero]

    filtered = dict(graph)
    filtered.update(
        pre=pre, post=post, sign=sign, syn_count=syn_count, pair_id=pair_id, inv_norm=inv_norm
    )
    return filtered


def compute_gain_group_id(
    pair_id: np.ndarray,
    type_id: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    super_class_id: np.ndarray,
    scheme: str,
    k: int = 2,
) -> tuple[np.ndarray, int]:
    """Derive a per-edge gain-sharing group id (D16), fully vectorized (no
    per-edge Python loop -- the real graph has 2.7M edges).

    fine:   one group per realized (pre_type, post_type) pair.
    coarse: one group per (pre_type, post_super_class).
    hybrid: a pair keeps its own group if it has >=k edges, else it's
            remapped to its (pre_type, post_super_class) coarse bucket --
            matches build_graph.py's hybrid_gain_sharing report exactly.
    """
    n_edges = len(pair_id)
    n_super_classes = int(super_class_id.max()) + 1 if len(super_class_id) else 0
    coarse_key = type_id[pre].astype(np.int64) * n_super_classes + super_class_id[post].astype(
        np.int64
    )

    if scheme == "fine":
        _, dense = np.unique(pair_id, return_inverse=True)
        dense = dense.astype(np.int64)
        n_groups = int(dense.max()) + 1 if n_edges else 0
        return dense, n_groups

    if scheme == "coarse":
        unique_keys, dense = np.unique(coarse_key, return_inverse=True)
        return dense.astype(np.int64), len(unique_keys)

    if scheme != "hybrid":
        raise ValueError(f"unknown gain_sharing scheme: {scheme!r}")

    _, dense_pair_id = np.unique(pair_id, return_inverse=True)
    n_pairs = int(dense_pair_id.max()) + 1 if n_edges else 0
    pair_counts = np.bincount(dense_pair_id, minlength=n_pairs)
    fine_mask = pair_counts >= k
    edge_is_fine = fine_mask[dense_pair_id]

    n_fine = int(fine_mask.sum())
    fine_group_of_pair = np.full(n_pairs, -1, dtype=np.int64)
    fine_group_of_pair[fine_mask] = np.arange(n_fine)

    leftover_mask = ~edge_is_fine
    unique_leftover, leftover_group = np.unique(coarse_key[leftover_mask], return_inverse=True)
    n_coarse = len(unique_leftover)

    gain_group_id = np.empty(n_edges, dtype=np.int64)
    gain_group_id[~leftover_mask] = fine_group_of_pair[dense_pair_id[~leftover_mask]]
    gain_group_id[leftover_mask] = n_fine + leftover_group

    return gain_group_id, n_fine + n_coarse


class ConnectomeBrain(nn.Module):
    """graph: a dict of numpy arrays with the graph.npz schema (pre, post,
    sign, syn_count, inv_norm, type_id, super_class_id, pair_id, dn_idx,
    input_idx_photoreceptor) -- not tied to file I/O, so tests can hand it a
    tiny synthetic graph. cfg: configs/brain.yaml's dict."""

    def __init__(
        self,
        graph: dict,
        cfg: dict,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.dt = float(cfg["dt"])
        self.tau_min_s = float(cfg["tau_min_s"])

        pre = np.asarray(graph["pre"], dtype=np.int64)
        post = np.asarray(graph["post"], dtype=np.int64)
        sign = np.asarray(graph["sign"], dtype=np.float64)
        syn_count = np.asarray(graph["syn_count"], dtype=np.float64)
        inv_norm = np.asarray(graph["inv_norm"], dtype=np.float64)
        type_id = np.asarray(graph["type_id"], dtype=np.int64)
        super_class_id = np.asarray(graph["super_class_id"], dtype=np.int64)
        pair_id = np.asarray(graph["pair_id"], dtype=np.int64)
        dn_idx = np.asarray(graph["dn_idx"], dtype=np.int64)
        input_idx = np.asarray(graph["input_idx_photoreceptor"], dtype=np.int64)

        self.n_nodes = int(len(type_id))
        self.n_types = int(type_id.max()) + 1 if len(type_id) else 0

        gain_group_id, n_groups = compute_gain_group_id(
            pair_id,
            type_id,
            pre,
            post,
            super_class_id,
            scheme=cfg.get("gain_sharing", "hybrid"),
            k=int(cfg.get("gain_sharing_k", 2)),
        )
        self.n_gain_groups = n_groups

        edge_const = sign * syn_count * inv_norm[post]  # [E], invariant-3 fixed data

        self.register_buffer("pre", torch.as_tensor(pre, dtype=torch.long, device=device))
        self.register_buffer("post", torch.as_tensor(post, dtype=torch.long, device=device))
        self.register_buffer("type_id", torch.as_tensor(type_id, dtype=torch.long, device=device))
        self.register_buffer(
            "gain_group_id", torch.as_tensor(gain_group_id, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "edge_const", torch.as_tensor(edge_const, dtype=dtype, device=device)
        )
        self.register_buffer("dn_idx", torch.as_tensor(dn_idx, dtype=torch.long, device=device))
        self.register_buffer(
            "input_idx", torch.as_tensor(input_idx, dtype=torch.long, device=device)
        )

        self.n_dn = int(len(dn_idx))
        self.n_input = int(len(input_idx))

        tau_init = float(cfg["tau_init_s"])
        self.tau = nn.Parameter(torch.full((self.n_types,), tau_init, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(self.n_types, dtype=dtype, device=device))

        gain0 = float(cfg["gain0"])
        g_init = softplus_inverse(gain0)
        self.g = nn.Parameter(
            torch.full((self.n_gain_groups,), g_init, dtype=dtype, device=device)
        )

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.n_nodes, dtype=self.dtype, device=self.device)

    def step(self, v: torch.Tensor, i_photo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One dt=1/120s substep. v: [B,N]. i_photo: [B, n_input] current at
        the photoreceptor input layer -- invariant 1: the brain's only
        visual input. Returns (v_next[B,N], r[B,N])."""
        r = torch.clamp(torch.relu(v), 0.0, 1.0)
        w = self.edge_const * F.softplus(self.g[self.gain_group_id])  # [E]
        pre_r = r[:, self.pre]  # [B,E]
        syn = torch.zeros_like(v)
        syn.index_add_(1, self.post, pre_r * w)

        i_in = torch.zeros_like(v)
        i_in.index_add_(1, self.input_idx, i_photo)

        tau = self.tau[self.type_id].clamp(min=self.tau_min_s)  # [N]
        bias = self.bias[self.type_id]  # [N]
        v_next = v + (self.dt / tau) * (-v + syn + bias + i_in)
        return v_next, r

    def frame_step(
        self, v: torch.Tensor, i_photo: torch.Tensor, n_substeps: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs n_substeps holding i_photo fixed (one 60Hz frame's worth of
        input at dt=1/120s). Returns (v_next, dn_rate_mean[B, n_dn]) -- the
        readout's only legal input (invariant 2)."""
        dn_rates = []
        for _ in range(n_substeps):
            v, r = self.step(v, i_photo)
            dn_rates.append(r[:, self.dn_idx])
        dn_rate_mean = torch.stack(dn_rates, dim=0).mean(dim=0)
        return v, dn_rate_mean

    def forward(self, v: torch.Tensor, i_photo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for step(), solely so torch.func.functional_call can drive
        this module with substituted (tau, bias, g) tensors -- used by the
        float64 gradcheck test, which needs a pure-function view of this
        module without duplicating step()'s math."""
        return self.step(v, i_photo)
