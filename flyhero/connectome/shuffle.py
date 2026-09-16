"""Degree-preserving shuffle of the connectome graph: the control condition
for invariant 6. Swaps postsynaptic targets while keeping each node's
in-degree, out-degree, and per-edge sign/syn_count multiset intact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from flyhero.utils.config import load_config


def degree_preserving_shuffle(
    pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, n_swaps_factor: int = 10
) -> np.ndarray:
    """Return a new `post` array via a double-edge-swap Markov chain: repeatedly
    pick two edges (a->b, c->d) and swap to (a->d, c->b) when that doesn't
    create a duplicate edge or a self-loop. Preserves in/out-degree exactly."""
    post = post.copy()
    n_edges = len(pre)
    existing = set(zip(pre.tolist(), post.tolist()))
    n_swaps = n_swaps_factor * n_edges

    for _ in range(n_swaps):
        i, j = rng.integers(0, n_edges, size=2)
        if i == j:
            continue
        a, b = pre[i], post[i]
        c, d = pre[j], post[j]
        if a == c or b == d:
            continue
        new_edge_1 = (a, d)
        new_edge_2 = (c, b)
        if a == d or c == b:
            continue
        if new_edge_1 in existing or new_edge_2 in existing:
            continue

        existing.discard((a, b))
        existing.discard((c, d))
        existing.add(new_edge_1)
        existing.add(new_edge_2)
        post[i] = d
        post[j] = b

    return post


def shuffle_graph(graph_path: Path, out_path: Path, seed: int) -> None:
    data = dict(np.load(graph_path, allow_pickle=False))
    rng = np.random.default_rng(seed)

    shuffled_post = degree_preserving_shuffle(data["pre"], data["post"], rng)
    data["post"] = shuffled_post

    np.savez(out_path, **data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg: dict[str, Any] = load_config(args.config)
    processed_dir = Path(cfg["processed_dir"])
    shuffle_graph(
        processed_dir / "graph.npz", processed_dir / "graph_shuffled.npz", args.seed
    )


if __name__ == "__main__":
    sys.exit(main())
