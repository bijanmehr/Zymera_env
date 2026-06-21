"""Swarm metrics: comm graph, connectivity, division-of-labor."""

from __future__ import annotations

from collections import deque

import numpy as np


def comm_adj(pos: np.ndarray, comm_r: int) -> np.ndarray:
    """(N,N) bool adjacency: Chebyshev distance <= comm_r, no self-loops."""
    n = len(pos)
    d = np.maximum(np.abs(pos[:, None, 0] - pos[None, :, 0]),
                   np.abs(pos[:, None, 1] - pos[None, :, 1]))
    return (d <= comm_r) & ~np.eye(n, dtype=bool)


def connected_components(adj: np.ndarray):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        comp, q, seen[s] = [], deque([s]), True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in range(n):
                if adj[u, v] and not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    return comps


def is_connected(adj: np.ndarray) -> bool:
    return len(connected_components(adj)) == 1


def gini(x) -> float:
    """Inequality of per-agent territory sizes (0 = perfectly balanced split)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
