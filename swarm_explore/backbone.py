"""v0 of the spanning-tree backbone architecture (the design we converged on).

Each tick the swarm grows a decentralized BFS spanning tree over its comm graph
(rooted at the min-ID agent of each connected component). An agent's ROLE is read
straight off the tree:
  - INTERNAL node (has children) -> RELAY: hold position to keep its parent+children
    links in range (the multi-hop spine).
  - LEAF -> FRONTIER: explore, but keep within comm range of its parent (connectivity
    by choice, at the link level) -- if the best frontier is out of parent range it
    stretches to the range edge, pulling the parent to follow (emergent migration).
  - ISOLATED -> RECONNECT toward last-known parent / swarm.

The tree is the single object that gives role + multi-hop connectivity + migration.
This is the deterministic v0; the GNN/attention upgrades plug into the same interface.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def spanning_tree(adj, comps):
    """Per-component BFS tree rooted at the min-ID agent. Returns
    (parent[n], children[n] lists, is_internal[n] bool). parent=-1 for roots and
    isolates. This is exactly what decentralized BFS-tree gossip converges to:
    each non-root picks the in-range neighbor with the lowest hop-count to root
    (tie-broken by lowest ID)."""
    n = adj.shape[0]
    parent = -np.ones(n, int)
    for comp in comps:
        root = min(comp)
        hop = {root: 0}
        q = deque([root])
        while q:
            u = q.popleft()
            for v in comp:
                if adj[u, v] and v not in hop:
                    hop[v] = hop[u] + 1
                    q.append(v)
        for v in comp:
            if v == root:
                continue
            cands = [u for u in comp if adj[v, u] and hop.get(u, 1 << 30) == hop[v] - 1]
            if cands:
                parent[v] = min(cands)
    children = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    is_internal = np.array([len(children[v]) > 0 for v in range(n)], bool)
    return parent, children, is_internal


def _snap(cands, tr, tc, dist):
    """nearest reachable known-free candidate cell to target (tr,tc)."""
    d2 = (cands[:, 0] - tr) ** 2 + (cands[:, 1] - tc) ** 2
    r, c = cands[int(np.argmin(d2))]
    return (int(r), int(c))


def relay_hold_target(i, pos, parent, children, seen, wall, dist):
    """Internal node: hold the centroid of its tree-neighbors (parent + children)
    so every spine link stays in range. As children explore outward the centroid
    drifts after them -> the spine migrates."""
    cur = (int(pos[i, 0]), int(pos[i, 1]))
    nbrs = list(children[i]) + ([int(parent[i])] if parent[i] >= 0 else [])
    if not nbrs:
        return cur
    pts = pos[nbrs].astype(float)
    cr, cc = pts.mean(0)
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return cur
    return _snap(cand, cr, cc, dist)


def leaf_target(i, pos, parent, kb_count, seen, wall, dist, comm_r):
    """Leaf: highest-info reachable known-free cell that keeps it within comm_r
    (Chebyshev) of its parent. If no good frontier is in range, head to the
    in-range cell closest to the best out-of-range frontier (stretch to the edge,
    pulling the parent). Root-leaf (no parent) explores freely."""
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return (int(pos[i, 0]), int(pos[i, 1]))
    gains = 1.0 / np.sqrt(kb_count[cand[:, 0], cand[:, 1]] + 1.0)
    d = dist[cand[:, 0], cand[:, 1]]
    score = 4.0 * gains - 0.04 * d
    if parent[i] < 0:                                   # root-leaf: explore freely
        idx = int(np.argmax(score))
        return (int(cand[idx, 0]), int(cand[idx, 1]))
    pr, pc = int(pos[parent[i], 0]), int(pos[parent[i], 1])
    cheb = np.maximum(np.abs(cand[:, 0] - pr), np.abs(cand[:, 1] - pc))
    in_range = cheb <= comm_r
    if in_range.any():
        masked = np.where(in_range, score, -1e18)
        idx = int(np.argmax(masked))
        return (int(cand[idx, 0]), int(cand[idx, 1]))
    # nothing in range: stretch toward the best frontier but stay linked
    best = cand[int(np.argmax(score))]
    within = cand[in_range] if in_range.any() else cand[cheb <= comm_r]
    if len(within) == 0:
        # move to the closest-to-parent reachable cell (pull back so we don't detach)
        return _snap(cand, pr, pc, dist)
    return _snap(within, best[0], best[1], dist)
