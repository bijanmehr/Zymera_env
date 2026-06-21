"""Topology gossip + relay-positioning — the scale-connectivity mechanism.

The per-agent connectivity weight (lambda role) is IMPOTENT at 32x32: ES fitness
was flat across every generation. Two structural reasons, both addressed here:

  1. A soft penalty measured against the NEAREST KNOWN neighbor cannot protect a
     GLOBAL property (single-component connectivity). -> topology gossip gives
     each agent its component's full comm graph, so it can reason about whether
     *it* is the bridge holding the swarm together.
  2. Once an agent fragments off, nothing pulls it back -- the penalty only
     shapes which frontier it picks, never says "go home". -> a relay role that
     actively REPOSITIONS toward the swarm centroid (an action, not a weight).

Everything stays decentralized: gossip only ever reconstructs the topology of an
agent's OWN connected component (flooding neighbor-lists within a component
recovers its subgraph); it never reveals anything across a broken link -- which
is exactly the information an agent legitimately cannot have, and the whole
reason connectivity is hard at scale.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def gossip_topology(adj, comps):
    """Within-component topology reconstruction. By flooding neighbor-lists,
    every agent in a connected component learns that component's full subgraph
    (and NOTHING about other components). Returns a per-agent N x N estimate."""
    n = adj.shape[0]
    est = [np.zeros((n, n), bool) for _ in range(n)]
    for comp in comps:
        idx = np.array(comp)
        sub = np.zeros((n, n), bool)
        sub[np.ix_(idx, idx)] = adj[np.ix_(idx, idx)]
        for a in comp:
            est[a] = sub                      # read-only downstream; shared ok
    return est


def global_cut_vertex(sub, i, comp):
    """Would removing agent i split its component? A TRUE cut-vertex test on the
    gossiped subgraph (vs. the conservative 2-hop approximation in swarm._is_cut_vertex).
    A cut-vertex is a bridge -> it must hold position, not chase a frontier."""
    others = [a for a in comp if a != i]
    if len(others) <= 1:
        return False
    seen = {others[0]}
    q = deque([others[0]])
    while q:
        u = q.popleft()
        for v in others:
            if sub[u, v] and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) != len(others)


def relay_target(i, pos, adj, last_nbr_pos, seen, wall, dist):
    """Connectivity-holding target.

    With LIVE neighbors the safest connectivity move is to HOLD position -- a
    relay (cut-vertex / edge leaf) is already where it's needed; moving it risks
    breaking the link it exists to protect. With NO neighbors it RECONNECTS,
    heading back toward the last heard neighbor positions (snapped to the nearest
    reachable known-free cell)."""
    cur = (int(pos[i, 0]), int(pos[i, 1]))
    nbrs = np.where(adj[i])[0]
    if len(nbrs) > 0:
        return cur                                       # hold the bridge / link
    if not last_nbr_pos:
        return cur                                       # nothing known -> hold
    pts = np.array(list(last_nbr_pos.values()), float)
    cr, cc = pts.mean(0)
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return cur
    d2 = (cand[:, 0] - cr) ** 2 + (cand[:, 1] - cc) ** 2
    r, c = cand[int(np.argmin(d2))]
    return (int(r), int(c))


def herd_target(i, pos, adj, last_nbr_pos, seen, wall, dist):
    """Return-to-herd target that REGAINS degree: head to the centroid of current
    neighbors (or last-heard neighbors if detached), snapped to the nearest
    reachable known-free cell. Unlike a static HOLD, this MOVES the agent inward,
    raising its degree and reshaping the graph -- which is what lets the role be
    RELEASED again. A hold is a fixed point that never clears its own trigger;
    this is why relays got stuck in the relay role."""
    cur = (int(pos[i, 0]), int(pos[i, 1]))
    nbrs = np.where(adj[i])[0]
    if len(nbrs) > 0:
        pts = pos[nbrs].astype(float)
    elif last_nbr_pos:
        pts = np.array(list(last_nbr_pos.values()), float)
    else:
        return cur                                       # nothing known -> hold
    cr, cc = pts.mean(0)
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return cur
    d2 = (cand[:, 0] - cr) ** 2 + (cand[:, 1] - cc) ** 2
    r, c = cand[int(np.argmin(d2))]
    return (int(r), int(c))


def degree_features(i, pos, adj, comp_size, d_ref, kb_count, seen, wall, n_agents, H, W):
    """Degree-based features for the learned degree policy (ES/MARL). All local or
    gossip-derivable: degree, neighbors lost from this agent's own peak (the budget
    signal), detached flag, component fraction |my comp|/N (gossip GLOBAL-split
    awareness), local unexplored density, coverage progress."""
    deg = int(adj[i].sum())
    f0 = deg / max(1, n_agents - 1)
    f1 = (d_ref - deg) / max(1.0, d_ref)                 # fraction lost from personal peak
    f2 = 1.0 if deg == 0 else 0.0                        # detached
    f3 = comp_size / max(1, n_agents)                    # my component as a fraction of the swarm
    r, c = int(pos[i, 0]), int(pos[i, 1])
    rad = 4
    r0, r1, c0, c1 = max(0, r - rad), min(H, r + rad + 1), max(0, c - rad), min(W, c + rad + 1)
    wfree = seen[r0:r1, c0:c1] & ~wall[r0:r1, c0:c1]
    wun = wfree & (kb_count[r0:r1, c0:c1] < 1)
    f4 = wun.sum() / max(1, wfree.sum())                 # local unexplored density
    free = seen & ~wall
    f5 = (free & (kb_count >= 1)).sum() / max(1, free.sum())   # coverage progress
    return np.array([f0, f1, f2, f3, f4, f5])


def relay_features(i, pos, adj, comp, sub, comm_r):
    """Two extra features for the learned relay-gate policy, beyond the 6 in
    swarm._agent_features: am I a global cut-vertex, and how close am I to losing
    my last link (edge pressure). Both are the signals a relay decision needs."""
    gcv = 1.0 if global_cut_vertex(sub, i, comp) else 0.0
    nbrs = np.where(adj[i])[0]
    if len(nbrs) > 0:
        mind = float(np.min(np.max(np.abs(pos[nbrs] - pos[i]), axis=1)))
        edge = min(mind, comm_r) / comm_r            # 1.0 = at the comm edge
    else:
        edge = 1.0                                   # detached
    return np.array([gcv, edge])
