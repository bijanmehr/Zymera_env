"""The SLAM controller's actuation half -- path-follow + collision shield.

The compass sets a goal (slow clock); the controller drives to it every tick
(fast clock) by BFS over known-free cells, emitting one primitive move. The
learned policy never emits these moves -- that is the whole point of the
goal/subgoal action representation. A reactive collision shield vetoes moves
onto a contested cell (last-resort backstop; connectivity shield is a planned
addition).

Action ids match zymera.env.ActionId: STAY=0, N=1, E=2, S=3, W=4.
"""

from __future__ import annotations

from collections import deque

import numpy as np

DELTAS = np.array([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])  # STAY,N,E,S,W


def bfs(free: np.ndarray, src):
    """BFS over 4-connected free cells from src. Returns (dist, parent)."""
    H, W = free.shape
    dist = np.full((H, W), np.inf)
    parent = -np.ones((H, W, 2), dtype=int)
    sr, sc = int(src[0]), int(src[1])
    free = free.copy()
    free[sr, sc] = True                       # always traversable from where we stand
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and not np.isfinite(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1
                parent[nr, nc] = (r, c)
                q.append((nr, nc))
    return dist, parent


def first_step(parent, src, goal):
    """Reconstruct path goal->src and return the first cell to step into."""
    src = (int(src[0]), int(src[1]))
    r, c = int(goal[0]), int(goal[1])
    if (r, c) == src:
        return src
    path = [(r, c)]
    while tuple(parent[r, c]) != (-1, -1) and (r, c) != src:
        r, c = int(parent[r, c][0]), int(parent[r, c][1])
        path.append((r, c))
    path.reverse()                            # src ... goal
    return path[1] if len(path) >= 2 else src


def action_for(src, nxt):
    dr, dc = int(nxt[0]) - int(src[0]), int(nxt[1]) - int(src[1])
    for a, (adr, adc) in enumerate(DELTAS):
        if (adr, adc) == (dr, dc):
            return a
    return 0


def collision_shield(actions, pos, wall, H, W):
    """Veto moves onto a contested target cell (lower index wins; loser STAYs).

    Returns (actions, n_vetoes). Mirrors the env's wall/bound clipping when
    computing targets so the shield reasons about real landing cells.
    """
    actions = np.array(actions, dtype=int)
    targets = []
    for i in range(len(pos)):
        dr, dc = DELTAS[actions[i]]
        nr = min(max(int(pos[i][0]) + int(dr), 0), H - 1)
        nc = min(max(int(pos[i][1]) + int(dc), 0), W - 1)
        if wall[nr, nc]:
            nr, nc = int(pos[i][0]), int(pos[i][1])
        targets.append((nr, nc))
    claimed = {}
    vetoes = 0
    for i in range(len(pos)):
        t = targets[i]
        if t in claimed and t != (int(pos[i][0]), int(pos[i][1])):
            actions[i] = 0                    # STAY
            vetoes += 1
        else:
            claimed[t] = i
    return actions, vetoes


def _clip_target(pos_i, action, wall, H, W):
    dr, dc = DELTAS[int(action)]
    nr = min(max(int(pos_i[0]) + int(dr), 0), H - 1)
    nc = min(max(int(pos_i[1]) + int(dc), 0), W - 1)
    if wall[nr, nc]:
        return int(pos_i[0]), int(pos_i[1])
    return int(nr), int(nc)


def connectivity_shield(actions, pos, wall, H, W, comm_r):
    """Hard connectivity guardrail (sequential): an agent takes its move only if
    that move does NOT increase the number of comm-graph components (and does not
    collide); otherwise it STAYs. Started from a connected cluster, this keeps the
    swarm's communication graph from ever fragmenting. Returns (actions, n_vetoes).
    """
    from .metrics import comm_adj, connected_components

    n = len(pos)
    committed = np.array(pos, dtype=int).copy()
    out = np.array(actions, dtype=int).copy()
    vetoes = 0
    for i in range(n):
        tgt = _clip_target(pos[i], out[i], wall, H, W)
        cur_comps = len(connected_components(comm_adj(committed, comm_r)))
        trial = committed.copy()
        trial[i] = tgt
        new_comps = len(connected_components(comm_adj(trial, comm_r)))
        if new_comps <= cur_comps:            # move only if it doesn't fragment
            committed[i] = tgt
        else:
            out[i] = 0
            vetoes += 1
    return out, vetoes
