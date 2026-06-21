"""Core helpers for the decentralized shared-exploration swarm.

World physics come from the zymera GridEnv + Sensor (host-side / Python-loop).
Everything here is plain numpy: comm graph, connectivity, BFS path-stepping,
the certainty field, and the evaluation metrics. No central controller — these
are utilities each agent (or the evaluator) calls locally.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

import zymera
from zymera.env import Body, World

# zymera ActionId: STAY=0, N=1, E=2, S=3, W=4
DELTAS = np.array([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])


# ---------- comm graph / connectivity ----------------------------------------
def comm_adj(pos, comm_r):
    n = len(pos)
    d = np.maximum(np.abs(pos[:, None, 0] - pos[None, :, 0]),
                   np.abs(pos[:, None, 1] - pos[None, :, 1]))
    return (d <= comm_r) & ~np.eye(n, dtype=bool)


def connected_components(adj):
    n = adj.shape[0]
    seen = np.zeros(n, bool)
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


def is_connected(adj):
    return adj.shape[0] <= 1 or len(connected_components(adj)) == 1


# ---------- BFS path-stepping over known-free cells ---------------------------
def bfs(free, src):
    H, W = free.shape
    dist = np.full((H, W), np.inf)
    parent = -np.ones((H, W, 2), int)
    sr, sc = int(src[0]), int(src[1])
    free = free.copy()
    free[sr, sc] = True
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
    src = (int(src[0]), int(src[1]))
    r, c = int(goal[0]), int(goal[1])
    if (r, c) == src:
        return src
    path = [(r, c)]
    while tuple(parent[r, c]) != (-1, -1) and (r, c) != src:
        r, c = int(parent[r, c][0]), int(parent[r, c][1])
        path.append((r, c))
    path.reverse()
    return path[1] if len(path) >= 2 else src


def action_for(src, nxt):
    dr, dc = int(nxt[0]) - int(src[0]), int(nxt[1]) - int(src[1])
    for a, (adr, adc) in enumerate(DELTAS):
        if (adr, adc) == (dr, dc):
            return a
    return 0


def collision_mask(actions, pos, wall):
    """HARD collision avoidance: no two agents land on the same cell. Processed
    in agent-ID order (lower ID has priority); an agent moves only into a cell
    that is currently free (vacated cells free up as lower-ID agents move first),
    otherwise it STAYs. A physical constraint — unlike connectivity, this is a
    hard mask, not a soft preference."""
    H, W = wall.shape
    actions = np.array(actions, dtype=int)
    occ = {(int(pos[k][0]), int(pos[k][1])): k for k in range(len(pos))}
    for i in range(len(pos)):
        cur = (int(pos[i][0]), int(pos[i][1]))
        dr, dc = DELTAS[actions[i]]
        nr = min(max(cur[0] + int(dr), 0), H - 1)
        nc = min(max(cur[1] + int(dc), 0), W - 1)
        if wall[nr, nc]:
            nr, nc = cur
        tgt = (nr, nc)
        if tgt == cur:
            continue
        if occ.get(tgt) is None:              # free -> move (frees own cell)
            occ.pop(cur, None)
            occ[tgt] = i
        else:                                 # occupied -> STAY
            actions[i] = 0
    return actions


# ---------- env construction (clustered, connected spawn) ---------------------
def random_wall(H, W, density, seed):
    rng = np.random.default_rng(seed + 777)
    wall = rng.random((H, W)) < density
    wall[0, 0] = False
    return wall


def clustered_spawn(wall, n, comm_r, seed):
    """n free cells, spread within comm_r//2 of a random centre -> connected start."""
    rng = np.random.default_rng(seed)
    free = np.argwhere(~wall)
    cr, cc = free[rng.integers(len(free))]
    rad = max(1, comm_r // 2)
    cheb = np.maximum(np.abs(free[:, 0] - cr), np.abs(free[:, 1] - cc))
    pool = free[cheb <= rad]
    if len(pool) < n:
        pool = free[np.argsort(np.abs(free[:, 0] - cr) + np.abs(free[:, 1] - cc))][:max(n, 1)]
    chosen = [tuple(int(x) for x in pool[0])]
    while len(chosen) < n and len(chosen) < len(pool):
        dmin = np.min([np.maximum(np.abs(pool[:, 0] - r), np.abs(pool[:, 1] - c))
                       for r, c in chosen], axis=0).astype(float)
        for r, c in chosen:
            dmin[(pool[:, 0] == r) & (pool[:, 1] == c)] = -1.0
        chosen.append(tuple(int(x) for x in pool[int(np.argmax(dmin))]))
    while len(chosen) < n:
        chosen.append(chosen[-1])
    return chosen[:n]


def make_env_world(H, W, wall, positions, sense_r, occlusion=True):
    env = zymera.make("empty-v0", grid_h=H, grid_w=W, n_agents=len(positions),
                      wall=wall, sense_radius=sense_r, occlusion=occlusion)
    pos = jnp.asarray(np.array(positions), jnp.int32)
    n = len(positions)
    explored = jnp.zeros((H, W), jnp.int32).at[pos[:, 0], pos[:, 1]].add(1)
    world = World(body=Body(position=pos), explored=explored,
                  step_count=jnp.zeros((), jnp.int32),
                  seen_by=jnp.zeros((n, H, W), bool), wall=jnp.asarray(wall, bool),
                  comm_graph=jnp.zeros((n, n), bool))
    return env, env._reset_observe(world)


# ---------- metrics -----------------------------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def evaluate(true_count, wall, conn_hits, steps, owner, n_agents, t_cov):
    """Objective metrics from the ground-truth operation field."""
    free = ~wall
    F = int(free.sum())
    operated = (true_count >= 1) & free
    coverage = operated.sum() / F
    # evenness over free cells (1 - Gini of counts); higher = more uniform
    counts_free = true_count[free]
    evenness = 1.0 - gini(counts_free)
    redundancy = float(true_count[free].sum()) / max(1, int(operated.sum()))
    owned = np.array([(owner == i).sum() for i in range(n_agents)])
    dol_gini = gini(owned)                       # territory inequality (lower = balanced split)
    return dict(coverage=float(coverage), connectivity=conn_hits / steps,
                evenness=float(evenness), redundancy=redundancy,
                dol_gini=float(dol_gini), t_cov=int(t_cov), F=F,
                owned=owned.tolist())
