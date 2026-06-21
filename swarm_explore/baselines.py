"""Reference baselines for the certainty-field coverage task.

These exist ONLY to frame the swarm's numbers:
  random_walk : the floor (no intelligence).
  lawnmower   : the ceiling — an OMNISCIENT, CENTRALIZED boustrophedon (uses the
                ground-truth wall map + a global schedule). NOT a valid decentralized
                solution; an efficiency reference to approach, never to submit.
The single-agent bar is just run_swarm(n_agents=1).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   evaluate, first_step, is_connected, make_env_world)


def _drive(env, world, H, W, wall, n_agents, comm_r, steps, seed, action_fn):
    true_count = np.zeros((H, W))
    owner = -np.ones((H, W), int)
    free_total = int((~wall).sum())
    conn_hits, t_cov = 0, steps
    key = jax.random.PRNGKey(seed + 1)
    for t in range(steps):
        pos = np.asarray(world.body.position)
        for i in range(n_agents):
            r, c = int(pos[i, 0]), int(pos[i, 1])
            if true_count[r, c] == 0:
                owner[r, c] = i
            true_count[r, c] += 1.0
        adj = comm_adj(pos, comm_r)
        if is_connected(adj):
            conn_hits += 1
        if (true_count >= 1).sum() / free_total >= 1.0 and t_cov == steps:
            t_cov = t
        key, sk = jax.random.split(key)
        actions = action_fn(pos, t, sk)
        actions = collision_mask(actions, pos, wall)    # HARD collision mask
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)
    pos = np.asarray(world.body.position)
    for i in range(n_agents):
        r, c = int(pos[i, 0]), int(pos[i, 1])
        if true_count[r, c] == 0:
            owner[r, c] = i
        true_count[r, c] += 1.0
    m = evaluate(true_count, wall, conn_hits, steps, owner, n_agents, t_cov)
    m.update(true_count=true_count, owner=owner, wall=wall, frames=[])
    return m


def random_walk(H, W, wall, n_agents, comm_r, sense_r, steps, seed):
    wall = np.asarray(wall, bool)
    positions = clustered_spawn(wall, n_agents, comm_r, seed)
    env, world = make_env_world(H, W, wall, positions, sense_r)
    rng = np.random.default_rng(seed + 5)
    return _drive(env, world, H, W, wall, n_agents, comm_r, steps, seed,
                  lambda pos, t, sk: rng.integers(0, 5, size=n_agents))


# ---- lawnmower (omniscient, centralized) ceiling ----------------------------
def _line_spawn(wall, n, H):
    S = max(1, H // n)
    W = wall.shape[1]
    out = []
    for k in range(n):
        r0, r1 = k * S, (H if k == n - 1 else (k + 1) * S)
        tr = r0 + (r1 - r0) // 2
        placed = None
        for col in range(W):
            for rr in sorted(range(r0, r1), key=lambda r: abs(r - tr)):
                if not wall[rr, col]:
                    placed = (rr, col)
                    break
            if placed:
                break
        out.append(placed or (min(tr, H - 1), 0))
    return out


class _Sweeper:
    def __init__(self, H, W, n):
        self.H, self.W, self.n = H, W, n
        self.S = max(1, H // n)
        self.schedule = []
        for off in range(self.S + (1 if H % n else 0)):
            cols = range(W) if off % 2 == 0 else range(W - 1, -1, -1)
            for col in cols:
                self.schedule.append((off, col))
        self.idx = 0
        self.strip_of = None

    def actions(self, pos, wall):
        if self.strip_of is None:
            order = np.argsort(np.asarray(pos)[:, 0])
            self.strip_of = {int(k): s for s, k in enumerate(order)}
        off, col = self.schedule[min(self.idx, len(self.schedule) - 1)]
        free = ~wall
        acts = np.zeros(self.n, int)
        ready = 0
        for k in range(self.n):
            strip = self.strip_of[k]
            r0, r1 = strip * self.S, (self.H if strip == self.n - 1 else (strip + 1) * self.S)
            tr = min(r0 + off, r1 - 1)
            if wall[tr, col]:
                cand = [r for r in range(r0, r1) if not wall[r, col]]
                if not cand:
                    ready += 1
                    continue
                tr = min(cand, key=lambda r: abs(r - (r0 + off)))
            src = (int(pos[k][0]), int(pos[k][1]))
            if src != (tr, col):
                dist, parent = bfs(free, src)
                if not np.isfinite(dist[tr, col]):
                    ready += 1
                    continue
                acts[k] = action_for(src, first_step(parent, src, (tr, col)))
            if abs(src[0] - tr) + abs(src[1] - col) <= 1:
                ready += 1
        if ready >= self.n - 1:
            self.idx = min(self.idx + 1, len(self.schedule) - 1)
        return acts


def lawnmower(H, W, wall, n_agents, comm_r, sense_r, steps, seed):
    wall = np.asarray(wall, bool)
    positions = _line_spawn(wall, n_agents, H)
    env, world = make_env_world(H, W, wall, positions, sense_r)
    sw = _Sweeper(H, W, n_agents)
    return _drive(env, world, H, W, wall, n_agents, comm_r, steps, seed,
                  lambda pos, t, sk: sw.actions(pos, wall))
