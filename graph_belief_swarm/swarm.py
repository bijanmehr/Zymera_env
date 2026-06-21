"""Swarm driver -- ties belief + compass + controller to the zymera env, and
runs the Emergence coupling (comm graph -> belief merge + claim sharing).

Drives the zymera GridEnv host-side (Python loop) so the numpy belief/compass/
controller can run each tick. One episode = one call to `run_episode`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import zymera
from zymera.env import Body, World

from .belief import Belief
from .compass import choose_goal
from .controller import (action_for, bfs, collision_shield,
                         connectivity_shield, first_step)
from .coverage_sweep import SweepController
from .metrics import comm_adj, connected_components, gini, is_connected


def _clustered_positions(wall, n, center, comm_r):
    """n free cells near a center, *spread* (farthest-point) within a disk of
    radius comm_r//2 so the swarm starts fully connected (max pairwise <= comm_r)
    yet with room to move -- avoids the gridlock a tight cluster causes."""
    free = np.argwhere(~wall)
    cr, cc = center
    rad = max(1, comm_r // 2)
    cheb = np.maximum(np.abs(free[:, 0] - cr), np.abs(free[:, 1] - cc))
    pool = free[cheb <= rad]
    if len(pool) < n:                          # fall back to nearest-by-Manhattan
        order = free[np.argsort(np.abs(free[:, 0] - cr) + np.abs(free[:, 1] - cc))]
        pool = order[:max(n, 1)]
    chosen = [tuple(int(x) for x in pool[0])]
    while len(chosen) < n and len(chosen) < len(pool):
        dmin = np.min(
            [np.maximum(np.abs(pool[:, 0] - r), np.abs(pool[:, 1] - c)) for r, c in chosen],
            axis=0,
        ).astype(float)
        for r, c in chosen:
            dmin[(pool[:, 0] == r) & (pool[:, 1] == c)] = -1.0
        chosen.append(tuple(int(x) for x in pool[int(np.argmax(dmin))]))
    while len(chosen) < n:                     # tiny pool: allow repeats
        chosen.append(chosen[-1])
    return chosen[:n]


def _line_positions(wall, n, H):
    """One agent per strip at the leftmost free cell near the strip centre -> a
    connected vertical line ready to rake (no formation warmup wasted)."""
    S = max(1, H // n)
    W = wall.shape[1]
    out = []
    for k in range(n):
        r0 = k * S
        r1 = H if k == n - 1 else (k + 1) * S
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


def _make_world(env, wall, positions):
    H, W = wall.shape
    n = len(positions)
    pos = jnp.asarray(np.array(positions), dtype=jnp.int32)
    explored = jnp.zeros((H, W), dtype=jnp.int32).at[pos[:, 0], pos[:, 1]].add(1)
    world = World(
        body=Body(position=pos),
        explored=explored,
        step_count=jnp.zeros((), dtype=jnp.int32),
        seen_by=jnp.zeros((n, H, W), dtype=bool),
        wall=jnp.asarray(wall, dtype=bool),
        comm_graph=jnp.zeros((n, n), dtype=bool),
    )
    return env._reset_observe(world)


def run_episode(H, W, wall, n_agents, comm_r, sense_r, steps, seed,
                *, merge=True, anti_crowding=True, keep_connected=False,
                shield="hard", tether_radius=None, strategy="compass",
                occlusion=True, record=False):
    """Run one coverage episode. Returns a metrics dict (+ frames if record)."""
    rng = np.random.default_rng(seed)
    wall = np.asarray(wall, dtype=bool)
    env = zymera.make("empty-v0", grid_h=H, grid_w=W, n_agents=n_agents,
                      wall=wall, sense_radius=sense_r, occlusion=occlusion)

    free_cells = np.argwhere(~wall)
    center = tuple(int(x) for x in free_cells[rng.integers(len(free_cells))])
    if strategy == "sweep":
        positions = _line_positions(wall, n_agents, H)        # pre-formed rake, col 0
    else:
        positions = _clustered_positions(wall, n_agents, center, comm_r)
    world = _make_world(env, wall, positions)

    beliefs = [Belief(H, W, wall) for _ in range(n_agents)]
    goals = [None] * n_agents
    sweeper = SweepController(H, W, n_agents) if strategy == "sweep" else None
    team_visited = np.zeros((H, W), dtype=bool)
    owner = -np.ones((H, W), dtype=int)
    per_agent_visits = np.zeros(n_agents, dtype=int)
    free_total = int((~wall).sum())

    key = jax.random.PRNGKey(seed)
    cov_curve, conn_hits, shield_total = [], 0, 0
    frames = []

    for _t in range(steps):
        pos_np = np.asarray(world.body.position)
        for i in range(n_agents):
            r, c = int(pos_np[i, 0]), int(pos_np[i, 1])
            if not team_visited[r, c]:
                team_visited[r, c] = True
                owner[r, c] = i
            per_agent_visits[i] += 1

        # --- NAVIGATION: each agent folds in what it currently senses --------
        vis = np.asarray(env.sensor.visibility(world))           # (N,H,W) bool
        for i in range(n_agents):
            beliefs[i].observe(vis[i], pos_np[i])

        # --- EMERGENCE: comm graph -> belief merge within components ---------
        adj = comm_adj(pos_np, comm_r)
        comps = connected_components(adj)
        if is_connected(adj):
            conn_hits += 1
        if merge:
            for comp in comps:
                if len(comp) > 1:
                    obs = np.zeros((H, W), dtype=bool)
                    vsd = np.zeros((H, W), dtype=bool)
                    for i in comp:
                        obs |= beliefs[i].observed
                        vsd |= beliefs[i].visited
                    for i in comp:
                        beliefs[i].observed = obs.copy()
                        beliefs[i].visited = vsd.copy()
        comp_of = {i: comp for comp in comps for i in comp}

        # --- GUIDANCE + CONTROL: pick this tick's actions --------------------
        if strategy == "sweep":
            actions = sweeper.actions(pos_np, wall)     # connected boustrophedon rake
            vetoes = 0
        else:
            actions = np.zeros(n_agents, dtype=int)
            centroid = (int(round(pos_np[:, 0].mean())), int(round(pos_np[:, 1].mean()))) \
                if keep_connected else None
            for i in range(n_agents):
                b = beliefs[i]
                src = (int(pos_np[i, 0]), int(pos_np[i, 1]))
                dist, parent = bfs(b.free, src)
                g = goals[i]
                # commit-until-reached (avoids the dithering the sensor causes)
                stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]]))
                if stale:
                    claims = []
                    if anti_crowding and merge:
                        for j in comp_of.get(i, [i]):
                            if j != i and goals[j] is not None:
                                claims.append(goals[j])
                    g = choose_goal(b, src, dist, claims, anti_crowding=anti_crowding,
                                    anchor=centroid, w_anchor=(0.6 if keep_connected else 0.0),
                                    tether_radius=((tether_radius or comm_r) if keep_connected else None))
                    goals[i] = g
                if g is not None:
                    actions[i] = action_for(src, first_step(parent, src, g))

            if keep_connected and shield == "hard":
                actions, vetoes = connectivity_shield(actions, pos_np, wall, H, W, comm_r)
            else:
                actions, vetoes = collision_shield(actions, pos_np, wall, H, W)
        shield_total += vetoes

        cov_curve.append(team_visited.sum() / free_total)
        if record:
            frames.append(dict(pos=pos_np.copy(), owner=owner.copy(),
                               adj=adj.copy(), goals=list(goals)))

        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, dtype=jnp.int32), sk)

    pos_np = np.asarray(world.body.position)
    for i in range(n_agents):
        r, c = int(pos_np[i, 0]), int(pos_np[i, 1])
        if not team_visited[r, c]:
            team_visited[r, c] = True
            owner[r, c] = i

    team_observed = np.zeros((H, W), dtype=bool)
    for b in beliefs:
        team_observed |= b.observed
    observed_coverage = (team_observed & ~wall).sum() / free_total

    owned = np.array([(owner == i).sum() for i in range(n_agents)])
    return dict(
        coverage=team_visited.sum() / free_total,
        observed_coverage=observed_coverage,
        connectivity=conn_hits / steps,
        redundancy=per_agent_visits.sum() / max(1, int(team_visited.sum())),
        gini=gini(owned),
        shield=shield_total,
        cov_curve=cov_curve,
        owner=owner,
        wall=wall,
        owned=owned,
        frames=frames,
    )
