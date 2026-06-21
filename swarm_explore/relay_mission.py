"""Relay-mission engine — the "delivered coverage" task + a 2-role (explorer/relay)
decentralized rollout with hybrid connectivity (thin backstop) and mission-safety floors.

A sensed cell counts for the team ONLY once its finder is path-connected to the home core
(the component containing the agent nearest the spawn centroid). The policy decides, per
agent, a ROLE (0=explorer, 1=relay) and a GOAL cell; the rollout A*-routes to the goal,
reverts only the moves that would split the comm graph (backstop), and books delivered
coverage + connectivity + mission-safety violations.

Increment 1: a heuristic *load-bearing-relay* baseline that validates the mechanics and
serves as the experiment's baseline. ES / PPO later swap in for `policy`.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.relay_mission
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   connected_components, first_step, is_connected, make_env_world, random_wall)
from .connect import herd_target
from .swarm import _choose_target
from . import arch3                       # reuse _features / relay_prob / init_theta for the learned role head

COMM_R = 5


# ---------------- graph helpers ----------------
def _reachable(adj, src, exclude=-1):
    """Set of nodes reachable from src in adj WITHOUT routing through `exclude`."""
    n = adj.shape[0]; seen = np.zeros(n, bool)
    if src == exclude:
        return seen
    seen[src] = True; stack = [src]
    while stack:
        u = stack.pop()
        for v in range(n):
            if v != exclude and adj[u, v] and not seen[v]:
                seen[v] = True; stack.append(v)
    return seen


def load_bearing_roles(adj, core_node, n):
    """relay[i]=True iff removing agent i strands some agent from the core (i is load-bearing)."""
    base = _reachable(adj, core_node)
    idx = np.arange(n)
    relay = np.zeros(n, bool)
    for i in range(n):
        if i == core_node:
            continue
        wo = _reachable(adj, core_node, exclude=i)
        if np.any(base & ~wo & (idx != i)):     # someone reachable WITH i but not without it
            relay[i] = True
    return relay


def connectivity_backstop(next_cell, pos, comm_r):
    """Revert only the moves that would split the graph (farthest-from-centre first).
    Guarantees one component as long as the current positions are connected."""
    prop = next_cell.copy()
    if is_connected(comm_adj(prop, comm_r)):
        return prop
    n = pos.shape[0]; cen = pos.mean(0)
    moved = [i for i in range(n) if (prop[i] != pos[i]).any()]
    moved.sort(key=lambda i: -((pos[i] - cen) ** 2).sum())
    for i in moved:
        prop[i] = pos[i]
        if is_connected(comm_adj(prop, comm_r)):
            break
    return prop


# ---------------- the rollout ----------------
def rollout_mission(H, W, wall, n, comm_r, sense_r, steps, seed, policy, *, backstop=True, decay=1.0, record=False):
    """decay<1 => latency-discounted delivery (Regime B): a find delivered `k` steps after
    discovery is worth decay**k, so relays that shorten the path home actually earn credit.
    decay=1 (default) recovers plain delivered coverage (== raw under a hard backstop)."""
    wall = np.asarray(wall, bool)
    pos0 = np.asarray(clustered_spawn(wall, n, comm_r, seed))
    env, world = make_env_world(H, W, wall, pos0, sense_r, True)
    home = pos0.mean(0)
    kb = [np.zeros((H, W)) for _ in range(n)]
    seen = [np.zeros((H, W), bool) for _ in range(n)]
    goals = [None] * n
    delivered = np.zeros((H, W), bool)
    disc_step = np.full((H, W), -1, int); deliv_step = np.full((H, W), -1, int)
    role_count = np.zeros(n)
    free_total = int((~wall).sum())
    conn_hits = 0; safety_viol = 0; relay_steps = 0; frames = []
    nbr_pos = [dict() for _ in range(n)]
    key = jax.random.PRNGKey(seed)

    for t in range(steps):
        pos = np.asarray(world.body.position).astype(int)
        for i in range(n):
            kb[i][pos[i, 0], pos[i, 1]] += 1.0
        vis = np.asarray(env.sensor.visibility(world))
        for i in range(n):
            seen[i] |= vis[i]
        adj = comm_adj(pos, comm_r)
        comps = connected_components(adj); comp_of = {i: c for c in comps for i in c}
        if is_connected(adj):
            conn_hits += 1
        for c in comps:                                   # gossip within a component
            if len(c) > 1:
                ss = np.zeros((H, W), bool); cc = np.zeros((H, W))
                for i in c:
                    ss |= seen[i]; np.maximum(cc, kb[i], out=cc)
                for i in c:
                    seen[i] = ss.copy(); kb[i] = cc.copy()
        core_node = int(np.argmin(((pos - home) ** 2).sum(1)))
        core_comp = comp_of[core_node]
        for i in core_comp:                               # DELIVERED = union seen of the core component
            delivered |= (seen[i] & ~wall)
        raw_now = np.zeros((H, W), bool)
        for i in range(n):
            raw_now |= seen[i]
        disc_step[(raw_now & (disc_step < 0)) & ~wall] = t        # first sensed
        deliv_step[(delivered & (deliv_step < 0)) & ~wall] = t    # first delivered (latency clock)
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    nbr_pos[i][j] = (int(pos[j, 0]), int(pos[j, 1]))

        roles, goals = policy(pos, adj, comps, comp_of, core_node, core_comp,
                              kb, seen, wall, nbr_pos, goals, comm_r, n)
        roles = np.asarray(roles, int)
        deg = adj.sum(1)
        for i in range(n):                                # mission-safety floors
            if deg[i] < (2 if roles[i] == 1 else 1):
                safety_viol += 1
        relay_steps += int((roles == 1).sum())
        role_count += (roles == 1)

        next_cell = pos.copy()                            # A*-route to each goal
        for i in range(n):
            src = (int(pos[i, 0]), int(pos[i, 1])); g = goals[i]
            if g is not None:
                dist, parent = bfs(seen[i] & ~wall, src)
                if np.isfinite(dist[g[0], g[1]]):
                    nc = first_step(parent, src, g)
                    next_cell[i] = [nc[0], nc[1]]
        if backstop:
            next_cell = connectivity_backstop(next_cell, pos, comm_r)
        actions = np.array([0 if (next_cell[i] == pos[i]).all()
                            else action_for((int(pos[i, 0]), int(pos[i, 1])),
                                            (int(next_cell[i, 0]), int(next_cell[i, 1])))
                            for i in range(n)], int)
        actions = collision_mask(actions, pos, wall)

        if record:
            team = np.zeros((H, W))
            for i in range(n):
                np.maximum(team, kb[i], out=team)
            frames.append(dict(pos=pos.copy(), count=team, adj=adj.copy(),
                               goal=list(goals), relay=roles == 1, lam=np.zeros(n)))
        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)

    raw = np.zeros((H, W), bool)
    for i in range(n):
        raw |= seen[i]
    mask = (deliv_step >= 0) & ~wall
    lat = np.clip(deliv_step[mask] - disc_step[mask], 0, None)
    delivered_score = float((decay ** lat).sum() / free_total)
    return dict(
        delivered_cov=float((delivered & ~wall).sum() / free_total),
        delivered_score=delivered_score,
        raw_cov=float((raw & ~wall).sum() / free_total),
        connectivity=conn_hits / steps,
        safety_viol=safety_viol / max(1, n * steps),
        relay_frac=relay_steps / max(1, n * steps),
        per_agent_relay=[round(float(x), 3) for x in role_count / steps],
        wall=wall, frames=frames)


# ---------------- heuristic baseline policy ----------------
def heuristic_policy(pos, adj, comps, comp_of, core_node, core_comp, kb, seen, wall, nbr_pos, goals_prev, comm_r, n):
    """Load-bearing agents relay (hold the bridge); everyone else explores frontiers."""
    roles = load_bearing_roles(adj, core_node, n).astype(int)
    goals = [None] * n
    for i in range(n):
        src = (int(pos[i, 0]), int(pos[i, 1]))
        dist, parent = bfs(seen[i] & ~wall, src)
        if roles[i] == 1:
            g = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
        else:
            g = goals_prev[i]
            if g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1.0:
                g = _choose_target(kb[i], seen[i], wall, src, dist, [], nbr_pos[i], comm_r, 0.0)
        goals[i] = g
    return roles, goals


def make_theta_policy(theta, K):
    """θ-parameterized policy: a LEARNED role head (arch3.relay_prob over local features)
    decides explorer/relay; the fixed tools (frontier / herd-bridge) execute. ES/PPO optimize θ."""
    def policy(pos, adj, comps, comp_of, core_node, core_comp, kb, seen, wall, nbr_pos, goals_prev, comm_r, n):
        H, W = wall.shape
        roles = np.zeros(n, int)
        for i in range(n):
            feats = arch3._features(i, pos, adj, comp_of.get(i, [i]), kb[i], seen[i], wall, comm_r, K, H, W, n)
            roles[i] = 1 if arch3.relay_prob(theta, feats) > 0.5 else 0
        goals = [None] * n
        for i in range(n):
            src = (int(pos[i, 0]), int(pos[i, 1])); dist, parent = bfs(seen[i] & ~wall, src)
            if roles[i] == 1:
                g = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
            else:
                g = goals_prev[i]
                if g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1.0:
                    g = _choose_target(kb[i], seen[i], wall, src, dist, [], nbr_pos[i], comm_r, 0.0)
            goals[i] = g
        return roles, goals
    return policy


def main():
    import os
    from . import gif
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "runs", "relay-mission-smoke"))
    os.makedirs(out, exist_ok=True)
    print("relay-mission heuristic baseline (delivered coverage, backstop on):", flush=True)
    for (H, n, steps) in [(16, 4, 80), (24, 6, 110), (32, 10, 140)]:
        S = 3; acc = np.zeros(5)
        for s in range(S):
            m = rollout_mission(H, H, random_wall(H, H, 0.05, s), n, COMM_R, 3, steps, s, heuristic_policy)
            acc += [m["delivered_cov"], m["raw_cov"], m["connectivity"], m["relay_frac"], m["safety_viol"]]
        d, r, c, rel, sv = acc / S
        print(f"  {H}x{H}/N{n}: delivered {d*100:4.0f}%  raw {r*100:4.0f}%  conn {c*100:4.0f}%  "
              f"relay {rel*100:3.0f}%  safety_viol {sv:.3f}", flush=True)
        m = rollout_mission(H, H, random_wall(H, H, 0.05, 0), n, COMM_R, 3, steps, 0, heuristic_policy, record=True)
        p = os.path.join(out, f"relay_{H}x{H}_n{n}.gif")
        gif.animate_run(m, p, title=f"relay-mission {H}x{H}/N{n} · delivered {m['delivered_cov']*100:.0f}% conn {m['connectivity']*100:.0f}%")
        print(f"     gif -> {p}", flush=True)


if __name__ == "__main__":
    main()
