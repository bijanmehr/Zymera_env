"""The 3-stack architecture, self-contained & stable, for 16x16 / 4 agents / 70 steps.

  Stack 1 — Compass (deterministic): frontier role -> uncertainty compass.
  Stack 2 — Role picker + switcher (ES): a small policy over graph-criticality features
            -> P(relay), wrapped in a hysteresis switcher (commit/release) so it doesn't
            flip-flop or stick. Frontier -> compass; relay -> hold (herd_target).
            Trained by ES with MISSION-SAFETY fitness: coverage - w_safety * (degree-budget
            violations), where an agent that has lost > K neighbors (degree < N-1-K) is a
            safety break.
  Stack 3 — Graph estimator (separate module estimator.py): can supply criticality features;
            here we use DIRECT gossiped-graph features (exact within a component) for
            stability, with the estimator validated/integrated separately.

This module is the runner + ES trainer for Stacks 1+2. No actor-critic (control = ES).
"""
from __future__ import annotations

import numpy as np
import jax

from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   connected_components, evaluate, first_step, is_connected, make_env_world)
from .connect import herd_target
from .swarm import _choose_target, _is_cut_vertex

import jax.numpy as jnp

# ---- Stack 2 policy: small MLP over graph-criticality features ----
SW_FEAT = 7
SW_HID = 16
_SH = [(SW_FEAT, SW_HID), (SW_HID,), (SW_HID, 1), (1,)]
SW_NPARAMS = sum(int(np.prod(s)) for s in _SH)


def init_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _SH[0]), np.zeros(_SH[1]), rng.normal(0, 0.5, _SH[2]), np.zeros(_SH[3])]
    return np.concatenate([p.ravel() for p in parts])


def _unflat(theta):
    out, k = [], 0
    for s in _SH:
        n = int(np.prod(s)); out.append(theta[k:k + n].reshape(s)); k += n
    return out


def relay_prob(theta, feats):
    W1, b1, W2, b2 = _unflat(theta)
    h = np.tanh(feats @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-float((h @ W2 + b2)[0])))


def _features(i, pos, adj, comp, kb, seen, wall, comm_r, K, H, W, n):
    """7 graph-criticality features for the role-switcher (all gossip-local)."""
    deg = int(adj[i].sum())
    full = n - 1
    f0 = deg / max(1, full)                               # normalized degree
    f1 = max(0.0, (full - K - deg)) / max(1, full)        # how far BELOW the safety floor (budget overspent)
    f2 = 1.0 if deg == 0 else 0.0                         # detached
    f3 = 1.0 if _is_cut_vertex(adj, i) else 0.0           # bridge (2-hop cut-vertex)
    f4 = len(comp) / n                                    # component fraction (global cohesion)
    r, c = int(pos[i, 0]), int(pos[i, 1]); rad = 4
    r0, r1, c0, c1 = max(0, r - rad), min(H, r + rad + 1), max(0, c - rad), min(W, c + rad + 1)
    wfree = seen[r0:r1, c0:c1] & ~wall[r0:r1, c0:c1]
    f5 = (wfree & (kb[r0:r1, c0:c1] < 1)).sum() / max(1, wfree.sum())     # local unexplored density
    free = seen & ~wall
    f6 = (free & (kb >= 1)).sum() / max(1, free.sum())    # coverage progress
    return np.array([f0, f1, f2, f3, f4, f5, f6])


def run_arch(H, W, wall, n, comm_r, sense_r, steps, seed, theta, *,
             K=2, hi=0.6, lo=0.4, lam_frontier=1.0, record=False):
    """Stacks 1+2 rollout. K = mission-safety neighbor budget (floor = n-1-K).
    Hysteresis switcher: enter relay if P(relay)>hi OR degree<floor; release if
    P(relay)<lo AND degree recovered to >= floor+1."""
    wall = np.asarray(wall, bool)
    floor = (n - 1) - K
    positions = clustered_spawn(wall, n, comm_r, seed)
    env, world = make_env_world(H, W, wall, positions, sense_r, True)
    kb = [np.zeros((H, W)) for _ in range(n)]; seen = [np.zeros((H, W), bool) for _ in range(n)]
    goals = [None] * n; relaying = np.zeros(n, bool)
    nbr_pos = [dict() for _ in range(n)]; nbr_claim = [dict() for _ in range(n)]
    true_count = np.zeros((H, W)); owner = -np.ones((H, W), int)
    free_total = int((~wall).sum())
    conn_hits = 0; coh_sum = 0.0; relay_steps = 0; viol_sum = 0; t_cov = steps; frames = []
    key = jax.random.PRNGKey(seed)
    for t in range(steps):
        pos = np.asarray(world.body.position)
        for i in range(n):
            r, c = int(pos[i, 0]), int(pos[i, 1]); kb[i][r, c] += 1.0
            if true_count[r, c] == 0:
                owner[r, c] = i
            true_count[r, c] += 1.0
        vis = np.asarray(env.sensor.visibility(world))
        for i in range(n):
            seen[i] |= vis[i]
        adj = comm_adj(pos, comm_r)
        if is_connected(adj):
            conn_hits += 1
        comps = connected_components(adj); comp_of = {i: comp for comp in comps for i in comp}
        coh_sum += max(len(c) for c in comps) / n
        for comp in comps:
            if len(comp) > 1:
                cc = np.zeros((H, W)); ss = np.zeros((H, W), bool)
                for i in comp:
                    np.maximum(cc, kb[i], out=cc); ss |= seen[i]
                for i in comp:
                    kb[i] = cc.copy(); seen[i] = ss.copy()
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    nbr_pos[i][j] = (int(pos[j, 0]), int(pos[j, 1])); nbr_claim[i][j] = goals[j]
        deg = adj.sum(1)
        viol_sum += int((deg < floor).sum())             # mission-safety: lost > K neighbors

        actions = np.zeros(n, int)
        for i in range(n):
            src = (int(pos[i, 0]), int(pos[i, 1])); dist, parent = bfs(seen[i] & ~wall, src)
            feats = _features(i, pos, adj, comp_of.get(i, [i]), kb[i], seen[i], wall, comm_r, K, H, W, n)
            p = relay_prob(theta, feats)
            # hysteresis switcher (hard safety: below floor forces relay)
            # switcher: the LEARNED policy (with hysteresis) decides relaying; the only HARD
            # override is full detachment (deg==0 -> reconnect). Mission-safety (degree<floor)
            # is driven through the ES FITNESS, not a hard force, so ES can relay SELECTIVELY.
            if relaying[i]:
                if p < lo and deg[i] >= max(1, floor):
                    relaying[i] = False
            else:
                if p > hi or deg[i] == 0:
                    relaying[i] = True
            if relaying[i]:
                g = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
            else:
                g = goals[i]
                stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1.0)
                if stale:
                    claims = [nbr_claim[i][j] for j in comp_of.get(i, [i])
                              if j < i and nbr_claim[i].get(j) is not None]
                    g = _choose_target(kb[i], seen[i], wall, src, dist, claims, nbr_pos[i], comm_r, lam_frontier)
                goals[i] = g
            if g is not None:
                actions[i] = action_for(src, first_step(parent, src, g))
        relay_steps += int(relaying.sum())
        if record:
            frames.append(dict(pos=pos.copy(), owner=owner.copy(), count=true_count.copy(),
                               adj=adj.copy(), lam=np.zeros(n), goal=list(goals), relay=relaying.copy()))
        actions = collision_mask(actions, pos, wall)
        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)
        if (true_count >= 1).sum() / free_total >= 1.0 and t_cov == steps:
            t_cov = t

    m = evaluate(true_count, wall, conn_hits, steps, owner, n, t_cov)
    m["largest_comp_frac"] = coh_sum / steps
    m["relay_frac"] = relay_steps / max(1, n * steps)
    m["safety_viol"] = viol_sum / max(1, n * steps)
    m["wall"] = wall; m["frames"] = frames
    return m
