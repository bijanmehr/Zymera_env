"""Decentralized shared-exploration swarm (v1: deterministic uncertainty compass).

Each agent (a robot with an ID) runs the same loop on purely local information:
  Perception  : transient local sensing (zymera Sensor, radius r).
  Communication: with in-range neighbors (range x > r) it max-merges its certainty
                 field, takes their seen-mask, and records their pos + intent.
  Local KB    : count[H,W] (its known certainty field), seen-mask, neighbor tracks.
  Mission     : uncertainty compass -> head to the lowest-certainty reachable cell,
                de-conflicted by lower-ID neighbors' claims, and (by CHOICE, never a
                hard mask) penalizing targets that would pull it out of comm range.
  Control     : one BFS step toward the target. Operating = +1 to the current cell.

`lambda_conn` is the coverage<->connectivity knob (0 = ignore connectivity).
`share=False` ablates communication. Nothing here is central — each agent acts on
its own KB + messages only. The ground-truth field is tracked for metrics ONLY.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

from . import core
from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   connected_components, evaluate, first_step, is_connected, make_env_world)
from .connect import (degree_features, gossip_topology, global_cut_vertex, herd_target,
                      relay_features, relay_target)
from .backbone import leaf_target, relay_hold_target, spanning_tree
from .policy import degree_policy, policy_lambda, relay_gate


def _info_gain(count):
    """Diminishing marginal certainty gain — fresh cells worth most."""
    return 1.0 / np.sqrt(count + 1.0)


def _choose_target(kb_count, seen, wall, src, dist, claims, neighbor_pos,
                   comm_r, lambda_conn):
    """Uncertainty compass: pick the reachable known-free cell maximizing
    info_gain - dist_cost - connectivity_penalty, hard-excluding lower-ID claims."""
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return None
    # estimate where the swarm is, from last-known neighbor positions (local info)
    if neighbor_pos:
        npos = np.array(list(neighbor_pos.values()))
    else:
        npos = None
    best, best_score, best_excluded = None, -1e18, True
    for r, c in cand:
        d = dist[r, c]
        gain = _info_gain(kb_count[r, c])           # high for un-/under-operated cells
        score = 4.0 * gain - 0.04 * d
        excluded = False
        for (cr, cc) in claims:                     # de-confliction vs lower-ID neighbors
            if abs(cr - r) <= 2 and abs(cc - c) <= 2:
                excluded = True
                break
        if lambda_conn > 0 and npos is not None:     # connectivity BY CHOICE (soft)
            nearest = np.min(np.maximum(np.abs(npos[:, 0] - r), np.abs(npos[:, 1] - c)))
            if nearest > comm_r:                     # target would strand me
                score -= lambda_conn * (nearest - comm_r)
        # prefer non-excluded; among same exclusion-class, higher score
        if (not excluded and best_excluded) or (excluded == best_excluded and score > best_score):
            best, best_score, best_excluded = (int(r), int(c)), score, excluded
    return best


def _is_cut_vertex(adj, i):
    """Would removing agent i split its current neighbors apart? Uses only the
    adjacency rows agent i would have via gossip (its own + its neighbors' rows),
    so it's a decentralized, conservative 2-hop cut-vertex test. True -> agent i
    is a bridge holding (sub)groups together -> it should take the RELAY role."""
    nbrs = np.where(adj[i])[0]
    if len(nbrs) <= 1:
        return False                      # a leaf is never a cut-vertex
    n = adj.shape[0]
    known = np.zeros((n, n), dtype=bool)
    for a in nbrs:                        # i knows its neighbors' full adjacency rows
        known[a] = adj[a]
        known[:, a] = adj[:, a]
    known[i, :] = False                   # remove i
    known[:, i] = False
    seen = {int(nbrs[0])}
    q = deque([int(nbrs[0])])
    while q:
        u = q.popleft()
        for v in range(n):
            if known[u, v] and v not in seen:
                seen.add(v)
                q.append(v)
    return not all(int(nb) in seen for nb in nbrs)


def _agent_features(i, pos, adj, kb_count, seen, wall, comm_r, H, W):
    """Local, decentralized features for the learned role policy."""
    n = adj.shape[0]
    nbrs = np.where(adj[i])[0]
    f0 = len(nbrs) / max(1, n - 1)                       # how connected I am locally
    f1 = 1.0 if _is_cut_vertex(adj, i) else 0.0          # am I a bridge?
    if len(nbrs) > 0:
        d = np.max(np.abs(pos[nbrs] - pos[i]), axis=1)
        f2 = min(float(np.min(d)), comm_r) / comm_r      # margin to nearest link (1=at edge)
    else:
        f2 = 1.0
    r, c = int(pos[i, 0]), int(pos[i, 1])
    rad = 4
    r0, r1, c0, c1 = max(0, r - rad), min(H, r + rad + 1), max(0, c - rad), min(W, c + rad + 1)
    wfree = seen[r0:r1, c0:c1] & ~wall[r0:r1, c0:c1]
    wun = wfree & (kb_count[r0:r1, c0:c1] < 1)
    f3 = wun.sum() / max(1, wfree.sum())                 # local unexplored density
    free = seen & ~wall
    f4 = (free & (kb_count >= 1)).sum() / max(1, free.sum())   # my coverage progress
    grp = np.vstack([pos[i][None], pos[nbrs]]) if len(nbrs) > 0 else pos[i][None]
    cen = grp.mean(0)
    f5 = float(np.max(np.abs(pos[i] - cen))) / max(1, comm_r)  # how peripheral I am
    return np.array([f0, f1, f2, f3, f4, f5])


def run_swarm(H, W, wall, n_agents, comm_r, sense_r, steps, seed, *,
              lambda_conn=1.0, theta=None, roles=False, lam_relay=8.0, lam_frontier=0.0,
              share=True, occlusion=True, record=False,
              topo_gossip=False, relay=False, relay_margin=1, gate_theta=None,
              degree_floor=None, hysteresis=1, degree_theta=None,
              backbone=False, gnn_theta=None, attn_theta=None):
    wall = np.asarray(wall, bool)
    positions = clustered_spawn(wall, n_agents, comm_r, seed)
    env, world = make_env_world(H, W, wall, positions, sense_r, occlusion)

    kb_count = [np.zeros((H, W)) for _ in range(n_agents)]      # each agent's KB field
    seen = [np.zeros((H, W), bool) for _ in range(n_agents)]    # known occupancy
    goals = [None] * n_agents
    role_relay = np.zeros(n_agents, bool)                       # persistent relay/frontier role (viz)
    returning = np.zeros(n_agents, bool)                        # degree-budget state: heading back to herd
    d_ref = np.zeros(n_agents)                                  # per-agent peak degree seen (budget reference)
    cur_lam = np.full(n_agents, float(lambda_conn))             # current per-agent role weight (viz)
    recent = [deque(maxlen=10) for _ in range(n_agents)]        # last-10 operated (msg payload)
    nbr_pos = [dict() for _ in range(n_agents)]                 # last-heard neighbor positions
    nbr_claim = [dict() for _ in range(n_agents)]               # last-heard neighbor intents

    true_count = np.zeros((H, W))                               # ground truth (metrics only)
    owner = -np.ones((H, W), int)
    key = jax.random.PRNGKey(seed)
    conn_hits, frames = 0, []
    largest_frac_sum, ncomp_sum, relay_steps = 0.0, 0.0, 0
    free_total = int((~wall).sum())
    t_cov = steps

    for t in range(steps):
        pos = np.asarray(world.body.position)

        # --- operate current cell (+1) ---
        for i in range(n_agents):
            r, c = int(pos[i, 0]), int(pos[i, 1])
            kb_count[i][r, c] += 1.0
            recent[i].append((r, c))
            if true_count[r, c] == 0:
                owner[r, c] = i
            true_count[r, c] += 1.0

        # --- perceive (local) ---
        vis = np.asarray(env.sensor.visibility(world))          # (N,H,W) currently visible
        for i in range(n_agents):
            seen[i] |= vis[i]

        # --- communicate with in-range neighbors (decentralized) ---
        adj = comm_adj(pos, comm_r)
        if is_connected(adj):
            conn_hits += 1
        comps = connected_components(adj)
        largest_frac_sum += max(len(c) for c in comps) / n_agents   # cohesion over time
        ncomp_sum += len(comps)
        # gossip is built only when actually used: by the gossip heuristic path or
        # by the learned gate's features. The no-gossip relay ablation skips it
        # entirely and falls back to the local 2-hop cut-vertex test.
        topo_est = gossip_topology(adj, comps) if (topo_gossip or gate_theta is not None) else None
        if share:
            for comp in comps:                                  # max-merge within a component
                if len(comp) > 1:
                    cc_max = np.zeros((H, W))
                    ss = np.zeros((H, W), bool)
                    for i in comp:
                        np.maximum(cc_max, kb_count[i], out=cc_max)
                        ss |= seen[i]
                    for i in comp:
                        kb_count[i] = cc_max.copy()
                        seen[i] = ss.copy()
        for i in range(n_agents):                               # share pos + intent (+last-10)
            for j in range(n_agents):
                if adj[i, j]:
                    nbr_pos[i][j] = (int(pos[j, 0]), int(pos[j, 1]))
                    nbr_claim[i][j] = goals[j]
                    if share:
                        for (r, c) in recent[j]:
                            kb_count[i][r, c] = max(kb_count[i][r, c], 1.0)

        comp_of = {i: comp for comp in comps for i in comp}

        # --- spanning-tree backbone (v0/v1/v2): role = position in the tree ---
        if backbone:
            tree_parent, tree_children, tree_internal = spanning_tree(adj, comps)

        # --- mission: uncertainty compass per agent ---
        actions = np.zeros(n_agents, int)
        for i in range(n_agents):
            src = (int(pos[i, 0]), int(pos[i, 1]))
            free_i = seen[i] & ~wall
            dist, parent = bfs(free_i, src)

            if backbone:
                # role read off the spanning tree (recomputed each tick as it moves).
                # v0: deterministic. v1: a GNN may FLIP a leaf->relay (learned role
                # refinement). v2: an attention head may re-rank the leaf's frontier goal.
                be_relay = bool(tree_internal[i])
                if gnn_theta is not None and not be_relay:
                    from .policy import gnn_relay
                    feats = _agent_features(i, pos, adj, kb_count[i], seen[i], wall, comm_r, H, W)
                    nbr_feats = [_agent_features(j, pos, adj, kb_count[j], seen[j], wall, comm_r, H, W)
                                 for j in np.where(adj[i])[0]]
                    be_relay = gnn_relay(gnn_theta, feats, nbr_feats) > 0.5
                role_relay[i] = be_relay
                cur_lam[i] = lam_relay if be_relay else 0.0
                if be_relay:
                    goals[i] = relay_hold_target(i, pos, tree_parent, tree_children, seen[i], wall, dist)
                else:
                    g = goals[i]
                    stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]])
                             or kb_count[i][g[0], g[1]] >= 1.0)
                    if stale:
                        if attn_theta is not None:
                            from .policy import attn_goal
                            goals[i] = attn_goal(attn_theta, i, pos, tree_parent, kb_count[i],
                                                 seen[i], wall, dist, comm_r)
                        else:
                            goals[i] = leaf_target(i, pos, tree_parent, kb_count[i], seen[i],
                                                   wall, dist, comm_r)
                g = goals[i]
            elif degree_floor is not None or degree_theta is not None:
                # === degree-signal role machine (replaces lambda) ===
                # Checked EVERY step (a safety constraint, harsh). The agent RETURNS
                # to the herd when its degree signal says so — either below a fixed
                # floor (lost > budget K neighbors), or per a LEARNED policy over the
                # degree features (degree_theta). Return MOVES it inward (regaining
                # degree), which clears its own trigger, so no agent gets stuck.
                deg = int(adj[i].sum())
                d_ref[i] = max(d_ref[i], deg)
                if degree_theta is not None:            # learned degree policy (ES/MARL)
                    comp_size = len(comp_of.get(i, [i]))   # |my component| via gossip = global-split signal
                    feats = degree_features(i, pos, adj, comp_size, d_ref[i], kb_count[i],
                                            seen[i], wall, n_agents, H, W)
                    returning[i] = degree_policy(degree_theta, feats) > 0.5
                else:                                   # fixed degree floor + hysteresis
                    if returning[i]:
                        if deg >= degree_floor + hysteresis:
                            returning[i] = False
                    elif deg < degree_floor:
                        returning[i] = True
                role_relay[i] = returning[i]
                cur_lam[i] = lam_relay if returning[i] else 0.0
                if returning[i]:                        # re-aim at the (moving) herd each step
                    goals[i] = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
                else:                                   # frontier: commit-until-reached, NO lambda
                    g = goals[i]
                    stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]])
                             or kb_count[i][g[0], g[1]] >= 1.0)
                    if stale:
                        claims = [nbr_claim[i][j] for j in comp_of.get(i, [i])
                                  if j < i and nbr_claim[i].get(j) is not None]
                        goals[i] = _choose_target(kb_count[i], seen[i], wall, src, dist,
                                                  claims, nbr_pos[i], comm_r, 0.0)
                g = goals[i]
            else:
                # === existing modes: cut-vertex relay / learned gate / soft lambda ===
                g = goals[i]
                stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]])
                         or kb_count[i][g[0], g[1]] >= 1.0)
                if stale:
                    claims = [nbr_claim[i][j] for j in comp_of.get(i, [i])
                              if j < i and nbr_claim[i].get(j) is not None]
                    my_comp = comp_of.get(i, [i])
                    be_relay = False
                    nbrs = np.where(adj[i])[0]
                    if relay or gate_theta is not None:
                        sub = topo_est[i] if topo_est is not None else None
                        if gate_theta is not None:     # learned relay-gate (v3)
                            feats = np.concatenate([
                                _agent_features(i, pos, adj, kb_count[i], seen[i], wall, comm_r, H, W),
                                relay_features(i, pos, adj, my_comp, sub, comm_r)])
                            be_relay = relay_gate(gate_theta, feats) > 0.5
                        else:                          # heuristic relay trigger (cut-vertex / detached / leaf-edge)
                            if topo_gossip:
                                gcv = global_cut_vertex(sub, i, my_comp)
                            else:
                                gcv = _is_cut_vertex(adj, i)
                            detached = len(nbrs) == 0
                            leaf_edge = False
                            if len(nbrs) >= 1 and len(nbrs) <= relay_margin:
                                mind = float(np.min(np.max(np.abs(pos[nbrs] - pos[i]), axis=1)))
                                leaf_edge = mind >= comm_r - 1
                            be_relay = gcv or detached or leaf_edge
                    if theta is not None:              # learned lambda policy (v2)
                        lam_i = policy_lambda(theta, _agent_features(
                            i, pos, adj, kb_count[i], seen[i], wall, comm_r, H, W))
                    elif roles:                        # heuristic role-picker
                        lam_i = lam_relay if _is_cut_vertex(adj, i) else lam_frontier
                    else:
                        lam_i = lambda_conn
                    cur_lam[i] = lam_relay if be_relay else lam_i
                    role_relay[i] = be_relay
                    if be_relay:                       # old cut-vertex mode: HOLD the bridge
                        g = relay_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
                    else:
                        g = _choose_target(kb_count[i], seen[i], wall, src, dist, claims,
                                           nbr_pos[i], comm_r, lam_i)
                    goals[i] = g
            if g is not None:
                actions[i] = action_for(src, first_step(parent, src, g))

        relay_steps += int(role_relay.sum())            # agent-steps in relay role (uniform across modes)

        # connectivity is a CHOICE (baked into target scoring) — no hard mask here.
        cov = (true_count >= 1).sum() / free_total
        if cov >= 1.0 and t_cov == steps:
            t_cov = t
        if record:
            frames.append(dict(pos=pos.copy(), owner=owner.copy(),
                               count=true_count.copy(), adj=adj.copy(),
                               lam=cur_lam.copy(), goal=list(goals),
                               relay=role_relay.copy()))

        actions = collision_mask(actions, pos, wall)    # HARD collision mask (ID-priority)
        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)

    # final operate of last positions
    pos = np.asarray(world.body.position)
    for i in range(n_agents):
        r, c = int(pos[i, 0]), int(pos[i, 1])
        if true_count[r, c] == 0:
            owner[r, c] = i
        true_count[r, c] += 1.0
    if (true_count >= 1).sum() / free_total >= 1.0 and t_cov == steps:
        t_cov = steps

    m = evaluate(true_count, wall, conn_hits, steps, owner, n_agents, t_cov)
    m.update(true_count=true_count, owner=owner, wall=wall, frames=frames,
             largest_comp_frac=largest_frac_sum / steps,    # avg cohesion (1.0 = always one piece)
             n_comp=ncomp_sum / steps,                       # avg # fragments
             relay_frac=relay_steps / max(1, n_agents * steps))  # share of agent-steps spent relaying
    return m
