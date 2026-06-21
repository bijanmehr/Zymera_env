"""CTDE GNN actor-critic with graph-shape estimation (16x16 default setup).

  Actor  (decentralized): a GCN over the agent's GOSSIPED component graph -> per-agent
          role (frontier / relay) + an estimate of the global graph SHAPE (adjacency).
          Out-of-component agents are masked 'unknown' -> the actor must predict their
          links (the privileged shape-supervision teaches this).
  Critic (centralized, training only): a GCN over the TRUE global graph -> team value.
  Train : A2C with Monte-Carlo returns (MC > TD here) + shape-estimation aux loss
          (BCE of predicted vs true adjacency). Env is host-side numpy; nets are JAX
          (policy-gradient needs no differentiable env). Role drives the backbone targets.

Both actor and critic operate on GRAPH SHAPE (the adjacency is the GCN operator), and
the actor's aux target is the true shape -> "both focus on the shape of the graph".

  PYTHONPATH=. .venv/bin/python -m swarm_explore.gnn_ac --episodes 400
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   connected_components, evaluate, first_step, is_connected, make_env_world,
                   random_wall)
from .connect import herd_target
from .swarm import _choose_target, _is_cut_vertex

F = 6          # node features: x, y, degree, coverage-progress, in-component, unknown
HID = 32
GAMMA = 0.99
COV_W = 30.0   # reward = COV_W * coverage-gain + COH_W * cohesion - SAFETY_PEN * violations
COH_W = 0.20
# MISSION SAFETY (degree budget): start fully connected (degree N-1); an agent may lose
# up to K_SAFE neighbors. Drop below the floor (N-1-K_SAFE) = "mission broken" for that
# agent -> heavy per-step penalty. The sharp, degree-tied signal that makes role choice
# track the graph (vs the diffuse global reward, which gave flat structure-blind roles).
K_SAFE = 2
SAFETY_PEN = 0.5     # heavy enough to discourage full detachment, light enough not to
                     # collapse into a clump (SAFETY_PEN=2.0 tanked coverage to ~18%)
W_AUX = 0.5    # shape-estimation loss weight
W_VAL = 0.5


# ---------------- networks (pure-JAX GCN) ----------------
def init_params(key):
    ks = jax.random.split(key, 7)
    def w(k, a, b):
        return jax.random.normal(k, (a, b)) * (0.7 / np.sqrt(a))
    return dict(W1=w(ks[0], F, HID), W2=w(ks[1], HID, HID),
                Wr=w(ks[2], HID, 2), We=w(ks[3], HID, HID),
                C1=w(ks[4], F, HID), C2=w(ks[5], HID, HID), Cv=w(ks[6], HID, 1))


def _gcn(A, X, W1, W2):
    n = A.shape[0]
    Ah = A + jnp.eye(n)
    Ah = Ah / Ah.sum(1, keepdims=True)
    H = jax.nn.relu(Ah @ (X @ W1))
    H = jax.nn.relu(Ah @ (H @ W2))
    return H


def actor_fwd(p, X, A):
    H = _gcn(A, X, p["W1"], p["W2"])
    role_logits = H @ p["Wr"]              # (N,2)
    Z = H @ p["We"]
    shape_logits = Z @ Z.T                 # (N,N) predicted adjacency logits
    return role_logits, shape_logits


def critic_fwd(p, Xc, Ac):
    H = _gcn(Ac, Xc, p["C1"], p["C2"])
    return (H.mean(0) @ p["Cv"])[0]


_ACT = jax.jit(actor_fwd)


# ---------------- features ----------------
def node_features(pos, adj, comp_set, cov_prog, H, W, n):
    X = np.zeros((n, F), np.float32)
    for j in range(n):
        if j in comp_set:
            X[j, 0] = pos[j, 0] / H
            X[j, 1] = pos[j, 1] / W
            X[j, 2] = adj[j].sum() / max(1, n - 1)
            X[j, 3] = cov_prog[j]
            X[j, 4] = 1.0                  # in-component
        else:
            X[j, 5] = 1.0                  # unknown
    return X


# ---------------- rollout (numpy env + JAX actor) ----------------
def rollout(params, H, W, wall, n, comm_r, sense_r, steps, seed, key, greedy=False, record=False):
    positions = clustered_spawn(wall, n, comm_r, seed)
    env, world = make_env_world(H, W, wall, positions, sense_r, True)
    kb = [np.zeros((H, W)) for _ in range(n)]
    seen = [np.zeros((H, W), bool) for _ in range(n)]
    goals = [None] * n
    nbr_pos = [dict() for _ in range(n)]; nbr_claim = [dict() for _ in range(n)]
    true_count = np.zeros((H, W)); owner = -np.ones((H, W), int)
    free_total = int((~wall).sum())
    conn_hits = 0; coh_sum = 0.0; relay_steps = 0; t_cov = steps; viol_sum = 0
    floor_deg = (n - 1) - K_SAFE                  # mission-safety degree floor
    shape_acc_sum = 0.0; shape_cnt = 0; frames = []
    aX, aA, aidx, arole, astep = [], [], [], [], []
    acut, aprelay, adeg = [], [], []            # per-sample probes: cut-vertex, P(relay), degree
    cXc, cAc, crew = [], [], []
    prev_cov = 0.0; pkey = key
    for t in range(steps):
        pos = np.asarray(world.body.position)
        for i in range(n):
            r, c = int(pos[i, 0]), int(pos[i, 1])
            kb[i][r, c] += 1.0
            if true_count[r, c] == 0:
                owner[r, c] = i
            true_count[r, c] += 1.0
        vis = np.asarray(env.sensor.visibility(world))
        for i in range(n):
            seen[i] |= vis[i]
        adj = comm_adj(pos, comm_r)
        if is_connected(adj):
            conn_hits += 1
        comps = connected_components(adj)
        coh_sum += max(len(cc) for cc in comps) / n
        for comp in comps:
            if len(comp) > 1:
                cc_max = np.zeros((H, W)); ss = np.zeros((H, W), bool)
                for i in comp:
                    np.maximum(cc_max, kb[i], out=cc_max); ss |= seen[i]
                for i in comp:
                    kb[i] = cc_max.copy(); seen[i] = ss.copy()
        # share pos + intent with in-range neighbors (for compass de-confliction + relay)
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    nbr_pos[i][j] = (int(pos[j, 0]), int(pos[j, 1])); nbr_claim[i][j] = goals[j]
        comp_of = {i: comp for comp in comps for i in comp}
        cov_prog = np.array([(seen[i] & ~wall & (kb[i] >= 1)).sum() / max(1, (seen[i] & ~wall).sum())
                             for i in range(n)], np.float32)
        Xc = node_features(pos, adj, set(range(n)), cov_prog, H, W, n)
        Ac = adj.astype(np.float32)

        roles = np.zeros(n, int)
        for comp in comps:
            cs = set(comp)
            Xb = node_features(pos, adj, cs, cov_prog, H, W, n)
            m = np.array([1.0 if j in cs else 0.0 for j in range(n)], np.float32)
            Ab = Ac * m[None, :] * m[:, None]
            rl, sl = _ACT(params, jnp.asarray(Xb), jnp.asarray(Ab))
            rl = np.asarray(rl)
            pred = (1.0 / (1.0 + np.exp(-np.asarray(sl)))) > 0.5      # predicted adjacency (shape)
            off = ~np.eye(n, dtype=bool)
            shape_acc_sum += float((pred[off] == (Ac[off] > 0.5)).mean()); shape_cnt += 1
            for i in comp:
                d = rl[i, 1] - rl[i, 0]
                p_relay = 1.0 / (1.0 + np.exp(-d))
                if greedy:
                    a = int(p_relay > 0.5)
                else:
                    pkey, sk = jax.random.split(pkey)
                    a = int(float(jax.random.uniform(sk)) < p_relay)
                roles[i] = a
                aX.append(Xb); aA.append(Ab); aidx.append(i); arole.append(a); astep.append(t)
                acut.append(1 if _is_cut_vertex(adj, i) else 0); aprelay.append(float(p_relay))
                adeg.append(int(adj[i].sum()))
        relay_steps += int(roles.sum())

        # roles -> targets: FRONTIER = uncertainty compass (free exploration);
        # RELAY = hold/reposition toward neighbors to maintain links. (No backbone crutch.)
        actions = np.zeros(n, int)
        for i in range(n):
            src = (int(pos[i, 0]), int(pos[i, 1]))
            dist, parent = bfs(seen[i] & ~wall, src)
            if roles[i] == 1:
                g = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
            else:
                g = goals[i]
                stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1.0)
                if stale:
                    claims = [nbr_claim[i][j] for j in comp_of.get(i, [i])
                              if j < i and nbr_claim[i].get(j) is not None]
                    g = _choose_target(kb[i], seen[i], wall, src, dist, claims, nbr_pos[i], comm_r, 0.0)
                goals[i] = g
            if g is not None:
                actions[i] = action_for(src, first_step(parent, src, g))
        actions = collision_mask(actions, pos, wall)
        pkey, sk = jax.random.split(pkey)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)

        cov = (true_count >= 1).sum() / free_total
        deg = adj.sum(1)                                   # per-agent comm degree
        viol = int((deg < floor_deg).sum())               # lost > K_SAFE neighbors -> safety break
        viol_sum += viol
        crew.append(COV_W * (cov - prev_cov) + COH_W * (max(len(cc) for cc in comps) / n)
                    - SAFETY_PEN * viol)
        prev_cov = cov
        cXc.append(Xc); cAc.append(Ac)
        if record:
            frames.append(dict(pos=pos.copy(), owner=owner.copy(), count=true_count.copy(),
                               adj=adj.copy(), lam=np.zeros(n), goal=list(goals),
                               relay=(roles == 1)))
        if cov >= 1.0 and t_cov == steps:
            t_cov = t

    m = evaluate(true_count, wall, conn_hits, steps, owner, n, t_cov)
    m["largest_comp_frac"] = coh_sum / steps
    m["relay_frac"] = relay_steps / max(1, n * steps)
    m["shape_acc"] = shape_acc_sum / max(1, shape_cnt)
    m["safety_viol"] = viol_sum / max(1, n * steps)        # frac of agent-steps below the degree floor
    m["wall"] = wall; m["frames"] = frames
    # MC returns
    crew = np.array(crew, np.float32); ret = np.zeros_like(crew)
    acc = 0.0
    for t in range(len(crew) - 1, -1, -1):
        acc = crew[t] + GAMMA * acc; ret[t] = acc
    traj = dict(aX=np.array(aX, np.float32), aA=np.array(aA, np.float32),
                aidx=np.array(aidx, np.int32), arole=np.array(arole, np.int32),
                astep=np.array(astep, np.int32), cXc=np.array(cXc, np.float32),
                cAc=np.array(cAc, np.float32), ret=ret,
                acut=np.array(acut, np.int32), aprelay=np.array(aprelay, np.float32))
    m["acut"] = np.array(acut, np.int32); m["aprelay"] = np.array(aprelay, np.float32)
    m["adeg"] = np.array(adeg, np.int32)
    return traj, m


# ---------------- A2C loss + update ----------------
def _loss(params, aX, aA, aidx, arole, astep, cXc, cAc, ret):
    rl, sl = jax.vmap(lambda X, A: actor_fwd(params, X, A))(aX, aA)      # (M,N,2),(M,N,N)
    M = aidx.shape[0]
    role_logits = rl[jnp.arange(M), aidx]                               # (M,2)
    logp_all = jax.nn.log_softmax(role_logits, -1)
    logp = logp_all[jnp.arange(M), arole]                               # (M,)
    values = jax.vmap(lambda X, A: critic_fwd(params, X, A))(cXc, cAc)  # (T,)
    adv = ret - jax.lax.stop_gradient(values)                          # (T,)
    a_adv = adv[astep]                                                  # (M,)
    a_adv = (a_adv - a_adv.mean()) / (a_adv.std() + 1e-6)
    policy_loss = -jnp.mean(logp * a_adv)
    value_loss = jnp.mean((ret - values) ** 2)
    # shape-estimation aux: predicted adjacency vs true adjacency (per actor sample's step)
    true_adj = cAc[astep]                                               # (M,N,N)
    bce = optax.sigmoid_binary_cross_entropy(sl, true_adj).mean()
    return policy_loss + W_VAL * value_loss + W_AUX * bce, (policy_loss, value_loss, bce)


def make_updater(opt):
    @jax.jit
    def upd(params, opt_state, aX, aA, aidx, arole, astep, cXc, cAc, ret):
        (loss, parts), g = jax.value_and_grad(_loss, has_aux=True)(
            params, aX, aA, aidx, arole, astep, cXc, cAc, ret)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, parts
    return upd


def merge(trajs):
    """concatenate B episode trajectories with global step offsets."""
    off = 0; aX = []; aA = []; aidx = []; arole = []; astep = []; cXc = []; cAc = []; ret = []
    for tr in trajs:
        aX.append(tr["aX"]); aA.append(tr["aA"]); aidx.append(tr["aidx"]); arole.append(tr["arole"])
        astep.append(tr["astep"] + off); cXc.append(tr["cXc"]); cAc.append(tr["cAc"]); ret.append(tr["ret"])
        off += len(tr["ret"])
    cat = lambda L: np.concatenate(L, 0)
    return (jnp.asarray(cat(aX)), jnp.asarray(cat(aA)), jnp.asarray(cat(aidx)),
            jnp.asarray(cat(arole)), jnp.asarray(cat(astep)), jnp.asarray(cat(cXc)),
            jnp.asarray(cat(cAc)), jnp.asarray(cat(ret)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H = W = a.grid; n = a.agents; sr = 3
    wall = random_wall(H, W, 0.05, 0)
    key = jax.random.PRNGKey(a.seed)
    key, ik = jax.random.split(key)
    params = init_params(ik)
    opt = optax.adam(a.lr); opt_state = opt.init(params)
    upd = make_updater(opt)

    n_updates = a.episodes // a.batch
    hist = []
    print(f"GNN actor-critic @ {H}x{H}/{n}/cr{a.comm_r}/{a.steps}st  {a.episodes}ep (batch {a.batch})", flush=True)
    for u in range(n_updates):
        trajs = []; metrics = []
        for b in range(a.batch):
            key, rk = jax.random.split(key)
            seed = int(jax.random.randint(rk, (), 0, 100000))
            wl = random_wall(H, W, 0.05, seed % 50)
            key, rk2 = jax.random.split(key)
            tr, m = rollout(params, H, W, wl, n, a.comm_r, sr, a.steps, seed, rk2)
            trajs.append(tr); metrics.append(m)
        batch = merge(trajs)
        params, opt_state, loss, parts = upd(params, opt_state, *batch)
        if u % 5 == 0 or u == n_updates - 1:
            cov = np.mean([m["coverage"] for m in metrics])
            conn = np.mean([m["connectivity"] for m in metrics])
            coh = np.mean([m["largest_comp_frac"] for m in metrics])
            rl = np.mean([m["relay_frac"] for m in metrics])
            sv = np.mean([m["safety_viol"] for m in metrics])
            pl, vl, bce = (float(x) for x in parts)
            hist.append(dict(update=u, cov=float(cov), conn=float(conn), coh=float(coh),
                             relay=float(rl), safety_viol=float(sv), shape_bce=bce))
            print(f"  upd {u:3d}  cov {cov*100:5.1f}%  conn {conn*100:4.0f}%  coh {coh*100:4.0f}%  "
                  f"relay {rl*100:3.0f}%  safety-viol {sv*100:4.1f}%  | loss(pg {pl:+.3f} v {vl:.3f} bce {bce:.3f})", flush=True)

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore",
                                        "gnnac-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "params.npz"), **{k: np.asarray(v) for k, v in params.items()})
    json.dump({"config": vars(a), "history": hist}, open(os.path.join(out, "train.json"), "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
