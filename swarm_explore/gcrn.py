"""Size-invariant recurrent graph belief (GCRN). Train on 16x16/4, transfer to 32x32/10.

Per agent i, a recurrent belief H_i[j] over every agent j:
  - GRU node memory (carries belief through blackouts),
  - message-pass over i's BELIEVED (component) graph,
  - bilinear decoder -> predicted full adjacency.
Size-invariant by design: shared weights, mean aggregation, comm-range-relative offsets,
capped raw degree (no N-dependent normalization). Method: privileged distillation
(predict the TRUE adjacency the simulator knows) via BPTT.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.gcrn --epochs 30
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
                   connected_components, first_step, make_env_world, random_wall)
from .connect import herd_target
from .swarm import _choose_target

COMM_R = 5
HID = 16
F_IN = 4          # per (perspective i, target j): observed, rel_dr, rel_dc, deg_j


def step_inputs(pos, adj, comps, n, comm_r=COMM_R):
    """Per-perspective GCRN inputs from one frame. Shared by collect_traj (training) and
    the runtime belief loop, so train/run use IDENTICAL feature construction.
    Returns inp (n,n,F_IN), adjb (n,n,n)."""
    cof = {i: set(c) for c in comps for i in c}
    deg = adj.sum(1)
    inp = np.zeros((n, n, F_IN), np.float32); adjb = np.zeros((n, n, n), np.float32)
    for i in range(n):
        comp = cof[i]
        for j in range(n):
            if j in comp:
                inp[i, j, 0] = 1.0
                inp[i, j, 1:3] = (pos[j] - pos[i]) / comm_r          # comm-range-relative offset
                inp[i, j, 3] = min(deg[j], 8) / 8.0                  # capped raw degree (size-invariant)
        idx = list(comp)
        for a in idx:
            for b in idx:
                if adj[a, b]:
                    adjb[i, a, b] = 1.0                              # i's believed (component) subgraph
    return inp, adjb


# ---------------- trajectory collection (numpy rollout, fixed behavior) ----------------
def collect_traj(grid, n, steps, seed):
    wall = random_wall(grid, grid, 0.05, seed)
    pos0 = clustered_spawn(wall, n, COMM_R, seed)
    env, world = make_env_world(grid, grid, wall, pos0, 3, True)
    kb = [np.zeros((grid, grid)) for _ in range(n)]; seen = [np.zeros((grid, grid), bool) for _ in range(n)]
    goals = [None] * n; ret = np.zeros(n, bool); npos = [dict() for _ in range(n)]
    INP = []; ADJB = []; TRUE = []                # per step: (n,n,F_IN), (n,n,n), (n,n)
    key = jax.random.PRNGKey(seed)
    for t in range(steps):
        pos = np.asarray(world.body.position)
        for i in range(n):
            kb[i][int(pos[i, 0]), int(pos[i, 1])] += 1
        vis = np.asarray(env.sensor.visibility(world))
        for i in range(n):
            seen[i] |= vis[i]
        adj = comm_adj(pos, COMM_R); comps = connected_components(adj)
        for c in comps:
            if len(c) > 1:
                cc = np.zeros((grid, grid)); ss = np.zeros((grid, grid), bool)
                for i in c: np.maximum(cc, kb[i], out=cc); ss |= seen[i]
                for i in c: kb[i] = cc.copy(); seen[i] = ss.copy()
        inp, adjb = step_inputs(pos, adj, comps, n)
        INP.append(inp); ADJB.append(adjb); TRUE.append(adj.astype(np.float32))
        # behavior: compass + degree-floor relay (floor 1) -> diverse spread/fragment data
        floor = 1; act = np.zeros(n, int)
        for i in range(n):
            for j in range(n):
                if adj[i, j]: npos[i][j] = (int(pos[j, 0]), int(pos[j, 1]))
            src = (int(pos[i, 0]), int(pos[i, 1])); dist, par = bfs(seen[i] & ~wall, src)
            if ret[i]:
                if deg[i] >= floor + 1: ret[i] = False
            elif deg[i] < floor: ret[i] = True
            if ret[i]:
                g = herd_target(i, pos, adj, npos[i], seen[i], wall, dist)
            else:
                g = goals[i]
                st = (g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1)
                if st: g = _choose_target(kb[i], seen[i], wall, src, dist, [], npos[i], COMM_R, 0.0)
                goals[i] = g
            if g is not None: act[i] = action_for(src, first_step(par, src, g))
        act = collision_mask(act, pos, wall)
        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(act, jnp.int32), sk)
    return np.array(INP), np.array(ADJB), np.array(TRUE)          # (T,n,n,F_IN),(T,n,n,n),(T,n,n)


# ---------------- GCRN (JAX) ----------------
def init_params(key):
    ks = jax.random.split(key, 12)
    def w(k, a, b): return jax.random.normal(k, (a, b)) * (0.6 / np.sqrt(a))
    z = lambda d: jnp.zeros((d,))
    return dict(
        # GRU (input F_IN, hidden HID)
        Wxz=w(ks[0], F_IN, HID), Whz=w(ks[1], HID, HID), bz=z(HID),
        Wxr=w(ks[2], F_IN, HID), Whr=w(ks[3], HID, HID), br=z(HID),
        Wxh=w(ks[4], F_IN, HID), Whh=w(ks[5], HID, HID), bh=z(HID),
        Wmp=w(ks[6], HID, HID), We=w(ks[7], HID, HID))


def gru(p, x, h):                          # x (.,F_IN), h (.,HID)
    zg = jax.nn.sigmoid(x @ p["Wxz"] + h @ p["Whz"] + p["bz"])
    rg = jax.nn.sigmoid(x @ p["Wxr"] + h @ p["Whr"] + p["br"])
    hh = jnp.tanh(x @ p["Wxh"] + (rg * h) @ p["Whh"] + p["bh"])
    return (1 - zg) * h + zg * hh


def step_one(p, H, inp, adjb):
    """H (n,n,HID), inp (n,n,F_IN), adjb (n,n,n) -> H', logits (n,n,n)."""
    n = H.shape[0]
    # GRU update per (perspective, target)
    H = jax.vmap(lambda Hi, xi: gru(p, xi, Hi))(H, inp)          # (n,n,HID)
    # message-pass over each perspective's believed graph (mean-normalized)
    def mp(Hi, Ab):
        A = Ab + jnp.eye(n)
        A = A / A.sum(1, keepdims=True)
        return jax.nn.relu(A @ (Hi @ p["Wmp"]))
    H = jax.vmap(mp)(H, adjb)                                    # (n,n,HID)
    # bilinear decode per perspective -> predicted adjacency (n,n)
    def dec(Hi):
        Z = Hi @ p["We"]; return Z @ Z.T
    logits = jax.vmap(dec)(H)                                    # (n,n,n) [perspective, j, k]
    return H, logits


def rollout_belief(p, INP, ADJB):
    """scan over time. INP (T,n,n,F_IN), ADJB (T,n,n,n) -> logits (T,n,n,n)."""
    n = INP.shape[1]
    H0 = jnp.zeros((n, n, HID))
    def f(H, x):
        inp, adjb = x
        H, lg = step_one(p, H, inp, adjb)
        return H, lg
    _, logits = jax.lax.scan(f, H0, (INP, ADJB))
    return logits


def loss_fn(p, INP, ADJB, TRUE):
    logits = rollout_belief(p, INP, ADJB)                       # (T,n,n,n)
    n = TRUE.shape[1]
    off = ~jnp.eye(n, dtype=bool)
    tgt = jnp.broadcast_to(TRUE[:, None, :, :], logits.shape)   # true adj, same for every perspective
    bce = optax.sigmoid_binary_cross_entropy(logits, tgt)
    return (bce * off[None, None]).sum() / (off.sum() * logits.shape[0] * n)


def accuracy(p, INP, ADJB, TRUE):
    logits = np.asarray(rollout_belief(p, jnp.asarray(INP), jnp.asarray(ADJB)))
    n = TRUE.shape[1]; off = ~np.eye(n, dtype=bool)
    pred = (1 / (1 + np.exp(-logits)) > 0.5)
    tgt = np.broadcast_to(TRUE[:, None, :, :], logits.shape) > 0.5
    return float((pred[:, :, off] == tgt[:, :, off]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    a = ap.parse_args()
    print("GCRN size-invariant belief — train @16x16/4, transfer @32x32/10", flush=True)
    # data
    tr = [collect_traj(16, 4, 70, s) for s in range(a.train_seeds)]
    te16 = [collect_traj(16, 4, 70, s) for s in range(100, 105)]
    te32 = [collect_traj(32, 10, 100, s) for s in range(200, 205)]
    print(f"  data: {len(tr)} train(16x16/4), {len(te16)} test16, {len(te32)} test32", flush=True)
    key = jax.random.PRNGKey(0); params = init_params(key)
    opt = optax.adam(a.lr); opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, INP, ADJB, TRUE):
        l, g = jax.value_and_grad(loss_fn)(params, INP, ADJB, TRUE)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, l

    def triv(tset):  # 'all-connected' baseline accuracy
        accs = []
        for INP, ADJB, TRUE in tset:
            n = TRUE.shape[1]; off = ~np.eye(n, dtype=bool); accs.append((TRUE[:, off] > 0.5).mean())
        return float(np.mean(accs))
    hist = []
    for e in range(a.epochs):
        ls = 0.0
        for INP, ADJB, TRUE in tr:
            params, opt_state, l = step(params, opt_state, jnp.asarray(INP), jnp.asarray(ADJB), jnp.asarray(TRUE))
            ls += float(l)
        if e % 5 == 0 or e == a.epochs - 1:
            a16 = np.mean([accuracy(params, *t) for t in te16])
            a32 = np.mean([accuracy(params, *t) for t in te32])
            hist.append(dict(epoch=e, loss=ls / len(tr), acc16=float(a16), acc32=float(a32)))
            print(f"  epoch {e:3d}  loss {ls/len(tr):.3f}  acc@16 {a16*100:.1f}%  acc@32(transfer) {a32*100:.1f}%", flush=True)
    print(f"  trivial baseline: 16x16 {triv(te16)*100:.1f}%   32x32 {triv(te32)*100:.1f}%", flush=True)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore",
                                        "gcrn-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "gcrn_params.npz"), **{k: np.asarray(v) for k, v in params.items()})
    json.dump({"config": vars(a), "history": hist, "trivial16": triv(te16), "trivial32": triv(te32)},
              open(os.path.join(out, "gcrn.json"), "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
