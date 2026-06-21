"""Experiment jobs for the overnight driver.

Each job:  def job(ctx, **knobs)  — does its work, records numbers via ctx.metric().
Two scales per job in the MANIFEST: 'smoke' (tiny, CPU, validates) and 'gpu' (full).

Status: ④ belief-multiscale and ③ pareto-wall are real; ② ⑤ ⑥ are runnable
scaffolds (they log their plan + status) pending the MARL/UED implementations.
"""
from __future__ import annotations
import collections
import json
import os

import numpy as np


# ============================================================ ④ multi-scale belief
def belief_multiscale(ctx, scales, epochs, n_train, n_test, batch, steps):
    """Train ONE size-invariant GCRN on a distribution of (grid, N), measure transfer at each scale."""
    import jax, jax.numpy as jnp, optax
    from swarm_explore import gcrn as G

    train, test = {}, {}                       # bucketed by N (shapes differ across scales)
    for gi, (grid, n) in enumerate(scales):
        tr = [G.collect_traj(grid, n, steps, 1000 * gi + s) for s in range(n_train)]
        te = [G.collect_traj(grid, n, steps, 9000 + 1000 * gi + s) for s in range(n_test)]
        train[n] = tuple(jnp.asarray(np.stack([t[k] for t in tr])) for k in range(3))
        test[n] = tuple(jnp.asarray(np.stack([t[k] for t in te])) for k in range(3))
        ctx.log(f"collected {grid}x{grid}/N={n}: {n_train} train + {n_test} test trajectories")

    p = G.init_params(jax.random.PRNGKey(0))
    opt = optax.adamw(5e-3, weight_decay=0.01); st = opt.init(p)

    def bloss(p, I, A, Tr):
        return jnp.mean(jax.vmap(G.loss_fn, in_axes=(None, 0, 0, 0))(p, I, A, Tr))

    @jax.jit
    def step(p, st, I, A, Tr):
        l, g = jax.value_and_grad(bloss)(p, I, A, Tr)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    def evaluate():
        return {f"acc@{n}": round(float(np.mean([G.accuracy(p, test[n][0][j], test[n][1][j], test[n][2][j])
                                                 for j in range(test[n][0].shape[0])])), 3) for n in test}

    rng = np.random.default_rng(0)
    for ep in range(epochs):
        for n in train:                         # one batch per scale per epoch (domain randomization)
            I, A, Tr = train[n]; B = I.shape[0]
            idx = rng.choice(B, size=min(batch, B), replace=False)
            p, st, _ = step(p, st, I[idx], A[idx], Tr[idx])
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            ctx.log(f"epoch {ep}: " + " ".join(f"{k}={v}" for k, v in evaluate().items()))

    for k, v in evaluate().items():
        ctx.metric(**{k: v})
    np.savez(os.path.join(ctx.dir, "belief_params.npz"), **{k: np.asarray(v) for k, v in p.items()})


# ============================================================ ③ coverage/connectivity Pareto + wall boundary
def pareto_wall(ctx, lams, scales, seeds, steps):
    """Sweep λ_conn × scale × seeds through the swarm engine → the Pareto + where connectivity stops being feasible."""
    from swarm_explore import arch3
    from swarm_explore.core import random_wall
    theta = arch3.init_theta(0).copy(); theta[-1] = -10.0    # compass-only baseline (P(relay)~0); relay/K lever is a later deepening
    agg = collections.defaultdict(list)
    total = len(lams) * len(scales) * seeds; done = 0
    for lam in lams:
        for (grid, n) in scales:
            for s in range(seeds):
                wall = random_wall(grid, grid, 0.05, s)
                m = arch3.run_arch(grid, grid, wall, n, 5, 3, steps, s, theta,
                                   K=max(1, n - 1 - 2), lam_frontier=lam)
                agg[(lam, grid, n)].append((m["coverage"], m["connectivity"], m["largest_comp_frac"]))
                done += 1
                if done % max(1, total // 8) == 0:
                    ctx.log(f"  {done}/{total} configs")
    out = []
    for (lam, grid, n), v in sorted(agg.items()):
        cov, conn, coh = (float(np.mean([x[i] for x in v])) for i in range(3))
        out.append(dict(lam=lam, grid=grid, n=n, cov=round(cov, 3), conn=round(conn, 3), coh=round(coh, 3)))
        ctx.log(f"  lam={lam} {grid}x{grid}/N={n}: cov {cov*100:.0f}% conn {conn*100:.0f}% coh {coh*100:.0f}%")
    json.dump(out, open(os.path.join(ctx.dir, "pareto.json"), "w"), indent=2)
    ctx.metric(configs=len(out), best_conn=round(max((o["conn"] for o in out), default=0.0), 3))


# ============================================================ ② MARL at scale  (scaffold)
def marl_scale(ctx, **k):
    ctx.log("SCAFFOLD ② MARL-32 — drive the examples/lib PPO stack (IPPO→CTDE→MAPPO) at 32x32, batched envs + seeds.")
    ctx.log("Plan: import examples.lib._marl_core; set GRID=32, N=10, NUM_ENVS=4096; warm-start ladder; "
            "record coverage / connectivity / fragmentation curves per seed. Verifies the geometric wall under full MARL.")
    ctx.metric(status="scaffold")


# ============================================================ ⑤ method tournament  (scaffold)
def method_tournament(ctx, **k):
    ctx.log("SCAFFOLD ⑤ tournament — compare the existing variants (independent / ctde / joint / frontier-attn) head-to-head,")
    ctx.log("then add QMIX / GNN / TarMAC. Standardise env+reward+eval; run method × seed × scale; emit a results table.")
    ctx.metric(status="scaffold")


# ============================================================ ⑥ UED curriculum  (scaffold)
def ued_curriculum(ctx, **k):
    ctx.log("SCAFFOLD ⑥ UED — PAIRED/ACCEL co-evolving connectivity-stressing maps (wall_density, comm_radius)")
    ctx.log("against the policy; outputs a robust policy + the discovered hard maps (the failure boundary).")
    ctx.metric(status="scaffold")


MANIFEST = [
    {"name": "belief-multiscale", "fn": belief_multiscale,
     "smoke": dict(scales=[(16, 4), (20, 5)], epochs=3, n_train=4, n_test=2, batch=4, steps=40),
     "gpu":   dict(scales=[(16, 4), (24, 6), (32, 10)], epochs=40, n_train=24, n_test=8, batch=24, steps=70)},
    {"name": "pareto-wall", "fn": pareto_wall,
     "smoke": dict(lams=[0.0, 1.0], scales=[(12, 3)], seeds=2, steps=40),
     "gpu":   dict(lams=[0.0, 0.5, 1.0, 2.0, 4.0], scales=[(16, 4), (24, 6), (32, 10)], seeds=5, steps=100)},
    {"name": "marl-scale", "fn": marl_scale, "smoke": {}, "gpu": {}},
    {"name": "method-tournament", "fn": method_tournament, "smoke": {}, "gpu": {}},
    {"name": "ued-curriculum", "fn": ued_curriculum, "smoke": {}, "gpu": {}},
]
