"""Head-to-head: GLOBAL giant-component vs LOCAL (cohesion leash + degree floor)
connectivity reward — same curriculum, same phase-1 warm-start, dev world.

Question: does a purely PARTIAL-INFO connectivity signal (each agent uses only
its own in-range neighbours — no λ₂, no global graph, no gossip) hold the swarm
together as well as the gossip-computable giant-component that scored 87% on 16×16?

Design: ONE shared phase-1 (spread + cover, no collision) → two phase-2 variants
warm-started from the SAME policy, differing ONLY in the connectivity reward, so
the connectivity signal is the only variable.

    python examples/run_local_conn.py
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

# ---- dev world (fast); comm_r=3 so the leash has ROOM (penalize nn ∈ (leash, comm_r]) ----
# NB: grid/n_agents/comm_r are env __init__ DEFAULT args (bound at import), so mutating
# the module constants alone doesn't size the env — we pass DEV explicitly to make_env.
# Mutating the constants here keeps the eval reads (cc.GRID/N_AGENTS/COMM_R/MAX_STEPS) in sync.
cc.GRID = 10
cc.N_AGENTS = 3
cc.MAX_STEPS = 35
cc.COMM_R = 3
cc.SPAWN_RADIUS = 1
DEV = dict(grid_h=10, grid_w=10, n_agents=3, comm_r=3, spawn_radius=1)  # vis_r=0, sense_r=1 keep defaults

import _marl_core as core          # noqa: E402  (after dev-world constants)
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC  # noqa: E402

N = cc.N_AGENTS
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
P1_ITERS, P2_ITERS = 120, 100


def graded_eval(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"),
                                    FrontierCommAttnAC(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, cc.MAX_STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
    return cov, giant, maxd, coll


base = core.new_run_dir()
print(f"LOCAL-vs-GLOBAL connectivity · dev {cc.GRID}×{cc.GRID}, N={N}, comm_r={cc.COMM_R} → {base}\n")

# ---- shared phase 1: spread + cover, NO collision constraint (identical warm-start) ----
core.ITERS = P1_ITERS
env1 = cc.make_env(w_cov=1.0, **SHAPE, **DEV)
b1 = fc.run(env1, FrontierCommAttnAC, "p1-spread", outdir=base, seed=1,
            target_cov=None, patience=999, min_iters=999, mask_collisions=False)
m1 = eqx.tree_deserialise_leaves(str(Path(b1["dir"]) / "checkpoint.eqx"),
                                 FrontierCommAttnAC(env1, jax.random.PRNGKey(0)))
print(f"--- phase1 spread {b1['cov']*100:.0f}% → two phase2 variants from the SAME warm-start ---\n")


def phase2(tag, kw):
    core.ITERS = P2_ITERS
    env2 = cc.make_env(w_cov=1.0, **SHAPE, **kw, **DEV)
    b2 = fc.run(env2, FrontierCommAttnAC, tag, outdir=base, seed=2, init_model=m1,
                target_cov=None, patience=999, min_iters=999, mask_collisions=True)
    return graded_eval(env2, Path(b2["dir"]))


# A · GLOBAL gossip-computable giant-component (the reward that gave 87% on 16×16)
A = phase2("A-global-giant", dict(w_conn=1.5, conn_cap=N - 1))
# B · LOCAL only — cohesion leash (clamped to comm_r) + degree floor, NO giant-component
B = phase2("B-local-leash-degree", dict(w_conn=0.0, w_cohesion=0.5, cohesion_leash=2.0,
                                        w_degree=1.0, degree_floor=1.0))

print("\n========= connectivity-signal head-to-head (eval 32 eps, masked) =========")
print(f"{'mode':<24}{'coverage':>10}{'giant':>9}{'max-dist':>10}{'collisions':>12}")
print(f"{'A · global giant-comp':<24}{A[0]*100:>9.1f}%{A[1]*100:>8.0f}%{A[2]:>10.1f}{A[3]:>12.2f}")
print(f"{'B · local leash+degree':<24}{B[0]*100:>9.1f}%{B[1]*100:>8.0f}%{B[2]:>10.1f}{B[3]:>12.2f}")
print(f"\nphase1 spread {b1['cov']*100:.0f}%   (both phase2 warm-started from it)")
print(f"artifacts: {base}")
