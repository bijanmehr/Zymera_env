"""Follow-up: LOOSEN the local connectivity signal. The leash=2 in run_local_conn
clumped the swarm (max-dist 2.7, coverage 48% vs global's 55%). Hypothesis: the
DEGREE FLOOR alone (each agent keeps ≥1 in-range neighbour) is the real anti-
fragmentation guard and permits a fully STRETCHED chain, while the leash just
over-tightens. Warm-start from the SAME phase-1 and sweep the looseness knob.

Reference (from run_local_conn): A·global 54.7%/84%   B·leash2+degree 48.2%/97%.

    python examples/run_local_conn2.py <phase1_dir>
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.GRID = 10
cc.N_AGENTS = 3
cc.MAX_STEPS = 35
cc.COMM_R = 3
cc.SPAWN_RADIUS = 1
DEV = dict(grid_h=10, grid_w=10, n_agents=3, comm_r=3, spawn_radius=1)

import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC  # noqa: E402

N = cc.N_AGENTS
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
P2_ITERS = 100

P1_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("experiments/marl/20260611-185029/p1-spread")


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
print(f"LOOSEN local connectivity · dev {cc.GRID}×{cc.GRID}, N={N}, comm_r={cc.COMM_R} → {base}")
print(f"warm-start phase1: {P1_DIR}\n")

env1 = cc.make_env(w_cov=1.0, **SHAPE, **DEV)
m1 = eqx.tree_deserialise_leaves(str(P1_DIR / "checkpoint.eqx"),
                                 FrontierCommAttnAC(env1, jax.random.PRNGKey(0)))


def phase2(tag, kw):
    core.ITERS = P2_ITERS
    env2 = cc.make_env(w_cov=1.0, **SHAPE, **kw, **DEV)
    b2 = fc.run(env2, FrontierCommAttnAC, tag, outdir=base, seed=2, init_model=m1,
                target_cov=None, patience=999, min_iters=999, mask_collisions=True)
    return graded_eval(env2, Path(b2["dir"]))


# C · degree-floor ONLY (no leash) — loosest local guard; permits chain up to 2·comm_r
C = phase2("C-degree-only", dict(w_conn=0.0, w_degree=1.0, degree_floor=1.0))
# D · degree-floor + LIGHT leash at the sensing edge (leash=comm_r-1=2 but weak) — middle ground
D = phase2("D-degree+lightleash", dict(w_conn=0.0, w_degree=1.0,
                                       w_cohesion=0.2, cohesion_leash=2.0))

print("\n========= LOOSENED local connectivity (eval 32 eps, masked) =========")
print(f"{'mode':<26}{'coverage':>10}{'giant':>9}{'max-dist':>10}{'collisions':>12}")
print(f"{'A · global giant (ref)':<26}{'54.7%':>10}{'84%':>9}{'4.5':>10}{'0.00':>12}")
print(f"{'B · leash2+degree (ref)':<26}{'48.2%':>10}{'97%':>9}{'2.7':>10}{'0.00':>12}")
print(f"{'C · degree-floor only':<26}{C[0]*100:>9.1f}%{C[1]*100:>8.0f}%{C[2]:>10.1f}{C[3]:>12.2f}")
print(f"{'D · degree+light leash':<26}{D[0]*100:>9.1f}%{D[1]*100:>8.0f}%{D[2]:>10.1f}{D[3]:>12.2f}")
print(f"\nartifacts: {base}")
