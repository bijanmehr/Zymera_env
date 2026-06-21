"""Reach >90% coverage on 32x32 with minimal comm-graph disruption using the
FrontierAttnAC model (frontier cross-attention, size-agnostic) + the degree signal.

The plain CNN IPPO clumps at 32x32 (redundancy 4x, coverage stalls). FrontierAttn
spreads (frontier-seeking attention) and is documented to generalize 16->24 at ~95%.
Strategy: train fast at 16x16 with the degree channels + connectivity reward, then
ZERO-SHOT transfer to 32x32; fine-tune at 32x32 if coverage < 90%.

Phases:
  python run_marl_frontier32.py train16
  python run_marl_frontier32.py eval32  <checkpoint.eqx>
  python run_marl_frontier32.py ft32    <checkpoint.eqx>
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

PHASE = sys.argv[1] if len(sys.argv) > 1 else "train16"
CKPT = sys.argv[2] if len(sys.argv) > 2 else ""

# grid knobs per phase, BEFORE importing the cores (they read T, conn constants).
# Train at 16x16 with N=4 -> DENSITY-MATCHED to 32x32/10 (10/1024 ~ 4/256), so
# spreading is REQUIRED to cover and coverage-vs-connectivity is a real tension.
# FrontierAttn's actor attends over CELL tokens (not agents) -> N-agnostic, so a
# policy trained at N=4 deploys at N=10.
if PHASE == "train16":
    cc.GRID, cc.N_AGENTS, cc.MAX_STEPS, cc.COMM_R = 16, 4, 70, 5
else:                                   # eval / fine-tune at 32x32 / 100 steps
    cc.GRID, cc.N_AGENTS, cc.MAX_STEPS, cc.COMM_R = 32, 10, 100, 5

import _marl_core as core               # noqa: E402
import _frontier_core as fc             # noqa: E402
import zymera                           # noqa: E402
from marl_frontier_attn import FrontierAttnAC   # noqa: E402

# PHASE 1 reward = the PROVEN coverage recipe (w_overlap=0.5 only) that took
# FrontierAttn to ~98% at 16x16 and generalized to larger grids. We keep the degree
# channels ON (harmless extra obs input; keeps obs_channels=7) so the connectivity
# fine-tune (ft32, which adds w_conn/w_degree) can warm-start without an arch change.
# Connectivity reward was capping the spread, so it is OFF here; coverage comes first.
REWARD = dict(vis_r=3, sense_r=3, spawn_radius=2,
              w_overlap=0.5, degree_channels=True)


def env_for(grid, n_obs, n_agents, **extra):
    return cc.make_env(grid_h=grid, grid_w=grid, n_agents=n_agents, comm_r=5,
                       n_obstacles=n_obs, conn_cap=n_agents - 1, **REWARD, **extra)


def eval_grid(env, model, steps, n=16, base=7):
    def policy(o, k):
        return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(base), n)
    tr = jax.vmap(lambda k: zymera.rollout(env, policy, steps, k))(keys)
    st = tr["world"]
    free = jnp.maximum((~st.wall[:, -1].reshape(n, -1)).sum(-1), 1)
    cov = st.team_explored[:, -1].reshape(n, -1).sum(-1) / free
    pos = st.pos
    d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
    conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= 5)))(d).mean(-1)
    return float(cov.mean()), float(cov.std()), float(conn.mean())


def main():
    if PHASE == "train16":
        core.ITERS = 200
        env = env_for(16, 8, 4)
        print(f"TRAIN FrontierAttn @16x16/10  obs_channels={env.obs_channels}", flush=True)
        best = fc.run(env, FrontierAttnAC, "frontier-deg16", seed=1,
                      target_cov=0.97, patience=80, min_iters=120)
        print(f"\ncheckpoint: {best['dir']}/checkpoint.eqx")
        print(f"train16 best coverage {best['cov']*100:.1f}%  connected {best['conn']*100:.0f}%")

    elif PHASE == "eval32":
        env = env_for(32, 30, 10)
        model = eqx.tree_deserialise_leaves(CKPT, FrontierAttnAC(env, jax.random.PRNGKey(0)))
        c, s, k = eval_grid(env, model, 100)
        print(f"ZERO-SHOT @32x32/100steps: coverage {c*100:.1f}% ± {s*100:.0f}  connected {k*100:.0f}%")

    elif PHASE == "ft32":
        core.ITERS = 60
        # add connectivity reward now (coverage already learned) to minimise graph disruption
        env = env_for(32, 30, 10, w_conn=1.0, w_degree=0.5)
        like = FrontierAttnAC(env, jax.random.PRNGKey(0))
        init = eqx.tree_deserialise_leaves(CKPT, like)
        print(f"FINE-TUNE @32x32 from {CKPT}  obs_channels={env.obs_channels}", flush=True)
        best = fc.run(env, FrontierAttnAC, "frontier-deg32-ft", seed=1,
                      target_cov=0.97, patience=40, min_iters=40, init_model=init)
        print(f"\ncheckpoint: {best['dir']}/checkpoint.eqx")
        print(f"ft32 best coverage {best['cov']*100:.1f}%  connected {best['conn']*100:.0f}%")


if __name__ == "__main__":
    main()
