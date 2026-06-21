"""Train FrontierAttn with the SOFT-CBF comm-graph reward — the test of whether
a barrier (λ₂ ≥ ε threshold, not a maximize) buys back the connected swarm while
keeping the architecture's high coverage. The CBF REPLACES the ad-hoc w_conn /
w_coll terms with discrete-time barrier violation penalties.

Early-exit is ON (plateau/divergence; coverage-target disabled since connectivity
is half the objective here).

    python examples/run_cbf.py
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import _frontier_core as fc
import zymera
from marl_frontier_attn import FrontierAttnAC

core.ITERS = 160
# Soft-CBF with α=1 → the residual becomes the STATE barrier relu(ε−λ₂), which
# penalizes the disconnected *state* every step (not just the transition, which
# the policy pays once then explores freely). ε=0.3 keeps a connected-chain margin
# (λ₂≈0.3 ≈ a d≈5-spaced chain). Stronger weights so connectivity competes with
# the frontier attention's coverage drive. CBF replaces the ad-hoc conn/coll terms.
REWARD = dict(w_cov=1.0, w_conn=0.0, w_coll=0.0, w_overlap=0.0,
              w_cbf_conn=15.0, w_cbf_coll=8.0, cbf_alpha=1.0, cbf_eps=0.3)

env = cc.make_env(**REWARD)
best = fc.run(env, FrontierAttnAC, "frontier-cbf", seed=1,
              target_cov=None, patience=70, min_iters=110)         # no coverage-target stop
vdir = Path(best["dir"])

# ---- eval (sample π, ε=0): coverage AND connectivity -----------------------
like = FrontierAttnAC(env, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

keys = jax.random.split(jax.random.PRNGKey(7), 32)
tr = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(keys)
st = tr["world"]
cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
pos = st.pos
d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)
maxd = jnp.where(~jnp.eye(cc.N_AGENTS, dtype=bool), d, 0).max((-1, -2))[:, -1]

print("\n================ Frontier + soft-CBF · eval (32 eps) ================")
print(f"coverage {float(cov.mean())*100:.1f}% ± {float(cov.std())*100:.0f}   "
      f"connected {float(conn.mean())*100:.0f}%   max-dist {float(maxd.mean()):.1f}")
print("\nreference: FrontierAttn (no CBF, w_conn=2) → 97.7% cov / 32% conn")
print("goal:      high coverage AND connected (the stretched chain)")
print(f"artifacts: {vdir}")
