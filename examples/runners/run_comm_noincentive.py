"""PROVE IT: inter-agent (comm) attention with NO connectivity/collision incentive.

Hypothesis: attention is capability, not incentive. Give each agent a full learned
attention over its in-range neighbours, but reward ONLY coverage (w_conn=0, w_coll=0,
no CBF, no overlap). Prediction: it still disperses (connectivity collapses, like the
plain FrontierAttn's 32%) and may even collide — because nothing makes min/max
distance valuable, no matter how well it can perceive its neighbours.

    python examples/run_comm_noincentive.py
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import _frontier_core as fc
import zymera
from marl_frontier_comm import FrontierCommAttnAC

core.ITERS = 120
REWARD = dict(w_cov=1.0, w_conn=0.0, w_coll=0.0, w_overlap=0.0)   # coverage ONLY — no incentive

env = cc.make_env(**REWARD)
best = fc.run(env, FrontierCommAttnAC, "comm-noincentive", seed=1,
              target_cov=None, patience=50, min_iters=80)
vdir = Path(best["dir"])

# eval: coverage / connectivity / collisions
like = FrontierCommAttnAC(env, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

keys = jax.random.split(jax.random.PRNGKey(7), 32)
tr = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(keys)
st = tr["world"]
cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
pos = st.pos
d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)
coll = (((d == 0) & off).sum((-1, -2)) / 2.0).mean(-1)
maxd = jnp.where(off, d, 0).max((-1, -2))[:, -1]

print("\n========= comm-attention, NO incentive · eval (32 eps) =========")
print(f"coverage {float(cov.mean())*100:.1f}%   connected {float(conn.mean())*100:.0f}%   "
      f"max-dist {float(maxd.mean()):.1f}   collisions {float(coll.mean()):.2f}")
print("\nprediction: disperses anyway (connected ≪ 100%) → attention ≠ incentive")
print("reference: FrontierAttn (no comm head, no conn reward) → 97.7% cov / 32% conn")
print(f"artifacts: {vdir}")
