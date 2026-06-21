"""Train the FrontierAttnAC variant (frontier cross-attention + return-norm) on
the best CTDE reward (ov0.5) and report: best-train coverage, a multi-episode
eval at the training size, and a ZERO-SHOT generalization probe on larger grids
(20×20, 24×24) the size-agnostic actor never trained on. Compares to CTDE 91.4%.

    python examples/run_frontier.py
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

core.ITERS = 180                         # match the CTDE ov0.5 run
REWARD = dict(w_overlap=0.5)             # the best CTDE reward shaping

# ---- train -----------------------------------------------------------------
env = cc.make_env(**REWARD)
best = fc.run(env, FrontierAttnAC, "frontier-attn", seed=1)
vdir = Path(best["dir"])

# ---- reload best checkpoint, eval by SAMPLING from π (ε=0) ------------------
like = FrontierAttnAC(env, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

def eval_grid(e, steps, n=32, base=7):
    keys = jax.random.split(jax.random.PRNGKey(base), n)
    tr = jax.vmap(lambda k: zymera.rollout(e, policy, steps, k))(keys)
    st = tr["world"]
    free = jnp.maximum((~st.wall[:, -1].reshape(n, -1)).sum(-1), 1)
    cov = st.team_explored[:, -1].reshape(n, -1).sum(-1) / free
    pos = st.pos
    d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
    conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)
    return float(cov.mean()), float(cov.std()), float(conn.mean())

print("\n================ FrontierAttn eval (sample π, ε=0) ================")
c, s, k = eval_grid(cc.make_env(**REWARD), cc.MAX_STEPS)
print(f"16×16 (train size, {cc.MAX_STEPS} steps): coverage {c*100:.1f}% ± {s*100:.0f}  connected {k*100:.0f}%")

# zero-shot to larger grids; scale the step budget by area so it's a fair coverage test
for G in (20, 24):
    steps = int(round(cc.MAX_STEPS * (G * G) / (cc.GRID ** 2)))
    c, s, k = eval_grid(cc.make_env(grid_h=G, grid_w=G, **REWARD), steps)
    print(f"{G}×{G} (ZERO-SHOT, {steps} steps): coverage {c*100:.1f}% ± {s*100:.0f}  connected {k*100:.0f}%")

print(f"\nbest-train coverage: {best['cov']*100:.1f}%   (CTDE ov0.5 baseline: 91.4%)")
print(f"artifacts: {vdir}")
