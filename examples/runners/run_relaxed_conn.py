"""Push connectivity under the RELAXED goal (graph need not be fully connected).
frontier-attn architecture (solves coverage) + the already-GRADED connectivity
reward (reach-1)/(N-1) turned up (w_conn 5, 10 vs the original 2). Measured by
GIANT-COMPONENT fraction (the right yardstick), not binary full-connected.

Target: coverage >90% AND giant-component high.

    python examples/run_relaxed_conn.py
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
N = cc.N_AGENTS
CONFIGS = [("wconn5", 5.0), ("wconn10", 10.0)]


def graded_eval(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"),
                                    FrontierAttnAC(env, jax.random.PRNGKey(0)))
    pol = lambda o, k: jax.random.categorical(k, m.logits(o), -1).astype(jnp.int32)
    tr = jax.vmap(lambda k: zymera.rollout(env, pol, cc.MAX_STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)   # (32,T+1,N,N)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)                          # (32,T+1,N,N)
    full = float(jax.vmap(jax.vmap(lambda r: r.all()))(reach).mean())
    giant = float((reach.sum(-1).max(-1) / N).mean())
    off = ~jnp.eye(N, dtype=bool)
    pairs = float(((reach & off).sum((-1, -2)) / (N * (N - 1))).mean())
    return cov, full, giant, pairs


base = core.new_run_dir()
print(f"relaxed-connectivity sweep (frontier-attn) → {base}\n")
rows = []
for name, wc in CONFIGS:
    env = cc.make_env(w_cov=1.0, w_conn=wc, w_coll=4.0, w_overlap=0.5)
    best = fc.run(env, FrontierAttnAC, name, outdir=base,
                  target_cov=None, patience=70, min_iters=110)
    rows.append((name, wc, graded_eval(env, Path(best["dir"]))))

print("\n========= relaxed-connectivity · eval (32 eps) =========")
print(f"{'config':10s} {'w_conn':>6} {'coverage':>9} {'full-conn':>10} {'giant-comp':>11} {'pairs':>7}")
print(f"{'(ref) frontier-attn':10s} {'2':>6} {'97.7%':>9} {'32%':>10} {'71%':>11} {'53%':>7}")
for name, wc, (c, f, g, p) in rows:
    print(f"{name:10s} {wc:6.0f} {c*100:8.1f}% {f*100:9.0f}% {g*100:10.0f}% {p*100:6.0f}%")
print(f"artifacts: {base}")
