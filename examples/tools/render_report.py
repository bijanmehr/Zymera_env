"""Reusable: HTML report (episode GIF w/ comm-graph overlay + coverage curve + visit
heatmap) for any saved checkpoint, for visual audit. Samples π (eps=0), embeds the
representative episode (coverage nearest the 32-episode mean).

    python examples/render_report.py <checkpoint_dir> <frontier|comm|ctde|independent|joint> [env_kw=val ...]
"""
import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import zymera
from zymera import viz
from marl_frontier_attn import FrontierAttnAC
from marl_frontier_comm import FrontierCommAttnAC
from marl_ctde import CTDE_AC
from marl_independent import IndependentAC
from marl_joint import JointAC

MODELS = {"frontier": FrontierAttnAC, "comm": FrontierCommAttnAC,
          "ctde": CTDE_AC, "independent": IndependentAC, "joint": JointAC}

vdir = Path(sys.argv[1])
Model = MODELS[sys.argv[2]]
env_kw = {}
for a in sys.argv[3:]:                       # optional env kwargs to match training (e.g. hard_collision=True)
    k, v = a.split("=")
    env_kw[k] = (True if v == "True" else False if v == "False"
                 else int(v) if v.lstrip("-").isdigit() else float(v))
env = cc.make_env(**env_kw)

model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), Model(env, jax.random.PRNGKey(0)))
def policy(o, k):
    return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)

trajs = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(
    jax.random.split(jax.random.PRNGKey(7), 32))
st = trajs["world"]
cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
pos = st.pos
d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
giant = float((reach.sum(-1).max(-1) / cc.N_AGENTS).mean())
full = float(jax.vmap(jax.vmap(lambda r: r.all()))(reach).mean())
coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
rep = int(np.argmin(np.abs(np.asarray(cov) - float(cov.mean()))))
ep = jax.tree_util.tree_map(lambda x: x[rep], st)
n = int(ep.pos.shape[0])

def snap(t):
    s = jax.tree_util.tree_map(lambda x: x[t], ep)
    return types.SimpleNamespace(visited=np.asarray(s.team_explored),
                                 explored=np.asarray(s.explored_by).sum(0),
                                 wall=np.asarray(s.wall),
                                 body=types.SimpleNamespace(position=np.asarray(s.pos)))

out = vdir / "report.html"
viz.make_report({"world": [snap(t) for t in range(n)]}, str(out),
                title=(f"{vdir.name} · {float(cov.mean())*100:.0f}% cov · "
                       f"giant-comp {giant*100:.0f}% · full-conn {full*100:.0f}% · coll {coll:.2f}"),
                fps=8, comm_radius=cc.COMM_R)
print(f"{vdir.name}: cov {float(cov.mean())*100:.1f}% giant {giant*100:.0f}% full {full*100:.0f}% coll {coll:.2f}")
print(f"wrote {out}")
