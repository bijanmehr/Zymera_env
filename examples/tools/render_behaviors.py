"""Render a 3-panel comparison of the learned behaviors (final-frame agent config +
comm graph + team coverage) for the report: frontier-attn (disperses), attention-alone
(colliding huddle), CBF+attention (disperses, clean collisions). Saves a PNG and embeds
it (base64) into docs/research-report.html under a new 'Behaviors' section."""
import base64
import io
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import zymera
from marl_frontier_attn import FrontierAttnAC
from marl_frontier_comm import FrontierCommAttnAC

PANELS = [
    ("frontier-attn\n98% cov · 32% conn · disperses",
     "experiments/marl/20260610-230935/frontier-attn", FrontierAttnAC),
    ("attention-alone (no incentive)\n84% · 69% · colliding huddle",
     "experiments/marl/20260611-015906/comm-noincentive", FrontierCommAttnAC),
    ("CBF + attention\n97% · 26% · clean collisions",
     "experiments/marl/20260611-024503/cbf_attention", FrontierCommAttnAC),
]

env = cc.make_env()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
colors = plt.cm.tab10.colors
for ax, (title, d, Model) in zip(axes, PANELS):
    model = eqx.tree_deserialise_leaves(str(Path(d) / "checkpoint.eqx"),
                                        Model(env, jax.random.PRNGKey(0)))
    def policy(o, k):
        return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)
    tr = zymera.rollout(env, policy, cc.MAX_STEPS, jax.random.PRNGKey(7))
    st = jax.tree_util.tree_map(lambda x: x[-1], tr["world"])
    team = np.asarray(st.team_explored)
    pos = np.asarray(st.pos)
    ax.imshow(team, cmap="Blues", vmin=0, vmax=1)
    d_ij = np.max(np.abs(pos[:, None] - pos[None]), -1)
    for i in range(cc.N_AGENTS):
        for j in range(i + 1, cc.N_AGENTS):
            if d_ij[i, j] <= cc.COMM_R:
                ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]], ":", color="gray", lw=1.3, zorder=2)
    for i in range(cc.N_AGENTS):
        ax.scatter(pos[i, 1], pos[i, 0], c=[colors[i]], s=160, edgecolors="black", lw=0.8, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Learned behaviors (final frame · dotted = in-comm-range edges)", fontsize=12)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
b64 = base64.b64encode(buf.getvalue()).decode()

# embed into the report just after the TOC
html = Path("docs/research-report.html").read_text()
block = (f'<section id="behaviors"><h2>0 · Learned behaviors at a glance</h2>'
         f'<img style="max-width:100%" src="data:image/png;base64,{b64}">'
         f'<p class="meta">Coverage is solved everywhere; the difference is the comm graph: '
         f'frontier-attn and CBF+attention disperse (few edges), attention-alone huddles. '
         f'Connectivity remains the open problem.</p></section>\n')
if 'id="behaviors"' not in html:
    html = html.replace("</div>\n", "</div>\n" + block, 1)  # after the TOC div
Path("docs/research-report.html").write_text(html)
print("embedded behavior panel into docs/research-report.html")
