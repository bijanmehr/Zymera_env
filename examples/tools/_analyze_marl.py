"""Train the best (joint) policy, roll out one greedy episode, and analyze the
learned spatial strategy: trajectories, per-agent territory, spread dynamics."""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
from marl_joint import JointAC

H = W = cc.GRID
N = cc.N_AGENTS
T = cc.MAX_STEPS

# ---- train the joint policy, keep the best ---------------------------------
env = cc.make_env()
model = JointAC(env, jax.random.PRNGKey(0))
opt_state, train_iter = core.make_train(env, model)
key = jax.random.PRNGKey(1)
best_cov, best = -1.0, model
for it in range(120):
    key, sk = jax.random.split(key)
    model, opt_state, loss, m = train_iter(model, opt_state, sk)
    if float(m[0]) > best_cov:
        best_cov, best = float(m[0]), model
print(f"trained joint · best train coverage {best_cov*100:.1f}%")
model = best

# ---- greedy rollout (argmax) -----------------------------------------------
key = jax.random.PRNGKey(7)
_, state = env.reset(key)
states = [state]
for t in range(T):
    a = jnp.argmax(model.logits(env._obs(state)), axis=-1).astype(jnp.int32)
    _, state, _, _, _ = env.step(state, a, key)
    states.append(state)

pos = np.array([np.asarray(s.pos) for s in states])            # (T+1,N,2)
explored = np.asarray(states[-1].explored_by)                  # (N,H,W)
team = np.asarray(states[-1].team_explored)                    # (H,W)
cov_t = np.array([np.asarray(s.team_explored).sum() / (H * W) for s in states])


def meandist(p):
    d = np.max(np.abs(p[:, None] - p[None, :]), -1)
    return d[np.triu_indices(N, 1)].mean()


spread_t = np.array([meandist(p) for p in pos])
per_agent = explored.reshape(N, -1).sum(-1)
overlap = int((explored.sum(0) >= 2).sum())
union = int(team.sum())

# ---- quantitative analysis -------------------------------------------------
print(f"\nfinal coverage           {cov_t[-1]*100:.1f}%   ({union}/{H*W} cells)")
print(f"coverage @ step 10/25/50 {cov_t[10]*100:.0f}% / {cov_t[25]*100:.0f}% / {cov_t[50]*100:.0f}%")
print(f"per-agent cells explored {per_agent.tolist()}")
print(f"cells seen by >=2 agents {overlap}  ({100*overlap/max(union,1):.0f}% of covered → redundancy {explored.reshape(N,-1).sum()/max(union,1):.2f}×)")
print(f"mean pairwise dist  start {spread_t[0]:.1f} → mid {spread_t[T//2]:.1f} → end {spread_t[-1]:.1f}  (formation band is [1,4])")
print(f"start positions {pos[0].tolist()}")
print(f"end positions   {pos[-1].tolist()}")

# ---- figure ----------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
colors = plt.cm.tab10.colors

# 1. trajectories
ax[0].imshow(team, cmap="Greys", alpha=0.25, vmin=0, vmax=1)
for i in range(N):
    ax[0].plot(pos[:, i, 1], pos[:, i, 0], "-", color=colors[i], lw=1.5, alpha=0.8)
    ax[0].plot(pos[0, i, 1], pos[0, i, 0], "o", mfc="none", mec=colors[i], ms=10)
    ax[0].plot(pos[-1, i, 1], pos[-1, i, 0], "o", color=colors[i], ms=9)
ax[0].set_title("agent trajectories (○ start, ● end)")

# 2. team coverage
ax[1].imshow(team, cmap="Blues", vmin=0, vmax=1)
ax[1].set_title(f"team coverage  {cov_t[-1]*100:.0f}%")

# 3. per-agent territory (cell → first agent index, by argmax of explored_by)
label = np.where(team, np.argmax(explored, 0), -1)
terr = np.full((H, W, 3), 1.0)
for i in range(N):
    terr[label == i] = colors[i][:3]
ax[2].imshow(terr)
ax[2].set_title("per-agent territory")

# 4. coverage + spread over time
ax2 = ax[3]
ax2.plot(cov_t * 100, color="#2c4f8a", label="coverage %")
ax2.set_ylabel("coverage %"); ax2.set_ylim(0, 100); ax2.set_xlabel("step")
axb = ax2.twinx()
axb.plot(spread_t, color="#c0392b", label="mean pairwise dist")
axb.axhspan(1, 4, color="green", alpha=0.08)
axb.set_ylabel("mean pairwise dist (band 1–4 shaded)")
ax2.set_title("coverage & spread over time")

for a in ax[:3]:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("/tmp/marl_analysis.png", dpi=110, bbox_inches="tight")
print("\nwrote /tmp/marl_analysis.png")
