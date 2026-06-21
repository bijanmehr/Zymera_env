"""Animate a trained comm-coverage MARL rollout into an audit GIF, same style as
the swarm_explore GIFs: light world, green = team-covered cells, black = walls,
blue dots = agents, grey lines = live comm graph, title = coverage% + CONNECTED/
SPLIT + #components. For auditing the 32x32 policy.

  PYTHONPATH=examples/lib python examples/tools/marl_gif.py <run_dir> <variant> <out.gif>
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import matplotlib.patches as mpatches        # noqa: E402
from matplotlib import animation             # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc                    # noqa: E402
cc.GRID = 32; cc.N_AGENTS = 10; cc.MAX_STEPS = 100; cc.COMM_R = 5
import zymera                                 # noqa: E402
from marl_independent import IndependentAC    # noqa: E402
from marl_ctde import CTDE_AC                 # noqa: E402
from marl_frontier_attn import FrontierAttnAC  # noqa: E402

MODELS = {"independent": IndependentAC, "CTDE": CTDE_AC, "frontier": FrontierAttnAC}
ENV_KW = dict(grid_h=32, grid_w=32, n_agents=10, comm_r=5, vis_r=3, sense_r=3,
              n_obstacles=30, spawn_radius=2, degree_channels=True)


def _components(d, comm_r):
    adj = (d <= comm_r) & ~np.eye(d.shape[0], dtype=bool)
    n = d.shape[0]; seen = np.zeros(n, bool); comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1; stack = [s]; seen[s] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and not seen[v]:
                    seen[v] = True; stack.append(v)
    return comps


def render(run_dir, variant, out, seed=2):
    env = cc.make_env(**ENV_KW)
    ckpt = Path(run_dir) / "checkpoint.eqx"
    model = eqx.tree_deserialise_leaves(str(ckpt), MODELS[variant](env, jax.random.PRNGKey(0)))

    def policy(o, k):
        return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)
    tr = zymera.rollout(env, policy, cc.MAX_STEPS, jax.random.PRNGKey(seed))
    st = tr["world"]
    team = np.asarray(st.team_explored)          # (T+1, H, W)
    pos = np.asarray(st.pos)                      # (T+1, N, 2)
    wall = np.asarray(st.wall[-1])               # (H, W)
    H, W = wall.shape
    N = pos.shape[1]
    free = int((~wall).sum())
    Tn = team.shape[0]

    def world_rgb(t):
        img = np.full((H, W, 3), 0.97)
        cov = team[t] & ~wall
        img[cov] = np.array([0.55, 0.85, 0.55])
        img[wall] = 0.0
        return img

    fig, ax = plt.subplots(figsize=(6.4, 6.8))

    def draw(t):
        ax.clear()
        ax.imshow(world_rgb(t), interpolation="nearest")
        p = pos[t]
        d = np.max(np.abs(p[:, None] - p[None]), -1)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] <= cc.COMM_R:
                    ax.plot([p[i, 1], p[j, 1]], [p[i, 0], p[j, 0]], color="#9aa0a6", lw=0.9, alpha=0.8, zorder=2)
        for i in range(N):
            ax.add_patch(mpatches.Circle((p[i, 1], p[i, 0]), 3, color="#1f6feb", alpha=0.06, zorder=1))
        ax.scatter(p[:, 1], p[:, 0], c="#1f6feb", edgecolors="white", s=85, zorder=5, linewidths=1.2)
        cov = team[t][~wall].mean() * 100
        nc = _components(d, cc.COMM_R)
        state = "CONNECTED" if nc == 1 else f"SPLIT×{nc}"
        ax.set_title(f"{variant} · 32×32 · step {t+1}/{Tn} · cover {cov:4.1f}% · {state}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)

    anim = animation.FuncAnimation(fig, draw, frames=Tn, interval=125)
    anim.save(out, writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    finalcov = team[-1][~wall].mean() * 100
    print(f"{variant}: final cover {finalcov:.1f}%  -> {out}")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3],
           seed=int(sys.argv[4]) if len(sys.argv) > 4 else 2)
