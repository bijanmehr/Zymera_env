"""Audit GIFs — animate a recorded run: certainty field filling (viridis),
walls (dark), agents (colored dots), comm links (white lines = the live comm
graph, so you can SEE connectivity hold/break), and per-step coverage + a
CONNECTED/SPLIT flag. Needs Pillow.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.gif
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import matplotlib.patches as mpatches        # noqa: E402
from matplotlib import animation             # noqa: E402

from .core import is_connected, random_wall   # noqa: E402
from .swarm import run_swarm                   # noqa: E402


def _world_rgb(cnt, wall, vmax):
    """Light world: near-white free, light-green covered (by certainty), black walls."""
    H, W = wall.shape
    img = np.full((H, W, 3), 0.97)                       # near-white base (unvisited free)
    cov = (cnt > 0) & ~wall
    t = np.clip(cnt / vmax, 0, 1)[cov]                   # deeper green with more certainty
    img[cov] = np.stack([1 - 0.5 * t, np.full(t.shape, 0.93), 1 - 0.5 * t], axis=-1)
    img[wall] = 0.0                                      # obstacles black
    return img


def animate_run(result, path, fps=8, title="swarm", sense_r=3):
    frames = result["frames"]
    if not frames:
        raise ValueError("no frames recorded — run_swarm(..., record=True)")
    wall = result["wall"]
    H, W = wall.shape
    vmax = max(1.0, float(frames[-1]["count"].max()))
    fig, ax = plt.subplots(figsize=(6.2, 6.6))

    def draw(t):
        ax.clear()
        fr = frames[t]
        cnt = fr["count"].astype(float)
        ax.imshow(_world_rgb(cnt, wall, vmax), interpolation="nearest")
        # grid lines
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color="#cfcfcf", linewidth=0.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        pos, adj, goals = fr["pos"], fr["adj"], fr.get("goal")
        # comm links (subtle grey)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if adj[i, j]:
                    ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                            color="#9aa0a6", lw=0.9, alpha=0.8, zorder=2)
        for i in range(len(pos)):
            r, c = float(pos[i, 0]), float(pos[i, 1])
            # sense range (Euclidean disk, radius = sense_r)
            ax.add_patch(mpatches.Circle((c, r), sense_r, color="#1f6feb", alpha=0.07, zorder=1))
            ax.add_patch(mpatches.Circle((c, r), sense_r, fill=False, color="#1f6feb",
                                         alpha=0.4, lw=0.8, zorder=1))
            # compass vector: arrow toward current goal
            if goals is not None and goals[i] is not None:
                dr, dc = goals[i][0] - r, goals[i][1] - c
                n = (dr * dr + dc * dc) ** 0.5
                if n > 1e-6:
                    L = 1.9
                    ax.annotate("", xy=(c + dc / n * L, r + dr / n * L), xytext=(c, r),
                                arrowprops=dict(arrowstyle="-|>", color="#d9480f", lw=1.7), zorder=4)
        # agents: frontier = blue, relay (holding/restoring the net) = amber
        relay = fr.get("relay")
        if relay is not None:
            colors = np.where(np.asarray(relay), "#f59f00", "#1f6feb")
        else:
            colors = "#1f6feb"
        ax.scatter(pos[:, 1], pos[:, 0], c=colors, edgecolors="white", s=95, zorder=5, linewidths=1.3)
        cov = (cnt >= 1)[~wall].mean() * 100
        state = "CONNECTED" if is_connected(adj) else "SPLIT"
        nrelay = int(np.asarray(relay).sum()) if relay is not None else 0
        extra = f" · relays {nrelay}" if relay is not None else ""
        ax.set_title(f"{title} · step {t+1}/{len(frames)} · cover {cov:4.1f}% · {state}{extra}", fontsize=10)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 // fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def scale_demo(gate_path=None, seed=2):
    """32x32 audit GIFs: baseline vs gossip+relay (and learned gate if given).
    Relay agents render amber so you can watch the network being actively held.
    Default seed=2 is representative (baseline fragments, relay holds it); headline
    numbers in the report are 5-seed averages, not this single illustrative run."""
    H = W = 32
    wall = random_wall(H, W, 0.05, seed)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore", "gifs"))
    os.makedirs(out, exist_ok=True)
    configs = [
        ("swarm32",        dict(n_agents=10, lambda_conn=1.0),
         "32x32  baseline  (soft λ — dead knob)"),
        ("swarm32_relay",  dict(n_agents=10, lambda_conn=1.0, topo_gossip=True, relay=True, relay_margin=0),
         "32x32  gossip+relay  (active bridge-holding)"),
    ]
    if gate_path and os.path.exists(gate_path):
        import numpy as _np
        configs.append(("swarm32_gate",
                        dict(n_agents=10, lambda_conn=1.0, topo_gossip=True, relay=True,
                             relay_margin=0, gate_theta=_np.load(gate_path)),
                        "32x32  learned relay-gate"))
    for tag, kw, title in configs:
        r = run_swarm(H, W, wall, comm_r=5, sense_r=3, steps=100, seed=seed, record=True, **kw)
        p = os.path.join(out, f"{tag}.gif")
        animate_run(r, p, title=title)
        print(f"{tag:14s} cover {r['coverage']*100:4.1f}%  conn {r['connectivity']*100:3.0f}%  "
              f"largest {r['largest_comp_frac']*100:3.0f}%  -> {p}")


def main():
    H = W = 16
    wall = random_wall(H, W, 0.05, 0)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore", "gifs"))
    os.makedirs(out, exist_ok=True)
    configs = [
        ("swarm_lam1", dict(n_agents=4, lambda_conn=1.0), "swarm  λ=1  (connectivity by choice)"),
        ("swarm_lam0", dict(n_agents=4, lambda_conn=0.0), "swarm  λ=0  (no connectivity pref)"),
        ("single",     dict(n_agents=1, lambda_conn=0.0), "single agent"),
    ]
    for tag, kw, title in configs:
        r = run_swarm(H, W, wall, comm_r=5, sense_r=3, steps=100, seed=0, record=True, **kw)
        p = os.path.join(out, f"{tag}.gif")
        animate_run(r, p, title=title)
        print(f"{tag:11s} cover {r['coverage']*100:4.1f}%  conn {r['connectivity']*100:3.0f}%  -> {p}")
    print(f"\nGIFs saved in {out}")


if __name__ == "__main__":
    main()
