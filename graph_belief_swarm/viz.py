"""Rendering: territory map (who-covered-what) + coverage-over-time curves."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import colors            # noqa: E402


def render_territories(result, path, title=""):
    """Color each cell by the agent that first covered it -> see division of labor."""
    owner, wall = result["owner"], result["wall"]
    H, W = owner.shape
    n_ag = int(owner.max()) + 1 if owner.max() >= 0 else 1

    base = plt.cm.tab10(np.linspace(0, 1, 10))
    palette = [(0.12, 0.12, 0.12, 1), (0.92, 0.92, 0.92, 1)] + [tuple(base[i % 10]) for i in range(n_ag)]
    cmap = colors.ListedColormap(palette)

    disp = np.ones((H, W))                # 1 = unvisited (light grey)
    disp[wall] = 0                        # 0 = wall (dark)
    own = owner >= 0
    disp[own] = owner[own] + 2            # 2.. = agent territories

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=len(palette) - 1, interpolation="nearest")
    if result.get("frames"):
        pos = result["frames"][-1]["pos"]
        ax.scatter(pos[:, 1], pos[:, 0], c="white", edgecolors="black",
                   s=70, zorder=3, linewidths=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def render_curves(results, labels, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, l in zip(results, labels):
        ax.plot(np.asarray(r["cov_curve"]) * 100, label=l, linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("team coverage (%)")
    ax.set_title("Coverage over time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
