"""Rendering: territory map (who first operated each cell), certainty-field
heatmap, and the coverage↔connectivity Pareto from the tension sweep. Robust /
headless (Agg); never required for the numbers."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import colors            # noqa: E402


def territory_map(result, path, title=""):
    owner, wall = result["owner"], result["wall"]
    H, W = owner.shape
    n = int(owner.max()) + 1 if owner.max() >= 0 else 1
    base = plt.cm.tab10(np.linspace(0, 1, 10))
    palette = [(0.12, 0.12, 0.12, 1), (0.92, 0.92, 0.92, 1)] + [tuple(base[i % 10]) for i in range(n)]
    cmap = colors.ListedColormap(palette)
    disp = np.ones((H, W))
    disp[wall] = 0
    own = owner >= 0
    disp[own] = owner[own] + 2
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=len(palette) - 1, interpolation="nearest")
    if result.get("frames"):
        pos = result["frames"][-1]["pos"]
        ax.scatter(pos[:, 1], pos[:, 0], c="white", edgecolors="black", s=70, zorder=3, lw=1.2)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def certainty_heatmap(result, path, title=""):
    cnt, wall = result["true_count"].astype(float), result["wall"]
    disp = np.ma.masked_where(wall, cnt)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(disp, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="operation count (certainty)")
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def pareto(sweep, path):
    """sweep: {lambda: row} where row[metric] = (mean, std)."""
    lams = sorted(sweep.keys(), key=float)
    cov = [sweep[l]["coverage"][0] * 100 for l in lams]
    conn = [sweep[l]["connectivity"][0] * 100 for l in lams]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(conn, cov, "o-", lw=2)
    for l, x, y in zip(lams, conn, cov):
        ax.annotate(f"λ={l}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("connectivity (% of steps graph whole)")
    ax.set_ylabel("coverage (%)")
    ax.set_title("Coverage ↔ connectivity Pareto (tension knob)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
