"""
Isometric (pseudo-3D) rendering — the "iso view", borrowed from
RedWithinBlue's mechanism: a 3D bar chart (``Axes3D.bar3d``) viewed from a
fixed isometric camera (``view_init(elev, azim)``).

The grid becomes 3D bars: visited floor tiles, raised wall blocks, agent
pillars, and — when ``seen_by`` is populated — fog columns over cells no
agent has ever seen.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from .render import _agent_color, _agg_figure, _fig_to_rgba, iter_worlds, save_gif

# camera + heights, tuned (as in RWB) so walls/fog don't occlude the interior
_ELEV, _AZIM = 32, -55
_FLOOR_H, _WALL_H, _AGENT_H, _FOG_H = 0.06, 0.5, 0.8, 0.22
_VISITED_RGBA = (0.26, 0.45, 0.71, 1.0)
_FREE_RGBA = (0.93, 0.93, 0.95, 1.0)
_WALL_RGBA = (0.18, 0.18, 0.20, 1.0)
_FOG_RGBA = (0.10, 0.10, 0.13, 0.55)


def draw_iso(ax, world, title=None, fog=True):
    """Draw one ``World`` snapshot as an isometric 3D scene on a 3D axis."""
    ax.clear()
    visited = np.asarray(world.visited)            # (H, W) bool
    wall = np.asarray(world.wall)                  # (H, W) bool
    pos = np.asarray(world.body.position)          # (N, 2)
    seen = np.asarray(world.seen_by)               # (N, H, W) bool
    H, W = visited.shape

    # floor tiles (x = col, y = row) on every non-wall cell
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs, ys = xs.ravel(), ys.ravel()
    free = (~wall).ravel()
    colors = np.tile(np.array(_FREE_RGBA), (xs.size, 1))
    colors[visited.ravel()] = _VISITED_RGBA
    ax.bar3d(xs[free], ys[free], np.zeros(int(free.sum())), 1, 1, _FLOOR_H,
             color=colors[free], shade=True, edgecolor=(1, 1, 1, 0.15), linewidth=0.2)

    # walls — dark raised blocks
    wr, wc = np.where(wall)
    if wr.size:
        ax.bar3d(wc, wr, np.zeros(wr.size), 1, 1, _WALL_H,
                 color=_WALL_RGBA, shade=True, edgecolor=(0, 0, 0, 0.4), linewidth=0.3)

    # fog — cells no agent has ever seen (only if seen_by is populated)
    if fog and seen.size and seen.any():
        unseen = (~seen.any(axis=0)) & (~wall)
        fr, fc = np.where(unseen)
        if fr.size:
            ax.bar3d(fc, fr, np.zeros(fr.size), 1, 1, _FOG_H, color=_FOG_RGBA, shade=False)

    # agents — thin pillars
    for i in range(pos.shape[0]):
        r, c = int(pos[i, 0]), int(pos[i, 1])
        ax.bar3d(c + 0.3, r + 0.3, 0, 0.4, 0.4, _AGENT_H,
                 color=_agent_color(i), shade=True, edgecolor="black", linewidth=0.4)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                              # invert so row 0 sits at the back
    ax.set_zlim(0, 1.2)
    ax.view_init(elev=_ELEV, azim=_AZIM)
    ax.set_box_aspect((W, H, max(H, W) * 0.35))
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)


def iso_gif(trajectory: dict, output_path: str, fps: int = 4) -> None:
    """Render a trajectory as an isometric animated gif."""
    n_frames, get_world = iter_worlds(trajectory)
    fig = _agg_figure((4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for t in range(n_frames):
        draw_iso(ax, get_world(t), title=f"step {t}")
        frames.append(_fig_to_rgba(fig))
    save_gif(frames, output_path, fps)
