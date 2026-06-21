"""
2D rendering — flat top-down draw helpers + trajectory → gif.

This module holds the shared drawing primitives (used by the gif exporter,
the isometric renderer, the HTML report, and the live keyboard viewer) plus
the gif-assembly mechanism borrowed from RedWithinBlue: render each frame to
an RGBA array on an Agg canvas, collect them, and write a gif with Pillow.

matplotlib + pillow are imported lazily / locally so importing the headless
core (``import zymera``) never pulls them in — only ``import zymera.viz``
does.
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np


# ---- agent colours ---------------------------------------------------------


def _agent_color(i: int) -> Any:
    """tab10 colour for agent ``i``; cycles after 10."""
    import matplotlib.cm as cm
    return cm.tab10.colors[i % 10]


# ---- 2D draw helpers (shared with the live viewer) -------------------------


def draw_frame(ax: Any, world: Any, title: Optional[str] = None) -> None:
    """Base scene: visited heatmap + one scatter point per agent."""
    ax.clear()
    visited = np.asarray(world.visited)
    positions = np.asarray(world.body.position)
    n_agents = positions.shape[0]

    ax.imshow(visited, cmap="Blues", vmin=0, vmax=1)
    for i in range(n_agents):
        ax.scatter(
            positions[i, 1], positions[i, 0],
            c=[_agent_color(i)], s=120, marker="o",
            edgecolors="black", linewidths=0.8, zorder=3,
        )
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_walls(ax: Any, world: Any) -> None:
    """Dark squares on wall cells."""
    wall = np.asarray(world.wall)
    rows, cols = np.where(wall)
    if rows.size:
        ax.scatter(cols, rows, c="dimgray", s=300, marker="s",
                   edgecolors="black", linewidths=0.6, zorder=2)


def draw_grid_lines(ax: Any, h: int, w: int) -> None:
    """Toggleable cell-edge grid lines."""
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)


def draw_comm_graph(ax: Any, world: Any, adj: Any) -> None:
    """Dotted lines between agents within comm range."""
    adj_np = np.asarray(adj)
    positions = np.asarray(world.body.position)
    n_agents = positions.shape[0]
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if bool(adj_np[i, j]):
                ax.plot(
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 0], positions[j, 0]],
                    linestyle=":", color="gray", linewidth=1.2,
                    alpha=0.7, zorder=2,
                )


def compute_fog_mask(world: Any, fog_mode: str, fog_radius: int,
                     selected_idx: int = 0) -> np.ndarray:
    """Return ``(H, W) bool`` — True where the cell is fogged (hidden)."""
    H, W = world.visited.shape
    if fog_mode == "off":
        return np.zeros((H, W), dtype=bool)

    positions = np.asarray(world.body.position)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    if fog_mode == "self":
        ar, ac = positions[selected_idx]
        dist = np.maximum(np.abs(rows - ar), np.abs(cols - ac))
        visible = dist <= fog_radius
    elif fog_mode == "all":
        visible = np.zeros((H, W), dtype=bool)
        for ar, ac in positions:
            dist = np.maximum(np.abs(rows - ar), np.abs(cols - ac))
            visible = visible | (dist <= fog_radius)
    else:
        return np.zeros((H, W), dtype=bool)
    return ~visible


def draw_fog(ax: Any, fog_mask: np.ndarray) -> None:
    """Dark RGBA overlay where ``fog_mask`` is True."""
    H, W = fog_mask.shape
    rgba = np.zeros((H, W, 4))
    rgba[..., 3] = fog_mask.astype(np.float32) * 0.55
    ax.imshow(rgba, zorder=1)


def draw_selection_highlight(ax: Any, world: Any, selected_idx: int) -> None:
    """Yellow ring around the currently-controlled agent."""
    positions = np.asarray(world.body.position)
    ax.scatter(
        positions[selected_idx, 1], positions[selected_idx, 0],
        s=320, facecolors="none", edgecolors="yellow",
        linewidths=2.5, zorder=4,
    )


# ---- gif machinery (shared by render_gif / iso_gif / report) ---------------


def _agg_figure(figsize: Tuple[float, float]) -> Any:
    """A standalone Agg-backed Figure — backend-independent, never touches
    the global pyplot state, always supports ``buffer_rgba``."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _fig_to_rgba(fig: Any) -> np.ndarray:
    """Rasterise an Agg figure to an ``(H, W, 4)`` uint8 RGBA array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def save_gif(frames: list, output_path: str, fps: int = 4) -> None:
    """Write a list of RGBA uint8 frames to an animated gif (Pillow)."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                 duration=max(1, 1000 // fps), loop=0)


def iter_worlds(trajectory: dict) -> Tuple[int, Callable[[int], Any]]:
    """``(n_frames, get_world)`` for a trajectory whose ``"world"`` is either
    a Python list of World snapshots or a stacked pytree (leading axis)."""
    worlds = trajectory["world"]
    if isinstance(worlds, list):
        return len(worlds), lambda t: worlds[t]
    import jax
    n = worlds.body.position.shape[0]
    return n, lambda t: jax.tree_util.tree_map(lambda x: x[t], worlds)


def render_gif(trajectory: dict, output_path: str, fps: int = 4) -> None:
    """Render a trajectory's visited mask + agent positions as a gif (flat)."""
    n_frames, get_world = iter_worlds(trajectory)
    fig = _agg_figure((4, 4))
    ax = fig.add_subplot(111)
    frames = [(_draw_and_grab(ax, fig, get_world(t), t)) for t in range(n_frames)]
    save_gif(frames, output_path, fps)


def _draw_and_grab(ax, fig, world, t):
    draw_frame(ax, world, title=f"step {t}")
    return _fig_to_rgba(fig)


def render_comm_gif(trajectory: dict, output_path: str, fps: int = 4,
                    comm_radius: Optional[int] = None) -> None:
    """Flat gif with dotted comm-graph edges between agents within Chebyshev
    ``comm_radius`` — for range-limited-communication / swarm tasks, so a
    stretched relay chain is visible as a connected path, not just dots."""
    n_frames, get_world = iter_worlds(trajectory)
    fig = _agg_figure((4, 4))
    ax = fig.add_subplot(111)
    frames = []
    for t in range(n_frames):
        world = get_world(t)
        draw_frame(ax, world, title=f"step {t}")
        if np.asarray(world.wall).any():
            draw_walls(ax, world)
        if comm_radius is not None:
            pos = np.asarray(world.body.position)
            d = np.max(np.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
            draw_comm_graph(ax, world, d <= comm_radius)
        frames.append(_fig_to_rgba(fig))
    save_gif(frames, output_path, fps)


# ---- CLI entry point (zymera-viz) -----------------------------------------


def main() -> None:
    """``zymera-viz`` — run a random rollout, dump a flat gif."""
    import argparse
    import jax

    from .. import make, random_policy, rollout

    parser = argparse.ArgumentParser(
        prog="zymera-viz", description="Render a zymera rollout as a gif.")
    parser.add_argument("--grid-h",   type=int, default=8)
    parser.add_argument("--grid-w",   type=int, default=8)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-steps",  type=int, default=32)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--fps",      type=int, default=4)
    parser.add_argument("--iso",      action="store_true", help="isometric 3D view")
    parser.add_argument("--output",   type=str, default="rollout.gif")
    args = parser.parse_args()

    env = make("empty-v0", grid_h=args.grid_h, grid_w=args.grid_w,
               n_agents=args.n_agents)
    traj = rollout(env, random_policy, args.n_steps, jax.random.PRNGKey(args.seed))
    if args.iso:
        from .iso import iso_gif
        iso_gif(traj, args.output, fps=args.fps)
    else:
        render_gif(traj, args.output, fps=args.fps)
    print(f"wrote {args.output}  ({args.n_steps} steps, {args.n_agents} agents on "
          f"{args.grid_h}×{args.grid_w}, seed {args.seed}{', iso' if args.iso else ''})")


if __name__ == "__main__":
    main()
