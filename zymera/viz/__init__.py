"""
zymera.viz — all visualization, kept out of the headless core.

``import zymera`` pulls in no plotting libraries; everything that draws lives
here and imports matplotlib / pillow lazily.

    from zymera import viz
    viz.render_gif(traj, "flat.gif")          # flat top-down gif
    viz.iso_gif(traj, "iso.gif")              # isometric 3D gif
    viz.make_report(traj, "report.html")      # self-contained HTML report
    viz.teleop(...)                           # interactive keyboard window
"""

from .render import (
    compute_fog_mask,
    draw_comm_graph,
    draw_fog,
    draw_frame,
    draw_grid_lines,
    draw_selection_highlight,
    draw_walls,
    iter_worlds,
    render_comm_gif,
    render_gif,
    save_gif,
)
from .iso import draw_iso, iso_gif
from .report import make_report
from .live import live_keyboard

teleop = live_keyboard

__all__ = [
    # gif export
    "render_gif", "render_comm_gif", "iso_gif", "save_gif",
    # report
    "make_report",
    # draw primitives
    "draw_frame", "draw_iso", "draw_walls", "draw_grid_lines",
    "draw_comm_graph", "draw_fog", "compute_fog_mask",
    "draw_selection_highlight", "iter_worlds",
    # interactive
    "teleop", "live_keyboard",
]
