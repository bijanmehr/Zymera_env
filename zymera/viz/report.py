"""
Self-contained HTML report — borrowed from RedWithinBlue's mechanism: an
f-string template with every image (charts + episode gif) base64-embedded
inline, so the ``report.html`` is a single portable file.

    from zymera.viz import make_report
    make_report(traj, "report.html", title="lawn-mower run", metrics={"loss": losses})
"""

import base64
import io
import os
import tempfile

import numpy as np

from .render import _agg_figure, iter_worlds, render_comm_gif, render_gif

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<style>
 body {{ font-family:-apple-system,system-ui,sans-serif; max-width:860px;
        margin:32px auto; color:#1a1a1a; padding:0 16px; }}
 h1 {{ font-size:24px; margin-bottom:2px; }}
 .meta {{ color:#666; font-size:14px; margin-top:0; }}
 .grid {{ display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; }}
 .card {{ border:1px solid #e1e4e8; border-radius:8px; padding:12px; }}
 .card h3 {{ margin:0 0 8px; font-size:14px; color:#444; font-weight:600; }}
 img {{ max-width:100%; display:block; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="meta">{summary}</p>
<div class="grid">
<div class="card"><h3>Episode</h3><img src="data:image/gif;base64,{gif}"></div>
{cards}
</div>
</body></html>
"""


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _coverage_chart(cov) -> str:
    fig = _agg_figure((4.2, 2.4))
    ax = fig.add_subplot(111)
    ax.plot(range(len(cov)), [c * 100 for c in cov], color="#2c4f8a", linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("coverage %"); ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_b64(fig)


def _heatmap_chart(explored) -> str:
    fig = _agg_figure((3.2, 3.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.asarray(explored), cmap="magma")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_b64(fig)


def _curve_chart(arr, label) -> str:
    fig = _agg_figure((4.2, 2.4))
    ax = fig.add_subplot(111)
    ax.plot(np.asarray(arr), color="#8a2c2c", linewidth=2)
    ax.set_xlabel("iteration"); ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_b64(fig)


def make_report(trajectory: dict, output_path: str, title: str = "zymera run",
                fps: int = 8, iso: bool = False, metrics: dict = None,
                comm_radius: int = None) -> None:
    """Write a single self-contained ``report.html``: an embedded episode gif
    plus coverage-over-time, the visit heatmap, and any training ``metrics``
    (a dict of ``name -> 1D array``).

    ``comm_radius`` (flat only): if set, the episode gif overlays dotted
    comm-graph edges between agents within that Chebyshev range — so a
    stretched relay chain shows up as a connected path.
    """
    n_frames, get_world = iter_worlds(trajectory)
    explored_final = np.asarray(get_world(n_frames - 1).explored)
    H, W = explored_final.shape
    n_agents = np.asarray(get_world(0).body.position).shape[0]
    cov = [float((np.asarray(get_world(t).explored) > 0).sum()) / (H * W)
           for t in range(n_frames)]

    # episode gif → base64 (flat or iso)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    if iso:
        from .iso import iso_gif
        iso_gif(trajectory, tmp.name, fps=fps)
    elif comm_radius is not None:
        render_comm_gif(trajectory, tmp.name, fps=fps, comm_radius=comm_radius)
    else:
        render_gif(trajectory, tmp.name, fps=fps)
    with open(tmp.name, "rb") as f:
        gif_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp.name)

    cards = [("Coverage over time", _coverage_chart(cov)),
             ("Visit counts", _heatmap_chart(explored_final))]
    for name, arr in (metrics or {}).items():
        cards.append((name, _curve_chart(arr, name)))
    cards_html = "\n".join(
        f'<div class="card"><h3>{name}</h3>'
        f'<img src="data:image/png;base64,{b64}"></div>' for name, b64 in cards)

    summary = (f"{H}×{W} grid · {n_agents} agent(s) · {n_frames - 1} steps · "
               f"final coverage {cov[-1] * 100:.0f}%")
    html = _TEMPLATE.format(title=title, summary=summary, gif=gif_b64, cards=cards_html)
    with open(output_path, "w") as f:
        f.write(html)
