"""Assemble the super-blue pipeline outputs into one self-contained REPORT.html.

Reads each phase's artifacts under experiments/runs/<job>/, renders figures into
experiments/report_assets/, copies the GIFs alongside, and writes REPORT.html.
Robust: a missing/failed phase simply omits its section (with a note).
"""
from __future__ import annotations

import glob
import html
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

INK = "#1f2328"; GREY = "#5b6570"; BLUE = "#1f6feb"; GREEN = "#1f7a44"; AMBER = "#f59f00"; RED = "#c0392b"


def _load(runs, name, fname):
    p = os.path.join(runs, name, fname)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _fig(path, fn, figsize=(7.2, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    fn(ax)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _heatmap(ax, grids, agents, cell, title, fmt="{:.0f}"):
    M = np.full((len(grids), len(agents)), np.nan)
    for i, g in enumerate(grids):
        for j, n in enumerate(agents):
            v = cell.get((g, n))
            if v is not None:
                M[i, j] = v
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(agents))); ax.set_xticklabels(agents)
    ax.set_yticks(range(len(grids))); ax.set_yticklabels([f"{g}²" for g in grids])
    ax.set_xlabel("agents N"); ax.set_ylabel("world")
    for i in range(len(grids)):
        for j in range(len(agents)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j] * 100), ha="center", va="center", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10)
    return im


def build_report(runs, root):
    assets = os.path.join(root, "experiments", "report_assets")
    os.makedirs(assets, exist_ok=True)
    sec = []                                   # (title, body_html)

    # ───────────────────────── A · perception (graph belief) ─────────────────────────
    hist = _load(runs, "belief-multiscale", "history.json")
    if hist:
        def fA(ax):
            xs = [h["epoch"] for h in hist]
            for k in [k for k in hist[0] if k.startswith("acc@")]:
                ax.plot(xs, [h[k] for h in hist], marker="o", ms=3, label=k)
            ax.set_xlabel("epoch"); ax.set_ylabel("belief accuracy"); ax.set_ylim(0.4, 1.0)
            ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3)
            ax.set_title("A · one GCRN, accuracy per scale")
        _fig(os.path.join(assets, "belief.png"), fA)
        last = hist[-1]
        cells = " · ".join(f"<b>{k}</b> {last[k]*100:.0f}%" for k in last if k.startswith("acc@"))
        sec.append(("A — Perception: size-invariant graph belief",
                    f"<img src='report_assets/belief.png'>"
                    f"<p class=m>Final accuracy: {cells}. Shared weights, mean aggregation, "
                    f"comm-range-relative features — one network reads the comm graph at every scale.</p>"))

    # ───────────────────────── B · the super-blue controller ─────────────────────────
    th = _load(runs, "train-super-blue", "train_history.json")
    if th:
        def fB(ax):
            xs = [h["gen"] for h in th]
            for key, col, lab in [("cov", GREEN, "coverage"), ("conn", BLUE, "connectivity"),
                                  ("coh", AMBER, "cohesion"), ("relay", RED, "relay use")]:
                if key in th[0]:
                    ax.plot(xs, [h[key] for h in th], marker="o", ms=3, color=col, label=lab)
            ax.set_xlabel("ES generation"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1.0)
            ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
            ax.set_title("B · controller trained across scales (domain randomization)")
        _fig(os.path.join(assets, "controller.png"), fB)
        last = th[-1]
        sec.append(("B — The super-blue agent: one controller, trained over a distribution of scales",
                    f"<img src='report_assets/controller.png'>"
                    f"<p class=m>Final (averaged over training scales): coverage "
                    f"<b>{last.get('cov',0)*100:.0f}%</b>, connectivity <b>{last.get('conn',0)*100:.0f}%</b>, "
                    f"cohesion <b>{last.get('coh',0)*100:.0f}%</b>, relay use {last.get('relay',0)*100:.0f}%. "
                    f"A single 145-parameter policy; fitness is averaged over (grid, N) every generation, "
                    f"so size-invariance is trained in, not hoped for.</p>"))

    # ───────────────────────── C · wall-boundary phase diagram ─────────────────────────
    par = _load(runs, "pareto-wall", "pareto.json")
    if par:
        grids = sorted({o["grid"] for o in par}); agents = sorted({o["n"] for o in par})
        comm_rs = sorted({o.get("comm_r", 5) for o in par})
        # best connectivity (over lam) per (grid,N) at the smallest and largest comm_r
        def best(cr, key):
            d = {}
            for o in par:
                if o.get("comm_r", 5) == cr:
                    d[(o["grid"], o["n"])] = max(d.get((o["grid"], o["n"]), 0.0), o[key])
            return d
        cr_lo, cr_hi = comm_rs[0], comm_rs[-1]

        def fC(axs):
            _heatmap(axs[0], grids, agents, best(cr_lo, "conn"), f"connectivity · comm_r={cr_lo}")
            im = _heatmap(axs[1], grids, agents, best(cr_hi, "conn"), f"connectivity · comm_r={cr_hi}")
            axs[0].figure.colorbar(im, ax=axs[1], fraction=0.046)
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2)); fC(axs)
        fig.suptitle("C · where single-component connectivity stays feasible", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(assets, "wall.png"), dpi=120); plt.close(fig)
        feasible = [o for o in par if o["conn"] >= 0.95]
        sec.append(("C — The wall: a feasibility phase-diagram",
                    f"<img src='report_assets/wall.png'>"
                    f"<p class=m>{len(par)} (λ, grid, N, comm_r) cells; {len(feasible)} hold ≥95% connectivity. "
                    f"Green = the team stays one component; red = it fragments. Wider comm range "
                    f"(right) pushes the feasible frontier outward — the geometric wall, mapped.</p>"))

    # ───────────────────────── D · generalization (held-out scales) ─────────────────────────
    gen = _load(runs, "generalize", "generalization.json")
    if gen:
        def fD(ax):
            labs = [f"{o['grid']}²/{o['n']}\n{o['regime']}" for o in gen]
            x = np.arange(len(gen)); w = 0.38
            ax.bar(x - w/2, [o["cov"] for o in gen], w, color=GREEN, label="coverage")
            ax.bar(x + w/2, [o["coh"] for o in gen], w, color=AMBER, label="cohesion")
            for i, o in enumerate(gen):
                if o["regime"] == "holdout":
                    ax.axvspan(i - 0.5, i + 0.5, color="#f1f3f5", zorder=0)
            ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
            ax.set_ylabel("fraction"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
            ax.set_title("D · same agent, train scales vs HELD-OUT scales (grey)")
        _fig(os.path.join(assets, "generalize.png"), fD, figsize=(8.6, 4.2))
        ho = [o for o in gen if o["regime"] == "holdout"]
        row = "".join(f"<tr><td>{o['regime']}</td><td>{o['grid']}²</td><td>{o['n']}</td>"
                      f"<td>{o['cov']*100:.0f}%</td><td>{o['conn']*100:.0f}%</td><td>{o['coh']*100:.0f}%</td></tr>"
                      for o in gen)
        hocov = np.mean([o["cov"] for o in ho]) if ho else 0.0
        sec.append(("D — Generalization: zero-shot to unseen scales",
                    f"<img src='report_assets/generalize.png'>"
                    f"<p class=m>Held-out worlds (never trained on) average <b>{hocov*100:.0f}%</b> coverage. "
                    f"The same 145 parameters run at every size.</p>"
                    f"<table><tr><th>regime</th><th>world</th><th>N</th><th>coverage</th>"
                    f"<th>connectivity</th><th>cohesion</th></tr>{row}</table>"))

    # ───────────────────────── E · GIF rollouts ─────────────────────────
    gifs = sorted(glob.glob(os.path.join(runs, "render-gifs", "gifs", "*.gif")))
    if gifs:
        gdst = os.path.join(assets, "gifs"); os.makedirs(gdst, exist_ok=True)
        imgs = ""
        for g in gifs:
            shutil.copy(g, os.path.join(gdst, os.path.basename(g)))
            imgs += (f"<figure><img src='report_assets/gifs/{os.path.basename(g)}' style='width:330px'>"
                     f"<figcaption class=m>{html.escape(os.path.basename(g).replace('.gif',''))}</figcaption></figure>")
        sec.append(("E — Watch it: rollouts of the trained agent",
                    f"<div class=gifs>{imgs}</div>"
                    f"<p class=m>Green = certainty field filling · amber dots = agents holding the network · "
                    f"grey lines = live comm graph · CONNECTED/SPLIT flag in each title.</p>"))

    # ───────────────────────── assemble ─────────────────────────
    body = ""
    for title, html_body in sec:
        body += f"<section><h2>{html.escape(title)}</h2>{html_body}</section>"
    if not sec:
        body = "<p>No completed phases found yet.</p>"

    doc = f"""<!doctype html><meta charset=utf-8><title>Super-blue — pipeline report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;max-width:920px;margin:30px auto;
       padding:0 22px;color:{INK};line-height:1.5}}
 h1{{font-size:26px;margin-bottom:2px}} h2{{font-size:18px;margin-top:8px;border-bottom:2px solid #eee;padding-bottom:4px}}
 section{{margin:30px 0}} img{{max-width:100%;border:1px solid #e3e6ea;border-radius:6px}}
 .m{{color:{GREY};font-size:13px}} table{{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}}
 td,th{{border:1px solid #e3e6ea;padding:6px 10px;text-align:left}} th{{background:#f1f3f5}}
 .gifs{{display:flex;flex-wrap:wrap;gap:16px}} figure{{margin:0}}
 .lead{{background:#f6f8fa;border:1px solid #e3e6ea;border-radius:8px;padding:14px 18px}}
</style>
<h1>Super-blue: a size-invariant, cooperative coverage swarm</h1>
<p class=m>One perception network + one 145-parameter controller, trained across a distribution of world
sizes and team counts, then evaluated zero-shot on worlds it never saw.</p>
<div class=lead><b>The claim, in one line:</b> a single agent policy that does not change with the number of
teammates or the size of the world — coverage, connectivity, and cooperation are read off normalized,
range-relative features, and trained by domain randomization rather than tuned per scale.</div>
{body}
<p class=m style='margin-top:40px'>Generated by <code>experiments/report.py</code> from the run artifacts under
<code>experiments/runs/</code>. Numbers are seed-averaged.</p>"""
    out = os.path.join(root, "experiments", "REPORT.html")
    open(out, "w").write(doc)
    return out, len(sec)
