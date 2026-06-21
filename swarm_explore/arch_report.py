"""Compare the architecture ladder on 16x16 / 4 agents / 70 steps / comm-range 5:
  baseline (uncertainty compass, soft lambda)
  v0  spanning-tree backbone (deterministic)
  v1  backbone + GNN role head (learned: when a leaf flips to relay)
  v2  backbone + attention goal head (learned: which frontier a leaf picks)

Reports STEP-BY-STEP curves (coverage & cohesion over time), a 5-seed summary
table, and an audit GIF per method, all embedded in a self-contained HTML report.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.arch_report
"""
from __future__ import annotations

import base64
import glob
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

from .core import connected_components, random_wall   # noqa: E402
from .swarm import run_swarm                           # noqa: E402
from .gif import animate_run                           # noqa: E402

H = W = 16
N = 4
STEPS = 70
COMM_R = 5
SENSE_R = 3
SEEDS = list(range(5))
GIF_SEED = 2
HERE = os.path.dirname(__file__)
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "experiments", "swarm_explore", "arch16")


def _latest(pat):
    fs = sorted(glob.glob(os.path.join(ROOT, "experiments/swarm_explore", pat)))
    return np.load(fs[-1]) if fs else None


def methods():
    gnn = _latest("arch-gnn-*/best_gnn.npy")
    attn = _latest("arch-attn-*/best_attn.npy")
    m = [("baseline (compass)", dict(lambda_conn=1.0)),
         ("v0 backbone", dict(backbone=True))]
    if gnn is not None:
        m.append(("v1 backbone+GNN", dict(backbone=True, gnn_theta=gnn)))
    if attn is not None:
        m.append(("v2 backbone+attn", dict(backbone=True, attn_theta=attn)))
    return m


def per_step(frames, free_total):
    cov, coh, conn = [], [], []
    for fr in frames:
        cnt = fr["count"]; adj = fr["adj"]
        cov.append((cnt >= 1).sum() / free_total)
        comps = connected_components(adj)
        coh.append(max(len(c) for c in comps) / N)
        conn.append(1.0 if len(comps) == 1 else 0.0)
    return np.array(cov), np.array(coh), np.array(conn)


def b64(path, mime):
    return (f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
            if os.path.exists(path) else "")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "gifs"), exist_ok=True)
    M = methods()
    summary, curves = [], {}

    for name, kw in M:
        # 5-seed averages
        acc = dict(coverage=0.0, connectivity=0.0, largest_comp_frac=0.0, redundancy=0.0,
                   evenness=0.0, dol_gini=0.0, relay_frac=0.0)
        for s in SEEDS:
            wall = random_wall(H, W, 0.05, s)
            r = run_swarm(H, W, wall, N, COMM_R, SENSE_R, STEPS, s, **kw)
            for k in acc:
                acc[k] += r[k]
        acc = {k: v / len(SEEDS) for k, v in acc.items()}
        acc["name"] = name
        summary.append(acc)
        # representative seed for curves + GIF
        wall = random_wall(H, W, 0.05, GIF_SEED)
        rr = run_swarm(H, W, wall, N, COMM_R, SENSE_R, STEPS, GIF_SEED, record=True, **kw)
        free_total = int((~wall).sum())
        curves[name] = per_step(rr["frames"], free_total)
        slug = name.split()[0].replace("(", "").replace(")", "")
        gpath = os.path.join(OUT, "gifs", f"{slug}.gif")
        animate_run(rr, gpath, title=name, sense_r=SENSE_R)
        print(f"{name:22s} cov {acc['coverage']*100:5.1f}%  conn {acc['connectivity']*100:4.0f}%  "
              f"cohesion {acc['largest_comp_frac']*100:4.0f}%  relay {acc['relay_frac']*100:3.0f}%  -> {os.path.basename(gpath)}",
              flush=True)

    # ---- step-by-step curves figure ----
    colors = {"baseline (compass)": "#9aa0a6", "v0 backbone": "#d9480f",
              "v1 backbone+GNN": "#1f6feb", "v2 backbone+attn": "#2b7a3b"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, _ in M:
        cov, coh, conn = curves[name]
        t = np.arange(1, len(cov) + 1)
        c = colors.get(name, "#555")
        ax[0].plot(t, cov * 100, color=c, label=name)
        ax[1].plot(t, coh * 100, color=c, label=name)
    ax[0].set_title("coverage over time"); ax[0].set_xlabel("step"); ax[0].set_ylabel("coverage %"); ax[0].set_ylim(0, 100)
    ax[1].set_title("cohesion (largest connected group) over time"); ax[1].set_xlabel("step"); ax[1].set_ylabel("%"); ax[1].set_ylim(0, 100)
    ax[0].grid(alpha=.3); ax[1].grid(alpha=.3); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle("Architecture ladder @ 16×16 / 4 agents / 70 steps / comm-range 5 (seed 2)")
    fig.tight_layout()
    curve_png = os.path.join(OUT, "curves.png")
    fig.savefig(curve_png, dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- HTML ----
    CSS = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;line-height:1.55;"
           "max-width:1050px;margin:0 auto;padding:0 24px 80px;background:#fafafa}"
           "h1{font-size:25px;border-bottom:3px solid #d9480f;padding-bottom:10px}"
           "h2{font-size:19px;margin-top:30px;border-top:1px solid #e3e6ea;padding-top:16px}"
           "table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}"
           "th,td{border:1px solid #e3e6ea;padding:8px 10px;text-align:left}th{background:#f1f3f5}"
           ".win{color:#2b7a3b;font-weight:700}.meta{color:#5b6570;font-size:13px}"
           "figure{margin:14px 0;text-align:center}figure img{max-width:100%;border:1px solid #e3e6ea;border-radius:8px;background:#fff}"
           "figcaption{font-size:12.5px;color:#5b6570;margin-top:6px}"
           ".gif{display:inline-block;width:47%;vertical-align:top;margin:1%}"
           ".callout{background:#edf2ff;border:1px solid #bcd0ff;border-radius:8px;padding:11px 15px;font-size:13.5px;margin:12px 0}")
    Hh = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Architecture ladder @16×16</title><style>{CSS}</style></head><body>"]
    Hh.append("<h1>Spanning-tree backbone architecture — v0 / v1 / v2 @ 16×16</h1>")
    Hh.append("<p class='meta'>16×16 grid · 4 agents · 70 steps · comm-range 5 · sense radius 3 · 5% obstacles · "
              "collision hard-masked · fully decentralized. Summary = 5-seed average; curves &amp; GIFs = seed 2.</p>")
    Hh.append("<div class='callout'><b>The ladder.</b> "
              "<b>v0</b> = deterministic spanning-tree backbone (role read off the tree: internal=relay, leaf=frontier; "
              "leaf stays within comm-range of its parent). "
              "<b>v1</b> = + a learned GNN role head (a leaf may flip to relay from its graph context). "
              "<b>v2</b> = + a learned attention head that ranks the leaf's candidate frontier cells. "
              "Amber dots = relay (spine) · blue = frontier (leaf).</div>")

    Hh.append("<h2>Summary (5-seed average)</h2>")
    Hh.append("<table><tr><th>method</th><th>coverage</th><th>full-connectivity</th><th>cohesion</th>"
              "<th>relay rate</th><th>redundancy</th><th>DoL-Gini</th></tr>")
    bestcov = max(s["coverage"] for s in summary)
    bestcoh = max(s["largest_comp_frac"] for s in summary)
    for s in summary:
        cc = " class='win'" if s["coverage"] == bestcov else ""
        hc = " class='win'" if s["largest_comp_frac"] == bestcoh else ""
        Hh.append(f"<tr><td>{s['name']}</td><td{cc}>{s['coverage']*100:.1f}%</td>"
                  f"<td>{s['connectivity']*100:.0f}%</td><td{hc}>{s['largest_comp_frac']*100:.0f}%</td>"
                  f"<td>{s['relay_frac']*100:.0f}%</td><td>{s['redundancy']:.2f}×</td><td>{s['dol_gini']:.2f}</td></tr>")
    Hh.append("</table>")

    Hh.append("<h2>Step-by-step (coverage &amp; cohesion over time)</h2>")
    Hh.append(f"<figure><img src='{b64(curve_png,'image/png')}'/>"
              "<figcaption>Left: coverage climbs over the 70 steps. Right: cohesion (largest connected group). "
              "The backbone variants trade a little coverage speed for a much steadier connected structure.</figcaption></figure>")

    Hh.append("<h2>Audit GIFs (seed 2)</h2>")
    for name, _ in M:
        slug = name.split()[0].replace("(", "").replace(")", "")
        g = b64(os.path.join(OUT, "gifs", f"{slug}.gif"), "image/gif")
        if g:
            Hh.append(f"<figure class='gif'><img src='{g}'/><figcaption>{name}</figcaption></figure>")
    Hh.append("</body></html>")
    outhtml = os.path.join(OUT, "report.html")
    open(outhtml, "w").write("\n".join(Hh))
    print(f"\nreport -> {outhtml}  ({round(os.path.getsize(outhtml)/1e6,1)} MB)")


if __name__ == "__main__":
    main()
