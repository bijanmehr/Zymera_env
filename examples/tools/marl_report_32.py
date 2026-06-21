"""Self-contained HTML audit report for a 32x32 comm-coverage MARL run: training
curves (coverage + connectivity), headline metrics, the disruption budget, and the
embedded rollout GIF. For later audit of the >90%-coverage / min-graph-disruption goal.

  PYTHONPATH=examples/lib python examples/tools/marl_report_32.py <variant_dir> <gif_path> <out_html> [more_variant_dir:gif ...]
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def _b64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_file(path):
    return base64.b64encode(Path(path).read_bytes()).decode() if Path(path).exists() else None


def _curve(m, label):
    it = m["iter"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(it, m["cov"] * 100, color="#2b7a3b", label="coverage %")
    ax.axhline(90, color="#2b7a3b", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("training iteration"); ax.set_ylabel("coverage %", color="#2b7a3b")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(it, m["conn"] * 100, color="#1f6feb", label="connected %")
    ax2.set_ylabel("connected % of steps", color="#1f6feb"); ax2.set_ylim(0, 100)
    ax.set_title(f"{label} — coverage vs connectivity during training")
    return _b64_png(fig)


CSS = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;line-height:1.55;
max-width:1000px;margin:0 auto;padding:0 24px 80px;background:#fafafa}
h1{font-size:26px;border-bottom:3px solid #2b7a3b;padding-bottom:10px;margin-top:34px}
h2{font-size:20px;margin-top:30px;border-top:1px solid #e3e6ea;padding-top:16px}
.meta{color:#5b6570;font-size:13.5px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:8px 11px;text-align:left}th{background:#f1f3f5}
.win{font-weight:700;color:#2b7a3b}
figure{margin:16px 0;text-align:center}figure img{max-width:100%;border:1px solid #e3e6ea;border-radius:8px;background:#fff}
figcaption{font-size:12.5px;color:#5b6570;margin-top:6px}
.callout{border-radius:8px;padding:12px 16px;margin:14px 0;font-size:14px;background:#edf2ff;border:1px solid #bcd0ff}
code{background:#f1f3f5;padding:1px 5px;border-radius:4px;font-size:13px}"""


def variant_block(vdir, gif):
    vdir = Path(vdir)
    m = dict(np.load(vdir / "metrics.npz"))
    best = json.loads((vdir / "best.json").read_text())
    cfg = json.loads((vdir / "config.json").read_text())
    label = vdir.name
    # best operating point: highest coverage; report its connectivity (disruption)
    bi = int(np.argmax(m["cov"]))
    cov, conn, maxd = m["cov"][bi] * 100, m["conn"][bi] * 100, m["maxd"][bi]
    # also the point with best (cov>=0.90 AND max conn) if one exists
    ok = np.where(m["cov"] >= 0.90)[0]
    bestconn = None
    if len(ok):
        j = ok[np.argmax(m["conn"][ok])]
        bestconn = (m["cov"][j] * 100, m["conn"][j] * 100, m["maxd"][j], int(j))
    H = []
    H.append(f"<h2>{label}</h2>")
    rw = cfg.get("reward", {})
    H.append(f"<p class='meta'>32×32 · {cfg.get('n_agents','?')} agents · comm_r {cfg.get('comm_r','?')} · "
             f"reward {rw} · obs_channels incl. degree signal.</p>")
    H.append("<table><tr><th>operating point</th><th>coverage</th><th>connected (of steps)</th>"
             "<th>graph disruption</th><th>max-dist</th><th>iter</th></tr>")
    cov_cls = "win" if cov >= 90 else ""
    H.append(f"<tr><td>best coverage</td><td class='{cov_cls}'>{cov:.1f}%</td><td>{conn:.0f}%</td>"
             f"<td>{100-conn:.0f}% of steps</td><td>{maxd:.1f}</td><td>{bi}</td></tr>")
    if bestconn:
        H.append(f"<tr><td>best connectivity @ cov≥90%</td><td class='win'>{bestconn[0]:.1f}%</td>"
                 f"<td class='win'>{bestconn[1]:.0f}%</td><td>{100-bestconn[1]:.0f}% of steps</td>"
                 f"<td>{bestconn[2]:.1f}</td><td>{bestconn[3]}</td></tr>")
    H.append("</table>")
    H.append(f"<figure><img src='data:image/png;base64,{_curve(m, label)}'/>"
             "<figcaption>Coverage (green) climbs while connectivity (blue) is held; "
             "dotted line = 90% coverage goal.</figcaption></figure>")
    g = _b64_file(gif)
    if g:
        H.append(f"<figure><img src='data:image/gif;base64,{g}'/>"
                 "<figcaption>Trained rollout (seed 2): green = team-covered, blue = agents, "
                 "grey = live comm links, title = coverage% + CONNECTED/SPLIT.</figcaption></figure>")
    # emergent-behavior analysis (auto-embedded if marl_analyze_32 wrote it into the run dir)
    epng = vdir / "emergence.png"
    ejson = vdir / "emergence.json"
    if epng.exists():
        H.append("<h2 style='border:none'>Emergent behavior</h2>")
        H.append(f"<figure><img src='data:image/png;base64,{_b64_file(epng)}'/>"
                 "<figcaption>Coverage/connectivity over time · # comm components · "
                 "formation (mean degree &amp; max stretch).</figcaption></figure>")
    if ejson.exists():
        s = json.loads(ejson.read_text())
        H.append("<div class='callout'>")
        H.append(f"<b>Formation:</b> final mean degree {s.get('final_meandeg',0):.1f}, "
                 f"max stretch {s.get('final_maxd',0):.1f} (comm range {cfg.get('comm_r','?')}), "
                 f"{s.get('final_ncomp',0):.1f} components. "
                 f"<b>Division of labour:</b> coverage Gini {s.get('dol_gini',0):.2f} "
                 f"(0 = perfectly even territories). ")
        eff_c = s.get('degree_signal_effect_cov', 0) * 100
        eff_k = s.get('degree_signal_effect_conn', 0) * 100
        H.append(f"<b>Degree-signal ablation:</b> zeroing the degree channels changes coverage "
                 f"by {eff_c:+.1f} pts and connectivity by {eff_k:+.1f} pts — "
                 f"{'the policy genuinely uses the signal' if abs(eff_c)+abs(eff_k) > 3 else 'weak reliance on the signal'}.")
        H.append("</div>")
    return "\n".join(H), (cov, conn, bestconn)


def main():
    out = sys.argv[2] if len(sys.argv) == 4 else sys.argv[-1]
    # pairs: arg list is (vdir, gif, out) for single, or vdir:gif ... out for multi
    pairs = []
    if len(sys.argv) == 4:
        pairs = [(sys.argv[1], sys.argv[2])]
        out = sys.argv[3]
    else:
        for a in sys.argv[1:-1]:
            vd, gif = a.split("::")
            pairs.append((vd, gif))
        out = sys.argv[-1]

    H = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>MARL 32×32 — coverage & connectivity</title>",
         f"<style>{CSS}</style></head><body>",
         "<h1>PPO MARL on 32×32 — >90% coverage with minimal comm-graph disruption</h1>",
         "<p class='meta'>Goal: drive the PPO swarm above 90% sensed coverage on the 32×32 world "
         "(10 agents · comm range 5 · vision radius 3 · 100 steps) while keeping the comm graph "
         "intact as much as possible, using the per-agent <b>degree signal</b> in the observation.</p>"]
    for vd, gif in pairs:
        block, _ = variant_block(vd, gif)
        H.append(block)
    H.append("</body></html>")
    Path(out).write_text("\n".join(H))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
