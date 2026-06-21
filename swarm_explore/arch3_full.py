"""Full validation of the 3-stack architecture @ 16x16 / 4 agents / 70 steps:
  - PARAMETER SWEEP over the mission-safety knobs (K budget x w_safety) for the ES
    role-switcher (Stack 2), each ES-trained over many maps.
  - pick BEST, then MULTI-MAP robustness (many seeds x obstacle densities).
  - compare to the compass-only baseline (Stack 1 alone) and v0 backbone.
  - fold in the separately-trained graph estimator (Stack 3) accuracy.
  - step-by-step self-contained HTML report + audit GIF.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.arch3_full
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os

import numpy as np

from .core import random_wall
from .swarm import run_swarm
from .arch3 import init_theta, run_arch, SW_NPARAMS
from .gif import animate_run

H = W = 16
N = 4
STEPS = 70
CR = 5
SR = 3
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "experiments", "swarm_explore", "arch3")


def es_train(K, w_safety, lam, w_coh=0.3, hi=0.6, lo=0.4, gens=8, pop=6, eval_seeds=4,
             sigma=0.15, alpha=0.08, seed=0):
    """Compact ES for the role-switcher (mission-safety fitness). Frontier uses soft-lambda
    `lam` as its connectivity base; the switcher adds selective relaying. Returns (theta, metrics)."""
    seeds = list(range(eval_seeds)); rng = np.random.default_rng(seed)
    theta = init_theta(seed)

    def metr(th, sds):
        cov = conn = coh = sv = rl = 0.0
        for s in sds:
            wall = random_wall(H, W, 0.05, s)
            m = run_arch(H, W, wall, N, CR, SR, STEPS, s, th, K=K, hi=hi, lo=lo, lam_frontier=lam)
            cov += m["coverage"]; conn += m["connectivity"]; coh += m["largest_comp_frac"]
            sv += m["safety_viol"]; rl += m["relay_frac"]
        k = len(sds); return dict(cov=cov/k, conn=conn/k, coh=coh/k, sv=sv/k, rl=rl/k)

    # fitness rewards BOTH coverage and connectivity, penalizes safety violations
    def fit(th):
        m = metr(th, seeds); return m["cov"] + 0.6 * m["conn"] + w_coh * m["coh"] - w_safety * m["sv"], m
    best = (fit(theta)[0], theta.copy())
    for g in range(gens):
        eps = rng.normal(0, 1, (pop, theta.size)); eps = np.concatenate([eps, -eps], 0)
        F = np.array([fit(theta + sigma * e)[0] for e in eps])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + alpha * (eps.T @ A) / (len(eps) * sigma)
        f, _ = fit(theta)
        if f > best[0]:
            best = (f, theta.copy())
    return best[1], metr(best[1], seeds)


def multimap(theta_or_none, K=2, lam=1.0, densities=(0.0, 0.05, 0.10), seeds=range(40, 64)):
    """Robustness: eval over many maps x obstacle densities. theta=None -> compass-only (lambda=1)."""
    rows = {}
    for d in densities:
        cov = conn = coh = sv = 0.0; covs = []
        for s in seeds:
            wall = random_wall(H, W, d, s)
            if theta_or_none is None:
                m = run_swarm(H, W, wall, N, CR, SR, STEPS, s, lambda_conn=1.0)
                m.setdefault("safety_viol", float("nan"))
            else:
                m = run_arch(H, W, wall, N, CR, SR, STEPS, s, theta_or_none, K=K, lam_frontier=lam)
            cov += m["coverage"]; conn += m["connectivity"]; coh += m["largest_comp_frac"]
            sv += m.get("safety_viol", 0.0); covs.append(m["coverage"])
        k = len(list(seeds))
        rows[d] = dict(cov=cov/k, conn=conn/k, coh=coh/k, sv=sv/k, cov_std=float(np.std(covs)))
    return rows


def b64png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64file(p, mime):
    return (f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()) if os.path.exists(p) else ""


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)

    # ---- 1) PARAMETER SWEEP (Stack 2): K budget x frontier-lambda x w_safety ----
    print("=== parameter sweep (K x lambda x w_safety) ===", flush=True)
    Ks = [2, 3]; Lams = [1.0]; Ws = [0.5, 1.5, 3.0]
    sweep = []
    for K in Ks:
        for lam in Lams:
            for w in Ws:
                theta, m = es_train(K, w, lam)
                score = m["cov"] + m["conn"] - m["sv"]            # excellent = BOTH high
                sweep.append(dict(K=K, lam=lam, w_safety=w, score=score, theta=theta, **m))
                print(f"  K={K} lam={lam} w_safety={w}: cov {m['cov']*100:4.0f}%  conn {m['conn']*100:3.0f}%  "
                      f"coh {m['coh']*100:3.0f}%  safety-viol {m['sv']*100:4.1f}%  relay {m['rl']*100:3.0f}%  score {score:.3f}", flush=True)
    best = max(sweep, key=lambda r: r["score"])
    print(f"  BEST: K={best['K']} lam={best['lam']} w_safety={best['w_safety']}  "
          f"(cov {best['cov']*100:.0f}% conn {best['conn']*100:.0f}% coh {best['coh']*100:.0f}%)", flush=True)
    np.save(os.path.join(OUT, "best_theta.npy"), best["theta"])

    # ---- 2) MULTI-MAP robustness: best 3-stack vs compass-only ----
    print("=== multi-map robustness (24 seeds x 3 densities) ===", flush=True)
    arch_mm = multimap(best["theta"], K=best["K"], lam=best["lam"])
    base_mm = multimap(None)
    for d in (0.0, 0.05, 0.10):
        a, b = arch_mm[d], base_mm[d]
        print(f"  obstacles {int(d*100):2d}%:  3-stack cov {a['cov']*100:4.0f}% conn {a['conn']*100:3.0f}% coh {a['coh']*100:3.0f}% sv {a['sv']*100:3.1f}%   |   compass cov {b['cov']*100:4.0f}% conn {b['conn']*100:3.0f}% coh {b['coh']*100:3.0f}%", flush=True)

    # ---- 3) estimator (Stack 3) accuracy (trained separately) ----
    est = None
    efs = sorted(glob.glob(os.path.join(ROOT, "experiments/swarm_explore/estimator-*/estimator.json")))
    if efs:
        ej = json.load(open(efs[-1])); est = dict(acc=ej["history"][-1]["test_acc"], base=ej["baseline_acc"])

    # ---- 4) GIF of best (seed 2) ----
    wall = random_wall(H, W, 0.05, 2)
    mrec = run_arch(H, W, wall, N, CR, SR, STEPS, 2, best["theta"], K=best["K"], record=True)
    gpath = os.path.join(OUT, "arch3.gif")
    animate_run({"frames": mrec["frames"], "wall": wall, "coverage": mrec["coverage"], "connectivity": mrec["connectivity"]},
                gpath, title="3-stack (compass + ES role-switcher)", sense_r=SR)

    # ---- 5) figures ----
    # sweep scatter: each config on the coverage-connectivity plane
    figS, axS = plt.subplots(figsize=(6, 4.4))
    for r in sweep:
        mk = "*" if r is best else "o"; sz = 220 if r is best else 70
        axS.scatter(r["cov"] * 100, r["conn"] * 100, s=sz, marker=mk,
                    c="#2b7a3b" if r is best else "#1f6feb", zorder=5 if r is best else 3)
        axS.annotate(f"K{r['K']} λ{r['lam']} w{r['w_safety']}", (r["cov"]*100, r["conn"]*100), fontsize=6.5, color="#444")
    axS.scatter([base_mm[0.05]["cov"]*100], [base_mm[0.05]["conn"]*100], s=120, marker="s",
                c="#9aa0a6", zorder=4, label="compass baseline")
    axS.legend(fontsize=8)
    axS.set_xlabel("coverage %"); axS.set_ylabel("full-connectivity %"); axS.set_title("parameter sweep on the coverage–connectivity plane")
    axS.grid(alpha=.3); sweep_png = b64png(figS)
    # robustness bars
    figR, axR = plt.subplots(figsize=(7, 4))
    ds = [0.0, 0.05, 0.10]; x = np.arange(len(ds)); wbar = 0.35
    axR.bar(x - wbar/2, [arch_mm[d]["cov"]*100 for d in ds], wbar, label="3-stack cov", color="#2b7a3b")
    axR.bar(x + wbar/2, [base_mm[d]["cov"]*100 for d in ds], wbar, label="compass cov", color="#9aa0a6")
    axR.plot(x - wbar/2, [arch_mm[d]["conn"]*100 for d in ds], "o-", color="#1f6feb", label="3-stack conn")
    axR.plot(x + wbar/2, [base_mm[d]["conn"]*100 for d in ds], "o--", color="#9aa0a6", label="compass conn")
    axR.set_xticks(x); axR.set_xticklabels([f"{int(d*100)}% obs" for d in ds]); axR.set_ylim(0, 100)
    axR.set_ylabel("%"); axR.set_title("multi-map robustness (24 maps/density)"); axR.legend(fontsize=8)
    rob_png = b64png(figR)

    # ---- 6) HTML ----
    CSS = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;line-height:1.55;max-width:1000px;"
           "margin:0 auto;padding:0 24px 80px;background:#fafafa}h1{font-size:25px;border-bottom:3px solid #2b7a3b;padding-bottom:10px}"
           "h2{font-size:19px;margin-top:30px;border-top:1px solid #e3e6ea;padding-top:16px}table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}"
           "th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:left}th{background:#f1f3f5}.win{color:#2b7a3b;font-weight:700}"
           ".meta{color:#5b6570;font-size:13px}figure{margin:14px 0;text-align:center}figure img{max-width:100%;border:1px solid #e3e6ea;border-radius:8px;background:#fff}"
           "figcaption{font-size:12.5px;color:#5b6570;margin-top:6px}.callout{background:#edf2ff;border:1px solid #bcd0ff;border-radius:8px;padding:11px 15px;font-size:13.5px;margin:12px 0}")
    Hh = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>3-stack architecture @16x16</title><style>{CSS}</style></head><body>"]
    Hh.append("<h1>3-Stack Swarm Architecture — validation @ 16×16 / 4 / 70 / comm-range 5</h1>")
    Hh.append("<p class='meta'>Stack 1 compass (deterministic) · Stack 2 role-picker+switcher (ES, mission-safety fitness, "
              "graph features, hysteresis) · Stack 3 graph estimator (supervised, separate). Trained separately, run together.</p>")

    Hh.append("<h2>Step 1 — Parameter sweep (Stack 2: K budget × frontier-λ × w_safety)</h2>")
    Hh.append("<p class='meta'>Each point = an ES-trained role-switcher (mission-safety fitness). Frontier uses soft-λ as its "
              "connectivity base; the switcher adds selective relaying. Score = coverage + connectivity − safety-violations "
              "(rewards BOTH axes). Grey square = compass-only baseline.</p>")
    Hh.append(f"<figure style='width:70%'><img src='{sweep_png}'/></figure>")
    Hh.append("<table><tr><th>K</th><th>λ_frontier</th><th>w_safety</th><th>coverage</th><th>connectivity</th><th>cohesion</th><th>safety-viol</th><th>relay</th><th>score</th></tr>")
    for r in sorted(sweep, key=lambda r: -r["score"]):
        cls = " class='win'" if r is best else ""
        Hh.append(f"<tr{cls}><td>{r['K']}</td><td>{r['lam']}</td><td>{r['w_safety']}</td><td>{r['cov']*100:.0f}%</td><td>{r['conn']*100:.0f}%</td>"
                  f"<td>{r['coh']*100:.0f}%</td><td>{r['sv']*100:.1f}%</td><td>{r['rl']*100:.0f}%</td><td>{r['score']:.3f}</td></tr>")
    Hh.append("</table>")
    Hh.append(f"<div class='callout'><b>Best config:</b> K={best['K']} (lose ≤{best['K']} neighbors), λ_frontier={best['lam']}, "
              f"w_safety={best['w_safety']} → cov {best['cov']*100:.0f}% / conn {best['conn']*100:.0f}% / "
              f"cohesion {best['coh']*100:.0f}% / safety-violations {best['sv']*100:.1f}%.</div>")

    Hh.append("<h2>Step 2 — Multi-map robustness (24 maps × 3 obstacle densities)</h2>")
    Hh.append(f"<figure><img src='{rob_png}'/></figure>")
    Hh.append("<table><tr><th>obstacles</th><th>3-stack cov</th><th>3-stack conn</th><th>3-stack coh</th><th>3-stack safety-viol</th>"
              "<th>compass cov</th><th>compass conn</th><th>compass coh</th></tr>")
    for d in ds:
        a, b = arch_mm[d], base_mm[d]
        Hh.append(f"<tr><td>{int(d*100)}%</td><td>{a['cov']*100:.0f}% ±{a['cov_std']*100:.0f}</td><td>{a['conn']*100:.0f}%</td>"
                  f"<td>{a['coh']*100:.0f}%</td><td>{a['sv']*100:.1f}%</td><td>{b['cov']*100:.0f}%</td><td>{b['conn']*100:.0f}%</td><td>{b['coh']*100:.0f}%</td></tr>")
    Hh.append("</table>")

    Hh.append("<h2>Step 3 — Stack 3 graph estimator (trained separately)</h2>")
    if est:
        Hh.append(f"<div class='callout'>Supervised graph-shape estimator: <b>{est['acc']*100:.0f}%</b> adjacency-prediction "
                  f"accuracy (trivial 'always-connected' baseline: {est['base']*100:.0f}%). Provides the graph-criticality "
                  "features the role-switcher reads; at 16×16/4 the gossiped graph is usually intact, so the run uses the "
                  "exact gossip features directly (the estimator matters more under fragmentation).</div>")
    else:
        Hh.append("<p class='meta'>(estimator not found — run <code>python -m swarm_explore.estimator</code>)</p>")

    Hh.append("<h2>Step 4 — Audit GIF (best config, seed 2)</h2>")
    g = b64file(gpath, "image/gif")
    if g:
        Hh.append(f"<figure style='width:55%'><img src='{g}'/><figcaption>Amber = relay (holding), blue = frontier (compass). "
                  "Role chosen by the ES switcher from graph-criticality features.</figcaption></figure>")
    Hh.append("</body></html>")
    open(os.path.join(OUT, "report.html"), "w").write("\n".join(Hh))
    json.dump({"sweep": [{k: (v if k != "theta" else None) for k, v in r.items()} for r in sweep],
               "best": {k: (best[k] if k != "theta" else None) for k in best},
               "arch_multimap": {str(d): arch_mm[d] for d in ds},
               "compass_multimap": {str(d): base_mm[d] for d in ds},
               "estimator": est}, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print(f"\nreport -> {os.path.join(OUT,'report.html')}", flush=True)


if __name__ == "__main__":
    main()
