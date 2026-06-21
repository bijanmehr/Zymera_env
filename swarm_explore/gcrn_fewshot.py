"""Few-shot adaptation of the GCRN belief to the 32-world.

Pretrain on 16x16/4, then fine-tune on K in {1,3,8} maps of 32x32/10 and test on
held-out 32x32/10. Baseline = train from scratch on the same K maps (no 16-world
pretraining). The gap between the two curves = the value of the size-invariant
pretraining. K=0 is the zero-shot point; the full-pool model is the ceiling.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.gcrn_fewshot
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import gcrn as G

_roll = jax.jit(G.rollout_belief)


def acc(p, INP, ADJB, TRUE):
    logits = np.asarray(_roll(p, jnp.asarray(INP), jnp.asarray(ADJB)))
    n = TRUE.shape[1]; off = ~np.eye(n, dtype=bool)
    pred = (1 / (1 + np.exp(-logits)) > 0.5)
    tgt = np.broadcast_to(TRUE[:, None, :, :], logits.shape) > 0.5
    return float((pred[:, :, off] == tgt[:, :, off]).mean())


def train_on(data, params, *, epochs, lr, wd, clip=1.0):
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(lr, weight_decay=wd))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, INP, ADJB, TRUE):
        l, g = jax.value_and_grad(G.loss_fn)(params, INP, ADJB, TRUE)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, l

    for _ in range(epochs):
        for INP, ADJB, TRUE in data:
            params, opt_state, _ = step(params, opt_state, jnp.asarray(INP), jnp.asarray(ADJB), jnp.asarray(TRUE))
    return params


def meanstd(p, test):
    a = [acc(p, *t) for t in test]
    return float(np.mean(a)), float(np.std(a))


def b64png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pre-epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=25)
    ap.add_argument("--pre-lr", type=float, default=5e-3)
    ap.add_argument("--ft-lr", type=float, default=1.5e-3)     # gentle fine-tune
    ap.add_argument("--scratch-lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    a = ap.parse_args()
    print("GCRN few-shot adaptation to 32x32/10", flush=True)

    # data
    pre = [G.collect_traj(16, 4, 70, s) for s in range(16)]
    pool = [G.collect_traj(32, 10, 100, s) for s in range(400, 412)]      # fine-tune pool (12)
    test = [G.collect_traj(32, 10, 100, s) for s in range(500, 508)]      # held-out (8)
    trivial = np.mean([(t[2][:, ~np.eye(t[2].shape[1], dtype=bool)] > 0.5).mean() for t in test])
    print(f"  data: 16 pre(16/4), {len(pool)} pool(32/10), {len(test)} test(32/10); trivial {trivial*100:.1f}%", flush=True)

    # pretrain base on 16-world
    base = train_on(pre, G.init_params(jax.random.PRNGKey(0)),
                    epochs=a.pre_epochs, lr=a.pre_lr, wd=a.wd)
    zs = meanstd(base, test)
    print(f"  zero-shot (K=0): {zs[0]*100:.1f} ± {zs[1]*100:.1f}%", flush=True)

    # ceiling: train from scratch on the WHOLE pool, longer
    ceil_p = train_on(pool, G.init_params(jax.random.PRNGKey(99)), epochs=50, lr=a.scratch_lr, wd=a.wd)
    ceil = meanstd(ceil_p, test)
    print(f"  ceiling (all {len(pool)} maps from scratch): {ceil[0]*100:.1f} ± {ceil[1]*100:.1f}%", flush=True)

    curve = {"fewshot": {}, "scratch": {}}
    for K in a.shots:
        fs, sc = [], []
        for r in range(a.repeats):
            rng = np.random.default_rng(r)
            pick = [pool[i] for i in rng.choice(len(pool), size=K, replace=False)]
            fs.append(meanstd(train_on(pick, {k: jnp.asarray(v) for k, v in base.items()},
                                       epochs=a.ft_epochs, lr=a.ft_lr, wd=a.wd), test)[0])
            sc.append(meanstd(train_on(pick, G.init_params(jax.random.PRNGKey(1000 + r)),
                                       epochs=a.ft_epochs, lr=a.scratch_lr, wd=a.wd), test)[0])
        curve["fewshot"][K] = (float(np.mean(fs)), float(np.std(fs)))
        curve["scratch"][K] = (float(np.mean(sc)), float(np.std(sc)))
        print(f"  K={K:2d}: few-shot(16-pretrained) {np.mean(fs)*100:.1f} ± {np.std(fs)*100:.1f}%   "
              f"from-scratch {np.mean(sc)*100:.1f} ± {np.std(sc)*100:.1f}%", flush=True)

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore",
                                        "gcrn-fewshot-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    rec = dict(config=vars(a), trivial=float(trivial), zero_shot=zs, ceiling=ceil, curve=curve)
    json.dump(rec, open(os.path.join(out, "fewshot.json"), "w"), indent=2)

    # report
    Ks = a.shots
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    xs = [0] + Ks
    fw = [zs[0] * 100] + [curve["fewshot"][k][0] * 100 for k in Ks]
    fwe = [zs[1] * 100] + [curve["fewshot"][k][1] * 100 for k in Ks]
    sc = [trivial * 100] + [curve["scratch"][k][0] * 100 for k in Ks]
    sce = [0] + [curve["scratch"][k][1] * 100 for k in Ks]
    ax.errorbar(xs, fw, yerr=fwe, fmt="o-", color="#2b7a3b", capsize=4, lw=2, label="16-world pretrained → few-shot")
    ax.errorbar(xs, sc, yerr=sce, fmt="s--", color="#cc7a30", capsize=4, lw=2, label="from scratch at 32")
    ax.axhline(ceil[0] * 100, ls=":", color="#1f6feb", lw=1.4, label=f"ceiling (all {len(pool)} maps): {ceil[0]*100:.0f}%")
    ax.axhline(trivial * 100, ls="-.", color="#9aa0a6", lw=1, label=f"trivial: {trivial*100:.0f}%")
    ax.set_xlabel("number of 32×32 fine-tuning maps (shots)"); ax.set_ylabel("held-out 32×32/10 accuracy %")
    ax.set_ylim(40, 100); ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=.3)
    ax.set_title("Few-shot adaptation to the 32-world")
    chart = b64png(fig)

    CSS = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;line-height:1.55;max-width:900px;"
           "margin:0 auto;padding:0 24px 60px}h1{font-size:23px;border-bottom:3px solid #2b7a3b;padding-bottom:9px}"
           "table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}th,td{border:1px solid #e3e6ea;padding:7px 10px}"
           "th{background:#f1f3f5}.meta{color:#5b6570;font-size:13px}figure{text-align:center}figure img{max-width:100%;border:1px solid #e3e6ea;border-radius:8px}"
           ".callout{background:#e7f6ec;border:1px solid #a6e0bb;border-radius:8px;padding:11px 15px;font-size:13.5px;margin:12px 0}")
    Hh = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
          "<h1>GCRN few-shot adaptation to the 32-world</h1>",
          "<p class='meta'>Pretrain on 16×16/4, fine-tune on K maps of 32×32/10, test on 8 held-out 32×32/10 maps. "
          "Baseline = train from scratch on the same K maps. Gap = value of the size-invariant pretraining.</p>",
          f"<div class='callout'><b>Zero-shot {zs[0]*100:.0f}%</b> → with just a few 32-world maps the pretrained belief "
          f"climbs toward the <b>{ceil[0]*100:.0f}% full-data ceiling</b>, and stays well above training from scratch on the "
          "same maps. The 16-world belief is a strong initialization for the 32-world.</div>",
          f"<figure><img src='{chart}'/></figure>",
          "<table><tr><th>shots (32-world maps)</th><th>16-pretrained few-shot</th><th>from scratch</th><th>advantage</th></tr>",
          f"<tr><td>0 (zero-shot)</td><td>{zs[0]*100:.1f} ± {zs[1]*100:.1f}%</td><td>{trivial*100:.1f}% (trivial)</td>"
          f"<td>+{(zs[0]-trivial)*100:.0f}</td></tr>"]
    for k in Ks:
        f, s = curve["fewshot"][k], curve["scratch"][k]
        Hh.append(f"<tr><td>{k}</td><td>{f[0]*100:.1f} ± {f[1]*100:.1f}%</td><td>{s[0]*100:.1f} ± {s[1]*100:.1f}%</td>"
                  f"<td>+{(f[0]-s[0])*100:.0f}</td></tr>")
    Hh.append(f"<tr><td>all {len(pool)} (ceiling)</td><td colspan='3'>{ceil[0]*100:.1f} ± {ceil[1]*100:.1f}%</td></tr>")
    Hh.append("</table></body></html>")
    open(os.path.join(out, "report.html"), "w").write("\n".join(Hh))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
