"""Belief-wired role-switcher (A+B) at 32x32/10.

Wires the FROZEN size-invariant GCRN belief (gcrn.py) into Stack 2:
  (A) role DECISION  — append belief-derived GLOBAL features to the switcher input:
       estimated global cut-vertex, estimated #components, believed component-fraction,
       belief confidence. (The 7 local-exact features can't express these.)
  (B) relay ACTION   — when relaying, head toward the believed-disconnected group
       (centroid of lost agents' last-known positions) to HEAL the split, instead of
       only holding locally. Falls back to herd (hold) when nothing is lost.

The belief is frozen; only the ES switcher (11-feature policy) is trained, at 32x32/10
with the mission-safety fitness (K=8 -> floor 1 for N=10).

  PYTHONPATH=. .venv/bin/python -m swarm_explore.arch3_belief --gens 8 --pop 6
"""
from __future__ import annotations

import argparse
import base64
import datetime
import glob
import io
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from .core import (action_for, bfs, clustered_spawn, collision_mask, comm_adj,
                   connected_components, evaluate, first_step, is_connected, make_env_world)
from .connect import herd_target
from .swarm import _choose_target, _is_cut_vertex
from .arch3 import _features
from . import gcrn as G

EXP = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore"))
_bstep = jax.jit(G.step_one)

# 11-feature switcher policy (7 local + 4 belief-global)
BFEAT = 11
BHID = 16
_BSH = [(BFEAT, BHID), (BHID,), (BHID, 1), (1,)]
B_NPARAMS = sum(int(np.prod(s)) for s in _BSH)


def init_btheta(seed=0):
    rng = np.random.default_rng(seed)
    # bias the output toward NOT relaying (b2 < 0) so ES starts in the exploring regime,
    # not the saturated all-relay clump basin; small weights keep initial logits modest.
    parts = [rng.normal(0, 0.3, _BSH[0]), np.zeros(_BSH[1]), rng.normal(0, 0.3, _BSH[2]), np.full(_BSH[3], -1.5)]
    return np.concatenate([p.ravel() for p in parts])


def warm_btheta(local_theta):
    """Warm-start the 11-feature policy from the trained 7-feature LOCAL switcher, with the
    4 belief-feature weights = 0. Initial behavior == the local switcher (known-good: ~60/36),
    so ES avoids BOTH the all-relay clump and the never-relay freeze; it learns the belief
    weights as a delta on a sane baseline."""
    from .arch3 import _unflat as lun, SW_FEAT
    W1l, b1l, W2l, b2l = lun(local_theta)              # (7,16),(16,),(16,1),(1,)
    W1 = np.zeros((BFEAT, BHID)); W1[:SW_FEAT] = W1l    # belief-feature rows start at 0
    return np.concatenate([W1.ravel(), b1l.ravel(), W2l.ravel(), b2l.ravel()])


def _bunflat(theta):
    out, k = [], 0
    for s in _BSH:
        nn = int(np.prod(s)); out.append(theta[k:k + nn].reshape(s)); k += nn
    return out


def brelay_prob(theta, feats):
    W1, b1, W2, b2 = _bunflat(theta)
    h = np.tanh(feats @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-float((h @ W2 + b2)[0])))


def load_belief():
    fs = sorted(glob.glob(os.path.join(EXP, "gcrn-*", "gcrn_params.npz")))
    d = np.load(fs[-1])
    return {k: jnp.asarray(d[k]) for k in d.files}


def _belief_features(i, Ahat_i, n):
    """4 GLOBAL features from agent i's predicted full adjacency."""
    A = (Ahat_i > 0.5).astype(float); np.fill_diagonal(A, 0.0)
    comps = connected_components(A); comp_of = {a: c for c in comps for a in c}
    cut = 1.0 if _is_cut_vertex(A, i) else 0.0                 # bridge on the ESTIMATED full graph
    fncomp = (len(comps) - 1) / max(1, n - 1)                  # estimated fragmentation (0 = one component)
    cf = len(comp_of[i]) / n                                  # believed component-fraction (incl. unseen)
    off = ~np.eye(n, dtype=bool)
    conf = float((np.abs(Ahat_i - 0.5) * 2)[off].mean())      # belief confidence (margin)
    return np.array([cut, fncomp, cf, conf])


def _toward(cen, seen, wall, dist):
    """reachable, seen, free cell closest to centroid cen."""
    free = seen & ~wall & np.isfinite(dist)
    ys, xs = np.where(free)
    if len(ys) == 0:
        return None
    k = int(np.argmin((ys - cen[0]) ** 2 + (xs - cen[1]) ** 2))
    return (int(ys[k]), int(xs[k]))


def _belief_relay_target(i, pos, adj, comp, nbr_pos, seen, wall, dist, n):
    """(B) head toward lost agents' last-known centroid to heal a split; else hold (herd)."""
    lost = [nbr_pos[j] for j in range(n) if j != i and j not in comp and j in nbr_pos]
    if lost:
        g = _toward(np.mean(lost, axis=0), seen, wall, dist)
        if g is not None:
            return g
    return herd_target(i, pos, adj, nbr_pos, seen, wall, dist)


def run_arch_belief(H, W, wall, n, comm_r, sense_r, steps, seed, theta, P, *,
                    K=8, hi=0.6, lo=0.4, lam_frontier=1.0, belief_action=True, record=False,
                    stochastic=False):
    """theta=None -> pure compass (no relay) baseline.
    stochastic=True (ES TRAINING): relay ~ Bernoulli(P) so P drives expected behavior
    smoothly -> non-flat ES gradient. stochastic=False (EVAL): hard hysteresis threshold."""
    wall = np.asarray(wall, bool)
    floor = (n - 1) - K
    positions = clustered_spawn(wall, n, comm_r, seed)
    env, world = make_env_world(H, W, wall, positions, sense_r, True)
    kb = [np.zeros((H, W)) for _ in range(n)]; seen = [np.zeros((H, W), bool) for _ in range(n)]
    goals = [None] * n; relaying = np.zeros(n, bool)
    nbr_pos = [dict() for _ in range(n)]; nbr_claim = [dict() for _ in range(n)]
    true_count = np.zeros((H, W)); owner = -np.ones((H, W), int); free_total = int((~wall).sum())
    conn_hits = 0; coh_sum = 0.0; relay_steps = 0; viol_sum = 0; t_cov = steps; frames = []
    Hb = jnp.zeros((n, n, G.HID))                                    # belief hidden state (carried)
    key = jax.random.PRNGKey(seed); rng_s = np.random.default_rng(seed + 12345)
    for t in range(steps):
        pos = np.asarray(world.body.position)
        for i in range(n):
            r, c = int(pos[i, 0]), int(pos[i, 1]); kb[i][r, c] += 1.0
            if true_count[r, c] == 0:
                owner[r, c] = i
            true_count[r, c] += 1.0
        vis = np.asarray(env.sensor.visibility(world))
        for i in range(n):
            seen[i] |= vis[i]
        adj = comm_adj(pos, comm_r)
        if is_connected(adj):
            conn_hits += 1
        comps = connected_components(adj); comp_of = {i: comp for comp in comps for i in comp}
        coh_sum += max(len(c) for c in comps) / n
        for comp in comps:
            if len(comp) > 1:
                cc = np.zeros((H, W)); ss = np.zeros((H, W), bool)
                for i in comp:
                    np.maximum(cc, kb[i], out=cc); ss |= seen[i]
                for i in comp:
                    kb[i] = cc.copy(); seen[i] = ss.copy()
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    nbr_pos[i][j] = (int(pos[j, 0]), int(pos[j, 1])); nbr_claim[i][j] = goals[j]
        deg = adj.sum(1)
        viol_sum += int((deg < floor).sum())
        # ---- frozen belief step (recurrent) ----
        inp, adjb = G.step_inputs(pos, adj, comps, n, comm_r)
        Hb, logits = _bstep(P, Hb, jnp.asarray(inp), jnp.asarray(adjb))
        Ahat = np.asarray(jax.nn.sigmoid(logits))               # (n,n,n) per-perspective predicted adjacency

        actions = np.zeros(n, int)
        for i in range(n):
            src = (int(pos[i, 0]), int(pos[i, 1])); dist, parent = bfs(seen[i] & ~wall, src)
            comp = comp_of.get(i, [i])
            if theta is not None:
                loc = _features(i, pos, adj, comp, kb[i], seen[i], wall, comm_r, K, H, W, n)   # 7 local
                glob = _belief_features(i, Ahat[i], n)                                          # 4 belief-global
                p = brelay_prob(theta, np.concatenate([loc, glob]))
                if stochastic:
                    relaying[i] = bool(rng_s.random() < p) or deg[i] == 0     # Bernoulli(P) -> smooth gradient
                elif relaying[i]:
                    if p < lo and deg[i] >= max(1, floor):
                        relaying[i] = False
                else:
                    if p > hi or deg[i] == 0:
                        relaying[i] = True
            if relaying[i]:
                if belief_action:
                    g = _belief_relay_target(i, pos, adj, comp, nbr_pos[i], seen[i], wall, dist, n)
                else:
                    g = herd_target(i, pos, adj, nbr_pos[i], seen[i], wall, dist)
            else:
                g = goals[i]
                stale = (g is None or src == g or not np.isfinite(dist[g[0], g[1]]) or kb[i][g[0], g[1]] >= 1.0)
                if stale:
                    claims = [nbr_claim[i][j] for j in comp if j < i and nbr_claim[i].get(j) is not None]
                    g = _choose_target(kb[i], seen[i], wall, src, dist, claims, nbr_pos[i], comm_r, lam_frontier)
                goals[i] = g
            if g is not None:
                actions[i] = action_for(src, first_step(parent, src, g))
        relay_steps += int(relaying.sum())
        if record:
            frames.append(dict(pos=pos.copy(), owner=owner.copy(), count=true_count.copy(),
                               adj=adj.copy(), lam=np.zeros(n), goal=list(goals), relay=relaying.copy()))
        actions = collision_mask(actions, pos, wall)
        key, sk = jax.random.split(key)
        _, world, _, _, _ = env.step(world, jnp.asarray(actions, jnp.int32), sk)
        if (true_count >= 1).sum() / free_total >= 1.0 and t_cov == steps:
            t_cov = t

    m = evaluate(true_count, wall, conn_hits, steps, owner, n, t_cov)
    m["largest_comp_frac"] = coh_sum / steps
    m["relay_frac"] = relay_steps / max(1, n * steps)
    m["safety_viol"] = viol_sum / max(1, n * steps)
    m["wall"] = wall; m["frames"] = frames
    return m


def _eval(theta, P, grid, n, seeds, *, K, belief_action=True, stochastic=False):
    cov = conn = coh = sv = rl = 0.0
    for s in seeds:
        from .core import random_wall
        wall = random_wall(grid, grid, 0.05, s)
        m = run_arch_belief(grid, grid, wall, n, 5, 3, 100, s, theta, P, K=K,
                            belief_action=belief_action, stochastic=stochastic)
        cov += m["coverage"]; conn += m["connectivity"]; coh += m["largest_comp_frac"]
        sv += m["safety_viol"]; rl += m["relay_frac"]
    k = len(seeds)
    return dict(cov=cov / k, conn=conn / k, coh=coh / k, sv=sv / k, rl=rl / k)


def b64png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.18)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)               # N=10 -> floor 1
    ap.add_argument("--w-safety", type=float, default=1.0)
    a = ap.parse_args()
    grid, n = 32, 10
    P = load_belief()
    tr = list(range(a.train_seeds)); ho = list(range(50, 56))     # held-out maps
    rng = np.random.default_rng(0)
    lt = os.path.join(EXP, "arch3", "best_theta_gen.npy")
    theta = warm_btheta(np.load(lt)) if os.path.exists(lt) else init_btheta(0)
    print(f"  init: {'warm-start from local switcher' if os.path.exists(lt) else 'random'}", flush=True)

    def fit(th):
        # coverage primary; connectivity bonus; PENALIZE relay-fraction so the degenerate
        # all-relay clump (cov~0) can never win; cohesion dropped (it's maxed by clumping).
        # stochastic=True: Bernoulli(P) relay -> P drives expected behavior -> non-flat gradient.
        m = _eval(th, P, grid, n, tr, K=a.K, stochastic=True)
        return m["cov"] + 0.5 * m["conn"] - 0.2 * m["rl"] - a.w_safety * m["sv"], m

    print(f"ES belief-wired switcher (A+B) @32x32/10  {a.gens}gen x {2*a.pop}pop x {a.train_seeds}seed  K={a.K}", flush=True)
    best = (-1e9, theta.copy(), None)
    for g in range(a.gens):
        eps = rng.normal(0, 1, (a.pop, theta.size)); eps = np.concatenate([eps, -eps], 0)
        F = np.array([fit(theta + a.sigma * e)[0] for e in eps])
        A_ = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + a.alpha * (eps.T @ A_) / (len(eps) * a.sigma)
        f, m = fit(theta)
        print(f"  gen {g:2d}  fit {f:.3f}  cov {m['cov']*100:4.0f}%  conn {m['conn']*100:4.0f}%  "
              f"coh {m['coh']*100:4.0f}%  sv {m['sv']*100:4.1f}%  relay {m['rl']*100:3.0f}%", flush=True)
        if f > best[0]:
            best = (f, theta.copy(), m)
    theta = best[1]

    # comparison ladder on held-out maps
    print("evaluating comparison ladder (held-out maps)...", flush=True)
    from .core import random_wall as _rw
    from .arch3 import run_arch as _run_local
    compass = _eval(None, P, grid, n, ho, K=a.K)
    lt = os.path.join(EXP, "arch3", "best_theta_gen.npy")
    local16 = None
    if os.path.exists(lt):
        lth = np.load(lt); cov = conn = coh = sv = 0.0
        for s in ho:
            m = _run_local(grid, grid, _rw(grid, grid, 0.05, s), n, 5, 3, 100, s, lth, K=a.K)
            cov += m["coverage"]; conn += m["connectivity"]; coh += m["largest_comp_frac"]; sv += m["safety_viol"]
        local16 = dict(cov=cov / len(ho), conn=conn / len(ho), coh=coh / len(ho), sv=sv / len(ho))
        print(f"  local (16-shot) : cov {local16['cov']*100:.0f}% conn {local16['conn']*100:.0f}% coh {local16['coh']*100:.0f}% sv {local16['sv']*100:.1f}%", flush=True)
    belief_AB = _eval(theta, P, grid, n, ho, K=a.K, belief_action=True)
    belief_Aonly = _eval(theta, P, grid, n, ho, K=a.K, belief_action=False)
    print(f"  compass     : cov {compass['cov']*100:.0f}% conn {compass['conn']*100:.0f}% coh {compass['coh']*100:.0f}% sv {compass['sv']*100:.1f}%", flush=True)
    print(f"  belief A    : cov {belief_Aonly['cov']*100:.0f}% conn {belief_Aonly['conn']*100:.0f}% coh {belief_Aonly['coh']*100:.0f}% sv {belief_Aonly['sv']*100:.1f}%", flush=True)
    print(f"  belief A+B  : cov {belief_AB['cov']*100:.0f}% conn {belief_AB['conn']*100:.0f}% coh {belief_AB['coh']*100:.0f}% sv {belief_AB['sv']*100:.1f}%", flush=True)

    out = os.path.join(EXP, "arch3-belief-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "btheta.npy"), theta)
    rec = dict(config=vars(a), compass=compass, local16=local16, belief_Aonly=belief_Aonly, belief_AB=belief_AB)
    json.dump({k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in rec.items()},
              open(os.path.join(out, "belief.json"), "w"), indent=2)

    # GIF + report
    from .core import random_wall
    from .gif import animate_run
    wall = random_wall(grid, grid, 0.05, 51)
    mr = run_arch_belief(grid, grid, wall, n, 5, 3, 100, 51, theta, P, K=a.K, belief_action=True, record=True)
    gp = os.path.join(out, "belief_AB.gif")
    animate_run({"frames": mr["frames"], "wall": wall, "coverage": mr["coverage"], "connectivity": mr["connectivity"]},
                gp, title="belief-wired switcher (A+B) @32x32/10", sense_r=3)

    import matplotlib.pyplot as plt
    rows = [("compass (no relay)", compass)]
    if local16:
        rows.append(("local switcher (16-shot, zero-shot)", local16))
    rows += [("belief, decision-only (A)", belief_Aonly), ("belief A+B", belief_AB)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); x = np.arange(len(rows)); wb = 0.27
    ax.bar(x - wb, [r[1]["cov"] * 100 for r in rows], wb, color="#2b7a3b", label="coverage")
    ax.bar(x, [r[1]["conn"] * 100 for r in rows], wb, color="#1f6feb", label="full-connectivity")
    ax.bar(x + wb, [r[1]["coh"] * 100 for r in rows], wb, color="#9aa0a6", label="cohesion")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8); ax.set_ylim(0, 100)
    ax.set_ylabel("%"); ax.legend(fontsize=8); ax.set_title("Belief-wired switcher @32×32/10 (held-out maps)")
    chart = b64png(fig)
    gif64 = "data:image/gif;base64," + base64.b64encode(open(gp, "rb").read()).decode() if os.path.exists(gp) else ""

    CSS = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;line-height:1.55;max-width:920px;"
           "margin:0 auto;padding:0 24px 60px}h1{font-size:23px;border-bottom:3px solid #1f6feb;padding-bottom:9px}"
           "h2{font-size:18px;margin-top:26px;border-top:1px solid #e3e6ea;padding-top:13px}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}"
           "th,td{border:1px solid #e3e6ea;padding:7px 10px}th{background:#f1f3f5}.meta{color:#5b6570;font-size:13px}figure{text-align:center}"
           "figure img{max-width:100%;border:1px solid #e3e6ea;border-radius:8px}.callout{background:#edf2ff;border:1px solid #bcd0ff;border-radius:8px;padding:11px 15px;font-size:13.5px;margin:12px 0}")
    Hh = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
          "<h1>Belief-wired role-switcher (A+B) @ 32×32 / 10 agents</h1>",
          "<p class='meta'>The frozen size-invariant GCRN belief feeds the ES role-switcher: (A) global graph-criticality "
          "features into the role decision, (B) belief-driven relay action (head toward the believed-disconnected group).</p>",
          f"<figure style='width:80%'><img src='{chart}'/></figure>",
          "<table><tr><th>policy</th><th>coverage</th><th>full-connectivity</th><th>cohesion</th><th>safety-viol</th></tr>"]
    for lbl, m in rows:
        Hh.append(f"<tr><td>{lbl}</td><td>{m['cov']*100:.0f}%</td><td>{m['conn']*100:.0f}%</td>"
                  f"<td>{m['coh']*100:.0f}%</td><td>{m['sv']*100:.1f}%</td></tr>")
    Hh.append("</table>")
    if gif64:
        Hh.append(f"<h2>Belief-wired policy (A+B), seed 51</h2><figure style='width:55%'><img src='{gif64}'/>"
                  "<figcaption>Amber = relay (belief-driven: heading to heal the split), blue = frontier.</figcaption></figure>")
    Hh.append("</body></html>")
    open(os.path.join(out, "report.html"), "w").write("\n".join(Hh))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
