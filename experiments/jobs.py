"""Experiment jobs for the overnight driver — the SUPER-BLUE pipeline.

Goal: one perception network + one size-invariant cooperative controller, trained
across a distribution of (grid, N), evaluated zero-shot on held-out scales, with a
GIF + HTML report at the end. Phases run in MANIFEST order (B/C produce artifacts
that D/E/F consume):

  A belief-multiscale   train the size-invariant GCRN belief across scales
  B train-super-blue    ES-train ONE controller; fitness averaged over (grid,N)  ← the agent
  C pareto-wall         parallel wall-boundary phase diagram (feasibility map)
  D generalize          eval the trained controller at train + HELD-OUT scales
  E render-gifs         GIF rollouts of the trained controller at several scales
  F assemble-report     one REPORT.html with curves, heatmaps, tables, GIFs

Each job:  def job(ctx, **knobs); records numbers via ctx.metric(). The MANIFEST gives
a 'smoke' (tiny, CPU) and 'gpu' (full) knob set per job.
"""
from __future__ import annotations
import collections
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "experiments", "runs")
CONTROLLER = os.path.join(RUNS, "train-super-blue", "controller_theta.npy")


# ============================================================ A · multi-scale belief
def belief_multiscale(ctx, scales, epochs, n_train, n_test, batch, steps):
    """Train ONE size-invariant GCRN on a distribution of (grid, N); measure transfer at each scale."""
    import jax, jax.numpy as jnp, optax
    from swarm_explore import gcrn as G

    train, test = {}, {}
    for gi, (grid, n) in enumerate(scales):
        tr = [G.collect_traj(grid, n, steps, 1000 * gi + s) for s in range(n_train)]
        te = [G.collect_traj(grid, n, steps, 9000 + 1000 * gi + s) for s in range(n_test)]
        train[n] = tuple(jnp.asarray(np.stack([t[k] for t in tr])) for k in range(3))
        test[n] = tuple(jnp.asarray(np.stack([t[k] for t in te])) for k in range(3))
        ctx.log(f"collected {grid}x{grid}/N={n}: {n_train} train + {n_test} test trajectories")

    p = G.init_params(jax.random.PRNGKey(0))
    opt = optax.adamw(5e-3, weight_decay=0.01); st = opt.init(p)

    def bloss(p, I, A, Tr):
        return jnp.mean(jax.vmap(G.loss_fn, in_axes=(None, 0, 0, 0))(p, I, A, Tr))

    @jax.jit
    def step(p, st, I, A, Tr):
        l, g = jax.value_and_grad(bloss)(p, I, A, Tr)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    def evaluate():
        return {f"acc@{n}": round(float(np.mean([G.accuracy(p, test[n][0][j], test[n][1][j], test[n][2][j])
                                                 for j in range(test[n][0].shape[0])])), 3) for n in test}

    rng = np.random.default_rng(0); history = []
    for ep in range(epochs):
        for n in train:
            I, A, Tr = train[n]; B = I.shape[0]
            idx = rng.choice(B, size=min(batch, B), replace=False)
            p, st, _ = step(p, st, I[idx], A[idx], Tr[idx])
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            acc = evaluate(); history.append(dict(epoch=ep, **acc))
            ctx.log(f"epoch {ep}: " + " ".join(f"{k}={v}" for k, v in acc.items()))

    for k, v in evaluate().items():
        ctx.metric(**{k: v})
    json.dump(history, open(os.path.join(ctx.dir, "history.json"), "w"), indent=2)
    np.savez(os.path.join(ctx.dir, "belief_params.npz"), **{k: np.asarray(v) for k, v in p.items()})


# ============================================================ B · train the super-blue controller
def train_super_blue(ctx, train_scales, gens, pop, sigma, alpha, w_conn, cost_relay, eval_seeds, steps):
    """OpenAI-ES on the 145-param relay/frontier controller. Fitness is averaged over a
    DISTRIBUTION of (grid, N) every generation -> a single size-invariant cooperative policy.
    The population x scales x seeds fitness evals fan out across CPU cores."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from swarm_explore import arch3
    from experiments._es_worker import eval_one

    theta = arch3.init_theta(0).astype(np.float64)
    rng = np.random.default_rng(0)
    workers = min(mp.cpu_count(), 24)
    ctx.log(f"ES: {gens} gens x {2*pop} pop x {eval_seeds} seeds over scales {train_scales} on {workers} cores")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))

    def evaluate(cands):
        """cands: list of (tid, theta) -> {tid: (fit, cov, conn, coh, relay)} seed/scale-averaged."""
        units = [(tid, th, g, n, 5, steps, s, max(1, n - 2), w_conn, cost_relay)
                 for (tid, th) in cands for (g, n) in train_scales for s in range(eval_seeds)]
        acc = collections.defaultdict(list)
        for tid, fit, cov, conn, coh, relay, _viol in pool.map(eval_one, units, chunksize=2):
            acc[tid].append((fit, cov, conn, coh, relay))
        return {tid: tuple(float(np.mean([a[i] for a in v])) for i in range(5)) for tid, v in acc.items()}

    history = []
    try:
        for g in range(gens):
            eps = rng.normal(0, 1, (pop, theta.size)); eps = np.concatenate([eps, -eps], 0)
            res = evaluate([(k, theta + sigma * eps[k]) for k in range(len(eps))])
            F = np.array([res[k][0] for k in range(len(eps))])
            A = (F - F.mean()) / (F.std() + 1e-8)
            theta = theta + alpha * (eps.T @ A) / (len(eps) * sigma)
            cur = evaluate([("cur", theta)])["cur"]
            history.append(dict(gen=g, fit=round(cur[0], 3), cov=round(cur[1], 3),
                                conn=round(cur[2], 3), coh=round(cur[3], 3), relay=round(cur[4], 3)))
            ctx.log(f"gen {g:2d}: fit {cur[0]:.3f}  cov {cur[1]*100:4.0f}%  conn {cur[2]*100:3.0f}%  "
                    f"coh {cur[3]*100:3.0f}%  relay {cur[4]*100:3.0f}%")
    finally:
        pool.shutdown()

    np.save(CONTROLLER, theta)
    json.dump(history, open(os.path.join(ctx.dir, "train_history.json"), "w"), indent=2)
    last = history[-1]
    ctx.metric(gens=gens, cov=last["cov"], conn=last["conn"], coh=last["coh"], relay=last["relay"])


# ============================================================ C · wall-boundary phase diagram
def pareto_wall(ctx, grids, agents, comm_rs, lams, seeds, steps):
    """Sweep λ × grid × N × comm_r × seeds (compass-only) -> where single-component connectivity
    stays feasible as the world scales. Parallel across CPU cores (engine pinned to CPU)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from experiments._pareto_worker import run_one
    configs = [(lam, g, n, cr, s, steps)
               for lam in lams for g in grids for n in agents for cr in comm_rs for s in range(seeds)]
    workers = min(mp.cpu_count(), 24)
    ctx.log(f"{len(configs)} configs across {workers} CPU cores (engine on CPU)…")
    agg = collections.defaultdict(list); done = 0
    raw = open(os.path.join(ctx.dir, "pareto_raw.jsonl"), "w")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run_one, configs, chunksize=4):
            lam, g, n, cr, cov, conn, coh = r
            agg[(lam, g, n, cr)].append((cov, conn, coh)); done += 1
            raw.write(json.dumps(r) + "\n"); raw.flush()
            if done % max(1, len(configs) // 20) == 0:
                ctx.log(f"  {done}/{len(configs)} configs")
    raw.close()
    out = []
    for (lam, g, n, cr), v in sorted(agg.items()):
        cov, conn, coh = (float(np.mean([x[i] for x in v])) for i in range(3))
        out.append(dict(lam=lam, grid=g, n=n, comm_r=cr, cov=round(cov, 3), conn=round(conn, 3), coh=round(coh, 3)))
    json.dump(out, open(os.path.join(ctx.dir, "pareto.json"), "w"), indent=2)
    conns = [o["conn"] for o in out]; feasible = [o for o in out if o["conn"] >= 0.95]
    ctx.metric(cells=len(out), best_conn=round(max(conns, default=0.0), 3),
               worst_conn=round(min(conns, default=0.0), 3), feasible_cells=len(feasible))
    ctx.log(f"wrote {len(out)} (λ,grid,N,comm_r) cells; {len(feasible)} reach ≥95% connectivity")


# ============================================================ D · generalization (held-out scales)
def generalize(ctx, train_scales, holdout_scales, seeds, steps):
    """Load the trained controller; eval at train scales AND held-out scales it never saw."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from swarm_explore import arch3
    from experiments._es_worker import eval_one
    theta = np.load(CONTROLLER) if os.path.exists(CONTROLLER) else arch3.init_theta(0)
    if not os.path.exists(CONTROLLER):
        ctx.log("no trained controller found — evaluating the untrained init (run train-super-blue first)")
    scales = [("train", g, n) for (g, n) in train_scales] + [("holdout", g, n) for (g, n) in holdout_scales]
    units = [(f"{tag}:{g}/{n}", theta, g, n, 5, steps, s, max(1, n - 2), 0.0, 0.0)
             for (tag, g, n) in scales for s in range(seeds)]
    pool = ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 24), mp_context=mp.get_context("spawn"))
    acc = collections.defaultdict(list)
    try:
        for tid, _fit, cov, conn, coh, _relay, _viol in pool.map(eval_one, units, chunksize=2):
            acc[tid].append((cov, conn, coh))
    finally:
        pool.shutdown()
    out = []
    for (tag, g, n) in scales:
        v = acc[f"{tag}:{g}/{n}"]
        o = dict(regime=tag, grid=g, n=n,
                 cov=round(float(np.mean([a[0] for a in v])), 3),
                 conn=round(float(np.mean([a[1] for a in v])), 3),
                 coh=round(float(np.mean([a[2] for a in v])), 3))
        out.append(o)
        ctx.log(f"{tag:7s} {g}x{g}/N={n}: cov {o['cov']*100:3.0f}%  conn {o['conn']*100:3.0f}%  coh {o['coh']*100:3.0f}%")
    json.dump(out, open(os.path.join(ctx.dir, "generalization.json"), "w"), indent=2)
    ho = [o for o in out if o["regime"] == "holdout"]
    ctx.metric(scales=len(out),
               holdout_cov=round(float(np.mean([o["cov"] for o in ho])), 3) if ho else 0.0,
               holdout_coh=round(float(np.mean([o["coh"] for o in ho])), 3) if ho else 0.0)


# ============================================================ E · GIF rollouts
def render_gifs(ctx, scales, steps):
    """Render the trained controller at several scales -> audit GIFs (certainty field + comm graph)."""
    from swarm_explore import arch3, gif
    from swarm_explore.core import random_wall
    theta = np.load(CONTROLLER) if os.path.exists(CONTROLLER) else arch3.init_theta(0)
    gdir = os.path.join(ctx.dir, "gifs"); os.makedirs(gdir, exist_ok=True)
    made = 0
    for (g, n) in scales:
        wall = random_wall(g, g, 0.05, 2)
        m = arch3.run_arch(g, g, wall, n, 5, 3, steps, 2, theta, K=max(1, n - 2), record=True)
        p = os.path.join(gdir, f"superblue_{g}x{g}_n{n}.gif")
        gif.animate_run(m, p, title=f"super-blue · {g}x{g}/N={n} · cov {m['coverage']*100:.0f}% "
                                    f"conn {m['connectivity']*100:.0f}%")
        made += 1
        ctx.log(f"rendered {g}x{g}/N={n}: cov {m['coverage']*100:.0f}% conn {m['connectivity']*100:.0f}% -> {os.path.basename(p)}")
    ctx.metric(gifs=made)


# ============================================================ F · assemble HTML report
def assemble_report(ctx):
    from experiments.report import build_report
    out, n = build_report(RUNS, ROOT)
    ctx.log(f"wrote {out} with {n} sections")
    ctx.metric(sections=n, report=os.path.relpath(out, ROOT))


# ============================================================ G · MARL method tournament (GPU, PPO)
def method_tournament(ctx, methods, scales, seeds, envs, iters):
    """Train each PPO variant (independent / CTDE / joint / frontier-attn) at each scale × seed
    on the GPU. Each run is a fresh subprocess (the MARL core fixes scale via module globals at
    import). Pegs the GPU; per-run failures are isolated; results stream to disk incrementally."""
    import subprocess
    import sys
    outdir = os.path.join(ctx.dir, "marl"); os.makedirs(outdir, exist_ok=True)
    runs = [(m, g, n, st, s) for m in methods for (g, n, st) in scales for s in range(seeds)]
    ctx.log(f"{len(runs)} MARL trainings (method × scale × seed) on the GPU, sequential")
    results = []
    for i, (m, g, n, st, s) in enumerate(runs):
        ctx.log(f"[{i+1}/{len(runs)}] {m} @ {g}x{g}/N={n} seed {s} (envs={envs}, iters={iters})…")
        cmd = [sys.executable, "-m", "experiments.marl_tournament",
               m, str(g), str(n), str(st), str(envs), str(iters), str(s), outdir]
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                               capture_output=True, text=True, timeout=9000)
            rj = os.path.join(outdir, f"result_{m}-{g}x{g}-n{n}-s{s}.json")
            if os.path.exists(rj):
                r = json.load(open(rj)); results.append(r)
                ctx.log(f"    cov {r['cov']*100:.0f}%  conn {r['conn']*100:.0f}%  maxd {r['maxd']:.1f}  (iter {r['iter']})")
            else:
                err = ((p.stderr or p.stdout or "").strip().splitlines() or ["no result"])[-1]
                ctx.log(f"    FAILED {m}: {err[:160]}")
        except Exception as e:
            ctx.log(f"    ERROR {m}: {e}")
        json.dump(results, open(os.path.join(ctx.dir, "tournament.json"), "w"), indent=2)
    by = collections.defaultdict(list)
    for r in results:
        by[(r["grid"], r["n"])].append(r)
    for (g, n), rs in sorted(by.items()):
        top = max(rs, key=lambda r: r["cov"])
        ctx.log(f"{g}x{g}/N={n}: best={top['variant']} cov {top['cov']*100:.0f}% conn {top['conn']*100:.0f}%")
    ctx.metric(runs=len(results), methods=len({r["variant"] for r in results}),
               best_cov=round(max((r["cov"] for r in results), default=0.0), 3))


# ============================================================ H · relay-mission (emergent explorer/relay)
def relay_mission(ctx, regimes, scales, seeds, gens, pop, sigma, alpha, w_conn, decay, steps_map):
    """ES-train ONE size-invariant explorer/relay controller per regime (A=hard backstop /
    coverage-under-connectivity, B=soft / latency-discounted delivery), domain-randomized over
    scales; eval vs the heuristic baseline; render GIFs. The headline coverage↔connectivity +
    emergent-roles experiment. Fitness evals fan out across CPU cores."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from swarm_explore import arch3, gif
    from swarm_explore.relay_mission import rollout_mission, make_theta_policy, heuristic_policy
    from swarm_explore.core import random_wall
    from experiments._relay_es_worker import eval_one

    workers = min(mp.cpu_count(), 24)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    results = {"scales": [list(s) for s in scales], "regimes": {}}
    gdir = os.path.join(ctx.dir, "gifs"); os.makedirs(gdir, exist_ok=True)
    rng = np.random.default_rng(0)
    try:
        for regime in regimes:
            theta = arch3.init_theta(0).astype(np.float64)

            def evaluate(cands, regime=regime):
                units = [(tid, th, regime, g, n, 5, steps_map[g], s, w_conn, decay)
                         for (tid, th) in cands for (g, n) in scales for s in range(seeds)]
                acc = collections.defaultdict(list)
                for r in pool.map(eval_one, units, chunksize=2):
                    acc[r[0]].append(r[1])
                return {tid: float(np.mean(v)) for tid, v in acc.items()}

            ctx.log(f"[{regime}] ES: {gens} gens x {2*pop} pop x {seeds} seeds over {scales} on {workers} cores")
            hist = []
            for gnum in range(gens):
                eps = rng.normal(0, 1, (pop, theta.size)); eps = np.concatenate([eps, -eps], 0)
                fits = evaluate([(k, theta + sigma * eps[k]) for k in range(len(eps))])
                F = np.array([fits[k] for k in range(len(eps))])
                A = (F - F.mean()) / (F.std() + 1e-8)
                theta = theta + alpha * (eps.T @ A) / (len(eps) * sigma)
                cur = evaluate([("cur", theta)])["cur"]; hist.append(round(cur, 4))
                if gnum % max(1, gens // 6) == 0 or gnum == gens - 1:
                    ctx.log(f"[{regime}] gen {gnum:2d}: fit {cur:.3f}")
            np.save(os.path.join(ctx.dir, f"theta_{regime}.npy"), theta)

            perscale = []
            for (g, n) in scales:                       # eval learned θ + heuristic baseline per scale
                E = max(3, seeds); L = np.zeros(5); Hh = np.zeros(5)
                for s in range(E):
                    wall = random_wall(g, g, 0.05, 100 + s); hard = (regime == "A"); dc = 1.0 if hard else decay
                    lm = rollout_mission(g, g, wall, n, 5, 3, steps_map[g], 100 + s, make_theta_policy(theta, n - 2), backstop=hard, decay=dc)
                    hm = rollout_mission(g, g, wall, n, 5, 3, steps_map[g], 100 + s, heuristic_policy, backstop=hard, decay=dc)
                    L += [lm["delivered_cov"], lm["raw_cov"], lm["connectivity"], lm["relay_frac"], np.std(lm["per_agent_relay"])]
                    Hh += [hm["delivered_cov"], hm["raw_cov"], hm["connectivity"], hm["relay_frac"], np.std(hm["per_agent_relay"])]
                L /= E; Hh /= E
                k = ("cov", "raw", "conn", "relay", "role_spread")
                perscale.append(dict(grid=g, n=n,
                                     learned={kk: round(L[i], 3) for i, kk in enumerate(k)},
                                     heuristic={kk: round(Hh[i], 3) for i, kk in enumerate(k)}))
                ctx.log(f"[{regime}] {g}x{g}/N{n}: learned cov {L[0]*100:.0f}% conn {L[2]*100:.0f}% relay {L[3]*100:.0f}% spread {L[4]:.2f} "
                        f"| heur cov {Hh[0]*100:.0f}% conn {Hh[2]*100:.0f}%")
            results["regimes"][regime] = dict(history=hist, per_scale=perscale)

            for (g, n) in [scales[0], scales[-1]]:       # GIFs at small + large scale
                hard = (regime == "A")
                m = rollout_mission(g, g, random_wall(g, g, 0.05, 2), n, 5, 3, steps_map[g], 2,
                                    make_theta_policy(theta, n - 2), backstop=hard, decay=(1.0 if hard else decay), record=True)
                gif.animate_run(m, os.path.join(gdir, f"relay_{regime}_{g}x{g}_n{n}.gif"),
                                title=f"relay-{regime} {g}x{g}/N{n} · cov {m['delivered_cov']*100:.0f}% conn {m['connectivity']*100:.0f}%")
    finally:
        pool.shutdown()

    json.dump(results, open(os.path.join(ctx.dir, "relay_results.json"), "w"), indent=2)
    best = {}
    for regime in regimes:
        big = results["regimes"][regime]["per_scale"][-1]["learned"]
        best[f"{regime}_big_cov"] = big["cov"]; best[f"{regime}_big_conn"] = big["conn"]
    ctx.metric(**best)


# ============================================================ I · relay PPO arm (structural credit) — scaffold
def relay_ppo_arm(ctx, **k):
    ctx.log("SCAFFOLD relay-PPO: the structural-credit arm — per-agent delivered-cell credit split β between")
    ctx.log("the finder explorer and the CRITICAL relay(s) on the delivery path (load-bearing only).")
    ctx.log("Plan: wire role+tool+termination policy into PPO with path-credit routing + articulation detection;")
    ctx.log("tests whether explicit structural credit rescues policy-gradient where AC failed (the ES arm is the de-risked baseline).")
    ctx.metric(status="scaffold")


MANIFEST = [
    {"name": "belief-multiscale", "fn": belief_multiscale,
     "smoke": dict(scales=[(16, 4), (20, 5)], epochs=3, n_train=4, n_test=2, batch=4, steps=40),
     "gpu":   dict(scales=[(16, 4), (20, 5), (24, 6), (28, 8), (32, 10)], epochs=60, n_train=24, n_test=8, batch=32, steps=70)},
    {"name": "train-super-blue", "fn": train_super_blue,
     "smoke": dict(train_scales=[(12, 3), (16, 4)], gens=2, pop=3, sigma=0.15, alpha=0.08, w_conn=0.4, cost_relay=0.15, eval_seeds=1, steps=40),
     "gpu":   dict(train_scales=[(16, 4), (24, 6), (32, 10)], gens=20, pop=12, sigma=0.15, alpha=0.08, w_conn=0.4, cost_relay=0.15, eval_seeds=3, steps=80)},
    {"name": "pareto-wall", "fn": pareto_wall,
     "smoke": dict(grids=[12, 16], agents=[3, 4], comm_rs=[4, 5], lams=[0.0, 1.0], seeds=2, steps=40),
     "gpu":   dict(grids=[16, 20, 24, 28, 32], agents=[4, 6, 8, 10, 12], comm_rs=[4, 5, 6, 8], lams=[0.0, 0.5, 1.0], seeds=8, steps=150)},
    {"name": "generalize", "fn": generalize,
     "smoke": dict(train_scales=[(16, 4)], holdout_scales=[(20, 5)], seeds=2, steps=40),
     "gpu":   dict(train_scales=[(16, 4), (24, 6), (32, 10)], holdout_scales=[(40, 16), (48, 20)], seeds=6, steps=120)},
    {"name": "render-gifs", "fn": render_gifs,
     "smoke": dict(scales=[(16, 4)], steps=40),
     "gpu":   dict(scales=[(16, 4), (24, 6), (32, 10), (40, 16)], steps=120)},
    {"name": "method-tournament", "fn": method_tournament,
     "smoke": dict(methods=["independent"], scales=[(12, 3, 20)], seeds=1, envs=2, iters=2),
     "gpu":   dict(methods=["frontier-attn", "attn", "gattn", "gattn-d4", "gattn-d6"],
                   scales=[(16, 4, 80), (24, 6, 100), (32, 10, 120)], seeds=2, envs=16, iters=200)},
    {"name": "relay-mission", "fn": relay_mission,
     "smoke": dict(regimes=["A", "B"], scales=[(12, 3), (16, 4)], seeds=1, gens=2, pop=3,
                   sigma=0.2, alpha=0.1, w_conn=0.5, decay=0.97, steps_map={12: 40, 16: 50}),
     "gpu":   dict(regimes=["A", "B"], scales=[(16, 4), (24, 6), (32, 10)], seeds=2, gens=25, pop=10,
                   sigma=0.15, alpha=0.08, w_conn=0.5, decay=0.97, steps_map={16: 120, 24: 200, 32: 280})},
    {"name": "relay-ppo-arm", "fn": relay_ppo_arm, "smoke": {}, "gpu": {}},
    {"name": "assemble-report", "fn": assemble_report, "smoke": {}, "gpu": {}},
]
