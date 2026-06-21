"""Architecture / learning-stack ablation — 8 variants x 3 seeds, seed-averaged.

Testbed: dev 10x10 / 3 agents / comm_r=3, single-phase COVERAGE only (PBRS shaping
on, NO collision / connectivity → isolates the coverage-learning question), budget
s55 (the regime where the baseline ~72% and efficiency is starting to collapse).
The metric that matters is FRESH-RATE (covered/slots = tiling efficiency), not raw cov.

Forks studied (one change each vs baseline):
  MEM           +own-trail & recency obs channels (5->7)   [memory, cheap GRU proxy]
  CRIT          per-agent critic (per-agent advantage)     [credit assignment]
  noFRONT       ablate spatial frontier attention head     [is frontier-attn real?]
  noCOMM        ablate inter-agent attention head          [is comm-attn real?]
  HIRES         stride-1 encoder (no 4x4 bottleneck)       [spatial precision]
  REVISIT       anti-revisit reward (team_explored)        [reward lever]
  MEM+REVISIT   memory channels + anti-revisit             [complementary combo]

    python -u examples/run_ablation.py
"""
import statistics as stt
import sys
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.GRID = 10
cc.N_AGENTS = 3
cc.COMM_R = 3
cc.SPAWN_RADIUS = 1
DEV = dict(grid_h=10, grid_w=10, n_agents=3, comm_r=3, spawn_radius=1)
CELLS, N, STEPS = 100, 3, 55

import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC as M  # noqa: E402

core.ITERS = 150
cc.MAX_STEPS = STEPS
core.T = STEPS                      # gae arange
fc.T = STEPS                        # rollout length + obs[:, :T]
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2, 3]

VARIANTS = [
    ("B0-baseline",   dict(), dict()),
    ("MEM",           dict(mem_channels=True), dict()),
    ("CRIT-peragent", dict(), dict(per_agent_critic=True)),
    ("noFRONT",       dict(), dict(use_frontier=False)),
    ("noCOMM",        dict(), dict(use_comm=False)),
    ("HIRES",         dict(), dict(hi_res=True)),
    ("REVISIT",       dict(w_revisit=0.3), dict()),
    ("MEM+REVISIT",   dict(mem_channels=True, w_revisit=0.3), dict()),
]


def cov_eval(env, build, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), build(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / CELLS).mean())
    slots = N * (STEPS + 1)
    return cov, cov * CELLS / slots          # coverage, fresh-rate


base = core.new_run_dir()
print(f"ABLATION · dev 10x10/3 · s{STEPS} · {len(VARIANTS)} variants x {len(SEEDS)} seeds · {base}\n", flush=True)

summary = []
for name, ekw, mkw in VARIANTS:
    build = partial(M, **mkw)
    covs, freshes = [], []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, **SHAPE, **ekw, **DEV)
        b = fc.run(env, build, f"{name}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999, mask_collisions=False)
        cov, fresh = cov_eval(env, build, Path(b["dir"]))
        covs.append(cov)
        freshes.append(fresh)
    mc = stt.mean(covs)
    sc = stt.pstdev(covs) if len(covs) > 1 else 0.0
    mf = stt.mean(freshes)
    summary.append((name, mc, sc, mf, covs))
    print(f"  ===> {name:16s} cov {mc*100:5.1f}% ±{sc*100:3.1f}  fresh {mf*100:4.0f}%  "
          f"seeds={[round(c*100, 1) for c in covs]}", flush=True)

print("\n========= ABLATION (dev s55, 3-seed mean) =========", flush=True)
b0 = next(s for s in summary if s[0].startswith("B0"))
print(f'{"variant":<16}{"cov":>9}{"±":>6}{"fresh":>8}{"Δcov vs B0":>12}')
for name, mc, sc, mf, covs in summary:
    dlt = (mc - b0[1]) * 100
    print(f"{name:<16}{mc*100:>8.1f}%{sc*100:>5.1f}{mf*100:>7.0f}%{dlt:>+11.1f}")
print(f"\n(B0 baseline {b0[1]*100:.1f}%; effects under ~5 pts are within seed noise)", flush=True)
print(f"artifacts: {base}", flush=True)
