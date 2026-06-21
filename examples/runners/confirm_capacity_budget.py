"""CONFIRM the bottleneck before committing to the GRU spend. Three hypotheses for
why coverage stalls ~57% (dev) / 69% (16x16):
  Q1 CAPACITY  — net too small/simple        -> fix = bigger/deeper net
  Q2 BUDGET    — 10% step-slack too tight     -> fix = more steps/agents (or accept high-80s)
  H3 MEMORY    — memoryless policy re-treads  -> fix = recurrence / memory (the GRU)

Disambiguate with two sweeps on the dev world (pure coverage: no collision, no
connectivity, from scratch):
  CAPACITY arm: D in {32,64,128} at fixed 35 steps  -> does width move coverage? (Q1)
  BUDGET   arm: D=64 at {35,55,80} steps             -> does slack move coverage? (Q2 vs H3)

THE KEY READ (budget arm): slots = N*(steps+1).
  * If MORE budget lifts the SAME memoryless net toward ~95% (waste stays low) ->
    the net CAN tile; it was BUDGET-bound (Q2), NOT memory-bound. No GRU needed.
  * If coverage plateaus while slack balloons (waste fraction RISES with budget) ->
    the memoryless policy re-treads regardless -> MEMORY is the real ceiling (H3).

    python -u examples/confirm_capacity_budget.py
"""
import sys
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
CELLS, N = 100, 3

import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
import marl_frontier_comm as mfc   # noqa: E402

SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
core.ITERS = 200   # train toward the CEILING, not the rate (so we measure capacity, not speed)


def set_budget(steps):
    cc.MAX_STEPS = steps
    core.T = steps     # gae's arange reads core.T at call time
    fc.T = steps       # _frontier_core.make_train rollout length + obs[:, :T] slice


def set_capacity(d, h):
    mfc.D = d
    mfc.HEADS = h


def n_params(env):
    m = mfc.FrontierCommAttnAC(env, jax.random.PRNGKey(0))
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))


def cov_eval(env, steps, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"),
                                    mfc.FrontierCommAttnAC(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, steps, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / CELLS).mean())
    slots = N * (steps + 1)
    waste = 1.0 - (cov * CELLS) / slots
    return cov, waste, slots


base = core.new_run_dir()
print(f"CONFIRM capacity vs budget · dev 10x10 / 3 agents · {base}\n", flush=True)

# (tag, D, heads, steps, arm)
CONFIGS = [
    ("D32-s35", 32, 4, 35, "cap"),
    ("D64-s35", 64, 4, 35, "cap/REF"),
    ("D128-s35", 128, 8, 35, "cap"),
    ("D64-s55", 64, 4, 55, "bud"),
    ("D64-s80", 64, 4, 80, "bud"),
]
rows = []
for tag, d, h, steps, arm in CONFIGS:
    set_capacity(d, h)
    set_budget(steps)
    env = cc.make_env(w_cov=1.0, **SHAPE, **DEV)
    npar = n_params(env)
    b = fc.run(env, mfc.FrontierCommAttnAC, f"conf-{tag}", outdir=base, seed=1,
               target_cov=None, patience=999, min_iters=999, mask_collisions=False)
    cov, waste, slots = cov_eval(env, steps, Path(b["dir"]))
    rows.append((tag, arm, d, h, steps, npar, cov, waste, slots))
    print(f"  >> {tag} [{arm}]: cov {cov*100:.1f}%  waste {waste*100:.0f}%  "
          f"slots {slots}  params {npar/1000:.0f}k", flush=True)

print("\n========= CAPACITY vs BUDGET confirmation =========", flush=True)
print(f"{'config':<10}{'arm':<9}{'D':>4}{'h':>3}{'steps':>6}{'params':>9}{'cov':>8}{'waste':>8}{'slack%':>8}")
for tag, arm, d, h, steps, npar, cov, waste, slots in rows:
    slack = 100 * (slots - CELLS) / CELLS
    print(f"{tag:<10}{arm:<9}{d:>4}{h:>3}{steps:>6}{npar/1000:>7.0f}k{cov*100:>7.1f}%{waste*100:>7.0f}%{slack:>7.0f}%")
print("\nReads: CAP arm flat across D => capacity NOT the bottleneck (Q1=no).", flush=True)
print("       BUD arm cov->~95% as steps grow => budget-bound (Q2=yes, no GRU needed).", flush=True)
print("       BUD arm cov plateaus while waste RISES => memoryless re-tread ceiling (H3, GRU needed).", flush=True)
print(f"artifacts: {base}", flush=True)
