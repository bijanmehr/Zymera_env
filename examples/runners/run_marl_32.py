"""Train PPO MARL on the 32x32 world to >90% coverage with MINIMAL comm-graph
disruption, using the degree observation signal.

Setting: 32x32 / 10 agents / comm_r 5 / vis_r 3 (sensed coverage) / 100 steps /
~3% obstacles / clustered spawn — matches the swarm_explore 32x32 setting.

Reward recipe for coverage + connectivity:
  - strong PBRS coverage shaping (w_shape1 frontier-seek + w_shape3 team blanket)
  - anti-overlap (spread / division of labour)
  - connectivity via conn_cap = N-1 (a stretched CONNECTED chain scores like a
    clump -> the swarm can spread AND stay linked) + a light degree floor
  - DEGREE CHANNELS in the obs so the policy self-regulates connectivity

Usage (env knobs set as module globals BEFORE importing the core, so T/denominators
are 32x32-correct):
  python examples/runners/run_marl_32.py independent
  python examples/runners/run_marl_32.py CTDE
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

# CLI: variant grid steps envs iters n_agents [warm_ckpt]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "independent"
GRID = int(sys.argv[2]) if len(sys.argv) > 2 else 32
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 160
NAG = int(sys.argv[6]) if len(sys.argv) > 6 else 10
WARM = sys.argv[7] if len(sys.argv) > 7 else ""

# --- set grid knobs BEFORE importing the core (it reads T, denominators) ---
cc.GRID = GRID
cc.N_AGENTS = NAG
cc.MAX_STEPS = STEPS
cc.COMM_R = 5

import _marl_core as core              # noqa: E402  (reads cc.* at import)
core.ENVS = ENVS
core.ITERS = ITERS
import equinox as eqx                  # noqa: E402
import jax                             # noqa: E402
from marl_independent import IndependentAC   # noqa: E402
from marl_ctde import CTDE_AC                 # noqa: E402

MODELS = {"independent": IndependentAC, "CTDE": CTDE_AC}

# obstacle count scales with area (~3%); 16x16->8, 32x32->30
ENV_KW = dict(
    grid_h=GRID, grid_w=GRID, n_agents=NAG, comm_r=5,
    vis_r=3, sense_r=3, n_obstacles=max(8, (GRID * GRID) * 3 // 100), spawn_radius=2,
    w_cov=1.0, w_conn=0.5, conn_cap=NAG - 1, w_coll=4.0,  # LIGHT connectivity (capped=roamer-friendly)
    w_overlap=1.0,                                         # anti-overlap -> spread / division of labour
    w_shape1=1.0, w_shape3=1.0, shape_gamma=0.99,          # PBRS coverage shaping (drives spreading)
    w_degree=0.2, degree_floor=1,                          # gentle local connectivity floor
    degree_channels=True,                                  # the DEGREE SIGNAL in the obs (self-regulate conn)
)


def main():
    env = cc.make_env(**ENV_KW)
    base = core.new_run_dir()
    print(f"run dir: {base}")
    print(f"variant: {VARIANT}  obs_channels: {env.obs_channels}  grid {env.H}x{env.W}  "
          f"N {env.n_agents}  T {cc.MAX_STEPS}  ENVS {core.ENVS}  ITERS {core.ITERS}"
          + (f"  warm={WARM}" if WARM else ""), flush=True)
    label = f"{VARIANT}-deg{GRID}"
    if WARM:
        # warm-start: build model, load weights, then continue training (size-agnostic encoder)
        model = MODELS[VARIANT](env, jax.random.PRNGKey(1))
        model = eqx.tree_deserialise_leaves(WARM, model)
        b = core.run(env, MODELS[VARIANT], label, outdir=base,
                     target_cov=0.97, patience=80, min_iters=60, init_model=model)
    else:
        b = core.run(env, MODELS[VARIANT], label, outdir=base,
                     target_cov=0.91, patience=70, min_iters=min(60, ITERS - 1))
    print(f"\nBEST {VARIANT}: coverage {b['cov']*100:.1f}%  connected {b['conn']*100:.0f}%  "
          f"max-dist {b['maxd']:.1f}  redundancy {b['redun']:.2f}x  (iter {b['iter']})")
    print(f"artifacts: {base}")


if __name__ == "__main__":
    main()
