# swarm_explore — Decentralized Emergent Shared-Exploration Swarm

Implements `GOAL.md`. Fully decentralized: each agent acts only on its own local
KB + messages from in-range neighbors. Built on the zymera `GridEnv` + `Sensor`
(host-side); runs on the project `.venv`.

## Run

```bash
# from the zymera_env project root, with the project venv
PYTHONPATH=. .venv/bin/python -m swarm_explore.run \
    --grid 16 --agents 4 --steps 100 --comm-r 5 --sense-r 3 --obstacle-density 0.05 --seeds 5
```

Prints a metrics table (single agent / random walk / lawnmower ceiling / decentralized
swarm), runs the **coverage↔connectivity tension sweep** over `lambda_conn`, and saves
`summary.json` + figures (territory map, certainty heatmap, Pareto) to
`../experiments/swarm_explore/<timestamp>/`.

## Modules
| file | role |
|---|---|
| `core.py` | comm graph, connectivity, BFS stepping, env construction, metrics |
| `swarm.py` | decentralized agent loop — Perception/Comms/KB/Mission(uncertainty compass)/Control |
| `baselines.py` | single agent, random walk (floor), lawnmower (omniscient ceiling — *reference only*) |
| `viz.py` | territory map, certainty heatmap, coverage↔connectivity Pareto |
| `run.py` | the experiment suite + save |

## Design (v1)
- **Certainty field:** each cell holds an operation count (= certainty), starts 0.
  Operate = +1 to the current cell, one op/tick. Merge by **max**. Reward = diminishing
  gain `1/√(count+1)` (so fresh cells pay most → coverage + evenness).
- **Mission = deterministic uncertainty compass:** head to the lowest-certainty reachable
  known-free cell; de-conflict by lower-ID neighbors' claims; **connectivity by choice**
  (soft penalty in target scoring when a target would strand the agent — *no hard mask*).
  `lambda_conn` is the coverage↔connectivity knob.
- **Decentralized throughout:** no central controller/map/schedule; ground-truth field is
  tracked for *metrics only*. The learned compass (RL/evolution) is the next build step;
  this v1 establishes the rig + baselines + the tension curve.

## Results (16×16, 4 agents, 100 steps, 5% obstacles, 5 seeds)

| condition | coverage | T_cov | connectivity | evenness | redundancy | DoL-gini |
|---|---|---|---|---|---|---|
| single agent | 40.2% | never | 100% | 0.39 | 1.03 | — |
| random walk | 35.6% | never | 54% | 0.21 | 4.69 | 0.15 |
| lawnmower (ceiling) | 97.1% | never | 97% | 0.61 | 1.70 | 0.04 |
| **swarm (decentralized)** | **98.9%** | **94.8** | **72%** | **0.76** | **1.67** | **0.04** |

Coverage↔connectivity Pareto (λ_conn): 0.0 → 100%/44%conn/T79.4 · 0.5 → 99.7%/70%/94.4 ·
1.0 → 98.9%/72%/94.8 · 2–4 → ~99%/71%/94.8 (connectivity saturates ~72%).

**Met:** coverage ≥98% (even, 0.76); beats a single agent 2.5× (single never finishes);
matches/edges the omniscient lawnmower while fully decentralized; emergent balanced
division of labor (DoL-gini 0.04, redundancy 1.67×).
**Missed:** connectivity tops out ~72% (target 90%). The "+20% AND ≥90%-connected" corner
does not exist for the deterministic compass — λ=0 is 100%@+20% but 44% connected; raising
λ buys connectivity only to a ~72% ceiling at a speed cost. This is the honest Pareto.

**Next:** the learned compass is the lever to push the connectivity frontier up while
keeping coverage — the deterministic soft-choice compass saturates at ~72%.

Run that produced these: `../experiments/swarm_explore/20260617-012915/`.
