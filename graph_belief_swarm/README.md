# Graph-Belief Swarm — prototype

Host-side (numpy) prototype of the graph-belief swarm design
(`../../graph-belief-swarm-whitepaper.html`). Built on the zymera `GridEnv` +
`Sensor` for world physics and partial observability; runs on the project
`.venv`. v1 is deterministic (no learning yet).

## Run

```bash
# from the zymera_env project root, with the project venv
PYTHONPATH=. .venv/bin/python -m graph_belief_swarm.run \
    --grid 16 --agents 4 --steps 70 --comm-r 5 --obstacle-density 0.05
```

Compares a single agent vs the connected swarm (two strategies), prints a
metrics table, and **checkpoints the run** to
`../experiments/graph_belief_swarm/<timestamp>/`:
`config.json`, `metrics.json`, `curves.npz`, per-condition `frames_*.npz`
(per-step positions / comm graph / ownership) and PNGs. Reload with
`persist.load_run(run_dir)` to regenerate data + visualizations later.

## Modules

| file | role |
|---|---|
| `belief.py` | per-agent relational belief (observed/visited maps; `merge` = graph-union on contact) |
| `compass.py` | **Guidance** — deterministic goal scorer (`gain − cost − crowding`, centroid anchor + tether) |
| `controller.py` | **Control** — BFS path-follow + collision shield + hard connectivity shield |
| `coverage_sweep.py` | **engineered** connected boustrophedon strip-sweep (high *visited* coverage) |
| `swarm.py` | drives the zymera env; runs the **Emergence** coupling (comm graph -> merge + claims) |
| `metrics.py` | comm graph, connectivity, division-of-labor |
| `persist.py` | checkpoint / reload runs (config, metrics, curves, frames) |
| `viz.py` | territory map + coverage curves |

## Status / findings (v1, deterministic — no learning yet)

Two notions of "coverage", both at **16×16 / 4 agents / 70 steps**:

- **Mapped (sensed) coverage** — the belief swarm maps **~99–100%** of the grid
  while ~89% connected (radius-3 sensors see far). Natural exploration metric.
- **Visited (stepped-on) coverage** — the connected strip-sweep
  (`strategy="sweep"`) reaches **100% (open) / ~90–92% (5% obstacles)** while
  **96–100% connected**, ~4× a single agent, redundancy ~1.2×. Dense (10%)
  obstacles in only 70 steps stay ~78% (detours + walled-off cells) — the regime
  where a *learned* policy (cf. prior PPO 86–91%) earns its keep.

Engineering notes that mattered:
- A **hard** connectivity guardrail + a *greedy* compass **gridlocks** (no
  coordinated migration, ~5% coverage); a **soft tether** keeps a belief-compass
  swarm cohesive *and* mobile.
- The sweep needs **lockstep-with-wait** advancing (advance one column/step but
  pause when an agent detours) and a **pre-formed vertical-line spawn** — without
  those it wastes half its ticks or chases a moving target.
- The greedy belief-compass alone re-treads (visited ~38%); the strip-sweep is
  the efficient *coverage* controller. The learned compass is the path to
  efficient coverage *and* emergent coordination together.

## Next steps

1. **Learned compass** — replace the greedy/engineered controllers with a policy
   (team objective + structured symmetry-breaker, not entropy); target >90%
   visited under dense obstacles where the hand-coded sweep falls short.
2. Connectivity-preserving controller that allows coordinated migration (fixes
   the hard-guardrail gridlock).
3. Data/viz: `persist.load_run` + saved frames support animations and territory
   maps from any checkpoint.
