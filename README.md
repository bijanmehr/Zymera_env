# Zymera_env

**JAX-native heterogeneous multi-agent environment for adversarial-robustness research** — plus the
decentralized swarm-exploration engine built on top of it. Like `gymnax`, but multi-agent: you write the
training loop, Zymera gives you the environment, the rollout primitive, and the visualisation.

This is the **engine layer** of the broader *Zymera* research program on covert micro→macro misbehaviour
and mission resilience in cooperative swarms (codename *RedWithinBlue* — a covert adversary inside a
cooperative team).

## About

Zymera_env is **active research code** — part of an ongoing project on adversarial robustness and
resilience in cooperative multi-agent / swarm systems. It is **work in progress**: interfaces,
experiments, and structure change frequently as the research evolves, so expect things to move (and
occasionally break) between commits.

This repository **supersedes** the earlier project
[**RedWithinBlue**](https://github.com/bijanmehr/RedWithinBlue), which is **no longer used** — the work
has been migrated and consolidated here.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"          # core + tests + plotting
# GPU users: install JAX with CUDA first, then the line above
```

Requires Python ≥ 3.10. Core deps: JAX, Equinox, optax, chex, numpy, msgspec, pyyaml.

## Five-line quick start

```python
import jax, zymera
from zymera import viz   # headless core; all drawing lives in zymera.viz

env  = zymera.make("empty-v0", grid_h=8, grid_w=8, n_agents=4)
traj = zymera.rollout(env, zymera.random_policy, 32, jax.random.PRNGKey(0))
viz.render_gif(traj, "rollout.gif")        # or viz.iso_gif / viz.make_report
```

## Repository layout

| path | what it is |
|---|---|
| `zymera/` | the env **library** (env + rollout + sensor + viz); the installable package |
| `swarm_explore/` | the **research engine** — decentralized connectivity-aware coverage (compass + ES role-switcher + recurrent graph belief) |
| `graph_belief_swarm/` | an earlier deterministic Compass / Controller / Emergence prototype |
| `examples/` | reusable PPO/MARL library, one-script runners, render/report tools |
| `tests/` | pytest suite |
| `docs/` | environment & network tutorials (HTML) |

> `experiments/` (run outputs — checkpoints, GIFs, reports) is **git-ignored** and regenerated locally.

## Environment contract

```python
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(state, action, key)

env.n_agents      # int
env.obs_dim       # int
env.n_actions     # int
```

`state` is a JAX pytree — passes through `jit` / `vmap` / `lax.scan` unchanged. `action` is `(N,) int32`.
`info` is a dict of raw env signals (`explored`, `step_count`, `seen_by`, `comm_graph`) read in your
reward / mission functions. The interface is mission-agnostic: gym-style `reset` / `step` over an opaque
state pytree. The built-in `"empty-v0"` env exposes the schema above; custom envs subclass `zymera.Env`.

## Public API (10 names)

| name | what |
| --- | --- |
| `zymera.make(name, **kw)` | construct a registered env |
| `zymera.list_envs()` | registered env names, sorted |
| `zymera.register_env(name, factory)` | add a custom env |
| `zymera.Env` | base class for custom envs |
| `env.reset(key)` → `(obs, state)` | reset, gym-style |
| `env.step(state, action, key)` → `(obs, state, r, done, info)` | step, gym-style |
| `zymera.rollout(env, policy, n_steps, key)` | `lax.scan` rollout, vmap-friendly |
| `zymera.random_policy(obs, key)` | one example policy |
| `zymera.viz.render_gif` / `iso_gif` | trajectory → flat or isometric gif |
| `zymera.viz.make_report(traj, path)` | self-contained HTML report (gif + charts) |
| `zymera.Sensor(radius, occlusion)` | partial-obs reading + `seen_by` |

Two CLIs ship for quick use:

```bash
zymera-viz  --grid-h 8 --grid-w 8 --n-agents 4 --output rollout.gif
zymera-keys --grid-h 10 --grid-w 10 --n-agents 1
```

## Obstacles + partial observability

Obstacles are real walls the env enforces; a `Sensor` gives each agent a local, optionally occluded view
(opt-in — `sense_radius=0` keeps the env pure-JAX, so `rollout` / `jit` / `vmap` are unaffected):

```python
env = zymera.make("empty-v0", grid_h=16, grid_w=16, n_agents=1,
                  n_obstacles=20, sense_radius=3, occlusion=True)
_, state = env.reset(key)
reading = env.sensor.observe_one(state, 0)   # {(r,c): blocked}, radius-3 disk
```

## The swarm-exploration engine (`swarm_explore/`)

A host-side engine for decentralized cooperative coverage that stays connected over a limited
communication range. Agents share a **certainty field** (operate +1 per cell, diminishing reward,
max-merge gossip within a component) and run three modules:

- **Compass** — deterministic frontier exploration (uncertainty − crowding).
- **Role-switcher** — an evolution-strategies policy over graph-criticality features that picks *relay*
  vs *frontier*, with a mission-safety degree-budget and hysteresis.
- **Graph belief** — a recurrent message-passing GNN (per-node GRU + bilinear decoder) estimating the
  global comm graph from partial info; **size-invariant** (trains at one scale, transfers to another).

```bash
PYTHONPATH=. .venv/bin/python -m swarm_explore.gcrn          # train the size-invariant belief
PYTHONPATH=. .venv/bin/python -m swarm_explore.arch3_belief  # belief-wired role-switcher
```

## Examples & tests

```bash
python examples/lawn_mower.py --grid 12 --obstacles 8 --output lawn.gif   # heuristic baseline
python examples/learn_template.py                                          # REINFORCE scaffold
pytest tests/ -q
```

## Documentation

- [`docs/tutorial-env.html`](docs/tutorial-env.html) — the environment (worlds, agents, observation, sensing)
- [`docs/tutorial-net.html`](docs/tutorial-net.html) — a network to control the agent (the learning loop)

```bash
cd docs && python3 -m http.server 8000   # http://localhost:8000
```

## Reference

A hands-on companion to the MARL textbook — POSG / Dec-POMDP, observation `o_i`, policy `π_i`, policy
gradient, parameter sharing, CTDE all map directly.

> Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer.
> *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches.* MIT Press, 2024.
> <https://www.marl-book.com>

## License

Research code; license TBD by the author.
