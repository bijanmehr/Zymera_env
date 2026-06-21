# examples/ — layout

Separated into the reusable library, the experiment scripts, and the render/report tools.
(The base `zymera` package lives at the repo root; everything here builds on it.)

```
examples/
├── lib/        reusable MARL library — imported by everything else
├── runners/    experiment scripts (run_*.py) — one per experiment
└── tools/      render / report / analysis utilities
```

## lib/ — the library
| module | what it is |
|---|---|
| `comm_coverage.py` | the grid comm-coverage env (1×1 visit-coverage, partial comms, all reward terms + the hard connectivity guardrail hooks) |
| `_marl_core.py` | shared PPO core (rollout, GAE, clipped update, run-dir + metrics) |
| `_frontier_core.py` | return-normalized PPO + the masked sampler — collision masking **and** the hard connectivity guardrail (`guard_conn`) |
| `marl_frontier_comm.py` | `FrontierCommAttnAC` — frontier + inter-agent attention actor, central critic |
| `marl_frontier_attn.py` | `FrontierAttnAC` — frontier-attention only |
| `marl_ctde.py` / `marl_independent.py` / `marl_joint.py` | the three original centralization-variant actor-critics |
| `cbf.py` | control-barrier-function terms (λ₂ / collision barriers) |

## runners/ — experiments
One self-contained script per experiment; each builds an env from `lib/`, trains, evals, prints a result.
Run them **from the repo root** so relative paths resolve:

```bash
python examples/runners/run_guardrail.py     # hard connectivity guardrail vs soft penalties
python examples/runners/run_curriculum.py    # spread → constrain curriculum
```

## tools/ — render / report
`render_*.py`, `report_ctde.py`, `_assemble_report*.py`, `_analyze_marl.py` — produce GIFs, HTML reports, analysis.

## results
Training outputs (checkpoints, `metrics.npz`, `config.json`, HTML/GIF) are written to
`experiments/marl/<timestamp>/` at the repo root — kept out of the code tree.
