# GOAL: Decentralized Emergent Shared-Exploration Swarm

**Mission.** In the `zymera_env` project (its env + `.venv`), build a **fully decentralized**
swarm that performs **emergent shared exploration** — drive every grid cell's *certainty* to
≥1, as **evenly** as possible, within a **strict timecap**, while the swarm keeps its **comm
graph connected by its own choice** — and run the experiment suite that proves it works and
**maps the coverage↔connectivity trade-off.**

## Non-negotiable constraints
- **Decentralized only.** Each agent acts solely on its **own local KB + messages from
  in-range neighbors**. No central controller, no global map, no ground-truth access, no
  central schedule.
- **Connectivity by choice, never by force.** Maintained via a **graduated reward penalty**
  (rises as a link weakens, big at the edge, huge on break) — **no action-masking guardrail.**
- **New clean folder:** `zymera_env/swarm_explore/`. **Do not touch `RedWithinBlue/`.** The old
  `graph_belief_swarm` sweep is a **non-emergent reference baseline only** — never the solution.
- Clean slate; ignore prior prototype numbers.

## The agent — robot = ID + four sections around a local KB
- **Perception:** transient local sensing within radius `r` (camera/radar); used in the moment,
  **not** auto-saved, not an action.
- **Communication:** dumb radio; send/receive minimal messages with neighbors in range `x`
  (`x > r`). Message = `{id, t, pos, intent, last-10 operated}`. **No message from a neighbor =
  link lost**; link signal/freshness **is** the connectivity sense.
- **Local KB:** curated, persistent memory (known certainty field + neighbor tracks + claims).
  **Only Mission writes it.** Merge incoming counts by **max**.
- **Mission:** the only agency; **sequential** operate; runs entirely off the KB. A **learned
  uncertainty-compass policy** — **scale/agent-count-invariant** (attention over KB cells + a
  variable neighbor set, local/relative coords) — picks a **target cell** (low certainty, not
  claimed by a lower-ID neighbor, keeps it connected), emits the target, broadcasts intent.
  **ID-priority: smaller ID goes first** breaks ties.
- **Control:** simple point-to-point driving to the target.

## Certainty field & objective
- Each cell holds an **operation count = certainty**; starts at **0** (unknown).
  **Operate = +1 to current cell, one op/tick.** Optional **forget-rate** decays certainty
  (**v1: forget = 0**; forget > 0 = persistent-monitoring stretch). Merge counts by **max**.
- **Objective:** every cell ≥ 1, as **even** as possible, within the timecap.

## Reward / fitness
- **Dense, diminishing** marginal certainty gain per operate (≈ `1/√(count+1)`) so fresh cells
  pay most → coverage **and** evenness emerge; **not flat** (no cell-farming).
- **Contribution-weighted** (each agent credited by its marginal lift to the *team* field) →
  division of labor and global connectivity fall out of the team result.
- **Graduated connectivity penalty** (approach → edge → break).

## Training
- Learn the compass with the codebase's **RL (PPO/CTDE + ValueNorm**, centralized *training*,
  decentralized *execution*). Add **low-dim evolution** to tune reward weights (hybrid).
  **Curriculum:** start small (8×8 / 2–3 agents, loose→strict timecap) → scale up;
  **domain-randomize** size/agents/obstacles. It's a **Dec-POMDP** — expect non-stationarity;
  lean on parameter-sharing, CTDE, value-norm.

## Scale & success
- **Target:** 16×16, 4 agents, **< 100 timesteps** (ideal ≈ min-coverage-time + ~20%).
- **Success =** full coverage (all ≥ 1), even distribution, connected **by choice**, fully
  decentralized, **beats a single agent decisively within the timecap**, **approaches the
  lawnmower** ceiling, and **trained-small generalizes** to bigger grids / more agents.

## Experiments (leaned — only genuine unknowns)
1. **Reference frame:** single agent (must *fail* the timecap), random walk (floor),
   lawnmower (ceiling).
2. **Headline:** full learned swarm @ 16×16/4/<100 — coverage, evenness, connectivity,
   ×single, vs lawnmower.
3. **Tension sweep (core experiment):** connectivity-weight × comm-range → the
   coverage/evenness ↔ connectivity **Pareto frontier**.
4. **Emergence + scaling:** division-of-labor & evenness metrics (territory map + field
   animation); train-small → **zero-shot** bigger/more-agents.
- *Light:* learned compass vs greedy heuristic.

## Deliverables
- Clean runnable package in `zymera_env/swarm_explore/`, on the `.venv`.
- **Build order:** Tier-0 first (env + certainty field + metrics + baselines — runs with no
  learning, validates the rig) → learned policy + training → experiments 2–4.
- **Checkpoints + metrics + curves + frames saved** (for later data/viz).
- An **honest report** (numbers, the Pareto curve, what's genuinely emergent, what's open).

## Honesty guardrails
- **Never** substitute a centralized/omniscient trick to hit a number — lawnmower is a
  reference only.
- If "connected AND +20%" proves infeasible, **report the real frontier** — don't fake it.
- **Verify by running**; report real numbers; clearly flag whatever didn't finish.
- Stay decentralized at **every** step; build incrementally.

**If time runs short, priority order:** Tier-0 (env+baselines+metrics) → headline →
tension sweep → emergence/scaling.

## Objective (measurable)

**Optimize — per-episode team fitness `J` (maximize):**

```
J = Σ_(operate events) 1/sqrt(count_before + 1)   −   λ_conn · Σ_t P_conn(t)
```
- 1st term = total **diminishing certainty gain**; its maximum ⇔ every cell covered, evenly.
- `P_conn(t)` = graduated connectivity penalty (0 when safely connected, ramps near comm
  range, large spike when any agent loses all links).
- Per-agent credit = that agent's **marginal contribution** to `J` (difference reward).
- `λ_conn` = the coverage↔connectivity knob to sweep (Experiment 3).

**Measure — exact definitions (16×16, 4 agents, cap = 100 steps, `F` = # free cells):**
- `coverage` = |{cells with count ≥ 1}| / F
- `T_cov`    = first step at which coverage = 1.0 (else = cap)
- `ρ` (connectivity) = fraction of steps the comm graph is a single component
- `evenness` = 1 − Gini(counts over free cells)
- `redundancy` = total operations / F
- `decentralization` = structural pass/fail (each action depends only on own KB + in-range msgs)

**Targets — what success aims at (aspirational; the deliverable is the measured frontier,
not a faked corner):**
| metric | target |
|---|---|
| coverage within cap | ≥ 0.98 (ideally 1.0) |
| `T_cov` | ≤ ~77 (≈ min 64 + 20%); hard ≤ 100 |
| connectivity `ρ` | ≥ 0.90, no permanent fragmentation |
| beats single agent | swarm coverage@100 ≥ 2× single's (single must NOT finish) |
| approaches ceiling | `T_cov` ≤ 1.3 × lawnmower's |
| evenness | ≥ 0.75 (Gini ≤ 0.25) |
| generalization | zero-shot ≥ 0.85 × in-distribution coverage at 32×32 / 8 agents |

**Honest note:** "`T_cov` ≤ +20% **AND** `ρ` ≥ 0.90" may be an infeasible corner. If it is,
report the achieved point on the coverage↔connectivity Pareto curve — do not force it.

