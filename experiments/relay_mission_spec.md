# Relay-mission experiment — spec

Goal: **one decentralized brain that self-organizes into explorers and relays**, hits
**>90% delivered coverage with >95% connectivity** across three world sizes, and chooses
*more* relays as the world grows — with the role split *measured* to be emergent.

## Task — "delivered coverage"
A sensed cell counts for the team **only once its finder is path-connected (through relays)
to the home core** (the connected component containing the spawn centroid). Raw vs delivered
coverage are tracked separately; the gap is the cost of bad teamwork.

## Agent (identical copy per agent, decentralized)
3-level brain, belief-conditioned:
- **L1 role**: explorer | relay
- **L2 tool**: explorer{best-frontier · A*-to-unknown · sweep}; relay{go-to-bridge · hold · shadow}
- **L3 execute**: A*/attention, with **learned termination** (when to stop & re-decide)
- conditions on: local view + **belief** (GCRN graph + delivery state + teammate roles) + gossip

## Mission safety (soft penalties; collision the only hard mask)
| role | keep ≥ neighbors | may strand others |
|---|---|---|
| explorer | **1** (a tether) | — |
| relay | **2** (a bridge has two ends) | **0** (release on hand-off) |
Constant across sizes. Resilience phase: explorer floor 1→2.

## Connectivity — hybrid (no pure masking)
- **Lagrangian constraint**: λ auto-climbs until connectivity ≥ 95% (soft-but-binding).
- **connectivity-preserving skills**: tools only propose still-connected goals.
- **thin backstop**: revert only the rare move that would actually split the graph.
→ ~100% connectivity, policy still able to stretch & recover.

## Reward (PPO arm) — one currency, delivered coverage
Per agent, per step:
- **delivered-cell credit**, split: (1−β)→finder explorer, β→**critical** relay(s) on the path
  (load-bearing only; empty bridges earn nothing). **β balances explorer/relay** (start 0.5).
- **team share** w_team≈0.2 (glue).
- **−safety penalty** (floors above).
- **−λ·disconnection** (adaptive).
- **−tiny switch cost** (anti-thrash).
ES arm needs none of this — just **team fitness** = delivered coverage s.t. connectivity.

## Learners — both arms, same architecture/task/metrics
- **ES** (proven for role control; team fitness; de-risked primary) — CPU.
- **PPO** (the new per-agent structural credit; tests if it rescues policy-gradient where AC failed) — GPU.

## Worlds (relay-necessity gradient)
16²/N4, 24²/N6, 32²/N10, comm_r 5. One size-invariant brain, domain-randomized over all three.

## The sweep (what the GPU is for)
- **Spine — coverage↔connectivity Pareto**: connectivity-pressure ε{soft,90,95,99} × scale{16,24,32}
  × design{old-flat-PPO, old-heuristic, new-ES, new-PPO} × seeds. Does the new design's frontier dominate the old's?
- **Rider — emergence**: β{0.3,0.5,0.7} × scale (new-PPO) → relay-fraction-vs-scale curve.
- fixed/ablated: belief on/off, backstop on/off, w_team, w_switch, K_r, comm_r.

## Metrics (prove emergence, not assert)
delivered coverage (primary) · connectivity · raw-vs-delivered gap · **MI(role; behavior)** ·
relay-fraction-vs-scale · relay-chain length · hand-off count · counterfactual relay value · safety violations.

## Outputs
per-size GIFs (watch roles + chains + hand-offs) + HTML report (Pareto plots, emergence curve, baseline table).

## Build order (verified increments)
1. **task engine + heuristic baseline** (validate delivered-coverage, connectivity, safety, credit, GIF) ← first
2. ES arm (team fitness) 3. PPO arm (per-agent credit) 4. sweep harness 5. metrics + report
