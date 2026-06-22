"""Coordination diagnostic — FIXED 100 steps, agents scaled with world size, the honest protocol.
Measures whether the swarm actually COORDINATES (divides the map) or just floods it, via:
  - coverage / connectivity / mission-safety, and
  - REDUNDANCY = Σ per-agent coverage / unique coverage (1.0 = perfect division; higher = overlap).
Plus an agent-sweep at 32² (does adding agents divide the work, or just pile on overlap?).
Current controllers only (heuristic, learned-A, learned-B) — the baseline the redesign must beat.

  PYTHONPATH=. .venv/bin/python -m experiments.coordination_test
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from swarm_explore.relay_mission import rollout_mission, make_theta_policy, heuristic_policy
from swarm_explore.core import random_wall

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RM = os.path.join(ROOT, "experiments", "runs", "relay-mission")
OUT = os.path.join(ROOT, "experiments", "runs", "coordination-test"); os.makedirs(OUT, exist_ok=True)
STEPS, SEEDS = 100, 3
SIZES = [(16, 4), (20, 5), (24, 6), (28, 8), (32, 10), (40, 16)]
SWEEP = [(32, k) for k in (6, 10, 14, 18)]      # agent sweep at fixed world


def load(name):
    p = os.path.join(RM, name)
    return np.load(p) if os.path.exists(p) else None


def run(label, th, hard, g, n):
    pol = heuristic_policy if th is None else make_theta_policy(th, n - 2)
    acc = np.zeros(4)
    for s in range(SEEDS):
        m = rollout_mission(g, g, random_wall(g, g, 0.05, s), n, 5, 3, STEPS, s, pol,
                            backstop=hard, decay=(1.0 if hard else 0.97))
        acc += [m["delivered_cov"], m["connectivity"], m["safety_viol"], m["redundancy"]]
    acc /= SEEDS
    return dict(policy=label, grid=g, n=n, cov=round(acc[0], 3), conn=round(acc[1], 3),
                safety=round(acc[2], 3), redundancy=round(acc[3], 2))


def main():
    tA, tB = load("theta_A.npy"), load("theta_B.npy")
    pols = [("heuristic", None, False), ("learned-A", tA, True), ("learned-B", tB, False)]
    log = open(os.path.join(OUT, "coordination.log"), "w")

    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    results = {"scaled": [], "sweep_32": []}
    say(f"COORDINATION DIAGNOSTIC · fixed {STEPS} steps · {SEEDS} seeds · redundancy 1.0=perfect division")
    say("--- scaled agents (constant ~density) ---")
    for (g, n) in SIZES:
        for label, th, hard in pols:
            if th is None or th is not None:
                r = run(label, th, hard, g, n); results["scaled"].append(r)
                say(f"  {g}²/N{n} {label:>10}: cov {r['cov']*100:3.0f}%  conn {r['conn']*100:3.0f}%  "
                    f"safety {r['safety']:.3f}  redundancy {r['redundancy']}")
        json.dump(results, open(os.path.join(OUT, "coordination.json"), "w"), indent=2)
    say("--- agent sweep at 32² (does adding agents DIVIDE or just OVERLAP?) ---")
    for (g, n) in SWEEP:
        r = run("learned-B", tB, False, g, n); results["sweep_32"].append(r)
        say(f"  32²/N{n:<2} learned-B: cov {r['cov']*100:3.0f}%  conn {r['conn']*100:3.0f}%  redundancy {r['redundancy']}")
        json.dump(results, open(os.path.join(OUT, "coordination.json"), "w"), indent=2)
    say("=== done ===")


if __name__ == "__main__":
    main()
