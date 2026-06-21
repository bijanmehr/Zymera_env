"""Persistence: checkpoint each experiment run so data and visualizations can
be regenerated later without re-running. Mirrors the zymera
experiments/<name>/<timestamp>/ layout.

Each run directory contains:
  config.json     — all run parameters (reproducible)
  metrics.json    — per-condition scalar metrics
  curves.npz      — per-condition coverage-over-time arrays
  frames_<cond>.npz — per-step positions, comm graph, ownership (for animation /
                      territory maps), for every condition that recorded frames.
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-").lower()


def save_run(base_dir: str, config: dict, named_results) -> str:
    """Persist one run. `named_results` = list of (label, result_dict)."""
    run_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    summary, curves = {}, {}
    for label, r in named_results:
        summary[label] = {k: float(r[k]) for k in
                          ("coverage", "observed_coverage", "connectivity",
                           "redundancy", "gini", "shield") if k in r}
        curves[_slug(label)] = np.asarray(r["cov_curve"], dtype=np.float32)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(run_dir, "curves.npz"), **curves)

    for label, r in named_results:
        if r.get("frames"):
            fr = r["frames"]
            np.savez(
                os.path.join(run_dir, f"frames_{_slug(label)}.npz"),
                positions=np.stack([f["pos"] for f in fr]).astype(np.int16),    # (T, N, 2)
                comm=np.stack([f["adj"] for f in fr]).astype(np.int8),          # (T, N, N)
                owner_seq=np.stack([f["owner"] for f in fr]).astype(np.int16),  # (T, H, W)
                owner=np.asarray(r["owner"], np.int16),                         # (H, W) final
                wall=np.asarray(r["wall"], np.int8),                            # (H, W)
                owned=np.asarray(r["owned"], np.int32),                         # (N,)
            )
    return run_dir


def load_run(run_dir: str) -> dict:
    """Load a saved run back (config, metrics, curves, and any frame archives)."""
    out = {"dir": run_dir}
    with open(os.path.join(run_dir, "config.json")) as f:
        out["config"] = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        out["metrics"] = json.load(f)
    out["curves"] = dict(np.load(os.path.join(run_dir, "curves.npz")))
    out["frames"] = {}
    for fn in os.listdir(run_dir):
        if fn.startswith("frames_") and fn.endswith(".npz"):
            out["frames"][fn[len("frames_"):-len(".npz")]] = dict(np.load(os.path.join(run_dir, fn)))
    return out
