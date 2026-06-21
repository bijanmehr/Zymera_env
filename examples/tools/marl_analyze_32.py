"""Emergent-behavior analysis for a trained 32x32 comm-coverage MARL policy.

Quantifies: coverage & connectivity over time, formation (chain vs blanket via
max-dist / mean degree), division of labour (per-agent unique coverage share),
and whether the policy actually USES the degree signal (ablate the degree channels
at eval -> does behavior change?). Saves a figure + a JSON of stats.

  PYTHONPATH=examples/lib python examples/tools/marl_analyze_32.py <variant_dir> <variant> <out_png> <out_json>
"""
import base64
import io
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
cc.GRID = 32; cc.N_AGENTS = 10; cc.MAX_STEPS = 100; cc.COMM_R = 5
import zymera
from marl_independent import IndependentAC
from marl_ctde import CTDE_AC
from marl_frontier_attn import FrontierAttnAC

MODELS = {"independent": IndependentAC, "CTDE": CTDE_AC, "frontier": FrontierAttnAC}
ENV_KW = dict(grid_h=32, grid_w=32, n_agents=10, comm_r=5, vis_r=3, sense_r=3,
              n_obstacles=30, spawn_radius=2, degree_channels=True)


def _ncomp(d, r):
    adj = (d <= r) & ~np.eye(d.shape[0], dtype=bool)
    n = d.shape[0]; seen = np.zeros(n, bool); c = 0
    for s in range(n):
        if seen[s]:
            continue
        c += 1; stk = [s]; seen[s] = True
        while stk:
            u = stk.pop()
            for v in range(n):
                if adj[u, v] and not seen[v]:
                    seen[v] = True; stk.append(v)
    return c


def rollout_stats(env, model, seeds=(0, 1, 2)):
    def policy(o, k):
        return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)
    covs, conns, ncs, mds, degs = [], [], [], [], []
    dol = []
    for s in seeds:
        tr = zymera.rollout(env, policy, cc.MAX_STEPS, jax.random.PRNGKey(s))
        st = tr["world"]
        wall = np.asarray(st.wall[-1]); free = int((~wall).sum())
        team = np.asarray(st.team_explored); pos = np.asarray(st.pos)
        covs.append(team[:, :, :][..., :, :].reshape(team.shape[0], -1).sum(-1) / free)
        T = pos.shape[0]
        cc_t, nc_t, md_t, dg_t = [], [], [], []
        for t in range(T):
            d = np.max(np.abs(pos[t][:, None] - pos[t][None]), -1)
            off = ~np.eye(env.n_agents, dtype=bool)
            cc_t.append(1.0 if _ncomp(d, cc.COMM_R) == 1 else 0.0)
            nc_t.append(_ncomp(d, cc.COMM_R))
            md_t.append(float(np.where(off, d, 0).max()))
            dg_t.append(float(((d <= cc.COMM_R) & off).sum(-1).mean()))
        conns.append(np.array(cc_t)); ncs.append(np.array(nc_t))
        mds.append(np.array(md_t)); degs.append(np.array(dg_t))
        # division of labour: per-agent share of cells it covered alone this step (approx via explored_by)
        eb = np.asarray(st.explored_by[-1])              # (N,H,W) each agent's accumulated vision
        owned = eb.sum((1, 2)).astype(float)
        dol.append(owned / max(owned.sum(), 1))
    return dict(
        cov=np.mean([c[-1] for c in covs]),
        cov_t=np.mean(np.stack(covs), 0),
        conn=np.mean([c.mean() for c in conns]),
        conn_t=np.mean(np.stack(conns), 0),
        ncomp_t=np.mean(np.stack(ncs), 0),
        maxd_t=np.mean(np.stack(mds), 0),
        meandeg_t=np.mean(np.stack(degs), 0),
        dol_gini=float(_gini(np.mean(np.stack(dol), 0))),
    )


def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(np.cumsum(x)) / np.cumsum(x)[-1]) / n)


def main():
    vdir, variant, out_png, out_json = sys.argv[1:5]
    env = cc.make_env(**ENV_KW)
    model = eqx.tree_deserialise_leaves(str(Path(vdir) / "checkpoint.eqx"),
                                        MODELS[variant](env, jax.random.PRNGKey(0)))
    full = rollout_stats(env, model)

    # degree-signal usage: feed CONSTANT degree channels (ablate the signal) -> does cov/conn change?
    env_ab = cc.make_env(**{**ENV_KW, "degree_channels": False})
    # reuse same weights only if channel count matches; instead test sensitivity by
    # zeroing the 2 degree channels inside a wrapper policy
    def ablated_logits(o):
        o2 = o.at[:, -2:].set(0.0)
        return model.logits(o2)

    class _Wrap:
        def logits(self, o):
            return ablated_logits(o)
    ab = rollout_stats(env, _Wrap())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    T = np.arange(len(full["cov_t"]))
    ax[0].plot(T, full["cov_t"] * 100, color="#2b7a3b", label="coverage")
    ax[0].plot(T, full["conn_t"] * 100, color="#1f6feb", label="connected")
    ax[0].axhline(90, ls=":", color="#2b7a3b", alpha=.6)
    ax[0].set_title("coverage & connectivity over time"); ax[0].set_xlabel("step"); ax[0].set_ylim(0, 100)
    ax[0].legend(fontsize=8)
    ax[1].plot(T, full["ncomp_t"], color="#d9480f"); ax[1].set_title("# comm components (1 = whole swarm)")
    ax[1].set_xlabel("step"); ax[1].set_ylim(0, env.n_agents)
    ax[2].plot(T, full["meandeg_t"], color="#7048e8", label="mean degree")
    ax[2].plot(T, full["maxd_t"], color="#999", label="max pairwise dist")
    ax[2].axhline(cc.COMM_R, ls=":", color="#999", alpha=.6); ax[2].set_title("formation: degree & stretch")
    ax[2].set_xlabel("step"); ax[2].legend(fontsize=8)
    fig.suptitle(f"{variant} 32×32 — emergent behavior")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    stats = dict(
        variant=variant,
        coverage=full["cov"], connectivity=full["conn"], dol_gini=full["dol_gini"],
        final_ncomp=float(full["ncomp_t"][-1]), final_meandeg=float(full["meandeg_t"][-1]),
        final_maxd=float(full["maxd_t"][-1]),
        ablate_degree_cov=ab["cov"], ablate_degree_conn=ab["conn"],
        degree_signal_effect_cov=full["cov"] - ab["cov"],
        degree_signal_effect_conn=full["conn"] - ab["conn"],
    )
    Path(out_json).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
