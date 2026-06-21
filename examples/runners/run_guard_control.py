"""Control: collision-masking ONLY (no connectivity mechanism) at comm_r=3, to isolate
how much of the guardrail's coverage edge is the connectivity guardrail vs collision masking."""
import statistics as stt, sys
from pathlib import Path
import equinox as eqx, jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
cc.N_AGENTS=3; cc.GRID=10
import _marl_core as core, _frontier_core as fc
from marl_frontier_comm import FrontierCommAttnAC as M
STEPS,N,COMM,CELLS=55,3,3,100
cc.MAX_STEPS=STEPS; core.T=STEPS; fc.T=STEPS; core.ITERS=200
ENV=dict(grid_h=10,grid_w=10,n_agents=N,comm_r=COMM,spawn_radius=2)
SHAPE=dict(w_shape1=1.0,w_shape3=2.0)
def evalp(env,vdir):
    m=eqx.tree_deserialise_leaves(str(vdir/"checkpoint.eqx"),M(env,jax.random.PRNGKey(0)))
    tr=jax.vmap(lambda k: fc._rollout_masked(env,m,STEPS,k,None))(jax.random.split(jax.random.PRNGKey(7),32))
    st=tr["world"]; cov=float((st.team_explored[:,-1].reshape(32,-1).sum(-1)/CELLS).mean())
    d=jnp.max(jnp.abs(st.pos[...,:,None,:]-st.pos[...,None,:,:]),-1)
    reach=jax.vmap(jax.vmap(jax.vmap(cc._reach)))(d<=COMM); cf=float((reach.sum(-1).max(-1)==N).mean())
    return cov,cf
base=core.new_run_dir(); print(f"CONTROL collision-only · comm_r={COMM} · {base}",flush=True)
res=[]
for sd in [1,2]:
    env=cc.make_env(w_cov=1.0,**SHAPE,**ENV)
    b=fc.run(env,M,f"collonly-s{sd}",outdir=base,seed=sd,target_cov=None,patience=999,min_iters=999,mask_collisions=True,guard_conn=False)
    res.append(evalp(env,Path(b["dir"])))
print(f"  ===> collision-only  cov {stt.mean([r[0] for r in res])*100:.1f}%  conn-frac {stt.mean([r[1] for r in res])*100:.0f}%  seeds={[round(r[0]*100,1) for r in res]}",flush=True)
