import torch, numpy as np
from hydra import compose, initialize_config_dir
from eval_lib.context import CFG_DIR, DATASETS_DIR, EDIT_MODEL_WEIGHTS_DIR
from connectpt.routes_generator import utils as lrnu
from connectpt.routes_generator.improvement_learning import (
    load_raw_graphs_and_lc_routes, make_improvement_batch, rollout_lc_improvement)
from connectpt.routes_generator.bee_colony import get_adjustment_degrees
from connectpt.routes_generator.torch_utils import get_batch_tensor_from_routes

dev=torch.device('cpu'); TARGET=0.2; NR=12; MINL=8; MAXL=15; STEPS=15; TRIM=1
D=DATASETS_DIR/"mixed_conn_adj_n50_r12_len8_15"
graphs, seed_routes = load_raw_graphs_and_lc_routes(D/"raw_graphs_subset.pkl", D)
N=len(graphs); perm=torch.randperm(N, generator=torch.Generator().manual_seed(0))
val=perm[int(0.9*N):].tolist()
print("graphs",N,"val",len(val),flush=True)
def mk():
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name="ppo_50nodes.yaml", overrides=["model=bestsofar_feb2023_trim","model.route_generator.kwargs.serial_halting=True","++run_name=e","++experiment.logdir=null"])
def build(wf,dis):
    _,_,_,cobj,m=lrnu.process_standard_experiment_cfg(mk(), run_name_prefix="e_", weights_required=False)
    cobj.ignore_stops_oob=True; cobj.set_enabled_components(disabled_components=dis or None)
    m.load_state_dict(torch.load(EDIT_MODEL_WEIGHTS_DIR/wf, map_location=dev)); m.to(dev).eval(); return m,cobj
MODELS={"conn_adj_mixed":("improvement_conn_adj_mixed.pt",["demand","route"]),
        "train_improvement":("improvement_lc_improvement_trim.pt",["connectivity"])}
for name,(wf,dis) in MODELS.items():
    m,cobj=build(wf,dis); cw=cobj.get_weights(dev); adjs=[]
    for gi in val:
        gb,rb=make_improvement_batch(graphs,seed_routes,torch.tensor([gi]),dev,training=False,target_n_routes=NR)
        with torch.no_grad():
            out=rollout_lc_improvement(m,cobj,gb,rb,MINL,MAXL,greedy=True,cost_weights=cw,max_route_edit_steps=STEPS,max_trim_actions_per_route=TRIM)
        imp=get_batch_tensor_from_routes(out[0].routes,dev,max_route_len=rb.shape[-1])
        nr=min(imp.shape[1],rb.shape[1]); w=min(imp.shape[-1],rb.shape[-1])
        adjs.append(get_adjustment_degrees(imp[:, :nr,:w], rb[:, :nr,:w], cobj.symmetric_routes, gap=0.1, mode="paper").mean().item())
    a=np.array(adjs)
    print(f"{name:18s} mean={a.mean():.3f} std={a.std():.3f} min={a.min():.3f} max={a.max():.3f} | >target: {(a>TARGET).sum()}/{len(a)} | >=target: {(a>=TARGET).sum()}/{len(a)}",flush=True)
    print("   per-graph adj:", [round(x,3) for x in a.tolist()],flush=True)
print("DONE",flush=True)
