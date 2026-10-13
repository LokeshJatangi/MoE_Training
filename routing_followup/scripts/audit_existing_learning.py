"""Evidence of expert learning in the original completed T4 checkpoints."""
import argparse,json
from pathlib import Path
import torch
from src.gpt import GPTConfig,TinyGPT
from src.routing_lab import grow

p=argparse.ArgumentParser()
p.add_argument('--evidence',default='../MoE_Clone_Lab-evidence')
p.add_argument('--output',default='analysis/existing_t4_learning.json')
a=p.parse_args();torch.set_num_threads(2)
root=Path(a.evidence);run='clone-first-20261012-133612';base=root/'checkpoints'/run
parent_ck=torch.load(base/'parent.pt',map_location='cpu',weights_only=True)
parent=parent_ck['model'];result={'provenance':'Audit of completed original T4 checkpoints; no new training', 'arms':[]}
for arm in ['hard','warmup','anneal']:
    ck=torch.load(base/f'E12-{arm}.pt',map_location='cpu',weights_only=True)
    model=grow(TinyGPT(GPTConfig(**parent_ck['model_config'])).upcycle(),3)
    ids=ck['optimizer']['param_groups'][0]['params'];names=list(model.named_parameters());assert len(ids)==len(names)
    moments={name:ck['optimizer']['state'].get(pid,{}) for (name,_),pid in zip(names,ids)}
    rows=[]
    for layer in range(4):
        experts=[]
        for e in range(12):
            change=0.;v_sum=0.;m_square=0.;states=0;steps=[]
            for matrix in ['gate','up','down']:
                name=f'blocks.{layer}.ffn.experts.{e}.{matrix}.weight'
                original=f'blocks.{layer}.ffn.experts.{e//3}.{matrix}.weight'
                change+=(ck['model'][name]-parent[original]).square().sum().item()
                state=moments[name]
                if state:
                    assert state['exp_avg'].shape==ck['model'][name].shape
                    states+=1;steps.append(int(state['step']))
                    m_square+=state['exp_avg'].square().sum().item()
                    v_sum+=state['exp_avg_sq'].sum().item()
            experts.append({'expert':e,'family':e//3,'lifetime_routes':int(ck.get('cumulative_routings',ck.get('cumulative_'+'assign'+'ments'))[layer,e]),
                'weight_change_l2':change**.5,'adam_matrices':states,'adam_steps':steps,
                'first_moment_l2':m_square**.5,'second_moment_sum':v_sum,
                'task_gradient_history_evidence':v_sum>0})
        rows.append(experts)
    result['arms'].append({'arm':arm,'layers':rows,
        'zero_lifetime_experts':sum(e['lifetime_routes']==0 for row in rows for e in row),
        'no_task_gradient_history_experts':sum(not e['task_gradient_history_evidence'] for row in rows for e in row),
        'unchanged_experts':sum(e['weight_change_l2']==0 for row in rows for e in row)})
p=Path(a.output);p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,indent=2))
for arm in result['arms']:print({k:v for k,v in arm.items() if k!='layers'})
