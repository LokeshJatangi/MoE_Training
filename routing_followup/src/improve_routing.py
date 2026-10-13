"""Controlled hard-routing repair study, with expert learning telemetry."""
import argparse,copy,hashlib,json,platform,time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .gpt import TinyGPT,GPTConfig
from .routing_lab import set_policy
from .learning_metrics import (build_arm,experts,gradient_norms,snapshot_experts,decay_removed_updates,
    fixed_hidden_probes,function_diversity,weight_drift,optimizer_expert_state,ablate_to_initial)


def sample(data,gen,n,context):
    start=torch.randint(len(data)-context,(n,),generator=gen)
    seq=data[start[:,None]+torch.arange(context+1)]
    return seq[:,:-1],seq[:,1:]


@torch.no_grad()
def evaluate(model,batch,device):
    model.eval();loss=0.;counts=None
    for x,y in zip(batch[0].split(4),batch[1].split(4)):
        logits,aux=model(x.to(device))
        loss+=F.cross_entropy(logits.flatten(0,1),y.to(device).flatten(),reduction='sum').item()
        current=torch.stack([a['counts'].cpu() for a in aux])
        counts=current if counts is None else counts+current
    return {'ce':loss/batch[1].numel(),'counts':counts}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/t4.json');p.add_argument('--device',default='cuda')
    p.add_argument('--data',default='data/tinystories_lab.npz');p.add_argument('--parent',default='artifacts/parent.pt')
    p.add_argument('--output',required=True);p.add_argument('--toy',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);cfg=json.loads(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(cfg['seed'])
    if a.toy:
        parent=TinyGPT(GPTConfig(vocab_size=13,context=8,layers=1,width=16,heads=4,intermediate=32)).upcycle()
        with torch.no_grad():parent.blocks[0].ffn.router.weight.zero_()
        tokens=torch.arange(13).repeat(400);data={k:tokens.clone() for k in ['train','validation','test']}
        provenance={'toy':True,'meaning':'Execution/telemetry smoke only; not language-generalization evidence.'}
    else:
        manifest=json.loads(Path('data_specs/parent_manifest.json').read_text())
        digest=hashlib.sha256(Path(a.parent).read_bytes()).hexdigest()
        if digest!=manifest['parent_sha256']:raise ValueError('Parent checkpoint checksum mismatch')
        saved=torch.load(a.parent,map_location='cpu',weights_only=True)
        parent=TinyGPT(GPTConfig(**saved['model_config'])).upcycle();parent.load_state_dict(saved['model'])
        expected=json.loads(Path('data_specs/manifest.json').read_text())
        actual=json.loads(Path(a.data).with_suffix('').joinpath('manifest.json').read_text())
        for key in ['revision','tokenizer_sha256','vocab_size','token_counts']:
            if actual[key]!=expected[key]:raise ValueError('Dataset mismatch: '+key)
        with np.load(a.data) as arrays:
            data={k:torch.from_numpy(arrays[k].astype('int64')) for k in ['train','validation','test']}
            if int(arrays['vocab_size'])!=parent.cfg.vocab_size:raise ValueError('Vocabulary mismatch')
        provenance={'parent_sha256':digest,'parent_source':manifest,'data_manifest':actual,
                    'npz_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest()}
    cfg['model']=vars(parent.cfg)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    env={'device':a.device,'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
         'torch':str(torch.__version__),'python':platform.python_version(),'precision':'FP32',
         'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src').glob('*.py')}}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    evals={k:sample(v,torch.Generator().manual_seed(9000+i),cfg['eval_sequences'],parent.cfg.context)
           for i,(k,v) in enumerate(data.items())}
    calibration=sample(data['train'],torch.Generator().manual_seed(9999),cfg['calibration_sequences'],parent.cfg.context)
    ablation=(evals['validation'][0][:cfg['ablation_sequences']],evals['validation'][1][:cfg['ablation_sequences']])
    parent.to(a.device).eval();hidden=fixed_hidden_probes(parent,calibration[0],a.device)
    with torch.no_grad():parent_logits=parent(calibration[0][:4].to(a.device))[0].cpu()
    parent.cpu();summary=[]
    checkpoints=Path('checkpoints')/out.name;checkpoints.mkdir(parents=True,exist_ok=False)
    for arm in cfg['arms']:
        model=build_arm(parent,arm,cfg).to(a.device)
        model.eval()
        with torch.no_grad():growth_drift=(model(calibration[0][:4].to(a.device))[0].cpu()-parent_logits).abs().max().item()
        if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        for b in model.blocks:b.ffn.route_generator=torch.Generator(device=a.device).manual_seed(cfg['seed']+100)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
        initial={(l,e):[p.detach().cpu().clone() for p in expert.parameters()] for l,e,expert in experts(model)}
        layers=len(model.blocks);E=len(model.blocks[0].ffn.experts)
        exposure=torch.zeros(layers,E);positive=torch.zeros(layers,E);gradient_sum=torch.zeros(layers,E)
        update_sum=torch.zeros(layers,E);update_samples=0;last_update=torch.zeros(layers,E);last_update_step=0;ring=deque(maxlen=cfg['window_steps'])
        gradient_ring=deque(maxlen=cfg['window_steps']);last_token_step=torch.full((layers,E),-1.)
        gen=torch.Generator().manual_seed(cfg['seed']+200);started=time.perf_counter();records=[]
        for i in range(cfg['steps']+1):
            exploration=float(arm=='warmup' and i<cfg['warmup_steps'])
            set_policy(model,exploration,bias_rate=cfg['bias_rate'] if arm in ['bias','family_bias'] else 0.)
            if i%cfg['eval_every']==0 or i==cfg['steps']:
                val=evaluate(model,evals['validation'],a.device);cal=evaluate(model,calibration,a.device)
                load=torch.stack(list(ring)).sum(0) if ring else torch.zeros_like(exposure)
                fractions=load/load.sum(-1,keepdim=True).clamp_min(1)
                entropy=-(fractions*fractions.clamp_min(1e-12).log()).sum(-1)
                r={'arm':arm,'step':i,'validation_ce':val['ce'],'exploration':exploration,
                   'training_window_tokens':min(i,cfg['window_steps'])*cfg['batch_size']*parent.cfg.context,
                   'window_route_fractions':fractions.tolist(),'window_zero_experts':(load==0).sum(-1).tolist(),
                   'hard_probe_zero_experts':(cal['counts']==0).sum(-1).tolist(),
                   'window_load_cv':(fractions.std(-1,unbiased=False)/fractions.mean(-1).clamp_min(1e-12)).tolist(),
                   'effective_experts':entropy.exp().tolist() if i else [None]*layers,'cumulative_tokens':exposure.tolist(),
                   'window_positive_task_gradient_steps':torch.stack(list(gradient_ring)).sum(0).tolist() if gradient_ring else torch.zeros_like(positive).tolist(),
                   'last_token_step':last_token_step.tolist(),
                   'positive_task_gradient_steps':positive.tolist(),'mean_task_gradient_norm':(gradient_sum/max(i,1)).tolist(),
                   'mean_task_gradient_norm_when_nonzero':(gradient_sum/positive.clamp_min(1)).tolist(),
                   'mean_decay_removed_update_norm':(update_sum/max(update_samples,1)).tolist(),
                   'update_samples':update_samples,'last_decay_removed_update_norm':last_update.tolist(),
                   'last_update_sample_step':last_update_step,'weight_drift':weight_drift(model,initial),
                   'family_output_relative_rms':function_diversity(model,hidden),
                   'expert_optimizer_state':optimizer_expert_state(model,opt),
                   'elapsed_seconds_including_diagnostics':time.perf_counter()-started}
                records.append(r)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
                print(json.dumps({k:r[k] for k in ['arm','step','validation_ce','window_zero_experts','hard_probe_zero_experts','effective_experts']}),flush=True)
            if i==cfg['steps']:break
            model.train();opt.zero_grad(set_to_none=True)
            x,y=(t.to(a.device) for t in sample(data['train'],gen,cfg['batch_size'],parent.cfg.context))
            logits,aux=model(x);ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
            ce.backward()
            task_norms=gradient_norms(model).cpu();positive+=(task_norms>0);gradient_sum+=task_norms
            gradient_ring.append((task_norms>0).float())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(norm) or not torch.isfinite(ce):raise RuntimeError('Nonfinite objective/gradient')
            capture=(i+1)%cfg['eval_every']==0
            before=snapshot_experts(model) if capture else None
            opt.step()
            if capture:
                last_update=decay_removed_updates(model,before,cfg['lr'],cfg['weight_decay']).cpu()
                update_sum+=last_update;update_samples+=1;last_update_step=i+1
            routed=torch.stack([item['counts'].detach().cpu() for item in aux]);exposure+=routed;ring.append(routed)
            last_token_step[routed>0]=i+1
        result={'arm':arm,'parameters':model.parameter_counts(),'growth_probe_max_logit_drift':growth_drift,
                'noise_std_by_layer':[float(cfg['noise_fraction']*source.ffn.router.weight.square().mean().sqrt()) if arm=='noise' else 0. for source,b in zip(parent.blocks,model.blocks)],
                'initial':records[0],'final':records[-1],
                'test_ce':evaluate(model,evals['test'],a.device)['ce'],
                'reset_ablation':ablate_to_initial(model,parent,ablation,a.device) if cfg['ablation'] else None}
        result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None
        summary.append(result);(out/'summary.json').write_text(json.dumps(summary,indent=2))
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
                    'routing_rng':[b.ffn.route_generator.get_state() for b in model.blocks],
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],
                    'config':cfg,'arm':arm,'step':cfg['steps'],'cumulative_tokens':exposure,
                    'positive_task_gradient_steps':positive,'gradient_sum':gradient_sum,
                    'update_sum':update_sum,'update_samples':update_samples,'last_update':last_update,
                    'last_update_step':last_update_step,'rolling_counts':list(ring),
                    'gradient_ring':list(gradient_ring),'last_token_step':last_token_step},checkpoints/f'{arm}.pt')
        del model,opt
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
    print('Completed:',[r['arm'] for r in summary],flush=True)

if __name__=='__main__':main()
