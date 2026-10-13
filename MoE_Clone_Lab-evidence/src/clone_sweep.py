"""Bounded clone-growth lab. Runs only when explicitly invoked; preserves all outcomes."""
import argparse,copy,gc,hashlib,json,platform,time
from collections import deque
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from src.gpt import TinyGPT,GPTConfig
from src.routing_lab import grow,set_policy,family_output_diversity
from src.train_gpt import sample,step,evaluate


def counts(model):
    return torch.stack([b.ffn.last_counts.detach().cpu() for b in model.blocks])


def save(model,opt,gen,path,cfg,extra):
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                'config':cfg,**extra},path)


def train_base(model,cfg,data,device,steps,seed):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['base_lr'],betas=(.9,.95),weight_decay=.01)
    gen=torch.Generator().manual_seed(seed)
    training_cfg={**cfg,'balance_coefficient':.01,'gradient_clip':1.}
    for _ in range(steps):
        step(model,opt,sample(data,gen,cfg['batch_size'],cfg['model']['context']),device,training_cfg)
    return model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/clone_t4.json')
    p.add_argument('--data',default='data/tinystories_lab.npz')
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='results/clone-lab')
    p.add_argument('--arms',nargs='+',default=['hard','warmup','anneal'])
    p.add_argument('--max-runs',type=int,default=3)
    a=p.parse_args()
    torch.set_num_threads(2)
    cfg=json.loads(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    arrays=np.load(a.data)
    data={k:torch.from_numpy(arrays[k].astype(np.int64)) for k in ('train','validation','test')}
    vocab=int(arrays['vocab_size'])
    model_cfg=GPTConfig(vocab_size=vocab,**cfg['model'])
    env={'torch':str(torch.__version__),'python':platform.python_version(),'device':a.device,
         'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
         'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    checkpoints=Path('checkpoints')/out.name;checkpoints.mkdir(parents=True,exist_ok=True)
    evals={k:sample(v,torch.Generator().manual_seed(9000+i),cfg['eval_sequences'],model_cfg.context) for i,(k,v) in enumerate(data.items())}
    calibration=sample(data['train'],torch.Generator().manual_seed(9999),cfg['calibration_sequences'],model_cfg.context)
    torch.manual_seed(cfg['seed'])
    base=TinyGPT(model_cfg).to(a.device)
    base_initial=evaluate(base,evals['validation'],a.device)['task_ce']
    train_base(base,cfg,data['train'],a.device,cfg['dense_steps'],cfg['seed'])
    dense_ce=evaluate(base,evals['validation'],a.device)['task_ce']
    parent=base.upcycle();del base
    train_base(parent,cfg,data['train'],a.device,cfg['parent_steps'],cfg['seed']+1)
    parent_ce=evaluate(parent,evals['validation'],a.device)['task_ce']
    parent_logits=parent.eval()(calibration[0].to(a.device))[0].detach().cpu()
    base_summary={'model':asdict(model_cfg),'initial_validation_ce':base_initial,'dense_validation_ce':dense_ce,
                  'parent_validation_ce':parent_ce,'parent_parameters':parent.parameter_counts(),
                  'parent_optimizer':'AdamW reset at dense -> parent MoE; balance coefficient .01'}
    (out/'base_summary.json').write_text(json.dumps(base_summary,indent=2))
    parent.cpu()
    torch.save({'model':parent.state_dict(),'model_config':asdict(model_cfg)},checkpoints/'parent.pt')
    summaries=[];runs=0
    for expert_count in cfg['expert_counts']:
        if expert_count%model_cfg.experts:
            raise ValueError('Expert count must be a multiple of original parent count')
        for arm in a.arms:
            if runs>=a.max_runs:
                (out/'summary.json').write_text(json.dumps(summaries,indent=2));return
            runs+=1
            name=f'E{expert_count}-{arm}'
            # Fresh independent branches from the SAME CPU parent, not sequentially trained growth.
            torch.manual_seed(cfg['seed']+2)
            model=grow(parent,expert_count//model_cfg.experts,
                       router_noise=cfg.get('router_noise',0.),family_routing=arm=='family')
            params=model.parameter_counts()
            estimate=params['total']*16
            result={'name':name,'experts':expert_count,'arm':arm,'parameters':params,
                    'estimated_parameter_optimizer_bytes':estimate,'status':'running'}
            if a.device.startswith('cuda'):
                free,_=torch.cuda.mem_get_info()
                if estimate>free*.75:
                    result['status']='estimated_memory_skip';summaries.append(result)
                    (out/'summary.json').write_text(json.dumps(summaries,indent=2));del model;continue
                torch.cuda.reset_peak_memory_stats()
            opt=None
            try:
                model.to(a.device)
                for b in model.blocks:
                    b.ffn.route_generator=torch.Generator(device=a.device).manual_seed(cfg['seed']+100)
                # Conversion drift is measured, NOT assumed to be zero for flat cloned routing.
                model.eval()
                with torch.no_grad():
                    grown_logits=model(calibration[0].to(a.device))[0].cpu()
                result['growth_max_logit_change']=(grown_logits-parent_logits).abs().max().item()
                result['growth_validation_ce']=evaluate(model,evals['validation'],a.device)['task_ce']
                opt=torch.optim.AdamW(model.parameters(),lr=cfg['continuation_lr'],betas=(.9,.95),weight_decay=.01)
                gen=torch.Generator().manual_seed(cfg['seed']+200)
                ring=deque(maxlen=cfg['window_steps'])
                exposure=torch.zeros(model_cfg.layers,expert_count)
                switch=None;ready_streak=0;hard_probe_ring=deque(maxlen=cfg['ready_checks'])
                started=time.perf_counter()
                balance_coefficient=.01 if arm=='aux' else 0.
                train_cfg={**cfg,'balance_coefficient':balance_coefficient,'gradient_clip':1.}
                for i in range(cfg['steps']+1):
                    if arm in ('hard','aux','bias'):
                        exploration=0.
                    elif arm in ('warmup','family'):
                        exploration=float(i<cfg['warmup_steps'])
                    elif arm=='anneal':
                        exploration=max(0.,1.-i/cfg['anneal_steps'])
                    elif arm=='adaptive':
                        exploration=float(switch is None)
                    else: raise ValueError('Unknown arm: '+arm)
                    if not exploration and switch is None: switch=i
                    set_policy(model,exploration,bias_rate=cfg['bias_rate'] if arm=='bias' else 0.)
                    if i%cfg['eval_every']==0 or i==cfg['steps']:
                        # Evaluation ALWAYS uses hard routing; this does not update routing bias/RNG.
                        val=evaluate(model,evals['validation'],a.device)
                        cal=evaluate(model,calibration,a.device)
                        hard_fractions=torch.tensor(cal['expert_fractions'])
                        hard_probe_ring.append(hard_fractions)
                        coverage=torch.stack(list(hard_probe_ring)).sum(0)
                        hard_zeros=(coverage==0).sum(-1).tolist()
                        diversity=[family_output_diversity(b.ffn) for b in model.blocks]
                        training_counts=torch.stack(list(ring)).sum(0) if ring else torch.zeros_like(exposure)
                        training_zero=(training_counts==0).sum(-1).tolist()
                        # This is an explicit experimental criterion, not a universal threshold.
                        ready=(len(hard_probe_ring)==cfg['ready_checks'] and max(hard_zeros)==0
                               and exposure.min().item()>=cfg['min_routings']
                               and i>=cfg['warmup_steps'])
                        if arm=='adaptive' and switch is None and ready:
                            switch=i
                        record={'name':name,'step':i,'exploration':exploration,'validation_ce':val['task_ce'],
                                'window_zero_routing_experts':training_zero,'hard_probe_zero_experts':hard_zeros,
                                'hard_probe_expert_fractions':cal['expert_fractions'],
                                'minimum_cumulative_routings':exposure.min().item(),
                                'family_relative_output_distance':diversity,'switch_step':switch,
                                'elapsed_seconds':time.perf_counter()-started}
                        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        print(json.dumps(record),flush=True)
                        result['final']=record
                        if time.perf_counter()-started>cfg['max_arm_seconds']:
                            result['status']='time_budget_exhausted';break
                    if i==cfg['steps']:
                        result['status']='completed';break
                    step(model,opt,sample(data['train'],gen,cfg['batch_size'],model_cfg.context),a.device,train_cfg)
                    routed=counts(model);ring.append(routed);exposure+=routed
                result['switch_step']=switch
                result['post_switch_steps']=max(0,result['final']['step']-switch) if switch is not None else 0
                result['test_ce']=evaluate(model,evals['test'],a.device)['task_ce']
                result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None
                route_rng=[b.ffn.route_generator.get_state() for b in model.blocks]
                save(model,opt,gen,checkpoints/f'{name}.pt',cfg,{'expert_count':expert_count,'arm':arm,
                      'step':result['final']['step'],'switch_step':switch,'routing_rng':route_rng,
                      'rolling_counts':list(ring),'cumulative_routings':exposure})
            except torch.cuda.OutOfMemoryError as exc:
                result.update(status='cuda_oom',error=str(exc))
                # Original parent checkpoint and previous logged measurements remain usable.
            except RuntimeError as exc:
                result.update(status='runtime_failure',error=str(exc))
            summaries.append(result)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2))
            del model,opt;gc.collect()
            if a.device.startswith('cuda'):torch.cuda.empty_cache()

if __name__=='__main__':main()
