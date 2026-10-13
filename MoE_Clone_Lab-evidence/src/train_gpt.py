"""Dense GPT -> FFN-upcycled MoE, with matched-token dense continuation."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .gpt import GPTConfig,TinyGPT,ExpertFFN
from .train import load_data


def sample(data,gen,n,context):
    starts=torch.randint(len(data)-context,(n,),generator=gen)
    sequence=data[starts[:,None]+torch.arange(context+1)]
    return sequence[:,:-1],sequence[:,1:]


def step(model,opt,batch,device,cfg):
    model.train()
    opt.zero_grad(set_to_none=True)
    x,y=(t.to(device) for t in batch)
    logits,aux=model(x)
    ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
    balance=torch.stack([a['balance'] for a in aux]).mean() if aux else ce.new_tensor(0.)
    objective=ce+cfg['balance_coefficient']*balance
    objective.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
    if not torch.isfinite(norm) or not torch.isfinite(objective):
        raise RuntimeError('Non-finite gradients/objective')
    router_grad=[b.ffn.router.weight.grad.norm().item() for b in model.blocks if isinstance(b.ffn,ExpertFFN)]
    opt.step()
    return {'batch_task_ce':ce.item(),'balance_loss':balance.item(),'objective':objective.item(),
            'gradient_norm_before_clip':norm.item(),'router_gradient_norm':router_grad}


@torch.no_grad()
def evaluate(model,batch,device):
    model.eval()
    loss=0.
    loads=[]
    entropy=[]
    for x,y in zip(batch[0].split(8),batch[1].split(8)):
        logits,aux=model(x.to(device))
        loss+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.to(device).reshape(-1),reduction='sum').item()
        if aux:
            counts=torch.stack([a['counts'] for a in aux]).cpu()
            if not loads:
                loads=[torch.zeros_like(counts)]
                entropy=[torch.zeros(len(aux))]
            loads[0]+=counts
            entropy[0]+=torch.tensor([a['entropy'].item() for a in aux])*y.numel()
    result={'task_ce':loss/batch[1].numel()}
    if loads:
        fractions=loads[0]/loads[0].sum(-1,keepdim=True)
        result.update(expert_fractions=fractions.tolist(),
                      router_entropy=(entropy[0]/batch[1].numel()).tolist(),
                      expert_load_cv=(fractions.std(-1,unbiased=False)/fractions.mean(-1)).tolist())
    return result


def phase(model,cfg,data,evals,seed,name,steps,lr,offset,device,checkpoint_dir,log_path):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(seed+(10000 if name=='dense_pretrain' else 20000))
    start=time.perf_counter()
    records=[]
    last={}
    for i in range(steps+1):
        if i%cfg['eval_every']==0 or i==steps:
            train=evaluate(model,evals['train'],device)
            validation=evaluate(model,evals['validation'],device)
            record={'seed':seed,'phase':name,'step':i,'global_step':offset+i,
                    'targets_seen':(offset+i)*cfg['batch_size']*cfg['model']['context'],
                    'train_ce':train['task_ce'],'validation_ce':validation['task_ce'],
                    'elapsed_seconds_including_eval':time.perf_counter()-start,**last}
            for k in ('expert_fractions','router_entropy','expert_load_cv'):
                if k in validation:
                    record[k]=validation[k]
            records.append(record)
            with log_path.open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if i==steps:
            break
        last=step(model,opt,sample(data,gen,cfg['batch_size'],cfg['model']['context']),device,cfg)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    saved={'model':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
           'config':cfg,'seed':seed,'phase':name,'step':steps}
    path=checkpoint_dir/f'{name}.pt'
    torch.save(saved,path)
    # Check on-disk model/Adam/sampler restoration with an identical next update.
    loaded=torch.load(path,map_location=device,weights_only=True)
    restored=copy.deepcopy(model)
    restored.load_state_dict(loaded['model'])
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=lr,betas=(.9,.95),weight_decay=cfg['weight_decay'])
    restored_opt.load_state_dict(loaded['optimizer'])
    restored_gen=torch.Generator()
    restored_gen.set_state(loaded['sampler_rng'].cpu())
    batch=sample(data,gen,cfg['batch_size'],cfg['model']['context'])
    restored_batch=sample(data,restored_gen,cfg['batch_size'],cfg['model']['context'])
    assert all(torch.equal(a,b) for a,b in zip(batch,restored_batch))
    reference=copy.deepcopy(model)
    ref_opt=torch.optim.AdamW(reference.parameters(),lr=lr,betas=(.9,.95),weight_decay=cfg['weight_decay'])
    ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    step(reference,ref_opt,batch,device,cfg)
    step(restored,restored_opt,restored_batch,device,cfg)
    error=max((a-b).abs().max().item() for a,b in zip(reference.parameters(),restored.parameters()))
    assert error<1e-6,error
    return records,{'wall_seconds_including_eval':elapsed,
                    'targets_per_second_including_eval':steps*cfg['batch_size']*cfg['model']['context']/elapsed,
                    'resume_next_update_parameter_error':error}


@torch.no_grad()
def generate(model,vocabulary,prompt,device,seed,length=200):
    gen=torch.Generator(device=device).manual_seed(seed)
    mapping={c:i for i,c in enumerate(vocabulary)}
    tokens=torch.tensor([[mapping[c] for c in prompt]],device=device)
    model.eval()
    for _ in range(length):
        logits,_=model(tokens[:,-model.cfg.context:])
        next_token=torch.multinomial((logits[:,-1]/.8).softmax(-1),1,generator=gen)
        tokens=torch.cat((tokens,next_token),dim=1)
    return ''.join(vocabulary[i] for i in tokens[0].tolist())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/gpt_cpu.json')
    p.add_argument('--data',default='data/tinyshakespeare.txt')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='results/gpt-cpu-pilot')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    cfg=json.loads(Path(a.config).read_text())
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    splits,manifest=load_data(Path(a.data),cfg['model']['context'])
    manifest['objective']='Causal next-character prediction at every position; input/target shifted by one.'
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    evals={k:sample(v,torch.Generator().manual_seed(9000+i),cfg['eval_sequences'],cfg['model']['context']) for i,(k,v) in enumerate(splits.items())}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError:
        commit=None
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}
    env={'python':platform.python_version(),'torch':str(torch.__version__),'device':a.device,'threads':a.threads,
         'gpu_name':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
         'git_commit':commit,'source_sha256':hashes,'precision':'FP32; no autocast/dropout'}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    summaries=[]
    samples={}
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        checkpoint_dir=Path('checkpoints')/out.name/str(seed)
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        model=TinyGPT(GPTConfig(vocab_size=len(manifest['vocabulary']),**cfg['model'])).to(a.device)
        dense_records,dense_timing=phase(model,cfg,splits['train'],evals,seed,'dense_pretrain',cfg['dense_steps'],cfg['dense_lr'],0,a.device,checkpoint_dir,out/'metrics.jsonl')
        model.eval()
        moe=model.upcycle().eval()
        with torch.no_grad():
            probe=evals['validation'][0].to(a.device)
            conversion_error=(model(probe)[0]-moe(probe)[0]).abs().max().item()
        assert conversion_error<1e-5,conversion_error
        torch.save({'model':moe.state_dict(),'config':cfg,'seed':seed},checkpoint_dir/'moe_converted.pt')
        original_experts={k:v.detach().clone() for k,v in moe.state_dict().items() if '.experts.' in k}
        branches={}
        for name,branch in [('dense_control',copy.deepcopy(model)),('moe_continue',moe)]:
            records,timing=phase(branch,cfg,splits['train'],evals,seed,name,cfg['continuation_steps'],cfg['continuation_lr'],cfg['dense_steps'],a.device,checkpoint_dir,out/'metrics.jsonl')
            branches[name]={'start':records[0],'final':records[-1],'test_ce':evaluate(branch,evals['test'],a.device)['task_ce'],**timing}
            samples[f'{seed}/{name}']=generate(branch,manifest['vocabulary'],'First Citizen:\n',a.device,1234)
        deltas={k:(v-original_experts[k]).norm().item() for k,v in moe.state_dict().items() if k in original_experts}
        all_updated=all(v>0 for v in deltas.values())
        acceptance={'dense_train_drops':dense_records[-1]['train_ce']<dense_records[0]['train_ce'],
                    'dense_validation_drops':dense_records[-1]['validation_ce']<dense_records[0]['validation_ce'],
                    'moe_train_drops':branches['moe_continue']['final']['train_ce']<dense_records[-1]['train_ce'],
                    'moe_validation_drops':branches['moe_continue']['final']['validation_ce']<dense_records[-1]['validation_ce'],
                    'all_expert_matrices_updated':all_updated,'conversion_preserved':conversion_error<1e-5}
        summary={'seed':seed,'model_config':asdict(model.cfg),'dense_parameters':model.parameter_counts(),
                 'moe_parameters':moe.parameter_counts(),'conversion_max_logit_error':conversion_error,
                 'initial':dense_records[0],'dense_end':dense_records[-1],'dense_timing':dense_timing,
                 'branches':branches,'expert_matrix_changes_l2':deltas,'acceptance':acceptance,
                 'peak_cuda_memory_allocated_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
        summaries.append(summary)
        (out/'summary.json').write_text(json.dumps(summaries,indent=2))
        (out/'generation_samples.json').write_text(json.dumps(samples,indent=2))
    print('Acceptance:',json.dumps([s['acceptance'] for s in summaries]),flush=True)
    if not all(all(s['acceptance'].values()) for s in summaries):
        raise SystemExit('Acceptance failure; preserve the run and investigate, without overwriting it.')

if __name__=='__main__':
    main()
