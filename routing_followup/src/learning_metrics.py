"""Expert telemetry: separate exposure, CE gradients, decay and learned behavior."""
import copy
import torch
from torch.nn import functional as F
from .routing_lab import grow


def build_arm(parent, arm, cfg):
    torch.manual_seed(cfg['seed']+2)
    model=grow(parent,cfg['copies'],family_routing=arm=='family_bias')
    if arm=='noise':
        for layer,(source,block) in enumerate(zip(parent.blocks,model.blocks)):
            weight=block.ffn.router.weight
            gen=torch.Generator(device=weight.device).manual_seed(cfg['seed']+1000+layer)
            with torch.no_grad():
                scale=source.ffn.router.weight.square().mean().sqrt()*cfg['noise_fraction']
                weight.add_(torch.randn(weight.shape,device=weight.device,generator=gen)*scale)
    return model


def experts(model):
    for layer,b in enumerate(model.blocks):
        for index,e in enumerate(b.ffn.experts):
            yield layer,index,e


def gradient_norms(model):
    layers=[]
    for b in model.blocks:
        norms=[]
        for expert in b.ffn.experts:
            square=sum((p.grad.detach().float().square().sum() for p in expert.parameters() if p.grad is not None),
                       b.ffn.router.weight.new_zeros(()))
            norms.append(square.sqrt())
        layers.append(torch.stack(norms))
    return torch.stack(layers)


def snapshot_experts(model):
    return {(layer,index):[p.detach().clone() for p in e.parameters()] for layer,index,e in experts(model)}


def decay_removed_updates(model,before,lr,weight_decay):
    rows=[]
    for b in model.blocks:
        values=[];layer=len(rows)
        for index,e in enumerate(b.ffn.experts):
            norm=e.gate.weight.new_zeros(())
            for previous,p in zip(before[layer,index],e.parameters()):
                # AdamW skips parameters with grad=None, including decay.
                decayed=previous*(1-lr*weight_decay) if p.grad is not None else previous
                norm=norm+(p.detach()-decayed).float().square().sum()
            values.append(norm.sqrt())
        rows.append(torch.stack(values))
    return torch.stack(rows)


@torch.no_grad()
def fixed_hidden_probes(parent,tokens,device,size=32):
    probes={};handles=[]
    for layer,block in enumerate(parent.blocks):
        def capture(module,args,layer=layer):
            x=args[0].detach().reshape(-1,args[0].shape[-1])
            ids=torch.linspace(0,len(x)-1,min(size,len(x)),device=x.device).long()
            probes[layer]=x[ids].clone().cpu()
        handles.append(block.ffn.register_forward_pre_hook(capture))
    parent.eval();parent(tokens.to(device))
    for handle in handles:handle.remove()
    return probes


@torch.no_grad()
def function_diversity(model,hidden):
    result=[]
    for layer,b in enumerate(model.blocks):
        x=hidden[layer].to(next(model.parameters()).device)
        outputs=torch.stack([e(x) for e in b.ffn.experts])
        families=[]
        for family in range(b.ffn.parents):
            out=outputs[family*b.ffn.copies:(family+1)*b.ffn.copies].flatten(1)
            denominator=out.square().mean().sqrt().clamp_min(1e-9)
            rms=torch.pdist(out)/out.shape[-1]**.5/denominator
            families.append(float(rms.mean().cpu()))
        result.append(families)
    return result


@torch.no_grad()
def weight_drift(model,initial):
    result=[]
    for layer,b in enumerate(model.blocks):
        row=[]
        for index,e in enumerate(b.ffn.experts):
            square=sum((p.detach().cpu()-base).square().sum().item()
                       for p,base in zip(e.parameters(),initial[layer,index]))
            denominator=sum(base.square().sum().item() for base in initial[layer,index])**.5
            row.append({'l2':square**.5,'relative_l2':square**.5/max(denominator,1e-12)})
        result.append(row)
    return result


def optimizer_expert_state(model,opt):
    return [[{'state_matrices':sum(bool(opt.state.get(p)) for p in e.parameters()),
              'matrix_steps':[int(opt.state[p]['step']) if p in opt.state else 0 for p in e.parameters()]}
             for e in b.ffn.experts] for b in model.blocks]


@torch.no_grad()
def ablate_to_initial(model,parent,batch,device):
    """Reset one expert at a time, keeping router weights fixed; held-out CE delta."""
    model.eval();x,y=(t.to(device) for t in batch)
    def ce():
        logits,_=model(x)
        return F.cross_entropy(logits.flatten(0,1),y.flatten()).item()
    baseline=ce();result=[]
    for layer,b in enumerate(model.blocks):
        row=[]
        for index,e in enumerate(b.ffn.experts):
            source=parent.blocks[layer].ffn.experts[index//b.ffn.copies]
            replacement=copy.deepcopy(source).to(device)
            b.ffn.experts[index]=replacement
            try:delta=ce()-baseline
            finally:b.ffn.experts[index]=e
            row.append(delta)
        result.append(row)
    return {'baseline_probe_ce':baseline,'targets':y.numel(),
            'expert_reset_ce_delta':result,
            'interpretation':'Positive: resetting these learned weights hurts this fixed probe in the current model context. Negative: reset helps. Zero is not evidence of useful learning.'}
