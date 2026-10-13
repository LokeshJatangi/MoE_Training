"""Controlled clone-family routing experiments; discrete sampled routes, not dense soft MoE."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .gpt import ExpertFFN

class CloneFFN(ExpertFFN):
    def __init__(self, parent, copies, router_noise=0., family_routing=False):
        nn.Module.__init__(self)
        self.top_k=parent.top_k
        self.parents=len(parent.experts)
        self.copies=copies
        self.family_routing=family_routing
        self.experts=nn.ModuleList([copy.deepcopy(e) for e in parent.experts for _ in range(copies)])
        device=parent.router.weight.device
        self.router=nn.Linear(parent.router.in_features,len(self.experts),bias=False).to(device)
        with torch.no_grad():
            self.router.weight.copy_(parent.router.weight.repeat_interleave(copies,dim=0))
            if router_noise:
                self.router.weight.add_(torch.randn_like(self.router.weight)*router_noise)
        self.register_buffer('routing_bias',torch.zeros(len(self.experts),device=device))
        self.exploration=0.
        self.temperature=1.
        self.bias_rate=0.
        self.route_generator=None
        self.last_counts=None
        self.last_hidden=None

    def sample_ids(self,scores,k):
        # Gumbel-top-k samples WITHOUT replacement. The discrete choice is not differentiated.
        uniform=torch.rand(scores.shape,device=scores.device,generator=self.route_generator).clamp_(1e-7,1-1e-7)
        gumbel=-torch.log(-torch.log(uniform))
        return (scores/self.temperature+gumbel).topk(k,dim=-1).indices

    def select(self,scores):
        if self.family_routing:
            family_scores=scores.reshape(-1,self.parents,self.copies).mean(-1)
            families=family_scores.topk(self.top_k,dim=-1).indices
            child_scores=scores.reshape(-1,self.parents,self.copies)[torch.arange(len(scores),device=scores.device)[:,None],families]
            children=child_scores.argmax(-1)
            hard=families*self.copies+children
            if self.training and self.exploration:
                sampled_children=self.sample_ids(child_scores.reshape(-1,self.copies),1).reshape(-1,self.top_k)
                sampled=families*self.copies+sampled_children
            else:
                return hard
        else:
            hard=scores.topk(self.top_k,dim=-1).indices
            if not (self.training and self.exploration):
                return hard
            sampled=self.sample_ids(scores,self.top_k)
        choose=torch.rand((len(scores),1),device=scores.device,generator=self.route_generator)<self.exploration
        return torch.where(choose,sampled,hard)

    def forward(self,hidden):
        shape=hidden.shape
        x=hidden.reshape(-1,shape[-1])
        logits=F.linear(x.float(),self.router.weight.float())
        scores=logits+self.routing_bias
        ids=self.select(scores)
        # Bias/noise changes selection only; mixing weights use unbiased selected logits.
        weights=logits.gather(1,ids).softmax(-1).to(x.dtype)
        output=torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            rows,slots=torch.where(ids==i)
            if rows.numel():
                output=output.index_add(0,rows,expert(x[rows])*weights[rows,slots,None])
        counts=torch.bincount(ids.flatten(),minlength=len(self.experts)).float()
        probs=logits.softmax(-1)
        balance=len(self.experts)*((counts/counts.sum()).detach()*probs.mean(0)).sum()
        self.last_counts=counts.detach()
        self.last_hidden=x[:32].detach()
        if self.training and self.bias_rate:
            with torch.no_grad():
                self.routing_bias.add_(self.bias_rate*torch.sign(counts.mean()-counts))
                self.routing_bias.sub_(self.routing_bias.mean())
        entropy=-(probs*probs.clamp_min(1e-9).log()).sum(-1).mean()
        return output.reshape(shape),{'counts':counts,'balance':balance,'entropy':entropy}


def grow(model,copies,router_noise=0.,family_routing=False):
    result=copy.deepcopy(model)
    for b in result.blocks:
        if not isinstance(b.ffn,ExpertFFN):
            raise ValueError('Grow an already-trained MoE; dense conversion is a separate phase.')
        b.ffn=CloneFFN(b.ffn,copies,router_noise,family_routing)
    return result


def set_policy(model,exploration,temperature=1.,bias_rate=0.):
    for b in model.blocks:
        b.ffn.exploration=exploration
        b.ffn.temperature=temperature
        b.ffn.bias_rate=bias_rate


@torch.no_grad()
def family_output_diversity(ffn):
    """Relative RMS output distance on the SAME hidden probes; not a switching theorem."""
    x=ffn.last_hidden
    if x is None or ffn.copies<2:
        return []
    distances=[]
    for parent in range(ffn.parents):
        indices=torch.linspace(0,ffn.copies-1,min(ffn.copies,8)).round().long().tolist()
        outputs=torch.stack([ffn.experts[parent*ffn.copies+j](x) for j in indices]).float()
        flat=outputs.flatten(1)
        scale=flat.square().mean().sqrt().clamp_min(1e-8)
        pair=torch.pdist(flat)/flat.shape[-1]**.5/scale
        distances.append(pair.mean().item())
    return distances
