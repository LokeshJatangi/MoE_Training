"""Small causal decoder-only Transformer and FFN-upcycled sparse MoE."""
import copy
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 65
    context: int = 128
    layers: int = 4
    width: int = 128
    heads: int = 4
    intermediate: int = 256
    experts: int = 4
    top_k: int = 2

class SwiGLU(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class ExpertFFN(nn.Module):
    def __init__(self, dense_ffn, width, experts, top_k):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError('Require 1 <= top_k <= experts')
        self.top_k = top_k
        self.experts = nn.ModuleList([copy.deepcopy(dense_ffn) for _ in range(experts)])
        self.router = nn.Linear(width, experts, bias=False)
        nn.init.normal_(self.router.weight, std=.02)

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        # Gate calculations in FP32 even when later using autocast.
        logits = F.linear(x.float(), self.router.weight.float())
        probs = logits.softmax(-1)
        values, ids = logits.topk(self.top_k, dim=-1)
        weights = values.softmax(-1).to(x.dtype)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(ids == i)
            if rows.numel():
                contribution = expert(x[rows]) * weights[rows, slots, None]
                output = output.index_add(0, rows, contribution)
        counts = torch.bincount(ids.flatten(), minlength=len(self.experts)).float()
        fraction = counts / ids.numel()
        balance = len(self.experts) * (fraction.detach() * probs.mean(0)).sum()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        return output.reshape(shape), {'balance': balance, 'counts': counts, 'entropy': entropy}

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError('width must be divisible by heads')
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3*cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q,k,v = [a.reshape(b,t,self.heads,d//self.heads).transpose(1,2) for a in (q,k,v)]
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        return self.proj(y.transpose(1,2).contiguous().reshape(b,t,d))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.norm_ffn = nn.LayerNorm(cfg.width)
        self.ffn = SwiGLU(cfg.width, cfg.intermediate)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        if isinstance(self.ffn, ExpertFFN):
            y, aux = self.ffn(self.norm_ffn(x))
        else:
            y, aux = self.ffn(self.norm_ffn(x)), None
        return x+y, aux

class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size,cfg.width)
        self.position_embedding = nn.Embedding(cfg.context,cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width,cfg.vocab_size,bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        for name,p in self.named_parameters():
            if name.endswith(('attn.proj.weight','ffn.down.weight')):
                nn.init.normal_(p, std=.02/math.sqrt(2*cfg.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.shape[1] > self.cfg.context:
            raise ValueError('Sequence exceeds configured context')
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(tokens.shape[1],device=tokens.device))
        aux=[]
        for block in self.blocks:
            x,a = block(x)
            if a is not None:
                aux.append(a)
        return self.head(self.norm(x)), aux

    def upcycle(self):
        result = copy.deepcopy(self)
        if any(isinstance(b.ffn,ExpertFFN) for b in result.blocks):
            raise ValueError('Model is already an MoE')
        for block in result.blocks:
            block.ffn = ExpertFFN(block.ffn,self.cfg.width,self.cfg.experts,self.cfg.top_k).to(self.head.weight.device)
        return result

    def parameter_counts(self):
        total=sum(p.numel() for p in self.parameters())
        inactive=0
        for block in self.blocks:
            if isinstance(block.ffn,ExpertFFN):
                each=sum(p.numel() for p in block.ffn.experts[0].parameters())
                inactive += (len(block.ffn.experts)-block.ffn.top_k)*each
        return {'total':total, 'active_parameter_convention':total-inactive,
                'convention':'All nonexpert parameters, including full tied embedding table, plus selected expert parameters. Not a FLOP count.'}
