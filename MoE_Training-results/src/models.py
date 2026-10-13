"""Literal affine predictor and sparse, function-preserving MoE conversion."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class LinearLM(nn.Module):
    def __init__(self, context, vocab):
        super().__init__()
        self.context, self.vocab = context, vocab
        self.linear = nn.Linear(context * vocab, vocab)

    def features(self, tokens):
        return F.one_hot(tokens, self.vocab).flatten(1).float()

    def forward(self, tokens):
        return self.linear(self.features(tokens)), None

class SparseMoE(nn.Module):
    def __init__(self, dense, experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError('Require 1 <= top_k <= experts')
        self.context, self.vocab = dense.context, dense.vocab
        self.top_k = top_k
        self.experts = nn.ModuleList([copy.deepcopy(dense.linear) for _ in range(experts)])
        self.router = nn.Linear(self.context * self.vocab, experts)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)

    def forward(self, tokens):
        x = F.one_hot(tokens, self.vocab).flatten(1).float()
        router_logits = self.router(x)
        probs = router_logits.softmax(-1)
        selected_logits, ids = router_logits.topk(self.top_k, dim=-1)
        weights = selected_logits.softmax(-1)
        output = x.new_zeros(x.shape[0], self.vocab)
        # Actual sparse dispatch: each expert sees only its assigned samples.
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(ids == i)
            if rows.numel():
                contribution = expert(x[rows]) * weights[rows, slots, None]
                output = output.index_add(0, rows, contribution)
        counts = torch.bincount(ids.flatten(), minlength=len(self.experts)).float()
        fraction = counts / ids.numel()
        balance = len(self.experts) * (fraction.detach() * probs.mean(0)).sum()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        return output, {'balance': balance, 'counts': counts, 'entropy': entropy}

def parameters(model):
    return sum(p.numel() for p in model.parameters())
