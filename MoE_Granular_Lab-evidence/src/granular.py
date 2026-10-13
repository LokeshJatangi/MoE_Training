"""Partition a trained SwiGLU into narrower, replicated sparse experts."""
import copy
import torch
from torch import nn
from .gpt import ExpertFFN, SwiGLU


class GranularFFN(ExpertFFN):
    def __init__(self, dense, partitions, replicas, top_k):
        nn.Module.__init__(self)
        width = dense.gate.in_features
        intermediate = dense.gate.out_features
        if partitions < 1 or replicas < 1 or intermediate % partitions:
            raise ValueError('Positive partitions/replicas required; partitions must divide FFN width')
        count = partitions * replicas
        if not 1 <= top_k <= count:
            raise ValueError('Require 1 <= top_k <= expert count')
        self.top_k = top_k
        self.partitions = partitions
        self.replicas = replicas
        # Normalized routing weights average experts. Scale by partition count:
        # uniform weights over one of each partition then reconstruct the sum.
        self.output_scale = partitions
        small = intermediate // partitions
        self.experts = nn.ModuleList()
        self.unit_indices = []
        for replica in range(replicas):
            for part in range(partitions):
                indices = torch.arange(part * small, (part + 1) * small, device=dense.gate.weight.device)
                expert = SwiGLU(width, small).to(device=dense.gate.weight.device, dtype=dense.gate.weight.dtype)
                with torch.no_grad():
                    expert.gate.weight.copy_(dense.gate.weight[indices])
                    expert.up.weight.copy_(dense.up.weight[indices])
                    expert.down.weight.copy_(dense.down.weight[:, indices])
                self.experts.append(expert)
                self.unit_indices.append(indices.cpu().tolist())
        self.router = nn.Linear(width, count, bias=False).to(dense.gate.weight.device)
        nn.init.normal_(self.router.weight, std=.02)

    def forward(self, hidden):
        output, aux = super().forward(hidden)
        return output * self.output_scale, aux


def partition_upcycle(model, partitions, replicas, top_k):
    if any(isinstance(b.ffn, ExpertFFN) for b in model.blocks):
        raise ValueError('Partition conversion requires a dense GPT')
    result = copy.deepcopy(model)
    for block in result.blocks:
        block.ffn = GranularFFN(block.ffn, partitions, replicas, top_k)
    result.cfg.intermediate //= partitions
    result.cfg.experts = partitions * replicas
    result.cfg.top_k = top_k
    return result
