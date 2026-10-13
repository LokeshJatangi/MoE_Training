"""Controlled initialization ablations for equal-sized narrow expert banks."""
import copy
import math
import torch
from .gpt import ExpertFFN
from .granular import GranularFFN

METHODS = ('contiguous', 'shuffled', 'drop', 'perturbed', 'random')


def narrow_upcycle(model, method, partitions=4, replicas=4, top_k=4, seed=17, noise_fraction=.01):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if any(isinstance(b.ffn, ExpertFFN) for b in model.blocks):
        raise ValueError('Expected a dense checkpoint')
    result = copy.deepcopy(model)
    for layer, block in enumerate(result.blocks):
        dense = block.ffn
        moe = GranularFFN(dense, partitions, replicas, top_k)
        gen = torch.Generator(device=dense.gate.weight.device).manual_seed(seed + 40000 + layer)
        h = dense.gate.out_features
        small = h // partitions
        permutation = torch.randperm(h, generator=gen, device=dense.gate.weight.device)
        for i, expert in enumerate(moe.experts):
            if method == 'shuffled':
                # A disjoint partition shared by replicas, with randomized units.
                indices = permutation[(i % partitions)*small:((i % partitions)+1)*small]
            elif method == 'drop':
                # Independently retain 1/partitions of units per expert, without
                # replacement within an expert; overlaps across experts allowed.
                indices = torch.randperm(h, generator=gen, device=dense.gate.weight.device)[:small]
            else:
                indices = torch.tensor(moe.unit_indices[i], device=dense.gate.weight.device)
            with torch.no_grad():
                if method == 'random':
                    for name, p in expert.named_parameters():
                        std = .02 / math.sqrt(2*model.cfg.layers) if name == 'down.weight' else .02
                        p.copy_(torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) * std)
                    moe.unit_indices[i] = []
                else:
                    expert.gate.weight.copy_(dense.gate.weight[indices])
                    expert.up.weight.copy_(dense.up.weight[indices])
                    expert.down.weight.copy_(dense.down.weight[:, indices])
                    moe.unit_indices[i] = indices.cpu().tolist()
                    if method == 'perturbed':
                        for p in expert.parameters():
                            rms = p.square().mean().sqrt()
                            p.add_(torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) * rms * noise_fraction)
        block.ffn = moe
    result.cfg.intermediate //= partitions
    result.cfg.experts = partitions * replicas
    result.cfg.top_k = top_k
    return result
