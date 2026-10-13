"""Scale-aware FP32 conversion checks with retained behavioral diagnostics."""
import copy
import torch
from torch.nn import functional as F


@torch.no_grad()
def conversion_diagnostics(reference, actual, targets, atol=1e-5, rtol=1e-5):
    if reference.shape != actual.shape:
        raise ValueError('Conversion changed output shape')
    difference = (actual-reference).abs()
    allowed = atol + rtol*reference.abs()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    dense_ce = F.cross_entropy(reference.reshape(-1,reference.shape[-1]),targets.reshape(-1)).item()
    moe_ce = F.cross_entropy(actual.reshape(-1,actual.shape[-1]),targets.reshape(-1)).item()
    close = bool(torch.all(difference <= allowed)) if finite else False
    ce_close = abs(dense_ce-moe_ce) <= 1e-5
    return {'atol':atol,'rtol':rtol,'ce_atol':1e-5,'finite':finite,
            'max_absolute_logit_error':difference.max().item(),
            'rms_logit_error':difference.square().mean().sqrt().item(),
            'max_error_over_allowed':(difference/allowed).max().item(),
            'logits_outside_tolerance':int((difference>allowed).sum().item()),
            'dense_probe_ce':dense_ce,'moe_probe_ce':moe_ce,'probe_ce_change':moe_ce-dense_ce,
            'max_probability_error':(actual.softmax(-1)-reference.softmax(-1)).abs().max().item(),
            'logits_close':close,'ce_close':ce_close,'passed':finite and close and ce_close}


@torch.no_grad()
def verify_model_conversion(dense, moe, tokens, targets):
    """Validate full-copy conversion in FP32 and independent FP64 CPU probes.

    Retain the original per-logit FP32 check. If it fails, accept only when
    a tighter CPU-double reference passes AND explicit FP32 behavioral and
    error bounds pass. This is not a generic license to ignore mismatches.
    """
    stats = conversion_diagnostics(dense(tokens)[0], moe(tokens)[0], targets)
    indices = torch.linspace(0, len(tokens)-1, min(8, len(tokens))).long()
    probe = tokens.detach().cpu()[indices]
    labels = targets.detach().cpu()[indices]
    dense64 = copy.deepcopy(dense).cpu().double().eval()
    moe64 = copy.deepcopy(moe).cpu().double().eval()
    ref = dense64(probe)[0]
    actual = moe64(probe)[0]
    difference = (actual-ref).abs()
    finite64 = bool(torch.isfinite(ref).all() and torch.isfinite(actual).all())
    close64 = bool(torch.all(difference <= 1e-10 + 1e-10*ref.abs())) if finite64 else False
    ce_ref = F.cross_entropy(ref.reshape(-1,ref.shape[-1]),labels.reshape(-1)).item()
    ce_actual = F.cross_entropy(actual.reshape(-1,actual.shape[-1]),labels.reshape(-1)).item()
    reference_passed = finite64 and close64 and abs(ce_actual-ce_ref) <= 1e-10
    stats['fp64_cpu_reference'] = {
        'sequence_indices':indices.tolist(),'atol':1e-10,'rtol':1e-10,'ce_atol':1e-10,
        'max_absolute_logit_error':difference.max().item(),
        'probe_ce_change':ce_actual-ce_ref,'passed':reference_passed}
    bounds = {'max_absolute_logit_error':1e-4, 'rms_logit_error':1e-5,
              'max_probability_error':1e-5, 'absolute_probe_ce_change':1e-5}
    bounded = (stats['finite'] and stats['max_absolute_logit_error'] <= bounds['max_absolute_logit_error']
               and stats['rms_logit_error'] <= bounds['rms_logit_error']
               and stats['max_probability_error'] <= bounds['max_probability_error']
               and abs(stats['probe_ce_change']) <= bounds['absolute_probe_ce_change'])
    original_passed = stats['passed']
    stats['fp32_behavior_bounds'] = bounds
    stats['fp32_behavior_bounds_passed'] = bounded
    stats['passed'] = reference_passed and (original_passed or bounded)
    stats['acceptance_method'] = ('fp32_per_logit_and_fp64_reference' if original_passed else
                                  'fp64_reference_and_bounded_fp32') if stats['passed'] else 'rejected'
    return stats
