"""Matched-token continuation: dense, full-copy MoE, and narrower partition MoE."""
import argparse
import copy
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
import torch
from .gpt import GPTConfig, TinyGPT
from .granular import partition_upcycle
from .train import load_data
from .train_gpt import sample, evaluate, phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/granular_t4.json')
    parser.add_argument('--data', default='data/tinystories_lab.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    data_path = Path(args.data)
    if data_path.suffix == '.npz':
        import numpy as np
        with np.load(data_path) as data:
            splits = {key: torch.from_numpy(data[key].astype('int64')) for key in ('train', 'validation', 'test')}
            vocab_size = int(data['vocab_size'])
        manifest_path = data_path.parent / 'tinystories_lab' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    else:
        splits, manifest = load_data(data_path, cfg['model']['context'])
        vocab_size = len(manifest['vocabulary'])
    manifest.update(token_array_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    token_counts={k: len(v) for k, v in splits.items()})
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'torch': str(torch.__version__), 'device': args.device,
        'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'precision': 'FP32', 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}
    }, indent=2))
    evals = {k: sample(v, torch.Generator().manual_seed(9000 + i), cfg['eval_sequences'], cfg['model']['context'])
             for i, (k, v) in enumerate(splits.items())}
    summaries = []
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        checkpoint_dir = Path('checkpoints') / out.name / str(seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        dense = TinyGPT(GPTConfig(vocab_size=vocab_size, **cfg['model'])).to(args.device)
        records, timing = phase(dense, cfg, splits['train'], evals, seed, 'dense_pretrain', cfg['dense_steps'],
                                cfg['dense_lr'], 0, args.device, checkpoint_dir, out / 'metrics.jsonl')
        # Keep only one branch on the GPU during continuation. All branches start
        # from this same CPU dense checkpoint; all Adam states reset identically.
        dense = dense.cpu().eval()
        branches = {}
        for name in ('dense_control', 'full_copy', 'granular'):
            torch.manual_seed(seed + 30000)
            if name == 'dense_control':
                branch = copy.deepcopy(dense)
            elif name == 'full_copy':
                branch = dense.upcycle()
            else:
                branch = partition_upcycle(dense, **cfg['conversion'])
            branch = branch.to(args.device).eval()
            original = {k: p.detach().cpu().clone() for k, p in branch.named_parameters() if '.experts.' in k}
            with torch.no_grad():
                probe = evals['validation'][0]
                before = dense(probe)[0]
                after = branch(probe.to(args.device))[0].cpu()
                drift = (before - after).abs().max().item()
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            continuation, branch_timing = phase(branch, cfg, splits['train'], evals, seed, name,
                cfg['continuation_steps'], cfg['continuation_lr'], cfg['dense_steps'], args.device,
                checkpoint_dir, out / 'metrics.jsonl')
            changes = {k: (p.detach().cpu() - original[k]).norm().item()
                       for k, p in branch.named_parameters() if k in original}
            h = branch.cfg.intermediate
            k = branch.cfg.top_k if name != 'dense_control' else 1
            branches[name] = {
                'model_config': asdict(branch.cfg), 'parameters': branch.parameter_counts(),
                'selected_ffn_width': h * k,
                'ffn_matrix_macs_per_token_per_layer': 3 * branch.cfg.width * h * k,
                'router_matrix_macs_per_token_per_layer': branch.cfg.width * branch.cfg.experts if original else 0,
                'conversion_max_logit_drift': drift,
                'conversion_ce_jump': continuation[0]['validation_ce'] - records[-1]['validation_ce'],
                'start': continuation[0], 'final': continuation[-1],
                'validation_drops_from_conversion': continuation[-1]['validation_ce'] < continuation[0]['validation_ce'],
                'validation_recovers_dense_checkpoint': continuation[-1]['validation_ce'] < records[-1]['validation_ce'],
                'expert_weight_changes_l2': changes, 'all_expert_matrices_updated': all(v > 0 for v in changes.values()) if changes else None,
                'test_ce': evaluate(branch, evals['test'], args.device)['task_ce'],
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                **branch_timing}
            del branch
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        summaries.append({'seed': seed, 'dense_initial': records[0], 'dense_end': records[-1],
                          'dense_timing': timing, 'branches': branches})
        (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    lines = ['# Narrow-expert continuation results', '',
             'Measured run. Matched tokens/data order and reset Adam; FFN matrix MACs exclude attention, activation, routing selection and dispatch overhead.', '',
             '| Seed | Arm | Total / active parameters | Conversion CE | Final validation CE | Loss drops / recovers dense |',
             '|---|---|---|---|---|---|']
    for result in summaries:
        for name, branch in result['branches'].items():
            p = branch['parameters']
            lines.append(f"| {result['seed']} | {name} | {p['total']} / {p['active_parameter_convention']} | {branch['start']['validation_ce']:.6f} | {branch['final']['validation_ce']:.6f} | {branch['validation_drops_from_conversion']} / {branch['validation_recovers_dense_checkpoint']} |")
    lines += ['', 'See summary.json for conversion drift, load fractions, gradient checks, checkpoint recovery, timings and parameter changes. No statistical quality or GPU efficiency claim follows from a one-seed CPU pilot.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
