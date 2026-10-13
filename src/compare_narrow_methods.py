"""Single-seed language experiment from an existing trained dense GPT checkpoint."""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import torch
from .gpt import GPTConfig, TinyGPT
from .narrow_methods import narrow_upcycle
from .train import load_data
from .train_gpt import phase, evaluate, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/narrow_methods_cpu.json')
    p.add_argument('--data', default='data/tinyshakespeare.txt')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    checkpoint = Path(cfg['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['seed'] == cfg['seed'] and saved['phase'] == 'dense_pretrain'
    cfg['model'] = saved['config']['model']
    cfg['dense_steps'] = saved['step']
    splits, manifest = load_data(Path(args.data), cfg['model']['context'])
    original_manifest = json.loads((Path(cfg['dense_evidence']) / 'data_manifest.json').read_text())
    assert manifest['sha256'] == original_manifest['sha256'], 'Dataset differs from pretraining'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    env = {'device': 'cpu', 'threads': 2, 'precision': 'FP32', 'python': platform.python_version(), 'torch': str(torch.__version__),
           'dense_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'source_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src').glob('*.py')}}
    (out/'environment.json').write_text(json.dumps(env, indent=2))
    seed = cfg['seed']
    torch.manual_seed(seed)
    dense = TinyGPT(GPTConfig(vocab_size=len(manifest['vocabulary']), **cfg['model']))
    dense.load_state_dict(saved['model'])
    dense.eval()
    evals = {k: sample(v, torch.Generator().manual_seed(9000+i), cfg['eval_sequences'], cfg['model']['context'])
             for i, (k, v) in enumerate(splits.items())}
    baseline = evaluate(dense, evals['validation'], 'cpu')['task_ce']
    probe = evals['validation'][0]
    with torch.no_grad():
        reference = dense(probe)[0]
    checkpoint_dir = Path('checkpoints')/out.name/str(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    summary = {'seed': seed, 'dense_checkpoint_validation_ce': baseline, 'branches': {}}
    for method in cfg['methods']:
        torch.manual_seed(seed+30000)
        if method == 'dense_control':
            branch = copy.deepcopy(dense)
        elif method == 'full_copy':
            branch = dense.upcycle()
        else:
            branch = narrow_upcycle(dense, method, seed=seed, **cfg['conversion'])
        with torch.no_grad():
            drift = (branch(probe)[0]-reference).abs().max().item()
        initial = {name: value.detach().clone() for name, value in branch.named_parameters() if '.experts.' in name}
        mapping = [b.ffn.unit_indices for b in branch.blocks] if method not in ('dense_control', 'full_copy') else None
        records, timing = phase(branch, cfg, splits['train'], evals, seed, method, cfg['continuation_steps'],
                                cfg['continuation_lr'], saved['step'], 'cpu', checkpoint_dir, out/'metrics.jsonl')
        changes = {name: (value-initial[name]).norm().item() for name, value in branch.named_parameters() if name in initial}
        final = records[-1]
        h = branch.cfg.intermediate
        k = 1 if method == 'dense_control' else branch.cfg.top_k
        summary['branches'][method] = {
            'parameters': branch.parameter_counts(), 'selected_ffn_width': h*k,
            'ffn_matrix_macs_per_token_per_layer': 3*branch.cfg.width*h*k,
            'conversion_max_logit_drift': drift, 'conversion_ce_jump': records[0]['validation_ce']-baseline,
            'start': records[0], 'final': final, 'unit_indices': mapping,
            'validation_drops': final['validation_ce']<records[0]['validation_ce'],
            'recovers_dense': final['validation_ce']<baseline,
            'expert_matrix_changes_l2': changes, 'all_expert_matrices_updated': all(v>0 for v in changes.values()) if changes else None,
            'final_probe_zero_experts_per_layer': [sum(v==0 for v in fractions) for fractions in final.get('expert_fractions',[])],
            **timing}
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
    # Test is evaluated only after all predeclared arms finish; never used for selection.
    for method in cfg['methods']:
        torch.manual_seed(seed+30000)
        if method == 'dense_control': model = copy.deepcopy(dense)
        elif method == 'full_copy': model = dense.upcycle()
        else: model = narrow_upcycle(dense, method, seed=seed, **cfg['conversion'])
        model.load_state_dict(torch.load(checkpoint_dir/f'{method}.pt', weights_only=True)['model'])
        summary['branches'][method]['test_ce'] = evaluate(model, evals['test'], 'cpu')['task_ce']
    (out/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
