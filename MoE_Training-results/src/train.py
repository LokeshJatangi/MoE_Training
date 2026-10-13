"""Run from repository root: python -m src.train --device cpu"""
import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import LinearLM, SparseMoE, parameters


def windows(data, starts, context):
    return data[starts[:, None] + torch.arange(context)], data[starts + context]


def load_data(path, context):
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    # Split raw text first: no input or target can cross a split boundary.
    a, b = int(.8 * len(text)), int(.9 * len(text))
    vocab = sorted(set(text[:a]))
    mapping = {c: i for i, c in enumerate(vocab)}
    # This corpus has the same character support in all splits; fail if changed.
    splits = {k: torch.tensor([mapping[c] for c in t]) for k, t in
              [('train', text[:a]), ('validation', text[a:b]), ('test', text[b:])]}
    manifest = {'source': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                'characters': len(text), 'vocabulary': vocab,
                'split_characters': {k: len(v) for k, v in splits.items()},
                'split_method': 'contiguous 80/10/10; split before constructing windows',
                'context': context}
    return splits, manifest


def fixed_batch(data, context, n, seed):
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(data) - context, (n,), generator=gen)
    return windows(data, starts, context)


@torch.no_grad()
def evaluate(model, batch, device):
    model.eval()
    losses, counts, entropies = [], [], []
    for x, y in zip(batch[0].split(512), batch[1].split(512)):
        logits, aux = model(x.to(device))
        losses.append(F.cross_entropy(logits, y.to(device), reduction='sum').item())
        if aux is not None:
            counts.append(aux['counts'].cpu())
            entropies.append((aux['entropy'].item(), len(y)))
    out = {'ce': sum(losses) / len(batch[1])}
    if counts:
        load = torch.stack(counts).sum(0)
        out['expert_fraction'] = (load / load.sum()).tolist()
        out['router_entropy'] = sum(e*n for e,n in entropies) / len(batch[1])
    return out


def update(model, optimizer, batch, device, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, aux = model(batch[0].to(device))
    ce = F.cross_entropy(logits, batch[1].to(device))
    objective = ce + (cfg['balance_coefficient'] * aux['balance'] if aux else 0)
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
    if not torch.isfinite(norm) or not torch.isfinite(objective):
        raise RuntimeError('Non-finite gradient or loss')
    optimizer.step()
    return {'batch_ce': ce.item(), 'objective': objective.item(), 'grad_norm': norm.item()}


def run_phase(model, cfg, splits, evals, device, seed, phase, steps, lr, offset, folder):
    # Both continuation branches deliberately reset AdamW at conversion.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    gen = torch.Generator().manual_seed(seed + (10000 if phase == 'dense_pretrain' else 20000))
    records = []
    start = time.perf_counter()
    last = {}
    for step in range(steps + 1):
        if step % cfg['eval_every'] == 0 or step == steps:
            tr = evaluate(model, evals['train'], device)
            va = evaluate(model, evals['validation'], device)
            rec = {'seed': seed, 'phase': phase, 'step': step, 'global_step': offset + step,
                   'targets_seen': (offset + step) * cfg['batch_size'],
                   'train_ce': tr['ce'], 'validation_ce': va['ce'],
                   'elapsed_seconds_including_eval': time.perf_counter() - start,
                   **last}
            if 'expert_fraction' in va:
                rec.update(expert_fraction=va['expert_fraction'], router_entropy=va['router_entropy'])
            records.append(rec)
            print(json.dumps(rec), flush=True)
        if step == steps:
            break
        indices = torch.randint(len(splits['train']) - cfg['context'], (cfg['batch_size'],), generator=gen)
        last = update(model, opt, windows(splits['train'], indices, cfg['context']), device, cfg)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                  'sampler_rng': gen.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                  'phase': phase, 'step': steps, 'seed': seed, 'config': cfg}
    checkpoint_path = folder / f'{phase}.pt'
    torch.save(checkpoint, checkpoint_path)
    # Verify actual disk reload, optimizer restoration, sampler restoration and next update.
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    clone = copy.deepcopy(model)
    clone.load_state_dict(loaded['model'])
    clone_opt = torch.optim.AdamW(clone.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    clone_opt.load_state_dict(loaded['optimizer'])
    clone_gen = torch.Generator()
    clone_gen.set_state(loaded['sampler_rng'].cpu())
    next_a = torch.randint(len(splits['train']) - cfg['context'], (cfg['batch_size'],), generator=gen)
    next_b = torch.randint(len(splits['train']) - cfg['context'], (cfg['batch_size'],), generator=clone_gen)
    assert torch.equal(next_a, next_b)
    # Work on a copy so verification cannot alter the measured final model.
    reference = copy.deepcopy(model)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    update(reference, ref_opt, windows(splits['train'], next_a, cfg['context']), device, cfg)
    update(clone, clone_opt, windows(splits['train'], next_b, cfg['context']), device, cfg)
    resume_error = max((x-y).abs().max().item() for x,y in zip(reference.parameters(), clone.parameters()))
    assert resume_error < 1e-6, resume_error
    return records, {'seconds_including_eval_and_save_excluded': elapsed,
                     'target_tokens_per_second_including_eval': steps * cfg['batch_size'] / elapsed,
                     'resume_next_step_max_parameter_error': resume_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/routing.json')
    parser.add_argument('--data', default='data/tinyshakespeare.txt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/routing')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads(Path(args.config).read_text())
    if args.seeds:
        cfg['seeds'] = args.seeds
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite previous evidence
    cfg_path = out / 'config.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))
    splits, manifest = load_data(Path(args.data), cfg['context'])
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    evals = {k: fixed_batch(v, cfg['context'], cfg['eval_samples'], 9000+i) for i,(k,v) in enumerate(splits.items())}
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src').glob('*.py'))}
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'platform': platform.platform(), 'device': args.device, 'threads': args.threads,
                   'git_commit': commit, 'source_sha256': source_hashes,
                   'gpu_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else None}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    all_records, summaries = [], []
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        checkpoint_dir = Path('checkpoints') / out.name / str(seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dense = LinearLM(cfg['context'], len(manifest['vocabulary'])).to(args.device)
        records, timing = run_phase(dense,cfg,splits,evals,args.device,seed,'dense_pretrain',
                                   cfg['dense_steps'],cfg['dense_lr'],0,checkpoint_dir)
        all_records += records
        initial, dense_end = records[0], records[-1]
        moe = SparseMoE(dense, cfg['experts'], cfg['top_k']).to(args.device)
        with torch.no_grad():
            probe = evals['validation'][0].to(args.device)
            max_error = (dense(probe)[0] - moe(probe)[0]).abs().max().item()
        assert max_error < 1e-5, max_error
        torch.save({'model':moe.state_dict(),'config':cfg,'seed':seed},checkpoint_dir/'moe_converted.pt')
        moe_initial_state = copy.deepcopy(moe.state_dict())
        branches = {}
        for name, model in [('dense_control', copy.deepcopy(dense)), ('moe_continue', moe)]:
            recs, stats = run_phase(model,cfg,splits,evals,args.device,seed,name,
                                   cfg['continuation_steps'],cfg['continuation_lr'],cfg['dense_steps'],checkpoint_dir)
            all_records += recs
            # Test is measured only at final checkpoints, never used for model selection.
            branches[name] = {'start':recs[0], 'final':recs[-1], 'test_ce':evaluate(model,evals['test'],args.device)['ce'], **stats}
        expert_deltas = [(expert.weight - moe_initial_state[f'experts.{i}.weight']).norm().item()
                         for i,expert in enumerate(moe.experts)]
        summary = {'seed':seed,'dense_parameters':parameters(dense),'moe_total_parameters':parameters(moe),
                   'moe_active_parameters_per_target':parameters(moe.router)+cfg['top_k']*parameters(dense.linear),
                   'conversion_max_logit_error':max_error,'initial':initial,'dense_end':dense_end,
                   'dense_pretrain_timing':timing, 'branches':branches, 'expert_weight_change_l2':expert_deltas,
                   'expert_pair_distance_l2':(moe.experts[0].weight-moe.experts[1].weight).norm().item()}
        summary['acceptance'] = {'dense_train_loss_drops':dense_end['train_ce'] < initial['train_ce'],
                                'dense_validation_loss_drops':dense_end['validation_ce'] < initial['validation_ce'],
                                'moe_train_loss_drops':branches['moe_continue']['final']['train_ce'] < dense_end['train_ce'],
                                'moe_validation_loss_drops':branches['moe_continue']['final']['validation_ce'] < dense_end['validation_ce'],
                                'all_experts_updated':all(x>0 for x in expert_deltas),
                                'conversion_preserved':max_error<1e-5}
        summaries.append(summary)
        (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
        with (out / 'metrics.jsonl').open('w') as f:
            for rec in all_records:
                f.write(json.dumps(rec)+'\n')
    keys = ['seed','phase','step','global_step','targets_seen','train_ce','validation_ce','elapsed_seconds_including_eval','batch_ce','objective','grad_norm','router_entropy']
    with (out / 'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); writer.writeheader(); writer.writerows(all_records)
    print('Acceptance:', json.dumps([s['acceptance'] for s in summaries]))
    if not all(all(s['acceptance'].values()) for s in summaries):
        raise SystemExit('One or more acceptance checks failed; preserve evidence and investigate.')

if __name__ == '__main__':
    main()
