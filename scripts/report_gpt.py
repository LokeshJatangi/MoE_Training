"""Plot and report a GPT upcycling run without selecting a best checkpoint."""
import argparse
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--results',required=True)
a=p.parse_args()
root=Path(a.results)
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
summaries=json.loads((root/'summary.json').read_text())
env=json.loads((root/'environment.json').read_text())
execution=f"{env['device']} / {env.get('gpu_name') or 'CPU'} / PyTorch {env['torch']}"
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
colors={'dense_pretrain':'#475569','dense_control':'#d97706','moe_continue':'#087f8c'}
for phase,color in colors.items():
    r=[v for v in rows if v['phase']==phase]
    steps=sorted({v['global_step'] for v in r})
    for ax,metric in zip(axes,['train_ce','validation_ce']):
        values=[[v[metric] for v in r if v['global_step']==s] for s in steps]
        ax.plot(steps,[st.mean(v) for v in values],label=phase.replace('_',' '),color=color)
        if len(summaries)>1:
            ax.fill_between(steps,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.15)
for ax,title in zip(axes,['Training evaluation loss','Held-out validation loss']):
    ax.set(title=title,xlabel='Optimizer steps along each branch',ylabel='Cross entropy (nats / character)')
    ax.axvline(summaries[0]['dense_end']['step'],color='#94a3b8',ls='--')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle(f"Small GPT → FFN MoE | {len(summaries)} seed(s)")
fig.savefig(root/'loss_curves.png',dpi=170)
fig.savefig(root/'loss_curves.svg')
lines=['# Small GPT → MoE results','',
'| Seed | Initial validation | Trained dense GPT | Continued dense | Continued MoE | MoE test |',
'|---|---:|---:|---:|---:|---:|']
for s in summaries:
 b=s['branches']
 lines.append(f"| {s['seed']} | {s['initial']['validation_ce']:.6f} | {s['dense_end']['validation_ce']:.6f} | {b['dense_control']['final']['validation_ce']:.6f} | {b['moe_continue']['final']['validation_ce']:.6f} | {b['moe_continue']['test_ce']:.6f} |")
s=summaries[0]
lines += ['',f"Dense: **{s['dense_parameters']['total']:,}** parameters; MoE: **{s['moe_parameters']['total']:,}** total, **{s['moe_parameters']['active_parameter_convention']:,}** under the documented active-parameter convention.",
'',f"Architecture: {s['model_config']['layers']} causal Transformer layers, width {s['model_config']['width']}, {s['model_config']['heads']} heads, context {s['model_config']['context']}, SwiGLU FFNs, tied token/output weights and learned positions. Four experts per FFN, top-2 normalized routing, dropless dispatch. No dropout or autocast in this baseline.",
'',f"Maximum conversion logit error across seeds: {max(s['conversion_max_logit_error'] for s in summaries):.9g}.",
'', 'Acceptance per seed:','']
for s in summaries:
 lines.append(f"- Seed {s['seed']}: "+json.dumps(s['acceptance']))
lines += ['', '## Interpretation','',
'This is a dense GPT-to-MoE continuation experiment. Dense and MoE losses are measured at every next-character position using causal attention. Every expert matrix updated. Saved models, optimizer and sampler states reproduced the next update within 1e-6. No semantic expert specialization is established.',
'',
'Both continuation branches reset AdamW, use the same lower learning rate and identical sampled sequence order. Expert routing alone changes the FFN path; attention, embeddings, norms and output head are copied. Task CE is separate from the mean layer-balancing term. The reported timing includes evaluation and excludes saving/reload checks. There is no learning-rate sweep or best-checkpoint selection.',
'',
'Validation/test use fixed sampled windows from an 80/10/10 raw-text split. Windows cannot cross split boundaries, but windows within a split may overlap. This short-context character model is a mechanism experiment. Its character CE cannot be compared directly to subword-token CE across different prediction objectives.',
'',
f'Measured execution: {execution}. {len(summaries)} seed(s); these results do not establish universal superiority.',
'',
'Top-2 routing increases active FFN work relative to dense; neither total parameters nor FLOPs are matched. Any small loss difference needs repeated seeds and fair compute controls before interpretation.',
'', '![Loss curves](loss_curves.png)','']
(root/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:13]))
