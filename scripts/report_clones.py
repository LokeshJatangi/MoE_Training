import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
root=Path(a.results)
rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()] if (root/'metrics.jsonl').exists() else []
summary=json.loads((root/'summary.json').read_text())
env=json.loads((root/'environment.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for name in sorted({r['name'] for r in rows}):
 r=[v for v in rows if v['name']==name]
 for ax,key in zip(axes,['validation_ce','hard_probe_zero_experts','exploration']):
  y=[max(v[key]) if isinstance(v[key],list) else v[key] for v in r]
  ax.plot([v['step'] for v in r],y,label=name)
for ax,title in zip(axes,['Hard-routing validation CE','Maximum per-layer hard-probe zero count','Training exploration probability']):
 ax.set(title=title,xlabel='Growth continuation step');ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.savefig(root/'clone_curves.png',dpi=160)
lines=['# Clone routing lab outcomes','',
f"All validation and calibration probes use HARD routing, even when training is stochastic. Recorded execution: {env['device']} / {env.get('gpu') or 'CPU'} / PyTorch {env['torch']}.",'',
'| Arm | Status | Total params | Switch | Post-switch steps | Final hard-probe zero count |',
'|---|---|---:|---:|---:|---|']
for s in summary:
 lines.append(f"| {s['name']} | {s['status']} | {s['parameters']['total']:,} | {s.get('switch_step')} | {s.get('post_switch_steps',0)} | {s.get('final',{}).get('hard_probe_zero_experts')} |")
lines+=['','A zero expert count on one batch is not permanent death. The training-window metric aggregates the last configured number of steps; the hard-probe metric aggregates recent fixed calibration probes. Neither guarantees future survival. Output distances are sampled within clone families and are diagnostics, not a validated switching threshold.','',
'Reported statuses distinguish completed runs from memory skips, CUDA OOM, runtime failures and time limits. A configured but unexecuted size is not a measured outcome.','',
'![Clone curves](clone_curves.png)','']
(root/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:14]))
