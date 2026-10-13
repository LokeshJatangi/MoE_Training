"""Audit uploaded completed Colab runs and regenerate submission figures; no training."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDIES={
 'gpt':('MoE_Training-results','gpt-colab-recovered-20261012-180346-953876'),
 'granular':('MoE_Granular_Lab-evidence','granular-20261012-122635-419439'),
 'clone':('MoE_Clone_Lab-evidence','clone-first-20261012-133612')}
out=ROOT/'analysis';out.mkdir(exist_ok=True)
audit={}
for name,(folder,run) in STUDIES.items():
 root=ROOT/folder;path=root/'results'/run
 env=json.loads((path/'environment.json').read_text())
 mismatches=[p for p,h in env['source_sha256'].items() if not (root/p).exists() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
 assert not mismatches,(name,mismatches)
 summary=json.loads((path/'summary.json').read_text())
 rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
 cfg=json.loads((path/'config.json').read_text())
 if name=='clone':
  assert len(summary)==3 and all(s['status']=='completed' and s['final']['step']==750 for s in summary)
 else:
  assert len(summary)==1
  assert all(b['final']['step']==cfg['continuation_steps'] for b in summary[0]['branches'].values())
  assert all(b['resume_next_update_parameter_error']<1e-6 for b in summary[0]['branches'].values())
 audit[name]={'evidence_root':folder,'run':run,'source_hashes_verified':len(env['source_sha256']),
              'source_hash_mismatches':mismatches,'gpu':env.get('gpu_name',env.get('gpu')),
              'python':env['python'],'torch':env['torch'],'seed':17,
              'summary_sha256':hashlib.sha256((path/'summary.json').read_bytes()).hexdigest(),
              'metrics_sha256':hashlib.sha256((path/'metrics.jsonl').read_bytes()).hexdigest(),
              'record_count':len(rows),'completed':True}
 if name=='clone':
  fig,axes=plt.subplots(1,2,figsize=(12,4.4))
  for s in summary:
   records=[r for r in rows if r['name']==s['name']]
   axes[0].plot([r['step'] for r in records],[r['validation_ce'] for r in records],label=s['arm'])
   trained=[r for r in records if r['step']>0]
   axes[1].plot([r['step'] for r in trained],[max(r['window_zero_routing_experts']) for r in trained],label=s['arm'])
  axes[0].set(xlabel='Continuation updates',ylabel='Validation CE (hard routing)')
  axes[1].set(xlabel='Continuation updates',ylabel='Max layer unused experts (last 50 updates)')
  maximum_zero=max(max(r['window_zero_routing_experts']) for r in rows if r['step']>0)
  axes[1].set_ylim(-.2,max(1,maximum_zero)+.5)
 else:
  fig,axes=plt.subplots(1,2,figsize=(12,4.4))
  for phase in dict.fromkeys(r['phase'] for r in rows):
   records=[r for r in rows if r['phase']==phase]
   axes[0].plot([r['global_step'] for r in records],[r['validation_ce'] for r in records],label=phase)
   if phase!='dense_pretrain':axes[1].plot([r['step'] for r in records],[r['validation_ce'] for r in records],label=phase)
  axes[0].set(xlabel='Pretraining + continuation updates',ylabel='Validation task CE')
  axes[1].set(xlabel='Continuation updates',ylabel='Validation task CE')
 for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle({'gpt':'Full-copy FFN upcycling — Tiny Shakespeare, T4, seed 17',
               'granular':'Narrow-expert upcycling — TinyStories, T4, seed 17',
               'clone':'Clone growth 4 → 12 experts — TinyStories, T4, seed 17'}[name])
 fig.tight_layout();fig.savefig(out/f'{name}_results.png',dpi=170);fig.savefig(out/f'{name}_results.svg');plt.close(fig)
# Data revision and tokenizer equality between the two independently prepared packages.
manifests=[]
for name in ('granular','clone'):
 folder=STUDIES[name][0];data=ROOT/folder/'data/tinystories_lab'
 m=json.loads((data/'manifest.json').read_text())
 assert hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()==m['tokenizer_sha256']
 manifests.append(m)
assert manifests[0]==manifests[1]
audit['tinystories_shared_manifest']=manifests[0]
(out/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v.get('source_hashes_verified') for k,v in audit.items() if k!='tinystories_shared_manifest'}))
