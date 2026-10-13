"""Find complete GPT dense-phase evidence with a surviving checkpoint."""
import json
from pathlib import Path


def find_dense_run(root, seed=17):
    root=Path(root)
    candidates=[]
    for run in (root/'results').glob('gpt-colab-*'):
        checkpoint=root/'checkpoints'/run.name/str(seed)/'dense_pretrain.pt'
        if not checkpoint.is_file() or checkpoint.stat().st_size==0:
            continue
        try:
            cfg=json.loads((run/'config.json').read_text())
            manifest=json.loads((run/'data_manifest.json').read_text())
            records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
            records=[r for r in records if r.get('seed')==seed and r.get('phase')=='dense_pretrain']
            if not manifest.get('sha256') or not records:
                continue
            if records[0]['step']!=0 or records[-1]['step']!=cfg['dense_steps']:
                continue
        except (OSError,ValueError,KeyError,TypeError):
            continue
        candidates.append((checkpoint.stat().st_mtime,run))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None
