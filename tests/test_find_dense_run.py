import json
from pathlib import Path
import tempfile
import unittest
from scripts.find_dense_run import find_dense_run

class FindDenseRunTests(unittest.TestCase):
    def make_run(self,root,name,completed=True,checkpoint=True):
        run=root/'results'/name;run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps({'dense_steps':1000}))
        (run/'data_manifest.json').write_text(json.dumps({'sha256':'fixture'}))
        (run/'metrics.jsonl').write_text('\n'.join(json.dumps({'seed':17,'phase':'dense_pretrain','step':s}) for s in [0,1000 if completed else 100]))
        if checkpoint:
            path=root/'checkpoints'/name/'17'/'dense_pretrain.pt';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
        return run

    def test_restarted_runtime_without_checkpoint_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertIsNone(find_dense_run(root))
            self.make_run(root,'gpt-colab-old',checkpoint=False)
            self.assertIsNone(find_dense_run(root))

    def test_discovers_surviving_complete_run_and_skips_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected=self.make_run(root,'gpt-colab-valid')
            self.make_run(root,'gpt-colab-partial',completed=False)
            self.assertEqual(find_dense_run(root),expected)
