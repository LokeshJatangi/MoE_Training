import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from scripts.run_logged import run_logged


class LoggedCommandTests(unittest.TestCase):
    def test_failure_surfaces_stderr_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failure.log'
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'deliberate training failure'):
                    run_logged([sys.executable, '-u', '-c',
                        "import sys; print('step 1 finished'); print('RuntimeError: deliberate training failure', file=sys.stderr); sys.exit(3)"], path)
            text=path.read_text()
            self.assertIn('step 1 finished',text)
            self.assertIn('RuntimeError: deliberate training failure',text)
            self.assertIn('Exit status: 3',text)

    def test_success_and_log_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'success.log'
            with contextlib.redirect_stdout(io.StringIO()):
                run_logged([sys.executable, '-u', '-c', "print('finished')"], path)
                with self.assertRaises(FileExistsError):
                    run_logged([sys.executable, '-c', "raise Exception('must not run')"], path)
            self.assertIn('finished',path.read_text())
            self.assertNotIn('must not run',path.read_text())
