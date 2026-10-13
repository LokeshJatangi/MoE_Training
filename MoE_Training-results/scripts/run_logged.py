"""Stream child output to notebook and disk, preserving its actual failure."""
from collections import deque
from pathlib import Path
import subprocess


def run_logged(command, log_path, cwd=None):
    """Run a command with combined stdout/stderr; refuse to replace its log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=40)
    with log_path.open('x', encoding='utf-8') as log:
        heading = f'Command: {command!r}\n'
        print(heading, end='', flush=True)
        log.write(heading)
        log.flush()
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
        log.write(f'\nExit status: {code}\n')
        log.flush()
    if code:
        details = ''.join(tail)[-20000:]
        raise RuntimeError(
            f'Command failed with exit status {code}. Full output: {log_path.resolve()}\n'
            f'Last output lines (the underlying error is here):\n{details}'
        )
    return log_path
