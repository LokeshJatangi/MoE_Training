"""Download and checksum the exact Tiny Shakespeare corpus."""
import hashlib
import urllib.request
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'data_specs/tinyshakespeare.json').read_text())
path=root/'data/tinyshakespeare.txt'
path.parent.mkdir(exist_ok=True)
raw=path.read_bytes() if path.exists() else urllib.request.urlopen(manifest['source'],timeout=60).read()
actual=hashlib.sha256(raw).hexdigest()
if actual != manifest['sha256']:
    raise SystemExit(f'Corpus checksum mismatch: {actual}; investigate upstream change before training.')
path.write_bytes(raw)
print(f'Verified {len(raw):,} bytes; SHA-256 {actual}')
