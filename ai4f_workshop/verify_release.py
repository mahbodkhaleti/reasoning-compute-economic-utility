"""Check the anonymous AI4F static release before replay; no inference."""
from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parent
expected=json.loads((base/'MANIFEST_SHA256.json').read_text())
for name,digest in expected.items():
    p=base/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Missing or changed AI4F release asset: '+name)
print(json.dumps({'status':'passed','static_files':len(expected),'new_inference':False}))
