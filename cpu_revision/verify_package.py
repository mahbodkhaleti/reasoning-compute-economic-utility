"""Verify delivered static files before regenerating outputs."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
count=0
for line in (root/'MANIFEST.sha256').read_text().splitlines():
 expected,relative=line.split('  ',1);p=(root/relative).resolve();assert p.is_relative_to(root)
 assert p.is_file(),relative
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==expected,relative
 count+=1
print(f'Static package integrity passed: {count} files.')
