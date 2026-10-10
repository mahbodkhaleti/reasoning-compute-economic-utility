#!/usr/bin/env python3
"""Build a deterministic compact source/audit bundle; no experiment or inference."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'ai4f_cpu_reference/'


def main():
    workshop = ROOT / 'ai4f_workshop'
    manifest = json.loads((workshop / 'MANIFEST_SHA256.json').read_text())
    content = {}
    for name, digest in manifest.items():
        data = (workshop / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('Workshop asset changed: ' + name)
        content['ai4f_workshop/' + name] = data
    content['ai4f_workshop/MANIFEST_SHA256.json'] = (workshop / 'MANIFEST_SHA256.json').read_bytes()
    content['reproduce_ai4f_workshop.sh'] = (ROOT / 'reproduce_ai4f_workshop.sh').read_bytes()
    content['README.md'] = b'''# Compact anonymous AI4F source and CPU audit

The current paper is ai4f_workshop/paper.pdf; its source is ai4f_workshop/main.tex.
Run `bash reproduce_ai4f_workshop.sh` from this directory. Python 3.10+ and
Poppler are required; no GPU, model weights or model API is used.
With LaTeX, run `make -C ai4f_workshop` and `make -C ai4f_workshop audit`.

This bundle checks saved numerical evidence and the prospective CPU reference.
It excludes the full historical inference archive. Full experiment replay is
available in the complete repository. CPU fixtures are not independent model
or service validation. Read ai4f_workshop/cpu_reference/CONTRACT.md.
The workshop README's links to parent repository documentation and releases
require the full repository; the commands above work within this compact bundle.
'''
    content['BUNDLE_MANIFEST.json'] = (json.dumps({
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(content.items())
    }, indent=2) + '\n').encode()
    destination = ROOT / 'releases/ai4f_cpu_reference.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w') as archive:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(PREFIX + name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(json.dumps({'status': 'passed', 'bundle': str(destination.relative_to(ROOT)),
                      'files': len(content), 'bytes': destination.stat().st_size}))


if __name__ == '__main__':
    main()
