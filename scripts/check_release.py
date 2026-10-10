#!/usr/bin/env python3
"""Check canonical paths, documentation links and the portable bundle."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT / 'release.json').read_text())
    for name in release['paths'].values():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            raise RuntimeError('Missing release path: ' + name)
    documents = ['README.md', 'docs/REPOSITORY_MAP.md', 'docs/REPRODUCE.md',
                 'docs/RELEASES.md', 'ai4f_workshop/README.md', 'ai4f_workshop/archive/README.md']
    links = 0
    for name in documents:
        source = ROOT / name
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', source.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            target = target.split('#', 1)[0]
            if not (source.parent / target).exists():
                raise RuntimeError('Broken link in ' + name + ': ' + target)
            links += 1
    prefix = 'ai4f_cpu_reference/'
    with zipfile.ZipFile(ROOT / release['paths']['compact_bundle']) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Corrupt compact bundle')
        manifest = json.loads(archive.read(prefix + 'BUNDLE_MANIFEST.json'))
        for name, digest in manifest.items():
            data = archive.read(prefix + name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError('Invalid bundled asset: ' + name)
            if name.startswith('ai4f_workshop/') or name == 'reproduce_ai4f_workshop.sh':
                if data != (ROOT / name).read_bytes():
                    raise RuntimeError('Stale bundle: ' + name)
    print(json.dumps({'status': 'passed', 'local_document_links': links,
                      'bundle_files_checked': len(manifest), 'new_inference': False}))


if __name__ == '__main__':
    main()
