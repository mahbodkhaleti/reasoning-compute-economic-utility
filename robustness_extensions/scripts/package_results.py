#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--zip", default="compute_arms_race_resolution_results.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    files = [x for x in output.rglob("*") if x.is_file()]
    code_files = [x for x in root.iterdir() if x.is_file() and x.name != args.zip]
    manifest = {
        "outputs": {str(x.relative_to(output)): sha256(x) for x in files},
        "code": {x.name: sha256(x) for x in code_files},
    }
    manifest_path = output / "SHA256_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    zip_path = root / args.zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"outputs/{path.relative_to(output)}")
        for path in code_files:
            archive.write(path, arcname=f"code/{path.name}")
    print(zip_path)


if __name__ == "__main__":
    main()
