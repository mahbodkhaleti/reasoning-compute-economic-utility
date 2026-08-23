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
    manifest_path = output / "SHA256_MANIFEST.json"
    # Auction-derived data and machine-specific logs are needed locally but are
    # deliberately not redistributed in the result archive.
    files = [
        x for x in output.rglob("*")
        if x.is_file()
        and x != manifest_path
        and x.relative_to(output).parts[0] not in {"data", "logs"}
    ]
    code_files = [x for x in root.iterdir() if x.is_file() and x.name != args.zip]
    package_files = [
        x for x in root.parent.iterdir()
        if x.is_file() and x.name not in {args.zip}
    ]
    documentation_files = [x for x in (root.parent / "docs").rglob("*") if x.is_file()]
    private_inputs = [x for x in (output / "data").glob("*") if x.is_file()]
    manifest = {
        "outputs": {str(x.relative_to(output)): sha256(x) for x in files},
        "code": {x.name: sha256(x) for x in code_files},
        "package": {x.name: sha256(x) for x in package_files},
        "documentation": {
            str(x.relative_to(root.parent)): sha256(x) for x in documentation_files
        },
        "inputs_not_redistributed": {
            str(x.relative_to(output)): sha256(x) for x in private_inputs
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    zip_path = root / args.zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"outputs/{path.relative_to(output)}")
        for path in code_files:
            archive.write(path, arcname=f"code/{path.name}")
        for path in package_files:
            archive.write(path, arcname=path.name)
        for path in documentation_files:
            archive.write(path, arcname=str(path.relative_to(root.parent)))
    print(zip_path)


if __name__ == "__main__":
    main()
