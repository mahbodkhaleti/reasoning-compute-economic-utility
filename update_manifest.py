#!/usr/bin/env python3
"""Regenerate the top-level SHA-256 manifest from the Git index."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDE = {"MANIFEST.sha256", "repository_validation.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    entries = []
    for relative in sorted(files):
        if relative in EXCLUDE:
            continue
        path = ROOT / relative
        if path.is_file():
            entries.append(f"{sha256(path)}  {relative}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to MANIFEST.sha256")


if __name__ == "__main__":
    main()
