#!/usr/bin/env python3
"""CPU-only entry point for the strict six-model synthetic merge."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--package-a",type=Path,required=True);p.add_argument("--package-b",type=Path,required=True);a=p.parse_args()
    script=a.package_b.resolve()/"src/merge_six_synthetic.py"
    if not script.exists():raise SystemExit(f"Missing merge implementation: {script}")
    subprocess.run([sys.executable,str(script),"--package-root",str(a.package_b.resolve()),"--existing4-package",str(a.package_a.resolve())],check=True)
if __name__=="__main__":main()
