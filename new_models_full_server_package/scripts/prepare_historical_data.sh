#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/results/historical/data"
cp "$ROOT/data/historical/scenarios.csv" "$ROOT/results/historical/data/scenarios.csv"
cp "$ROOT/data/historical/auction_splits.csv" "$ROOT/results/historical/data/auction_splits.csv"
cp "$ROOT/data/historical/dataset_summary.csv" "$ROOT/results/historical/data/dataset_summary.csv"
cp "$ROOT/data/historical/metadata.json" "$ROOT/results/historical/data/metadata.json"
(cd "$ROOT/data/historical" && shasum -a 256 -c SHA256SUMS)
