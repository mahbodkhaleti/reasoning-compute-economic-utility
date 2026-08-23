# Data acquisition and provenance

The source is the Kaggle mirror `onlineauctions/online-auctions-dataset`, cited
in the paper and accessed on 2026-08-16. The expected `auction.csv` is 1,043,530
bytes with SHA-256:

```text
c05d10bafd4f33aadb2c70ab562a23937bd11c8d32a7c73244c829c06302486e
```

Install and authenticate the Kaggle CLI, then run:

```bash
bash data/download_online_auctions.sh
mkdir -p online-auctions-dataset
cp data/source/auction.csv online-auctions-dataset/auction.csv
```

The raw CSV is deliberately excluded from Git. The Kaggle mirror labels its
license as unknown, so this repository does not assert redistribution rights.
The checked-in `robustness_extensions/server_outputs/data/` directory contains
the exact processed inputs used by the saved offline analyses. Do not publish
those derivative files until the authors have confirmed that their planned
distribution is permitted.

`prepare_scenarios.py` constructs 216 scenarios deterministically using master
seed 20260819. It creates category-specific 70/30 construction/held-out splits,
candidate sets, held-out scores, and the scenario identifiers consumed by the
runner. Candidate generation and option ordering use SHA-256-derived integer
seeds rather than Python's process-randomized hash.
