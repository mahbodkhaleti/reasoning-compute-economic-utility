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
For exact audit and offline replay, the release preserves the transformed split,
summary, and scenario files under
`new_models_full_server_package/data/historical/`. They are not covered by the
repository's MIT license and are provided without any new license claim. The
same artifacts can be reconstructed locally by the deterministic preparation
code below.

`prepare_scenarios.py` constructs 216 scenarios deterministically using master
seed 20260819. It creates category-specific 70/30 construction/held-out splits,
candidate sets, held-out scores, and the scenario identifiers consumed by the
runner. Candidate generation and option ordering use SHA-256-derived integer
seeds rather than Python's process-randomized hash.

The expected reconstructed files and historical SHA-256 values are:

```text
auction_splits.csv  fb3ff11bac65c30a51afea0d0811bd24164851ed3c638040537885145d17cdef
dataset_summary.csv 4efba13152356c7c8105e57f57c1ec9316dead4ef9f810673f579ecd2fb489ae
scenarios.csv       12a846cb3f9362ad9be5c5830d5aef71d7300328ce2399ac87cf2ef60f9986e6
```

These hashes verify either the preserved artifacts or a local reconstruction;
they are not a grant of redistribution rights.
