# Locally reconstructed data

This directory is populated by `scripts/prepare_scenarios.py`. Its CSV/JSON
outputs are intentionally not redistributed because the source dataset's
Kaggle mirror reports an unknown license. Run `data/download_online_auctions.sh`
from the repository root and then `reproduce_offline.sh`; the wrapper verifies
the source checksum and deterministically recreates the required files.
