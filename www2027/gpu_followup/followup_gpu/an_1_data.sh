#!/usr/bin/env bash
# Download 4 AuctionNet delivery periods (7-10), build competing prices, win curves and the 216 scenarios. CPU only.
set -euo pipefail
cd "$(dirname "$0")"; source env.sh; mkdir -p logs
env_a/bin/python -m pytest -q auctionnet_runner/tests -k "not prefix_reuse" 2>&1 | tee logs/an_tests.log
env_a/bin/python -m auctionnet_runner.run prepare-data 2>&1 | tee logs/an_prepare.log
env_a/bin/python -m auctionnet_runner.run build-scenarios 2>&1 | tee logs/an_scenarios.log
tar czf an_data_check.tgz results/auctionnet/*.json results/auctionnet/scenarios*.csv logs/an_*.log
echo "Send me an_data_check.tgz"
