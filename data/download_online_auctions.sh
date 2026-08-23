#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/source"
EXPECTED_SHA256="c05d10bafd4f33aadb2c70ab562a23937bd11c8d32a7c73244c829c06302486e"

mkdir -p "${SOURCE_DIR}"
kaggle datasets download \
  -d onlineauctions/online-auctions-dataset \
  -p "${SOURCE_DIR}" \
  --unzip

AUCTION_CSV="${SOURCE_DIR}/auction.csv"
if [[ ! -f "${AUCTION_CSV}" ]]; then
  echo "Expected ${AUCTION_CSV}, but the Kaggle archive layout differs." >&2
  exit 1
fi

ACTUAL_SHA256="$(shasum -a 256 "${AUCTION_CSV}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Checksum mismatch for ${AUCTION_CSV}" >&2
  echo "Expected: ${EXPECTED_SHA256}" >&2
  echo "Actual:   ${ACTUAL_SHA256}" >&2
  exit 1
fi

echo "Verified ${AUCTION_CSV}"
