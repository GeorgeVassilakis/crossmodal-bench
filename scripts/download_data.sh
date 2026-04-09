#!/usr/bin/env bash
# Download the pre-built ProVABGS + DESI + Legacy Survey cross-matched dataset.
# ~3.9 GB, contains 4000 galaxies with images, spectra, and photometry.
#
# NOTE: This is DESI EDR/SV3 data that AION-1 was trained on.
# Use for pipeline validation / proof-of-concept only.
# For a clean held-out benchmark, swap in DESI DR1 main-survey data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
mkdir -p "${DATA_DIR}"

URL="https://users.flatironinstitute.org/~polymathic/data/provabgs_desi_ls.hdf5"
OUTPUT="${DATA_DIR}/provabgs_desi_ls.hdf5"

if [ -f "${OUTPUT}" ]; then
    echo "Data already exists at ${OUTPUT}"
    exit 0
fi

echo "Downloading provabgs_desi_ls.hdf5 (~3.9 GB)..."
wget -O "${OUTPUT}" "${URL}" || curl -L -o "${OUTPUT}" "${URL}"
echo "Done. Saved to ${OUTPUT}"
