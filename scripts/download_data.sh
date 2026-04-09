#!/usr/bin/env bash
# Download the pre-built ProVABGS + DESI + Legacy Survey cross-matched dataset.
# ~3.9 GB, contains 4000 galaxies with images, spectra, and photometry.
#
# NOTE: This is a small unseen eval set (~4000 objects) used for
# AION-1 spectrum-generation benchmarking.
# For an alternate benchmark, see scripts/prepare_data.py for building
# a DESI DR1 main-survey dataset.

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
