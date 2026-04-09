# Cross-Modal Spectral Prediction Benchmark

Evaluates AION-1's ability to predict DESI galaxy spectra from Legacy Survey
images alone, using held-out DESI DR1 main-survey galaxies (AION-1 was only
trained on the ~1M EDR/SV3 subset).

## Quick Start

```bash
# 1. Prepare evaluation dataset (~5K matched image+spectrum pairs)
python scripts/prepare_data.py --config configs/default.yaml

# 2. Run AION-1 inference (image → spectrum prediction)
python scripts/run_inference.py --config configs/default.yaml

# 3. Compute Tier 1 metrics
python scripts/evaluate.py

# 4. View results
jupyter notebook notebooks/results.ipynb
```

## Metrics (Tier 1: Spectral Fidelity)

- Per-wavelength reduced χ²
- Continuum R² (polynomial fit comparison)
- Residual structure analysis (systematic bias, noise per wavelength)
- All metrics stratified by galaxy type (BGS / LRG / ELG / QSO)
