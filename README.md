# Cross-Modal Spectral Prediction Benchmark

Evaluates AION-1's ability to predict DESI galaxy spectra from Legacy Survey
images alone, using the ProVABGS + DESI + Legacy Survey cross-matched dataset
(~4000 galaxies with images, spectra, and photometry).

**Note:** The current dataset (`provabgs_desi_ls.hdf5`) uses DESI EDR/SV3 data
that AION-1 was trained on. This is for pipeline validation only. For a clean
held-out benchmark, swap in DESI DR1 main-survey data.

## Quick Start

```bash
# 1. Download the pre-built dataset (~3.9 GB)
bash scripts/download_data.sh

# 2. Run AION-1 inference (image -> spectrum prediction, two modes)
python scripts/run_inference.py

# 3. Compute Tier 1 metrics
python scripts/evaluate.py

# 4. View results
jupyter notebook notebooks/results.ipynb
```

## Metrics (Tier 1: Spectral Fidelity)

- Per-wavelength reduced chi-squared
- Continuum R-squared (polynomial fit comparison)
- Residual structure analysis (systematic bias, noise per wavelength)
- All metrics stratified by redshift bin

## Inference Modes

1. **Image only**: Legacy Survey 4-band image -> DESI spectrum
2. **Image + photometry**: Legacy Survey image + g/r/i/z broadband fluxes -> DESI spectrum

Comparing the two modes measures the marginal value of spatially-resolved
imaging beyond broadband colors.
