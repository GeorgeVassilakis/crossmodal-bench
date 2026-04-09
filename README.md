# Cross-Modal Spectral Prediction Benchmark

Evaluates AION-1's ability to predict DESI galaxy spectra from Legacy Survey
images alone, using the ProVABGS + DESI + Legacy Survey cross-matched dataset
(~4000 galaxies with images, spectra, and photometry).

**Note:** The current dataset (`provabgs_desi_ls.hdf5`) is a small unseen eval
set (~4000 objects) for AION-1 spectrum-generation benchmarking. The
`prepare_data.py` pipeline can be used to build an alternate benchmark from
DESI DR1 main-survey data.

## Install

Create or activate an environment with Python 3.11+, then install the benchmark
dependencies:

```bash
python -m pip install -r requirements.txt
```

This benchmark tests cross-modal spectrum generation in flux space: given
Legacy Survey imaging, optionally with broadband photometry, generate a DESI
spectrum and compare it to the observed spectrum. The goal is not to measure
whether AION can reconstruct spectra it already sees, but whether it can
produce scientifically faithful spectra from another modality. To make the
numbers interpretable, the benchmark evaluates AION in two settings, image-only
and image-plus-photometry, alongside two lower-bound baselines and one
upper-bound oracle. The lower bounds are a leave-one-out mean-spectrum baseline
and a leave-one-out photometry nearest-neighbor baseline. The upper bound is a
codec oracle: the true DESI spectrum is encoded and decoded through the current
spectrum codec and then scored in the same flux-space metrics. This separates
three questions cleanly: whether the task is trivial, whether AION is beating
simple shortcuts, and whether the remaining gap is due to the codec or the
model.

On the current eval set, that ladder is well separated. The mean-spectrum
baseline is very poor, the photometry nearest-neighbor baseline is much
stronger but still clearly worse than AION, AION image-only and AION
image-plus-photometry both perform substantially better, and the codec oracle
is better still. In median per-object chi-squared, the ordering is roughly
29.98 for the mean baseline, 4.71 for photometry nearest-neighbor, 1.61 for
AION image-only, 1.53 for AION image-plus-photometry, and 0.87 for the codec
oracle. That means AION-1 is doing real cross-modal generation rather than
barely beating a retrieval shortcut, but it is also not codec-limited: the
oracle leaves clear model headroom. With the benchmark's continuum-shape metric,
the oracle is near-perfect, while AION captures much of the broad continuum
shape but still falls short on full spectral fidelity, indicating that the
remaining gap is in detailed generation quality rather than only coarse
continuum information.

## Quick Start

```bash
# 1. Download the pre-built dataset (~3.9 GB)
bash scripts/download_data.sh

# 2. Run AION-1 inference (image -> spectrum prediction, two modes)
python scripts/run_inference.py

# 3. Run the codec oracle (recommended for benchmark interpretation)
python scripts/run_oracle.py

# 4. Compute metrics, baselines, and skill summaries
python scripts/evaluate.py

# 5. View results (with Jupyter installed)
jupyter notebook notebooks/results.ipynb
```

`evaluate.py` will still run without `artifacts/oracle.hdf5`, but the oracle is
recommended because it makes the benchmark much easier to interpret.

## Metrics (Tier 1: Spectral Fidelity)

- Per-wavelength reduced chi-squared
- Continuum R-squared (weighted smooth continuum-shape comparison)
- Residual structure analysis (systematic bias, noise per wavelength)
- All metrics stratified by redshift bin
- Comparison against mean-spectrum and photometry nearest-neighbor baselines
- Comparison against a codec oracle upper bound and normalized skill summaries

## Inference Modes

1. **Image only**: Legacy Survey 4-band image -> DESI spectrum
2. **Image + photometry**: Legacy Survey image + g/r/i/z broadband fluxes -> DESI spectrum

Comparing the two modes measures the marginal value of spatially-resolved
imaging beyond broadband colors.
