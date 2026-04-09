"""Compute Tier 1 spectral fidelity metrics.

Reads artifacts/ground_truth.hdf5 and artifacts/predictions.hdf5,
produces artifacts/metrics.json and artifacts/metrics_per_object.parquet.
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import yaml


# Common rest-frame emission/absorption lines to mask for continuum fitting (Angstrom)
REST_FRAME_LINES = {
    "OII": 3727,
    "Hdelta": 4101,
    "Hgamma": 4340,
    "Hbeta": 4861,
    "OIII_4959": 4959,
    "OIII_5007": 5007,
    "NII_6548": 6548,
    "Halpha": 6563,
    "NII_6583": 6583,
    "SII_6717": 6717,
    "SII_6731": 6731,
}
LINE_MASK_HALFWIDTH_A = 15.0  # mask +/- this many Angstrom around each line
CAMERA_BOUNDARIES_A = [4500, 5900, 7500]


def save_summary_plots(
    output_dir: str,
    wavelength: np.ndarray,
    z_spec: np.ndarray,
    spec_mask: np.ndarray,
    ivar: np.ndarray,
    true_flux: np.ndarray,
    pred_image_only: np.ndarray,
    pred_image_phot: np.ndarray,
    per_obj: pd.DataFrame,
    results_image: dict,
    results_phot: dict,
) -> None:
    """Write a compact set of static plots alongside the scalar metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    wav = wavelength

    # Plot 1: Per-wavelength chi2 and normalized residual bias
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(wav, results_image["chi2_per_wavelength"], color="steelblue", lw=0.7, label="Image only")
    ax1.plot(wav, results_phot["chi2_per_wavelength"], color="darkorange", lw=0.7, label="Image+phot")
    ax1.axhline(1.0, color="black", ls="--", lw=0.5, alpha=0.5)
    for boundary in CAMERA_BOUNDARIES_A:
        ax1.axvline(boundary, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax1.set_ylabel("Mean reduced chi2")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Per-Wavelength Reduced Chi-Squared")

    ax2.plot(
        wav,
        results_image["residuals"]["mean_norm_residual"],
        color="steelblue",
        lw=0.5,
        alpha=0.8,
        label="Image only",
    )
    ax2.plot(
        wav,
        results_phot["residuals"]["mean_norm_residual"],
        color="darkorange",
        lw=0.5,
        alpha=0.8,
        label="Image+phot",
    )
    ax2.axhline(0, color="black", ls="--", lw=0.5)
    for boundary in CAMERA_BOUNDARIES_A:
        ax2.axvline(boundary, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax2.set_ylabel("Mean norm. residual")
    ax2.set_xlabel("Wavelength [A]")
    ax2.set_title("Systematic Bias")
    ax2.set_xlim(float(wav[0]), float(wav[-1]))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chi2_per_wavelength.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Per-object chi2 histograms by redshift bin
    z_bin_list = sorted(set(per_obj["z_bin"].values))
    n_bins = len(z_bin_list)
    fig, axes = plt.subplots(1, n_bins, figsize=(4 * n_bins, 3.5), sharey=True)
    if n_bins == 1:
        axes = [axes]

    bins = np.logspace(-1, 2.5, 50)
    for ax, zbin in zip(axes, z_bin_list):
        mask = per_obj["z_bin"] == zbin
        chi2_img = per_obj.loc[mask, "chi2_image_only"].dropna()
        chi2_phot = per_obj.loc[mask, "chi2_image_phot"].dropna()

        ax.hist(chi2_img, bins=bins, alpha=0.5, color="steelblue", label="Image only")
        ax.hist(chi2_phot, bins=bins, alpha=0.5, color="darkorange", label="Image+phot")
        ax.axvline(3.0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Reduced chi2")
        ax.set_title(f"{zbin} (n={int(mask.sum())})")
        if ax == axes[0]:
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    plt.suptitle("Per-Object Reduced Chi-Squared Distribution", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chi2_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Improvement from adding photometry
    fig, ax = plt.subplots(figsize=(7, 5))
    delta_chi2 = per_obj["chi2_image_only"] - per_obj["chi2_image_phot"]
    scatter = ax.scatter(
        per_obj["z_spec"],
        delta_chi2,
        c=per_obj["chi2_image_only"],
        s=10,
        alpha=0.35,
        cmap="viridis",
        edgecolors="none",
    )
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Spectroscopic redshift")
    ax.set_ylabel("Delta chi2 (image-only minus image+phot)")
    ax.set_title("Photometry Gain Across Redshift")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("Image-only chi2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chi2_gain_vs_redshift.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Residual heatmap for image-only predictions
    sort_idx = np.argsort(z_spec)
    valid = ~spec_mask & (ivar > 0)
    norm_resid = (pred_image_only - true_flux) * np.sqrt(np.where(valid, ivar, 0))
    norm_resid[~valid] = np.nan

    step = max(1, len(sort_idx) // 500)
    display_idx = sort_idx[::step]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        norm_resid[display_idx],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        extent=[wav[0], wav[-1], z_spec[display_idx[-1]], z_spec[display_idx[0]]],
        interpolation="nearest",
    )
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Redshift")
    ax.set_title("Normalized Residual Heatmap (Image Only)")
    plt.colorbar(im, ax=ax, label="Normalized residual", shrink=0.8)

    for line_rest in REST_FRAME_LINES.values():
        z_range = np.linspace(z_spec[display_idx[0]], z_spec[display_idx[-1]], 100)
        obs_wav = line_rest * (1 + z_range)
        in_range = (obs_wav > wav[0]) & (obs_wav < wav[-1])
        ax.plot(obs_wav[in_range], z_range[in_range], "k--", lw=0.5, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_heatmap_image_only.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_line_mask(wavelength: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    """Create a boolean mask [N, L] that is True at emission/absorption line locations.

    Used to exclude lines when fitting the continuum.
    """
    N = len(z_spec)
    L = len(wavelength)
    line_mask = np.zeros((N, L), dtype=bool)

    for line_rest in REST_FRAME_LINES.values():
        for i in range(N):
            line_obs = line_rest * (1 + z_spec[i])
            line_mask[i] |= np.abs(wavelength - line_obs) < LINE_MASK_HALFWIDTH_A

    return line_mask


def per_wavelength_chi2(
    pred: np.ndarray,
    true: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Mean chi2 at each wavelength across the sample. Shape [L]."""
    valid = ~mask & (ivar > 0)
    residual_sq = (pred - true) ** 2 * ivar
    residual_sq[~valid] = np.nan
    return np.nanmean(residual_sq, axis=0)


def per_object_chi2(
    pred: np.ndarray,
    true: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Reduced chi2 per object (mean over wavelength). Shape [N]."""
    valid = ~mask & (ivar > 0)
    residual_sq = (pred - true) ** 2 * ivar
    residual_sq[~valid] = np.nan
    return np.nanmean(residual_sq, axis=1)


def continuum_r2(
    pred: np.ndarray,
    true: np.ndarray,
    wavelength: np.ndarray,
    spec_mask: np.ndarray,
    z_spec: np.ndarray,
    poly_degree: int = 7,
) -> np.ndarray:
    """Per-object R^2 of predicted vs true continuum shape. Shape [N]."""
    line_mask = make_line_mask(wavelength, z_spec)
    N = pred.shape[0]
    r2_values = np.full(N, np.nan)

    for i in range(N):
        # Fit continuum only on pixels that are not masked and not near emission lines
        fit_mask = ~spec_mask[i] & ~line_mask[i]
        if fit_mask.sum() < poly_degree + 10:
            continue

        x = wavelength[fit_mask]
        # Normalize wavelength for numerical stability
        x_norm = (x - x.mean()) / x.std()

        try:
            true_coeffs = np.polyfit(x_norm, true[i, fit_mask], poly_degree)
            true_cont = np.polyval(true_coeffs, x_norm)

            pred_cont = pred[i, fit_mask]

            ss_res = np.sum((pred_cont - true_cont) ** 2)
            ss_tot = np.sum((true_cont - true_cont.mean()) ** 2)

            if ss_tot > 0:
                r2_values[i] = 1.0 - ss_res / ss_tot
        except (np.linalg.LinAlgError, ValueError):
            continue

    return r2_values


def residual_analysis(
    pred: np.ndarray,
    true: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute residual statistics across the sample."""
    valid = ~mask & (ivar > 0)
    residual = pred - true
    residual[~valid] = np.nan

    # Normalized residual: should be ~ N(0,1) if well-calibrated
    norm_residual = residual * np.sqrt(np.where(valid, ivar, np.nan))

    return {
        "mean_residual": np.nanmean(residual, axis=0),  # [L]
        "std_residual": np.nanstd(residual, axis=0),  # [L]
        "mean_norm_residual": np.nanmean(norm_residual, axis=0),  # [L]
        "std_norm_residual": np.nanstd(norm_residual, axis=0),  # [L]
    }


def classify_redshift_bin(z_spec: np.ndarray) -> np.ndarray:
    """Assign galaxies to redshift bins for stratification."""
    bins = np.empty(len(z_spec), dtype="U10")
    bins[z_spec < 0.1] = "z<0.1"
    bins[(z_spec >= 0.1) & (z_spec < 0.2)] = "0.1<z<0.2"
    bins[(z_spec >= 0.2) & (z_spec < 0.3)] = "0.2<z<0.3"
    bins[(z_spec >= 0.3) & (z_spec < 0.5)] = "0.3<z<0.5"
    bins[z_spec >= 0.5] = "z>0.5"
    return bins


def stratified_summary(
    chi2_per_obj: np.ndarray,
    cont_r2: np.ndarray,
    strata: np.ndarray,
    chi2_threshold: float = 3.0,
) -> pd.DataFrame:
    """Summary table of metrics per stratum (redshift bin or target type)."""
    unique_strata = sorted(set(strata))
    rows = []
    for label in list(unique_strata) + ["ALL"]:
        if label == "ALL":
            mask = np.ones(len(chi2_per_obj), dtype=bool)
        else:
            mask = strata == label

        n = mask.sum()
        if n == 0:
            continue

        chi2_vals = chi2_per_obj[mask]
        r2_vals = cont_r2[mask]

        rows.append({
            "stratum": label,
            "n_objects": int(n),
            "median_chi2": float(np.nanmedian(chi2_vals)),
            "mean_chi2": float(np.nanmean(chi2_vals)),
            "frac_good": float(np.nanmean(chi2_vals < chi2_threshold)),
            "median_continuum_r2": float(np.nanmedian(r2_vals)),
            "mean_continuum_r2": float(np.nanmean(r2_vals)),
        })

    return pd.DataFrame(rows)


def evaluate_mode(
    pred_flux: np.ndarray,
    pred_mask: np.ndarray,
    true_flux: np.ndarray,
    ivar: np.ndarray,
    spec_mask: np.ndarray,
    wavelength: np.ndarray,
    z_spec: np.ndarray,
    strata: np.ndarray,
    mode_name: str,
    poly_degree: int = 7,
    chi2_threshold: float = 3.0,
) -> dict:
    """Run all Tier 1 metrics for one inference mode."""
    print(f"\n  === {mode_name} ===")

    # Combine masks: ground-truth mask + prediction mask
    combined_mask = spec_mask | pred_mask

    # Per-wavelength chi2
    chi2_lam = per_wavelength_chi2(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Median per-wavelength chi2: {np.nanmedian(chi2_lam):.3f}")

    # Per-object chi2
    chi2_obj = per_object_chi2(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Median per-object chi2: {np.nanmedian(chi2_obj):.3f}")
    print(f"  Fraction chi2 < {chi2_threshold}: {np.nanmean(chi2_obj < chi2_threshold):.3f}")

    # Continuum R2
    print("  Computing continuum R2...")
    cont_r2_vals = continuum_r2(
        pred_flux, true_flux, wavelength, combined_mask, z_spec, poly_degree
    )
    print(f"  Median continuum R2: {np.nanmedian(cont_r2_vals):.3f}")

    # Residual analysis
    residuals = residual_analysis(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Mean normalized residual (bias): {np.nanmean(residuals['mean_norm_residual']):.4f}")

    # Stratified summary
    summary = stratified_summary(chi2_obj, cont_r2_vals, strata, chi2_threshold)
    print(f"\n  Stratified summary:\n{summary.to_string(index=False)}")

    return {
        "chi2_per_wavelength": chi2_lam,
        "chi2_per_object": chi2_obj,
        "continuum_r2": cont_r2_vals,
        "residuals": residuals,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-modal predictions")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Config YAML"
    )
    parser.add_argument(
        "--ground-truth", default="artifacts/ground_truth.hdf5"
    )
    parser.add_argument(
        "--predictions", default="artifacts/predictions.hdf5"
    )
    parser.add_argument(
        "--output-dir", default="artifacts"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    config_path = os.path.join(repo_root, args.config)
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    gt_path = os.path.join(repo_root, args.ground_truth)
    pred_path = os.path.join(repo_root, args.predictions)
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth (produced by run_inference.py)
    print(f"Loading ground truth from {gt_path}")
    with h5py.File(gt_path, "r") as f:
        true_flux = f["spectrum_flux"][:]
        ivar = f["spectrum_ivar"][:]
        spec_mask = f["spectrum_mask"][:].astype(bool)
        wavelength = f["spectrum_lambda"][:]
        z_spec = f["z_spec"][:]

    if wavelength.ndim == 2:
        wavelength = wavelength[0]

    # Load predictions
    print(f"Loading predictions from {pred_path}")
    with h5py.File(pred_path, "r") as f:
        if "image_only_done" in f.attrs and not bool(f.attrs["image_only_done"]):
            raise RuntimeError("Image-only predictions are incomplete; rerun inference before evaluation.")
        if "image_phot_done" in f.attrs and not bool(f.attrs["image_phot_done"]):
            raise RuntimeError("Image+phot predictions are incomplete; rerun inference before evaluation.")

        pred_image_only = f["pred_flux_image_only"][:]
        pred_image_phot = f["pred_flux_image_phot"][:]
        if "pred_mask_image_only" in f:
            pred_mask_image_only = f["pred_mask_image_only"][:].astype(bool)
        elif "pred_mask" in f:
            pred_mask_image_only = f["pred_mask"][:].astype(bool)
        else:
            pred_mask_image_only = np.zeros_like(spec_mask, dtype=bool)

        if "pred_mask_image_phot" in f:
            pred_mask_image_phot = f["pred_mask_image_phot"][:].astype(bool)
        elif "pred_mask" in f:
            pred_mask_image_phot = f["pred_mask"][:].astype(bool)
        else:
            pred_mask_image_phot = np.zeros_like(spec_mask, dtype=bool)

    # Stratify by redshift bin (provabgs data is all BGS)
    strata = classify_redshift_bin(z_spec)
    print(f"Evaluating {len(true_flux)} objects...")
    unique, counts = np.unique(strata, return_counts=True)
    for s, c in zip(unique, counts):
        print(f"  {s}: {c}")

    # Evaluate both modes
    poly_degree = cfg.get("poly_degree", 7)
    chi2_threshold = cfg.get("chi2_good_threshold", 3.0)

    results_image = evaluate_mode(
        pred_image_only, pred_mask_image_only, true_flux, ivar, spec_mask,
        wavelength, z_spec, strata,
        "Image Only", poly_degree, chi2_threshold,
    )

    results_phot = evaluate_mode(
        pred_image_phot, pred_mask_image_phot, true_flux, ivar, spec_mask,
        wavelength, z_spec, strata,
        "Image + Photometry", poly_degree, chi2_threshold,
    )

    # Save metrics JSON
    metrics = {
        "image_only": {
            "median_chi2": float(np.nanmedian(results_image["chi2_per_object"])),
            "mean_chi2": float(np.nanmean(results_image["chi2_per_object"])),
            "frac_good": float(np.nanmean(results_image["chi2_per_object"] < chi2_threshold)),
            "median_continuum_r2": float(np.nanmedian(results_image["continuum_r2"])),
            "summary": results_image["summary"].to_dict(orient="records"),
        },
        "image_phot": {
            "median_chi2": float(np.nanmedian(results_phot["chi2_per_object"])),
            "mean_chi2": float(np.nanmean(results_phot["chi2_per_object"])),
            "frac_good": float(np.nanmean(results_phot["chi2_per_object"] < chi2_threshold)),
            "median_continuum_r2": float(np.nanmedian(results_phot["continuum_r2"])),
            "summary": results_phot["summary"].to_dict(orient="records"),
        },
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save per-object metrics as parquet for the notebook
    per_obj = pd.DataFrame({
        "z_bin": strata,
        "z_spec": z_spec,
        "chi2_image_only": results_image["chi2_per_object"],
        "chi2_image_phot": results_phot["chi2_per_object"],
        "cont_r2_image_only": results_image["continuum_r2"],
        "cont_r2_image_phot": results_phot["continuum_r2"],
    })
    per_obj_path = os.path.join(output_dir, "metrics_per_object.parquet")
    per_obj.to_parquet(per_obj_path, index=False)
    print(f"Saved per-object metrics to {per_obj_path}")

    # Save per-wavelength arrays for plotting
    np.savez(
        os.path.join(output_dir, "per_wavelength.npz"),
        wavelength=wavelength,
        chi2_image_only=results_image["chi2_per_wavelength"],
        chi2_image_phot=results_phot["chi2_per_wavelength"],
        mean_residual_image_only=results_image["residuals"]["mean_residual"],
        mean_residual_image_phot=results_phot["residuals"]["mean_residual"],
        std_residual_image_only=results_image["residuals"]["std_residual"],
        std_residual_image_phot=results_phot["residuals"]["std_residual"],
        mean_norm_residual_image_only=results_image["residuals"]["mean_norm_residual"],
        mean_norm_residual_image_phot=results_phot["residuals"]["mean_norm_residual"],
    )
    print(f"Saved per-wavelength arrays to {output_dir}/per_wavelength.npz")

    save_summary_plots(
        output_dir=output_dir,
        wavelength=wavelength,
        z_spec=z_spec,
        spec_mask=spec_mask,
        ivar=ivar,
        true_flux=true_flux,
        pred_image_only=pred_image_only,
        pred_image_phot=pred_image_phot,
        per_obj=per_obj,
        results_image=results_image,
        results_phot=results_phot,
    )
    print(f"Saved summary plots to {output_dir}")


if __name__ == "__main__":
    main()
