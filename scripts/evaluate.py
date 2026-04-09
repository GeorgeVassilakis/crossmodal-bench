"""Compute Tier 1 spectral fidelity metrics and benchmark baselines.

Reads artifacts/ground_truth.hdf5 plus model predictions and optional
artifacts/oracle.hdf5, then writes metrics.json, metrics_per_object.parquet,
per_wavelength.npz, and summary plots.
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
PHOTOMETRY_KEYS = ("flux_g", "flux_r", "flux_i", "flux_z")


def save_summary_plots(
    output_dir: str,
    wavelength: np.ndarray,
    z_spec: np.ndarray,
    spec_mask: np.ndarray,
    ivar: np.ndarray,
    true_flux: np.ndarray,
    per_obj: pd.DataFrame,
    plot_methods: list[dict[str, object]],
) -> None:
    """Write a compact set of static plots alongside the scalar metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_styles = {
        "image_only": {"color": "steelblue"},
        "image_phot": {"color": "darkorange"},
        "oracle": {"color": "seagreen"},
        "baseline_mean": {"color": "firebrick"},
        "baseline_phot_nn": {"color": "mediumpurple"},
    }

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    wav = wavelength

    # Plot 1: Per-wavelength chi2 and normalized residual bias
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for method in plot_methods:
        key = str(method["key"])
        label = str(method["display_name"])
        result = method["results"]
        style = method_styles.get(key, {})
        ax1.plot(
            wav,
            result["chi2_per_wavelength"],
            lw=0.8,
            label=label,
            **style,
        )
    ax1.axhline(1.0, color="black", ls="--", lw=0.5, alpha=0.5)
    for boundary in CAMERA_BOUNDARIES_A:
        ax1.axvline(boundary, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax1.set_ylabel("Mean reduced chi2")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Per-Wavelength Reduced Chi-Squared")

    for method in plot_methods:
        key = str(method["key"])
        result = method["results"]
        style = method_styles.get(key, {})
        ax2.plot(
            wav,
            result["residuals"]["mean_norm_residual"],
            lw=0.6,
            alpha=0.85,
            **style,
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
    histogram_methods = [method for method in plot_methods if method["key"] in {"image_only", "image_phot"}]
    if len(histogram_methods) < 2:
        histogram_methods = plot_methods[: min(2, len(plot_methods))]
    z_bin_list = sorted(set(per_obj["z_bin"].values))
    n_bins = len(z_bin_list)
    fig, axes = plt.subplots(1, n_bins, figsize=(4 * n_bins, 3.5), sharey=True)
    if n_bins == 1:
        axes = [axes]

    bins = np.logspace(-1, 2.5, 50)
    for ax, zbin in zip(axes, z_bin_list):
        mask = per_obj["z_bin"] == zbin
        for method in histogram_methods:
            key = str(method["key"])
            col = f"chi2_{key}"
            if col not in per_obj:
                continue
            style = method_styles.get(key, {})
            ax.hist(
                per_obj.loc[mask, col].dropna(),
                bins=bins,
                alpha=0.45,
                label=str(method["display_name"]),
                **style,
            )
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
    if {"chi2_image_only", "chi2_image_phot"}.issubset(per_obj.columns):
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
    residual_method = next(
        (method for method in plot_methods if method["key"] == "image_only"),
        plot_methods[0] if plot_methods else None,
    )
    if residual_method is None:
        return

    residual_flux = residual_method["pred_flux"]
    residual_title = str(residual_method["display_name"])
    residual_filename = str(residual_method["key"])
    sort_idx = np.argsort(z_spec)
    valid = ~spec_mask & (ivar > 0)
    norm_resid = (residual_flux - true_flux) * np.sqrt(np.where(valid, ivar, 0))
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
    ax.set_title(f"Normalized Residual Heatmap ({residual_title})")
    plt.colorbar(im, ax=ax, label="Normalized residual", shrink=0.8)

    for line_rest in REST_FRAME_LINES.values():
        z_range = np.linspace(z_spec[display_idx[0]], z_spec[display_idx[-1]], 100)
        obs_wav = line_rest * (1 + z_range)
        in_range = (obs_wav > wav[0]) & (obs_wav < wav[-1])
        ax.plot(obs_wav[in_range], z_range[in_range], "k--", lw=0.5, alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"residual_heatmap_{residual_filename}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_line_mask(wavelength: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    """Create a boolean mask [N, L] that is True at emission/absorption line locations."""
    n_obj = len(z_spec)
    n_wave = len(wavelength)
    line_mask = np.zeros((n_obj, n_wave), dtype=bool)

    for line_rest in REST_FRAME_LINES.values():
        for i in range(n_obj):
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
    n_obj = pred.shape[0]
    r2_values = np.full(n_obj, np.nan)

    for i in range(n_obj):
        fit_mask = ~spec_mask[i] & ~line_mask[i]
        if fit_mask.sum() < poly_degree + 10:
            continue

        x = wavelength[fit_mask]
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

    norm_residual = residual * np.sqrt(np.where(valid, ivar, np.nan))

    return {
        "mean_residual": np.nanmean(residual, axis=0),
        "std_residual": np.nanstd(residual, axis=0),
        "mean_norm_residual": np.nanmean(norm_residual, axis=0),
        "std_norm_residual": np.nanstd(norm_residual, axis=0),
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
    """Summary table of metrics per stratum."""
    unique_strata = sorted(set(strata))
    rows = []
    for label in list(unique_strata) + ["ALL"]:
        if label == "ALL":
            mask = np.ones(len(chi2_per_obj), dtype=bool)
        else:
            mask = strata == label

        n_obj = mask.sum()
        if n_obj == 0:
            continue

        chi2_vals = chi2_per_obj[mask]
        r2_vals = cont_r2[mask]

        rows.append({
            "stratum": label,
            "n_objects": int(n_obj),
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

    combined_mask = spec_mask | pred_mask

    chi2_lam = per_wavelength_chi2(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Median per-wavelength chi2: {np.nanmedian(chi2_lam):.3f}")

    chi2_obj = per_object_chi2(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Median per-object chi2: {np.nanmedian(chi2_obj):.3f}")
    print(f"  Fraction chi2 < {chi2_threshold}: {np.nanmean(chi2_obj < chi2_threshold):.3f}")

    print("  Computing continuum R2...")
    cont_r2_vals = continuum_r2(
        pred_flux, true_flux, wavelength, combined_mask, z_spec, poly_degree
    )
    print(f"  Median continuum R2: {np.nanmedian(cont_r2_vals):.3f}")

    residuals = residual_analysis(pred_flux, true_flux, ivar, combined_mask)
    print(f"  Mean normalized residual (bias): {np.nanmean(residuals['mean_norm_residual']):.4f}")

    summary = stratified_summary(chi2_obj, cont_r2_vals, strata, chi2_threshold)
    print(f"\n  Stratified summary:\n{summary.to_string(index=False)}")

    return {
        "chi2_per_wavelength": chi2_lam,
        "chi2_per_object": chi2_obj,
        "continuum_r2": cont_r2_vals,
        "residuals": residuals,
        "summary": summary,
    }


def validate_array_shape(name: str, array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    """Check that loaded arrays match the expected shape."""
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {expected_shape}")


def validate_wavelength_grid(name: str, array: np.ndarray, wavelength: np.ndarray) -> None:
    """Check that a stored wavelength grid matches ground truth."""
    if array.shape != wavelength.shape or not np.array_equal(array, wavelength):
        raise ValueError(
            f"{name} has wavelength grid shape {array.shape}, expected {wavelength.shape} matching ground truth"
        )


def build_leave_one_out_mean_baseline(
    true_flux: np.ndarray,
    ivar: np.ndarray,
    spec_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each spectrum with the leave-one-out global mean spectrum."""
    valid = ~spec_mask & (ivar > 0)
    valid_float = valid.astype(np.float32)
    masked_flux = np.where(valid, true_flux, 0.0).astype(np.float32)

    flux_sum = masked_flux.sum(axis=0, dtype=np.float64)
    valid_count = valid_float.sum(axis=0, dtype=np.float64)

    leave_one_out_sum = flux_sum[None, :] - masked_flux
    leave_one_out_count = valid_count[None, :] - valid_float

    pred_mask = leave_one_out_count <= 0
    pred_flux = np.zeros_like(true_flux, dtype=np.float32)
    np.divide(
        leave_one_out_sum,
        leave_one_out_count,
        out=pred_flux,
        where=~pred_mask,
    )
    return pred_flux, pred_mask


def build_photometry_nn_baseline(
    true_flux: np.ndarray,
    spec_mask: np.ndarray,
    photometry: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve each spectrum from its nearest photometric neighbor, excluding itself."""
    features = np.stack([photometry[key] for key in PHOTOMETRY_KEYS], axis=1).astype(np.float64)
    features = np.sign(features) * np.log10(1.0 + np.abs(features))

    col_medians = np.nanmedian(features, axis=0)
    for j in range(features.shape[1]):
        bad = ~np.isfinite(features[:, j])
        if np.any(bad):
            features[bad, j] = col_medians[j]

    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    features = (features - feature_mean) / feature_std

    sq_norm = np.sum(features**2, axis=1, keepdims=True)
    dist2 = sq_norm + sq_norm.T - 2.0 * features @ features.T
    dist2 = np.maximum(dist2, 0.0)
    np.fill_diagonal(dist2, np.inf)

    nn_index = np.argmin(dist2, axis=1)
    pred_flux = true_flux[nn_index].copy().astype(np.float32)
    pred_mask = spec_mask[nn_index].copy().astype(bool)
    return pred_flux, pred_mask, nn_index.astype(np.int32)


def metrics_dict_from_results(results: dict, chi2_threshold: float) -> dict:
    """Convert an evaluation result bundle into a JSON-serializable summary."""
    return {
        "median_chi2": float(np.nanmedian(results["chi2_per_object"])),
        "mean_chi2": float(np.nanmean(results["chi2_per_object"])),
        "frac_good": float(np.nanmean(results["chi2_per_object"] < chi2_threshold)),
        "median_continuum_r2": float(np.nanmedian(results["continuum_r2"])),
        "summary": results["summary"].to_dict(orient="records"),
    }


def compute_normalized_skill(
    baseline_chi2: float,
    model_chi2: float,
    oracle_chi2: float,
) -> float:
    """Return normalized improvement over a baseline toward the oracle."""
    if not np.isfinite(baseline_chi2) or not np.isfinite(model_chi2) or not np.isfinite(oracle_chi2):
        return float("nan")
    denominator = baseline_chi2 - oracle_chi2
    if denominator <= 0 or np.isclose(denominator, 0.0):
        return float("nan")
    return float((baseline_chi2 - model_chi2) / denominator)


def build_normalized_skill_table(
    per_obj: pd.DataFrame,
    strata: np.ndarray,
    model_keys: list[str],
    baseline_keys: list[str],
    oracle_key: str = "oracle",
) -> pd.DataFrame:
    """Summarize median-chi2 skill for each model relative to each baseline."""
    oracle_col = f"chi2_{oracle_key}"
    if oracle_col not in per_obj:
        return pd.DataFrame(columns=[
            "stratum",
            "method",
            "baseline",
            "baseline_median_chi2",
            "model_median_chi2",
            "oracle_median_chi2",
            "normalized_skill",
        ])

    rows = []
    labels = ["ALL"] + sorted(set(strata))
    for label in labels:
        if label == "ALL":
            mask = np.ones(len(per_obj), dtype=bool)
        else:
            mask = strata == label

        oracle_median = float(np.nanmedian(per_obj.loc[mask, oracle_col]))
        for baseline_key in baseline_keys:
            baseline_col = f"chi2_{baseline_key}"
            if baseline_col not in per_obj:
                continue
            baseline_median = float(np.nanmedian(per_obj.loc[mask, baseline_col]))
            for model_key in model_keys:
                model_col = f"chi2_{model_key}"
                if model_col not in per_obj:
                    continue
                model_median = float(np.nanmedian(per_obj.loc[mask, model_col]))
                rows.append({
                    "stratum": label,
                    "method": model_key,
                    "baseline": baseline_key,
                    "baseline_median_chi2": baseline_median,
                    "model_median_chi2": model_median,
                    "oracle_median_chi2": oracle_median,
                    "normalized_skill": compute_normalized_skill(
                        baseline_median,
                        model_median,
                        oracle_median,
                    ),
                })

    return pd.DataFrame(rows)


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
        "--oracle", default="artifacts/oracle.hdf5"
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
    oracle_path = os.path.join(repo_root, args.oracle)
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading ground truth from {gt_path}")
    with h5py.File(gt_path, "r") as f:
        true_flux = f["spectrum_flux"][:]
        ivar = f["spectrum_ivar"][:]
        spec_mask = f["spectrum_mask"][:].astype(bool)
        wavelength = f["spectrum_lambda"][:]
        z_spec = f["z_spec"][:]
        photometry = {
            key: f[key][:] for key in PHOTOMETRY_KEYS if key in f
        }

    if wavelength.ndim == 2:
        wavelength = wavelength[0]

    expected_shape = true_flux.shape
    validate_array_shape("spectrum_ivar", ivar, expected_shape)
    validate_array_shape("spectrum_mask", spec_mask, expected_shape)

    print(f"Loading predictions from {pred_path}")
    method_specs: list[dict[str, object]] = []
    with h5py.File(pred_path, "r") as f:
        if "image_only_done" in f.attrs and not bool(f.attrs["image_only_done"]):
            raise RuntimeError("Image-only predictions are incomplete; rerun inference before evaluation.")
        if "image_phot_done" in f.attrs and not bool(f.attrs["image_phot_done"]):
            raise RuntimeError("Image+phot predictions are incomplete; rerun inference before evaluation.")
        if "wavelength" in f:
            validate_wavelength_grid("predictions", f["wavelength"][:], wavelength)

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

    validate_array_shape("pred_flux_image_only", pred_image_only, expected_shape)
    validate_array_shape("pred_flux_image_phot", pred_image_phot, expected_shape)
    validate_array_shape("pred_mask_image_only", pred_mask_image_only, expected_shape)
    validate_array_shape("pred_mask_image_phot", pred_mask_image_phot, expected_shape)

    method_specs.extend([
        {
            "key": "image_only",
            "display_name": "Image Only",
            "pred_flux": pred_image_only,
            "pred_mask": pred_mask_image_only,
        },
        {
            "key": "image_phot",
            "display_name": "Image + Photometry",
            "pred_flux": pred_image_phot,
            "pred_mask": pred_mask_image_phot,
        },
    ])

    if os.path.exists(oracle_path):
        print(f"Loading oracle from {oracle_path}")
        with h5py.File(oracle_path, "r") as f:
            if "done" in f.attrs and not bool(f.attrs["done"]):
                raise RuntimeError("Oracle artifact is incomplete; rerun oracle generation before evaluation.")
            if "wavelength" in f:
                validate_wavelength_grid("oracle", f["wavelength"][:], wavelength)
            pred_flux_oracle = f["pred_flux_oracle"][:]
            if "pred_mask_oracle" in f:
                pred_mask_oracle = f["pred_mask_oracle"][:].astype(bool)
            else:
                pred_mask_oracle = np.zeros_like(spec_mask, dtype=bool)

        validate_array_shape("pred_flux_oracle", pred_flux_oracle, expected_shape)
        validate_array_shape("pred_mask_oracle", pred_mask_oracle, expected_shape)
        method_specs.append({
            "key": "oracle",
            "display_name": "Codec Oracle",
            "pred_flux": pred_flux_oracle,
            "pred_mask": pred_mask_oracle,
        })
    else:
        print(f"Oracle file not found at {oracle_path}; skipping oracle comparisons.")

    print("Building CPU baselines...")
    pred_flux_mean, pred_mask_mean = build_leave_one_out_mean_baseline(
        true_flux=true_flux,
        ivar=ivar,
        spec_mask=spec_mask,
    )
    method_specs.append({
        "key": "baseline_mean",
        "display_name": "Mean Spectrum Baseline",
        "pred_flux": pred_flux_mean,
        "pred_mask": pred_mask_mean,
    })

    phot_nn_index = None
    missing_phot = [key for key in PHOTOMETRY_KEYS if key not in photometry]
    if missing_phot:
        print(f"Skipping photometry NN baseline; missing photometry columns: {missing_phot}")
    else:
        pred_flux_phot_nn, pred_mask_phot_nn, phot_nn_index = build_photometry_nn_baseline(
            true_flux=true_flux,
            spec_mask=spec_mask,
            photometry=photometry,
        )
        method_specs.append({
            "key": "baseline_phot_nn",
            "display_name": "Photometry NN Baseline",
            "pred_flux": pred_flux_phot_nn,
            "pred_mask": pred_mask_phot_nn,
        })

    strata = classify_redshift_bin(z_spec)
    print(f"Evaluating {len(true_flux)} objects...")
    unique, counts = np.unique(strata, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    poly_degree = cfg.get("poly_degree", 7)
    chi2_threshold = cfg.get("chi2_good_threshold", 3.0)

    results_by_key: dict[str, dict] = {}
    for spec in method_specs:
        results_by_key[spec["key"]] = evaluate_mode(
            pred_flux=spec["pred_flux"],
            pred_mask=spec["pred_mask"],
            true_flux=true_flux,
            ivar=ivar,
            spec_mask=spec_mask,
            wavelength=wavelength,
            z_spec=z_spec,
            strata=strata,
            mode_name=spec["display_name"],
            poly_degree=poly_degree,
            chi2_threshold=chi2_threshold,
        )

    plot_methods = [
        {
            "key": spec["key"],
            "display_name": spec["display_name"],
            "pred_flux": spec["pred_flux"],
            "results": results_by_key[spec["key"]],
        }
        for spec in method_specs
    ]

    metrics = {
        spec["key"]: metrics_dict_from_results(results_by_key[spec["key"]], chi2_threshold)
        for spec in method_specs
    }

    per_obj = pd.DataFrame({
        "z_bin": strata,
        "z_spec": z_spec,
    })
    for spec in method_specs:
        key = spec["key"]
        per_obj[f"chi2_{key}"] = results_by_key[key]["chi2_per_object"]
        per_obj[f"cont_r2_{key}"] = results_by_key[key]["continuum_r2"]
    if phot_nn_index is not None:
        per_obj["phot_nn_index"] = phot_nn_index

    baseline_keys = [spec["key"] for spec in method_specs if str(spec["key"]).startswith("baseline_")]
    model_keys = [key for key in ["image_only", "image_phot"] if key in results_by_key]
    skill_table = build_normalized_skill_table(
        per_obj=per_obj,
        strata=strata,
        model_keys=model_keys,
        baseline_keys=baseline_keys,
    )
    metrics["normalized_skill"] = {
        "metric": "median_per_object_chi2",
        "rows": skill_table.to_dict(orient="records"),
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    per_obj_path = os.path.join(output_dir, "metrics_per_object.parquet")
    per_obj.to_parquet(per_obj_path, index=False)
    print(f"Saved per-object metrics to {per_obj_path}")

    per_wavelength_payload: dict[str, np.ndarray] = {"wavelength": wavelength}
    for spec in method_specs:
        key = str(spec["key"])
        result = results_by_key[key]
        per_wavelength_payload[f"chi2_{key}"] = result["chi2_per_wavelength"]
        per_wavelength_payload[f"mean_residual_{key}"] = result["residuals"]["mean_residual"]
        per_wavelength_payload[f"std_residual_{key}"] = result["residuals"]["std_residual"]
        per_wavelength_payload[f"mean_norm_residual_{key}"] = result["residuals"]["mean_norm_residual"]
        per_wavelength_payload[f"std_norm_residual_{key}"] = result["residuals"]["std_norm_residual"]

    per_wavelength_path = os.path.join(output_dir, "per_wavelength.npz")
    np.savez(per_wavelength_path, **per_wavelength_payload)
    print(f"Saved per-wavelength arrays to {per_wavelength_path}")

    save_summary_plots(
        output_dir=output_dir,
        wavelength=wavelength,
        z_spec=z_spec,
        spec_mask=spec_mask,
        ivar=ivar,
        true_flux=true_flux,
        per_obj=per_obj,
        plot_methods=plot_methods,
    )
    print(f"Saved summary plots to {output_dir}")


if __name__ == "__main__":
    main()
