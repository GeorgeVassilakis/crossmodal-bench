from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PATH = REPO_ROOT / "scripts" / "evaluate.py"


def load_evaluate_module():
    module_name = "crossmodal_bench_evaluate_test"
    spec = importlib.util.spec_from_file_location(module_name, EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_fixture_files(tmp_path: Path, *, oracle_done: bool = True, preds_done: bool = True):
    n_obj = 5
    n_wave = 24
    wavelength = np.linspace(3600.0, 9800.0, n_wave, dtype=np.float32)
    z_spec = np.array([0.05, 0.15, 0.25, 0.35, 0.55], dtype=np.float32)

    base = np.linspace(0.8, 1.2, n_wave, dtype=np.float32)
    true_flux = np.stack([
        base + 0.05 * i + 0.02 * np.sin(wavelength / 300.0 + i)
        for i in range(n_obj)
    ]).astype(np.float32)
    ivar = np.full_like(true_flux, 4.0, dtype=np.float32)
    spec_mask = np.zeros_like(true_flux, dtype=bool)
    spec_mask[1, 3] = True
    spec_mask[3, 10] = True

    gt_path = tmp_path / "ground_truth.hdf5"
    with h5py.File(gt_path, "w") as f:
        f.create_dataset("spectrum_flux", data=true_flux)
        f.create_dataset("spectrum_ivar", data=ivar)
        f.create_dataset("spectrum_mask", data=spec_mask)
        f.create_dataset("spectrum_lambda", data=wavelength)
        f.create_dataset("z_spec", data=z_spec)
        f.create_dataset("flux_g", data=np.array([1.0, 1.1, 2.0, 2.1, 5.0], dtype=np.float32))
        f.create_dataset("flux_r", data=np.array([1.2, 1.3, 2.2, 2.3, 5.2], dtype=np.float32))
        f.create_dataset("flux_i", data=np.array([1.4, 1.5, 2.4, 2.5, 5.4], dtype=np.float32))
        f.create_dataset("flux_z", data=np.array([1.6, 1.7, 2.6, 2.7, 5.6], dtype=np.float32))

    pred_path = tmp_path / "predictions.hdf5"
    with h5py.File(pred_path, "w") as f:
        f.create_dataset("wavelength", data=wavelength)
        f.create_dataset("pred_flux_image_only", data=true_flux + 0.08)
        f.create_dataset("pred_flux_image_phot", data=true_flux + 0.03)
        f.create_dataset("pred_mask_image_only", data=np.zeros_like(spec_mask, dtype=bool))
        f.create_dataset("pred_mask_image_phot", data=np.zeros_like(spec_mask, dtype=bool))
        f.attrs["image_only_done"] = preds_done
        f.attrs["image_phot_done"] = preds_done

    oracle_path = tmp_path / "oracle.hdf5"
    with h5py.File(oracle_path, "w") as f:
        f.create_dataset("wavelength", data=wavelength)
        f.create_dataset("pred_flux_oracle", data=true_flux + 0.01)
        f.create_dataset("pred_mask_oracle", data=np.zeros_like(spec_mask, dtype=bool))
        f.attrs["done"] = oracle_done

    config_path = tmp_path / "config.yaml"
    config_path.write_text("poly_degree: 3\nchi2_good_threshold: 3.0\n")

    return gt_path, pred_path, oracle_path, config_path


def test_helper_functions_cover_core_benchmark_logic():
    module = load_evaluate_module()

    true_flux = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [10.0, 11.0, 12.0],
    ], dtype=np.float32)
    ivar = np.ones_like(true_flux)
    spec_mask = np.array([
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ])

    pred_mean, mask_mean = module.build_leave_one_out_mean_baseline(true_flux, ivar, spec_mask)
    expected_first = np.array([6.0, 11.0, 8.0], dtype=np.float32)
    np.testing.assert_allclose(pred_mean[0], expected_first)
    assert mask_mean.shape == spec_mask.shape

    photometry = {
        "flux_g": np.array([1.0, 1.1, 9.0], dtype=np.float32),
        "flux_r": np.array([1.2, 1.3, 9.2], dtype=np.float32),
        "flux_i": np.array([1.4, 1.5, 9.4], dtype=np.float32),
        "flux_z": np.array([1.6, 1.7, 9.6], dtype=np.float32),
    }
    pred_nn, mask_nn, nn_index = module.build_photometry_nn_baseline(true_flux, spec_mask, photometry)
    assert pred_nn.shape == true_flux.shape
    assert mask_nn.shape == spec_mask.shape
    np.testing.assert_array_equal(nn_index, np.array([1, 0, 1], dtype=np.int32))

    assert module.compute_normalized_skill(5.0, 3.0, 1.0) == pytest.approx(0.5)
    assert np.isnan(module.compute_normalized_skill(1.0, 1.2, 1.0))


def test_evaluate_main_writes_metrics_with_oracle_and_baselines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_evaluate_module()
    gt_path, pred_path, oracle_path, config_path = write_fixture_files(tmp_path)
    output_dir = tmp_path / "artifacts"

    def fake_to_parquet(self, path, index=False):
        Path(path).write_text("parquet stub\n")

    def fake_save_summary_plots(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots_stub.txt").write_text("ok\n")

    monkeypatch.setattr(module.pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    monkeypatch.setattr(module, "save_summary_plots", fake_save_summary_plots)

    argv = [
        "evaluate.py",
        "--config", str(config_path),
        "--ground-truth", str(gt_path),
        "--predictions", str(pred_path),
        "--oracle", str(oracle_path),
        "--output-dir", str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "image_only" in metrics
    assert "image_phot" in metrics
    assert "oracle" in metrics
    assert "baseline_mean" in metrics
    assert "baseline_phot_nn" in metrics
    assert "normalized_skill" in metrics
    assert metrics["oracle"]["median_chi2"] < metrics["image_only"]["median_chi2"]
    assert metrics["image_phot"]["median_chi2"] < metrics["image_only"]["median_chi2"]
    assert metrics["baseline_phot_nn"]["median_chi2"] <= metrics["baseline_mean"]["median_chi2"]
    assert metrics["normalized_skill"]["rows"]

    per_wavelength = np.load(output_dir / "per_wavelength.npz")
    assert "chi2_oracle" in per_wavelength.files
    assert "chi2_baseline_mean" in per_wavelength.files
    assert "chi2_baseline_phot_nn" in per_wavelength.files

    assert (output_dir / "metrics_per_object.parquet").exists()
    assert (output_dir / "plots_stub.txt").exists()


def test_evaluate_main_rejects_incomplete_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_evaluate_module()
    gt_path, pred_path, oracle_path, config_path = write_fixture_files(tmp_path, oracle_done=False)
    output_dir = tmp_path / "artifacts"

    monkeypatch.setattr(module.pd.DataFrame, "to_parquet", lambda self, path, index=False: None, raising=False)
    monkeypatch.setattr(module, "save_summary_plots", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config", str(config_path),
            "--ground-truth", str(gt_path),
            "--predictions", str(pred_path),
            "--oracle", str(oracle_path),
            "--output-dir", str(output_dir),
        ],
    )

    with pytest.raises(RuntimeError, match="Oracle artifact is incomplete"):
        module.main()
