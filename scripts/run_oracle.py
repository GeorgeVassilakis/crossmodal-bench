"""Run a codec round-trip oracle for the benchmark spectra.

Reads data/provabgs_desi_ls.hdf5, encodes each DESI spectrum with the AION
spectrum codec, decodes it back to flux space, and saves resumable outputs to
artifacts/oracle.hdf5.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
import yaml

from run_inference import load_provabgs_data, save_ground_truth


def prepare_oracle_store(
    output_path: str,
    wavelength: np.ndarray,
    n_objects: int,
    n_wavelength: int,
    device: str,
) -> h5py.File:
    """Open a checkpointable oracle store, creating datasets if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f = h5py.File(output_path, "a")

    expected_shape = (n_objects, n_wavelength)

    def ensure_dataset(name: str, dtype: np.dtype) -> None:
        if name not in f:
            f.create_dataset(name, shape=expected_shape, dtype=dtype)
            return
        if f[name].shape != expected_shape:
            raise ValueError(
                f"Existing dataset {name} has shape {f[name].shape}, expected {expected_shape}. "
                f"Use a different --output path or remove the stale oracle file."
            )

    ensure_dataset("pred_flux_oracle", np.float32)
    ensure_dataset("pred_mask_oracle", np.bool_)

    if "wavelength" not in f:
        f.create_dataset("wavelength", data=wavelength)
    elif f["wavelength"].shape != wavelength.shape or not np.array_equal(f["wavelength"][:], wavelength):
        raise ValueError(
            "Existing oracle file has a different wavelength grid. "
            "Use a different --output path or remove the stale oracle file."
        )

    f.attrs["device"] = device
    f.attrs["n_objects"] = n_objects
    f.attrs["n_wavelength"] = n_wavelength
    if "next_index" not in f.attrs:
        f.attrs["next_index"] = 0
    if "done" not in f.attrs:
        f.attrs["done"] = False
    f.flush()
    return f


def run_oracle(
    data: dict[str, np.ndarray],
    output_path: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Round-trip the ground-truth spectra through the current spectrum codec."""
    from aion.codecs import CodecManager
    from aion.modalities import DESISpectrum

    spectrum_flux = data["spectrum_flux"]
    spectrum_ivar = data["spectrum_ivar"]
    spectrum_mask = data["spectrum_mask"]
    spectrum_lambda = data["spectrum_lambda"]

    if spectrum_lambda.ndim == 2:
        wavelength = spectrum_lambda[0]
    else:
        wavelength = spectrum_lambda

    n_objects, n_wavelength = spectrum_flux.shape
    gt_path = os.path.join(os.path.dirname(output_path), "ground_truth.hdf5")
    save_ground_truth(data, gt_path, wavelength)

    oracle_store = prepare_oracle_store(
        output_path=output_path,
        wavelength=wavelength,
        n_objects=n_objects,
        n_wavelength=n_wavelength,
        device=device,
    )

    if bool(oracle_store.attrs["done"]):
        print("Oracle checkpoint complete, skipping.")
        oracle_store.close()
        return

    next_index = int(oracle_store.attrs["next_index"])
    if next_index > 0:
        print(f"Resuming oracle from object {next_index}/{n_objects}")

    print(f"Running codec oracle on {n_objects} galaxies, batch_size={batch_size}")
    codec = CodecManager(device=device)

    t0 = time.time()
    for start in range(next_index, n_objects, batch_size):
        end = min(start + batch_size, n_objects)
        if start % (batch_size * 10) == 0:
            print(f"  Batch {start // batch_size + 1}/{(n_objects + batch_size - 1) // batch_size}")

        batch_flux = torch.tensor(spectrum_flux[start:end], dtype=torch.float32, device=device)
        batch_ivar = torch.tensor(spectrum_ivar[start:end], dtype=torch.float32, device=device)
        batch_mask = torch.tensor(spectrum_mask[start:end], dtype=torch.bool, device=device)
        if spectrum_lambda.ndim == 2:
            batch_wavelength = torch.tensor(
                spectrum_lambda[start:end], dtype=torch.float32, device=device
            ).contiguous()
        else:
            batch_wavelength = torch.tensor(
                wavelength, dtype=torch.float32, device=device
            ).unsqueeze(0).expand(end - start, -1).contiguous()

        spectrum_mod = DESISpectrum(
            flux=batch_flux,
            ivar=batch_ivar,
            mask=batch_mask,
            wavelength=batch_wavelength,
        )

        with torch.no_grad():
            tokens = codec.encode(spectrum_mod)
            decoded = codec.decode(tokens, DESISpectrum, wavelength=batch_wavelength)

        oracle_store["pred_flux_oracle"][start:end] = decoded.flux.cpu().numpy()
        oracle_store["pred_mask_oracle"][start:end] = decoded.mask.cpu().numpy()
        oracle_store.attrs["next_index"] = end
        oracle_store.flush()

    oracle_store.attrs["next_index"] = n_objects
    oracle_store.attrs["done"] = True
    oracle_store.flush()
    oracle_store.close()

    dt = time.time() - t0
    print(f"Oracle round-trip: {dt:.1f}s ({dt / n_objects * 1000:.1f}ms/galaxy)")
    print(f"Saved oracle artifact to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run codec oracle round-trip")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--data", default="data/provabgs_desi_ls.hdf5",
        help="Input HDF5 (provabgs_desi_ls.hdf5 or custom eval dataset)",
    )
    parser.add_argument(
        "--output", default="artifacts/oracle.hdf5", help="Output oracle HDF5"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size override"
    )
    parser.add_argument(
        "--device", default=None, help="Device override"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    config_path = os.path.join(repo_root, args.config)
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    data_path = os.path.join(repo_root, args.data)
    output_path = os.path.join(repo_root, args.output)

    print(f"Loading data from {data_path}")
    data = load_provabgs_data(data_path)

    run_oracle(
        data=data,
        output_path=output_path,
        batch_size=args.batch_size or cfg.get("batch_size", 32),
        device=args.device or cfg.get("device", "cuda"),
    )


if __name__ == "__main__":
    main()
