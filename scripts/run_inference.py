"""Run AION-1 cross-modal inference: predict DESI spectra from Legacy Survey images.

Reads data/provabgs_desi_ls.hdf5 (or custom eval HDF5), runs inference in two
modes (image-only and image+photometry), saves predictions to artifacts/predictions.hdf5.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
import yaml


def load_provabgs_data(path: str) -> dict[str, np.ndarray]:
    """Load the provabgs_desi_ls.hdf5 file and return arrays in a standard format."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")

        data = {
            "images": np.array(f["legacysurvey_image_flux"]),
            "spectrum_flux": np.array(f["desi_spectrum_flux"]),
            "spectrum_ivar": np.array(f["desi_spectrum_ivar"]),
            "spectrum_mask": np.array(f["desi_spectrum_mask"]).astype(bool),
            "spectrum_lambda": np.array(f["desi_spectrum_lambda"]),
            "flux_g": np.array(f["legacysurvey_FLUX_G"]),
            "flux_r": np.array(f["legacysurvey_FLUX_R"]),
            "flux_i": np.array(f["legacysurvey_FLUX_I"]),
            "flux_z": np.array(f["legacysurvey_FLUX_Z"]),
            "z_spec": np.array(f["provabgs_Z_HP"]),
        }

        # RGB images for visualization (if present)
        if "legacysurvey_image_rgb" in f:
            data["image_rgb"] = np.array(f["legacysurvey_image_rgb"])

    print(f"  Loaded {data['images'].shape[0]} galaxies")
    print(f"  Image shape: {data['images'].shape}")
    print(f"  Spectrum shape: {data['spectrum_flux'].shape}")
    print(f"  Wavelength range: {data['spectrum_lambda'].min():.0f} - {data['spectrum_lambda'].max():.0f} A")

    return data


def run_inference(
    data: dict[str, np.ndarray],
    output_path: str,
    model_name: str = "polymathic-ai/aion-base",
    batch_size: int = 32,
    device: str = "cuda",
):
    from aion import AION
    from aion.codecs import CodecManager
    from aion.modalities import (
        DESISpectrum,
        LegacySurveyFluxG,
        LegacySurveyFluxI,
        LegacySurveyFluxR,
        LegacySurveyFluxZ,
        LegacySurveyImage,
    )

    # Load model and codec
    print(f"Loading model: {model_name}")
    model = AION.from_pretrained(model_name).to(device).eval()
    codec = CodecManager(device=device)

    images = data["images"]
    # Use wavelength from first spectrum (shared grid)
    wavelength = data["spectrum_lambda"]
    if wavelength.ndim == 2:
        wavelength = wavelength[0]  # all rows share the same grid

    N = images.shape[0]
    n_wave = len(wavelength)
    wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32, device=device)

    print(f"Running inference on {N} galaxies, batch_size={batch_size}")

    # Pre-allocate output arrays
    pred_flux_image_only = np.zeros((N, n_wave), dtype=np.float32)
    pred_flux_image_phot = np.zeros((N, n_wave), dtype=np.float32)
    pred_mask = np.zeros((N, n_wave), dtype=bool)

    # ---------- Mode 1: Image only ----------
    print("\n--- Mode 1: Image only ---")
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        if start % (batch_size * 10) == 0:
            print(f"  Batch {start // batch_size + 1}/{(N + batch_size - 1) // batch_size}")

        batch_images = torch.tensor(
            images[start:end], dtype=torch.float32, device=device
        )

        image_mod = LegacySurveyImage(
            flux=batch_images,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        tokens = codec.encode(image_mod)

        with torch.no_grad():
            logits = model(tokens, target_modality=DESISpectrum)

        pred_tokens = {
            "tok_spectrum_desi": logits["tok_spectrum_desi"].argmax(dim=-1)
        }

        wl = wavelength_tensor.unsqueeze(0).expand(B, -1)
        pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)

        pred_flux_image_only[start:end] = pred_spectrum.flux.cpu().numpy()
        pred_mask[start:end] = pred_spectrum.mask.cpu().numpy()

    t1 = time.time()
    print(f"  Image-only inference: {t1 - t0:.1f}s ({(t1 - t0) / N * 1000:.1f}ms/galaxy)")

    # ---------- Mode 2: Image + photometry ----------
    print("\n--- Mode 2: Image + photometry ---")
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        if start % (batch_size * 10) == 0:
            print(f"  Batch {start // batch_size + 1}/{(N + batch_size - 1) // batch_size}")

        batch_images = torch.tensor(
            images[start:end], dtype=torch.float32, device=device
        )

        image_mod = LegacySurveyImage(
            flux=batch_images,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        fg = LegacySurveyFluxG(value=torch.tensor(data["flux_g"][start:end], dtype=torch.float32, device=device))
        fr = LegacySurveyFluxR(value=torch.tensor(data["flux_r"][start:end], dtype=torch.float32, device=device))
        fi = LegacySurveyFluxI(value=torch.tensor(data["flux_i"][start:end], dtype=torch.float32, device=device))
        fz = LegacySurveyFluxZ(value=torch.tensor(data["flux_z"][start:end], dtype=torch.float32, device=device))

        tokens = codec.encode(image_mod, fg, fr, fi, fz)

        with torch.no_grad():
            logits = model(tokens, target_modality=DESISpectrum)

        pred_tokens = {
            "tok_spectrum_desi": logits["tok_spectrum_desi"].argmax(dim=-1)
        }

        wl = wavelength_tensor.unsqueeze(0).expand(B, -1)
        pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)

        pred_flux_image_phot[start:end] = pred_spectrum.flux.cpu().numpy()

    t1 = time.time()
    print(f"  Image+phot inference: {t1 - t0:.1f}s ({(t1 - t0) / N * 1000:.1f}ms/galaxy)")

    # ---------- Save predictions ----------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving predictions to {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("pred_flux_image_only", data=pred_flux_image_only)
        f.create_dataset("pred_flux_image_phot", data=pred_flux_image_phot)
        f.create_dataset("pred_mask", data=pred_mask)
        f.create_dataset("wavelength", data=wavelength)
        f.attrs["model_name"] = model_name
        f.attrs["n_objects"] = N
        f.attrs["n_wavelength"] = n_wave

    # Also save ground truth alongside for convenience
    gt_path = output_path.replace("predictions", "ground_truth")
    print(f"Saving ground truth to {gt_path}")
    with h5py.File(gt_path, "w") as f:
        f.create_dataset("spectrum_flux", data=data["spectrum_flux"])
        f.create_dataset("spectrum_ivar", data=data["spectrum_ivar"])
        f.create_dataset("spectrum_mask", data=data["spectrum_mask"])
        f.create_dataset("spectrum_lambda", data=wavelength)
        f.create_dataset("z_spec", data=data["z_spec"])
        if "image_rgb" in data:
            f.create_dataset("image_rgb", data=data["image_rgb"])
        f.create_dataset("images", data=images, compression="gzip", compression_opts=1)
        f.create_dataset("flux_g", data=data["flux_g"])
        f.create_dataset("flux_r", data=data["flux_r"])
        f.create_dataset("flux_i", data=data["flux_i"])
        f.create_dataset("flux_z", data=data["flux_z"])

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run AION-1 cross-modal inference")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--data", default="data/provabgs_desi_ls.hdf5",
        help="Input HDF5 (provabgs_desi_ls.hdf5 or custom eval dataset)",
    )
    parser.add_argument(
        "--output", default="artifacts/predictions.hdf5", help="Output predictions HDF5"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    config_path = os.path.join(repo_root, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_path = os.path.join(repo_root, args.data)
    output_path = os.path.join(repo_root, args.output)

    print(f"Loading data from {data_path}")
    data = load_provabgs_data(data_path)

    run_inference(
        data=data,
        output_path=output_path,
        model_name=cfg.get("model_name", "polymathic-ai/aion-base"),
        batch_size=cfg.get("batch_size", 32),
        device=cfg.get("device", "cuda"),
    )


if __name__ == "__main__":
    main()
