"""Run AION-1 cross-modal inference: predict DESI spectra from Legacy Survey images.

Reads data/eval_dataset.hdf5, runs inference in two modes (image-only and
image+photometry), and saves predictions to artifacts/predictions.hdf5.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
import yaml


def run_inference(
    eval_path: str,
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
    model = AION.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    codec = CodecManager(device=device)

    # Load evaluation dataset
    print(f"Loading eval dataset: {eval_path}")
    with h5py.File(eval_path, "r") as f:
        images = f["image"][:]  # [N, 4, 160, 160]
        wavelength = f["spectrum_lambda"][:]  # [7781]
        flux_g_arr = f["flux_g"][:]
        flux_r_arr = f["flux_r"][:]
        flux_i_arr = f["flux_i"][:]
        flux_z_arr = f["flux_z"][:]

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
            print(f"  Batch {start//batch_size + 1}/{(N + batch_size - 1)//batch_size}")

        batch_images = torch.tensor(
            images[start:end], dtype=torch.float32, device=device
        )

        # Encode image
        image_mod = LegacySurveyImage(
            flux=batch_images,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        tokens = codec.encode(image_mod)

        # Predict spectrum
        with torch.no_grad():
            logits = model(tokens, target_modality=DESISpectrum)

        pred_tokens = {
            "tok_spectrum_desi": logits["tok_spectrum_desi"].argmax(dim=-1)
        }

        # Decode to flux on the DESI wavelength grid
        wl = wavelength_tensor.unsqueeze(0).expand(B, -1)
        pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)

        pred_flux_image_only[start:end] = pred_spectrum.flux.cpu().numpy()
        if start == 0:
            # Save mask from first batch (should be same for all via codec)
            pred_mask[start:end] = pred_spectrum.mask.cpu().numpy()

    t1 = time.time()
    print(f"  Image-only inference: {t1 - t0:.1f}s ({(t1-t0)/N*1000:.1f}ms/galaxy)")

    # ---------- Mode 2: Image + photometry ----------
    print("\n--- Mode 2: Image + photometry ---")
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        if start % (batch_size * 10) == 0:
            print(f"  Batch {start//batch_size + 1}/{(N + batch_size - 1)//batch_size}")

        batch_images = torch.tensor(
            images[start:end], dtype=torch.float32, device=device
        )

        # Encode image + photometry
        image_mod = LegacySurveyImage(
            flux=batch_images,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        fg = LegacySurveyFluxG(value=torch.tensor(flux_g_arr[start:end], dtype=torch.float32, device=device))
        fr = LegacySurveyFluxR(value=torch.tensor(flux_r_arr[start:end], dtype=torch.float32, device=device))
        fi = LegacySurveyFluxI(value=torch.tensor(flux_i_arr[start:end], dtype=torch.float32, device=device))
        fz = LegacySurveyFluxZ(value=torch.tensor(flux_z_arr[start:end], dtype=torch.float32, device=device))

        tokens = codec.encode(image_mod, fg, fr, fi, fz)

        # Predict spectrum
        with torch.no_grad():
            logits = model(tokens, target_modality=DESISpectrum)

        pred_tokens = {
            "tok_spectrum_desi": logits["tok_spectrum_desi"].argmax(dim=-1)
        }

        wl = wavelength_tensor.unsqueeze(0).expand(B, -1)
        pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)

        pred_flux_image_phot[start:end] = pred_spectrum.flux.cpu().numpy()

    t1 = time.time()
    print(f"  Image+phot inference: {t1 - t0:.1f}s ({(t1-t0)/N*1000:.1f}ms/galaxy)")

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

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run AION-1 cross-modal inference")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--eval-dataset", default="data/eval_dataset.hdf5", help="Input eval HDF5"
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

    eval_path = os.path.join(repo_root, args.eval_dataset)
    output_path = os.path.join(repo_root, args.output)

    run_inference(
        eval_path=eval_path,
        output_path=output_path,
        model_name=cfg.get("model_name", "polymathic-ai/aion-base"),
        batch_size=cfg.get("batch_size", 32),
        device=cfg.get("device", "cuda"),
    )


if __name__ == "__main__":
    main()
