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

        if "legacysurvey_image_scale" in f:
            data["image_scale"] = np.array(f["legacysurvey_image_scale"])

        if "legacysurvey_image_band" in f:
            data["image_band"] = np.array(f["legacysurvey_image_band"])

        # RGB images for visualization (if present)
        if "legacysurvey_image_rgb" in f:
            data["image_rgb"] = np.array(f["legacysurvey_image_rgb"])

    print(f"  Loaded {data['images'].shape[0]} galaxies")
    print(f"  Image shape: {data['images'].shape}")
    print(f"  Spectrum shape: {data['spectrum_flux'].shape}")
    print(f"  Wavelength range: {data['spectrum_lambda'].min():.0f} - {data['spectrum_lambda'].max():.0f} A")
    if "image_scale" in data:
        scale = data["image_scale"]
        print(
            "  Image scale stats:"
            f" shape={scale.shape}, dtype={scale.dtype},"
            f" min={scale.min():.6g}, max={scale.max():.6g}, mean={scale.mean():.6g}"
        )

    return data


def generate_spectrum_tokens(
    model,
    codec,
    target_modality,
    device: str,
    decoding_steps: int,
    *modalities,
) -> dict[str, torch.Tensor]:
    """Generate spectrum tokens using AION's iterative MaskGIT sampler."""
    from aion.fourm.generate import (
        GenerationSampler,
        build_chained_generation_schedules,
        init_empty_target_modality,
        init_full_input_modality,
    )

    token_dict = codec.encode(*modalities)
    mod_dict: dict[str, dict[str, torch.Tensor]] = {}

    for token_key, tensor in token_dict.items():
        mod_dict[token_key] = {"tensor": tensor}
        init_full_input_modality(mod_dict, model.modality_info, token_key, device)

    init_empty_target_modality(
        mod_dict,
        model.modality_info,
        target_modality.token_key,
        batch_size=list(token_dict.values())[0].shape[0],
        num_tokens=target_modality.num_tokens,
        device=device,
    )

    schedule = build_chained_generation_schedules(
        cond_domains=list(token_dict.keys()),
        target_domains=[target_modality.token_key],
        tokens_per_target=[target_modality.num_tokens],
        autoregression_schemes=["maskgit"],
        decoding_steps=[decoding_steps],
        token_decoding_schedules=["cosine"],
        temps=[0.0],
        temp_schedules=["constant"],
        cfg_scales=[1.0],
        cfg_schedules=["constant"],
        modality_info=model.modality_info,
    )

    sampler = GenerationSampler(model)
    generated = sampler.generate(mod_dict, schedule, verbose=False)
    return {target_modality.token_key: generated[target_modality.token_key]["tensor"]}


def run_inference(
    data: dict[str, np.ndarray],
    output_path: str,
    model_name: str = "polymathic-ai/aion-base",
    batch_size: int = 32,
    device: str = "cuda",
    spectrum_decoding_steps: int = 12,
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
    print(f"Spectrum decoding: MaskGIT, {spectrum_decoding_steps} steps")

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
        with torch.no_grad():
            pred_tokens = generate_spectrum_tokens(
                model,
                codec,
                DESISpectrum,
                device,
                spectrum_decoding_steps,
                image_mod,
            )

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

        with torch.no_grad():
            pred_tokens = generate_spectrum_tokens(
                model,
                codec,
                DESISpectrum,
                device,
                spectrum_decoding_steps,
                image_mod,
                fg,
                fr,
                fi,
                fz,
            )

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
        spectrum_decoding_steps=cfg.get("spectrum_decoding_steps", 12),
    )


if __name__ == "__main__":
    main()
