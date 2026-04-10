"""Run AION-1 cross-modal inference: predict DESI spectra from Legacy Survey inputs.

Reads data/provabgs_desi_ls.hdf5 (or custom eval HDF5), runs inference in three
modes (image-only, photometry-only, and image+photometry), saves predictions to
artifacts/predictions.hdf5.
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

        optional_fields = {
            "targetid": ("targetid", "provabgs_TARGETID", "TARGETID"),
            "ra": ("ra", "provabgs_ra", "legacysurvey_ra", "desi_ra"),
            "dec": ("dec", "provabgs_dec", "legacysurvey_dec", "desi_dec"),
            "f_fiber": ("f_fiber", "provabgs_f_fiber"),
            "fibmag_r": ("fibmag_r", "provabgs_FIBMAG_R"),
        }
        for output_key, candidate_keys in optional_fields.items():
            for candidate in candidate_keys:
                if candidate in f:
                    data[output_key] = np.array(f[candidate])
                    break

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
    codec,
    sampler,
    schedule,
    target_domain: str,
    target_num_tokens: int,
    *modalities,
) -> dict[str, torch.Tensor]:
    """Generate spectrum tokens using AION's iterative MaskGIT sampler."""
    from aion.fourm.generate import init_empty_target_modality, init_full_input_modality

    token_dict = codec.encode(*modalities)
    mod_dict: dict[str, dict[str, torch.Tensor]] = {}

    for token_key, tensor in token_dict.items():
        mod_dict[token_key] = {"tensor": tensor}
        init_full_input_modality(mod_dict, sampler.model.modality_info, token_key, codec.device)

    init_empty_target_modality(
        mod_dict,
        sampler.model.modality_info,
        target_domain,
        batch_size=list(token_dict.values())[0].shape[0],
        num_tokens=target_num_tokens,
        device=codec.device,
    )

    generated = sampler.generate(mod_dict, schedule, verbose=False)
    return {target_domain: generated[target_domain]["tensor"]}


def build_maskgit_schedule(
    model,
    cond_domains: list[str],
    target_domain: str,
    target_num_tokens: int,
    decoding_steps: int,
):
    from aion.fourm.generate import build_chained_generation_schedules

    return build_chained_generation_schedules(
        cond_domains=cond_domains,
        target_domains=[target_domain],
        tokens_per_target=[target_num_tokens],
        autoregression_schemes=["maskgit"],
        decoding_steps=[decoding_steps],
        token_decoding_schedules=["cosine"],
        temps=[0.0],
        temp_schedules=["constant"],
        cfg_scales=[1.0],
        cfg_schedules=["constant"],
        modality_info=model.modality_info,
    )


def make_batched_scalar(values: np.ndarray, start: int, end: int, device: str) -> torch.Tensor:
    """Return scalar conditioning values with explicit [batch, 1] shape."""
    return torch.tensor(
        values[start:end], dtype=torch.float32, device=device
    ).reshape(-1, 1)


def build_photometry_modalities(data: dict[str, np.ndarray], start: int, end: int, device: str):
    """Build scalar photometry modalities for the requested batch."""
    from aion.modalities import LegacySurveyFluxG, LegacySurveyFluxI, LegacySurveyFluxR, LegacySurveyFluxZ

    return (
        LegacySurveyFluxG(value=make_batched_scalar(data["flux_g"], start, end, device)),
        LegacySurveyFluxR(value=make_batched_scalar(data["flux_r"], start, end, device)),
        LegacySurveyFluxI(value=make_batched_scalar(data["flux_i"], start, end, device)),
        LegacySurveyFluxZ(value=make_batched_scalar(data["flux_z"], start, end, device)),
    )


def save_ground_truth(
    data: dict[str, np.ndarray],
    gt_path: str,
    wavelength: np.ndarray,
) -> None:
    """Write ground truth once so evaluation can proceed independently of inference restarts."""
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)

    def ensure_dataset(f: h5py.File, name: str, dataset_data: np.ndarray, **kwargs) -> None:
        if name in f:
            return
        f.create_dataset(name, data=dataset_data, **kwargs)

    mode = "a" if os.path.exists(gt_path) else "w"
    if mode == "w":
        print(f"Saving ground truth to {gt_path}")
    else:
        print(f"Ensuring ground truth metadata in {gt_path}")

    with h5py.File(gt_path, mode) as f:
        ensure_dataset(f, "spectrum_flux", data["spectrum_flux"])
        ensure_dataset(f, "spectrum_ivar", data["spectrum_ivar"])
        ensure_dataset(f, "spectrum_mask", data["spectrum_mask"])
        ensure_dataset(f, "spectrum_lambda", wavelength)
        ensure_dataset(f, "z_spec", data["z_spec"])
        if "image_rgb" in data:
            ensure_dataset(f, "image_rgb", data["image_rgb"])
        ensure_dataset(f, "images", data["images"], compression="gzip", compression_opts=1)
        ensure_dataset(f, "flux_g", data["flux_g"])
        ensure_dataset(f, "flux_r", data["flux_r"])
        ensure_dataset(f, "flux_i", data["flux_i"])
        ensure_dataset(f, "flux_z", data["flux_z"])

        for key in ["targetid", "ra", "dec", "f_fiber", "fibmag_r"]:
            if key in data:
                ensure_dataset(f, key, data[key])


def prepare_prediction_store(
    output_path: str,
    model_name: str,
    wavelength: np.ndarray,
    n_objects: int,
    n_wavelength: int,
    spectrum_decoding_steps: int,
) -> h5py.File:
    """Open a checkpointable predictions store, creating datasets if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f = h5py.File(output_path, "a")

    expected_shape = (n_objects, n_wavelength)

    def ensure_prediction_dataset(name: str, dtype: np.dtype) -> None:
        if name not in f:
            f.create_dataset(name, shape=expected_shape, dtype=dtype)
            return
        if f[name].shape != expected_shape:
            raise ValueError(
                f"Existing dataset {name} has shape {f[name].shape}, expected {expected_shape}. "
                f"Use a different --output path or remove the stale predictions file."
            )

    for name, dtype in [
        ("pred_flux_image_only", np.float32),
        ("pred_flux_phot_only", np.float32),
        ("pred_flux_image_phot", np.float32),
        ("pred_mask_image_only", np.bool_),
        ("pred_mask_phot_only", np.bool_),
        ("pred_mask_image_phot", np.bool_),
        ("pred_mask", np.bool_),
    ]:
        ensure_prediction_dataset(name, dtype)

    if "wavelength" not in f:
        f.create_dataset("wavelength", data=wavelength)
    elif f["wavelength"].shape != wavelength.shape or not np.array_equal(f["wavelength"][:], wavelength):
        raise ValueError(
            "Existing predictions file has a different wavelength grid. "
            "Use a different --output path or remove the stale predictions file."
        )

    existing_model_name = f.attrs.get("model_name")
    if existing_model_name not in (None, model_name):
        raise ValueError(
            f"Existing predictions file was created for model {existing_model_name!r}, "
            f"but current run uses {model_name!r}."
        )

    existing_steps = f.attrs.get("spectrum_decoding_steps")
    if existing_steps not in (None, spectrum_decoding_steps):
        raise ValueError(
            f"Existing predictions file was created with spectrum_decoding_steps={existing_steps}, "
            f"but current run uses {spectrum_decoding_steps}."
        )

    f.attrs["model_name"] = model_name
    f.attrs["n_objects"] = n_objects
    f.attrs["n_wavelength"] = n_wavelength
    f.attrs["spectrum_decoding_steps"] = spectrum_decoding_steps
    if "image_only_next_index" not in f.attrs:
        f.attrs["image_only_next_index"] = 0
    if "image_phot_next_index" not in f.attrs:
        f.attrs["image_phot_next_index"] = 0
    if "phot_only_next_index" not in f.attrs:
        f.attrs["phot_only_next_index"] = 0
    if "image_only_done" not in f.attrs:
        f.attrs["image_only_done"] = False
    if "image_phot_done" not in f.attrs:
        f.attrs["image_phot_done"] = False
    if "phot_only_done" not in f.attrs:
        f.attrs["phot_only_done"] = False
    f.flush()
    return f


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
    from aion.fourm.generate import GenerationSampler
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
    sampler = GenerationSampler(model)

    image_only_schedule = build_maskgit_schedule(
        model=model,
        cond_domains=[LegacySurveyImage.token_key],
        target_domain=DESISpectrum.token_key,
        target_num_tokens=DESISpectrum.num_tokens,
        decoding_steps=spectrum_decoding_steps,
    )
    image_phot_schedule = build_maskgit_schedule(
        model=model,
        cond_domains=[
            LegacySurveyImage.token_key,
            LegacySurveyFluxG.token_key,
            LegacySurveyFluxR.token_key,
            LegacySurveyFluxI.token_key,
            LegacySurveyFluxZ.token_key,
        ],
        target_domain=DESISpectrum.token_key,
        target_num_tokens=DESISpectrum.num_tokens,
        decoding_steps=spectrum_decoding_steps,
    )
    phot_only_schedule = build_maskgit_schedule(
        model=model,
        cond_domains=[
            LegacySurveyFluxG.token_key,
            LegacySurveyFluxR.token_key,
            LegacySurveyFluxI.token_key,
            LegacySurveyFluxZ.token_key,
        ],
        target_domain=DESISpectrum.token_key,
        target_num_tokens=DESISpectrum.num_tokens,
        decoding_steps=spectrum_decoding_steps,
    )

    images = data["images"]
    # Use wavelength from first spectrum (shared grid)
    wavelength = data["spectrum_lambda"]
    if wavelength.ndim == 2:
        wavelength = wavelength[0]  # all rows share the same grid

    N = images.shape[0]
    n_wave = len(wavelength)
    wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32, device=device)
    gt_path = output_path.replace("predictions", "ground_truth")

    save_ground_truth(data, gt_path, wavelength)
    pred_store = prepare_prediction_store(
        output_path=output_path,
        model_name=model_name,
        wavelength=wavelength,
        n_objects=N,
        n_wavelength=n_wave,
        spectrum_decoding_steps=spectrum_decoding_steps,
    )

    print(f"Running inference on {N} galaxies, batch_size={batch_size}")
    print(f"Spectrum decoding: MaskGIT, {spectrum_decoding_steps} steps")

    # ---------- Mode 1: Image only ----------
    image_only_done = bool(pred_store.attrs["image_only_done"])
    image_only_start = int(pred_store.attrs["image_only_next_index"])
    if image_only_done:
        print("\n--- Mode 1: Image only ---")
        print("  Checkpoint complete, skipping.")
    else:
        print("\n--- Mode 1: Image only ---")
        if image_only_start > 0:
            print(f"  Resuming from object {image_only_start}/{N}")
        t0 = time.time()

        for start in range(image_only_start, N, batch_size):
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
                    codec,
                    sampler,
                    image_only_schedule,
                    DESISpectrum.token_key,
                    DESISpectrum.num_tokens,
                    image_mod,
                )

            wl = wavelength_tensor.unsqueeze(0).expand(B, -1).contiguous()
            pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)
            pred_flux = pred_spectrum.flux.cpu().numpy()
            pred_mask = pred_spectrum.mask.cpu().numpy()

            pred_store["pred_flux_image_only"][start:end] = pred_flux
            pred_store["pred_mask_image_only"][start:end] = pred_mask
            pred_store["pred_mask"][start:end] = pred_mask
            pred_store.attrs["image_only_next_index"] = end
            pred_store.flush()

        pred_store.attrs["image_only_next_index"] = N
        pred_store.attrs["image_only_done"] = True
        pred_store.flush()
        t1 = time.time()
        print(f"  Image-only inference: {t1 - t0:.1f}s ({(t1 - t0) / N * 1000:.1f}ms/galaxy)")

    # ---------- Mode 2: Photometry only ----------
    phot_only_done = bool(pred_store.attrs["phot_only_done"])
    phot_only_start = int(pred_store.attrs["phot_only_next_index"])
    if phot_only_done:
        print("\n--- Mode 2: Photometry only ---")
        print("  Checkpoint complete, skipping.")
    else:
        print("\n--- Mode 2: Photometry only ---")
        if phot_only_start > 0:
            print(f"  Resuming from object {phot_only_start}/{N}")
        t0 = time.time()

        for start in range(phot_only_start, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            if start % (batch_size * 10) == 0:
                print(f"  Batch {start // batch_size + 1}/{(N + batch_size - 1) // batch_size}")

            fg, fr, fi, fz = build_photometry_modalities(data, start, end, device)

            with torch.no_grad():
                pred_tokens = generate_spectrum_tokens(
                    codec,
                    sampler,
                    phot_only_schedule,
                    DESISpectrum.token_key,
                    DESISpectrum.num_tokens,
                    fg,
                    fr,
                    fi,
                    fz,
                )

            wl = wavelength_tensor.unsqueeze(0).expand(B, -1).contiguous()
            pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)
            pred_flux = pred_spectrum.flux.cpu().numpy()
            pred_mask = pred_spectrum.mask.cpu().numpy()

            pred_store["pred_flux_phot_only"][start:end] = pred_flux
            pred_store["pred_mask_phot_only"][start:end] = pred_mask
            pred_store.attrs["phot_only_next_index"] = end
            pred_store.flush()

        pred_store.attrs["phot_only_next_index"] = N
        pred_store.attrs["phot_only_done"] = True
        pred_store.flush()
        t1 = time.time()
        print(f"  Phot-only inference: {t1 - t0:.1f}s ({(t1 - t0) / N * 1000:.1f}ms/galaxy)")

    # ---------- Mode 3: Image + photometry ----------
    image_phot_done = bool(pred_store.attrs["image_phot_done"])
    image_phot_start = int(pred_store.attrs["image_phot_next_index"])
    if image_phot_done:
        print("\n--- Mode 3: Image + photometry ---")
        print("  Checkpoint complete, skipping.")
    else:
        print("\n--- Mode 3: Image + photometry ---")
        if image_phot_start > 0:
            print(f"  Resuming from object {image_phot_start}/{N}")
        t0 = time.time()

        for start in range(image_phot_start, N, batch_size):
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
            fg, fr, fi, fz = build_photometry_modalities(data, start, end, device)

            with torch.no_grad():
                pred_tokens = generate_spectrum_tokens(
                    codec,
                    sampler,
                    image_phot_schedule,
                    DESISpectrum.token_key,
                    DESISpectrum.num_tokens,
                    image_mod,
                    fg,
                    fr,
                    fi,
                    fz,
                )

            wl = wavelength_tensor.unsqueeze(0).expand(B, -1).contiguous()
            pred_spectrum = codec.decode(pred_tokens, DESISpectrum, wavelength=wl)
            pred_flux = pred_spectrum.flux.cpu().numpy()
            pred_mask = pred_spectrum.mask.cpu().numpy()

            pred_store["pred_flux_image_phot"][start:end] = pred_flux
            pred_store["pred_mask_image_phot"][start:end] = pred_mask
            pred_store.attrs["image_phot_next_index"] = end
            pred_store.flush()

        pred_store.attrs["image_phot_next_index"] = N
        pred_store.attrs["image_phot_done"] = True
        pred_store.flush()
        t1 = time.time()
        print(f"  Image+phot inference: {t1 - t0:.1f}s ({(t1 - t0) / N * 1000:.1f}ms/galaxy)")

    pred_store.close()
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
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config for this run",
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
        batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32),
        device=cfg.get("device", "cuda"),
        spectrum_decoding_steps=cfg.get("spectrum_decoding_steps", 12),
    )


if __name__ == "__main__":
    main()
