"""Build the cross-modal spectral prediction evaluation dataset.

Produces data/eval_dataset.hdf5 with ~5K matched (Legacy Survey image, DESI spectrum)
pairs from DESI DR1 main-survey galaxies that AION-1 never saw during training.

Requires access to the Flatiron cluster filesystem.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import healpy as hp
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from glob import glob

# Suppress DESI logging noise
os.environ.setdefault("DESI_LOGLEVEL", "WARNING")


# ---------------------------------------------------------------------------
# BAD_HANDLES from MMU v1 — coadd files that don't exist on the cluster mirror
# ---------------------------------------------------------------------------
BAD_HANDLES: dict[str, list[int]] = {
    "bright": [9836, 4802, 4561, 4730],
    "dark": [26535, 15051, 10844, 9913],
    "backup": [10786, 10810],
}


def _as_str_array(col) -> np.ndarray:
    """Decode a FITS byte/str column into a stripped-string numpy array."""
    out = np.empty(len(col), dtype=object)
    for i, v in enumerate(col):
        if isinstance(v, (bytes, np.bytes_)):
            out[i] = v.decode().strip()
        else:
            out[i] = str(v).strip()
    return out.astype(str)


def load_and_filter_desi_catalog(cfg: dict) -> Table:
    """Load zall-pix-iron.fits and apply selection cuts."""
    print(f"Loading DESI catalog: {cfg['desi_zcatalog']}")
    catalog = Table.read(cfg["desi_zcatalog"])
    print(f"  Full catalog: {len(catalog)} rows")

    # Core selection (matches MMU v1 exactly)
    survey = _as_str_array(catalog["SURVEY"])
    objtype = _as_str_array(catalog["OBJTYPE"])
    mask = survey == cfg.get("desi_survey", "main")
    mask &= np.asarray(catalog["MAIN_PRIMARY"]).astype(bool)
    mask &= objtype == "TGT"
    mask &= np.asarray(catalog["COADD_FIBERSTATUS"]) == 0

    # Exclude BAD_HANDLES
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    for bad_program, bad_hp in BAD_HANDLES.items():
        mask &= ~((program == bad_program) & np.isin(healpix, bad_hp))

    # Additional quality cuts for evaluation
    spectype = _as_str_array(catalog["SPECTYPE"])
    zwarn = np.asarray(catalog["ZWARN"])
    deltachi2 = np.asarray(catalog["DELTACHI2"])
    z = np.asarray(catalog["Z"])

    # Allow both GALAXY and QSO for a richer benchmark
    mask &= (spectype == "GALAXY") | (spectype == "QSO")
    mask &= zwarn == 0
    mask &= deltachi2 > cfg.get("desi_deltachi2_min", 25.0)
    mask &= z > 0.001  # exclude stars / bad fits

    catalog = catalog[mask]
    print(f"  After selection: {len(catalog)} rows")

    # Add HEALPix column at nside=16 (nest) for matching to Legacy Survey tiles
    nside = cfg.get("healpix_nside", 16)
    catalog["HEALPIX_16"] = hp.ang2pix(
        nside,
        np.asarray(catalog["TARGET_RA"]),
        np.asarray(catalog["TARGET_DEC"]),
        lonlat=True,
        nest=True,
    )

    return catalog


def find_overlapping_healpix(catalog: Table, legacy_root: str) -> list[int]:
    """Find HEALPix pixels that have both DESI targets and Legacy Survey images."""
    # Find existing Legacy Survey HDF5 files
    legacy_files = glob(os.path.join(legacy_root, "healpix=*/001-of-001.hdf5"))
    legacy_hp = set()
    for f in legacy_files:
        # Extract healpix number from path: .../healpix=1234/001-of-001.hdf5
        dirname = os.path.basename(os.path.dirname(f))
        hp_val = int(dirname.split("=")[1])
        legacy_hp.add(hp_val)
    print(f"  Legacy Survey: {len(legacy_hp)} HEALPix tiles on disk")

    # Intersect with DESI HEALPix
    desi_hp = set(np.unique(catalog["HEALPIX_16"]))
    overlap = sorted(desi_hp & legacy_hp)
    print(f"  Overlapping HEALPix pixels: {len(overlap)}")

    return overlap


def crossmatch_one_tile(
    desi_subset: Table,
    legacy_root: str,
    healpix: int,
    radius_arcsec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-match DESI targets to Legacy Survey objects in one HEALPix tile.

    Returns (desi_indices, legacy_indices, separations_arcsec) for matches
    within the given radius.
    """
    hdf5_path = os.path.join(
        legacy_root, f"healpix={healpix}", "001-of-001.hdf5"
    )
    if not os.path.exists(hdf5_path):
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    with h5py.File(hdf5_path, "r") as f:
        ls_ra = f["ra"][:]
        ls_dec = f["dec"][:]

    desi_coord = SkyCoord(
        ra=np.asarray(desi_subset["TARGET_RA"]) * u.deg,
        dec=np.asarray(desi_subset["TARGET_DEC"]) * u.deg,
    )
    ls_coord = SkyCoord(ra=ls_ra * u.deg, dec=ls_dec * u.deg)

    # Match DESI → Legacy Survey (find nearest Legacy neighbor for each DESI target)
    idx, sep, _ = desi_coord.match_to_catalog_sky(ls_coord)
    within = sep.arcsec < radius_arcsec

    desi_idx = np.where(within)[0]
    legacy_idx = idx[within]
    seps = sep.arcsec[within]

    return desi_idx, legacy_idx, seps


def classify_target_type(catalog: Table) -> np.ndarray:
    """Classify DESI targets into BGS/LRG/ELG/QSO using PROGRAM + redshift."""
    program = _as_str_array(catalog["PROGRAM"])
    spectype = _as_str_array(catalog["SPECTYPE"])
    z = np.asarray(catalog["Z"])

    types = np.full(len(catalog), "OTHER", dtype="U8")
    types[program == "bright"] = "BGS"
    types[(program == "dark") & (z < 0.5)] = "LRG"
    types[(program == "dark") & (z >= 0.5)] = "ELG"
    types[spectype == "QSO"] = "QSO"

    return types


def stratified_sample(
    indices: np.ndarray,
    target_types: np.ndarray,
    n_total: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Return a stratified subsample of indices, balanced across target types."""
    rng = np.random.default_rng(seed)
    strata = ["BGS", "LRG", "ELG", "QSO"]
    per_stratum = n_total // len(strata)

    selected = []
    for s in strata:
        mask = target_types == s
        pool = indices[mask]
        n = min(per_stratum, len(pool))
        if n > 0:
            chosen = rng.choice(pool, size=n, replace=False)
            selected.append(chosen)
            print(f"  {s}: sampled {n} / {len(pool)} available")

    return np.concatenate(selected)


def fetch_desi_spectra(
    catalog: Table,
    coadd_root: str,
) -> dict[str, np.ndarray]:
    """Read coadd FITS files and extract combined spectra for the given targets.

    Groups targets by (SURVEY, PROGRAM, HEALPIX) and reads each coadd file once.
    """
    import desispec.io
    from desispec import coaddition
    from scipy.optimize import curve_fit

    def _gauss(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    survey = _as_str_array(catalog["SURVEY"])
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    target_ids = np.asarray(catalog["TARGETID"])

    # Group by (survey, program, healpix)
    groups: dict[tuple, list[int]] = {}
    for i in range(len(catalog)):
        key = (survey[i], program[i], int(healpix[i]))
        groups.setdefault(key, []).append(i)

    # Pre-allocate output arrays (we'll figure out wavelength length from first file)
    all_flux = {}
    all_ivar = {}
    all_mask = {}
    wavelength = None

    n_groups = len(groups)
    for gi, ((s, p, h), row_indices) in enumerate(groups.items()):
        coadd_path = os.path.join(coadd_root, f"coadd-{s}-{p}-{h}.fits")
        if not os.path.exists(coadd_path):
            print(f"  WARNING: missing coadd file {coadd_path}, skipping {len(row_indices)} targets")
            continue

        tids = target_ids[row_indices]
        if gi % 50 == 0:
            print(f"  Processing coadd {gi+1}/{n_groups}: {coadd_path} ({len(tids)} targets)")

        try:
            spectra = desispec.io.read_spectra(coadd_path).select(targets=tids)
            combined = coaddition.coadd_cameras(spectra)
        except Exception as e:
            print(f"  WARNING: failed to process {coadd_path}: {e}")
            continue

        wav = np.asarray(combined.wave["brz"], dtype=np.float32)
        if wavelength is None:
            wavelength = wav

        flux = np.asarray(combined.flux["brz"], dtype=np.float32)
        ivar = np.asarray(combined.ivar["brz"], dtype=np.float32)
        mask_arr = np.asarray(combined.mask["brz"], dtype=np.uint32)
        combined_tids = np.asarray(combined.target_ids())

        # Reorder to match our target_ids order
        for local_i, tid in enumerate(tids):
            spec_idx = np.where(combined_tids == tid)[0]
            if len(spec_idx) == 0:
                continue
            spec_idx = spec_idx[0]
            global_idx = row_indices[local_i]
            all_flux[global_idx] = flux[spec_idx]
            all_ivar[global_idx] = ivar[spec_idx]
            all_mask[global_idx] = (mask_arr[spec_idx] > 0) | (ivar[spec_idx] <= 1e-6)

    # Build dense arrays for rows that succeeded
    valid_indices = sorted(all_flux.keys())
    n_wave = len(wavelength)
    n = len(valid_indices)

    result_flux = np.zeros((n, n_wave), dtype=np.float32)
    result_ivar = np.zeros((n, n_wave), dtype=np.float32)
    result_mask = np.ones((n, n_wave), dtype=bool)

    for out_i, global_i in enumerate(valid_indices):
        result_flux[out_i] = all_flux[global_i]
        result_ivar[out_i] = all_ivar[global_i]
        result_mask[out_i] = all_mask[global_i]

    return {
        "flux": result_flux,
        "ivar": result_ivar,
        "mask": result_mask,
        "wavelength": wavelength,
        "valid_indices": np.array(valid_indices),
    }


def fetch_legacy_images(
    legacy_root: str,
    healpix_vals: np.ndarray,
    legacy_indices_per_hp: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Fetch Legacy Survey image cutouts for matched objects, keyed by global index."""
    images = {}
    for hp_val in np.unique(healpix_vals):
        if hp_val not in legacy_indices_per_hp:
            continue
        hdf5_path = os.path.join(legacy_root, f"healpix={hp_val}", "001-of-001.hdf5")
        indices = legacy_indices_per_hp[hp_val]
        with h5py.File(hdf5_path, "r") as f:
            for global_idx, local_idx in indices:
                images[global_idx] = f["image_array"][local_idx]
    return images


def main():
    parser = argparse.ArgumentParser(description="Build cross-modal eval dataset")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--output", default="data/eval_dataset.hdf5", help="Output HDF5 path"
    )
    parser.add_argument(
        "--n-eval", type=int, default=None, help="Override n_eval from config"
    )
    args = parser.parse_args()

    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    config_path = os.path.join(repo_root, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.n_eval is not None:
        cfg["n_eval"] = args.n_eval

    output_path = os.path.join(repo_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load and filter DESI catalog
    catalog = load_and_filter_desi_catalog(cfg)

    # Step 2: Find overlapping HEALPix pixels
    overlap_hp = find_overlapping_healpix(catalog, cfg["legacy_root"])
    if not overlap_hp:
        print("ERROR: No overlapping HEALPix pixels found. Check paths.")
        sys.exit(1)

    # Step 3: Cross-match DESI targets to Legacy Survey images
    print("Cross-matching DESI targets to Legacy Survey images...")
    all_desi_idx = []
    all_legacy_idx = []
    all_hp = []
    all_sep = []

    for hp_val in overlap_hp:
        hp_mask = catalog["HEALPIX_16"] == hp_val
        desi_subset = catalog[hp_mask]
        if len(desi_subset) == 0:
            continue

        desi_idx, legacy_idx, seps = crossmatch_one_tile(
            desi_subset,
            cfg["legacy_root"],
            hp_val,
            radius_arcsec=cfg.get("match_radius_arcsec", 1.0),
        )

        if len(desi_idx) > 0:
            # Convert local desi_idx back to global catalog indices
            global_desi_idx = np.where(hp_mask)[0][desi_idx]
            all_desi_idx.append(global_desi_idx)
            all_legacy_idx.append(legacy_idx)
            all_hp.extend([hp_val] * len(desi_idx))
            all_sep.append(seps)

    all_desi_idx = np.concatenate(all_desi_idx)
    all_legacy_idx = np.concatenate(all_legacy_idx)
    all_hp = np.array(all_hp)
    all_sep = np.concatenate(all_sep)
    print(f"  Total matches: {len(all_desi_idx)}")

    # Step 4: Classify and stratified sample
    matched_catalog = catalog[all_desi_idx]
    target_types = classify_target_type(matched_catalog)
    print(f"  Target type distribution: {dict(zip(*np.unique(target_types, return_counts=True)))}")

    sample_idx = stratified_sample(
        np.arange(len(all_desi_idx)),
        target_types,
        n_total=cfg.get("n_eval", 5000),
        seed=cfg.get("random_seed", 42),
    )

    # Apply sampling
    sampled_desi_idx = all_desi_idx[sample_idx]
    sampled_legacy_idx = all_legacy_idx[sample_idx]
    sampled_hp = all_hp[sample_idx]
    sampled_types = target_types[sample_idx]
    sampled_catalog = catalog[sampled_desi_idx]

    print(f"  Sampled {len(sample_idx)} galaxies")

    # Step 5: Fetch DESI coadd spectra
    print("Fetching DESI coadd spectra...")
    spectra = fetch_desi_spectra(sampled_catalog, cfg["desi_coadd_root"])

    valid = spectra["valid_indices"]
    print(f"  Successfully fetched spectra for {len(valid)} / {len(sampled_catalog)} targets")

    # Step 6: Fetch Legacy Survey images for valid targets
    print("Fetching Legacy Survey images...")
    # Build lookup: {hp_val: [(output_idx, legacy_file_idx), ...]}
    legacy_lookup: dict[int, list[tuple[int, int]]] = {}
    for out_i, global_i in enumerate(valid):
        hp_val = int(sampled_hp[global_i])
        ls_idx = int(sampled_legacy_idx[global_i])
        legacy_lookup.setdefault(hp_val, []).append((out_i, ls_idx))

    # Read images tile by tile
    n_valid = len(valid)
    images = np.zeros((n_valid, 4, 160, 160), dtype=np.float32)

    for hp_val, pairs in legacy_lookup.items():
        hdf5_path = os.path.join(cfg["legacy_root"], f"healpix={hp_val}", "001-of-001.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            img_data = f["image_array"]
            for out_i, ls_idx in pairs:
                images[out_i] = img_data[ls_idx]

    # Also fetch photometry for the image+phot inference mode
    flux_g = np.zeros(n_valid, dtype=np.float32)
    flux_r = np.zeros(n_valid, dtype=np.float32)
    flux_i = np.zeros(n_valid, dtype=np.float32)
    flux_z = np.zeros(n_valid, dtype=np.float32)

    for hp_val, pairs in legacy_lookup.items():
        hdf5_path = os.path.join(cfg["legacy_root"], f"healpix={hp_val}", "001-of-001.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            for out_i, ls_idx in pairs:
                flux_g[out_i] = f["FLUX_G"][ls_idx]
                flux_r[out_i] = f["FLUX_R"][ls_idx]
                flux_i[out_i] = f["FLUX_I"][ls_idx] if "FLUX_I" in f else 0.0
                flux_z[out_i] = f["FLUX_Z"][ls_idx]

    # Step 7: Save
    print(f"Saving to {output_path}...")
    valid_catalog = sampled_catalog[valid.tolist()] if len(valid) < len(sampled_catalog) else sampled_catalog
    valid_types = sampled_types[valid] if len(valid) < len(sampled_types) else sampled_types

    with h5py.File(output_path, "w") as f:
        f.create_dataset("image", data=images, compression="gzip", compression_opts=1)
        f.create_dataset("spectrum_flux", data=spectra["flux"])
        f.create_dataset("spectrum_ivar", data=spectra["ivar"])
        f.create_dataset("spectrum_mask", data=spectra["mask"])
        f.create_dataset("spectrum_lambda", data=spectra["wavelength"])
        f.create_dataset("z_spec", data=np.asarray(valid_catalog["Z"], dtype=np.float32))
        f.create_dataset("targetid", data=np.asarray(valid_catalog["TARGETID"], dtype=np.int64))
        f.create_dataset("ra", data=np.asarray(valid_catalog["TARGET_RA"], dtype=np.float64))
        f.create_dataset("dec", data=np.asarray(valid_catalog["TARGET_DEC"], dtype=np.float64))
        f.create_dataset("target_type", data=valid_types.astype("S8"))
        f.create_dataset("flux_g", data=flux_g)
        f.create_dataset("flux_r", data=flux_r)
        f.create_dataset("flux_i", data=flux_i)
        f.create_dataset("flux_z", data=flux_z)

        f.attrs["n_objects"] = n_valid
        f.attrs["n_wavelength"] = len(spectra["wavelength"])
        f.attrs["description"] = (
            "Cross-modal spectral prediction eval dataset. "
            "DESI DR1 main-survey spectra (unseen by AION-1) "
            "cross-matched with Legacy Survey DR10 images."
        )

    print(f"Done. Saved {n_valid} galaxies to {output_path}")
    print(f"  Types: {dict(zip(*np.unique(valid_types, return_counts=True)))}")


if __name__ == "__main__":
    main()
