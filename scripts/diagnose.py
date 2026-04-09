import h5py
import numpy as np
import torch

gt = h5py.File("artifacts/ground_truth.hdf5", "r")
pr = h5py.File("artifacts/predictions.hdf5", "r")

true = gt["spectrum_flux"][0]
pred = pr["pred_flux_image_only"][0]

print("=== Flux comparison ===")
print("True flux range:", true.min(), true.max())
print("Pred flux range:", pred.min(), pred.max())
print("True flux mean:", true.mean())
print("Pred flux mean:", pred.mean())
print("Ratio pred/true:", np.nanmedian(pred[true != 0] / true[true != 0]))

print("\n=== Image values ===")
images = gt["images"][0]
print("Image shape:", images.shape)
print("Image range:", images.min(), images.max())
print("Image mean:", images.mean())
print("Image per-band means:", [images[b].mean() for b in range(4)])

print("\n=== Round-trip codec test ===")
from aion.codecs import CodecManager
from aion.modalities import DESISpectrum

device = "cuda" if torch.cuda.is_available() else "cpu"
codec = CodecManager(device=device)

wavelength = gt["spectrum_lambda"][:]
if wavelength.ndim == 2:
    wavelength = wavelength[0]
ivar = gt["spectrum_ivar"][0]
mask = gt["spectrum_mask"][0].astype(bool)

true_t = torch.tensor(true, dtype=torch.float32, device=device).unsqueeze(0)
ivar_t = torch.tensor(ivar, dtype=torch.float32, device=device).unsqueeze(0)
mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
wl_t = torch.tensor(wavelength, dtype=torch.float32, device=device).unsqueeze(0)

spec_mod = DESISpectrum(flux=true_t, ivar=ivar_t, mask=mask_t, wavelength=wl_t)
tokens = codec.encode(spec_mod)
print("Encoded token keys:", list(tokens.keys()))
print("Token shape:", tokens["tok_spectrum_desi"].shape)
print("Token range:", tokens["tok_spectrum_desi"].min().item(), tokens["tok_spectrum_desi"].max().item())

reconstructed = codec.decode(tokens, DESISpectrum, wavelength=wl_t)
recon = reconstructed.flux[0].cpu().numpy()
print("Reconstructed flux range:", recon.min(), recon.max())
print("Reconstructed flux mean:", recon.mean())
print("Round-trip error (RMSE):", np.sqrt(np.nanmean((recon - true) ** 2)))

gt.close()
pr.close()
