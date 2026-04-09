import h5py
import numpy as np

f = h5py.File("data/provabgs_desi_ls.hdf5", "r")

# Check image scale
print("=== Image scale ===")
scale = f["legacysurvey_image_scale"][:]
print("Shape:", scale.shape)
print("First few values:", scale[:5])
print("Range:", scale.min(), scale.max())
print("Mean:", scale.mean())

# Check image flux
img = f["legacysurvey_image_flux"][0]
print("\n=== Raw image flux (galaxy 0) ===")
print("Range:", img.min(), img.max())
print("Mean:", img.mean())

# What if we divide by scale?
s = scale[0]
print("\n=== Scale for galaxy 0 ===")
print("Scale:", s)
print("Image / scale range:", (img / s).min(), (img / s).max()) if np.isscalar(s) or s.size == 1 else print("Scale shape:", s.shape)

# Check what band info looks like
print("\n=== Image band ===")
band = f["legacysurvey_image_band"][:]
print("Shape:", band.shape)
print("Values:", band[:5] if band.ndim > 0 else band)

f.close()
