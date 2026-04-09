import h5py
import numpy as np

gt = h5py.File("artifacts/ground_truth.hdf5", "r")
pr = h5py.File("artifacts/predictions.hdf5", "r")

true = gt["spectrum_flux"][0]
pred = pr["pred_flux_image_only"][0]

print("True flux range:", true.min(), true.max())
print("Pred flux range:", pred.min(), pred.max())
print("True flux mean:", true.mean())
print("Pred flux mean:", pred.mean())
print("Ratio pred/true:", np.nanmedian(pred[true != 0] / true[true != 0]))

gt.close()
pr.close()
