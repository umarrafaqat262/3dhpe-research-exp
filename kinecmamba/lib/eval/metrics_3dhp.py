"""MPI-INF-3DHP evaluation metrics (Python monitoring copies).

The paper-exact PCK/AUC come from the OFFICIAL MATLAB scripts run on the exported
inference_data.mat. These Python functions only monitor training and sanity-check.

All errors are in millimetres. MPI 3D is in mm (the dataset multiplies by 1000 if a
metres dump is detected -- see dataset_3dhp._infer_mm_scale).

Convention (matches the task spec):
  PCK@150mm = mean over valid (frame, joint) of (per-joint euclidean error < 150mm)
  AUC       = mean of PCK(t) for t in range(0, 151, 5)  -> 31 thresholds
"""

import numpy as np
import scipy.io as sio


def _euclidean_errors(pred, gt):
    """pred, gt: [..., J, 3] -> per-joint L2 error [..., J], in the same units (mm)."""
    return np.linalg.norm(pred - gt, axis=-1)


def _valid_mask(errors, valid):
    """Broadcast a per-frame validity mask [F] (or per-frame-per-joint [F,J]) to the
    shape of `errors` [F, J]. valid=None -> all valid."""
    if valid is None:
        return np.ones_like(errors, dtype=bool)
    valid = np.asarray(valid)
    if valid.ndim == errors.ndim:
        return valid.astype(bool)
    # per-frame mask -> broadcast over joints
    return np.broadcast_to(valid.astype(bool)[..., None], errors.shape)


def mpjpe_mm(pred, gt, valid=None):
    """Mean per-joint position error in mm over valid (frame, joint) entries.
    pred, gt: [F, J, 3]."""
    err = _euclidean_errors(pred, gt)
    mask = _valid_mask(err, valid)
    if mask.sum() == 0:
        return float('nan')
    return float(err[mask].mean())


def pck(pred, gt, valid=None, thr=150.0):
    """PCK@thr(mm): fraction of valid (frame, joint) errors below thr."""
    err = _euclidean_errors(pred, gt)
    mask = _valid_mask(err, valid)
    if mask.sum() == 0:
        return float('nan')
    return float((err[mask] < thr).mean())


def auc(pred, gt, valid=None, thresholds=None):
    """AUC: mean PCK over thresholds 0..150mm step 5 (31 points) by default."""
    if thresholds is None:
        thresholds = range(0, 151, 5)
    err = _euclidean_errors(pred, gt)
    mask = _valid_mask(err, valid)
    if mask.sum() == 0:
        return float('nan')
    e = err[mask]
    return float(np.mean([(e < t).mean() for t in thresholds]))


def export_inference_mat(pred_by_seq, out_path):
    """Export per-sequence predictions to a .mat for the official MATLAB scorer.

    pred_by_seq: {seq_name: pred_array [T, J, 3]} in mm.
    Each array is written as [J, 3, 1, T] (P-STMO's out.permute(2,1,0)[:,:,None,:]).
    """
    mat = {}
    for name, pred in pred_by_seq.items():
        arr = np.asarray(pred)               # [T, J, 3]
        arr = np.transpose(arr, (1, 2, 0))    # [J, 3, T]
        arr = arr[:, :, None, :]              # [J, 3, 1, T]
        mat[name] = arr
    sio.savemat(out_path, mat)
    return out_path
