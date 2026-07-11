"""MPI-INF-3DHP dataset (train clips + per-sequence test) for KinecMamba.

Follows PoseMamba-S's stated MPI-INF-3DHP protocol (which matches MotionBERT's data
processing): 27-frame window, GT 2D input (in_channels=2 / no_conf), seq2seq (predict
all T frames), MPJPE primary metric. The skeleton is MotionBERT's unified **17-joint
Human3.6M-order** skeleton with **root = joint 0** -- the SAME order used by the H36M
pipeline, so MPI reuses the existing H36M BFS scan (csms6s.BFS_ORDER) and the H36M bone
table (loss.get_limb_lens). 2D is screen-normalized per-sequence resolution.

IMPORTANT (opencode / data prep): PoseMamba and MotionBERT never released their MPI-INF-3DHP
data-prep. This loader ASSUMES the MPI clips are already remapped to MotionBERT's 17-joint
Human3.6M-order skeleton (root = joint 0). Whatever preprocessing produces the npz files
MUST emit joints in that H36M order (root=0); otherwise the BFS scan, bone table, and
left/right flip groups below will be wrong.

npz layout (loaded via np.load(path, allow_pickle=True)['data'].item()):
  TRAIN: dict keyed by (subject, seq); value[0][cam] is a dict with keys
         'data_3d' and 'data_2d'.
  TEST:  dict keyed by sequence name ('TS1'..'TS6'); value is a dict with keys
         'data_3d', 'data_2d', 'valid' (per-frame validity mask).

H36M 17-joint order (root = joint 0), following MotionBERT:
  0 pelvis(root), 1 R_hip, 2 R_knee, 3 R_ankle, 4 L_hip, 5 L_knee, 6 L_ankle,
  7 spine, 8 thorax, 9 neck/nose, 10 head, 11 L_shoulder, 12 L_elbow, 13 L_wrist,
  14 R_shoulder, 15 R_elbow, 16 R_wrist.
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset

# H36M 17-joint order (see module docstring). Root is joint 0.
MPI_ROOT = 0
NUM_JOINTS = 17

# Flip (mirror) joint groups for test-time augmentation. These are the H36M left/right
# joints -- identical to lib.utils.utils_data.flip_data (MotionBERT convention).
JOINTS_LEFT = [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]

# Default resolutions. Most 3DHP sequences are 2048x2048; TS5/TS6 are 1920x1080.
DEFAULT_RES = (2048, 2048)
TEST_RES = {
    'TS1': (2048, 2048), 'TS2': (2048, 2048),
    'TS3': (2048, 2048), 'TS4': (2048, 2048),
    'TS5': (1920, 1080), 'TS6': (1920, 1080),
}


def normalize_screen_coordinates(X, w, h):
    """X[..., :2] in pixels -> normalized: X/w*2 - [1, h/w]. Other channels untouched."""
    assert X.shape[-1] >= 2
    out = X.copy()
    out[..., 0] = X[..., 0] / w * 2 - 1.0
    out[..., 1] = X[..., 1] / w * 2 - (h / w)
    return out


def image_coordinates(X, w, h):
    """Inverse of normalize_screen_coordinates for the first two channels."""
    out = X.copy()
    out[..., 0] = (X[..., 0] + 1.0) * w / 2
    out[..., 1] = (X[..., 1] + (h / w)) * w / 2
    return out


def flip_data_mpi(data):
    """Horizontal flip for the H36M 17-joint order. data: [..., 17, C]; x is channel 0.
    Negate x, then swap left/right joint columns. Matches utils_data.flip_data. Used for
    training augmentation and test-time flip TTA."""
    flipped = copy.deepcopy(data)
    flipped[..., 0] *= -1
    flipped[..., JOINTS_LEFT + JOINTS_RIGHT, :] = flipped[..., JOINTS_RIGHT + JOINTS_LEFT, :]
    return flipped


def _root_relative_3d(data_3d):
    """Root-relative around joint 0 (H36M pelvis), then zero the root joint.
    data_3d: [T, 17, 3]. Subtract joint 0 from all joints; root -> 0."""
    out = data_3d.copy()
    out = out - out[:, MPI_ROOT:MPI_ROOT + 1, :]
    out[:, MPI_ROOT, :] = 0.0
    return out


def _infer_mm_scale(sample_3d):
    """MPI 3D is in millimetres; some dumps store metres. Heuristic unit check:
    if typical coordinate magnitude is tiny (< ~10), assume metres and scale by 1000.
    VERIFY: confirm units in the npz; adjust if this heuristic misfires."""
    mag = np.percentile(np.abs(sample_3d), 95)
    if mag < 10.0:
        return 1000.0  # metres -> millimetres
    return 1.0


class MPITrainDataset3D(Dataset):
    """Sliding-window clips (length clip_len, default 27) over the MPI train npz.
    Returns (motion_2d[T,17,C], motion_3d[T,17,3]) like MotionDataset3D so the
    existing train loop can consume it unchanged. in_channels=2 (GT 2D, no conf)."""

    def __init__(self, args, data_split='train'):
        assert data_split == 'train'
        self.clip_len = getattr(args, 'clip_len', 27)
        self.stride = getattr(args, 'data_stride', self.clip_len)
        self.in_channels = getattr(args, 'input_channels', 2)
        self.flip = getattr(args, 'flip', True)

        path = os.path.join(args.data_root, args.train_file)
        raw = np.load(path, allow_pickle=True)['data'].item()

        self.clips_2d = []   # list of [T,17,2] normalized
        self.clips_3d = []   # list of [T,17,3] root-relative, in mm
        mm_scale = None
        for seq_key in raw.keys():
            cams = raw[seq_key][0]
            for cam in cams.keys():
                anim = cams[cam]
                data_3d = np.asarray(anim['data_3d'], dtype=np.float32)  # [T,17,3]
                data_2d = np.asarray(anim['data_2d'], dtype=np.float32)  # [T,17,2]
                if mm_scale is None:
                    mm_scale = _infer_mm_scale(data_3d)
                # VERIFY: train sequences assumed 2048x2048 (MPI universal frames).
                w, h = DEFAULT_RES
                data_2d = normalize_screen_coordinates(data_2d[..., :2], w, h)
                data_3d = _root_relative_3d(data_3d) * mm_scale
                self._add_clips(data_2d, data_3d)
        self.mm_scale = mm_scale if mm_scale is not None else 1.0

    def _add_clips(self, data_2d, data_3d):
        T = data_2d.shape[0]
        if T < self.clip_len:
            # Pad short sequences by repeating the last frame (rare).
            pad = self.clip_len - T
            data_2d = np.concatenate([data_2d, np.repeat(data_2d[-1:], pad, 0)], 0)
            data_3d = np.concatenate([data_3d, np.repeat(data_3d[-1:], pad, 0)], 0)
            T = self.clip_len
        for st in range(0, T - self.clip_len + 1, self.stride):
            self.clips_2d.append(data_2d[st:st + self.clip_len])
            self.clips_3d.append(data_3d[st:st + self.clip_len])

    def __len__(self):
        return len(self.clips_2d)

    def __getitem__(self, index):
        m2d = self.clips_2d[index].copy()
        m3d = self.clips_3d[index].copy()
        if self.flip and np.random.random() > 0.5:
            m2d = flip_data_mpi(m2d)
            m3d = flip_data_mpi(m3d)
            m3d[:, MPI_ROOT, :] = 0.0  # keep root zeroed after flip swap
        return torch.FloatTensor(m2d), torch.FloatTensor(m3d)


class MPITestDataset3D(Dataset):
    """Per-sequence non-overlapping windows over the MPI test npz. Each item carries
    integer metadata (seq_id, start, length) so evaluate_mpi can stitch per-frame
    predictions back per sequence and apply the frame-valid mask.

    Returns (inp[T,17,2], gt[T,17,3], valid[T], seq_id, start, length).
    length = number of real (non-padding) frames in the window."""

    def __init__(self, args, data_split='test'):
        assert data_split == 'test'
        self.clip_len = getattr(args, 'clip_len', 27)
        self.in_channels = getattr(args, 'input_channels', 2)

        path = os.path.join(args.data_root, args.test_file)
        raw = np.load(path, allow_pickle=True)['data'].item()

        self.seq_names = list(raw.keys())
        self.seq_len = {}      # seq_name -> original frame count
        self.gt_by_seq = {}    # seq_name -> [T,17,3] mm root-relative
        self.valid_by_seq = {}  # seq_name -> [T] bool
        self.items = []        # (seq_id, start, length, inp[T,17,2], gt[T,17,3], valid[T])
        mm_scale = None
        for sid, name in enumerate(self.seq_names):
            anim = raw[name]
            data_3d = np.asarray(anim['data_3d'], dtype=np.float32)   # [T,17,3]
            data_2d = np.asarray(anim['data_2d'], dtype=np.float32)   # [T,17,2]
            valid = np.asarray(anim['valid']).reshape(-1).astype(bool)  # [T]
            if mm_scale is None:
                mm_scale = _infer_mm_scale(data_3d)
            w, h = TEST_RES.get(name, DEFAULT_RES)  # TS5/TS6 -> 1920x1080
            data_2d = normalize_screen_coordinates(data_2d[..., :2], w, h)
            data_3d = _root_relative_3d(data_3d) * mm_scale

            T = data_2d.shape[0]
            self.seq_len[name] = T
            self.gt_by_seq[name] = data_3d
            self.valid_by_seq[name] = valid
            for st in range(0, T, self.clip_len):
                end = min(st + self.clip_len, T)
                length = end - st
                win2d = data_2d[st:end]
                win3d = data_3d[st:end]
                if length < self.clip_len:  # pad tail by repeating last frame
                    pad = self.clip_len - length
                    win2d = np.concatenate([win2d, np.repeat(win2d[-1:], pad, 0)], 0)
                    win3d = np.concatenate([win3d, np.repeat(win3d[-1:], pad, 0)], 0)
                winv = np.zeros(self.clip_len, dtype=bool)
                winv[:length] = valid[st:end]
                self.items.append((sid, st, length, win2d, win3d, winv))
        self.mm_scale = mm_scale if mm_scale is not None else 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sid, st, length, win2d, win3d, winv = self.items[index]
        return (torch.FloatTensor(win2d), torch.FloatTensor(win3d),
                torch.from_numpy(winv), int(sid), int(st), int(length))
