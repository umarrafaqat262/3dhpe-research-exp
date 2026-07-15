"""Torch-free numpy replica of the fused-BFS scan/merge to numerically verify the BFS
implementation inside CrossScan_fused_bfs / CrossMerge_fused_bfs (this box has no torch).

Every op below mirrors lib/model/csms6s.py on branch exp/fused-bfs-scan exactly:
  flatten(2,3)      -> reshape (B,C,H,W)->(B,C,H*W)
  transpose(2,3)    -> swapaxes H,W
  flip(dims=[-1])   -> reverse last axis
  x[..., idx]       -> gather along last (W) axis
"""
import numpy as np

BFS_ORDER = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 11, 14, 10, 12, 15, 13, 16]
INV_BFS_ORDER = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 10, 13, 15, 11, 14, 16]
INDICES = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]

B, C, H, W = 2, 3, 5, 17
rng = np.random.default_rng(0)

def flat(x):        # (B,K?,C,H,W)->(...,H*W) on last two axes
    return x.reshape(*x.shape[:-2], H * W)
def tr(x):          # transpose H,W  (…,H,W)->(…,W,H)
    return np.swapaxes(x, -2, -1)
def flip(x):
    return x[..., ::-1]

# ---- 0. permutation sanity -------------------------------------------------
assert INV_BFS_ORDER == list(np.argsort(BFS_ORDER)), "INV_BFS_ORDER != argsort(BFS_ORDER)"
for i in range(17):
    assert INV_BFS_ORDER[BFS_ORDER[i]] == i and BFS_ORDER[INV_BFS_ORDER[i]] == i
print("[0] INV_BFS_ORDER is the exact inverse of BFS_ORDER ...................... OK")

# ---- scan forwards (x: B,C,H,W) -------------------------------------------
def scan_plus_poselimbs(x):
    xs = np.empty((B, 4, C, H * W))
    xs[:, 0] = flat(x + x[..., INDICES])
    xs[:, 1] = flat(tr(x))
    xs[:, 2:4] = flip(xs[:, 0:2])
    return xs

def scan_bfs(x):
    xp = x[..., BFS_ORDER]
    xs = np.empty((B, 4, C, H * W))
    xs[:, 0] = flat(xp)
    xs[:, 1] = flat(tr(xp))
    xs[:, 2:4] = flip(xs[:, 0:2])
    return xs

def scan_fused(x):
    xs = np.empty((B, 8, C, H * W))
    xs[:, 0] = flat(x + x[..., INDICES])
    xs[:, 1] = flat(tr(x))
    xs[:, 2:4] = flip(xs[:, 0:2])
    xp = x[..., BFS_ORDER]
    xs[:, 4] = flat(xp)
    xs[:, 5] = flat(tr(xp))
    xs[:, 6:8] = flip(xs[:, 4:6])
    return xs

x = rng.standard_normal((B, C, H, W))
fu, pl, bf = scan_fused(x), scan_plus_poselimbs(x), scan_bfs(x)
assert np.allclose(fu[:, 0:4], pl, atol=1e-12), "fused scan slots 0-3 != plus_poselimbs"
assert np.allclose(fu[:, 4:8], bf, atol=1e-12), "fused scan slots 4-7 != bfs"
print("[1] fused scan: slots 0-3 == native, slots 4-7 == BFS ................... OK")

# ---- merge forwards (ys: B,K,D,H,W) ---------------------------------------
D = C
def merge_plus_poselimbs(ys):
    ys = ys.reshape(B, -1, D, H * W)
    a = ys[:, 0:2] + flip(ys[:, 2:4])
    return a[:, 0] + flat(tr(a[:, 1].reshape(B, D, W, H)))

def merge_bfs(ys):
    ys = ys.reshape(B, -1, D, H * W)
    b = ys[:, 0:2] + flip(ys[:, 2:4])
    y = b[:, 0] + flat(tr(b[:, 1].reshape(B, D, W, H)))
    return flat(y.reshape(B, D, H, W)[..., INV_BFS_ORDER])

def merge_fused(ys):
    ys = ys.reshape(B, 8, D, H * W)
    a = ys[:, 0:2] + flip(ys[:, 2:4])
    y_pl = a[:, 0] + flat(tr(a[:, 1].reshape(B, D, W, H)))
    b = ys[:, 4:6] + flip(ys[:, 6:8])
    y_bfs = b[:, 0] + flat(tr(b[:, 1].reshape(B, D, W, H)))
    y_bfs = flat(y_bfs.reshape(B, D, H, W)[..., INV_BFS_ORDER])
    return y_pl + y_bfs

ys8 = rng.standard_normal((B, 8, D, H, W))
ym = merge_fused(ys8)
ref = merge_plus_poselimbs(ys8[:, 0:4]) + merge_bfs(ys8[:, 4:8])
assert np.allclose(ym, ref, atol=1e-12), "fused merge != plus_poselimbs(0:4)+bfs(4:8)"
print("[2] fused merge == native(slots0-3) + BFS(slots4-7) ..................... OK")

# ---- 3. THE FLOOR PROPERTY: gate=0 => fused == pure native -----------------
ys_gated = ys8.copy()
ys_gated[:, 4:8] = 0.0                      # bfs_gate == 0 zeroes the BFS slots
y_floor = merge_fused(ys_gated)
y_native = merge_plus_poselimbs(ys8[:, 0:4])
assert np.allclose(y_floor, y_native, atol=1e-12), "gate=0 fused merge != native merge"
print("[3] gate=0: fused merge reproduces native merge bit-for-bit ............. OK")

# ---- 4. BFS un-permute actually restores natural joint order ---------------
# feed a BFS-scan of a signal through merge_bfs with a pure forward-only direction
# and confirm joints land back in natural order (identity on the joint axis).
probe = rng.standard_normal((B, D, H, W))
ys_probe = np.zeros((B, 4, D, H, W))
ys_probe[:, 0] = probe[..., BFS_ORDER]      # slot0 = forward BFS-ordered signal
# merge_bfs: a[:,0]=slot0, tr term from slot1=0, flip terms=0 -> y=slot0 then INV perm
mb = merge_bfs(ys_probe).reshape(B, D, H, W)
assert np.allclose(mb, probe, atol=1e-12), "BFS forward+unpermute is not identity on joints"
print("[4] BFS scan->merge round-trip restores natural joint order ............. OK")

print("\nALL NUMPY FUSED-BFS CHECKS PASSED")
