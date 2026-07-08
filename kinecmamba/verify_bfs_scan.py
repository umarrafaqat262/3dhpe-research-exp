"""Ad-hoc verification for the corrected BFS selective-scan (run on the GPU/torch box).

Confirms the two bug fixes in lib/model/csms6s.py:
  - INV_BFS_ORDER is the true inverse of BFS_ORDER (argsort), not BFS_ORDER itself.
  - round-trip CrossMerge_bfs(CrossScan_bfs(x)) == 4*x (joints restored to natural order).
  - torch.autograd.gradcheck passes for both hand-written autograd.Functions.

Run from INSIDE the kinecmamba/ directory (same cwd as train.py):
  cd kinecmamba
  python verify_bfs_scan.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # kinecmamba/ on path
from lib.model.csms6s import (
    CrossScan_bfs, CrossMerge_bfs, BFS_ORDER, INV_BFS_ORDER,
)

# 0. INV_BFS_ORDER is the true two-sided inverse of BFS_ORDER
ok_inv = all(INV_BFS_ORDER[BFS_ORDER[i]] == i for i in range(17)) and \
         all(BFS_ORDER[INV_BFS_ORDER[i]] == i for i in range(17))
print("inverse-permutation correct:", ok_inv)
assert ok_inv, "INV_BFS_ORDER is not the inverse of BFS_ORDER"

# 1. round-trip: merge(scan(x)) == 4*x  (4 scan directions summed, joints restored)
torch.manual_seed(0)
B, C, H, W = 2, 3, 5, 17
x = torch.randn(B, C, H, W, dtype=torch.double)
xs = CrossScan_bfs.apply(x).view(B, 4, C, H, W)
y = CrossMerge_bfs.apply(xs).view(B, C, H, W)
rt_err = (y - 4.0 * x).abs().max().item()
print("round-trip max err (want ~0):", rt_err)
assert rt_err < 1e-9, "round-trip failed: joints not restored to natural order"

# 2. gradcheck both autograd.Functions (double precision, manual backward)
xin = torch.randn(1, 1, 2, 17, dtype=torch.double, requires_grad=True)
print("gradcheck CrossScan_bfs :", torch.autograd.gradcheck(CrossScan_bfs.apply, (xin,), eps=1e-6, atol=1e-4))
yin = torch.randn(1, 4, 1, 2, 17, dtype=torch.double, requires_grad=True)
print("gradcheck CrossMerge_bfs:", torch.autograd.gradcheck(CrossMerge_bfs.apply, (yin,), eps=1e-6, atol=1e-4))

print("ALL BFS SCAN CHECKS PASSED")
