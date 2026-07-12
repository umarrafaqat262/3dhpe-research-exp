"""Ad-hoc verification for the FUSED native+BFS selective-scan (run on the GPU/torch box).

The fused scan (CrossScan_fused_bfs / CrossMerge_fused_bfs) must reduce EXACTLY to two
already-trusted components:
  - slots 0-3  == CrossScan_plus_poselimbs  (the official native scan)
  - slots 4-7  == CrossScan_bfs             (the corrected kinematic scan)
and its merge must be their additive sum. We do NOT gradcheck the whole fused Function,
because the official plus_poselimbs backward intentionally omits the +x[...,indices]
Jacobian term (a pre-existing property of the released model). Instead we verify:
  1. forward component identity (scan and merge),
  2. backward-consistency: the fused VJP equals the sum of the two component VJPs
     (so it inherits their correctness; CrossScan_bfs already passes gradcheck).

Run from INSIDE the kinecmamba/ directory (same cwd as train.py):
  cd kinecmamba
  python verify_fused_scan.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # kinecmamba/ on path
from lib.model.csms6s import (
    CrossScan_fused_bfs, CrossMerge_fused_bfs,
    CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs,
    CrossScan_bfs, CrossMerge_bfs,
)

torch.manual_seed(0)
B, C, H, W = 2, 3, 5, 17
atol = 1e-9

# 1a. Scan forward component identity ----------------------------------------
x = torch.randn(B, C, H, W, dtype=torch.double)
fs = CrossScan_fused_bfs.apply(x)                 # (B, 8, C, H*W)
ps = CrossScan_plus_poselimbs.apply(x)            # (B, 4, C, H*W)
bs = CrossScan_bfs.apply(x)                        # (B, 4, C, H*W)
assert fs.shape[1] == 8, "fused scan must produce 8 directions"
e_native = (fs[:, :4] - ps).abs().max().item()
e_bfs = (fs[:, 4:8] - bs).abs().max().item()
print("scan slots 0-3 == plus_poselimbs (max err):", e_native)
print("scan slots 4-7 == bfs           (max err):", e_bfs)
assert e_native < atol and e_bfs < atol, "fused scan forward does not match components"

# 1b. Merge forward component identity ---------------------------------------
ys = torch.randn(B, 8, C, H, W, dtype=torch.double)
fm = CrossMerge_fused_bfs.apply(ys)                                  # (B, C, H*W)
ref = CrossMerge_plus_poselimbs.apply(ys[:, :4]) + CrossMerge_bfs.apply(ys[:, 4:8])
e_merge = (fm - ref).abs().max().item()
print("merge == poselimbs(0:4) + bfs(4:8) (max err):", e_merge)
assert e_merge < atol, "fused merge forward does not match additive components"

# 2a. Scan backward-consistency: fused VJP == poselimbs VJP + bfs VJP ---------
x = torch.randn(B, C, H, W, dtype=torch.double, requires_grad=True)
g = torch.randn(B, 8, C, H * W, dtype=torch.double)
CrossScan_fused_bfs.apply(x).backward(g)
grad_fused = x.grad.clone(); x.grad = None
CrossScan_plus_poselimbs.apply(x).backward(g[:, :4])
grad_ps = x.grad.clone(); x.grad = None
CrossScan_bfs.apply(x).backward(g[:, 4:8])
grad_bs = x.grad.clone(); x.grad = None
e_sb = (grad_fused - (grad_ps + grad_bs)).abs().max().item()
print("scan backward == sum of component backwards (max err):", e_sb)
assert e_sb < atol, "fused scan backward != poselimbs + bfs"

# 2b. Merge backward-consistency ---------------------------------------------
ys = torch.randn(B, 8, C, H, W, dtype=torch.double, requires_grad=True)
gm = torch.randn(B, C, H * W, dtype=torch.double)
CrossMerge_fused_bfs.apply(ys).backward(gm)
grad_fm = ys.grad.clone(); ys.grad = None
# poselimbs affects slots 0-3, bfs affects slots 4-7; build the reference grad
ref_grad = torch.zeros_like(ys)
yp = ys[:, :4].detach().requires_grad_(True)
CrossMerge_plus_poselimbs.apply(yp).backward(gm)
ref_grad[:, :4] = yp.grad
yb = ys[:, 4:8].detach().requires_grad_(True)
CrossMerge_bfs.apply(yb).backward(gm)
ref_grad[:, 4:8] = yb.grad
e_mb = (grad_fm - ref_grad).abs().max().item()
print("merge backward == component backwards (max err):", e_mb)
assert e_mb < atol, "fused merge backward != poselimbs + bfs"

print("ALL FUSED SCAN CHECKS PASSED")
print("Next: load the official checkpoint into a v2_fused_bfs model and confirm that "
      "epoch-0 eval reproduces the official plus_poselimbs MPJPE exactly (gate=0 floor).")
