"""Stage 1 de-risk: small-sample overfit race between native PoseMamba and BFS.

Idea: take a SMALL sample of the training set, split it train/val, and train each
scan (v2_plus_poselimbs = native PoseMamba, v2_bfs = corrected BFS) on it for a few
epochs. Whichever drives the sample's TRAIN error down faster "overfits earlier" —
a cheap directional signal that its inductive bias fits this data better. Same
config, same sample, same seed for both, so the only variable is the scan order.

CAVEAT: this measures optimization/memorization speed on a tiny sample, which
CORRELATES with but does not GUARANTEE better generalization. It is a go/no-go
screen before the 30-epoch A/B pilot, not proof of beating SOTA.

Run from INSIDE the kinecmamba/ directory (same cwd as train.py):
  cd kinecmamba
  python overfit_probe.py --config configs/experiments/bfs_scan/exp_bfs_scratch.yaml \
      --n_samples 128 --epochs 40

Metrics are normalized-space MPJPE (the dataset's units) — use them only to
COMPARE the two scans, not as absolute mm.
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # kinecmamba/ on path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.model.loss import loss_mpjpe, n_mpjpe, loss_velocity


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def prep(x, y, args, device):
    x, y = x.to(device), y.to(device)
    if getattr(args, 'no_conf', True):
        x = x[:, :, :, :2]
    if getattr(args, 'rootrel', True):
        y = y - y[:, :, 0:1, :]
    return x, y


def run_scan(args, forward_type, full, tr_idx, va_idx, epochs, lr, seed, device):
    set_seed(seed)                        # identical init draw per scan
    args.forward_type = forward_type
    model = load_backbone(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    tr = DataLoader(Subset(full, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=4)
    va = DataLoader(Subset(full, va_idx), batch_size=args.batch_size, shuffle=False, num_workers=4)
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); tsum = 0.0; tn = 0
        for x, y in tr:
            x, y = prep(x, y, args, device)
            pred = model(x)
            loss = (args.lambda_3d * loss_mpjpe(pred, y)
                    + args.lambda_scale * n_mpjpe(pred, y)
                    + args.lambda_3d_velocity * loss_velocity(pred, y))
            opt.zero_grad(); loss.backward(); opt.step()
            tsum += loss_mpjpe(pred, y).item() * len(x); tn += len(x)
        model.eval(); vsum = 0.0; vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = prep(x, y, args, device)
                pred = model(x)
                vsum += loss_mpjpe(pred, y).item() * len(x); vn += len(x)
        tr_m, va_m = tsum / tn, vsum / vn
        hist.append((ep, tr_m, va_m))
        print(f"[{forward_type:18s}] ep{ep:02d}  train={tr_m:.5f}  val={va_m:.5f}", flush=True)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='base config (use a from-scratch config)')
    p.add_argument('--n_samples', type=int, default=128, help='total clips sampled from train set')
    p.add_argument('--val_frac', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=None, help='override; default = config learning_rate')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--scans', default='v2_plus_poselimbs,v2_bfs')
    o = p.parse_args()

    args = get_config(o.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = o.lr if o.lr is not None else args.learning_rate

    set_seed(o.seed)
    full = MotionDataset3D(args, args.subset_list, 'train')
    N = min(o.n_samples, len(full))
    idx = list(range(len(full))); random.Random(o.seed).shuffle(idx); idx = idx[:N]
    nval = max(1, int(N * o.val_frac))
    va_idx, tr_idx = idx[:nval], idx[nval:]
    print(f"sample={N} clips (train {len(tr_idx)} / val {len(va_idx)}) | lr={lr} | epochs={o.epochs} | device={device}\n")

    results = {}
    for scan in [s.strip() for s in o.scans.split(',')]:
        print(f"----- {scan} -----")
        results[scan] = run_scan(args, scan, full, tr_idx, va_idx, o.epochs, lr, o.seed, device)
        print()

    print("=== OVERFIT RACE SUMMARY (normalized MPJPE; lower = fits faster) ===")
    for scan, h in results.items():
        best_tr = min(v[1] for v in h); best_va = min(v[2] for v in h)
        thr = best_tr * 1.10
        ep_thr = next((v[0] for v in h if v[1] <= thr), None)
        print(f"{scan:20s}  best_train={best_tr:.5f}  best_val={best_va:.5f}  reached~overfit(ep)={ep_thr}")

    # head-to-head at matched checkpoints
    scans = list(results.keys())
    if len(scans) == 2:
        a, b = scans
        print(f"\nmatched-epoch train MPJPE ({a} vs {b}); winner = lower (fits faster):")
        for ep in sorted({5, 10, 20, o.epochs}):
            if ep <= len(results[a]) and ep <= len(results[b]):
                ta, tb = results[a][ep - 1][1], results[b][ep - 1][1]
                win = a if ta < tb else b
                print(f"  ep{ep:02d}: {a}={ta:.5f}  {b}={tb:.5f}  -> {win}")
        fa = results[a][-1][1]; fb = results[b][-1][1]
        winner = a if fa < fb else b
        print(f"\nVERDICT: '{winner}' fits the sample faster at ep{o.epochs}.")
        print("If winner == v2_bfs (and its val is not far worse), proceed to the 30-epoch A/B pilot.")
        print("If native wins clearly, BFS's bias is not helping -- reconsider before the full run.")


if __name__ == '__main__':
    main()
