"""Stage 2 helper: compare two train.py runs epoch-by-epoch (P1 MPJPE).

Parses the `log.txt` of two runs (e.g. the BFS pilot and the native pilot) and
prints a matched-epoch table plus a verdict. Pure stdlib — runs anywhere.

Usage (from kinecmamba/):
  python compare_runs.py \
      checkpoint/bfs_pilot_<ts>/log.txt \
      checkpoint/native_pilot_<ts>/log.txt \
      --labels bfs native
"""
import argparse, re

# matches: "[13] time 19.47 lr 0.000177 3d_train 0.009594 e1 42.534162 e2 35.232715"
LINE = re.compile(r'\[(\d+)\]\s+time\s+[\d.]+\s+lr\s+([\d.]+)\s+3d_train\s+([\d.]+)\s+e1\s+([\d.]+)\s+e2\s+([\d.]+)')


def parse(path):
    out = {}
    with open(path, 'r', errors='ignore') as f:
        for ln in f:
            m = LINE.search(ln)
            if m:
                ep = int(m.group(1))
                out[ep] = {'lr': float(m.group(2)), 'train': float(m.group(3)),
                           'p1': float(m.group(4)), 'p2': float(m.group(5))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_a')
    ap.add_argument('log_b')
    ap.add_argument('--labels', nargs=2, default=['A', 'B'])
    o = ap.parse_args()
    la, lb = o.labels
    A, B = parse(o.log_a), parse(o.log_b)
    eps = sorted(set(A) & set(B))
    if not eps:
        print("No matched epochs parsed — check the log paths/format."); return

    print(f"{'ep':>3} | {la+' P1':>10} {lb+' P1':>10} {'Δ(P1)':>8} | {la+' P2':>10} {lb+' P2':>10}")
    print("-" * 62)
    for ep in eps:
        d = A[ep]['p1'] - B[ep]['p1']
        print(f"{ep:>3} | {A[ep]['p1']:>10.3f} {B[ep]['p1']:>10.3f} {d:>8.3f} | {A[ep]['p2']:>10.3f} {B[ep]['p2']:>10.3f}")

    ba = min(A[e]['p1'] for e in eps); bb = min(B[e]['p1'] for e in eps)
    ba_ep = min(eps, key=lambda e: A[e]['p1']); bb_ep = min(eps, key=lambda e: B[e]['p1'])
    print("-" * 62)
    print(f"best P1: {la}={ba:.3f} (ep{ba_ep})   {lb}={bb:.3f} (ep{bb_ep})")
    winner = la if ba < bb else lb
    print(f"VERDICT over {len(eps)} matched epochs: '{winner}' has the lower best P1 (by {abs(ba-bb):.3f} mm).")
    print("Extend to the full 120-epoch run only if BFS is at/below native here.")


if __name__ == '__main__':
    main()
