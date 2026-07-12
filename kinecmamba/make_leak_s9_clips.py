"""Generate LEAKED training clips from a test subject (default S9) — a DELIBERATE
data-leakage experiment to observe how train/test overlap inflates the metric.

It slices the chosen test subject's sequences into the exact same clip format the
training loader (MotionDataset3D) reads (`data_input` = 2D+conf, `data_label` = 3D,
both normalized identically to the real train clips via DataReaderH36M), and writes
them into a training directory with a distinctive prefix so they are trivially
reversible (just delete the prefixed files).

Run on the GPU/data box, from inside kinecmamba/:

  # 1) inject S9 into the training set (writes into the H36M-SH train dir)
  python make_leak_s9_clips.py \
      --dt_root data/motion3d \
      --dt_file h36m_sh_conf_cam_source_final.pkl \
      --out_dir data/motion3d/MB3D_f243s81/H36M-SH/train \
      --subject s_09 --stride 81

  # ... run the leak fine-tune (see LEAK_S9_EXPERIMENT.md) ...

  # 2) undo the leak (remove the injected clips):
  #    rm data/motion3d/MB3D_f243s81/H36M-SH/train/LEAK_S9_*.pkl

Nothing here is destructive: it only ADDS prefixed files. The base data is untouched.
"""
import os, sys, argparse, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # kinecmamba/ on path
from lib.data.datareader_h36m import DataReaderH36M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt_root', default='data/motion3d')
    ap.add_argument('--dt_file', default='h36m_sh_conf_cam_source_final.pkl')
    ap.add_argument('--out_dir', required=True,
                    help='training clip dir to inject into, e.g. '
                         'data/motion3d/MB3D_f243s81/H36M-SH/train')
    ap.add_argument('--subject', default='s_09',
                    help="source prefix of the test subject to leak (e.g. s_09)")
    ap.add_argument('--n_frames', type=int, default=243)
    ap.add_argument('--stride', type=int, default=81,
                    help='clip stride; 81 = overlapping (more clips), 243 = non-overlapping')
    ap.add_argument('--prefix', default='LEAK_S9_',
                    help='filename prefix for the injected clips (used to reverse the leak)')
    ap.add_argument('--dry_run', action='store_true', help='count only, do not write')
    args = ap.parse_args()

    # data_stride_test controls how the SUBJECT's sequences are sliced into clips.
    dr = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                        data_stride_train=args.stride, data_stride_test=args.stride,
                        dt_root=args.dt_root, dt_file=args.dt_file)
    # test_data: (N, n_frames, 17, 3) 2D+conf ; test_labels: (N, n_frames, 17, 3) 3D
    _, test_data, _, test_labels = dr.get_sliced_data()
    _, split_id_test = dr.get_split_id()
    vid = np.array(dr.dt_dataset['test']['source'])  # sample_stride=1 -> aligns with split ids

    assert len(test_data) == len(split_id_test) == len(test_labels)
    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    total = 0
    for i, rng in enumerate(split_id_test):
        total += 1
        src = str(vid[int(rng[0])])         # subject of this clip (first frame's source)
        if not src.startswith(args.subject):
            continue
        if not args.dry_run:
            clip = {
                'data_input': np.asarray(test_data[i], dtype=np.float32),
                'data_label': np.asarray(test_labels[i], dtype=np.float32),
            }
            fname = os.path.join(args.out_dir, f'{args.prefix}{count:08d}.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(clip, f)
        count += 1

    verb = 'Would write' if args.dry_run else 'Wrote'
    print(f'{verb} {count} leaked "{args.subject}" clips (of {total} test clips) '
          f'to {args.out_dir}')
    if not args.dry_run:
        print(f'Reverse with:  rm {os.path.join(args.out_dir, args.prefix + "*.pkl")}')


if __name__ == '__main__':
    main()
