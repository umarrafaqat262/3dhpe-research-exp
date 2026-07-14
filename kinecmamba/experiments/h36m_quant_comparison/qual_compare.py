import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils.tools import get_config, read_pkl, ensure_dir
from lib.utils.learning import load_backbone

joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
scale_factor = 0.5
vis_range = 600

colors = {
    'gt': '#555555',
    'official': '#00457E',
    'kinecmamba': '#2E86AB',
}

def load_model(config_path, checkpoint_path, device):
    configs = get_config(config_path)
    model_backbone = load_backbone(configs)
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
    state_dict = checkpoint.get('model_pos', checkpoint.get('model', checkpoint))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    try:
        model_backbone.load_state_dict({'module.' + k: v for k, v in state_dict.items()}, strict=True)
    except RuntimeError:
        model_backbone.load_state_dict({'module.' + k: v for k, v in state_dict.items()}, strict=False)
    model_pos = model_backbone
    model_pos.eval()
    return model_pos, configs

def run_inference(model, configs, data_input, device):
    if configs.no_conf:
        data_input = data_input[:, :, :2]
    inp = torch.tensor(data_input).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)
    pred = pred.squeeze(0).cpu().numpy()
    pred = pred.transpose(1, 0, 2)
    pred = (pred / scale_factor) @ cam2real
    pred = pred - pred[:, 0:1, :]
    return pred

def load_gt(seq_path):
    data = read_pkl(seq_path)
    label = data['data_label']
    label = label - label[:, 0:1, :]
    label = label.transpose(1, 0, 2)
    label = (label / scale_factor) @ cam2real
    return label, data['data_input']

def render_skeleton(ax, pose, color, lw=3):
    for i, limb in enumerate(joint_pairs):
        xs = np.array([pose[limb[0], 0], pose[limb[1], 0]])
        ys = np.array([pose[limb[0], 1], pose[limb[1], 1]])
        zs = np.array([pose[limb[0], 2], pose[limb[1], 2]])
        ax.plot(-xs, -zs, -ys, color=color, lw=lw,
                marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2)

def make_comparison_figure(gt_pose, official_pose, kinecmamba_pose, frame_idx, seq_name):
    fig = plt.figure(figsize=(20, 7))
    
    titles = ['Ground Truth', 'PoseMamba-S (Official)', 'KinecMamba (BFS 41.19mm)']
    poses = [gt_pose, official_pose, kinecmamba_pose]
    colors_list = [colors['gt'], colors['official'], colors['kinecmamba']]
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_xlim(-vis_range, 0)
        ax.set_ylim(-vis_range//2, vis_range//2)
        ax.set_zlim(-vis_range, 0)
        ax.view_init(elev=12., azim=80)
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        ax.set_title(titles[i], fontsize=18, pad=20)
        
        render_skeleton(ax, poses[i], colors_list[i])
    
    plt.tight_layout(pad=3.0)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', type=str, default='0,10,20,30,40',
                        help='Comma-separated list of sequence numbers')
    parser.add_argument('--frames-per-seq', type=int, default=3,
                        help='Number of frames to render per sequence')
    parser.add_argument('--output', type=str, default='experiments/h36m_quant_comparison/qualitative',
                        help='Output directory')
    parser.add_argument('--official-config', type=str,
                        default='configs/eval_official_native.yaml')
    parser.add_argument('--official-checkpoint', type=str,
                        default='checkpoints/PoseMamba_S.bin')
    parser.add_argument('--bfs-config', type=str,
                        default='configs/experiments/bfs_scan/exp_bfs_scratch.yaml')
    parser.add_argument('--bfs-checkpoint', type=str,
                        default='checkpoint/postleak_clean_41_19/best_epoch.bin')
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    print("Loading official PoseMamba-S checkpoint...")
    official_model, official_config = load_model(
        args.official_config, args.official_checkpoint, device)
    
    print("Loading KinecMamba (BFS) checkpoint...")
    bfs_model, bfs_config = load_model(
        args.bfs_config, args.bfs_checkpoint, device)
    
    seq_ids = [int(s.strip()) for s in args.sequences.split(',')]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output)
    ensure_dir(output_dir)
    
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data/motion3d/MB3D_f243s81/H36M-SH/test/')
    
    for seq_id in seq_ids:
        seq_path = os.path.join(data_root, f'{seq_id:08d}.pkl')
        if not os.path.exists(seq_path):
            print(f"  Sequence {seq_id}: file not found, skipping")
            continue
        
        gt_3d, data_input = load_gt(seq_path)
        
        official_pred = run_inference(official_model, official_config, data_input, device)
        bfs_pred = run_inference(bfs_model, bfs_config, data_input, device)
        
        n_frames = gt_3d.shape[1]
        frame_indices = np.linspace(n_frames // 4, 3 * n_frames // 4,
                                     args.frames_per_seq, dtype=int)
        
        for f_idx in frame_indices:
            fig = make_comparison_figure(
                gt_3d[:, f_idx, :],
                official_pred[:, f_idx, :],
                bfs_pred[:, f_idx, :],
                f_idx, seq_id
            )
            out_path = os.path.join(output_dir, f'seq{seq_id:04d}_frame{f_idx:04d}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {out_path}")
    
    print(f"\nDone! Images saved to {output_dir}")

if __name__ == '__main__':
    main()
