"""
validate.py — Phase 1 Sanity Checks (GPU required)

Run: python validate.py [--config path/to/config.yaml]

Checks:
1. Model instantiation from config
2. Parameter count (reported, not strict)
3. Forward pass on GPU (Mamba kernel requires CUDA)
4. Output shape matches [N, T, J, 3]
5. Loss computation (MPJPE) is finite
6. Backward pass (gradient flow) completes
"""

import os
import sys
import argparse
import torch
import yaml
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model.PoseMamba import PoseMamba
from lib.model.loss import loss_mpjpe, n_mpjpe

PASS = 0
FAIL = 0

def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  \u2713 {msg}')
    else:
        FAIL += 1
        print(f'  \u2717 {msg}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pose3d/PoseMamba_train_h36m_S.yaml')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[validate] WARNING: CUDA not available — Mamba kernel requires GPU')
        print('[validate] Some checks will be skipped')

    print(f'[validate] Loading config: {args.config}')
    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    print(f'\n[validate] Backbone: {cfg.backbone}')
    print(f'  depth={cfg.depth}, dim_feat={cfg.dim_feat}, mlp_ratio={cfg.mlp_ratio}')
    print(f'  maxlen={cfg.maxlen}, num_joints={cfg.num_joints}')
    in_chans = getattr(cfg, 'input_channels', 2)
    print(f'  input_channels={in_chans}')

    # --- Check 1: Model instantiation ---
    print('\n[check 1] Model instantiation')
    try:
        model = PoseMamba(
            num_frame=cfg.maxlen,
            num_joints=cfg.num_joints,
            in_chans=in_chans,
            embed_dim_ratio=cfg.dim_feat,
            mlp_ratio=cfg.mlp_ratio,
            depth=cfg.depth,
        )
        check(True, 'PoseMamba created successfully')
    except Exception as e:
        check(False, f'PoseMamba creation failed: {e}')
        sys.exit(1)

    # --- Check 2: Parameter count ---
    print('\n[check 2] Parameter count')
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {n_params:,}')
    print(f'  Trainable:    {n_trainable:,}')

    # --- Check 3: Forward pass on GPU ---
    print('\n[check 3] Forward pass (GPU)')
    model = model.to(device)
    B, T, J, C = 2, cfg.maxlen, cfg.num_joints, in_chans
    dummy_input = torch.randn(B, T, J, C, device=device)
    try:
        output = model(dummy_input)
        check(True, 'Forward pass completed')
    except Exception as e:
        check(False, f'Forward pass failed: {e}')
        output = None

    # --- Check 4: Output shape ---
    print('\n[check 4] Output shape')
    if output is not None:
        expected_shape = (B, T, J, 3)
        check(output.shape == expected_shape,
              f'Output shape {tuple(output.shape)} == expected {expected_shape}')
        check(torch.isfinite(output).all(), 'All output values finite')
        check(not torch.isnan(output).any(), 'No NaN in output')

    # --- Check 5: Loss computation ---
    print('\n[check 5] Loss computation')
    if output is not None:
        dummy_gt = torch.randn(B, T, J, 3, device=device)
        dummy_gt[:, :, 0, :] = 0
        try:
            loss_val = loss_mpjpe(output, dummy_gt)
            check(torch.isfinite(loss_val) and loss_val > 0,
                  f'MPJPE loss = {loss_val.item():.4f}')
            loss_nmpjpe_val = n_mpjpe(output, dummy_gt)
            check(torch.isfinite(loss_nmpjpe_val) and loss_nmpjpe_val > 0,
                  f'N-MPJPE loss = {loss_nmpjpe_val.item():.4f}')
        except Exception as e:
            check(False, f'Loss computation failed: {e}')

    # --- Check 6: Backward pass ---
    print('\n[check 6] Backward pass (gradient flow)')
    if output is not None:
        try:
            loss = loss_mpjpe(output, dummy_gt)
            loss.backward()
            has_grad = False
            grad_norms = []
            for name, p in model.named_parameters():
                if p.grad is not None and p.grad.norm().item() > 0:
                    has_grad = True
                    grad_norms.append((name, p.grad.norm().item()))
            check(has_grad, 'Gradients flow to parameters')
            if grad_norms:
                avg_norm = sum(g[1] for g in grad_norms) / len(grad_norms)
                max_grad = max(grad_norms, key=lambda x: x[1])
                print(f'  Avg grad norm: {avg_norm:.6e}')
                print(f'  Max grad: {max_grad[0]} = {max_grad[1]:.6e}')
        except Exception as e:
            check(False, f'Backward pass failed: {e}')

    # --- Summary ---
    print(f'\n{"="*50}')
    print(f'Results: {PASS} passed, {FAIL} failed')
    if FAIL > 0:
        print('VALIDATION FAILED')
        sys.exit(1)
    else:
        print('VALIDATION PASSED')
        sys.exit(0)

if __name__ == '__main__':
    main()
