"""
benchmark.py — Multi-experiment comparison runner

Runs all experiments through Tier 1 (zero-shot) and Tier 2 (1 epoch),
collects MPJPE, P-MPJPE, gradient stats, and module diagnostics.
Outputs results/benchmark_results.csv

Usage: python experiments/benchmark.py
"""

import os
import sys
import csv
import time
import re
import subprocess
import json
import tempfile
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE), 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON = '/home/ubuntu/miniforge3/envs/posemamba/bin/python'
TRAIN_SCRIPT = os.path.join(BASE, '..', 'train.py')
CONFIG_DIR = os.path.join(BASE, '..', 'configs', 'experiments')

EXPERIMENTS = [
    ('A1_baseline',    'exp_A1_baseline.yaml'),
    ('A2_confidence',  'exp_A2_confidence.yaml'),
    ('A5_bone_vectors','exp_A5_bone_vectors.yaml'),
    ('A3_ssi',         'exp_A3_ssi.yaml'),
    ('A4_msm',         'exp_A4_msm.yaml'),
    ('B1_hypergcn',    'exp_B1_hypergcn.yaml'),
    ('B2_gat_head',    'exp_B2_gat_head.yaml'),
    ('C1_ssi_msm',     'exp_C1_ssi_msm.yaml'),
    ('B3_v2_default',  'exp_B3_v2.yaml'),
]

RESULTS_FILE = os.path.join(RESULTS_DIR, 'benchmark_results.csv')

def run_capture(cmd, timeout=1800):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

CSV_FIELDS = ['experiment', 'timestamp', 'config', 'params', 'grad_flow',
              'grad_avg', 'grad_max', 'grad_max_name',
              'ssi_w_ent', 'msm_delta_std', 'msm_delta_mean',
              'hypergcn_gate', 'gat_head',
              'train_time', 'mpjpe_p1', 'p_mpjpe', 'loss_3d']

def extract_mpjpe(text):
    m = re.search(r'Protocol #1 Error \(MPJPE\):([0-9.]+)', text)
    p1 = float(m.group(1)) if m else None
    m = re.search(r'Protocol #2 Error \(P-MPJPE\):([0-9.]+)', text)
    p2 = float(m.group(1)) if m else None
    m = re.search(r'3d_train ([0-9.]+)', text)
    loss = float(m.group(1)) if m else None
    return p1, p2, loss

def run_zero_shot(config_path):
    """Run zero-shot diagnostics in a subprocess."""
    # Escape config_path for the inner script
    escaped_path = config_path.replace('\\', '\\\\').replace("'", "\\'")
    script = '''
import sys; sys.path.insert(0, "kinecmamba")
from easydict import EasyDict
import yaml, torch
with open("''' + escaped_path + '''") as f:
    cfg = EasyDict(yaml.safe_load(f))
from lib.utils.learning import load_backbone
m = load_backbone(cfg)
n = sum(p.numel() for p in m.parameters())
print("PARAMS=%d" % n)
m = m.cuda()
x = torch.randn(2, 243, 17, cfg.get("input_channels", 2)).cuda()
y = m(x)
loss = y.mean()
loss.backward()
total_norm = 0.0
max_norm = 0.0
max_name = "none"
count = 0
for name, p in m.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        total_norm += gn
        count += 1
        if gn > max_norm:
            max_norm = gn
            max_name = name
avg_norm = total_norm / max(1, count)
print("GRAD_AVG=%.6f" % avg_norm)
print("GRAD_MAX=%.6f (%s)" % (max_norm, max_name))
print("GRAD_FLOW=%s" % ("OK" if total_norm > 0 else "DEAD"))
if hasattr(m, "ssi") and hasattr(m.ssi, "W"):
    w_ent = -(m.ssi.W.softmax(dim=-1) * m.ssi.W.softmax(dim=-1).log()).sum(dim=-1).mean().item()
    print("SSI_W_ENT=%.4f" % w_ent)
if hasattr(m, "msm") and hasattr(m.msm, "bias"):
    delta_std = m.msm.bias.std().item()
    delta_mean = m.msm.bias.mean().item()
    print("MSM_DELTA_STD=%.6f" % delta_std)
    print("MSM_DELTA_MEAN=%.6f" % delta_mean)
if hasattr(m, "hypergcn") and hasattr(m.hypergcn, "alpha"):
    gate_val = m.hypergcn.alpha.sigmoid().mean().item()
    print("HYPERGCN_GATE=%.4f" % gate_val)
if hasattr(m, "head") and hasattr(m.head, "W_query"):
    print("GAT_HEAD=True")
'''
    cmd = [PYTHON, '-c', script]
    stdout, stderr, rc = run_capture(cmd, timeout=120)
    if rc != 0:
        print('  ERROR in zero-shot: %s' % stderr[:500])
        return None
    results = {}
    for line in stdout.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('PARAMS='):
            results['params'] = int(line.split('=')[1])
        elif line.startswith('GRAD_AVG='):
            results['grad_avg'] = float(line.split('=')[1])
        elif line.startswith('GRAD_MAX='):
            parts = line.split('=')
            val_name = parts[1].split('(')
            results['grad_max'] = float(val_name[0].strip())
            results['grad_max_name'] = val_name[1].rstrip(')') if len(val_name) > 1 else ''
        elif line.startswith('GRAD_FLOW='):
            results['grad_flow'] = line.split('=')[1]
        elif line.startswith('SSI_W_ENT='):
            results['ssi_w_ent'] = float(line.split('=')[1])
        elif line.startswith('MSM_DELTA_STD='):
            results['msm_delta_std'] = float(line.split('=')[1])
        elif line.startswith('MSM_DELTA_MEAN='):
            results['msm_delta_mean'] = float(line.split('=')[1])
        elif line.startswith('HYPERGCN_GATE='):
            results['hypergcn_gate'] = float(line.split('=')[1])
        elif line.startswith('GAT_HEAD='):
            results['gat_head'] = line.split('=')[1]
    return results, stdout

def run_experiment(name, config_file):
    config_path = os.path.join(CONFIG_DIR, config_file)
    output_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    print('\n%s' % ('=' * 60))
    print('  BENCHMARK: %s' % name)
    print('  Config: %s' % config_file)
    print('  Output: %s' % output_dir)
    print('%s' % ('=' * 60))

    results = {
        'experiment': name,
        'timestamp': datetime.now().isoformat(),
        'config': config_file,
    }

    # Tier 1: Zero-shot diagnostics
    print('\n[TIER 1] Zero-shot diagnostics...')
    zero_shot = run_zero_shot(config_path)
    if zero_shot is None:
        print('  SKIPPED %s (zero-shot failed)' % name)
        return None
    zs_results, zs_stdout = zero_shot
    results.update(zs_results)
    for k, v in zs_results.items():
        if k in ('params',):
            print('  Params: %s' % f'{v:,}')
        elif k in ('grad_avg', 'grad_max'):
            print('  %s: %.6f' % (k, v))
        elif k == 'grad_flow':
            print('  Gradient flow: %s' % v)
        elif k == 'ssi_w_ent':
            print('  SSI W entropy: %.4f' % v)
        elif k == 'msm_delta_std':
            print('  MSM delta std: %.6f' % v)
        elif k == 'msm_delta_mean':
            print('  MSM delta mean: %.6f' % v)
        elif k == 'hypergcn_gate':
            print('  HyperGCN gate: %.4f' % v)
        elif k == 'gat_head':
            print('  GAT Head: %s' % v)

    # Tier 2: Train for 1 epoch
    print('\n[TIER 2] Training (1 epoch)...')

    # Create a temp config with epochs=1
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['epochs'] = 1
    cfg['no_eval'] = False
    # Override checkpoint dir to avoid cluttering wandb/checkpoints
    if 'checkpoint' in cfg:
        cfg['checkpoint'] = output_dir

    tmp_config = os.path.join(output_dir, '_benchmark_config.yaml')
    with open(tmp_config, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    train_out = os.path.join(output_dir, 'train_log.txt')
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        '--config', tmp_config,
        '-c', output_dir,
        '--wandb', 'False',
    ]
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    start = time.time()
    stdout, stderr, rc = run_capture(cmd, timeout=7200)
    elapsed = time.time() - start
    results['train_time'] = round(elapsed, 1)

    # Log output goes to stderr (via logging), so merge both for parsing
    combined = stdout + '\n' + stderr
    with open(train_out, 'w') as f:
        f.write(combined)

    # Always try to parse MPJPE (printed before eval layer_hooks crash)
    p1, p2, loss = extract_mpjpe(combined)
    results['mpjpe_p1'] = p1
    results['p_mpjpe'] = p2
    results['loss_3d'] = loss

    if rc != 0:
        print('  ERROR (non-zero rc=%d): %s' % (rc, stderr[:200]))
        results['train_error'] = stderr[:200]
    print('  Time: %.1fs' % results['train_time'])
    print('  MPJPE (P1): %s' % ('%.2f' % p1 if p1 else 'N/A'))
    print('  P-MPJPE (P2): %s' % ('%.2f' % p2 if p2 else 'N/A'))
    print('  3D loss: %s' % ('%.4f' % loss if loss else 'N/A'))

    return results


def main():
    all_results = []

    for name, config in EXPERIMENTS:
        result = run_experiment(name, config)
        if result:
            all_results.append(result)
            _save_results(all_results)
        else:
            print('  SKIPPED %s due to error' % name)

    # Final summary
    print('\n\n%s' % ('=' * 70))
    print('  BENCHMARK RESULTS')
    print('%s' % ('=' * 70))
    header = '{:<20} {:>8} {:>8} {:>10} {:>8} {:>8}'
    sep = '{:<20} {:>8} {:>8} {:>10} {:>8} {:>8}'
    print(header.format('Experiment', 'MPJPE', 'P-MPJPE', 'Params', 'Time', 'Grad'))
    print(sep.format('-' * 20, '-' * 8, '-' * 8, '-' * 10, '-' * 8, '-' * 8))
    for r in all_results:
        p1 = '%.2f' % r['mpjpe_p1'] if r.get('mpjpe_p1') is not None else 'N/A'
        p2 = '%.2f' % r['p_mpjpe'] if r.get('p_mpjpe') is not None else 'N/A'
        params = f'{r.get("params", 0):,}'
        t = '%.0fs' % r.get('train_time', 0)
        g = r.get('grad_flow', 'N/A')
        print(header.format(r['experiment'], p1, p2, params, t, g))
    print('%s' % ('=' * 70))
    print('Results saved to: %s' % RESULTS_FILE)

def _save_results(all_results):
    if not all_results:
        return
    with open(RESULTS_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print('  [saved] %s' % RESULTS_FILE)

if __name__ == '__main__':
    main()
