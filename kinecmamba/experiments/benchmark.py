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
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE), 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON = '/home/ubuntu/miniforge3/envs/posemamba/bin/python'
TRAIN_SCRIPT = os.path.join(BASE, '..', 'train.py')
CONFIG_DIR = os.path.join(BASE, '..', 'configs', 'experiments')

EXPERIMENTS = [
    # (name, config, expected_params)
    ('A1_baseline',    'exp_A1_baseline.yaml',     1470211),
    ('A2_confidence',  'exp_A2_confidence.yaml',    1470275),
    ('A5_bone_vectors','exp_A5_bone_vectors.yaml',  1470403),
    ('A3_ssi',         'exp_A3_ssi.yaml',           1474211),
    ('A4_msm',         'exp_A4_msm.yaml',           1478915),
    ('B1_hypergcn',    'exp_B1_hypergcn.yaml',      1482883),
    ('B2_gat_head',    'exp_B2_gat_head.yaml',      1478659),
    ('C1_ssi_msm',     'exp_C1_ssi_msm.yaml',       1483300),
    ('B3_v2_default',  'exp_B3_v2.yaml',            1470211),
]

RESULTS_FILE = os.path.join(RESULTS_DIR, 'benchmark_results.csv')

def run_capture(cmd, timeout=1800):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def extract_mpjpe(text):
    m = re.search(r'Protocol #1 Error \(MPJPE\):([0-9.]+)', text)
    p1 = float(m.group(1)) if m else None

    m = re.search(r'Protocol #2 Error \(P-MPJPE\):([0-9.]+)', text)
    p2 = float(m.group(1)) if m else None

    m = re.search(r'3d_train ([0-9.]+)', text)
    loss = float(m.group(1)) if m else None

    return p1, p2, loss

def run_experiment(name, config_file):
    config_path = os.path.join(CONFIG_DIR, config_file)
    output_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  BENCHMARK: {name}')
    print(f'  Config: {config_file}')
    print(f'  Output: {output_dir}')
    print(f'{"="*60}')

    results = {
        'experiment': name,
        'timestamp': datetime.now().isoformat(),
        'config': config_file,
    }

    # Tier 1: Zero-shot validation
    print('\n[TIER 1] Zero-shot validation...')
    cmd = [PYTHON, '-c', f'''
import sys; sys.path.insert(0, "kinecmamba")
from easydict import EasyDict
import yaml, torch
with open("{config_path}") as f:
    cfg = EasyDict(yaml.safe_load(f))
from lib.utils.learning import load_backbone
m = load_backbone(cfg)
n = sum(p.numel() for p in m.parameters())
print(f"PARAMS={{n}}")

# Gradient check: one forward/backward on random input
m = m.cuda()
x = torch.randn(2, 243, 17, cfg.get("input_channels", 2)).cuda()
y = m(x)
loss = y.mean()
loss.backward()

total_norm = 0
max_norm = 0
max_name = ""
for name, p in m.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        total_norm += gn
        if gn > max_norm:
            max_norm = gn
            max_name = name
print(f"GRAD_AVG={{total_norm/len(list(m.parameters())):.6f}}")
print(f"GRAD_MAX={{max_norm:.6f}} ({max_name})")
print(f"GRAD_FLOW={{'OK' if total_norm > 0 else 'DEAD'}}")

# Module diagnostics
if hasattr(m, "ssi") and hasattr(m.ssi, "W"):
    w_ent = -(m.ssi.W.softmax(dim=-1) * m.ssi.W.softmax(dim=-1).log()).sum(dim=-1).mean().item()
    print(f"SSI_W_ENT={{w_ent:.4f}}")
if hasattr(m, "msm") and hasattr(m.msm, "bias"):
    delta_std = m.msm.bias.std().item()
    delta_mean = m.msm.bias.mean().item()
    print(f"MSM_DELTA_STD={{delta_std:.6f}}")
    print(f"MSM_DELTA_MEAN={{delta_mean:.6f}}")
if hasattr(m, "hypergcn") and hasattr(m.hypergcn, "alpha"):
    gate_val = m.hypergcn.alpha.sigmoid().mean().item()
    print(f"HYPERGCN_GATE={{gate_val:.4f}}")
if hasattr(m, "head") and hasattr(m.head, "W_query"):
    print(f"GAT_HEAD=True")
''']
    stdout, stderr, rc = run_capture(cmd, timeout=120)
    if rc != 0:
        print(f'  ERROR: {stderr[:500]}')
        return None

    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('PARAMS='):
            results['params'] = int(line.split('=')[1])
            print(f'  Params: {results["params"]:,}')
        elif line.startswith('GRAD_AVG='):
            results['grad_avg'] = float(line.split('=')[1])
        elif line.startswith('GRAD_MAX='):
            parts = line.split('=')
            results['grad_max'] = float(parts[1].split()[0])
        elif line.startswith('GRAD_FLOW='):
            results['grad_flow'] = line.split('=')[1]
            print(f'  Gradient flow: {results["grad_flow"]}')
        elif line.startswith('SSI_W_ENT='):
            results['ssi_w_ent'] = float(line.split('=')[1])
            print(f'  SSI W entropy: {results["ssi_w_ent"]:.4f}')
        elif line.startswith('MSM_DELTA_STD='):
            results['msm_delta_std'] = float(line.split('=')[1])
        elif line.startswith('MSM_DELTA_MEAN='):
            results['msm_delta_mean'] = float(line.split('=')[1])
            print(f'  MSM delta bias: mean={results["msm_delta_mean"]:.4f} std={results["msm_delta_std"]:.4f}')
        elif line.startswith('HYPERGCN_GATE='):
            results['hypergcn_gate'] = float(line.split('=')[1])
            print(f'  HyperGCN gate: {results["hypergcn_gate"]:.4f}')
        elif line.startswith('GAT_HEAD='):
            results['gat_head'] = line.split('=')[1]

    # Tier 2: Train for 1 epoch
    print('\n[TIER 2] Training (1 epoch, no_eval)...')
    train_out = os.path.join(output_dir, 'train_log.txt')
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        '--config', config_path,
        '-c', output_dir,
        '--wandb', 'False',
    ]
    # Override to 1 epoch with eval
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    start = time.time()
    stdout, stderr, rc = run_capture(cmd, timeout=3600)
    elapsed = time.time() - start
    results['train_time'] = round(elapsed, 1)

    with open(train_out, 'w') as f:
        f.write(stdout)

    # Parse validation results
    p1, p2, loss = extract_mpjpe(stdout)
    results['mpjpe_p1'] = p1
    results['p_mpjpe'] = p2
    results['loss_3d'] = loss

    print(f'  Time: {results["train_time"]}s')
    print(f'  MPJPE: {p1}')
    print(f'  P-MPJPE: {p2}')

    return results

def main():
    all_results = []

    for name, config, expected_params in EXPERIMENTS:
        result = run_experiment(name, config)
        if result:
            all_results.append(result)
            # Save intermediate results
            _save_results(all_results)
        else:
            print(f'  SKIPPED {name} due to error')

    # Final summary
    print(f'\n\n{"="*70}')
    print(f'  BENCHMARK RESULTS')
    print(f'{"="*70}')
    print(f'{"Experiment":<20} {"MPJPE":>8} {"P-MPJPE":>8} {"Params":>10} {"Time":>8} {"Grad":>8}')
    print(f'{ "-"*20:>20} {"-"*8:>8} {"-"*8:>8} {"-"*10:>10} {"-"*8:>8} {"-"*8:>8}')
    for r in all_results:
        p1 = f'{r.get("mpjpe_p1", "N/A"):.2f}' if r.get("mpjpe_p1") else "N/A"
        p2 = f'{r.get("p_mpjpe", "N/A"):.2f}' if r.get("p_mpjpe") else "N/A"
        params = f'{r.get("params", 0):,}'
        t = f'{r.get("train_time", 0):.0f}s'
        g = r.get("grad_flow", "N/A")
        print(f'{r["experiment"]:<20} {p1:>8} {p2:>8} {params:>10} {t:>8} {g:>8}')
    print(f'{"="*70}')
    print(f'Results saved to: {RESULTS_FILE}')

def _save_results(all_results):
    if not all_results:
        return
    keys = all_results[0].keys()
    with open(RESULTS_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)
    print(f'  [saved] {RESULTS_FILE}')

if __name__ == '__main__':
    main()
