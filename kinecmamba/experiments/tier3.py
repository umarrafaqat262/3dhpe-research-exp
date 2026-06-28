"""
tier3.py — Tier 3 benchmark: 5 epochs on winning experiments

Usage: python experiments/tier3.py
"""

import os
import sys
import csv
import time
import re
import subprocess
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE), 'experiments', 'results')
TIER3_DIR = os.path.join(RESULTS_DIR, 'tier3')
os.makedirs(TIER3_DIR, exist_ok=True)

PYTHON = '/home/ubuntu/miniforge3/envs/posemamba/bin/python'
TRAIN_SCRIPT = os.path.join(BASE, '..', 'train.py')
CONFIG_DIR = os.path.join(BASE, '..', 'configs', 'experiments')

# Winners from Tier 2 — ranked by ΔMPJPE
EXPERIMENTS = [
    ('B2_gat_head',   'exp_B2_gat_head.yaml',   1482499),
    ('A3_ssi',        'exp_A3_ssi.yaml',        1474596),
    ('C1_ssi_msm',    'exp_C1_ssi_msm.yaml',    1483300),
    ('B1_hypergcn',   'exp_B1_hypergcn.yaml',   1478473),
]

EPOCHS = 5
RESULTS_FILE = os.path.join(TIER3_DIR, 'tier3_results.csv')

CSV_FIELDS = ['experiment', 'timestamp', 'config', 'params',
              'train_time', 'mpjpe_p1', 'p_mpjpe', 'loss_3d']

def run_capture(cmd, timeout=14400):
    full_env = os.environ.copy()
    full_env['PYTHONWARNINGS'] = 'ignore'
    full_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full_env)
    return result.stdout, result.stderr, result.returncode

def extract_mpjpe(text):
    m = re.search(r'Protocol #1 Error \(MPJPE\):([0-9.]+)', text)
    p1 = float(m.group(1)) if m else None
    m = re.search(r'Protocol #2 Error \(P-MPJPE\):([0-9.]+)', text)
    p2 = float(m.group(1)) if m else None
    return p1, p2

def run_tier3(name, config_file, expected_params):
    config_path = os.path.join(CONFIG_DIR, config_file)
    output_dir = os.path.join(TIER3_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    print('\n%s' % ('=' * 70))
    print('  TIER 3: %s (%d epochs)' % (name, EPOCHS))
    print('  Config: %s' % config_file)
    print('  Output: %s' % output_dir)
    print('%s' % ('=' * 70))

    results = {
        'experiment': name,
        'timestamp': datetime.now().isoformat(),
        'config': config_file,
        'params': expected_params,
    }

    # Create temp config with epochs=EPOCHS
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['epochs'] = EPOCHS
    cfg['no_eval'] = False
    if 'checkpoint' in cfg:
        cfg['checkpoint'] = output_dir

    tmp_config = os.path.join(output_dir, '_tier3_config.yaml')
    with open(tmp_config, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    train_out = os.path.join(output_dir, 'train_log.txt')
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        '--config', tmp_config,
        '-c', output_dir,
        '--wandb', 'False',
    ]

    print('  Training started at %s...' % datetime.now().strftime('%H:%M'))
    start = time.time()
    stdout, stderr, rc = run_capture(cmd, timeout=14400)
    elapsed = time.time() - start
    results['train_time'] = round(elapsed, 1)

    combined = stdout + '\n' + stderr
    with open(train_out, 'w') as f:
        f.write(combined)

    p1, p2 = extract_mpjpe(combined)
    results['mpjpe_p1'] = p1
    results['p_mpjpe'] = p2

    print('  Finished at %s (%.1f min)' % (datetime.now().strftime('%H:%M'), elapsed / 60))
    if rc != 0:
        print('  RC=%d (post-eval crash likely, MPJPE captured)' % rc)
    print('  MPJPE (P1): %s' % ('%.2f' % p1 if p1 else 'N/A'))
    print('  P-MPJPE (P2): %s' % ('%.2f' % p2 if p2 else 'N/A'))
    print('  Δ vs baseline @1ep: %smm' % ('%.1f' % (p1 - 205.07) if p1 else 'N/A'))

    return results

def main():
    all_results = []
    for name, config, params in EXPERIMENTS:
        result = run_tier3(name, config, params)
        if result:
            all_results.append(result)
            _save_results(all_results)
        else:
            print('  SKIPPED %s due to error' % name)

    print('\n\n%s' % ('=' * 70))
    print('  TIER 3 RESULTS (%d epochs)' % EPOCHS)
    print('%s' % ('=' * 70))
    h = '{:<18} {:>10} {:>10} {:>8}'
    print(h.format('Experiment', 'MPJPE', 'P-MPJPE', 'Time'))
    print(h.format('-' * 18, '-' * 10, '-' * 10, '-' * 8))
    for r in all_results:
        p1 = '%.2f' % r['mpjpe_p1'] if r.get('mpjpe_p1') else 'N/A'
        p2 = '%.2f' % r['p_mpjpe'] if r.get('p_mpjpe') else 'N/A'
        t = '%.0fm' % (r['train_time'] / 60) if r.get('train_time') else 'N/A'
        print(h.format(r['experiment'], p1, p2, t))
    print('%s' % ('=' * 70))
    print('Results saved to: %s' % RESULTS_FILE)

def _save_results(all_results):
    if not all_results:
        return
    with open(RESULTS_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)

if __name__ == '__main__':
    main()
