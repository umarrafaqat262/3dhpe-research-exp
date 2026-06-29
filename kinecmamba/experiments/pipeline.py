"""
pipeline.py — Full multi-fidelity comparison pipeline

Runs remaining experiments + baseline comparisons for Tier 3 & 4.
Pushes results to GitHub after each tier.

Usage:
  setsid python -u kinecmamba/experiments/pipeline.py > pipeline_output.log 2>&1 &

The setsid + nohup ensures this survives terminal closure.
"""

import os
import sys
import csv
import time
import re
import glob
import subprocess
import yaml
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE, '..', '..')
RESULTS_DIR = os.path.join(os.path.dirname(BASE), 'experiments', 'results')

PYTHON = '/home/ubuntu/miniforge3/envs/posemamba/bin/python'
TRAIN_SCRIPT = os.path.join(BASE, '..', 'train.py')
CONFIG_DIR = os.path.join(BASE, '..', 'configs', 'experiments')

# ─── Tier 3: Baseline at 5 epochs (for fair comparison) ───
TIER3_BASELINE = ('A1_baseline_5ep', 'exp_A1_baseline.yaml', 1470211, 5)

# ─── Tier 4: Top 2 winners from Tier 3 + baseline at 24 epochs ───
TIER4_EXPERIMENTS = [
    ('B1_hypergcn',   'exp_B1_hypergcn.yaml',   1478473, 24),
    ('C1_ssi_msm',    'exp_C1_ssi_msm.yaml',    1483300, 24),
    ('A1_baseline',   'exp_A1_baseline.yaml',   1470211, 24),
]

CSV_FIELDS = ['experiment', 'timestamp', 'config', 'params', 'epochs',
              'train_time', 'mpjpe_p1_last', 'mpjpe_p1_best',
              'p_mpjpe_last', 'p_mpjpe_best', 'loss_3d', 'delta_vs_baseline']

BASELINE_MPJPE_5EP = None  # Will be filled after Tier 3 baseline


def run_capture(cmd, timeout=14400):
    full_env = os.environ.copy()
    full_env['PYTHONWARNINGS'] = 'ignore'
    full_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full_env)
    return result.stdout, result.stderr, result.returncode


def extract_mpjpe_last_epoch(text):
    """Extract MPJPE from the LAST epoch (fixes bug of extracting first)."""
    p1_matches = re.findall(r'Protocol #1 Error \(MPJPE\):([0-9.]+)', text)
    p2_matches = re.findall(r'Protocol #2 Error \(P-MPJPE\):([0-9.]+)', text)
    loss_matches = re.findall(r'3d_train ([0-9.]+)', text)

    p1_last = float(p1_matches[-1]) if p1_matches else None
    p2_last = float(p2_matches[-1]) if p2_matches else None
    loss = float(loss_matches[-1]) if loss_matches else None

    # Best (minimum) MPJPE across all epochs
    p1_best = min(float(x) for x in p1_matches) if p1_matches else None
    p2_best = min(float(x) for x in p2_matches) if p2_matches else None

    return p1_last, p1_best, p2_last, p2_best, loss


def run_experiment(name, config_file, expected_params, num_epochs):
    config_path = os.path.join(CONFIG_DIR, config_file)
    tier_dir = 'tier3' if num_epochs <= 5 else 'tier4'
    output_dir = os.path.join(RESULTS_DIR, tier_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    print('\n%s' % ('=' * 70))
    print('  %s: %s (%d epochs)' % (tier_dir.upper(), name, num_epochs))
    print('  Config: %s' % config_file)
    print('  Output: %s' % output_dir)
    print('%s' % ('=' * 70))

    # Create temp config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['epochs'] = num_epochs
    cfg['no_eval'] = False
    if 'checkpoint' in cfg:
        cfg['checkpoint'] = output_dir

    tmp_config = os.path.join(output_dir, '_run_config.yaml')
    with open(tmp_config, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    train_out = os.path.join(output_dir, 'train_log.txt')
    cmd = [PYTHON, TRAIN_SCRIPT, '--config', tmp_config, '-c', output_dir, '--wandb', 'False']

    print('  Started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M'))
    start = time.time()
    stdout, stderr, rc = run_capture(cmd, timeout=max(14400, num_epochs * 1500))
    elapsed = time.time() - start
    combined = stdout + '\n' + stderr
    with open(train_out, 'w') as f:
        f.write(combined)

    p1_last, p1_best, p2_last, p2_best, loss = extract_mpjpe_last_epoch(combined)
    finished = datetime.now().strftime('%H:%M')

    results = {
        'experiment': name,
        'timestamp': datetime.now().isoformat(),
        'config': config_file,
        'params': expected_params,
        'epochs': num_epochs,
        'train_time': round(elapsed, 1),
        'mpjpe_p1_last': p1_last,
        'mpjpe_p1_best': p1_best,
        'p_mpjpe_last': p2_last,
        'p_mpjpe_best': p2_best,
        'loss_3d': loss,
    }

    print('  Finished: %s (%.1f min)' % (finished, elapsed / 60))
    if rc != 0:
        print('  RC=%d (non-fatal, MPJPE captured from log)' % rc)
    print('  MPJPE last epoch: %s' % ('%.2f' % p1_last if p1_last else 'N/A'))
    print('  MPJPE best epoch: %s' % ('%.2f' % p1_best if p1_best else 'N/A'))
    print('  P-MPJPE last:     %s' % ('%.2f' % p2_last if p2_last else 'N/A'))
    print('  P-MPJPE best:     %s' % ('%.2f' % p2_best if p2_best else 'N/A'))

    return results


def git_push(message):
    """Stage, commit, and push to origin/exp/comparison."""
    print('\n--- Git push: %s ---' % message)
    try:
        cmd = ['git', 'add', '-A']
        subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=30)
        cmd = ['git', 'commit', '-m', message]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        print('  commit: %s' % (r.stdout.strip()[:100] if r.returncode == 0 else 'nothing to commit'))
        cmd = ['git', 'push', 'origin', 'exp/comparison']
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        print('  push: %s' % ('OK' if r.returncode == 0 else r.stderr[:200]))
    except Exception as e:
        print('  git error: %s' % e)


def save_results(all_results, filename):
    if not all_results:
        return
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print('  [saved] %s' % filename)


def print_summary(title, results, baseline_key=None):
    print('\n\n%s' % ('=' * 75))
    print('  %s' % title)
    print('%s' % ('=' * 75))
    h = '{:<18} {:>10} {:>10} {:>10} {:>10}'
    print(h.format('Experiment', 'MPJPE_last', 'MPJPE_best', 'P-MPJPE', 'Δ vs base'))
    print(h.format('-' * 18, '-' * 10, '-' * 10, '-' * 10, '-' * 10))
    baseline_best = None
    for r in results:
        if baseline_key and r['experiment'] == baseline_key:
            baseline_best = r.get('mpjpe_p1_best')
            break
    for r in results:
        p1l = '%.1f' % r['mpjpe_p1_last'] if r.get('mpjpe_p1_last') else 'N/A'
        p1b = '%.1f' % r['mpjpe_p1_best'] if r.get('mpjpe_p1_best') else 'N/A'
        p2 = '%.1f' % r.get('p_mpjpe_best') if r.get('p_mpjpe_best') else 'N/A'
        if baseline_best and r.get('mpjpe_p1_best'):
            delta = r['mpjpe_p1_best'] - baseline_best
            d = '%+.1f' % delta
        else:
            d = '---'
        print(h.format(r['experiment'], p1l, p1b, p2, d))
    print('%s' % ('=' * 75))


def main():
    print('Pipeline started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('Tier 3 results already collected (from earlier tier3.py run)')
    print()

    # ─── Read existing Tier 3 results (corrected with last-epoch values) ───
    tier3_file = os.path.join(RESULTS_DIR, 'tier3', 'tier3_results_corrected.csv')
    tier3_results = []

    # ─── Tier 3補完: Run baseline at 5 epochs ───
    print('\n' + '#' * 75)
    print('#  TIER 3 SUPPLEMENT: Baseline at 5 epochs')
    print('#' * 75)
    tier3_dir = os.path.join(RESULTS_DIR, 'tier3')
    os.makedirs(tier3_dir, exist_ok=True)
    baseline_5ep = run_experiment(*TIER3_BASELINE)
    tier3_results.append(baseline_5ep)
    save_results(tier3_results, tier3_file)

    # Read existing tier3 results and add baseline
    existing_tier3 = []
    csv_path = os.path.join(tier3_dir, 'tier3_results.csv')
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fix: extract correct last-epoch values from logs
                name = row['experiment']
                    log_files = list(glob.glob(os.path.join(tier3_dir, '%s_2026*/log.txt' % name)))
                if log_files:
                    with open(log_files[0]) as lf:
                        log_text = lf.read()
                    p1_last, p1_best, p2_last, p2_best, loss = extract_mpjpe_last_epoch(log_text)
                    row['mpjpe_p1_last'] = p1_last
                    row['mpjpe_p1_best'] = p1_best
                    row['p_mpjpe_last'] = p2_last
                    row['p_mpjpe_best'] = p2_best
                    row['loss_3d'] = loss
                    row['epochs'] = 5
                existing_tier3.append(row)

    all_tier3 = existing_tier3 + tier3_results
    save_results(all_tier3, os.path.join(tier3_dir, 'tier3_results_corrected.csv'))
    print_summary('TIER 3 CORRECTED RESULTS (5 epochs)', all_tier3, baseline_key='A1_baseline_5ep')

    git_push('exp/tier3: corrected last-epoch values + baseline@5ep comparison')

    # ─── Tier 4: Top 3 winners + baseline at 24 epochs ───
    print('\n\n' + '#' * 75)
    print('#  TIER 4: Full training (24 epochs) — Top 3 + Baseline')
    print('#' * 75)
    tier4_dir = os.path.join(RESULTS_DIR, 'tier4')
    os.makedirs(tier4_dir, exist_ok=True)
    tier4_results = []

    for exp in TIER4_EXPERIMENTS:
        result = run_experiment(*exp)
        tier4_results.append(result)
        save_results(tier4_results, os.path.join(tier4_dir, 'tier4_results.csv'))
        git_push('exp/tier4: %s complete (%d epochs)' % (exp[0], exp[3]))

    print_summary('TIER 4 FINAL RESULTS (24 epochs)', tier4_results, baseline_key='A1_baseline')
    save_results(tier4_results, os.path.join(tier4_dir, 'tier4_results.csv'))
    git_push('exp/tier4: all experiments complete')

    print('\n\n' + '=' * 75)
    print('  PIPELINE COMPLETE: %s' % datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('=' * 75)


if __name__ == '__main__':
    main()
