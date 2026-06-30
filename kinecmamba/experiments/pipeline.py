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
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE, '..', '..')
RESULTS_DIR = os.path.join(os.path.dirname(BASE), 'experiments', 'results')

PYTHON = '/home/ubuntu/miniforge3/envs/posemamba/bin/python'
TRAIN_SCRIPT = os.path.join(BASE, '..', 'train.py')
CONFIG_DIR = os.path.join(BASE, '..', 'configs', 'experiments')

# ─── Tier 3: Baseline at 5 epochs (for fair comparison) ───
TIER3_BASELINE = ('A1_baseline_5ep', 'exp_A1_baseline.yaml', 1470211, 5, False)

# ─── Tier 4: Baseline first (for comparison), then Top 2 winners at 30 epochs ───
TIER4_EXPERIMENTS = [
    ('A1_baseline',   'exp_A1_baseline.yaml',   1470211, 30),
    ('B1_hypergcn',   'exp_B1_hypergcn.yaml',   1478473, 30),
    ('C1_ssi_msm',    'exp_C1_ssi_msm.yaml',    1483300, 30),
]

# ─── Tier 5: Resume from Tier 4 checkpoints → 120 epochs (paper recipe) ───
TIER5_EXPERIMENTS = [
    ('A1_baseline',   'exp_A1_baseline.yaml',   1470211, 120),
    ('B1_hypergcn',   'exp_B1_hypergcn.yaml',   1478473, 120),
    ('C1_ssi_msm',    'exp_C1_ssi_msm.yaml',    1483300, 120),
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


def find_latest_checkpoint(tier_dir, name):
    """Find latest_epoch.bin from any timestamped subdirectory or base dir."""
    # Check base dir first
    base_bin = os.path.join(tier_dir, name, 'latest_epoch.bin')
    if os.path.exists(base_bin):
        return base_bin
    # Check timestamped subdirs (most recent by name sort)
    pattern = os.path.join(tier_dir, '%s_2026*/latest_epoch.bin' % name)
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]
    return None


def load_checkpoint_epoch(bin_path):
    """Load checkpoint and return epoch number."""
    ckpt = torch.load(bin_path, map_location='cpu', weights_only=False)
    return ckpt.get('epoch', 0)


def run_experiment(name, config_file, expected_params, num_epochs, use_wandb=True):
    config_path = os.path.join(CONFIG_DIR, config_file)
    tier_dir = 'tier3' if num_epochs <= 5 else 'tier4'
    output_dir = os.path.join(RESULTS_DIR, tier_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    print('\n%s' % ('=' * 70))
    print('  %s: %s (%d epochs)' % (tier_dir.upper(), name, num_epochs))
    print('  Config: %s' % config_file)
    print('  Output: %s' % output_dir)
    print('%s' % ('=' * 70))

    # Find existing checkpoint from ANY timestamped subdir or base dir
    existing_ckpt = find_latest_checkpoint(os.path.join(RESULTS_DIR, tier_dir), name)
    resume_epoch = 0
    if existing_ckpt:
        resume_epoch = load_checkpoint_epoch(existing_ckpt)
        print('  FOUND checkpoint epoch %d at: %s' % (resume_epoch, existing_ckpt))
        print('  Will resume → %d epochs total' % num_epochs)
    else:
        print('  Starting from scratch: %d epochs' % num_epochs)

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
    wandb_flag = 'True' if use_wandb else 'False'
    cmd = [PYTHON, TRAIN_SCRIPT, '--config', tmp_config, '-c', output_dir, '--wandb', wandb_flag]
    # Pass -r with full path to existing checkpoint so train.py can find it
    # even though train.py appends a timestamp to -c path
    if existing_ckpt:
        cmd.extend(['-r', existing_ckpt])

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


def run_tier5_experiment(name, config_file, expected_params, num_epochs):
    """Resume from Tier 4 checkpoint and train to 120 epochs."""
    config_path = os.path.join(CONFIG_DIR, config_file)
    tier5_dir = os.path.join(RESULTS_DIR, 'tier5')
    output_dir = os.path.join(tier5_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    # Find Tier 4 checkpoint
    tier4_dir = os.path.join(RESULTS_DIR, 'tier4')
    existing_ckpt = find_latest_checkpoint(tier4_dir, name)

    if not existing_ckpt:
        print('  ERROR: No Tier 4 checkpoint found for %s' % name)
        return None

    resume_epoch = load_checkpoint_epoch(existing_ckpt)
    print('\n%s' % ('=' * 70))
    print('  TIER 5: %s (%d epochs, resume from epoch %d)' % (name, num_epochs, resume_epoch))
    print('  Config: %s' % config_file)
    print('  Tier 4 checkpoint: %s' % existing_ckpt)
    print('  Output: %s' % output_dir)
    print('%s' % ('=' * 70))

    # Create temp config with epochs=120
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
    cmd = [PYTHON, TRAIN_SCRIPT, '--config', tmp_config, '-c', output_dir,
           '--wandb', 'True', '-r', existing_ckpt]

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
        if baseline_key and r.get('experiment') == baseline_key:
            try:
                baseline_best = float(r.get('mpjpe_p1_best', 0) or 0)
            except (ValueError, TypeError):
                pass
            break
    for r in results:
        p1l_raw = r.get('mpjpe_p1_last')
        p1b_raw = r.get('mpjpe_p1_best')
        p2_raw = r.get('p_mpjpe_best')
        try:
            p1l = '%.1f' % float(p1l_raw) if p1l_raw else 'N/A'
        except (ValueError, TypeError):
            p1l = 'N/A'
        try:
            p1b = '%.1f' % float(p1b_raw) if p1b_raw else 'N/A'
        except (ValueError, TypeError):
            p1b = 'N/A'
        try:
            p2 = '%.1f' % float(p2_raw) if p2_raw else 'N/A'
        except (ValueError, TypeError):
            p2 = 'N/A'
        if baseline_key and r.get('experiment') == baseline_key:
            try:
                baseline_best = float(p1b_raw)
            except (ValueError, TypeError):
                baseline_best = None
        try:
            cur_best = float(p1b_raw) if p1b_raw else None
        except (ValueError, TypeError):
            cur_best = None
        if baseline_best and cur_best:
            d = '%+.1f' % (cur_best - baseline_best)
        else:
            d = '---'
        print(h.format(r.get('experiment', '?'), p1l, p1b, p2, d))
    print('%s' % ('=' * 75))


def main():
    print('Pipeline started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('Tier 3 results already collected (from earlier tier3.py run)')
    print()

    # ─── Read existing Tier 3 results (corrected with last-epoch values) ───
    tier3_file = os.path.join(RESULTS_DIR, 'tier3', 'tier3_results_corrected.csv')
    tier3_results = []

    # ─── Tier 3補完: Run baseline at 5 epochs (skip if already done) ───
    tier3_dir = os.path.join(RESULTS_DIR, 'tier3')
    tier3_corrected = os.path.join(tier3_dir, 'tier3_results_corrected.csv')
    existing_tier3 = []

    if os.path.exists(tier3_corrected):
        print('Tier 3 corrected results already exist, skipping Tier 3 baseline run')
        with open(tier3_corrected) as f:
            existing_tier3 = list(csv.DictReader(f))
        print_summary('TIER 3 CORRECTED RESULTS (5 epochs)', existing_tier3, baseline_key='A1_baseline_5ep')
    else:
        print('\n' + '#' * 75)
        print('#  TIER 3 SUPPLEMENT: Baseline at 5 epochs')
        print('#' * 75)
        os.makedirs(tier3_dir, exist_ok=True)
        baseline_5ep = run_experiment(*TIER3_BASELINE)
        tier3_results.append(baseline_5ep)
        save_results(tier3_results, tier3_file)

        # Read existing tier3 results and add baseline
        csv_path = os.path.join(tier3_dir, 'tier3_results.csv')
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
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
        save_results(all_tier3, tier3_corrected)
        git_push('exp/tier3: corrected last-epoch values + baseline@5ep comparison')

    # ─── Tier 4: Top 2 winners + baseline at 30 epochs (wandb online) ───
    print('\n\n' + '#' * 75)
    print('#  TIER 4: Full training (30 epochs) — Top 2 + Baseline')
    print('#' * 75)
    tier4_dir = os.path.join(RESULTS_DIR, 'tier4')
    os.makedirs(tier4_dir, exist_ok=True)
    tier4_results = []

    for exp in TIER4_EXPERIMENTS:
        name, config, params, num_epochs = exp
        # Check if already completed target epochs (search all timestamped dirs)
        existing_ckpt = find_latest_checkpoint(tier4_dir, name)
        if existing_ckpt:
            ckpt_epoch = load_checkpoint_epoch(existing_ckpt)
            if ckpt_epoch >= num_epochs:
                print('\n  SKIP %s: already at epoch %d (target=%d)' % (name, ckpt_epoch, num_epochs))
                # Still collect results from existing log
                log_files = list(glob.glob(os.path.join(tier4_dir, '%s_2026*/log.txt' % name)))
                if log_files:
                    with open(log_files[0]) as lf:
                        p1l, p1b, p2l, p2b, loss = extract_mpjpe_last_epoch(lf.read())
                    tier4_results.append({
                        'experiment': name, 'config': config, 'params': params,
                        'epochs': ckpt_epoch, 'mpjpe_p1_last': p1l, 'mpjpe_p1_best': p1b,
                        'p_mpjpe_last': p2l, 'p_mpjpe_best': p2b, 'loss_3d': loss,
                    })
                continue
        result = run_experiment(*exp)
        tier4_results.append(result)
        save_results(tier4_results, os.path.join(tier4_dir, 'tier4_results.csv'))
        git_push('exp/tier4: %s complete (%d epochs)' % (name, num_epochs))

    print_summary('TIER 4 FINAL RESULTS (30 epochs)', tier4_results, baseline_key='A1_baseline')
    save_results(tier4_results, os.path.join(tier4_dir, 'tier4_results.csv'))
    git_push('exp/tier4: all experiments complete')

    # ─── Tier 5: Resume from Tier 4 → 120 epochs (wandb online) ───
    print('\n\n' + '#' * 75)
    print('#  TIER 5: Extended training (30→120 epochs) — A1, B1, C1')
    print('#' * 75)
    tier5_dir = os.path.join(RESULTS_DIR, 'tier5')
    os.makedirs(tier5_dir, exist_ok=True)
    tier5_results = []

    for exp in TIER5_EXPERIMENTS:
        name, config, params, num_epochs = exp
        # Check if already completed target epochs
        existing_ckpt = find_latest_checkpoint(tier5_dir, name)
        if existing_ckpt:
            ckpt_epoch = load_checkpoint_epoch(existing_ckpt)
            if ckpt_epoch >= num_epochs:
                print('\n  SKIP %s: already at epoch %d (target=%d)' % (name, ckpt_epoch, num_epochs))
                log_files = list(glob.glob(os.path.join(tier5_dir, '%s_2026*/log.txt' % name)))
                if log_files:
                    with open(log_files[0]) as lf:
                        p1l, p1b, p2l, p2b, loss = extract_mpjpe_last_epoch(lf.read())
                    tier5_results.append({
                        'experiment': name, 'config': config, 'params': params,
                        'epochs': ckpt_epoch, 'mpjpe_p1_last': p1l, 'mpjpe_p1_best': p1b,
                        'p_mpjpe_last': p2l, 'p_mpjpe_best': p2b, 'loss_3d': loss,
                    })
                continue
        result = run_tier5_experiment(name, config, params, num_epochs)
        if result:
            tier5_results.append(result)
            save_results(tier5_results, os.path.join(tier5_dir, 'tier5_results.csv'))
            git_push('exp/tier5: %s complete (%d epochs)' % (name, num_epochs))

    print_summary('TIER 5 FINAL RESULTS (120 epochs)', tier5_results, baseline_key='A1_baseline')
    save_results(tier5_results, os.path.join(tier5_dir, 'tier5_results.csv'))
    git_push('exp/tier5: all experiments complete')

    print('\n\n' + '=' * 75)
    print('  PIPELINE COMPLETE: %s' % datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('=' * 75)


if __name__ == '__main__':
    main()
