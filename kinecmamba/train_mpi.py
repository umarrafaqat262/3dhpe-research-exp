"""MPI-INF-3DHP train + eval for KinecMamba.

Faithful sibling of train.py (same args, model build, AdamW + LR schedule, EMA,
checkpoint-load incl. -e eval-only, loss assembly WITH bone losses) but:
  - dataset comes from lib/data/dataset_3dhp.py (train clips + per-seq test),
  - set_bfs_order('mpi') is called after import when forward_type == 'v2_bfs',
  - evaluate() is replaced by evaluate_mpi(): flip-TTA, per-sequence prediction
    accumulation, Python MPJPE/PCK@150/AUC, and export of inference_data.mat.
  - in_channels = 2 (GT 2D input, no confidence).

Do NOT use this for H36M; H36M stays on train.py.
"""

import os
import numpy as np
import argparse
import errno
import math
import datetime
import time
import copy
import random
import prettytable
import yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_3dhp import (MPITrainDataset3D, MPITestDataset3D,
                                   flip_data_mpi, MPI_ROOT,
                                   JOINTS_LEFT, JOINTS_RIGHT)
from lib.eval.metrics_3dhp import mpjpe_mm, pck, auc, export_inference_mat
import logger
from logger import colorlogger

# Switch the BFS cross-scan joint permutation to MPI order (no effect on H36M runs,
# which never import this module). Only meaningful for forward_type == 'v2_bfs'.
from lib.model import csms6s


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/mpi_inf_3dhp/exp_mpi_bfs.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--wandb', default=True, type=lambda x: x.lower() == 'true', help='enable wandb logging')
    parser.add_argument('--wandb_project', default='PoseMamba-mpi', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', default='msds24068-itu-lahore', type=str, help='wandb entity name')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    """Exponential moving average of model weights (identical to train.py)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)


def ema_swap_in(model, ema):
    backup = {k: model.state_dict()[k].detach().clone() for k in ema.shadow}
    msd = model.state_dict()
    for k in ema.shadow:
        msd[k].copy_(ema.shadow[k])
    return backup


def ema_swap_out(model, backup):
    msd = model.state_dict()
    for k, v in backup.items():
        msd[k].copy_(v)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss, is_best=False):
    log.info(f'Saving checkpoint to {chk_path}')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss
    }, chk_path)
    if wandb.run is not None and is_best:
        artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model', metadata={'epoch': epoch + 1, 'min_loss': min_loss})
        artifact.add_file(chk_path)
        wandb.log_artifact(artifact)


def _flip_torch(x):
    """Horizontal flip of a torch tensor [B,T,17,C] in MPI joint order."""
    out = x.clone()
    out[..., 0] *= -1
    out[..., JOINTS_LEFT + JOINTS_RIGHT, :] = out[..., JOINTS_RIGHT + JOINTS_LEFT, :]
    return out


@torch.no_grad()
def evaluate_mpi(args, model_pos, test_loader, checkpoint_dir):
    """Per-sequence eval with flip-TTA. Accumulates predictions per sequence,
    computes Python MPJPE/PCK@150/AUC (monitoring), and writes inference_data.mat
    (per-seq keys, [J,3,1,T]) for the official MATLAB scorer."""
    log.info('INFO: Testing (MPI-INF-3DHP)')
    model_pos.eval()

    ds = test_loader.dataset
    seq_names = ds.seq_names
    # per-sequence prediction buffers (mm), full sequence length
    pred_buf = {name: np.zeros((ds.seq_len[name], 17, 3), dtype=np.float32) for name in seq_names}

    for batch in tqdm(test_loader):
        batch_input, batch_gt, batch_valid, seq_id, start, length = batch
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]

        if args.flip:
            pred1 = model_pos(batch_input)
            pred_flip = model_pos(_flip_torch(batch_input))
            pred2 = _flip_torch(pred_flip)
            pred = (pred1 + pred2) / 2
        else:
            pred = model_pos(batch_input)

        # root-relative around joint 14 (both pred and GT use the MPI pelvis)
        pred = pred - pred[:, :, MPI_ROOT:MPI_ROOT + 1, :]
        pred[:, :, MPI_ROOT, :] = 0.0
        pred = pred.cpu().numpy()

        for b in range(pred.shape[0]):
            name = seq_names[int(seq_id[b])]
            st = int(start[b]); ln = int(length[b])
            pred_buf[name][st:st + ln] = pred[b, :ln]

    # metrics (monitoring only; paper numbers come from the MATLAB scorer on the .mat)
    all_pred, all_gt, all_valid = [], [], []
    summary = prettytable.PrettyTable()
    summary.field_names = ['seq', 'MPJPE(mm)', 'PCK@150', 'AUC']
    for name in seq_names:
        p = pred_buf[name]
        g = ds.gt_by_seq[name]
        v = ds.valid_by_seq[name]
        summary.add_row([name,
                         f'{mpjpe_mm(p, g, v):.2f}',
                         f'{pck(p, g, v):.4f}',
                         f'{auc(p, g, v):.4f}'])
        all_pred.append(p); all_gt.append(g); all_valid.append(v)
    all_pred = np.concatenate(all_pred, 0)
    all_gt = np.concatenate(all_gt, 0)
    all_valid = np.concatenate(all_valid, 0)
    e_mpjpe = mpjpe_mm(all_pred, all_gt, all_valid)
    e_pck = pck(all_pred, all_gt, all_valid)
    e_auc = auc(all_pred, all_gt, all_valid)
    log.info(summary)
    log.info(f'Overall MPJPE {e_mpjpe:.2f}mm  PCK@150 {e_pck:.4f}  AUC {e_auc:.4f}')

    # export inference_data.mat for the official MATLAB PCK/AUC scripts
    out_path = os.path.join(checkpoint_dir, 'inference_data.mat')
    export_inference_mat(pred_buf, out_path)
    log.info(f'Exported inference data to {out_path}')

    return e_mpjpe, e_pck, e_auc


def train_epoch(args, model_pos, train_loader, losses, optimizer, accum_steps=1, ema=None):
    """MPI train epoch. GT arrives already root-relative (root=14 zeroed) from the
    dataset, so we do NOT re-root here (unlike train.py's H36M path). Full loss
    assembly with bone losses is kept."""
    model_pos.train()
    optimizer.zero_grad()
    w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4])
    if torch.cuda.is_available():
        w_mpjpe = w_mpjpe.cuda()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]

        predicted_3d_pos = model_pos(batch_input)

        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        # MPI-specific bone losses: MPI's 17-joint skeleton (root=14) has a different
        # joint order than H36M, so use the MPI limb table (loss.py get_limb_lens_mpi).
        loss_lv = loss_limb_var_mpi(predicted_3d_pos)
        loss_lg = loss_limb_gt_mpi(predicted_3d_pos, batch_gt)
        loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)

        dif_seq = predicted_3d_pos[:, 1:, :, :] - predicted_3d_pos[:, :-1, :, :]
        weights_joints = torch.ones_like(dif_seq)
        weights_joints = torch.mul(weights_joints.permute(0, 1, 3, 2), w_mpjpe).permute(0, 1, 3, 2)
        loss_diff = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

        loss_total = args.lambda_3d * loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     getattr(args, 'lambda_3dw', 0.0) * loss_3d_w + \
                     getattr(args, 'lambda_diff', 0.0) * loss_diff

        losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total = loss_total / accum_steps
        loss_total.backward()

        if (idx + 1) % accum_steps == 0:
            if hasattr(args, 'grad_clip_norm') and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_pos.parameters(), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model_pos)


def get_beijing_timestamp():
    local_offset = time.localtime().tm_gmtoff
    beijing_offset = int(8 * 60 * 60)
    offset = local_offset - beijing_offset
    timestamp = int(datetime.datetime.now().timestamp())
    return timestamp - offset


def train_with_config(args, opts):
    opts.checkpoint = opts.checkpoint + '_' + datetime.datetime.fromtimestamp(get_beijing_timestamp()).strftime('%Y_%m_%d_T_%H_%M_%S')
    global log
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    log = colorlogger(opts.checkpoint, log_name='log.txt')
    log.info(args)
    with open(os.path.join(opts.checkpoint, 'config.yaml'), 'w') as f:
        yaml.dump(args, f, sort_keys=False)
    log.info(f"Number of GPUs found:{torch.cuda.device_count()}")

    # Switch BFS scan order to MPI-INF-3DHP joints for the BFS forward type.
    if getattr(args, 'forward_type', '') == 'v2_bfs':
        csms6s.set_bfs_order('mpi')
        log.info(f'BFS scan order set to MPI: {csms6s.BFS_ORDER}')

    if opts.wandb:
        run_name = os.path.basename(opts.checkpoint)
        wandb.init(project=opts.wandb_project, entity=opts.wandb_entity, name=run_name, config=vars(args))
        wandb.config.update({'checkpoint_dir': opts.checkpoint})
        log.info(f'Wandb initialized: {run_name}')

    log.info('Loading MPI-INF-3DHP dataset...')
    num_workers = min(8, os.cpu_count() or 4)
    trainloader_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': num_workers,
                          'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}
    testloader_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': num_workers,
                         'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}

    if not opts.evaluate:
        train_dataset = MPITrainDataset3D(args, 'train')
        train_loader = DataLoader(train_dataset, **trainloader_params)
    test_dataset = MPITestDataset3D(args, 'test')
    test_loader = DataLoader(test_dataset, **testloader_params)

    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = sum(p.numel() for p in model_backbone.parameters())
    log.info(f'INFO: Trainable parameter count:{model_params}')
    if opts.wandb and wandb.run is not None:
        wandb.config.update({'model_params': model_params}, allow_val_change=True)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone).cuda()

    # Checkpoint loading (mirrors train.py: finetune via -p/-ms, resume, or eval -e)
    checkpoint = None
    if getattr(args, 'finetune', False):
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
        log.info(f'Loading checkpoint {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda s, l: s, weights_only=False)
        missing, unexpected = model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
        if missing:
            log.info(f'Missing keys (ignored): {missing}')
        if unexpected:
            log.info(f'Unexpected keys (ignored): {unexpected}')
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint {chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda s, l: s, weights_only=False)
            missing, unexpected = model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
            if missing:
                log.info(f'Missing keys (ignored): {missing}')
            if unexpected:
                log.info(f'Unexpected keys (ignored): {unexpected}')
    model_pos = model_backbone

    if getattr(args, 'partial_train', None):
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        accum_steps = getattr(args, 'accum_steps', 1)
        warmup_epochs = getattr(args, 'warmup_epochs', 0)
        lr_scheduler = getattr(args, 'lr_scheduler', 'exponential')
        log.info(f'INFO: Training on {len(train_loader)}(3D) batches')
        log.info(f'Gradient accumulation steps: {accum_steps}, effective batch size: {args.batch_size * accum_steps}')
        use_ema = getattr(args, 'use_ema', False)
        ema = EMA(model_pos, decay=getattr(args, 'ema_decay', 0.999)) if use_ema else None
        if use_ema:
            log.info(f'EMA enabled, decay {getattr(args, "ema_decay", 0.999)}')
        if opts.resume and checkpoint is not None:
            st = checkpoint['epoch']
            if checkpoint.get('optimizer') is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            lr = checkpoint['lr']
            if checkpoint.get('min_loss') is not None:
                min_loss = checkpoint['min_loss']

        for epoch in range(st, args.epochs):
            log.info(f'Training epoch {epoch}.')
            start_time = time.time()
            losses = {k: AverageMeter() for k in ['3d_pos', '3d_scale', '3d_velocity', 'lv', 'lg', 'total']}

            if lr_scheduler == 'cosine':
                if epoch < warmup_epochs:
                    lr = args.learning_rate * (epoch + 1) / max(1, warmup_epochs)
                else:
                    progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
                    lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                lr = args.learning_rate * (lr_decay ** (epoch - st))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            train_epoch(args, model_pos, train_loader, losses, optimizer, accum_steps=accum_steps, ema=ema)
            elapsed = (time.time() - start_time) / 60

            if args.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (epoch + 1, elapsed, lr, losses['3d_pos'].avg))
                e_mpjpe = losses['3d_pos'].avg
            else:
                if ema is not None:
                    _bk = ema_swap_in(model_pos, ema)
                    e_mpjpe, e_pck, e_auc = evaluate_mpi(args, model_pos, test_loader, opts.checkpoint)
                    ema_swap_out(model_pos, _bk)
                else:
                    e_mpjpe, e_pck, e_auc = evaluate_mpi(args, model_pos, test_loader, opts.checkpoint)
                log.info('[%d] time %.2f lr %f 3d_train %f MPJPE %f PCK %f AUC %f' % (
                    epoch + 1, elapsed, lr, losses['3d_pos'].avg, e_mpjpe, e_pck, e_auc))
                if opts.wandb:
                    wandb.log({'epoch': epoch + 1, 'MPJPE(mm)': e_mpjpe, 'PCK@150': e_pck, 'AUC': e_auc,
                               'loss_3d_pos': losses['3d_pos'].avg, 'loss_total': losses['total'].avg, 'lr': lr},
                              step=epoch + 1)

            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin')
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(os.path.join(opts.checkpoint, f'epoch_{epoch}.bin'), epoch, lr, optimizer, model_pos, min_loss)
            if e_mpjpe < min_loss:
                min_loss = e_mpjpe
                if ema is not None:
                    _bk = ema_swap_in(model_pos, ema)
                    save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss, is_best=True)
                    ema_swap_out(model_pos, _bk)
                else:
                    save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss, is_best=True)

    if opts.evaluate:
        evaluate_mpi(args, model_pos, test_loader, opts.checkpoint)

    if opts.wandb:
        wandb.finish()


if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
