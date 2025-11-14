#!/usr/bin/env python
"""
Waymo Training Script with Experiment Management
Supports:
- Multiple experiment configurations
- Sample testing mode
- Auto-resume from checkpoints
- Organized output structure
"""

import argparse
import os
import sys
import copy
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmcv.runner import init_dist
from projects.configs.occformer_waymo.experiments import EXPERIMENTS, get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Waymo OccFormer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--exp-name', required=True,
                       help=f'experiment name: {list(EXPERIMENTS.keys())}')
    parser.add_argument('--sample-test', action='store_true',
                       help='use only ~10 samples for testing')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'],
                       default='none', help='job launcher')
    parser.add_argument('--deterministic', action='store_true',
                       help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def apply_experiment_config(cfg, exp_config):
    """Apply experiment-specific configuration overrides."""

    # Learning rate
    if 'lr' in exp_config:
        cfg.optimizer.lr = exp_config['lr']

    # LR schedule
    if 'lr_config' in exp_config:
        cfg.lr_config = exp_config['lr_config']

    # Data augmentation
    if 'bda_aug_conf' in exp_config:
        cfg.bda_aug_conf = exp_config['bda_aug_conf']

    # Data config
    if 'data_config' in exp_config:
        cfg.data_config = exp_config['data_config']

    # Model backbone
    if 'model_backbone' in exp_config:
        cfg.model.img_backbone = exp_config['model_backbone']

    # Model neck
    if 'model_neck_in_channels' in exp_config:
        cfg.model.img_neck.in_channels = exp_config['model_neck_in_channels']
    if 'model_neck_upsample_strides' in exp_config:
        cfg.model.img_neck.upsample_strides = exp_config['model_neck_upsample_strides']

    # Optimizer
    if 'optimizer' in exp_config:
        cfg.optimizer = exp_config['optimizer']

    # Runner
    if 'runner' in exp_config:
        cfg.runner = exp_config['runner']

    # Mask2Former queries
    if 'mask2former_num_queries' in exp_config:
        cfg.mask2former_num_queries = exp_config['mask2former_num_queries']
        cfg.model.pts_bbox_head.num_queries = exp_config['mask2former_num_queries']

    # Batch size
    if 'samples_per_gpu' in exp_config:
        cfg.data.samples_per_gpu = exp_config['samples_per_gpu']

    # Sample test mode
    if 'data_train_load_interval' in exp_config:
        cfg.data.train.load_interval = exp_config['data_train_load_interval']
    if 'evaluation_interval' in exp_config:
        cfg.evaluation.interval = exp_config['evaluation_interval']

    return cfg


def main():
    args = parse_args()

    # Load base config
    cfg = Config.fromfile(args.config)

    # Get experiment configuration
    exp_config, description = get_config(args.exp_name, args.sample_test)

    # Apply experiment config
    cfg = apply_experiment_config(cfg, exp_config)

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = f'results/{args.exp_name}/model'

    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Resume from checkpoint
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    elif os.path.exists(os.path.join(cfg.work_dir, 'latest.pth')):
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')

    # Initialize distributed training
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)

    # Dump config (avoid yapf formatting issues)
    import shutil
    shutil.copy(args.config, os.path.join(cfg.work_dir, os.path.basename(args.config)))

    # Set random seed
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup logger
    timestamp = None
    if args.resume_from is not None:
        # Keep original timestamp when resuming
        import re
        match = re.search(r'\d{8}_\d{6}', args.resume_from)
        if match:
            timestamp = match.group()

    logger = get_root_logger(
        log_file=os.path.join(cfg.work_dir, f'{args.exp_name}.log'),
        log_level=cfg.log_level
    )

    # Log experiment info
    logger.info('=' * 80)
    logger.info(f'Experiment: {args.exp_name}')
    logger.info(f'Description: {description}')
    if args.sample_test:
        logger.info('Mode: SAMPLE TEST (~10 samples)')
    else:
        logger.info('Mode: FULL TRAINING')
    logger.info(f'Work directory: {cfg.work_dir}')
    if cfg.get('resume_from'):
        logger.info(f'Resume from: {cfg.resume_from}')
    logger.info('=' * 80)

    # Log config
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    # Build model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()

    # Train model
    train_model(
        model,
        datasets,
        cfg,
        distributed=(args.launcher != 'none'),
        validate=True,
        timestamp=timestamp,
        meta=dict(
            exp_name=args.exp_name,
            config=cfg.pretty_text,
        )
    )


if __name__ == '__main__':
    main()
