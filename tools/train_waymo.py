"""Waymo training entry point with experiment presets."""

import argparse
import os
import sys
import copy
import torch

# Optional: reduce CUDA allocator fragmentation.
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20'

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmcv.runner import init_dist
from projects.configs.occformer_waymo.experiments import EXPERIMENTS, get_config
from projects.mmdet3d_plugin.datasets import (
    CustomWaymoDataset,
    CustomWaymoDataset_T
)

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

    # Model backbone (mmcv Config: update keys explicitly)
    if 'model_backbone' in exp_config:
        backbone_cfg = exp_config['model_backbone']
        for key, value in backbone_cfg.items():
            if key == 'init_cfg':
                # Handle nested dict
                for init_key, init_value in value.items():
                    cfg.model.img_backbone.init_cfg[init_key] = init_value
            else:
                cfg.model.img_backbone[key] = value

    # Model neck
    if 'model_neck_in_channels' in exp_config:
        # Update list by replacing elements
        cfg.model.img_neck['in_channels'] = exp_config['model_neck_in_channels']
    if 'model_neck_upsample_strides' in exp_config:
        cfg.model.img_neck['upsample_strides'] = exp_config['model_neck_upsample_strides']

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

    # Optimizer config (for gradient accumulation, etc.)
    if 'optimizer_config' in exp_config:
        cfg.optimizer_config = exp_config['optimizer_config']

    # Sample test mode
    if 'data_train_load_interval' in exp_config:
        cfg.data.train.load_interval = exp_config['data_train_load_interval']
    if 'data_val_load_interval' in exp_config:
        cfg.data.val.load_interval = exp_config['data_val_load_interval']
    if 'data_test_load_interval' in exp_config:
        cfg.data.test.load_interval = exp_config['data_test_load_interval']
    if 'evaluation_interval' in exp_config:
        cfg.evaluation.interval = exp_config['evaluation_interval']

    # Use Waymo-specific loader with resizing
    if exp_config.get('use_waymo_loader', False):
        target_size = exp_config.get('target_occ_size', [256, 256, 32])

        # Update train pipeline
        for i, step in enumerate(cfg.train_pipeline):
            if step['type'] == 'LoadSemKittiAnnotation':
                cfg.train_pipeline[i] = dict(
                    type='LoadWaymoOccAnnotation',
                    bda_aug_conf=step.get('bda_aug_conf', cfg.bda_aug_conf),
                    is_train=step.get('is_train', True),
                    point_cloud_range=step.get('point_cloud_range', cfg.point_cloud_range),
                    target_occ_size=target_size,
                    resize_method='nearest'
                )

        # Update test pipeline
        for i, step in enumerate(cfg.test_pipeline):
            if step['type'] == 'LoadSemKittiAnnotation':
                cfg.test_pipeline[i] = dict(
                    type='LoadWaymoOccAnnotation',
                    bda_aug_conf=step.get('bda_aug_conf', cfg.bda_aug_conf),
                    is_train=step.get('is_train', False),
                    point_cloud_range=step.get('point_cloud_range', cfg.point_cloud_range),
                    target_occ_size=target_size,
                    resize_method='nearest'
                )

    # Override occ_size if specified (for small grid experiments)
    if 'occ_size' in exp_config:
        cfg.occ_size = exp_config['occ_size']
        cfg.data.train.occ_size = exp_config['occ_size']
        cfg.data.val.occ_size = exp_config['occ_size']
        cfg.data.test.occ_size = exp_config['occ_size']

    # Use focal loss for class imbalance
    if exp_config.get('use_focal_loss', False):
        # Build loss configuration - Focal Loss handles class balancing internally
        cfg.model.pts_bbox_head.loss_cls = dict(
            type='OccupancyFocalLoss',
            gamma=2.5,
            reduction='mean',
            loss_weight=2.0,
            ignore_index=255,
            num_classes=len(cfg.class_names)
        )
    # Use weighted CrossEntropy for class imbalance (more memory efficient than focal loss)
    elif exp_config.get('use_class_weights', False):
        class_weights = exp_config.get('class_weights', None)
        if class_weights is not None:
            cfg.model.pts_bbox_head.loss_cls = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=class_weights
            )

    return cfg


def main():
    args = parse_args()

    # Clear CUDA cache before starting (helps when jobs reuse GPUs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Deterministic cudnn can reduce variability in memory use
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load base config
    cfg = Config.fromfile(args.config)

    # Auto-detect available GPUs and override config
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        cfg.gpu_ids = list(range(num_gpus))
        print(f"Detected {num_gpus} GPU(s): {cfg.gpu_ids}")
    else:
        cfg.gpu_ids = []
        print("No GPUs detected, using CPU")

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

    # Debug: Log model backbone configuration
    logger.info('Model backbone configuration:')
    logger.info(f'  Type: {cfg.model.img_backbone.type}')
    logger.info(f'  Arch: {cfg.model.img_backbone.arch}')
    logger.info(f'  Checkpoint: {cfg.model.img_backbone.init_cfg.checkpoint}')
    logger.info('Model neck configuration:')
    logger.info(f'  In channels: {cfg.model.img_neck.in_channels}')

    # Debug: Log pipeline configuration
    logger.info('Train pipeline:')
    for i, step in enumerate(cfg.train_pipeline):
        logger.info(f'  [{i}] {step.get("type", "unknown")}')
    logger.info('Test pipeline:')
    for i, step in enumerate(cfg.test_pipeline):
        logger.info(f'  [{i}] {step.get("type", "unknown")}')

    # Log config (skip pretty_text to avoid yapf issues)
    logger.info(f'Config file: {args.config}')
    logger.info(f'Experiment config applied: {exp_config}')

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
    # Disable validation if evaluation interval is non-positive
    validate_flag = True
    if cfg.get('evaluation', None):
        interval = cfg.evaluation.get('interval', 1)
        if interval <= 0:
            validate_flag = False

    train_model(
        model,
        datasets,
        cfg,
        distributed=(args.launcher != 'none'),
        validate=validate_flag,
        timestamp=timestamp,
        meta=dict(
            exp_name=args.exp_name,
            config_file=args.config,
        )
    )


if __name__ == '__main__':
    main()
