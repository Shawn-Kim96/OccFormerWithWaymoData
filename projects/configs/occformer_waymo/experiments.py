# All Waymo Experiment Configurations
# Import this file and access configs by key

from .waymo_base import *

# Define all experiment variations
EXPERIMENTS = {
    'test': {
        'lr': 1e-4,
        'description': 'test'
    },
    
    # Baseline
    'baseline': {
        'lr': 1e-4,
        'description': 'Baseline configuration',
    },

    # Faster baseline (smaller data + shorter schedule)
    'baseline_fast': {
        'lr': 1e-4,
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
        'description': 'Baseline with 1/10 data and 30 epochs (fast check)',
    },

    # Learning rate variations
    'lr_5e5': {
        'lr': 5e-5,
        'lr_config': dict(policy='step', step=[25, 28]),
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
        'description': 'Lower learning rate (5e-5)',
    },

    'lr_5e4': {
        'lr': 5e-4,
        'lr_config': dict(policy='step', step=[15, 22]),
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
        'description': 'Higher learning rate (5e-4)',
    },

    # Augmentation variations
    'strong_aug': {
        'bda_aug_conf': dict(
            rot_lim=(-10.0, 10.0),
            scale_lim=(0.85, 1.15),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0.5,
        ),
        'data_config': {
            'input_size': (640, 960),
            'resize': (-0.15, 0.15),
            'rot': (-8.0, 8.0),
            'flip': True,
            'crop_h': (0.0, 0.0),
            'resize_test': 0.00,
        },
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'evaluation_interval': 0,        # disable val during training; eval later manually
        'runner': dict(type='EpochBasedRunner', max_epochs=35),
        'description': 'Strong data augmentation',
    },

    'weak_aug': {
        'bda_aug_conf': dict(
            rot_lim=(0, 0),
            scale_lim=(0.98, 1.02),
            flip_dx_ratio=0.3,
            flip_dy_ratio=0.3,
            flip_dz_ratio=0.3,
        ),
        'data_config': {
            'input_size': (640, 960),
            'resize': (-0.03, 0.05),
            'rot': (-2.0, 2.0),
            'flip': True,
            'crop_h': (0.0, 0.0),
            'resize_test': 0.00,
        },
        'description': 'Weak data augmentation',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    # Model architecture variations
    'efficientnet_b4': {
        'model_backbone': dict(
            type='CustomEfficientNet',
            arch='b4',
            drop_path_rate=0.2,
            frozen_stages=0,
            norm_eval=False,
            out_indices=(2, 3, 4, 5, 6),
            with_cp=True,
            init_cfg=dict(type='Pretrained', prefix='backbone',
                         checkpoint='ckpts/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth'),
        ),
        'model_neck_in_channels': [32, 56, 160, 448, 1792],
        'samples_per_gpu': 2,
        'description': 'Smaller backbone (EfficientNet-B4)',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    'resnet101': {
        'model_backbone': dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            with_cp=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='ckpts/resnet101-5d3b4d8f.pth'  # Pre-downloaded for HPC
            ),
        ),
        'model_neck_in_channels': [256, 512, 1024, 2048],
        'model_neck_upsample_strides': [0.5, 1, 2, 4],
        'lr': 5e-5,
        'description': 'ResNet-101 backbone',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    # Training strategy variations
    'cosine_lr': {
        'lr_config': dict(
            policy='CosineAnnealing',
            min_lr=1e-6,
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.1,
        ),
        'runner': dict(type='EpochBasedRunner', max_epochs=40),
        'description': 'Cosine learning rate schedule',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    'sgd': {
        'optimizer': dict(
            type='SGD',
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001,
        ),
        'lr_config': dict(
            policy='CosineAnnealing',
            min_lr=1e-5,
            warmup='linear',
            warmup_iters=1000,
            warmup_ratio=0.001,
        ),
        'description': 'SGD optimizer',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    # Efficiency variation
    'reduced_queries': {
        'mask2former_num_queries': 50,
        'samples_per_gpu': 2,
        'description': 'Reduced queries for efficiency',
        'data_train_load_interval': 10,  # use 1/10 train samples
        'data_val_load_interval': 10,    # shrink val/test similarly
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,        # disable val during training; eval later manually
    },

    # ===== IMPROVED EXPERIMENTS WITH FIXES =====

    'improved_baseline': {
        'lr': 1e-4,
        'description': 'Fixed baseline: resize GT to model size + focal loss',
        'data_train_load_interval': 10,  # Keep 10% for memory constraints
        'data_val_load_interval': 10,
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=40),
        'evaluation_interval': 5,  # Evaluate every 5 epochs
        # Use Waymo-specific loading with resizing
        'use_waymo_loader': True,
        'target_occ_size': [256, 256, 32],  # Resize GT to match model
        # Use focal loss for class imbalance
        'use_focal_loss': True,
    },

    'improved_fast': {
        'lr': 1e-4,
        'description': 'Fast improved version with all fixes',
        'data_train_load_interval': 10,
        'data_val_load_interval': 10,
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=30),
        'evaluation_interval': 0,  # Disable during training, eval manually
        'use_waymo_loader': True,
        'target_occ_size': [256, 256, 32],
        'use_focal_loss': True,
    },

    'improved_small_grid': {
        'lr': 1e-4,
        'description': 'Use native GT size (200,200,16) - more memory efficient',
        'data_train_load_interval': 10,
        'data_val_load_interval': 10,
        'data_test_load_interval': 10,
        'runner': dict(type='EpochBasedRunner', max_epochs=35),
        'evaluation_interval': 5,
        # Keep GT size, change model to match
        'use_waymo_loader': True,
        'target_occ_size': None,  # Don't resize, use (200,200,16)
        'occ_size': [200, 200, 16],  # Override model config
        'use_focal_loss': True,
    },
}


def get_config(exp_name, sample_test=False):
    """
    Get configuration for a specific experiment.

    Args:
        exp_name: Name of the experiment (key from EXPERIMENTS dict)
        sample_test: If True, use only 10 samples for testing

    Returns:
        config dict with all settings
    """
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")

    # Start with base config
    config = {}

    # Get experiment-specific overrides
    exp_config = EXPERIMENTS[exp_name].copy()
    description = exp_config.pop('description', '')

    # Apply overrides
    for key, value in exp_config.items():
        config[key] = value

    # Sample test mode
    if sample_test:
        # Subsample aggressively for quick sanity checks
        config['data_train_load_interval'] = 10000  # train ~1/100
        config['data_val_load_interval'] = 800    # val ≈10 samples (8069/800 ≈ 10)
        config['data_test_load_interval'] = 800   # test ≈10 samples
        config['runner'] = dict(type='EpochBasedRunner', max_epochs=2)
        # Disable eval/save-best for quick smoke test (outputs are dummy)
        config['evaluation_interval'] = 1

    return config, description
