# Baseline Configuration for Waymo Dataset
# This serves as the baseline for comparing different configurations
# Learning rate: 1e-4, Backbone: EfficientNet-B7, Standard augmentation

_base_ = './occformer_waymo.py'

# Explicitly define baseline settings
lr = 1e-4
max_epochs = 30
backbone_type = 'efficientnet-b7'

# Standard data augmentation
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,
)

optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# Log settings for experiment tracking
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# Evaluation
evaluation = dict(
    interval=1,
    save_best='waymo_SSC_mIoU',
    rule='greater',
)
