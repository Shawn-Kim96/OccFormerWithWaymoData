# Experiment: Different Backbone Architecture (ResNet-101)
# Compare with baseline (EfficientNet-B7) to analyze architecture impact
# Expected: Different feature representations, different performance characteristics

_base_ = './occformer_waymo_baseline.py'

# Model configuration with ResNet-101
model = dict(
    type='OccupancyFormer',
    img_backbone=dict(
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
            checkpoint='torchvision://resnet101'
        ),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],  # ResNet-101 channel dims
        upsample_strides=[0.5, 1, 2, 4],
        out_channels=[128, 128, 128, 128]
    ),
)

# ResNet might need different learning rate
optimizer = dict(
    type='AdamW',
    lr=5e-5,  # Lower LR for ResNet
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
)
