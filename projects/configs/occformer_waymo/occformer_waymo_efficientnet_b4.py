# Experiment: Smaller Backbone (EfficientNet-B4)
# Compare with baseline (B7) to analyze model size vs performance tradeoff
# Expected: Faster training, lower memory, potentially lower accuracy

_base_ = './occformer_waymo_baseline.py'

# Model configuration with EfficientNet-B4
model = dict(
    type='OccupancyFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b4',  # Changed from b7
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='ckpts/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth'
        ),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[32, 56, 160, 448, 1792],  # B4 channel dims
        upsample_strides=[0.25, 0.5, 1, 2, 2],
        out_channels=[128, 128, 128, 128, 128]
    ),
)

# Can increase batch size due to smaller model
data = dict(
    samples_per_gpu=2,  # Increased from 1
    workers_per_gpu=4,
)
