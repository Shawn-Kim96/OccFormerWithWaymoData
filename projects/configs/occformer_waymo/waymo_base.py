# Base configuration for all Waymo experiments
# This contains all shared settings

_base_ = [
    '../_base_/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

sync_bn = True
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# GPU and seed settings
gpu_ids = [0, 1, 2, 3]  # Use 4 GPUs (list instead of range for config)
seed = 0

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Camera configuration
camera_used = [0, 1, 2, 3, 4]  # all 5 cameras
num_views = 5

# Class names
class_names = [
    'unlabeled', 'car', 'truck', 'bus', 'other-vehicle', 'pedestrian',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
num_class = len(class_names)

# Point cloud range
point_cloud_range = [0, -40, -3, 70.4, 40, 4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

# Data configuration
data_config = {
    'input_size': (640, 960),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    'cams': ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'],
    'Ncams': 5,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 70.0, 0.5],
}

# 3D encoder settings
numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

# Mask2Former settings
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel // 3  # Integer division
mask2former_num_heads = voxel_out_channels // 32

# Model architecture
model = dict(
    type='OccupancyFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'
        ),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.25, 0.5, 1, 2, 2],
        out_channels=[128, 128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        numC_input=640,
        cam_channels=33,
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    img_bev_encoder_backbone=dict(
        type='OccupancyEncoder',
        num_stage=len(voxel_num_layer),
        in_channels=numC_Trans,
        block_numbers=voxel_num_layer,
        block_inplanes=voxel_channels,
        block_strides=voxel_strides,
        out_indices=voxel_out_indices,
        with_cp=True,
        norm_cfg=norm_cfg,
    ),
    img_bev_encoder_neck=dict(
        type='MSDeformAttnPixelDecoder3D',
        strides=[2, 4, 8, 16],
        in_channels=voxel_channels,
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        norm_cfg=norm_cfg,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=voxel_out_channels,
                    num_heads=6,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(embed_dims=voxel_out_channels),
                feedforward_channels=voxel_out_channels * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=voxel_out_channels // 3,
            normalize=True),
    ),
    pts_bbox_head=dict(
        type='Mask2FormerOccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        num_queries=mask2former_num_queries,
        num_occupancy_classes=num_class,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        point_cloud_range=point_cloud_range,
    ),
    train_cfg=dict(
        pts=dict(
            num_points=12544 * 4,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'),
        )),
    test_cfg=dict(
        pts=dict(
            semantic_on=True,
            panoptic_on=False,
            instance_on=False)),
)

# Dataset paths
dataset_type = 'CustomWaymoDataset_T'
data_root = 'data/waymo_v1-3-1/kitti_format'
ann_file_train = 'data/waymo_v1-3-1/occ3d_waymo/waymo_infos_train.filtered.pkl'
ann_file_val = 'data/waymo_v1-3-1/occ3d_waymo/waymo_infos_val.filtered.pkl'
pose_file = 'data/waymo_v1-3-1/occ3d_waymo/cam_infos.pkl'
pose_file_val = 'data/waymo_v1-3-1/occ3d_waymo/cam_infos_vali.pkl'

# Data augmentation
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,
)

# Pipelines
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True,
         data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),  # Use 'kitti' for Waymo KITTI format
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
         is_train=True, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False,
         data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
         is_train=False, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=True),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Data config
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # Set to 0 to avoid multiprocessing issues
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        split='training',
        pose_file=pose_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        camera_used=camera_used,
        num_views=num_views,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        history_len=1,
        load_interval=1,
        withimage=True,
        input_sample_policy=dict(type='normal'),  # Normal sampling policy
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        split='validation',
        pose_file=pose_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        camera_used=camera_used,
        num_views=num_views,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        input_sample_policy=dict(type='normal'),  # Normal sampling policy
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        split='validation',
        pose_file=pose_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        camera_used=camera_used,
        num_views=num_views,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        input_sample_policy=dict(type='normal'),  # Normal sampling policy
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# Optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        },
        norm_decay_mult=0.0))

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# Learning rate
lr_config = dict(
    policy='step',
    step=[20, 25],
)

# Checkpointing
checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# Training
runner = dict(type='EpochBasedRunner', max_epochs=30)

# Evaluation
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='waymo_SSC_mIoU',
    rule='greater',
)

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
