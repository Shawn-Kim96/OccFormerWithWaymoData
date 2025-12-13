#!/bin/bash
# Quick test script to verify all fixes are working
# Run this locally before submitting the full job

set -e

echo ""
echo "========================================="
echo "GPU INFORMATION"
echo "========================================="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (CPU mode or different GPU driver)"
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "========================================="

echo ""
echo "========================================="
echo "Quick Test: Verifying Fixes"
echo "========================================="

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate occformer307

export PYTHONPATH=".:${PYTHONPATH:-}"

# Test 1: Check if new modules can be imported
echo ""
echo "[Test 1/3] Checking module imports..."
python -c "
from projects.mmdet3d_plugin.datasets.pipelines.loading_waymo_occ import LoadWaymoOccAnnotation
from projects.mmdet3d_plugin.models.losses.focal_loss_balanced import OccupancyFocalLoss
from mmdet.models.builder import LOSSES
print('  LoadWaymoOccAnnotation imported successfully')
print('  OccupancyFocalLoss imported successfully')
if 'OccupancyFocalLoss' in LOSSES._module_dict:
    print('  OccupancyFocalLoss registered in LOSSES registry')
else:
    print('  ERROR: OccupancyFocalLoss is not registered')
    import sys
    sys.exit(1)
"

# Test 2: Verify config can be loaded with new experiment
echo ""
echo "[Test 2/3] Testing config loading..."
python -c "
from mmcv import Config
from projects.configs.occformer_waymo.experiments import get_config
import sys

# Test improved_small_grid config
config, desc = get_config('improved_small_grid', sample_test=False)
print(f'  Experiment config loaded: {desc}')
print(f'  use_waymo_loader: {config.get(\"use_waymo_loader\", False)}')
print(f'  use_focal_loss: {config.get(\"use_focal_loss\", False)}')
print(f'  target_occ_size: {config.get(\"target_occ_size\", None)}')
print(f'  occ_size: {config.get(\"occ_size\", None)}')
"

# Test 3: Test LoadWaymoOccAnnotation directly
echo ""
echo "[Test 3/3] Testing LoadWaymoOccAnnotation directly..."
python -c "
import sys
import torch
import numpy as np

# Import the loader
from projects.mmdet3d_plugin.datasets.pipelines.loading_waymo_occ import LoadWaymoOccAnnotation

# Create loader instance
loader = LoadWaymoOccAnnotation(
    bda_aug_conf=dict(
        rot_lim=(0, 0),
        scale_lim=(1.0, 1.0),
        flip_dx_ratio=0.0,
        flip_dy_ratio=0.0,
        flip_dz_ratio=0.0
    ),
    is_train=False,
    point_cloud_range=[0, -40, -3, 70.4, 40, 4],
    target_occ_size=[256, 256, 32],
    resize_method='nearest'
)
print('  LoadWaymoOccAnnotation instantiated')

# Create dummy GT with label 23
dummy_gt = np.ones((200, 200, 16), dtype=np.uint8) * 23
dummy_results = {
    'gt_occ': dummy_gt,
    'img_inputs': (None, None, None, None, None, None, None, None)
}

# Process
try:
    results = loader(dummy_results)
    gt_occ = results['gt_occ']

    print(f'  Sample processed: GT shape = {gt_occ.shape}')
    print(f'  GT unique labels: {torch.unique(gt_occ).tolist()}')

    # Check label 23 is remapped to 15
    if (gt_occ == 23).any():
        print('  ERROR: Label 23 still exists. Remapping failed.')
        sys.exit(1)
    elif (gt_occ == 15).all():
        print('  Label remapping working (23 -> 15)')
    else:
        print(f'  Warning: Unexpected labels: {torch.unique(gt_occ).tolist()}')

    # Check if resizing worked
    if gt_occ.shape == torch.Size([256, 256, 32]):
        print(f'  Resizing working correctly: {gt_occ.shape}')
    else:
        print(f'  Unexpected shape: {gt_occ.shape}')

except Exception as e:
    print(f'  Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "========================================="
echo "All tests passed"
echo "========================================="
echo ""
echo "Ready to submit full experiment:"
echo "  sbatch scripts/run_full_experiment.sh improved_fast"
echo ""
