#!/usr/bin/env python
"""
Minimal test to find bottleneck
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config

# Import custom datasets
from projects.mmdet3d_plugin.datasets import CustomWaymoDataset_T

print("="*80)
print("STEP 1: Loading config...")
start = time.time()
cfg = Config.fromfile('projects/configs/occformer_waymo/waymo_base.py')
cfg.data.train.load_interval = 100
print(f"✓ Config loaded in {time.time()-start:.2f}s")

print("\n" + "="*80)
print("STEP 2: Creating dataset (NO pipeline)...")
start = time.time()

# Create dataset without pipeline to isolate the issue
dataset_cfg = cfg.data.train.copy()
dataset_cfg['pipeline'] = []  # Remove pipeline

from mmdet3d.datasets import build_dataset
try:
    dataset = build_dataset(dataset_cfg)
    print(f"✓ Dataset created in {time.time()-start:.2f}s")
    print(f"  Length: {len(dataset)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("STEP 3: Testing with minimal pipeline...")
start = time.time()

# Test with only image loading
dataset_cfg['pipeline'] = [
    dict(type='LoadMultiViewImageFromFiles_OccFormer', is_train=True,
         data_config=cfg.data_config, img_norm_cfg=cfg.img_norm_cfg),
]

try:
    dataset = build_dataset(dataset_cfg)
    print(f"✓ Dataset with LoadImage created in {time.time()-start:.2f}s")

    print("\nSTEP 4: Loading first sample (image only)...")
    start = time.time()
    data = dataset[0]
    print(f"✓ First sample loaded in {time.time()-start:.2f}s")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("STEP 5: Testing with full pipeline...")
start = time.time()

dataset_cfg['pipeline'] = cfg.train_pipeline
try:
    dataset = build_dataset(dataset_cfg)
    print(f"✓ Dataset with full pipeline created in {time.time()-start:.2f}s")

    print("\nLoading first sample (full pipeline)...")
    start = time.time()
    data = dataset[0]
    print(f"✓ First sample loaded in {time.time()-start:.2f}s")

    print("\n" + "="*80)
    print("SUCCESS! All tests passed.")
    print("="*80)

except Exception as e:
    print(f"✗ Failed at full pipeline: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  The bottleneck is in the full pipeline!")
    print("   Likely culprit: CreateDepthFromLiDAR or LoadSemKittiAnnotation")
