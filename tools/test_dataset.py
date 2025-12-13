#!/usr/bin/env python
"""
Quick dataset test to diagnose data loading issues
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config
from mmdet3d.datasets import build_dataset

# Import custom datasets to register them
from projects.mmdet3d_plugin.datasets import (
    CustomWaymoDataset,
    CustomWaymoDataset_T
)

def test_dataset():
    # Load config
    cfg = Config.fromfile('projects/configs/occformer_waymo/waymo_base.py')

    # Apply sample test mode
    cfg.data.train.load_interval = 100

    print("="*80)
    print("Building dataset...")
    print(f"Dataset type: {cfg.data.train.type}")
    print(f"Data root: {cfg.data.train.data_root}")
    print(f"Ann file: {cfg.data.train.ann_file}")
    print(f"Pose file: {cfg.data.train.pose_file}")
    print("="*80)

    # Build dataset
    try:
        dataset = build_dataset(cfg.data.train)
        print("Dataset created successfully")
        print(f"  Dataset length: {len(dataset)}")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("Testing first sample...")
    print("="*80)

    # Get data info first (without loading)
    print("Checking dataset info...")
    print(f"  Total samples: {len(dataset)}")
    if hasattr(dataset, 'data_infos'):
        print(f"  data_infos available: {len(dataset.data_infos) if dataset.data_infos else 0}")

    # Try to load first sample with detailed debugging
    try:
        print("\nAttempting to load sample 0...")
        print("  Calling dataset.__getitem__(0)...")

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Dataset loading timed out after 30 seconds")

        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        try:
            data = dataset[0]
            signal.alarm(0)  # Cancel alarm
            print("First sample loaded successfully")
            print(f"  Keys: {data.keys() if hasattr(data, 'keys') else 'N/A'}")
        except TimeoutError as e:
            signal.alarm(0)
            print(f"TIMEOUT: {e}")
            print("\nThis suggests the data loading is stuck in:")
            print("  - Image file reading")
            print("  - Point cloud processing")
            print("  - Annotation loading")
            print("\nCheck if files exist:")
            if hasattr(dataset, 'data_infos') and len(dataset.data_infos) > 0:
                info = dataset.data_infos[0]
                print(f"  Sample 0 info: {info}")
            return

    except Exception as e:
        print(f"Failed to load first sample: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("SUCCESS: Dataset is working!")
    print("="*80)

if __name__ == '__main__':
    test_dataset()
