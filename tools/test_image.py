#!/usr/bin/env python
"""
Test image loading speed
"""
import time
import os
import glob

print("Testing image loading speed...")

# Find first 5 images
img_dir = 'data/waymo_v1-3-1/kitti_format/training/image_0'
images = sorted(glob.glob(os.path.join(img_dir, '*.png')))[:5]

if not images:
    print(f"No images found in {img_dir}")
    exit(1)

print(f"Found {len(images)} test images")

# Test 1: File size
print("\n" + "="*80)
print("TEST 1: File sizes")
for img in images:
    size_mb = os.path.getsize(img) / (1024*1024)
    print(f"  {os.path.basename(img)}: {size_mb:.2f} MB")

# Test 2: Reading with PIL
print("\n" + "="*80)
print("TEST 2: Loading with PIL (Image.open)")
from PIL import Image

for img in images:
    start = time.time()
    im = Image.open(img)
    elapsed = time.time() - start
    print(f"  {os.path.basename(img)}: {elapsed:.3f}s ({im.size})")

# Test 3: Reading with OpenCV
print("\n" + "="*80)
print("TEST 3: Loading with OpenCV (cv2.imread)")
import cv2

for img in images:
    start = time.time()
    im = cv2.imread(img)
    elapsed = time.time() - start
    print(f"  {os.path.basename(img)}: {elapsed:.3f}s ({im.shape})")

# Test 4: Reading with mmcv
print("\n" + "="*80)
print("TEST 4: Loading with mmcv.imread")
import mmcv

for img in images:
    start = time.time()
    im = mmcv.imread(img)
    elapsed = time.time() - start
    print(f"  {os.path.basename(img)}: {elapsed:.3f}s ({im.shape})")

print("\n" + "="*80)
print("Analysis:")
print("  - If > 1 second per image: File system is slow (NFS issue)")
print("  - If > 10 MB per image: Images are too large")
print("  - Normal: < 0.1 second per image")
