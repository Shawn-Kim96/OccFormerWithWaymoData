#!/usr/bin/env python
"""
Download ResNet-101 pretrained weights for offline use on HPC
Run this on a machine with internet access
"""

import torch
import torchvision
import os

print("=" * 60)
print("Downloading ResNet-101 Pretrained Weights")
print("=" * 60)
print("")

# Create ckpts directory
os.makedirs('ckpts', exist_ok=True)

# Download ResNet-101
print("Downloading ResNet-101 from torchvision...")
model = torchvision.models.resnet101(pretrained=True)

# Save state dict
output_path = 'ckpts/resnet101-5d3b4d8f.pth'
torch.save(model.state_dict(), output_path)

# Check file
file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
print(f"✓ ResNet-101 downloaded: {output_path}")
print(f"  Size: {file_size:.1f} MB")
print("")
print("=" * 60)
print("Upload this file to HPC:")
print(f"  scp {output_path} user@hpc:/path/to/OccFormerWithWaymoData/ckpts/")
print("=" * 60)
