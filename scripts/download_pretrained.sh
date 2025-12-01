#!/bin/bash

# Download Pretrained Weights for Waymo Experiments
# Run this script on a machine with internet access BEFORE uploading to HPC

echo "========================================="
echo "Downloading Pretrained Weights"
echo "========================================="
echo ""

# Create checkpoint directory
mkdir -p ckpts
cd ckpts

# Download EfficientNet-B7 (for baseline and most experiments)
echo "Downloading EfficientNet-B7..."
if [ ! -f "efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth" ]; then
    wget https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth
    echo "✓ EfficientNet-B7 downloaded"
else
    echo "✓ EfficientNet-B7 already exists"
fi
echo ""

# Download EfficientNet-B4 (for efficientnet_b4 experiment)
echo "Downloading EfficientNet-B4..."
if [ ! -f "efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth" ]; then
    wget https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth
    echo "✓ EfficientNet-B4 downloaded"
else
    echo "✓ EfficientNet-B4 already exists"
fi
echo ""

# Download ResNet-101
echo "Downloading ResNet-101..."
if [ ! -f "resnet101-5d3b4d8f.pth" ]; then
    echo "Running Python script to download ResNet-101..."
    cd ..
    python scripts/download_resnet.py
    cd ckpts
    if [ -f "resnet101-5d3b4d8f.pth" ]; then
        echo "✓ ResNet-101 downloaded"
    else
        echo "✗ ResNet-101 download failed"
        echo "  Please run manually: python scripts/download_resnet.py"
    fi
else
    echo "✓ ResNet-101 already exists"
fi
echo ""

# Check downloads
cd ..
echo "========================================="
echo "Download Summary"
echo "========================================="
ls -lh ckpts/
echo ""
echo "✓ Pretrained weights ready!"
echo "Upload the 'ckpts/' directory to HPC"
echo "========================================="
