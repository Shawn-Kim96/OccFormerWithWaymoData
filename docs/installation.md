# Installation Guide (P100 / CUDA 11.3)

This guide extends the original instructions with the fixes required for building CUDA ops on Tesla P100 (compute capability 6.0) using a local CUDA 11.3 extraction.

## 0. Prerequisites
- GPU: Tesla P100 (sm_60)
- Local CUDA 11.3 extracted at `~/cuda113_extracted/builds/*`
  - nvcc: `~/cuda113_extracted/builds/cuda_nvcc/bin/nvcc`
  - cudart: `~/cuda113_extracted/builds/cuda_cudart/{include,lib64}`
  - cublas: `~/cuda113_extracted/builds/libcublas/{include,lib64}`
  - cusparse: `~/cuda113_extracted/builds/libcusparse/{include,lib64}`
  - cupti: `~/cuda113_extracted/builds/cuda_cupti/lib64`

## 1. Conda environment
```bash
conda create -n occformer python=3.7 -y
conda activate occformer
```

## 2. PyTorch
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## 3. mmcv / mmdet / mmseg
```bash
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

## 4. CUDA build env (P100)
```bash
# nvcc wrapper to bypass compiler check
mkdir -p ~/nvcc_wrap
cat > ~/nvcc_wrap/nvcc <<'EOF'
#!/bin/bash
exec /home/018219422/cuda113_extracted/builds/cuda_nvcc/bin/nvcc -allow-unsupported-compiler "$@"
EOF
chmod +x ~/nvcc_wrap/nvcc

# CUDA & arch
export PATH=~/nvcc_wrap:$PATH
export CUDA_HOME=$HOME/cuda113_extracted/builds/cuda_nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/cuda113_extracted/builds/cuda_cudart/lib64:$HOME/cuda113_extracted/builds/cuda_cupti/lib64:$HOME/cuda113_extracted/builds/libcublas/lib64:$HOME/cuda113_extracted/builds/libcusparse/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0+PTX"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
export NVCC_FLAGS="-allow-unsupported-compiler"

# Compiler (install gcc/g++ 9 via conda to avoid nvcc GCC>10 block)
conda install -y -c conda-forge gxx_linux-64=9.5.0
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CPATH=/usr/include:$CPATH  # for crypt.h and other system headers
```

## 5. Build mmdet3d (sm_60 kernels)
```bash
# Clean caches
rm -rf ~/.cache/torch_extensions/*bev_pool* ~/.cache/torch_extensions/*spconv* ~/.cache/torch_extensions/*bevpool_cuda*
cd ~/OccFormerWithWaymoData/mmdetection3d
rm -rf build
python setup.py clean

# Install
pip install -e . --no-cache-dir --no-build-isolation
```
Ensure the nvcc log shows `-gencode=arch=compute_60,code=sm_60` and ccbin pointing to the conda gcc9.

## 6. Other deps
```bash
pip install -r docs/requirements.txt
```

## 7. Run (same shell, env exports still valid)
Train:
```bash
python tools/train_waymo.py projects/configs/occformer_waymo/waymo_base.py --exp-name baseline_fast
```
Eval:
```bash
PYTHONPATH=.:$PYTHONPATH python tools/test.py projects/configs/occformer_waymo/waymo_base.py \
  results/baseline_fast/model/latest.pth --eval mIoU \
  --cfg-options data.val.load_interval=10 data.test.load_interval=10 \
  > results/baseline_fast/logs/evaluate.log 2>&1
```
Note: IoU table in logs is printed as percentages (e.g., 92.02 => IoU=0.9202).

## 8. Node isolation (recommended)
If different nodes have different driver/compiler setups, create separate conda envs per node and rebuild `mmdetection3d` on each node to ensure CUDA ops match the local GPU/driver.***
