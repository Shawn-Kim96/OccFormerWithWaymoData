# Environment Setting (P100 / CUDA 11.3)

This document captures the environment fixes applied to build and run OccFormer on Tesla P100 (compute capability 6.0) using a locally extracted CUDA 11.3 toolkit.

## CUDA layout (local extraction)
- nvcc: `~/cuda113_extracted/builds/cuda_nvcc/bin/nvcc`
- cudart: `~/cuda113_extracted/builds/cuda_cudart/{include,lib64}`
- cublas: `~/cuda113_extracted/builds/libcublas/{include,lib64}`
- cusparse: `~/cuda113_extracted/builds/libcusparse/{include,lib64}`
- cupti: `~/cuda113_extracted/builds/cuda_cupti/lib64`

## Conda GCC (to avoid nvcc GCC>10 check)
Install gcc/g++ 9 via conda and point CC/CXX to it:
```bash
conda install -y -c conda-forge gxx_linux-64=9.5.0
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CPATH=/usr/include:$CPATH  # for crypt.h and other system headers
```

## nvcc wrapper (bypass unsupported compiler check)
```bash
mkdir -p ~/nvcc_wrap
cat > ~/nvcc_wrap/nvcc <<'EOF'
#!/bin/bash
exec /home/018219422/cuda113_extracted/builds/cuda_nvcc/bin/nvcc -allow-unsupported-compiler "$@"
EOF
chmod +x ~/nvcc_wrap/nvcc
```

## Exported environment (per shell)
```bash
export PATH=~/nvcc_wrap:$PATH
export CUDA_HOME=$HOME/cuda113_extracted/builds/cuda_nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/cuda113_extracted/builds/cuda_cudart/lib64:$HOME/cuda113_extracted/builds/cuda_cupti/lib64:$HOME/cuda113_extracted/builds/libcublas/lib64:$HOME/cuda113_extracted/builds/libcusparse/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0+PTX"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
export NVCC_FLAGS="-allow-unsupported-compiler"
```

## mmdet3d build steps (clean + install)
```bash
rm -rf ~/.cache/torch_extensions/*bev_pool* ~/.cache/torch_extensions/*spconv* ~/.cache/torch_extensions/*bevpool_cuda*
cd ~/OccFormerWithWaymoData/mmdetection3d
rm -rf build
python setup.py clean
pip install -e . --no-cache-dir --no-build-isolation
```
Verify nvcc logs show `-gencode=arch=compute_60,code=sm_60` and ccbin points to the conda gcc9.

## Runtime notes
- Keep the above exports active when running train/eval.
- IoU tables printed by evaluation are shown in percentage format (e.g., 92.02 => IoU=0.9202).
- If different nodes have different drivers/compilers, create separate conda envs per node and rebuild `mmdetection3d` on each node.***
