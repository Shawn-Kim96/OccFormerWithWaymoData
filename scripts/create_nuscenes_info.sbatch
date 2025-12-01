#!/bin/bash
#SBATCH -J nusc_infos
#SBATCH -p gpuqs
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/nusc_infos.%j.out
#SBATCH -e logs/nusc_infos.%j.err

echo "Allocated GPUs:"
nvidia-smi


export CUDA_HOME=$HOME/opt/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONPATH=$PWD:$PYTHONPATH

python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ./data/nuscenes \
    --max-sweeps 10
