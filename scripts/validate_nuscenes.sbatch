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
export PYTHONPATH=/fs/atipa/home/018219422/OccFormerWithWaymoData:$PYTHONPATH


python tools/test.py \
    projects/configs/occformer_nusc/occformer_nusc_r50_256x704.py \
    ckpts/occformer_nusc_r50.pth \
    --eval ssc