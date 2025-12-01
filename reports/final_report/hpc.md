HPC setting for cuda-11.3
```
export CUDA_HOME=$HOME/opt/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONPATH=$HOME/OccFormerWithWaymoData:$PYTHONPATH
```