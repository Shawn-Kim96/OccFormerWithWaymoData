#!/bin/bash

#SBATCH --job-name=waymo_exp
#SBATCH --output=results/%x/%x_%j.out
#SBATCH --error=results/%x/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
<<<<<<< HEAD
#SBATCH --time=14-00:00:00
=======
#SBATCH --time=48:00:00
>>>>>>> 74934aa9c50c8e423dd9c698c49ac95958c35237
#SBATCH --partition=gpuql

# Usage: sbatch scripts/run_experiment.sh <exp_name> [sample_test]
# Example: sbatch scripts/run_experiment.sh baseline
# Example (test mode): sbatch scripts/run_experiment.sh baseline sample_test

# Parse arguments
EXP_NAME=${1:-baseline}
SAMPLE_TEST=${2:-""}

# Set experiment directory
EXP_DIR="results/${EXP_NAME}"
mkdir -p ${EXP_DIR}/model
mkdir -p ${EXP_DIR}/logs

# Print job information
echo "========================================="
echo "Waymo Experiment: ${EXP_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Output directory: ${EXP_DIR}"
if [ -n "${SAMPLE_TEST}" ]; then
    echo "Mode: SAMPLE TEST (10 samples)"
else
    echo "Mode: FULL TRAINING"
fi
echo "========================================="

# Load modules (adjust for your HPC)
# module load cuda/11.3
# module load python/3.8
# conda activate your_env

# Set Python path to include project root
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Check for checkpoint to resume from
CHECKPOINT=""
if [ -f "${EXP_DIR}/model/latest.pth" ]; then
    CHECKPOINT="--resume-from ${EXP_DIR}/model/latest.pth"
    echo "Resuming from checkpoint: ${EXP_DIR}/model/latest.pth"
fi

# Build config file path
CONFIG_FILE="projects/configs/occformer_waymo/waymo_base.py"

# Determine how many GPUs SLURM gave us (fallback to 1)
NPROC=${SLURM_GPUS_ON_NODE:-1}

# Build training command
TRAIN_ARGS="--work-dir ${EXP_DIR}/model --exp-name ${EXP_NAME}"

if [ -n "${SAMPLE_TEST}" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --sample-test"
fi

# Run training
echo "Starting training at $(date)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs detected: ${NPROC}"
echo "Arguments: ${TRAIN_ARGS}"
echo ""

python -m torch.distributed.launch \
    --nproc_per_node=${NPROC} \
    --master_port=29500 \
    tools/train_waymo.py \
    ${CONFIG_FILE} \
    ${TRAIN_ARGS} \
    ${CHECKPOINT} \
    --launcher pytorch \
    --deterministic \
    2>&1 | tee ${EXP_DIR}/logs/train_${SLURM_JOB_ID}.log
# Capture exit status from pipe
TRAIN_STATUS=${PIPESTATUS[0]}

# Check training status
if [ ${TRAIN_STATUS} -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "Finished at: $(date)"
    echo "========================================="

    # Run evaluation (if best checkpoint exists)
    if [ -f "${EXP_DIR}/model/best_waymo_SSC_mIoU.pth" ]; then
        echo "Running evaluation..."
        python tools/test.py \
            ${CONFIG_FILE} \
            ${EXP_DIR}/model/best_waymo_SSC_mIoU.pth \
            --eval mIoU \
            --work-dir ${EXP_DIR} \
            2>&1 | tee ${EXP_DIR}/logs/eval_${SLURM_JOB_ID}.log
    else
        echo "Warning: Best checkpoint not found, skipping evaluation"
    fi

else
    echo ""
    echo "========================================="
    echo "Training failed or interrupted!"
    echo "Check logs at: ${EXP_DIR}/logs/train_${SLURM_JOB_ID}.log"
    echo "========================================="
    exit 1
fi

echo ""
echo "Results saved in: ${EXP_DIR}"
echo "  - Model: ${EXP_DIR}/model/"
echo "  - Logs: ${EXP_DIR}/logs/"