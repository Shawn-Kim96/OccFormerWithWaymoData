#!/bin/bash

#SBATCH --job-name=waymo_eval
#SBATCH --output=results/%x/%x_%j.out
#SBATCH --error=results/%x/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpuqm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "results/${SLURM_JOB_NAME}"

DEFAULT_EXPERIMENTS=(efficientnet_b4 lr_5e4 sgd strong_aug)
if [ "$#" -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
    echo "No experiments specified for evaluation."
    exit 1
fi

CONFIG_FILE="projects/configs/occformer_waymo/waymo_base.py"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "========================================="
echo "Waymo evaluation job ${SLURM_JOB_ID}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Config: ${CONFIG_FILE}"
echo "========================================="

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    EXP_DIR="results/${EXP_NAME}"
    CKPT_PATH="${EXP_DIR}/model/latest.pth"
    LOG_DIR="${EXP_DIR}/logs"
    LOG_FILE="${LOG_DIR}/evaluate.log"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "Checkpoint not found for ${EXP_NAME}: ${CKPT_PATH}"
        exit 1
    fi

    mkdir -p "${LOG_DIR}"
    echo ""
    echo ">>> Evaluating ${EXP_NAME}"
    echo "    Checkpoint: ${CKPT_PATH}"
    echo "    Log: ${LOG_FILE}"

    python tools/test.py \
        "${CONFIG_FILE}" \
        "${CKPT_PATH}" \
        --eval mIoU \
        --cfg-options data.val.load_interval=10 data.test.load_interval=10 \
        > "${LOG_FILE}" 2>&1

    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "Evaluation failed for ${EXP_NAME}. See ${LOG_FILE}"
        exit ${STATUS}
    fi

    echo "Completed evaluation for ${EXP_NAME}. Results logged to ${LOG_FILE}"
done

echo ""
echo "All requested evaluations completed."
