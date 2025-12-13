#!/bin/bash

# Check Progress of All Experiments

echo "========================================="
echo "Waymo Experiment Progress"
echo "========================================="
echo ""

# List of experiments
EXPERIMENTS=(
    "baseline"
    "lr_5e5"
    "lr_2e4"
    "strong_aug"
    "weak_aug"
    "efficientnet_b4"
    "resnet101"
    "cosine_lr"
    "sgd"
    "reduced_queries"
)

# Check each experiment
for exp in "${EXPERIMENTS[@]}"; do
    EXP_DIR="results/${exp}"

    if [ ! -d "${EXP_DIR}" ]; then
        echo "${exp}: Not started"
        continue
    fi

    # Check for checkpoint
    if [ -f "${EXP_DIR}/model/latest.pth" ]; then
        # Get latest log
        LATEST_LOG=$(ls -t ${EXP_DIR}/logs/train_*.log 2>/dev/null | head -1)

        if [ -n "${LATEST_LOG}" ]; then
            # Extract last epoch
            LAST_EPOCH=$(grep "Epoch \[" ${LATEST_LOG} | tail -1 | grep -oP 'Epoch \[\K[0-9]+' || echo "?")
            BEST_MIOU=$(grep "best_waymo_SSC_mIoU" ${LATEST_LOG} | tail -1 | grep -oP 'mIoU: \K[0-9.]+' || echo "?")

            echo "${exp}: Epoch ${LAST_EPOCH}, Best mIoU: ${BEST_MIOU}"
        else
            echo "${exp}: Running (no log yet)"
        fi
    elif [ -f "${EXP_DIR}/model/best_waymo_SSC_mIoU.pth" ]; then
        # Completed
        EVAL_LOG="${EXP_DIR}/logs/eval_*.log"
        if [ -f ${EVAL_LOG} ]; then
            FINAL_MIOU=$(grep "mIoU" ${EVAL_LOG} | tail -1 | grep -oP 'mIoU: \K[0-9.]+' || echo "?")
            echo "${exp}: Completed, Final mIoU: ${FINAL_MIOU}"
        else
            echo "${exp}: Completed"
        fi
    else
        echo "${exp}: Queued or failed"
    fi
done

echo ""
echo "========================================="
echo "SLURM Queue Status"
echo "========================================="
squeue -u $USER --format="%.10i %.15j %.10T %.10M %.6D %R"
echo ""
