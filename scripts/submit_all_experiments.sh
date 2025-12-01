#!/bin/bash

# Submit All Waymo Experiments to SLURM
# This script submits all experiments as separate jobs

# Experiment list
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

# Sample test mode flag
SAMPLE_TEST=${1:-""}

if [ -n "${SAMPLE_TEST}" ]; then
    echo "========================================="
    echo "Submitting experiments in SAMPLE TEST mode"
    echo "Each will use ~10 samples for quick testing"
    echo "========================================="
else
    echo "========================================="
    echo "Submitting experiments in FULL TRAINING mode"
    echo "========================================="
fi

echo ""
echo "Total experiments to submit: ${#EXPERIMENTS[@]}"
echo ""

# Track job IDs
JOB_IDS=()

# Submit each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo "Submitting: ${exp}"

    # Submit job
    if [ -n "${SAMPLE_TEST}" ]; then
        JOB_OUTPUT=$(sbatch scripts/run_experiment.sh ${exp} sample_test)
    else
        JOB_OUTPUT=$(sbatch scripts/run_experiment.sh ${exp})
    fi

    # Extract job ID
    JOB_ID=$(echo ${JOB_OUTPUT} | awk '{print $4}')
    JOB_IDS+=($JOB_ID)

    echo "  Job ID: ${JOB_ID}"
    echo ""

    # Small delay to avoid overwhelming scheduler
    sleep 2
done

echo "========================================="
echo "All experiments submitted!"
echo "========================================="
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Check status:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor specific job:"
echo "  tail -f results/<exp_name>/<exp_name>_<job_id>.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${JOB_IDS[@]}"
echo ""
