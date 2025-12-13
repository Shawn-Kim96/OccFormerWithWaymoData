#!/bin/bash
#SBATCH --job-name=waymo_full
#SBATCH --output=results/%x/%x_%j.out
#SBATCH --error=results/%x/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpuql

# Train + evaluate (and optionally visualize) a Waymo experiment.
# Usage: sbatch scripts/run_full_experiment.sh <exp_name>

set -e  # Exit on error

# Parse arguments
EXP_NAME=${1:-improved_fast}

# Set experiment directory
EXP_DIR="results/${EXP_NAME}"
mkdir -p ${EXP_DIR}/{model,logs,analysis,videos,plots}

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate occformer307

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# ============================================================================
# SECTION 1: TRAINING
# ============================================================================

echo ""
echo "========================================="
echo "GPU INFORMATION"
echo "========================================="
nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "========================================="

echo ""
echo "========================================="
echo "SECTION 1/4: TRAINING"
echo "========================================="
echo "Experiment: ${EXP_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Output directory: ${EXP_DIR}"
echo "========================================="

# Check for checkpoint to resume from
CHECKPOINT=""
if [ -f "${EXP_DIR}/model/latest.pth" ]; then
    CHECKPOINT="--resume-from ${EXP_DIR}/model/latest.pth"
    echo "Resuming from checkpoint: ${EXP_DIR}/model/latest.pth"
fi

# Config file
CONFIG_FILE="projects/configs/occformer_waymo/waymo_base.py"

# Determine GPU count
NPROC=${SLURM_GPUS_ON_NODE:-1}

# Training arguments
TRAIN_ARGS="--work-dir ${EXP_DIR}/model --exp-name ${EXP_NAME}"

# Run training
echo ""
echo "Starting training at $(date)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs detected: ${NPROC}"
echo "Arguments: ${TRAIN_ARGS}"
echo ""

if [ ${NPROC} -gt 1 ]; then
    # Multi-GPU training
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
else
    # Single-GPU training
    python tools/train_waymo.py \
        ${CONFIG_FILE} \
        ${TRAIN_ARGS} \
        ${CHECKPOINT} \
        --launcher none \
        --deterministic \
        2>&1 | tee ${EXP_DIR}/logs/train_${SLURM_JOB_ID}.log
fi

TRAIN_STATUS=${PIPESTATUS[0]}

if [ ${TRAIN_STATUS} -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "Training failed! Exiting..."
    echo "Check logs at: ${EXP_DIR}/logs/train_${SLURM_JOB_ID}.log"
    echo "========================================="
    exit 1
fi

echo ""
echo "Training completed successfully at $(date)"

# ============================================================================
# SECTION 2: EVALUATION
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 2/4: EVALUATION"
echo "========================================="

# Find best checkpoint
BEST_CKPT="${EXP_DIR}/model/best_waymo_SSC_mIoU.pth"
LATEST_CKPT="${EXP_DIR}/model/latest.pth"

if [ -f "${BEST_CKPT}" ]; then
    EVAL_CKPT="${BEST_CKPT}"
    echo "Using best checkpoint: ${BEST_CKPT}"
elif [ -f "${LATEST_CKPT}" ]; then
    EVAL_CKPT="${LATEST_CKPT}"
    echo "Using latest checkpoint: ${LATEST_CKPT}"
else
    echo "No checkpoint found! Skipping evaluation."
    EVAL_CKPT=""
fi

if [ -n "${EVAL_CKPT}" ]; then
    echo "Running evaluation on validation set..."
    python tools/test.py \
        ${CONFIG_FILE} \
        ${EVAL_CKPT} \
        --eval mIoU \
        --launcher none \
        2>&1 | tee ${EXP_DIR}/logs/eval_${SLURM_JOB_ID}.log

    echo "Evaluation completed at $(date)"
else
    echo "Skipping evaluation (no checkpoint)"
fi

# ============================================================================
# SECTION 3: VISUALIZATION
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 3/4: VISUALIZATION & ANALYSIS"
echo "========================================="

if [ -n "${EVAL_CKPT}" ]; then
    echo "Generating visualizations and analysis..."

    # Run comprehensive analysis (creates video + plots + report)
    python tools/visualize_and_analyze.py \
        ${CONFIG_FILE} \
        ${EVAL_CKPT} \
        --out-dir ${EXP_DIR}/analysis \
        --num-samples 100 \
        --video-fps 5 \
        --device cuda:0 \
        2>&1 | tee ${EXP_DIR}/logs/visualize_${SLURM_JOB_ID}.log

    # Copy video and plots to experiment root for easy access
    if [ -f "${EXP_DIR}/analysis/videos/prediction_visualization.mp4" ]; then
        cp ${EXP_DIR}/analysis/videos/prediction_visualization.mp4 ${EXP_DIR}/
        echo "Video copied to: ${EXP_DIR}/prediction_visualization.mp4"
    fi

    if [ -d "${EXP_DIR}/analysis/plots" ]; then
        cp -r ${EXP_DIR}/analysis/plots ${EXP_DIR}/
        echo "Plots copied to: ${EXP_DIR}/plots/"
    fi

    echo "Visualization completed at $(date)"
else
    echo "Skipping visualization (no checkpoint)"
fi

# ============================================================================
# SECTION 4: SUMMARY REPORT
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 4/4: GENERATING SUMMARY"
echo "========================================="

# Create experiment summary
SUMMARY_FILE="${EXP_DIR}/EXPERIMENT_SUMMARY.md"

cat > ${SUMMARY_FILE} << EOF
# Waymo Experiment: ${EXP_NAME}

**Job ID:** ${SLURM_JOB_ID}
**Date:** $(date)
**Node:** ${SLURM_NODELIST}
**GPUs:** ${NPROC}

---

## Configuration

- **Config file:** ${CONFIG_FILE}
- **Checkpoint:** ${EVAL_CKPT}
- **Working directory:** ${EXP_DIR}

---

## Results

### Training
- **Log:** [train_${SLURM_JOB_ID}.log](logs/train_${SLURM_JOB_ID}.log)
- **Status:** Completed successfully

### Evaluation
- **Log:** [eval_${SLURM_JOB_ID}.log](logs/eval_${SLURM_JOB_ID}.log)

### Visualization
- **Video:** [prediction_visualization.mp4](prediction_visualization.mp4)
- **Analysis report:** [analysis/analysis_report.txt](analysis/analysis_report.txt)
- **Plots:** [plots/](plots/)

---

## Key Files

\`\`\`
${EXP_DIR}/
├── model/                          # Model checkpoints
│   ├── best_waymo_SSC_mIoU.pth    # Best checkpoint
│   └── latest.pth                  # Latest checkpoint
├── logs/                           # Training logs
│   ├── train_${SLURM_JOB_ID}.log
│   ├── eval_${SLURM_JOB_ID}.log
│   └── visualize_${SLURM_JOB_ID}.log
├── analysis/                       # Analysis results
│   ├── analysis_report.txt        # Detailed analysis report
│   ├── frames/                    # Individual frames
│   ├── videos/                    # Generated videos
│   └── plots/                     # Statistical plots
├── prediction_visualization.mp4    # Main video output
├── plots/                          # Copy of analysis plots
│   ├── per_class_iou.png
│   ├── confusion_matrix.png
│   └── class_distribution.png
└── EXPERIMENT_SUMMARY.md          # This file
\`\`\`

---

## Next Steps

1. **Review metrics:** Check \`analysis/analysis_report.txt\` for detailed performance analysis
2. **Watch video:** Open \`prediction_visualization.mp4\` to see qualitative results
3. **Analyze plots:** Review plots in \`plots/\` directory
4. **Compare experiments:** Use \`scripts/compare_experiments.sh\` to compare with other runs

---

**Generated by:** run_full_experiment.sh
**Finished at:** $(date)
EOF

echo "Summary report created: ${SUMMARY_FILE}"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "========================================="
echo "EXPERIMENT COMPLETE!"
echo "========================================="
echo ""
echo "Experiment name: ${EXP_NAME}"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Duration:        Started at job submission, finished at $(date)"
echo ""
echo "Results location: ${EXP_DIR}/"
echo ""
echo "Quick access:"
echo "  - Video:    ${EXP_DIR}/prediction_visualization.mp4"
echo "  - Report:   ${EXP_DIR}/analysis/analysis_report.txt"
echo "  - Plots:    ${EXP_DIR}/plots/"
echo "  - Logs:     ${EXP_DIR}/logs/"
echo "  - Summary:  ${EXP_DIR}/EXPERIMENT_SUMMARY.md"
echo ""
echo "========================================="

# Clean up temporary frame files to save space (optional)
# Uncomment if you want to delete individual frames after video creation
# echo "Cleaning up temporary frames..."
# rm -rf ${EXP_DIR}/analysis/frames/

exit 0
