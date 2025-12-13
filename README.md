# Waymo Fine-tuning Quick Start Guide

## IMPORTANT: Pretrained Weights (HPC has no internet!)

**Run this BEFORE uploading to HPC:**

```bash
# On your local machine with internet
bash scripts/download_pretrained.sh
```

This downloads all required pretrained weights:
- `ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth` (255 MB)
- `ckpts/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth` (75 MB)
- `ckpts/resnet101-5d3b4d8f.pth` (~170 MB)

**Total size: ~500 MB**

Then upload the entire `ckpts/` directory to HPC.

## Directory Structure

All results are organized under `results/` directory:

```
results/
├── baseline/
│   ├── model/              # Checkpoints
│   │   ├── latest.pth      # For resuming
│   │   ├── best_waymo_SSC_mIoU.pth
│   │   └── epoch_*.pth
│   └── logs/               # Training logs
│       ├── train_<job_id>.log
│       └── eval_<job_id>.log
├── lr_5e5/
│   ├── model/
│   └── logs/
├── lr_2e4/
...
└── reduced_queries/
    ├── model/
    └── logs/
```

## Available Experiments

All configurations are defined in `projects/configs/occformer_waymo/experiments.py`:

| Experiment | Description | Key Changes |
|------------|-------------|-------------|
| `baseline_fast` | Standard configuration | LR=1e-4, B7, normal aug |
| `lr_5e5` | Lower learning rate | LR=5e-5 |
| `lr_2e4` | Higher learning rate | LR=2e-4 |
| `strong_aug` | Strong augmentation | Rot ±10°, Scale 0.85-1.15 |
| `weak_aug` | Weak augmentation | Minimal augmentation |
| `resnet101` | Different backbone | ResNet-101 |
| `sgd` | SGD optimizer | SGD instead of AdamW |
| `improved_imbalance_q30` | Weighted CE | Change loss for solving imbalance class |

## Step-by-Step Usage

### 0. Prepare Pretrained Weights (BEFORE HPC Upload!)

On your **local machine with internet**:
```bash
bash scripts/download_pretrained.sh
```

Upload to HPC:
```bash
# From your local machine
scp -r ckpts/ your_username@hpc.sjsu.edu:/home/018219422/OccFormerWithWaymoData/
```

Verify on HPC:
```bash
ls -lh ckpts/
# Should see:
# - efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth (255 MB)
# - efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth (75 MB)
```

### 1. Sample Test (Recommended First!)

Test with ~10 samples to verify everything works:

```bash
# Test single experiment
sbatch scripts/run_experiment.sh baseline sample_test

# Test all experiments (fast)
bash scripts/submit_all_experiments.sh sample_test
```

Each sample test should complete in **~1 hour**.

### 2. Full Training

After verifying sample test works:

```bash
# Single experiment
sbatch scripts/run_experiment.sh baseline

# All experiments
bash scripts/submit_all_experiments.sh
```

Each full experiment takes **~8-12 hours**.

### 3. Check Progress

```bash
# Quick status
bash scripts/check_progress.sh

# Detailed queue info
squeue -u $USER

# Watch specific experiment
tail -f results/baseline/logs/train_<job_id>.log
```

### 4. Resume Interrupted Training

Training **automatically resumes** from latest checkpoint if interrupted.

Manual resume:
```bash
sbatch scripts/run_experiment.sh baseline
# Will detect results/baseline/model/latest.pth and resume
```

## Experiment Configuration

Edit `scripts/run_experiment.sh` to adjust SLURM settings:

```bash
#SBATCH --time=48:00:00        # Max time
#SBATCH --mem=64G              # Memory
#SBATCH --partition=gpu        # Your partition name
```

## Time Estimates

### Sample Test Mode (~10 samples)
- Training: 2 epochs × ~30 min = **1 hour**
- Total for 10 experiments: **~10 hours** (parallel)

### Full Training Mode
- Baseline (30 epochs): **~58 hours**

## File Reference

- `scripts/run_experiment.sh` - Main SBATCH script
- `scripts/submit_all_experiments.sh` - Submit all experiments
- `scripts/check_progress.sh` - Monitor progress
- `tools/train_waymo.py` - Training script
- `projects/configs/occformer_waymo/experiments.py` - Experiment configs
- `projects/configs/occformer_waymo/waymo_base.py` - Base config
