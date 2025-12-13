# Quick Start Guide

Minimal commands to spin up the improved Waymo experiments.

---

## Step 1: Smoke test (~30s)
```bash
bash scripts/quick_test.sh
```
Confirms imports, loader resize, and loss registration.

---

## Step 2: Launch an experiment (~5s)
**Memory-friendly (recommended)**
```bash
sbatch scripts/run_full_experiment.sh improved_small_grid
```
- Uses native 200×200×16 GT
- Focal loss for imbalance
- ~40% lower memory

**Higher resolution alternative**
```bash
sbatch scripts/run_full_experiment.sh improved_fast
```
- Resizes GT to 256×256×32 for finer detail (needs more memory)

---

## Step 3: Inspect outputs
```bash
# Video (if visualization was run)
ls results/improved_small_grid/videos/

# Logs
tail -f results/improved_small_grid/logs/train_*.log

# Plots / reports (after visualization)
ls results/improved_small_grid/plots/
```

---

## Monitoring
```bash
# Training progress
tail -f results/improved_small_grid/logs/train_*.log

# Cluster job status
squeue -u $USER

# Quick loss snapshot
grep "loss_cls" results/improved_small_grid/logs/train_*.log | tail -20
```

---

## Next Steps
- Use more data:
  ```python
  'data_train_load_interval': 5,  # 20% of data
  'runner': dict(type='EpochBasedRunner', max_epochs=50),
  ```
- Compare experiments:
  ```bash
  python tools/compare_experiments.py --results-dir results
  ```

---

## FAQ
- Slow training? Check GPU utilization with `nvidia-smi`.
- OOM? Use `improved_small_grid`, keep `samples_per_gpu=1`, or drop to EfficientNet-B4.
- Missing video? Install OpenCV/matplotlib (`pip install opencv-python matplotlib seaborn`).

See also: `IMPROVEMENTS_README.md` and `DIAGNOSTIC_REPORT.md`.
