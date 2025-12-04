#!/usr/bin/env python
"""
Complete Visualization and Analysis Tool for Waymo Occupancy Results
Generates:
1. Video visualizations (BEV + 3D)
2. Per-class IoU analysis
3. Confusion matrix
4. Failure case detection
5. Statistical reports
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint


# Waymo class colors (matching nuScenes-style palette)
WAYMO_COLORS = np.array([
    [255, 158, 0],    # 0 general_object
    [255, 99, 71],    # 1 vehicle (red)
    [255, 140, 0],    # 2 pedestrian (orange)
    [255, 215, 0],    # 3 sign (gold)
    [138, 43, 226],   # 4 cyclist (purple)
    [255, 20, 147],   # 5 traffic_light (pink)
    [139, 69, 19],    # 6 pole (brown)
    [255, 165, 0],    # 7 construction_cone
    [0, 191, 255],    # 8 bicycle (blue)
    [148, 0, 211],    # 9 motorcycle
    [105, 105, 105],  # 10 building (gray)
    [34, 139, 34],    # 11 vegetation (green)
    [85, 107, 47],    # 12 tree_trunk (dark green)
    [128, 128, 128],  # 13 road (gray)
    [189, 183, 107],  # 14 walkable (tan)
    [220, 220, 220],  # 15 free (light gray)
]) / 255.0


CLASS_NAMES = [
    'general_object', 'vehicle', 'pedestrian', 'sign', 'cyclist',
    'traffic_light', 'pole', 'construction_cone', 'bicycle', 'motorcycle',
    'building', 'vegetation', 'tree_trunk', 'road', 'walkable', 'free'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize and Analyze Waymo Results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', default='results/analysis', help='output directory')
    parser.add_argument('--num-samples', type=int, default=50, help='number of samples to visualize')
    parser.add_argument('--video-fps', type=int, default=5, help='video FPS')
    parser.add_argument('--device', default='cuda:0', help='device')
    args = parser.parse_args()
    return args


class WaymoAnalyzer:
    """Comprehensive analyzer for Waymo occupancy results."""

    def __init__(self, cfg, model, dataset, out_dir, device='cuda:0'):
        self.cfg = cfg
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.out_dir = Path(out_dir)
        self.device = device
        self.num_classes = len(CLASS_NAMES)

        # Create output directories
        (self.out_dir / 'frames').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'videos').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'plots').mkdir(parents=True, exist_ok=True)

        # Metrics accumulators
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.failure_cases = []

    def visualize_bev(self, gt_occ, pred_occ, slice_idx=8):
        """Create BEV visualization with GT and prediction side-by-side."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Ground truth
        gt_slice = gt_occ[:, :, slice_idx]
        gt_colored = WAYMO_COLORS[gt_slice]
        axes[0].imshow(gt_colored)
        axes[0].set_title('Ground Truth (BEV)', fontsize=14)
        axes[0].axis('off')

        # Prediction
        pred_slice = pred_occ[:, :, slice_idx]
        pred_colored = WAYMO_COLORS[pred_slice]
        axes[1].imshow(pred_colored)
        axes[1].set_title('Prediction (BEV)', fontsize=14)
        axes[1].axis('off')

        # Difference (correct = white, wrong = colored by GT class)
        diff = (gt_slice != pred_slice).astype(np.float32)
        diff_colored = diff[:, :, None] * gt_colored + (1 - diff[:, :, None])
        axes[2].imshow(diff_colored)
        axes[2].set_title('Errors (colored by GT class)', fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()
        return fig

    def compute_metrics(self, gt_occ, pred_occ):
        """Compute IoU metrics and update confusion matrix."""
        gt_flat = gt_occ.flatten()
        pred_flat = pred_occ.flatten()

        # Per-class IoU
        for c in range(self.num_classes):
            gt_mask = (gt_flat == c)
            pred_mask = (pred_flat == c)

            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()

            self.intersection[c] += intersection
            self.union[c] += union

        # Confusion matrix
        for gt_c in range(self.num_classes):
            for pred_c in range(self.num_classes):
                count = ((gt_flat == gt_c) & (pred_flat == pred_c)).sum()
                self.confusion_matrix[gt_c, pred_c] += count

    def detect_failure_cases(self, sample_idx, gt_occ, pred_occ, iou_threshold=0.1):
        """Detect samples with particularly poor performance."""
        # Compute sample-level mIoU
        ious = []
        for c in range(self.num_classes):
            gt_mask = (gt_occ == c)
            pred_mask = (pred_occ == c)

            if gt_mask.sum() == 0:
                continue

            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)

        sample_miou = np.mean(ious) if ious else 0.0

        if sample_miou < iou_threshold:
            self.failure_cases.append({
                'sample_idx': sample_idx,
                'miou': sample_miou,
                'present_classes': np.unique(gt_occ).tolist()
            })

    def run_inference(self, num_samples):
        """Run inference on dataset and collect results."""
        print(f"\n[1/4] Running inference on {num_samples} samples...")

        frames = []

        for i in tqdm(range(min(num_samples, len(self.dataset)))):
            # Get data
            data = self.dataset[i]
            if data is None:
                continue

            batch = collate([data], samples_per_gpu=1)
            batch = scatter(batch, [self.device])[0]

            # Forward pass
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **batch)

            # Extract GT and prediction
            gt_occ = batch['gt_occ'][0].cpu().numpy()  # [H, W, D]
            pred_logits = result[0]['output_voxels']  # [C, H, W, D]
            pred_occ = pred_logits.argmax(0).cpu().numpy()  # [H, W, D]

            # Compute metrics
            self.compute_metrics(gt_occ, pred_occ)
            self.detect_failure_cases(i, gt_occ, pred_occ)

            # Create visualization
            fig = self.visualize_bev(gt_occ, pred_occ, slice_idx=8)
            frame_path = self.out_dir / 'frames' / f'frame_{i:04d}.png'
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            frames.append(str(frame_path))

        return frames

    def create_video(self, frames, fps=5):
        """Create video from frames."""
        print(f"\n[2/4] Creating video from {len(frames)} frames...")

        if not frames:
            print("No frames to create video!")
            return

        # Read first frame to get dimensions
        first_frame = cv2.imread(frames[0])
        height, width, _ = first_frame.shape

        # Create video writer
        video_path = self.out_dir / 'videos' / 'prediction_visualization.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame_path in tqdm(frames):
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()
        print(f"Video saved to: {video_path}")

    def plot_results(self):
        """Generate all analysis plots."""
        print("\n[3/4] Generating analysis plots...")

        # Per-class IoU
        ious = self.intersection / (self.union + 1e-6)
        miou = ious.mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(self.num_classes)
        bars = ax.bar(x, ious * 100, color=WAYMO_COLORS)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylabel('IoU (%)', fontsize=12)
        ax.set_title(f'Per-Class IoU (mIoU={miou*100:.2f}%)', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=miou*100, color='r', linestyle='--', label=f'Mean: {miou*100:.2f}%')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / 'plots' / 'per_class_iou.png', dpi=150)
        plt.close()

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(14, 12))
        # Normalize by GT (rows)
        cm_normalized = self.confusion_matrix / (self.confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Normalized Confusion Matrix', fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(self.out_dir / 'plots' / 'confusion_matrix.png', dpi=150)
        plt.close()

        # Class frequency in predictions vs GT
        gt_freq = self.confusion_matrix.sum(axis=1)
        pred_freq = self.confusion_matrix.sum(axis=0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(self.num_classes)
        width = 0.35
        ax.bar(x - width/2, gt_freq / gt_freq.sum() * 100, width, label='Ground Truth', alpha=0.8)
        ax.bar(x + width/2, pred_freq / pred_freq.sum() * 100, width, label='Prediction', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title('Class Distribution: GT vs Prediction', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(self.out_dir / 'plots' / 'class_distribution.png', dpi=150)
        plt.close()

        print(f"Plots saved to: {self.out_dir / 'plots'}")

    def generate_report(self):
        """Generate text report with all statistics."""
        print("\n[4/4] Generating analysis report...")

        ious = self.intersection / (self.union + 1e-6)
        miou = ious.mean()

        report_path = self.out_dir / 'analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WAYMO OCCUPANCY PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean IoU (mIoU):     {miou * 100:.2f}%\n")
            f.write(f"Samples analyzed:    {len(self.dataset)}\n")
            f.write(f"Failure cases:       {len(self.failure_cases)}\n\n")

            # Per-class IoU
            f.write("PER-CLASS IoU\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'IoU (%)':<10} {'GT Count':<15} {'Pred Count':<15}\n")
            f.write("-" * 80 + "\n")

            gt_counts = self.confusion_matrix.sum(axis=1)
            pred_counts = self.confusion_matrix.sum(axis=0)

            for c in range(self.num_classes):
                f.write(f"{CLASS_NAMES[c]:<20} {ious[c]*100:>8.2f}  "
                       f"{int(gt_counts[c]):>12}    {int(pred_counts[c]):>12}\n")

            # Top failure cases
            f.write("\n\nTOP 10 FAILURE CASES\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Sample ID':<12} {'mIoU (%)':<12} {'Present Classes'}\n")
            f.write("-" * 80 + "\n")

            sorted_failures = sorted(self.failure_cases, key=lambda x: x['miou'])[:10]
            for case in sorted_failures:
                classes_str = ', '.join([CLASS_NAMES[c] for c in case['present_classes'] if c < self.num_classes])
                f.write(f"{case['sample_idx']:<12} {case['miou']*100:>10.2f}  {classes_str}\n")

            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            # Find worst performing classes
            worst_classes = np.argsort(ious)[:5]
            f.write("Classes needing improvement (lowest IoU):\n")
            for c in worst_classes:
                if gt_counts[c] > 0:  # Only if class exists in GT
                    f.write(f"  - {CLASS_NAMES[c]}: {ious[c]*100:.2f}% IoU\n")

            # Check if model is biased towards certain classes
            pred_ratios = pred_counts / (gt_counts + 1e-6)
            over_predicted = np.where(pred_ratios > 2.0)[0]
            under_predicted = np.where(pred_ratios < 0.5)[0]

            if len(over_predicted) > 0:
                f.write("\nOver-predicted classes (pred/GT > 2.0):\n")
                for c in over_predicted:
                    f.write(f"  - {CLASS_NAMES[c]}: {pred_ratios[c]:.2f}x GT frequency\n")

            if len(under_predicted) > 0:
                f.write("\nUnder-predicted classes (pred/GT < 0.5):\n")
                for c in under_predicted:
                    if gt_counts[c] > 100:  # Only if class is reasonably common
                        f.write(f"  - {CLASS_NAMES[c]}: {pred_ratios[c]:.2f}x GT frequency\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"\nAnalysis report saved to: {report_path}")
        print(f"\nSummary:")
        print(f"  mIoU:         {miou*100:.2f}%")
        print(f"  Best class:   {CLASS_NAMES[np.argmax(ious)]} ({ious.max()*100:.2f}%)")
        print(f"  Worst class:  {CLASS_NAMES[np.argmax(ious)]} ({ious.min()*100:.2f}%)")


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Build dataset
    val_cfg = cfg.data.val.copy()
    dataset = build_dataset(val_cfg)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Build model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print(f"Checkpoint loaded: {args.checkpoint}")

    # Create analyzer
    analyzer = WaymoAnalyzer(cfg, model, dataset, args.out_dir, args.device)

    # Run full analysis pipeline
    frames = analyzer.run_inference(args.num_samples)
    analyzer.create_video(frames, fps=args.video_fps)
    analyzer.plot_results()
    analyzer.generate_report()

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
