#!/usr/bin/env python
"""
Compare multiple Waymo experiments and generate comparison report.
"""
import argparse
import os
from pathlib import Path
import re
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CLASS_NAMES = [
    'general_object', 'vehicle', 'pedestrian', 'sign', 'cyclist',
    'traffic_light', 'pole', 'construction_cone', 'bicycle', 'motorcycle',
    'building', 'vegetation', 'tree_trunk', 'road', 'walkable', 'free'
]


def parse_analysis_report(report_path):
    """Parse analysis report to extract metrics."""
    metrics = {'per_class_iou': {}}

    with open(report_path, 'r') as f:
        lines = f.readlines()

    # Find mIoU
    for line in lines:
        if 'Mean IoU (mIoU):' in line:
            miou = float(line.split(':')[-1].strip().replace('%', ''))
            metrics['miou'] = miou

    # Find per-class IoU section
    in_class_section = False
    for line in lines:
        if 'PER-CLASS IoU' in line:
            in_class_section = True
            continue

        if in_class_section and '---' in line:
            continue

        if in_class_section and line.strip():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in CLASS_NAMES:
                class_name = parts[0]
                try:
                    iou = float(parts[1])
                    metrics['per_class_iou'][class_name] = iou
                except:
                    pass

    return metrics


def find_experiments(results_dir='results'):
    """Find all completed experiments."""
    experiments = {}

    results_path = Path(results_dir)
    if not results_path.exists():
        return experiments

    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Check if experiment has analysis report
        report_path = exp_dir / 'analysis' / 'analysis_report.txt'
        if report_path.exists():
            metrics = parse_analysis_report(report_path)
            experiments[exp_dir.name] = {
                'path': str(exp_dir),
                'metrics': metrics
            }

    return experiments


def plot_comparison(experiments, out_dir):
    """Generate comparison plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_names = list(experiments.keys())
    mious = [exp['metrics'].get('miou', 0) for exp in experiments.values()]

    # Plot 1: mIoU comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('husl', len(exp_names))
    bars = ax.bar(range(len(exp_names)), mious, color=colors)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)', fontsize=12)
    ax.set_title('Experiment Comparison: Overall mIoU', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mious)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / 'miou_comparison.png', dpi=150)
    plt.close()

    # Plot 2: Per-class IoU comparison (heatmap)
    class_iou_matrix = []
    for exp_name in exp_names:
        class_ious = []
        per_class = experiments[exp_name]['metrics'].get('per_class_iou', {})
        for class_name in CLASS_NAMES:
            iou = per_class.get(class_name, 0)
            class_ious.append(iou)
        class_iou_matrix.append(class_ious)

    class_iou_matrix = np.array(class_iou_matrix)

    fig, ax = plt.subplots(figsize=(16, len(exp_names) + 2))
    sns.heatmap(class_iou_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=exp_names,
                vmin=0, vmax=100, cbar_kws={'label': 'IoU (%)'}, ax=ax)
    ax.set_title('Per-Class IoU Comparison', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / 'per_class_comparison.png', dpi=150)
    plt.close()

    print(f"Comparison plots saved to: {out_dir}")


def generate_markdown_report(experiments, out_path):
    """Generate markdown comparison report."""
    exp_names = sorted(experiments.keys())

    with open(out_path, 'w') as f:
        f.write("# Waymo Experiment Comparison\n\n")
        f.write(f"**Experiments analyzed:** {len(exp_names)}\n\n")
        f.write("---\n\n")

        # Overall ranking
        f.write("## Overall Ranking (by mIoU)\n\n")
        ranked = sorted(exp_names, key=lambda x: experiments[x]['metrics'].get('miou', 0), reverse=True)

        f.write("| Rank | Experiment | mIoU (%) |\n")
        f.write("|------|------------|----------|\n")
        for i, exp_name in enumerate(ranked, 1):
            miou = experiments[exp_name]['metrics'].get('miou', 0)
            f.write(f"| {i} | {exp_name} | {miou:.2f} |\n")

        # Detailed comparison table
        f.write("\n## Detailed Per-Class Comparison\n\n")
        f.write("| Class | " + " | ".join(exp_names) + " |\n")
        f.write("|-------|" + "------|" * len(exp_names) + "\n")

        for class_name in CLASS_NAMES:
            row = [class_name]
            for exp_name in exp_names:
                per_class = experiments[exp_name]['metrics'].get('per_class_iou', {})
                iou = per_class.get(class_name, 0)
                row.append(f"{iou:.2f}")
            f.write("| " + " | ".join(row) + " |\n")

        # Best experiment per class
        f.write("\n## Best Experiment Per Class\n\n")
        f.write("| Class | Best Experiment | IoU (%) |\n")
        f.write("|-------|-----------------|----------|\n")

        for class_name in CLASS_NAMES:
            best_exp = None
            best_iou = 0

            for exp_name in exp_names:
                per_class = experiments[exp_name]['metrics'].get('per_class_iou', {})
                iou = per_class.get(class_name, 0)
                if iou > best_iou:
                    best_iou = iou
                    best_exp = exp_name

            f.write(f"| {class_name} | {best_exp or 'N/A'} | {best_iou:.2f} |\n")

        f.write("\n---\n\n")
        f.write("**Generated by:** compare_experiments.py\n")

    print(f"Markdown report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Waymo Experiments')
    parser.add_argument('--results-dir', default='results', help='results directory')
    parser.add_argument('--out-dir', default='results/comparison', help='output directory')
    args = parser.parse_args()

    # Find experiments
    print("Scanning for completed experiments...")
    experiments = find_experiments(args.results_dir)

    if not experiments:
        print(f"No completed experiments found in {args.results_dir}/")
        print("Make sure experiments have analysis reports in analysis/analysis_report.txt")
        return

    print(f"Found {len(experiments)} completed experiments:")
    for exp_name in sorted(experiments.keys()):
        miou = experiments[exp_name]['metrics'].get('miou', 0)
        print(f"  - {exp_name}: mIoU = {miou:.2f}%")

    # Generate comparison
    print("\nGenerating comparison plots...")
    plot_comparison(experiments, args.out_dir)

    print("Generating comparison report...")
    report_path = Path(args.out_dir) / 'COMPARISON_REPORT.md'
    generate_markdown_report(experiments, report_path)

    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Results saved to: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
