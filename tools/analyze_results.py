#!/usr/bin/env python
"""
Analyze Waymo Experiment Results
Collects results from all experiments and generates comparison reports.
"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Experiments
EXPERIMENTS = {
    'baseline': 'Baseline (LR=1e-4, B7)',
    'lr_5e5': 'Lower LR (5e-5)',
    'lr_2e4': 'Higher LR (2e-4)',
    'strong_aug': 'Strong Augmentation',
    'weak_aug': 'Weak Augmentation',
    'efficientnet_b4': 'EfficientNet-B4',
    'resnet101': 'ResNet-101',
    'cosine_lr': 'Cosine LR',
    'sgd': 'SGD Optimizer',
    'reduced_queries': 'Reduced Queries',
}


def parse_log(log_path):
    """Parse training log to extract metrics."""
    metrics = {
        'best_miou': 0.0,
        'final_epoch': 0,
        'training_time_hours': 0.0,
    }

    if not os.path.exists(log_path):
        return metrics

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # Parse mIoU
            if 'mIoU:' in line:
                try:
                    miou = float(line.split('mIoU:')[1].split()[0])
                    metrics['best_miou'] = max(metrics['best_miou'], miou)
                except:
                    pass

            # Parse epoch
            if 'Epoch [' in line:
                try:
                    epoch_str = line.split('Epoch [')[1].split(']')[0]
                    epoch = int(epoch_str.split('/')[0])
                    metrics['final_epoch'] = max(metrics['final_epoch'], epoch)
                except:
                    pass

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return metrics


def collect_results():
    """Collect results from all experiments."""
    results = []

    for exp_key, exp_name in EXPERIMENTS.items():
        exp_dir = f"results/{exp_key}"
        model_dir = f"{exp_dir}/model"
        log_dir = f"{exp_dir}/logs"

        result = {
            'Experiment': exp_name,
            'Key': exp_key,
            'Status': 'Not Run',
            'mIoU': 0.0,
            'Epochs': 0,
            'Model Size (MB)': 0.0,
        }

        # Check if experiment was run
        if os.path.exists(model_dir):
            # Check for best checkpoint
            best_ckpt = f"{model_dir}/best_waymo_SSC_mIoU.pth"
            if os.path.exists(best_ckpt):
                result['Status'] = 'Completed'
                size_mb = os.path.getsize(best_ckpt) / (1024 * 1024)
                result['Model Size (MB)'] = f"{size_mb:.1f}"

                # Parse log
                log_files = glob.glob(f"{log_dir}/train_*.log")
                if log_files:
                    latest_log = max(log_files, key=os.path.getmtime)
                    metrics = parse_log(latest_log)
                    result['mIoU'] = metrics['best_miou']
                    result['Epochs'] = metrics['final_epoch']

            elif os.path.exists(f"{model_dir}/latest.pth"):
                result['Status'] = 'Running'

                # Parse log for progress
                log_files = glob.glob(f"{log_dir}/train_*.log")
                if log_files:
                    latest_log = max(log_files, key=os.path.getmtime)
                    metrics = parse_log(latest_log)
                    result['mIoU'] = metrics['best_miou']
                    result['Epochs'] = metrics['final_epoch']

        results.append(result)

    return pd.DataFrame(results)


def plot_results(df):
    """Generate comparison plots."""
    completed = df[df['Status'] == 'Completed'].copy()

    if len(completed) == 0:
        print("No completed experiments to plot.")
        return

    # Create output directory
    os.makedirs('results/plots', exist_ok=True)

    # Sort by mIoU
    completed = completed.sort_values('mIoU', ascending=False)

    # Convert model size to numeric
    completed['Model Size Numeric'] = completed['Model Size (MB)'].astype(float)

    # Plot 1: Performance Comparison
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    bars = plt.barh(completed['Key'], completed['mIoU'])
    plt.xlabel('mIoU (%)', fontsize=12)
    plt.ylabel('Experiment', fontsize=12)
    plt.title('Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlim(0, max(completed['mIoU']) * 1.1)

    # Color best in green
    colors = ['green' if x == completed['mIoU'].max() else 'steelblue'
              for x in completed['mIoU']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add values
    for i, v in enumerate(completed['mIoU']):
        plt.text(v + 0.5, i, f'{v:.2f}', va='center')

    plt.subplot(1, 2, 2)
    bars = plt.barh(completed['Key'], completed['Epochs'])
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Experiment', fontsize=12)
    plt.title('Training Duration', fontsize=14, fontweight='bold')

    # Add values
    for i, v in enumerate(completed['Epochs']):
        plt.text(v + 0.5, i, str(int(v)), va='center')

    plt.tight_layout()
    plt.savefig('results/plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/performance_comparison.png")
    plt.close()

    # Plot 2: Size vs Performance
    plt.figure(figsize=(10, 6))
    plt.scatter(completed['Model Size Numeric'], completed['mIoU'],
               s=200, alpha=0.6, c=range(len(completed)), cmap='viridis')

    for idx, row in completed.iterrows():
        plt.annotate(row['Key'],
                    (row['Model Size Numeric'], row['mIoU']),
                    fontsize=9, alpha=0.8)

    plt.xlabel('Model Size (MB)', fontsize=12)
    plt.ylabel('mIoU (%)', fontsize=12)
    plt.title('Model Size vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/size_vs_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/size_vs_performance.png")
    plt.close()


def generate_report(df):
    """Generate markdown report."""
    report = []
    report.append("# Waymo Fine-tuning Results\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n")

    # Summary
    completed = df[df['Status'] == 'Completed']
    running = df[df['Status'] == 'Running']
    not_run = df[df['Status'] == 'Not Run']

    report.append("\n## Summary\n")
    report.append(f"- **Completed:** {len(completed)}/{len(df)}")
    report.append(f"- **Running:** {len(running)}")
    report.append(f"- **Not Started:** {len(not_run)}\n")

    # Best result
    if len(completed) > 0:
        best = completed.loc[completed['mIoU'].idxmax()]
        report.append("\n### Best Configuration\n")
        report.append(f"- **Experiment:** {best['Experiment']}")
        report.append(f"- **mIoU:** {best['mIoU']:.2f}%")
        report.append(f"- **Epochs:** {int(best['Epochs'])}\n")

    # Results table
    report.append("\n## All Results\n")
    report.append(df.to_markdown(index=False))

    # Analysis
    if len(completed) >= 2:
        report.append("\n\n## Analysis\n")

        # Learning rate comparison
        lr_exps = completed[completed['Key'].str.contains('lr|baseline')]
        if len(lr_exps) > 0:
            report.append("\n### Learning Rate Impact\n")
            report.append(lr_exps[['Experiment', 'mIoU', 'Epochs']].to_markdown(index=False))

        # Augmentation comparison
        aug_exps = completed[completed['Key'].str.contains('aug|baseline')]
        if len(aug_exps) > 0:
            report.append("\n### Data Augmentation Impact\n")
            report.append(aug_exps[['Experiment', 'mIoU', 'Epochs']].to_markdown(index=False))

        # Architecture comparison
        arch_exps = completed[completed['Key'].str.contains('efficientnet|resnet|baseline')]
        if len(arch_exps) > 0:
            report.append("\n### Model Architecture Impact\n")
            report.append(arch_exps[['Experiment', 'mIoU', 'Model Size (MB)']].to_markdown(index=False))

    # Save report
    with open('results/report.md', 'w') as f:
        f.write('\n'.join(report))

    print("Saved: results/report.md")


def main():
    print("=" * 60)
    print("Analyzing Waymo Experiments")
    print("=" * 60)

    # Collect results
    df = collect_results()

    # Save CSV
    df.to_csv('results/comparison.csv', index=False)
    print("Saved: results/comparison.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(df)

    # Generate report
    print("\nGenerating report...")
    generate_report(df)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Check results/ directory for outputs")
    print("=" * 60)


if __name__ == '__main__':
    main()
