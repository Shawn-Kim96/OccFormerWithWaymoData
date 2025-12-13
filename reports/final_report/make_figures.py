#!/usr/bin/env python

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


REPORT_DIR = Path(__file__).resolve().parent


def find_repo_root(start_dir: Path) -> Path:
    for p in [start_dir] + list(start_dir.parents):
        if (p / "results").is_dir() and (p / "projects").is_dir():
            return p
        if (p / ".git").exists():
            return p
    return start_dir


REPO_ROOT = find_repo_root(REPORT_DIR)
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = REPORT_DIR / "figures"

CLASS_ORDER = [
    "general_object",
    "vehicle",
    "pedestrian",
    "sign",
    "cyclist",
    "traffic_light",
    "pole",
    "construction_cone",
    "bicycle",
    "motorcycle",
    "building",
    "vegetation",
    "tree_trunk",
    "road",
    "walkable",
    "free",
]


def parse_evaluate_log(log_path: Path):
    lines = log_path.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Class") and "|" in line:
            start = i
    if start is None:
        return None

    records = []
    for line in lines[start:]:
        if "|" not in line:
            continue
        s = line.strip()
        if not s or set(s) <= set("-+| "):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) != 2:
            continue
        cls, val = parts
        try:
            iou = float(val)
        except ValueError:
            continue
        records.append((cls, iou))

    if not records:
        return None
    return pd.DataFrame(records, columns=["class", "iou"])


def main():
    sns.set_theme(style="whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class = {}

    for log_path in RESULTS_ROOT.glob("*/logs/evaluate.log"):
        exp = log_path.parent.parent.name
        df = parse_evaluate_log(log_path)
        if df is None:
            continue

        miou_row = df[df["class"] == "mIoU"]
        if miou_row.empty:
            continue
        miou = float(miou_row["iou"].iloc[0])
        rows.append({"experiment": exp, "mIoU": miou})

        df_pc = df[df["class"] != "mIoU"].copy()
        df_pc = df_pc[df_pc["class"].isin(CLASS_ORDER)]
        df_pc["class"] = pd.Categorical(df_pc["class"], categories=CLASS_ORDER, ordered=True)
        df_pc = df_pc.sort_values("class")
        per_class[exp] = df_pc

    summary = pd.DataFrame(rows).sort_values("mIoU", ascending=False)
    summary.to_csv(REPORT_DIR / "miou_summary.csv", index=False)

    # mIoU across experiments
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=summary, x="experiment", y="mIoU", color="#4C72B0")
    ax.set_ylabel("mIoU")
    ax.set_xlabel("Experiment")
    ax.set_title("Waymo val mIoU by experiment (from evaluate.log)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "miou_by_experiment.png", dpi=200)
    plt.close()

    # Per-class IoU (baseline_fast)
    if "baseline_fast" in per_class:
        df = per_class["baseline_fast"]
        plt.figure(figsize=(10, 4.5))
        ax = sns.barplot(data=df, x="class", y="iou", color="#55A868")
        ax.set_ylabel("IoU")
        ax.set_xlabel("Class")
        ax.set_title("Per-class IoU (baseline_fast)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_class_baseline_fast.png", dpi=200)
        plt.close()

    print(f"Wrote {REPORT_DIR / 'miou_summary.csv'}")
    print(f"Wrote figures under {OUT_DIR}/")


if __name__ == "__main__":
    main()
