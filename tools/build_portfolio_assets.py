#!/usr/bin/env python3
"""Build reproducible web assets and a manifest for the OccFormer portfolio site."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_TITLE = "Camera-only 3D Semantic Occupancy on Waymo with OccFormer"
PROJECT_SUBTITLE = (
    "A reproducible Waymo adaptation of OccFormer covering dataset integration, training, "
    "evaluation, failure analysis, and qualitative occupancy demos."
)
ABSTRACT = (
    "This project turns the public OccFormer codebase into a working Waymo occupancy pipeline inside "
    "MMDetection3D. The core work was systems integration: aligning Waymo KITTI-format sensor exports with "
    "Occ3D-Waymo labels, reconciling voxel-grid mismatches, fixing label semantics, and stabilizing training "
    "under a single 12 GB GPU budget."
)
ENGINEERING_LESSONS = [
    "Aligned Waymo image exports with Occ3D-Waymo occupancy labels and fixed the free-space label convention.",
    "Bridged a GT voxel size mismatch (200×200×16) against the default model grid (256×256×32).",
    "Stabilized training and evaluation in an HPC environment with tight memory and CUDA/toolchain constraints.",
    "Tracked failed experiments and environment issues instead of hiding them, then folded those findings back into a reproducible baseline.",
]
REPRO_STEPS = [
    "Use the provided MMDetection3D + OccFormer configs under projects/configs/occformer_waymo/.",
    "Inspect the final report and generated figures under reports/final_report/ for experiment rationale and metrics.",
    "Run the portfolio asset build script to regenerate the web bundle and manifest from repository artifacts.",
    "Use the GitHub Pages workflow to rebuild, validate, and publish the static site from the repo state.",
]


@dataclass
class AssetRecord:
    id: str
    section: str
    kind: str
    src: str
    title: str
    caption: str
    alt: str
    source: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None


@dataclass
class SectionRecord:
    id: str
    eyebrow: str
    title: str
    body: str
    asset_ids: list[str]


def resolve_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here] + list(here.parents):
        if (candidate / "reports" / "final_report").is_dir() and (candidate / "projects").is_dir():
            return candidate
    raise RuntimeError("Could not resolve repository root")


def extra_source_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = [repo_root]
    state_root = os.environ.get("OMX_TEAM_STATE_ROOT")
    if state_root:
        state_path = Path(state_root).resolve()
        if state_path.name == "state" and state_path.parent.name == ".omx":
            roots.append(state_path.parent.parent)
    unique: list[Path] = []
    seen = set()
    for root in roots:
        root = root.resolve()
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def find_first(roots: Iterable[Path], rel_path: str) -> Path:
    for root in roots:
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(rel_path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dest: Path) -> tuple[int, int]:
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    with Image.open(dest) as image:
        width, height = image.size
    return width, height


def write_video_derivatives(src: Path, video_dest: Path, poster_dest: Path, filmstrip_dest: Path, *, target_width: int = 960) -> tuple[int, int, float]:
    ensure_dir(video_dest.parent)
    ensure_dir(poster_dest.parent)
    ensure_dir(filmstrip_dest.parent)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Video has invalid metadata: {src}")

    scale = min(1.0, target_width / width)
    out_width = max(2, int(width * scale) // 2 * 2)
    out_height = max(2, int(height * scale) // 2 * 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_dest), fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not write video: {video_dest}")

    strip_indices = {int(round(i * (frame_count - 1) / 3)) for i in range(4)}
    strip_frames = []
    first_frame_rgb = None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if first_frame_rgb is None:
            first_frame_rgb = rgb
        if idx in strip_indices:
            strip_frames.append(rgb)
        idx += 1

    cap.release()
    writer.release()

    if first_frame_rgb is None:
        raise RuntimeError(f"Could not read any frames from {src}")

    Image.fromarray(first_frame_rgb).save(poster_dest, quality=92)

    while len(strip_frames) < 4:
        strip_frames.append(first_frame_rgb)
    resized_strip = [Image.fromarray(frame).resize((320, 180)) for frame in strip_frames[:4]]
    filmstrip = Image.new("RGB", (320 * 4, 180))
    for offset, frame in enumerate(resized_strip):
        filmstrip.paste(frame, (320 * offset, 0))
    filmstrip.save(filmstrip_dest, quality=92)

    duration = frame_count / fps if fps else 0.0
    return out_width, out_height, duration


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_class_rankings(iou_csv: Path, out_path: Path) -> tuple[int, int]:
    rows = [row for row in read_csv_rows(iou_csv) if row["experiment"] == "baseline_fast" and row["class"] != "mIoU"]
    rows.sort(key=lambda row: float(row["iou"]), reverse=True)
    top = rows[:8]
    labels = [row["class"].replace("_", " ") for row in top]
    values = [float(row["iou"]) for row in top]

    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(labels, values, color="#6C8FF8")
    plt.ylabel("IoU (%)")
    plt.title("Best baseline: strongest classes on Waymo validation")
    plt.xticks(rotation=30, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    with Image.open(out_path) as img:
        return img.size


def plot_experiment_tradeoffs(summary_csv: Path, settings_csv: Path, out_path: Path) -> tuple[int, int]:
    summary_rows = {row["experiment"]: row for row in read_csv_rows(summary_csv)}
    settings_rows = {row["experiment"]: row for row in read_csv_rows(settings_csv)}
    experiments = [name for name in summary_rows.keys() if name in settings_rows]
    experiments.sort(key=lambda name: float(summary_rows[name]["mIoU"]), reverse=True)
    labels = []
    values = []
    annotations = []
    for experiment in experiments:
        summary = summary_rows[experiment]
        settings = settings_rows[experiment]
        labels.append(experiment.replace("_", " "))
        values.append(float(summary["mIoU"]))
        annotations.append(f"{settings['backbone']} • {settings['loss']}")

    plt.figure(figsize=(10.5, 5.2))
    bars = plt.barh(labels, values, color="#2A9D8F")
    plt.xlabel("mIoU")
    plt.title("Experiment leaderboard with backbone/loss context")
    plt.gca().invert_yaxis()
    for bar, value, annotation in zip(bars, values, annotations):
        y = bar.get_y() + bar.get_height() / 2
        plt.text(value + 0.25, y, f"{value:.2f}  {annotation}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    with Image.open(out_path) as img:
        return img.size


def parse_remote_url(repo_root: Path) -> str | None:
    config_path = repo_root / ".git" / "config"
    if not config_path.exists():
        return None
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"url = (.+github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^.\n]+)(?:\.git)?)", text)
    if not match:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    return f"https://{owner.lower()}.github.io/{repo}/"


def build_manifest(repo_root: Path, generated_dir: Path) -> dict:
    roots = extra_source_roots(repo_root)
    report_root = repo_root / "reports" / "final_report"
    figures_root = report_root / "figures"

    ensure_dir(generated_dir)

    assets: list[AssetRecord] = []

    def add_image(asset_id: str, section: str, title: str, caption: str, alt: str, source_rel: str, dest_name: str) -> None:
        src = find_first(roots, source_rel)
        dest = generated_dir / dest_name
        width, height = copy_image(src, dest)
        assets.append(AssetRecord(
            id=asset_id,
            section=section,
            kind="image",
            src=dest.relative_to(repo_root).as_posix(),
            title=title,
            caption=caption,
            alt=alt,
            source=source_rel,
            width=width,
            height=height,
            size_bytes=dest.stat().st_size,
        ))

    add_image(
        "system-architecture",
        "method",
        "System architecture",
        "Pipeline view showing the repo's Waymo data flow, training stack, and evaluation path.",
        "System architecture diagram for the OccFormer Waymo adaptation.",
        "reports/final_report/figures/system_architecture.png",
        "system-architecture.png",
    )
    add_image(
        "occformer-framework",
        "method",
        "OccFormer framework",
        "Core model framing from the final report: lift image features to 3D, reason jointly, and decode occupancy masks.",
        "OccFormer framework figure from the final report.",
        "reports/final_report/figures/occformer_framework.jpg",
        "occformer-framework.jpg",
    )
    add_image(
        "miou-experiments",
        "results",
        "mIoU by experiment",
        "Validation mIoU across the major Waymo experiment variants documented in the report.",
        "Bar chart of validation mIoU by experiment.",
        "reports/final_report/figures/miou_by_experiment.png",
        "miou-by-experiment.png",
    )
    add_image(
        "per-class-baseline",
        "results",
        "Per-class IoU",
        "Per-class IoU for the best completed baseline_fast run, highlighting strong road/free-space performance and weaker rare classes.",
        "Per-class IoU chart for the baseline_fast experiment.",
        "reports/final_report/figures/per_class_baseline_fast.png",
        "per-class-baseline-fast.png",
    )

    baseline_video = find_first(roots, "reports/baseline_fast_scene0.mp4")
    hero_video = generated_dir / "hero-baseline-fast.mp4"
    hero_poster = generated_dir / "hero-baseline-fast-poster.jpg"
    hero_strip = generated_dir / "hero-baseline-fast-filmstrip.jpg"
    hero_width, hero_height, hero_duration = write_video_derivatives(
        baseline_video,
        hero_video,
        hero_poster,
        hero_strip,
    )
    assets.append(AssetRecord(
        id="hero-video",
        section="hero",
        kind="video",
        src=hero_video.relative_to(repo_root).as_posix(),
        title="Waymo occupancy demo",
        caption="A lightweight hero clip from the best completed baseline_fast run, used as the landing-page qualitative demo.",
        alt="Short hero video showing camera views and bird's-eye occupancy predictions for the Waymo baseline.",
        source="reports/baseline_fast_scene0.mp4",
        width=hero_width,
        height=hero_height,
        duration_seconds=round(hero_duration, 2),
        size_bytes=hero_video.stat().st_size,
    ))
    assets.append(AssetRecord(
        id="hero-poster",
        section="hero",
        kind="image",
        src=hero_poster.relative_to(repo_root).as_posix(),
        title="Hero poster frame",
        caption="First decoded frame from the hero video for poster/fallback rendering.",
        alt="Poster frame extracted from the Waymo occupancy demo video.",
        source="reports/baseline_fast_scene0.mp4",
        width=hero_width,
        height=hero_height,
        size_bytes=hero_poster.stat().st_size,
    ))
    assets.append(AssetRecord(
        id="hero-filmstrip",
        section="qualitative",
        kind="image",
        src=hero_strip.relative_to(repo_root).as_posix(),
        title="Qualitative filmstrip",
        caption="Four evenly sampled frames to keep the qualitative story visible even before the video plays.",
        alt="Filmstrip with four frames from the baseline Waymo occupancy demo.",
        source="reports/baseline_fast_scene0.mp4",
        width=1280,
        height=180,
        size_bytes=hero_strip.stat().st_size,
    ))

    class_rankings = generated_dir / "baseline-class-rankings.png"
    rankings_width, rankings_height = plot_class_rankings(report_root / "iou_result.csv", class_rankings)
    assets.append(AssetRecord(
        id="baseline-class-rankings",
        section="results",
        kind="image",
        src=class_rankings.relative_to(repo_root).as_posix(),
        title="Best recovered classes",
        caption="Top per-class IoU values from baseline_fast, useful for reading where the model is currently strongest.",
        alt="Bar chart ranking the strongest classes recovered by the baseline_fast model.",
        source="reports/final_report/iou_result.csv",
        width=rankings_width,
        height=rankings_height,
        size_bytes=class_rankings.stat().st_size,
    ))

    tradeoffs = generated_dir / "experiment-tradeoffs.png"
    tradeoffs_width, tradeoffs_height = plot_experiment_tradeoffs(
        report_root / "miou_summary.csv",
        report_root / "experiment_settings.csv",
        tradeoffs,
    )
    assets.append(AssetRecord(
        id="experiment-tradeoffs",
        section="results",
        kind="image",
        src=tradeoffs.relative_to(repo_root).as_posix(),
        title="Experiment trade-offs",
        caption="The leaderboard pairs each experiment's mIoU with the backbone/loss combination that produced it.",
        alt="Horizontal bar chart showing experiment leaderboard with backbone and loss annotations.",
        source="reports/final_report/miou_summary.csv + reports/final_report/experiment_settings.csv",
        width=tradeoffs_width,
        height=tradeoffs_height,
        size_bytes=tradeoffs.stat().st_size,
    ))

    sections = [
        SectionRecord(
            id="overview",
            eyebrow="Overview",
            title="What was built in this repo",
            body=(
                "This repo is not just a paper reproduction. It contains the data plumbing, config work, training/evaluation scripts, "
                "analysis artifacts, and final reporting needed to make OccFormer run on Waymo occupancy data end to end."
            ),
            asset_ids=["hero-filmstrip"],
        ),
        SectionRecord(
            id="method",
            eyebrow="Method",
            title="Adapt the model, then make the pipeline survive reality",
            body=(
                "The project keeps OccFormer's camera-only 3D occupancy framing, then focuses on the hard systems work: aligning labels, "
                "bridging voxel-grid mismatches, and keeping MMDetection3D training/evaluation stable in an HPC setting."
            ),
            asset_ids=["system-architecture", "occformer-framework"],
        ),
        SectionRecord(
            id="results",
            eyebrow="Results",
            title="Quantitative evidence from the report and logs",
            body=(
                "The best fully completed run in this repo is baseline_fast at 13.41 mIoU. The rest of the page keeps those results grounded in the report tables and generated figures."
            ),
            asset_ids=["miou-experiments", "per-class-baseline", "baseline-class-rankings", "experiment-tradeoffs"],
        ),
        SectionRecord(
            id="engineering",
            eyebrow="Engineering lessons",
            title="Most of the difficulty was integration, not architecture changes",
            body=(
                "The repo documents concrete failures—registry ordering issues, image loader mismatches, SLURM quirks, and GPU-memory limits—so the page can show what was learned, not just the final score."
            ),
            asset_ids=[],
        ),
        SectionRecord(
            id="reproducibility",
            eyebrow="Reproducibility",
            title="Everything on this page traces back to repo evidence",
            body=(
                "Every figure, metric, and media block is copied or derived from tracked report assets, CSV summaries, or demo videos already present in the repository."
            ),
            asset_ids=[],
        ),
    ]

    summary_rows = read_csv_rows(report_root / "miou_summary.csv")
    ranking = [
        {"experiment": row["experiment"], "miou": float(row["mIoU"])}
        for row in summary_rows
    ]
    ranking.sort(key=lambda row: row["miou"], reverse=True)

    manifest = {
        "project": {
            "title": PROJECT_TITLE,
            "subtitle": PROJECT_SUBTITLE,
            "abstract": ABSTRACT,
            "pages_url_guess": parse_remote_url(repo_root),
            "repo_url": "https://github.com/Shawn-Kim96/OccFormerWithWaymoData",
            "report_url": "reports/final_report/main.pdf",
            "report_source_url": "reports/final_report/main.tex",
            "metrics": {
                "best_miou": 13.41,
                "best_experiment": "baseline_fast",
                "hero_duration_seconds": round(hero_duration, 2),
                "hero_size_mb": round(hero_video.stat().st_size / 1024 / 1024, 2),
                "experiments_summarized": len(ranking),
            },
        },
        "highlights": [
            "Built a working Waymo occupancy pipeline inside MMDetection3D rather than changing the core architecture.",
            "Documented the best completed validation score (13.41 mIoU) and the experiment trade-offs behind it.",
            "Converted report artifacts into a lightweight static site and reproducible web asset bundle.",
        ],
        "engineering_lessons": ENGINEERING_LESSONS,
        "repro_steps": REPRO_STEPS,
        "experiment_ranking": ranking,
        "sections": [asdict(section) for section in sections],
        "assets": [asdict(asset) for asset in assets],
    }

    manifest_path = generated_dir / "portfolio-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    repo_root = resolve_repo_root()
    generated_dir = repo_root / "site" / "assets" / "generated"
    manifest = build_manifest(repo_root, generated_dir)
    print(json.dumps({
        "generated_dir": generated_dir.relative_to(repo_root).as_posix(),
        "asset_count": len(manifest["assets"]),
        "pages_url_guess": manifest["project"]["pages_url_guess"],
    }, indent=2))


if __name__ == "__main__":
    main()
