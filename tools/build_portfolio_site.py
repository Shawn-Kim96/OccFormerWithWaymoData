#!/usr/bin/env python3
"""Build a static portfolio site for the OccFormer Waymo project."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "final_report"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "site"
DEFAULT_VIDEO_SIZE = (1280, 720)
DEFAULT_VIDEO_FPS = 12

REQUIRED_SOURCE_FILES = {
    "system_architecture": REPO_ROOT / "reports" / "final_report" / "figures" / "system_architecture.png",
    "occformer_framework": REPO_ROOT / "reports" / "final_report" / "figures" / "occformer_framework.jpg",
    "miou_by_experiment": REPO_ROOT / "reports" / "final_report" / "figures" / "miou_by_experiment.png",
    "per_class_baseline_fast": REPO_ROOT / "reports" / "final_report" / "figures" / "per_class_baseline_fast.png",
    "per_class_all": REPO_ROOT / "reports" / "final_report" / "figures" / "per_class_all.png",
    "report_pdf": REPO_ROOT / "reports" / "final_report" / "main.pdf",
}

SECTION_COPY = {
    "title": "OccFormer on Waymo",
    "subtitle": "Camera-only 3D semantic occupancy on Waymo, adapted from OccFormer and hardened into a reproducible MMDetection3D pipeline under a single 12 GB GPU budget.",
    "summary": "This portfolio page condenses the repo's report, configs, figures, and scripts into a single public-facing research case study with deterministic asset generation and static deployment.",
    "method_summary": "The system lifts five synchronized camera views into a voxel-aligned 3D representation, runs dual-path 3D reasoning, and predicts a 16-class occupancy grid over Waymo scenes.",
    "results_summary": "The best completed baseline_fast run reached 13.41 mIoU on the validation split while operating on 10% data slices for practical iteration speed.",
    "engineering_summary": "Most of the work was systems integration: data alignment, label remapping, voxel-grid mismatch repair, memory pressure reduction, and workflow stabilization on HPC nodes.",
}

ENGINEERING_CHALLENGES = [
    {"title": "Loader mismatch", "body": "Swapped nuScenes-style assumptions for a Waymo/KITTI-compatible multiview loader so image tensors and metadata line up with the occupancy pipeline."},
    {"title": "Pose ordering", "body": "Corrected a silent two-camera pose mismatch that otherwise corrupts feature lifting without always crashing training."},
    {"title": "Free-space labels", "body": "Remapped Occ3D-Waymo free-space label 23 to the repo's class index 15 so supervision and evaluation agree."},
    {"title": "Voxel grid alignment", "body": "Handled 200×200×16 ground truth against 256×256×32 model outputs with explicit resizing safeguards for training and visualization."},
    {"title": "12 GB GPU ceiling", "body": "Reduced query counts, used faster presets, and documented failure modes caused by large [B,Q,H,W,D] occupancy tensors."},
]

REPRO_COMMANDS = [
    "conda activate occformer307",
    "export PYTHONPATH=$(pwd):$PYTHONPATH",
    "sbatch scripts/run_experiment.sh baseline_fast",
    "python tools/test.py projects/configs/occformer_waymo/waymo_base.py results/baseline_fast/model/latest.pth --eval mIoU --launcher none",
    "python tools/build_portfolio_site.py --output-dir site",
    "python tools/validate_portfolio_site.py --site-dir site --strict",
]

SOURCE_MATRIX = [
    ("Hero", "Derived reel from report figures and experiment metrics", ["system_architecture", "occformer_framework", "miou_by_experiment", "per_class_baseline_fast"]),
    ("Method", "System diagrams from final report", ["system_architecture", "occformer_framework"]),
    ("Results", "Evaluation charts regenerated from report assets", ["miou_by_experiment", "per_class_all", "per_class_baseline_fast"]),
    ("Reproducibility", "Commands and links from report + repo", ["report_pdf"]),
]


@dataclass
class AssetRecord:
    key: str
    role: str
    title: str
    section: str
    source: str
    relative_path: str
    kind: str
    alt_text: str
    caption: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "path": self.relative_path,
            "kind": self.kind,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
        }


class BuildError(RuntimeError):
    pass


HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <meta name=\"description\" content=\"{description}\" />
  <link rel=\"stylesheet\" href=\"styles.css\" />
</head>
<body>
  <header class=\"hero\">
    <div class=\"hero-copy\">
      <p class=\"eyebrow\">Research portfolio · static GitHub Pages build</p>
      <h1>{title}</h1>
      <p class=\"lede\">{subtitle}</p>
      <p class=\"summary\">{summary}</p>
      <div class=\"actions\">
        <a class=\"button primary\" href=\"assets/docs/final-report.pdf\">Read final report</a>
        <a class=\"button\" href=\"https://github.com/Shawn-Kim96/OccFormerWithWaymoData\">Browse repository</a>
        <a class=\"button\" href=\"assets/data/portfolio-manifest.json\">Download manifest</a>
      </div>
      <ul class=\"stats\">{stats_cards}</ul>
    </div>
    <div class=\"hero-media\">
      <video autoplay muted loop playsinline controls poster=\"assets/media/hero-poster.jpg\">
        <source src=\"assets/media/hero-reel.mp4\" type=\"video/mp4\" />
      </video>
      <p class=\"caption\">{hero_caption}</p>
    </div>
  </header>

  <main>
    <section id=\"overview\" class=\"panel\">
      <div class=\"section-heading\">
        <p class=\"eyebrow\">Overview</p>
        <h2>Why this project matters</h2>
      </div>
      <div class=\"two-col\">
        <p>{overview_body}</p>
        <div class=\"source-list\">
          <h3>Grounded sources</h3>
          <ul>{overview_sources}</ul>
        </div>
      </div>
    </section>

    <section id=\"method\" class=\"panel\">
      <div class=\"section-heading\">
        <p class=\"eyebrow\">Method</p>
        <h2>OccFormer adapted to Waymo occupancy labels</h2>
      </div>
      <p>{method_summary}</p>
      <div class=\"media-grid\">{method_cards}</div>
    </section>

    <section id=\"results\" class=\"panel\">
      <div class=\"section-heading\">
        <p class=\"eyebrow\">Results</p>
        <h2>Quantitative signal from completed experiments</h2>
      </div>
      <p>{results_summary}</p>
      <div class=\"media-grid\">{results_cards}</div>
      <div class=\"table-wrap\">{results_table}</div>
    </section>

    <section id=\"engineering\" class=\"panel\">
      <div class=\"section-heading\">
        <p class=\"eyebrow\">Engineering lessons</p>
        <h2>What actually consumed the time</h2>
      </div>
      <p>{engineering_summary}</p>
      <div class=\"challenge-grid\">{challenge_cards}</div>
    </section>

    <section id=\"reproducibility\" class=\"panel\">
      <div class=\"section-heading\">
        <p class=\"eyebrow\">Reproducibility</p>
        <h2>How to rebuild the artifacts</h2>
      </div>
      <div class=\"two-col\">
        <div>
          <p>The site build is deterministic: report figures are promoted into the site bundle, the hero reel is generated from repo-backed images, and validation is performed from the emitted manifest.</p>
          <pre><code>{repro_commands}</code></pre>
        </div>
        <div class=\"source-list\">
          <h3>Source-to-section matrix</h3>
          <ul>{source_matrix}</ul>
        </div>
      </div>
    </section>
  </main>

  <footer class=\"footer\">
    <p>Generated {generated_at}. This static site is built from repo artifacts under <code>reports/</code>, <code>projects/configs/</code>, and <code>tools/</code>.</p>
  </footer>
</body>
</html>
"""

CSS_CONTENT = """
:root {
  color-scheme: dark;
  --bg: #07111f;
  --panel: rgba(9, 20, 36, 0.85);
  --panel-soft: rgba(20, 36, 59, 0.65);
  --border: rgba(143, 192, 255, 0.16);
  --text: #ebf2ff;
  --muted: #a0b6d6;
  --accent: #6fb5ff;
  --accent-strong: #95d0ff;
  --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
  --radius: 22px;
  --max-width: 1180px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  background:
    radial-gradient(circle at top left, rgba(35, 112, 215, 0.28), transparent 32%),
    radial-gradient(circle at top right, rgba(39, 196, 167, 0.18), transparent 28%),
    var(--bg);
  color: var(--text);
  line-height: 1.6;
}
a { color: inherit; }
.hero, main, .footer { width: min(calc(100% - 2rem), var(--max-width)); margin: 0 auto; }
.hero {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 1.5rem;
  padding: 3rem 0 2rem;
  align-items: center;
}
.panel, .hero-copy, .hero-media, .footer {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}
.hero-copy, .hero-media, .panel, .footer { padding: 1.5rem; }
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.75rem;
  color: var(--accent-strong);
  margin: 0 0 0.75rem;
}
h1, h2, h3 { line-height: 1.15; margin: 0 0 0.75rem; }
h1 { font-size: clamp(2.4rem, 5vw, 4.5rem); }
h2 { font-size: clamp(1.5rem, 2.4vw, 2.4rem); }
.lede { font-size: 1.15rem; color: var(--text); margin-top: 0; }
.summary, .caption, .source-list, .footer, .challenge-card p { color: var(--muted); }
.actions { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.5rem 0; }
.button {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.8rem 1.15rem;
  text-decoration: none;
  background: rgba(255, 255, 255, 0.03);
}
.button.primary { background: linear-gradient(135deg, var(--accent), #55e1ca); color: #08111b; font-weight: 700; }
.stats {
  list-style: none;
  padding: 0;
  margin: 1rem 0 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}
.stats li, .challenge-card, .media-card, .source-list, pre {
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: 18px;
}
.stats li { padding: 0.9rem 1rem; }
.stats strong { display: block; font-size: 1.35rem; }
.hero-media video, .media-card img { width: 100%; border-radius: 16px; display: block; }
.hero-media video { background: #02060a; aspect-ratio: 16 / 9; object-fit: cover; }
main { display: grid; gap: 1.25rem; padding-bottom: 2rem; }
.section-heading { margin-bottom: 1rem; }
.two-col {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: 1rem;
}
.media-grid, .challenge-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}
.media-card { overflow: hidden; }
.media-card .copy, .challenge-card { padding: 1rem; }
.media-card h3, .challenge-card h3 { margin-bottom: 0.4rem; }
.table-wrap { overflow-x: auto; margin-top: 1rem; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.8rem; border-bottom: 1px solid var(--border); text-align: left; }
pre { padding: 1rem; overflow-x: auto; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.source-list ul { margin: 0; padding-left: 1.15rem; }
.footer { margin-bottom: 2rem; }
@media (max-width: 900px) {
  .hero, .two-col, .media-grid, .challenge-grid { grid-template-columns: 1fr; }
  .stats { grid-template-columns: 1fr; }
}
""".strip()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _require_sources() -> None:
    missing = [str(path) for path in REQUIRED_SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise BuildError("Missing required source files:\n" + "\n".join(missing))


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _render_stat_cards(stats: list[tuple[str, str]]) -> str:
    return "".join(f"<li><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></li>" for label, value in stats)


def _render_asset_card(asset: AssetRecord) -> str:
    return textwrap.dedent(f"""
        <article class=\"media-card\">
          <img src=\"{html.escape(asset.relative_path)}\" alt=\"{html.escape(asset.alt_text)}\" loading=\"lazy\" />
          <div class=\"copy\">
            <h3>{html.escape(asset.title)}</h3>
            <p>{html.escape(asset.caption)}</p>
            <p><strong>Source:</strong> <code>{html.escape(asset.source)}</code></p>
          </div>
        </article>
    """).strip()


def _render_challenge_card(item: dict[str, str]) -> str:
    return f"<article class=\"challenge-card\"><h3>{html.escape(item['title'])}</h3><p>{html.escape(item['body'])}</p></article>"


def _render_results_table(summary_rows: list[dict[str, str]]) -> str:
    body = "".join(f"<tr><td>{html.escape(row['experiment'])}</td><td>{html.escape(row['mIoU'])}</td></tr>" for row in summary_rows)
    return f"<table><thead><tr><th>Experiment</th><th>mIoU</th></tr></thead><tbody>{body}</tbody></table>"


def _render_source_matrix() -> str:
    items = []
    for section_name, description, refs in SOURCE_MATRIX:
        items.append(f"<li><strong>{html.escape(section_name)}:</strong> {html.escape(description)} <br /><span>{html.escape(', '.join(refs))}</span></li>")
    return "".join(items)


def _load_slide_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        src_w, src_h = rgb.size
        scale = max(target_w / src_w, target_h / src_h)
        resized = rgb.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
        left = max((resized.width - target_w) // 2, 0)
        top = max((resized.height - target_h) // 2, 0)
        cropped = resized.crop((left, top, left + target_w, top + target_h))
        return cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)


def _overlay_copy(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    cv2.rectangle(overlay, (48, h - 200), (w - 48, h - 48), (4, 11, 22), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, title, (72, h - 136), cv2.FONT_HERSHEY_DUPLEX, 1.2, (237, 245, 255), 2, cv2.LINE_AA)
    for idx, line in enumerate(textwrap.wrap(subtitle, width=58)):
        cv2.putText(frame, line, (72, h - 96 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (205, 220, 242), 2, cv2.LINE_AA)
    return frame


def _write_hero_video(output_path: Path, poster_path: Path) -> dict[str, float | int]:
    slide_specs = [
        (REQUIRED_SOURCE_FILES["system_architecture"], "End-to-end pipeline", "Waymo sensor exports, occupancy labels, dataset hooks, training, evaluation, and demo generation in one loop."),
        (REQUIRED_SOURCE_FILES["occformer_framework"], "OccFormer core model", "Five-camera inputs are lifted into a voxel-aligned 3D volume and decoded into 16-class occupancy predictions."),
        (REQUIRED_SOURCE_FILES["miou_by_experiment"], "Completed experiment sweep", "baseline_fast leads the finished runs at 13.41 mIoU while operating on a 10% data slice for feasible iteration."),
        (REQUIRED_SOURCE_FILES["per_class_baseline_fast"], "Per-class behavior", "Road, walkable space, and vehicles carry most of the signal while rare classes stay difficult under constrained compute."),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, DEFAULT_VIDEO_FPS, DEFAULT_VIDEO_SIZE)
    total_frames = 0
    first_frame = None
    for slide_path, title, subtitle in slide_specs:
        base = _load_slide_image(slide_path, DEFAULT_VIDEO_SIZE)
        for index in range(DEFAULT_VIDEO_FPS * 2):
            alpha = index / max(DEFAULT_VIDEO_FPS * 2 - 1, 1)
            zoom = 1.0 + alpha * 0.05
            scaled = cv2.resize(base, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            crop_x = max((scaled.shape[1] - DEFAULT_VIDEO_SIZE[0]) // 2, 0)
            crop_y = max((scaled.shape[0] - DEFAULT_VIDEO_SIZE[1]) // 2, 0)
            frame = scaled[crop_y:crop_y + DEFAULT_VIDEO_SIZE[1], crop_x:crop_x + DEFAULT_VIDEO_SIZE[0]].copy()
            frame = _overlay_copy(frame, title, subtitle)
            if first_frame is None:
                first_frame = frame.copy()
            writer.write(frame)
            total_frames += 1
    writer.release()
    if first_frame is None:
        raise BuildError("Hero video generation produced no frames")
    Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)).save(poster_path, quality=92)
    return {"width": DEFAULT_VIDEO_SIZE[0], "height": DEFAULT_VIDEO_SIZE[1], "duration_seconds": round(total_frames / DEFAULT_VIDEO_FPS, 2), "size_bytes": output_path.stat().st_size}


def _find_latest_context_snapshot() -> str | None:
    context_dir = REPO_ROOT / ".omx" / "context"
    if not context_dir.is_dir():
        return None
    matches = sorted(context_dir.glob("occformer-portfolio-page-*.md"))
    return str(matches[-1].relative_to(REPO_ROOT)) if matches else None


def _build_manifest(output_dir: Path) -> dict[str, Any]:
    data_dir = output_dir / "assets" / "data"
    media_dir = output_dir / "assets" / "media"
    docs_dir = output_dir / "assets" / "docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for key in ["system_architecture", "occformer_framework", "miou_by_experiment", "per_class_all", "per_class_baseline_fast"]:
        src = REQUIRED_SOURCE_FILES[key]
        _copy_file(src, media_dir / src.name.lower().replace(" ", "-"))
    _copy_file(REQUIRED_SOURCE_FILES["report_pdf"], docs_dir / "final-report.pdf")

    hero_video_path = media_dir / "hero-reel.mp4"
    hero_poster_path = media_dir / "hero-poster.jpg"
    hero_meta = _write_hero_video(hero_video_path, hero_poster_path)

    summary_rows = _read_csv_rows(REPORT_DIR / "miou_summary.csv")
    per_class_rows = _read_csv_rows(REPORT_DIR / "iou_result.csv")
    experiment_rows = _read_csv_rows(REPORT_DIR / "experiment_settings.csv")
    best_row = max(summary_rows, key=lambda row: float(row["mIoU"]))

    asset_specs = []
    for key, role, title, section, caption in [
        ("system_architecture", "architecture_image", "System architecture", "method", "The repo's pipeline spans data alignment, dataset hooks, training/evaluation, and artifact generation."),
        ("occformer_framework", "framework_image", "OccFormer framework", "method", "OccFormer's view transformation and 3D reasoning stack, reused as the backbone of the Waymo adaptation."),
        ("miou_by_experiment", "experiment_chart", "mIoU by experiment", "results", "Completed runs show baseline_fast leading the available validation experiments."),
        ("per_class_all", "per_class_chart", "Per-class IoU across experiments", "results", "Cross-experiment comparison highlights class imbalance and geometry sensitivity."),
        ("per_class_baseline_fast", "per_class_detail_chart", "Per-class IoU for baseline_fast", "results", "The strongest completed run still shows the typical occupancy gap between common and rare classes."),
    ]:
        src = REQUIRED_SOURCE_FILES[key]
        rel = f"assets/media/{src.name.lower().replace(' ', '-')}"
        width, height = _image_size(output_dir / rel)
        asset_specs.append(AssetRecord(key=key, role=role, title=title, section=section, source=str(src.relative_to(REPO_ROOT)), relative_path=rel, kind="image", alt_text=title, caption=caption, width=width, height=height, size_bytes=(output_dir / rel).stat().st_size))

    asset_specs.append(AssetRecord(key="hero_reel", role="hero_video", title="Hero reel", section="hero", source="Derived from reports/final_report/figures/*", relative_path="assets/media/hero-reel.mp4", kind="video", alt_text="Hero video summarizing architecture and results slides", caption="A lightweight autoplay reel built from repo-backed report figures for GitHub Pages delivery.", width=int(hero_meta["width"]), height=int(hero_meta["height"]), duration_seconds=float(hero_meta["duration_seconds"]), size_bytes=int(hero_meta["size_bytes"])))
    asset_specs.append(AssetRecord(key="hero_poster", role="hero_poster", title="Hero poster", section="hero", source="Derived from reports/final_report/figures/system_architecture.png", relative_path="assets/media/hero-poster.jpg", kind="image", alt_text="Poster frame for the hero reel", caption="Poster frame for browsers or bandwidth conditions where autoplay is unavailable.", width=int(hero_meta["width"]), height=int(hero_meta["height"]), size_bytes=hero_poster_path.stat().st_size))
    asset_specs.append(AssetRecord(key="final_report_pdf", role="report_pdf", title="Final report PDF", section="reproducibility", source="reports/final_report/main.pdf", relative_path="assets/docs/final-report.pdf", kind="document", alt_text="Final report PDF", caption="The full write-up used as the narrative source for this portfolio page.", size_bytes=(docs_dir / "final-report.pdf").stat().st_size))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": {"name": "OccFormerWithWaymoData", "path": str(REPO_ROOT), "source_context": _find_latest_context_snapshot()},
        "project": {"title": SECTION_COPY["title"], "subtitle": SECTION_COPY["subtitle"], "summary": SECTION_COPY["summary"], "best_experiment": best_row, "experiments": summary_rows, "experiment_settings": experiment_rows},
        "source_matrix": [{"section": section, "description": description, "sources": refs} for section, description, refs in SOURCE_MATRIX],
        "required_sections": ["hero", "overview", "method", "results", "engineering", "reproducibility"],
        "assets": [asset.to_dict() for asset in asset_specs],
        "narrative": {"overview": SECTION_COPY["summary"], "method": SECTION_COPY["method_summary"], "results": SECTION_COPY["results_summary"], "engineering": SECTION_COPY["engineering_summary"], "repro_commands": REPRO_COMMANDS, "engineering_challenges": ENGINEERING_CHALLENGES, "per_class_rows_sample": per_class_rows[:18]},
    }
    (data_dir / "portfolio-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _asset_lookup(manifest: dict[str, Any], role: str) -> AssetRecord:
    for item in manifest["assets"]:
        if item["role"] == role:
            return AssetRecord(key=item["key"], role=item["role"], title=item["title"], section=item["section"], source=item["source"], relative_path=item["path"], kind=item["kind"], alt_text=item["alt_text"], caption=item["caption"], width=item.get("width"), height=item.get("height"), duration_seconds=item.get("duration_seconds"), size_bytes=item.get("size_bytes"))
    raise KeyError(f"Missing asset role: {role}")


def _render_html(manifest: dict[str, Any], output_dir: Path) -> None:
    experiments = manifest["project"]["experiments"]
    best = manifest["project"]["best_experiment"]
    stats = [("Best completed mIoU", best["mIoU"]), ("Finished experiment variants", str(len(experiments))), ("Camera views", "5"), ("Occupancy classes", "16"), ("Training data fraction", "10%"), ("GPU budget", "12 GB")]
    overview_sources = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in ["reports/final_report/main.tex", "projects/configs/occformer_waymo/experiments.py", "projects/configs/occformer_waymo/waymo_base.py", "reports/final_report/iou_result.csv"])
    method_cards = "".join(_render_asset_card(_asset_lookup(manifest, role)) for role in ["architecture_image", "framework_image"])
    results_cards = "".join(_render_asset_card(_asset_lookup(manifest, role)) for role in ["experiment_chart", "per_class_chart", "per_class_detail_chart"])
    challenge_cards = "".join(_render_challenge_card(item) for item in ENGINEERING_CHALLENGES)
    page_html = HTML_TEMPLATE.format(
        title=html.escape(SECTION_COPY["title"]),
        description=html.escape(SECTION_COPY["summary"]),
        subtitle=html.escape(SECTION_COPY["subtitle"]),
        summary=html.escape(SECTION_COPY["summary"]),
        stats_cards=_render_stat_cards(stats),
        hero_caption=html.escape(_asset_lookup(manifest, "hero_video").caption),
        overview_body=html.escape("The project turns a forked research codebase into a reproducible Waymo occupancy system. It packages the report's architecture, quantitative results, and engineering lessons into a static artifact that is easy to share, validate, and host on GitHub Pages."),
        overview_sources=overview_sources,
        method_summary=html.escape(SECTION_COPY["method_summary"]),
        method_cards=method_cards,
        results_summary=html.escape(SECTION_COPY["results_summary"]),
        results_cards=results_cards,
        results_table=_render_results_table(experiments),
        engineering_summary=html.escape(SECTION_COPY["engineering_summary"]),
        challenge_cards=challenge_cards,
        repro_commands=html.escape("\n".join(REPRO_COMMANDS)),
        source_matrix=_render_source_matrix(),
        generated_at=html.escape(manifest["generated_at"]),
    )
    (output_dir / "index.html").write_text(page_html)
    (output_dir / "styles.css").write_text(CSS_CONTENT + "\n")
    (output_dir / ".nojekyll").write_text("\n")


def build_site(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    _require_sources()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(output_dir)
    _render_html(manifest, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where the static site will be written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_site(args.output_dir)
    print(f"Built site at {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'assets/data/portfolio-manifest.json'}")
    print(f"Best experiment: {manifest['project']['best_experiment']['experiment']} ({manifest['project']['best_experiment']['mIoU']} mIoU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
