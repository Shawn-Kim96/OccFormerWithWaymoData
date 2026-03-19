#!/usr/bin/env python3
"""Validate the generated OccFormer portfolio site."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

REQUIRED_SECTIONS = ["hero", "overview", "method", "results", "engineering", "reproducibility"]
REQUIRED_ROLES = ["hero_video", "architecture_image", "framework_image", "experiment_chart", "per_class_chart", "report_pdf"]
LINK_RE = re.compile(r"(?:src|href)=\"([^\"]+)\"")


class ValidationError(RuntimeError):
    pass


def _load_manifest(site_dir: Path) -> dict[str, Any]:
    path = site_dir / "assets" / "data" / "portfolio-manifest.json"
    if not path.exists():
        raise ValidationError(f"Manifest missing: {path}")
    return json.loads(path.read_text())


def _validate_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValidationError(f"Invalid image dimensions for {path}")
        return {"width": width, "height": height}


def _validate_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValidationError(f"Video did not open: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValidationError(f"Video has no decodable first frame: {path}")
    if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
        raise ValidationError(f"Video metadata invalid: {path}")
    if float(frame.mean()) < 3.0:
        raise ValidationError(f"Video first frame appears blank: {path}")
    return {"width": width, "height": height, "frame_count": frame_count, "fps": fps, "duration_seconds": round(frame_count / fps, 2)}


def _resolve_local_links(site_dir: Path, html_path: Path) -> list[str]:
    missing = []
    for link in LINK_RE.findall(html_path.read_text()):
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not (html_path.parent / link).resolve().exists():
            missing.append(link)
    return missing


def validate_site(site_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(site_dir)
    index_path = site_dir / "index.html"
    styles_path = site_dir / "styles.css"
    if not index_path.exists():
        raise ValidationError(f"Missing index.html in {site_dir}")
    if not styles_path.exists():
        raise ValidationError(f"Missing styles.css in {site_dir}")

    assets = manifest.get("assets", [])
    assets_by_role = {asset["role"]: asset for asset in assets}
    missing_roles = [role for role in REQUIRED_ROLES if role not in assets_by_role]
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in manifest.get("required_sections", [])]

    checks, missing_targets, corrupt_assets = [], [], []
    for asset in assets:
        asset_path = site_dir / asset["path"]
        if not asset_path.exists():
            missing_targets.append(asset["path"])
            continue
        try:
            if asset["kind"] == "image":
                meta = _validate_image(asset_path)
            elif asset["kind"] == "video":
                meta = _validate_video(asset_path)
            else:
                meta = {"size_bytes": asset_path.stat().st_size}
            checks.append({"path": asset["path"], "kind": asset["kind"], **meta})
        except ValidationError as exc:
            corrupt_assets.append(str(exc))

    broken_local_links = _resolve_local_links(site_dir, index_path)
    technical_pass = not (missing_roles or missing_sections or missing_targets or corrupt_assets or broken_local_links)
    technical_score = max(0, 100 - 8 * len(missing_roles) - 8 * len(missing_sections) - 10 * len(missing_targets) - 12 * len(corrupt_assets) - 10 * len(broken_local_links))

    quality_categories = {
        "clarity": 5 if len(manifest.get("narrative", {}).get("engineering_challenges", [])) >= 4 else 3,
        "correctness": 5 if technical_pass else max(1, 4 - len(corrupt_assets) - len(broken_local_links)),
        "coverage": 5 if not missing_roles and not missing_sections else max(1, 5 - len(missing_roles) - len(missing_sections)),
        "usefulness": 5 if len(manifest.get("narrative", {}).get("repro_commands", [])) >= 4 else 3,
    }
    quality_score = round(sum(quality_categories.values()) / (len(quality_categories) * 5) * 100, 2)
    composite_score = round(technical_score * 0.6 + quality_score * 0.4, 2)

    return {
        "site_dir": str(site_dir),
        "technical": {"pass": technical_pass, "score": technical_score, "missing_roles": missing_roles, "missing_sections": missing_sections, "missing_targets": missing_targets, "corrupt_assets": corrupt_assets, "broken_local_links": broken_local_links},
        "quality": {"score": quality_score, "categories": quality_categories},
        "composite_score": composite_score,
        "threshold_pass": technical_pass and composite_score >= 85 and min(quality_categories.values()) >= 3,
        "checks": checks,
    }


def write_reports(site_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    data_dir = site_dir / "assets" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "validator-report.json"
    md_path = data_dir / "validator-report.md"
    json_path.write_text(json.dumps(report, indent=2))
    lines = ["# Portfolio Validator Report", "", f"- Technical pass: {'PASS' if report['technical']['pass'] else 'FAIL'}", f"- Technical score: {report['technical']['score']}", f"- Quality score: {report['quality']['score']}", f"- Composite score: {report['composite_score']}", f"- Threshold pass: {'PASS' if report['threshold_pass'] else 'FAIL'}", "", "## Quality Categories"]
    for key, value in report["quality"]["categories"].items():
        lines.append(f"- {key}: {value}/5")
    lines.extend(["", "## Technical Issues", f"- Missing roles: {report['technical']['missing_roles'] or 'none'}", f"- Missing sections: {report['technical']['missing_sections'] or 'none'}", f"- Missing targets: {report['technical']['missing_targets'] or 'none'}", f"- Corrupt assets: {report['technical']['corrupt_assets'] or 'none'}", f"- Broken local links: {report['technical']['broken_local_links'] or 'none'}"])
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_site(args.site_dir)
    json_path, md_path = write_reports(args.site_dir, report)
    print(json.dumps(report, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if (not args.strict or report["threshold_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
