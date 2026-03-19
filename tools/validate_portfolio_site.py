#!/usr/bin/env python3
"""Validate the static portfolio site, generated assets, and optional deployed Pages URL."""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import cv2
from PIL import Image


REQUIRED_ASSET_IDS = {
    "hero-video",
    "system-architecture",
    "occformer-framework",
    "miou-experiments",
    "per-class-baseline",
}
REQUIRED_SECTION_IDS = {
    "overview",
    "method",
    "results",
    "engineering",
    "reproducibility",
}


class RefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.section_ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("src", "href", "poster"):
            value = attrs.get(key)
            if value:
                self.refs.append(value)
        if tag == "section" and attrs.get("id"):
            self.section_ids.add(attrs["id"])


def local_ref_to_path(site_dir: Path, ref: str) -> Path | None:
    if ref.startswith(("http://", "https://", "mailto:", "javascript:", "#")):
        return None
    clean = ref.split("?", 1)[0].split("#", 1)[0]
    return (site_dir / clean).resolve()


def check_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            return False, "zero dimensions"
        return True, f"{width}x{height}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def check_video(path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "could not open"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "no decodable first frame"
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return False, "invalid metadata"
        return True, f"{width}x{height}, {frame_count / fps:.2f}s"
    finally:
        cap.release()


def fetch_url(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 400, f"HTTP {status}"
    except urllib.error.URLError as exc:
        return False, str(exc)


def score_categories(manifest: dict, parser: RefParser, broken_refs: list[str], asset_failures: list[str]) -> dict[str, int]:
    assets = manifest["assets"]
    asset_ids = {asset["id"] for asset in assets}
    clarity = 5 if manifest.get("highlights") and len(assets) >= 7 else 3
    correctness = 5 if not broken_refs and not asset_failures and REQUIRED_ASSET_IDS.issubset(asset_ids) else 2
    coverage = 5 if REQUIRED_SECTION_IDS.issubset(parser.section_ids | {section["id"] for section in manifest.get("sections", [])}) else 2
    usefulness = 5 if manifest.get("repro_steps") and manifest.get("engineering_lessons") else 3
    return {
        "clarity": clarity,
        "correctness": correctness,
        "coverage": coverage,
        "usefulness": usefulness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", default="site", help="Static site directory")
    parser.add_argument("--deployed-url", default=None, help="Optional deployed Pages URL to smoke-check")
    args = parser.parse_args()

    site_dir = Path(args.site_dir).resolve()
    index_path = site_dir / "index.html"
    manifest_path = site_dir / "assets" / "generated" / "portfolio-manifest.json"
    report_json = site_dir / "assets" / "generated" / "validation-report.json"
    report_md = site_dir / "assets" / "generated" / "validation-report.md"

    html = index_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ref_parser = RefParser()
    ref_parser.feed(html)

    broken_refs: list[str] = []
    for ref in ref_parser.refs:
        path = local_ref_to_path(site_dir, ref)
        if path is None:
            continue
        if not path.exists():
            broken_refs.append(ref)

    asset_failures: list[str] = []
    asset_results: list[dict[str, str | bool]] = []
    for asset in manifest.get("assets", []):
        asset_path = (site_dir.parent / asset["src"]).resolve()
        if not asset_path.exists():
            asset_failures.append(f"missing: {asset['id']}")
            asset_results.append({"id": asset["id"], "ok": False, "detail": "missing"})
            continue
        if asset["kind"] == "video":
            ok, detail = check_video(asset_path)
        else:
            ok, detail = check_image(asset_path)
        asset_results.append({"id": asset["id"], "ok": ok, "detail": detail})
        if not ok:
            asset_failures.append(f"{asset['id']}: {detail}")

    section_ids = {section["id"] for section in manifest.get("sections", [])}
    missing_asset_ids = sorted(REQUIRED_ASSET_IDS - {asset["id"] for asset in manifest.get("assets", [])})
    missing_sections = sorted(REQUIRED_SECTION_IDS - section_ids)

    categories = score_categories(manifest, ref_parser, broken_refs, asset_failures)
    composite_score = int(round((sum(categories.values()) / 20) * 100))

    deployed_checks = []
    deployed_failures = []
    if args.deployed_url:
        base = args.deployed_url.rstrip("/") + "/"
        for suffix in ["", "assets/generated/portfolio-manifest.json"]:
            ok, detail = fetch_url(base + suffix)
            deployed_checks.append({"url": base + suffix, "ok": ok, "detail": detail})
            if not ok:
                deployed_failures.append(f"{base + suffix}: {detail}")

    passed = not any([
        missing_asset_ids,
        missing_sections,
        broken_refs,
        asset_failures,
        deployed_failures,
        composite_score < 85,
        any(value < 3 for value in categories.values()),
    ])

    report = {
        "passed": passed,
        "technical_threshold": {
            "missing_required_assets": missing_asset_ids,
            "missing_required_sections": missing_sections,
            "broken_local_refs": broken_refs,
            "asset_failures": asset_failures,
            "deployed_failures": deployed_failures,
        },
        "quality_threshold": {
            "composite_score": composite_score,
            "categories": categories,
        },
        "asset_results": asset_results,
        "deployed_checks": deployed_checks,
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Portfolio Validation Report",
        "",
        f"- Passed: {'YES' if passed else 'NO'}",
        f"- Composite score: {composite_score}/100",
        f"- Category scores: {categories}",
        f"- Missing required assets: {missing_asset_ids or 'none'}",
        f"- Missing required sections: {missing_sections or 'none'}",
        f"- Broken local refs: {broken_refs or 'none'}",
        f"- Asset failures: {asset_failures or 'none'}",
        f"- Deployed failures: {deployed_failures or 'none'}",
        "",
        "## Asset checks",
    ]
    for item in asset_results:
        lines.append(f"- {'PASS' if item['ok'] else 'FAIL'} `{item['id']}` — {item['detail']}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
