#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003
"""Compose three exact saved active-tension face endpoint panels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "data/91-active-tension-viewer"
OUTCOME = ROOT / "data/93-active-tension-face-outcome.json"
PNG = VIEWER / "active-tension-face-comparison.png"
PDF = VIEWER / "active-tension-face-comparison.pdf"
RECEIPT = VIEWER / "active-tension-face-comparison-receipt.json"

CASES = (
    ("gain-0000", "Gain 0 · T = 0 MPa · rest control"),
    ("gain-0100", "Gain 1 · T = 0.0246575 MPa"),
    ("gain-0300", "Gain 3 · T = 0.0739726 MPa"),
)
RENDER_SIZE = (1800, 1325)
ENDPOINT_CROP = (900, 0, 1800, 1200)


def digest(path: Path) -> dict[str, str | int]:
    """Return a streaming content receipt."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def font(size: int) -> ImageFont.FreeTypeFont:
    """Load a system font with the needed Unicode glyphs."""
    return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)


def centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    y: int,
    text_font: ImageFont.FreeTypeFont,
) -> None:
    """Draw text centered across the canvas width."""
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, y), text, fill="#172321", font=text_font
    )


def main() -> None:
    """Crop exact endpoint halves and compose them without resampling."""
    if PNG.exists() or PDF.exists() or RECEIPT.exists():
        raise FileExistsError(
            "refusing to replace an existing active-tension comparison"
        )

    outcome = json.loads(OUTCOME.read_text(encoding="utf-8"))
    accepted = {row["label"]: row for row in outcome["accepted_cases"]}
    failed = {row["label"]: row for row in outcome["failed_cases"]}
    if tuple(accepted) != tuple(case_id for case_id, _ in CASES):
        raise ValueError("producer accepted-case ordering differs from the figure")
    failure = failed.get("gain-1000")
    if failure is None or not failure["absent_state_artifacts"]:
        raise ValueError("gain 10 must remain an explicit absent-state failure")

    panels = []
    source_receipts = []
    common_camera = None
    for case_id, label in CASES:
        image_path = VIEWER / case_id / "full-head.png"
        case_receipt_path = VIEWER / case_id / "receipt.json"
        case_receipt = json.loads(case_receipt_path.read_text(encoding="utf-8"))
        image_receipt = digest(image_path)
        if image_receipt != case_receipt["outputs"]["full-head.png"]:
            raise ValueError(f"render receipt mismatch: {image_path}")
        if (
            case_receipt["sources"]["endpoint"]["sha256"]
            != accepted[case_id]["artifacts"]["state.vtu"]["sha256"]
        ):
            raise ValueError(f"producer endpoint mismatch: {case_id}")
        with Image.open(image_path) as source:
            if source.size != RENDER_SIZE:
                raise ValueError(f"unexpected render dimensions: {source.size}")
            panels.append(source.convert("RGB").crop(ENDPOINT_CROP))
        if common_camera is None:
            common_camera = case_receipt["full_head_camera"]
        elif case_receipt["full_head_camera"] != common_camera:
            raise ValueError("full-head cameras differ across accepted cases")
        source_receipts.append(
            {
                "id": case_id,
                "label": label,
                "image": image_receipt,
                "case_receipt": digest(case_receipt_path),
                "endpoint": case_receipt["sources"]["endpoint"],
            }
        )

    title_height = 100
    caption_height = 86
    footer_height = 100
    panel_width, panel_height = panels[0].size
    width = len(panels) * panel_width
    height = title_height + caption_height + panel_height + footer_height
    canvas = Image.new("RGB", (width, height), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    centered(
        draw, "Manual active tension · exact saved face equilibria", width, 22, font(38)
    )

    for index, ((case_id, label), panel) in enumerate(zip(CASES, panels, strict=True)):
        x = index * panel_width
        motion = accepted[case_id]["motion"]
        draw.text((x + 20, title_height + 9), label, fill="#172321", font=font(25))
        detail = (
            f"surface motion {motion['surface_rms_mm']:.3f} mm · "
            f"post-hoc Smile projection {motion['smile_projection']:.3f}"
        )
        draw.text((x + 20, title_height + 47), detail, fill="#33413e", font=font(20))
        canvas.paste(panel, (x, title_height + caption_height))
        if index:
            draw.line(
                (x, title_height, x, title_height + caption_height + panel_height),
                fill="#172321",
                width=2,
            )

    footer_y = title_height + caption_height + panel_height
    centered(
        draw,
        "Shared camera · true scale · no geometry processing · Smile target used only for post-hoc comparison",
        width,
        footer_y + 15,
        font(23),
    )
    centered(
        draw,
        "Gain 10 omitted: fixed 10,000-step forward budget exhausted; no accepted saved state exists",
        width,
        footer_y + 52,
        font(21),
    )
    canvas.save(PNG, optimize=True)
    canvas.save(
        PDF,
        "PDF",
        resolution=150.0,
        title="Manual active tension face comparison",
        subject="Exact saved target-independent active-tension equilibria at shared true scale",
        creator="91-plot-active-tension.py",
    )

    receipt = {
        "schema_version": 1,
        "scope": "Raster-only three-column composition of exact endpoint halves from shared-camera full-head renders",
        "source_script": digest(Path(__file__)),
        "producer_outcome": digest(OUTCOME),
        "source_render_size_px": list(RENDER_SIZE),
        "endpoint_crop_px": list(ENDPOINT_CROP),
        "resampling": "none",
        "geometry_processing": "none",
        "full_head_camera": common_camera,
        "sources": source_receipts,
        "omitted_case": {
            "id": failure["label"],
            "status": failure["status"],
            "reason": failure["failure_reason"],
            "failure_artifact": failure["artifacts"]["failure.json"],
            "placeholder_geometry": False,
        },
        "outputs": {"png": digest(PNG), "pdf": digest(PDF)},
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
