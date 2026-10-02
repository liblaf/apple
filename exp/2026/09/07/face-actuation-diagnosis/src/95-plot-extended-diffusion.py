#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003
"""Compose six exact saved field-diffusion endpoints at one shared scale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "data/95-extended-diffusion-viewer"
BASELINE_SUMMARY = ROOT / "data/80-forward-field-diffusion/summary.json"
EXTENDED_SUMMARY = ROOT / "data/94-extended-field-diffusion/summary.json"
SURFACE_SUMMARY = ROOT / "data/95-extended-diffusion-surface/summary.json"
FULL_PNG = VIEWER / "extended-diffusion-full-head-comparison.png"
FULL_PDF = VIEWER / "extended-diffusion-full-head-comparison.pdf"
MOUTH_PNG = VIEWER / "extended-diffusion-mouth-comparison.png"
MOUTH_PDF = VIEWER / "extended-diffusion-mouth-comparison.pdf"
RECEIPT = VIEWER / "extended-diffusion-comparison-receipt.json"

CASES = (
    ("baseline", "unchanged"),
    ("mild", "diffused"),
    ("strong", "diffused"),
    ("r010", "diffused"),
    ("r001", "diffused"),
    ("component_constant", "component-constant limit"),
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
    """Load a system font with scientific Unicode glyphs."""
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
        ((width - (box[2] - box[0])) / 2, y),
        text,
        fill="#172321",
        font=text_font,
    )


def achieved_ratio(case: dict[str, Any], source_roughness: float) -> float:
    """Return the producer-recorded achieved roughness ratio."""
    field = case.get("field")
    if isinstance(field, dict) and "roughness_ratio" in field:
        return float(field["roughness_ratio"])
    return float(case["roughness"]) / source_roughness


def compose(
    panels: list[Image.Image],
    sources: list[dict[str, Any]],
    title: str,
    png: Path,
    pdf: Path,
) -> None:
    """Compose one 2x3 view from exact endpoint crops without resampling."""
    columns, rows = 3, 2
    title_height = 100
    caption_height = 112
    footer_height = 130
    panel_width, panel_height = panels[0].size
    width = columns * panel_width
    height = title_height + rows * (caption_height + panel_height) + footer_height
    canvas = Image.new("RGB", (width, height), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    centered(draw, title, width, 22, font(38))

    for index, (panel, source) in enumerate(zip(panels, sources, strict=True)):
        column, row = index % columns, index // columns
        x = column * panel_width
        caption_y = title_height + row * (caption_height + panel_height)
        surface = source["surface_case"]["surface"]
        expression = surface["expression"]
        hp5 = surface["scales"]["5mm"]["normal_displacement_highpass"]["mouth_10mm"][
            "rms_mm"
        ]
        case_title = (
            f"R/R₀ = {source['achieved_roughness_ratio']:.6f} · {source['short_label']}"
        )
        detail = (
            f"motion {expression['displacement_rms_mm']:.3f} mm · "
            f"fit residual {expression['residual_rms_mm']:.3f} mm · "
            f"mouth HP₅ {hp5:.3f} mm"
        )
        draw.text((x + 20, caption_y + 12), case_title, fill="#172321", font=font(27))
        draw.text((x + 20, caption_y + 58), detail, fill="#33413e", font=font(20))
        canvas.paste(panel, (x, caption_y + caption_height))
        if column:
            draw.line(
                (x, caption_y, x, caption_y + caption_height + panel_height),
                fill="#172321",
                width=2,
            )
        if row:
            draw.line(
                (x, caption_y, x + panel_width, caption_y), fill="#172321", width=2
            )

    footer_y = title_height + rows * (caption_height + panel_height)
    centered(
        draw,
        "One Raw6 source field · one displacement seed · independent equilibria · shared camera · true scale",
        width,
        footer_y + 18,
        font(24),
    )
    centered(
        draw,
        "HP₅ = 5 mm-scale rest-normal displacement high-pass RMS within the intrinsic 10 mm mouth region",
        width,
        footer_y + 59,
        font(21),
    )
    centered(
        draw,
        "Smile target is post-hoc context; no output geometry smoothing or deformation exaggeration",
        width,
        footer_y + 94,
        font(21),
    )
    canvas.save(png, optimize=True)
    canvas.save(
        pdf,
        "PDF",
        resolution=150.0,
        title=title,
        subject="Six exact saved face equilibria at shared true scale",
        creator="95-plot-extended-diffusion.py",
    )


def main() -> None:
    """Validate sources and compose exact endpoint crops without resampling."""
    outputs = (FULL_PNG, FULL_PDF, MOUTH_PNG, MOUTH_PDF, RECEIPT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to replace an existing diffusion comparison")

    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    extended = json.loads(EXTENDED_SUMMARY.read_text(encoding="utf-8"))
    surface_summary = json.loads(SURFACE_SUMMARY.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or extended.get("status") != "completed":
        raise ValueError("both forward producers must have completed successfully")
    source_roughness = float(baseline["field_preparation"]["source_roughness"])
    producer_cases = {
        **{case["id"]: case for case in baseline["cases"]},
        **{case["id"]: case for case in extended["cases"]},
    }
    surface_cases = {case["id"]: case for case in surface_summary["cases"]}
    expected = tuple(case_id for case_id, _ in CASES)
    if tuple(producer_cases) != expected or tuple(surface_cases) != expected:
        raise ValueError("producer and surface case ordering must match the figure")

    full_panels: list[Image.Image] = []
    mouth_panels: list[Image.Image] = []
    sources: list[dict[str, Any]] = []
    cameras: dict[str, Any] = {"full_head": None, "mouth_closeup": None}
    for case_id, short_label in CASES:
        producer = producer_cases[case_id]
        if producer.get("status") != "equilibrium_valid":
            raise ValueError(f"invalid equilibrium cannot be plotted: {case_id}")
        surface = surface_cases[case_id]
        case_receipt_path = VIEWER / case_id / "receipt.json"
        case_receipt = json.loads(case_receipt_path.read_text(encoding="utf-8"))
        image_receipts: dict[str, Any] = {}
        for filename, panels in (
            ("full-head.png", full_panels),
            ("mouth-closeup.png", mouth_panels),
        ):
            image_path = VIEWER / case_id / filename
            image_receipt = digest(image_path)
            if image_receipt != case_receipt["outputs"][filename]:
                raise ValueError(f"render receipt mismatch: {image_path}")
            with Image.open(image_path) as image:
                if image.size != RENDER_SIZE:
                    raise ValueError(f"unexpected render dimensions: {image.size}")
                panels.append(image.convert("RGB").crop(ENDPOINT_CROP))
            image_receipts[filename] = image_receipt
        endpoint = case_receipt["sources"]["endpoint"]
        producer_endpoint = producer["mesh"]
        if endpoint["sha256"] != producer_endpoint["sha256"]:
            raise ValueError(f"render/producer endpoint mismatch: {case_id}")
        if endpoint["sha256"] != surface["endpoint"]["sha256"]:
            raise ValueError(f"render/surface endpoint mismatch: {case_id}")
        for key, receipt_key in (
            ("full_head", "full_head_camera"),
            ("mouth_closeup", "mouth_closeup_camera"),
        ):
            if cameras[key] is None:
                cameras[key] = case_receipt[receipt_key]
            elif case_receipt[receipt_key] != cameras[key]:
                raise ValueError(f"{key} cameras differ across cases")
        sources.append(
            {
                "id": case_id,
                "short_label": short_label,
                "achieved_roughness_ratio": achieved_ratio(producer, source_roughness),
                "producer_case": producer,
                "surface_case": surface,
                "images": image_receipts,
                "case_receipt": digest(case_receipt_path),
            }
        )

    compose(
        full_panels,
        sources,
        "Conservative field diffusion · full-head exact saved equilibria",
        FULL_PNG,
        FULL_PDF,
    )
    compose(
        mouth_panels,
        sources,
        "Conservative field diffusion · mouth close-up at shared true scale",
        MOUTH_PNG,
        MOUTH_PDF,
    )

    receipt = {
        "schema_version": 1,
        "scope": (
            "Raster-only 2x3 compositions of exact endpoint halves from shared-camera "
            "full-head and mouth-closeup renders"
        ),
        "source_script": digest(Path(__file__)),
        "producer_summaries": [digest(BASELINE_SUMMARY), digest(EXTENDED_SUMMARY)],
        "surface_summary": digest(SURFACE_SUMMARY),
        "source_render_size_px": list(RENDER_SIZE),
        "endpoint_crop_px": list(ENDPOINT_CROP),
        "resampling": "none",
        "geometry_processing": "none",
        "cameras": cameras,
        "sources": sources,
        "outputs": {
            "full_head_png": digest(FULL_PNG),
            "full_head_pdf": digest(FULL_PDF),
            "mouth_png": digest(MOUTH_PNG),
            "mouth_pdf": digest(MOUTH_PDF),
        },
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
