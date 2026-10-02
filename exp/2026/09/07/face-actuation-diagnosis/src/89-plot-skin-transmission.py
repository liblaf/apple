#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, RUF001, TRY003
"""Compose a 2 by 2 mouth comparison from exact rendered endpoint panels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "data/89-skin-transmission-viewer"
PNG = VIEWER / "skin-transmission-mouth-2x2.png"
PDF = VIEWER / "skin-transmission-mouth-2x2.pdf"
RECEIPT = VIEWER / "skin-transmission-mouth-2x2-receipt.json"

CASES = (
    ("baseline_skin_0", "Unchanged field · skin factor 0.00"),
    ("baseline_skin_0.12", "Unchanged field · skin factor 0.12"),
    ("strong_skin_0", "Strong field diffusion · skin factor 0.00"),
    ("strong_skin_0.12", "Strong field diffusion · skin factor 0.12"),
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
    """Load a bundled system font with the needed Unicode glyphs."""
    return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)


def centered(
    draw: ImageDraw.ImageDraw, text: str, y: int, text_font: ImageFont.FreeTypeFont
) -> None:
    """Draw centered text across the two-column canvas."""
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text(((1800 - (box[2] - box[0])) / 2, y), text, fill="#172321", font=text_font)


def main() -> None:
    """Crop only endpoint halves and compose them without resampling."""
    if PNG.exists() or PDF.exists() or RECEIPT.exists():
        raise FileExistsError("refusing to replace an existing 2 by 2 comparison")

    source_receipts = []
    endpoint_panels = []
    camera = None
    for case_id, label in CASES:
        image_path = VIEWER / case_id / "mouth-closeup.png"
        receipt_path = VIEWER / case_id / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        image_receipt = digest(image_path)
        if image_receipt != receipt["outputs"]["mouth-closeup.png"]:
            raise ValueError(f"render receipt mismatch: {image_path}")
        if image_receipt["bytes"] <= 0:
            raise ValueError(f"empty render: {image_path}")
        with Image.open(image_path) as source:
            if source.size != RENDER_SIZE:
                raise ValueError(f"unexpected render dimensions: {source.size}")
            endpoint_panels.append(source.convert("RGB").crop(ENDPOINT_CROP))
        if camera is None:
            camera = receipt["mouth_closeup_camera"]
        elif receipt["mouth_closeup_camera"] != camera:
            raise ValueError("mouth cameras differ across the four cases")
        source_receipts.append(
            {
                "id": case_id,
                "label": label,
                "image": image_receipt,
                "case_receipt": digest(receipt_path),
            }
        )

    title_height = 104
    caption_height = 58
    footer_height = 68
    panel_width, panel_height = endpoint_panels[0].size
    canvas = Image.new(
        "RGB",
        (
            2 * panel_width,
            title_height + 2 * (caption_height + panel_height) + footer_height,
        ),
        "#f7f6f2",
    )
    draw = ImageDraw.Draw(canvas)
    centered(draw, "Skin transmission × activation-field diffusion", 22, font(38))

    y = title_height
    for row in range(2):
        for column in range(2):
            index = row * 2 + column
            x = column * panel_width
            label = CASES[index][1]
            draw.rectangle((x, y, x + panel_width, y + caption_height), fill="#f7f6f2")
            draw.text((x + 22, y + 13), label, fill="#172321", font=font(25))
            canvas.paste(endpoint_panels[index], (x, y + caption_height))
        draw.line(
            (panel_width, y, panel_width, y + caption_height + panel_height),
            fill="#172321",
            width=2,
        )
        y += caption_height + panel_height
        if row == 0:
            draw.line((0, y, 2 * panel_width, y), fill="#172321", width=2)

    footer = "Exact saved endpoint geometry · identical camera · true scale · no geometry processing"
    centered(draw, footer, y + 18, font(23))
    canvas.save(PNG, optimize=True)
    canvas.save(
        PDF,
        "PDF",
        resolution=150.0,
        title="Skin transmission by activation-field diffusion",
        subject="Exact saved mouth endpoint comparison at shared camera and true scale",
        creator="89-plot-skin-transmission.py",
    )

    payload = {
        "schema_version": 1,
        "scope": "Raster-only 2 by 2 composition of exact endpoint halves from the four shared-camera mouth renders",
        "source_script": digest(Path(__file__)),
        "source_render_size_px": list(RENDER_SIZE),
        "endpoint_crop_px": list(ENDPOINT_CROP),
        "resampling": "none",
        "geometry_processing": "none",
        "mouth_closeup_camera": camera,
        "sources": source_receipts,
        "outputs": {"png": digest(PNG), "pdf": digest(PDF)},
    }
    RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
