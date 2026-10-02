# Copyright (c) 2026 liblaf
# ruff: noqa: RUF001
"""Compose a readable preview from decoded animation keyframes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    render: Path = Path("92-smile-mouthopen-transition-render")
    output: Path = Path("95-transition-preview")


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/TTF/DejaVuSans{suffix}.ttf", size)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.render / "manifest.json")
    manifest = json.loads(source.read_text())
    video = Path(manifest["video"]["path"])
    assert record(video)["sha256"] == manifest["video"]["sha256"]
    out = cherries.output(cfg.output)
    out.mkdir(exist_ok=False)
    video_indices = [0, 59, 89, 119, 178]
    state_indices = [0, 30, 60, 90, 120]
    selection = "+".join(f"eq(n\\,{i})" for i in video_indices)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "vfr",
            str(out / "keyframe-%02d.png"),
        ],
        check=True,
    )
    page = Image.new("RGB", (2048, 1024), "#101315")
    draw = ImageDraw.Draw(page)
    draw.text(
        (1024, 22),
        "Smile → MouthOpen",
        font=font(40, bold=True),
        fill="white",
        anchor="mt",
    )
    draw.text(
        (1024, 72),
        "Activation tensors and prescribed jaw pose transition together",
        font=font(23),
        fill="#dddddd",
        anchor="mt",
    )
    draw.text((24, 123), "Full tetmesh shape", font=font(23, bold=True), fill="white")
    draw.text(
        (24, 530), "Principal activation mode", font=font(23, bold=True), fill="white"
    )
    for column, index in enumerate(state_indices):
        alpha = (1 - np.cos(np.pi * index / 120)) / 2
        label = f"α = {alpha:.3f}"
        if index == 0:
            label = "Smile · α = 0"
        elif index == 120:
            label = "MouthOpen · α = 1"
        x = 24 + column * 400
        draw.text(
            (x + 192, 158), label, font=font(22, bold=True), fill="#eeeeee", anchor="mt"
        )
        with Image.open(out / f"keyframe-{column + 1:02d}.png") as frame:
            for left, y in ((0, 191), (960, 570)):
                panel = frame.crop((left, 145, left + 960, 925)).resize(
                    (384, 312), Image.Resampling.LANCZOS
                )
                page.paste(panel, (x, y))
    draw.text(
        (24, 910),
        "121 re-equilibrated states · Full tensors drive the shape; the strongest mode is shown.",
        font=font(21),
        fill="#dddddd",
    )
    draw.text(
        (24, 946),
        "Contact off; limited inversions and boundary intersections remain. Mechanical validity is not established.",
        font=font(19),
        fill="#bbbbbb",
    )
    image = out / "smile-to-mouthopen-preview.png"
    page.save(image)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "video": record(video),
                "render_manifest": record(source),
                "video_frame_indices": video_indices,
                "state_indices": state_indices,
                "preview": record(image),
                "source": record(Path(__file__)),
                "method": "Decoded MP4 frames, panel crops and new labels; no new physics or geometry interpolation",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
