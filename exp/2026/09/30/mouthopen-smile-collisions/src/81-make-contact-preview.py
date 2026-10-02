# ruff: noqa: C901, E402, EM101, EM102, PLR0915, TRY003
"""Make a compact contact sheet from the completed fixed-activation movie."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
sys.path.insert(0, str(STRESS_SRC))
from experiment import Profile

LOG = logging.getLogger(__name__)
FPS = 30
STATE_COUNT = 121
VIDEO_FRAMES = 179
STATE_INDICES = (0, 30, 60, 90, 120)
VIDEO_INDICES = tuple(0 if i == 0 else 29 + i for i in STATE_INDICES)
PAGE_SIZE = (2400, 960)
PANEL_SIZE = (800, 325)
PANEL_CROP = (0, 145, 1920, 925)
BACKGROUND = "#101315"
FOREGROUND = "#f0f0f0"
FONT_DIR = Path("/usr/share/fonts/TTF")


class Config(cherries.BaseConfig):
    source: Path = Path("60-fixed-activation-contact-render")
    output: Path = Path("81-contact-keyframe-preview-003")


def record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    movie = source / "mouthopen-to-smile-fixed-activation-contact.mp4"
    render_manifest_path = source / "manifest.json"
    render_manifest = json.loads(render_manifest_path.read_text())
    if render_manifest.get("status") != "complete":
        raise ValueError("source render manifest is not complete")
    if (
        render_manifest.get("frame_count") != STATE_COUNT
        or render_manifest.get("video_frames") != VIDEO_FRAMES
        or render_manifest.get("fps") != FPS
    ):
        raise ValueError("source render does not have the expected 121-state movie")
    if len(render_manifest.get("state_receipts", [])) != STATE_COUNT:
        raise ValueError("source render lacks 121 state receipts")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to decode movie frames")
    selected = "+".join(f"eq(n\\,{index})" for index in VIDEO_INDICES)
    with tempfile.TemporaryDirectory(prefix="mouthopen-keyframes-") as temporary:
        temporary_path = Path(temporary)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(movie),
                "-vf",
                f"select={selected}",
                "-fps_mode",
                "passthrough",
                str(temporary_path / "frame-%02d.png"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        extracted = sorted(temporary_path.glob("frame-*.png"))
        if len(extracted) != len(STATE_INDICES):
            raise ValueError(f"decoded {len(extracted)} keyframes; expected 5")

        output_manifest = cherries.output(cfg.output / "manifest.json", mkdir=True)
        output = output_manifest.parent
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"preview output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        preview = Image.new("RGB", PAGE_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(preview)
        font_title = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 34)
        font_label = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 24)
        draw.text(
            (PAGE_SIZE[0] // 2, 32),
            "MouthOpen → Smile · Fixed activation, re-equilibrated contact states",
            font=font_title,
            fill="white",
            anchor="mt",
        )
        font_footer = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 20)
        draw.text(
            (PAGE_SIZE[0] // 2, 850),
            "Left: complete tetmesh boundary · Right: principal activation · Shared amplitude scale a = 0-60",
            font=font_footer,
            fill="#d9d9d9",
            anchor="mt",
        )
        all_inversions = [
            int(render_manifest["state_receipts"][i]["inverted_cells"])
            for i in range(STATE_COUNT)
        ]
        draw.text(
            (PAGE_SIZE[0] // 2, 884),
            f"Bone/eye contact only · all frames: {min(all_inversions)}-{max(all_inversions)} inverted tets · exploratory",
            font=font_footer,
            fill="#d9d9d9",
            anchor="mt",
        )

        layout = ((0, 0), (1, 0), (2, 0), (0.5, 1), (1.5, 1))
        image_receipts: list[dict[str, Any]] = []
        for state_index, video_index, path, (column, row) in zip(
            STATE_INDICES, VIDEO_INDICES, extracted, layout, strict=True
        ):
            beta = float(render_manifest["state_receipts"][state_index]["beta_smile"])
            expected_beta = (1 - np.cos(np.pi * state_index / (STATE_COUNT - 1))) / 2
            if not np.isclose(beta, expected_beta, rtol=0, atol=2e-15):
                raise ValueError(f"state {state_index} has unexpected beta {beta}")
            with Image.open(path) as decoded:
                if decoded.size != (1920, 1080):
                    raise ValueError(
                        f"decoded keyframe has unexpected size {decoded.size}"
                    )
                panel = decoded.crop(PANEL_CROP).resize(
                    PANEL_SIZE, Image.Resampling.LANCZOS
                )
            x = int(column * PANEL_SIZE[0])
            y = 104 + row * 384
            if row == 1:
                y += 4
            caption = (
                "MouthOpen endpoint"
                if state_index == 0
                else "Smile endpoint"
                if state_index == STATE_COUNT - 1
                else "Transition"
            )
            draw.text(
                (x + PANEL_SIZE[0] // 2, y - 10),
                f"State {state_index:03d} · β={beta:.2f} · {caption}",
                font=font_label,
                fill=FOREGROUND,
                anchor="ms",
            )
            preview.paste(panel, (x, y))
            image_receipts.append(
                {
                    "state_index": state_index,
                    "decoded_video_frame_index": video_index,
                    "beta_smile": beta,
                    "image_crop_xyxy": list(PANEL_CROP),
                    "panel_xy": [x, y],
                    "decoded_keyframe": record(path),
                }
            )
            panel.close()

        preview_path = output / "contact-keyframes.png"
        preview.save(preview_path)
        preview.close()
        output_manifest.write_text(
            json.dumps(
                {
                    "schema": "fixed-activation-contact-keyframe-preview-v1",
                    "source_video": record(movie),
                    "source_render_manifest": record(render_manifest_path),
                    "preview": record(preview_path),
                    "fps": FPS,
                    "state_count": STATE_COUNT,
                    "video_frame_count": VIDEO_FRAMES,
                    "state_to_video_frame_mapping": "state 0 maps to video frame 0; state i>0 maps to frame 29+i, following 30 initial copies",
                    "panel_policy": "crop the baked renderer header/footer to avoid duplicated text; preserve the face and activation panels at common rendered scale",
                    "panels": image_receipts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    LOG.info("Wrote compact contact preview to %s", preview_path)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
