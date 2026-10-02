# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915
"""Play the verified activation transition backward with corrected annotations."""

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
    output: Path = Path("96-mouthopen-to-smile-animation")


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/TTF/DejaVuSans{suffix}.ttf", size)


def main(cfg: Config) -> None:
    manifest_path = cherries.input(cfg.render / "manifest.json")
    source_manifest = json.loads(manifest_path.read_text())
    video = Path(source_manifest["video"]["path"])
    assert record(video)["sha256"] == source_manifest["video"]["sha256"]
    summary_path = Path(source_manifest["source_summary"]["path"])
    assert record(summary_path)["sha256"] == source_manifest["source_summary"]["sha256"]
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    assert len(summary["frames"]) == 121
    assert source_manifest["timing"]["total_video_frames"] == 179
    assert source_manifest["timing"]["extra_endpoint_frames_each"] == 29
    out = cherries.output(cfg.output)
    out.mkdir(exist_ok=False)
    target = out / "mouthopen-to-smile-transition.mp4"
    decoder_command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        "reverse",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    encoder_command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "1920x1080",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    mapping = []
    with (
        (out / "decoder.log").open("wb") as decode_log,
        (out / "encoder.log").open("wb") as encode_log,
        subprocess.Popen(
            decoder_command, stdout=subprocess.PIPE, stderr=decode_log
        ) as decoder,
        subprocess.Popen(
            encoder_command, stdin=subprocess.PIPE, stderr=encode_log
        ) as encoder,
    ):
        assert decoder.stdout is not None
        assert encoder.stdin is not None
        for index in range(179):
            pixels = decoder.stdout.read(1920 * 1080 * 3)
            assert len(pixels) == 1920 * 1080 * 3
            image = Image.frombytes("RGB", (1920, 1080), pixels)
            progress = min(max(index - 29, 0), 120)
            source_state = 120 - progress
            alpha = float(summary["frames"][source_state]["alpha"])
            beta = 1 - alpha
            np.testing.assert_allclose(
                beta, (1 - np.cos(np.pi * progress / 120)) / 2, rtol=0, atol=1e-15
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1919, 144), fill="#101315")
            draw.text(
                (960, 25),
                "MouthOpen → Smile activation transition",
                font=font(34, bold=True),
                fill="white",
                anchor="mt",
            )
            draw.text(
                (480, 105),
                "Re-equilibrated tetmesh shape",
                font=font(23, bold=True),
                fill="#e6e6e6",
                anchor="mt",
            )
            draw.text(
                (1440, 105),
                "Principal activation mode",
                font=font(23, bold=True),
                fill="#e6e6e6",
                anchor="mt",
            )
            draw.text(
                (1860, 73),
                f"Frame {progress + 1}/121 · Smile blend {beta:.3f}",
                font=font(22),
                fill="#d9d9d9",
                anchor="rt",
            )
            encoder.stdin.write(image.tobytes())
            if index in {0, 89, 178}:
                image.save(out / f"frame-{index:03d}.png")
            mapping.append(
                {
                    "video_frame": index,
                    "source_video_frame": 178 - index,
                    "source_state_index": source_state,
                    "smile_blend": beta,
                }
            )
        assert decoder.stdout.read(1) == b""
        encoder.stdin.close()
        assert decoder.wait() == 0
        assert encoder.wait() == 0
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(target),
            ]
        )
    )
    stream = probe["streams"][0]
    assert stream["width"] == 1920
    assert stream["height"] == 1080
    assert int(stream["nb_frames"]) == 179
    assert stream["avg_frame_rate"] == "30/1"
    assert abs(float(probe["format"]["duration"]) - 179 / 30) < 1e-5
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True
    )
    (out / "ffprobe.json").write_text(json.dumps(probe, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "reverse-activation-animation-v1",
                "status": "complete",
                "source_video": record(video),
                "source_render_manifest": record(manifest_path),
                "numerical_summary": record(summary_path),
                "video": record(target),
                "source": record(Path(__file__)),
                "video_frames": mapping,
                "state_count": 121,
                "video_frame_count": 179,
                "fps": 30,
                "duration_seconds": float(probe["format"]["duration"]),
                "decoder_command": decoder_command,
                "encoder_command": encoder_command,
                "full_stream_decode_passed": True,
                "description": "Reverse playback of the previously verified quasistatic states; activation and prescribed jaw pose traverse the same path backward. Header and progress annotations are updated. No new forward solves or independent reverse continuation.",
            },
            indent=2,
        )
        + "\n"
    )
    cherries.log_metrics({"video_frames": 179, "equilibrium_states": 121, "fps": 30})


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
