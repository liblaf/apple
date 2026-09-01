"""Rebase and validate receipts after parallel ParaView renders are consolidated."""

from __future__ import annotations

# ruff: noqa: C901, EM102, TRY003
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return path.stat().st_size, hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rebase(root: Path, source_root: Path | None) -> None:
    case_dirs = sorted(path for path in root.glob("*__*") if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"no rendered cases beneath {root}")
    cases = []
    rates = set()
    for case_dir in case_dirs:
        receipt_path = case_dir / "render-receipt.json"
        video_path = case_dir / "evolution.mp4"
        receipt = read_json(receipt_path)
        if receipt.get("status") != "ok":
            raise ValueError(f"non-ok case receipt: {receipt_path}")
        source_path = Path(receipt["source"]["path"])
        if source_root is not None:
            _dimension, separator, case_name = case_dir.name.partition("__")
            if not separator or not case_name:
                raise ValueError(f"invalid rendered case directory: {case_dir}")
            source_path = source_root / case_name / "history.vtu.series"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_size, source_sha256 = digest(source_path)
        recorded_source = receipt["source"]
        if source_size != recorded_source.get(
            "bytes"
        ) or source_sha256 != recorded_source.get("sha256"):
            raise ValueError(f"source digest mismatch: {source_path}")
        recorded_source.update(
            {
                "path": str(source_path.resolve()),
                "bytes": source_size,
                "sha256": source_sha256,
            }
        )
        size, sha256 = digest(video_path)
        video = receipt["video"]
        if size != video.get("bytes") or sha256 != video.get("sha256"):
            raise ValueError(f"video digest mismatch: {video_path}")
        counts = {
            int(receipt["source_timestep_count"]),
            int(receipt["png_frame_count"]),
            int(video["frame_count"]),
            int(video["ffprobe"]["streams"][0]["nb_frames"]),
        }
        if len(counts) != 1:
            raise ValueError(f"frame-count mismatch: {receipt_path}")
        rate = video["ffprobe"]["streams"][0]["r_frame_rate"]
        if rate != "30/1" or video["ffprobe"]["streams"][0]["pix_fmt"] != "yuv420p":
            raise ValueError(f"unexpected encoding: {receipt_path}")
        rates.add(int(video["fps"]))
        video["path"] = str(video_path.resolve())
        write_json(receipt_path, receipt)
        cases.append(receipt)
    if rates != {30}:
        raise ValueError(f"inconsistent FPS beneath {root}: {rates}")
    write_json(
        root / "render-receipt.json",
        {"status": "ok", "fps": 30, "case_count": len(cases), "cases": cases},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="canonical case root used to rebase and validate history paths",
    )
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve() if args.source_root is not None else None
    if source_root is not None and not source_root.is_dir():
        raise NotADirectoryError(source_root)
    for root in args.roots:
        rebase(root.resolve(), source_root)


if __name__ == "__main__":
    main()
