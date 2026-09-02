"""Render the exact mixed-root h=.20 2-D comparison with ParaView.

The four source histories are intentionally split between the immutable
canonical attempt and the later NONSTATIONARY/EXPLORATORY releases.  This
wrapper reuses the hardened exact-frame renderer but preflights that mapping
before it permits any output to be written.
"""

from __future__ import annotations

# ruff: noqa: C416, C901, EM101, EM102, SIM102, TRY003
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_renderer() -> Any:
    path = Path(__file__).with_name("20-render-paraview.py")
    spec = importlib.util.spec_from_file_location("h020_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_renderer()
CASES = (
    ("h020-direct", "canonical"),
    ("h020-shared", "canonical"),
    ("h020-shared-release", "NONSTATIONARY_EXPLORATORY"),
    ("h020-shared-release_zero_u", "NONSTATIONARY_EXPLORATORY"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object: {path}")
    return result


def exact_source(root: Path, case_name: str, provenance: str) -> Any:
    case_dir = root / case_name
    summary_path = case_dir / "summary.json"
    series_path = case_dir / "history.vtu.series"
    if not summary_path.is_file() or not series_path.is_file():
        raise FileNotFoundError(f"missing complete case receipt: {case_dir}")
    summary = load_json(summary_path)
    if summary.get("case", {}).get("name") != case_name:
        raise ValueError(f"case-name mismatch: {summary_path}")
    if provenance == "NONSTATIONARY_EXPLORATORY":
        if summary.get("continuation", {}).get("seed_status") != "NONSTATIONARY/EXPLORATORY":
            raise ValueError(f"exploratory status missing: {summary_path}")
    inverse = summary.get("inverse")
    if not isinstance(inverse, dict) or not isinstance(inverse.get("evaluations"), int):
        raise TypeError(f"missing inverse evaluation receipt: {summary_path}")
    series = load_json(series_path)
    files = series.get("files")
    if not isinstance(files, list) or len(files) != inverse["evaluations"]:
        raise ValueError(f"incomplete history: {series_path}")
    manifest: list[dict[str, Any]] = []
    for step, entry in enumerate(files):
        expected = f"frames/step-{step:04d}.vtu"
        if not isinstance(entry, dict) or entry.get("time") != float(step) or entry.get("name") != expected:
            raise ValueError(f"nonconsecutive history item {step}: {series_path}")
        frame = case_dir / expected
        if not frame.is_file():
            raise FileNotFoundError(frame)
        manifest.append({"step": step, "time": float(step), **R.digest(frame)})
    final = case_dir / "final.vtu"
    if not final.is_file() or sha256(final) != manifest[-1]["sha256"]:
        raise ValueError(f"final.vtu must byte-match final history state: {case_dir}")
    label = f"2d__{case_name}__{provenance}"
    return R.Source(series_path.resolve(), label, case_name, summary, manifest)


def preflight(canonical_root: Path, exploratory_root: Path) -> list[Any]:
    expected_by_root = {
        canonical_root: {"h020-direct", "h020-shared"},
        exploratory_root: {"h020-shared-release", "h020-shared-release_zero_u"},
    }
    for root, expected in expected_by_root.items():
        if not root.is_dir():
            raise NotADirectoryError(root)
        found = {path.parent.name for path in root.rglob("history.vtu.series")}
        if found != expected:
            raise ValueError(f"exact histories required in {root}: expected {expected}, got {found}")
    sources = [
        exact_source(
            canonical_root if provenance == "canonical" else exploratory_root,
            case_name,
            provenance,
        )
        for case_name, provenance in CASES
    ]
    if tuple(source.case_name for source in sources) != tuple(item[0] for item in CASES):
        raise AssertionError("exact case ordering lost")
    return sources


def version(command: str) -> str:
    return subprocess.run(
        [shutil.which(command) or command, "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    canonical_root = args.canonical_root.resolve()
    exploratory_root = args.exploratory_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be empty: {output_root}")
    sources = preflight(canonical_root, exploratory_root)
    dimensions, scalar_ranges, per_case_bounds = R.scan_ranges(sources)
    if set(dimensions.values()) != {2}:
        raise ValueError(f"all comparison sources must be 2-D: {dimensions}")
    bounds = None
    for value in per_case_bounds.values():
        bounds = value if bounds is None else R.union_bounds(bounds, value)
    if bounds is None:
        raise RuntimeError("no union camera bounds")
    output_root.mkdir(parents=True, exist_ok=True)
    receipts = [
        R.render(source, output_root, dimensions[source.path], scalar_ranges, bounds)
        for source in sources
    ]
    R.write_json(
        output_root / "render-receipt.json",
        {
            "status": "ok",
            "exact_cases": [source.case_name for source in sources],
            "provenance": {case_name: provenance for case_name, provenance in CASES},
            "exploratory_label": "NONSTATIONARY/EXPLORATORY",
            "fps": R.FPS,
            "one_saved_state_per_video_frame": True,
            "shared_union_camera": {"bounds": list(bounds), **R.camera_spec(2, bounds, 3)},
            "shared_scalar_ranges": {name: list(value) for (dimension, name), value in scalar_ranges.items() if dimension == 2},
            "software": {"pvpython": sys.executable, "paraview_version": R.paraview_version(), "pyvista": "not used", "ffmpeg": version("ffmpeg"), "ffprobe": version("ffprobe")},
            "renderer_source": R.digest(Path(__file__)),
            "cases": receipts,
        },
    )


if __name__ == "__main__":
    main()
