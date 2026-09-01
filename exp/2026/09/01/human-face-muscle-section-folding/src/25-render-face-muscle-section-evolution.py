"""Render the exact 201-state id64 material-section history at 30 FPS."""

from __future__ import annotations

# ruff: noqa: EM101, TRY003
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "data" / "10-face-muscle-section"
HISTORY = ROOT / "data" / "15-face-muscle-section-history"
OUT = ROOT / "data" / "25-face-muscle-section-evolution"
PRIMARY = "20-human-face-smile-no-skin-lr3"
STEPS = tuple(range(201))
ARRAYS = ("DetF", "DetAinv", "DetG", "DoubleInverted", "ActivationNorm")
SIZE = (3000, 1000)
FPS = 30


def digest(path: Path) -> dict[str, object]:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def digest_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(digest(path)["sha256"].encode())
    return h.hexdigest()


def camera(
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    span = np.asarray(
        (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    )
    center = np.asarray(
        (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )
    )
    return {
        "position": (center + 2.6 * max(span) * np.asarray((1.0, 0.8, 0.65))).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0, 0, 1],
        "parallel_projection": True,
        "parallel_scale": float(0.62 * max(span)),
    }


def setup(plotter: pv.Plotter, spec: dict[str, object]) -> None:
    plotter.set_background("white")
    plotter.enable_parallel_projection()
    plotter.camera.position = spec["position"]
    plotter.camera.focal_point = spec["focal_point"]
    plotter.camera.up = spec["view_up"]
    plotter.camera.parallel_scale = spec["parallel_scale"]


def panel(
    plotter: pv.Plotter,
    reference: pv.UnstructuredGrid,
    state: pv.UnstructuredGrid,
    scalar: str,
    clim: tuple[float, float],
    step: int,
    spec: dict[str, object],
) -> None:
    setup(plotter, spec)
    plotter.add_mesh(
        reference, color="#9aa0a6", style="wireframe", line_width=1.2, opacity=0.72
    )
    plotter.add_mesh(
        state,
        scalars=scalar,
        clim=clim,
        cmap="coolwarm",
        show_edges=True,
        edge_color="#202124",
        line_width=0.35,
        scalar_bar_args={"title": scalar, "fmt": "%.2g"},
    )
    double_inverted = state.threshold((0.5, 1.5), scalars="DoubleInverted")
    if double_inverted.n_cells:
        plotter.add_mesh(
            double_inverted,
            color="#ff00b8",
            show_edges=True,
            edge_color="black",
            line_width=1.5,
            opacity=1.0,
        )
    label = f"step {step:03d} / 200 | {scalar}\ngray: reference wireframe"
    if scalar == "DetG":
        label += " | magenta/black: double-inverted\nlocal coordinates; no exaggeration"
    plotter.add_text(label, font_size=12, color="black", position="upper_left")


def validate_history() -> tuple[list[Path], pv.UnstructuredGrid, dict[str, object]]:
    summary = json.loads((HISTORY / "summary.json").read_text())
    receipt = json.loads((HISTORY / "receipt.json").read_text())
    if summary != receipt or summary["frame_count"] != len(STEPS):
        raise ValueError("history summary/receipt contract mismatch")
    if summary["primary_case"] != PRIMARY:
        raise ValueError("unexpected source case")
    if summary["inverse_steps"] != {
        "exact_consecutive": True,
        "first": 0,
        "last": 200,
    }:
        raise ValueError("history is not the exact 0..200 sequence")
    series_path = HISTORY / "history.vtu.series"
    series = json.loads(series_path.read_text())
    entries = series.get("files")
    expected_names = [f"frames/step-{step:03d}.vtu" for step in STEPS]
    if (
        not isinstance(entries, list)
        or [entry.get("name") for entry in entries] != expected_names
    ):
        raise ValueError("series does not enumerate the exact expected frames")
    if [entry.get("time") for entry in entries] != list(STEPS):
        raise ValueError("series times do not match inverse steps")
    frames = [HISTORY / name for name in expected_names]
    if any(not frame.is_file() for frame in frames):
        raise FileNotFoundError("history frame missing")
    reference = pv.read(SECTION / f"{PRIMARY}-section-reference.vtu")
    if reference.n_cells != 31:
        raise ValueError("expected the matched 31-cell reference section")
    first = pv.read(frames[0])
    if first.n_cells != reference.n_cells or set(first.cell_data) < set(ARRAYS):
        raise ValueError("history frame array contract mismatch")
    return frames, reference, summary


def ranges_and_bounds(
    frames: list[Path], reference: pv.UnstructuredGrid
) -> tuple[
    dict[str, tuple[float, float]], tuple[float, float, float, float, float, float]
]:
    extrema = {name: [0.0, 0.0] for name in ARRAYS[:3]}
    bounds = [list(reference.bounds)]
    for frame in frames:
        grid = pv.read(frame)
        if grid.n_cells != reference.n_cells or set(grid.cell_data) < set(ARRAYS):
            message = f"frame contract mismatch: {frame}"
            raise ValueError(message)
        bounds.append(list(grid.bounds))
        for name, bounds_for_array in extrema.items():
            values = np.asarray(grid.cell_data[name], dtype=float)
            if not np.isfinite(values).all():
                message = f"nonfinite {name}: {frame}"
                raise ValueError(message)
            bounds_for_array[0] = min(bounds_for_array[0], float(values.min()))
            bounds_for_array[1] = max(bounds_for_array[1], float(values.max()))
    stacked = np.asarray(bounds)
    union = (
        float(stacked[:, 0].min()),
        float(stacked[:, 1].max()),
        float(stacked[:, 2].min()),
        float(stacked[:, 3].max()),
        float(stacked[:, 4].min()),
        float(stacked[:, 5].max()),
    )
    ranges = {}
    for name, (lo, hi) in extrema.items():
        limit = max(abs(lo), abs(hi))
        if not limit > 0:
            message = f"degenerate scalar range: {name}"
            raise ValueError(message)
        ranges[name] = (-limit, limit)
    return ranges, union


def render_frames(
    frames: list[Path],
    reference: pv.UnstructuredGrid,
    ranges: dict[str, tuple[float, float]],
    spec: dict[str, object],
) -> list[Path]:
    png_dir = OUT / "frames"
    png_dir.mkdir()
    rendered = []
    for step, frame in enumerate(frames):
        state = pv.read(frame)
        plot = pv.Plotter(
            shape=(1, 3), off_screen=True, window_size=SIZE, lighting="light kit"
        )
        for index, name in enumerate(ranges):
            plot.subplot(0, index)
            panel(plot, reference, state, name, ranges[name], step, spec)
        output = png_dir / f"frame-{step:03d}.png"
        plot.screenshot(output)
        plot.close()
        rendered.append(output)
    return rendered


def render_video(frames: list[Path]) -> tuple[Path, dict[str, object]]:
    video = OUT / "face-muscle-section-evolution.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(OUT / "frames" / "frame-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(probe.stdout)
    stream = facts["streams"][0]
    count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(facts["format"]["duration"])
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or stream["r_frame_rate"] != "30/1"
        or count != len(frames)
        or abs(duration - len(frames) / FPS) > 1 / FPS
    ):
        message = f"video contract mismatch: {facts}"
        raise ValueError(message)
    return video, {"stream": stream, "format": facts["format"]}


def write_pvsm(reference: Path, frame: Path) -> Path:
    from paraview.simple import (
        GetActiveViewOrCreate,
        SaveState,
        Show,
        XMLUnstructuredGridReader,
    )

    view = GetActiveViewOrCreate("RenderView")
    reference_source = XMLUnstructuredGridReader(FileName=[str(reference)])
    state_source = XMLUnstructuredGridReader(FileName=[str(frame)])
    Show(reference_source, view)
    Show(state_source, view)
    output = OUT / "face-muscle-section-evolution.pvsm"
    SaveState(str(output))
    return output


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    frames, reference, history_summary = validate_history()
    ranges, bounds = ranges_and_bounds(frames, reference)
    spec = camera(bounds)
    OUT.mkdir(parents=True)
    rendered = render_frames(frames, reference, ranges, spec)
    if len(rendered) != len(frames) or any(not path.is_file() for path in rendered):
        raise ValueError("rendered PNG sequence is incomplete")
    video, probe = render_video(rendered)
    pvsm = write_pvsm(SECTION / f"{PRIMARY}-section-reference.vtu", frames[0])
    receipt = {
        "status": "ok",
        "primary_case": PRIMARY,
        "inverse_steps": {"first": 0, "last": 200, "exact_consecutive": True},
        "dimensions": {"section_cells": 31, "png_size": list(SIZE)},
        "arrays": list(ARRAYS),
        "scalar_ranges": ranges,
        "camera": spec,
        "render_contract": {
            "source_frame_count": len(frames),
            "png_frame_count": len(rendered),
            "video_frame_count": len(rendered),
            "fps": FPS,
            "no_interpolation_or_duplication": True,
            "reference_wireframe": True,
            "double_inverted_overlay": "magenta fill with black edges",
            "no_deformation_exaggeration": True,
        },
        "sources": {
            "history_summary": digest(HISTORY / "summary.json"),
            "history_receipt": digest(HISTORY / "receipt.json"),
            "history_series": digest(HISTORY / "history.vtu.series"),
            "reference_section": digest(SECTION / f"{PRIMARY}-section-reference.vtu"),
            "all_history_frames_sha256": digest_paths(frames),
            "history_validation": history_summary["validation"],
        },
        "outputs": {
            "video": digest(video),
            "pvsm": digest(pvsm),
            "all_png_frames_sha256": digest_paths(rendered),
        },
        "video_probe": probe,
    }
    (OUT / "render-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
