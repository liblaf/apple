"""Render only the three exact saved h=.20 pork histories as filled materials.

Run with ParaView's ``pvpython``.  This is strictly a visualization program:
it reads already saved VTU states, writes one PNG for each exact saved state,
and packages those PNGs at 30 FPS.  It neither imports a solver nor performs
any forward/inverse physics evaluation.

The only admissible histories are h020-direct, h020-shared, and the explicit
NONSTATIONARY/EXPLORATORY h020-shared-release continuation.  In particular,
the ``*_zero_u`` branch is intentionally not an input to this report.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, PLR0912, SLF001, TRY003
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import paraview.simple as pvs
from paraview import servermanager, vtk

FPS = 30
IMAGE_SIZE = (1800, 1000)
FAT_RGB = (0.929, 0.694, 0.125)
MUSCLE_RGB = (0.796, 0.153, 0.153)
EDGE_RGB = (0.10, 0.11, 0.14)
TARGET_RGB = (0.95, 0.22, 0.62)
CASES = (
    ("h020-direct", "canonical"),
    ("h020-shared", "canonical"),
    ("h020-shared-release", "NONSTATIONARY/EXPLORATORY"),
)


@dataclass(frozen=True)
class Source:
    case_name: str
    provenance: str
    path: Path
    summary: dict[str, Any]
    manifest: tuple[dict[str, Any], ...]

    @property
    def label(self) -> str:
        return f"2d__{self.case_name}__{self.provenance.replace('/', '-')}"

    @property
    def output_name(self) -> str:
        return {
            "h020-direct": "direct",
            "h020-shared": "shared",
            "h020-shared-release": "shared-release",
        }[self.case_name]


def digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reset() -> None:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()


def version(command: str) -> str:
    executable = shutil.which(command) or command
    return subprocess.run(
        [executable, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]


def paraview_version() -> str:
    manager = servermanager.vtkSMProxyManager
    return ".".join(
        str(value)
        for value in (
            manager.GetVersionMajor(),
            manager.GetVersionMinor(),
            manager.GetVersionPatch(),
        )
    )


def sha256(path: Path) -> str:
    return digest(path)["sha256"]


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def exact_source(root: Path, case_name: str, provenance: str) -> Source:
    case_dir = root / case_name
    summary_path, series_path, final_path = (
        case_dir / name for name in ("summary.json", "history.vtu.series", "final.vtu")
    )
    if (
        not summary_path.is_file()
        or not series_path.is_file()
        or not final_path.is_file()
    ):
        raise FileNotFoundError(f"incomplete saved history: {case_dir}")
    summary = json_object(summary_path)
    case = summary.get("case")
    if not isinstance(case, dict):
        raise TypeError(f"missing case receipt: {summary_path}")
    required_case = {
        "name": case_name,
        "length": 1.0,
        "height": 0.2,
        "muscle_layout": "band",
        "poisson": 0.49,
    }
    if any(case.get(key) != value for key, value in required_case.items()):
        raise ValueError(f"not the specified h=.20 baseline family: {summary_path}")
    if provenance == "canonical" and case.get("protocol") not in {"direct", "shared"}:
        raise ValueError(f"unexpected canonical protocol: {summary_path}")
    if provenance != "canonical":
        continuation = summary.get("continuation")
        if (
            not isinstance(continuation, dict)
            or continuation.get("seed_status") != "NONSTATIONARY/EXPLORATORY"
        ):
            raise ValueError(
                f"release must remain explicitly exploratory: {summary_path}"
            )
        if case.get("protocol") != "shared_then_release":
            raise ValueError(f"unexpected release protocol: {summary_path}")
    inverse = summary.get("inverse")
    if not isinstance(inverse, dict) or not isinstance(inverse.get("evaluations"), int):
        raise TypeError(f"missing saved-state count: {summary_path}")
    series = json_object(series_path)
    entries = series.get("files")
    if not isinstance(entries, list) or len(entries) != inverse["evaluations"]:
        raise ValueError(f"incomplete saved-state series: {series_path}")
    manifest: list[dict[str, Any]] = []
    for step, entry in enumerate(entries):
        expected = f"frames/step-{step:04d}.vtu"
        if (
            not isinstance(entry, dict)
            or entry.get("name") != expected
            or entry.get("time") != float(step)
        ):
            raise ValueError(
                f"nonconsecutive or mismatched saved state {step}: {series_path}"
            )
        frame = case_dir / expected
        if not frame.is_file():
            raise FileNotFoundError(frame)
        manifest.append({"step": step, "time": float(step), **digest(frame)})
    if sha256(final_path) != manifest[-1]["sha256"]:
        raise ValueError(f"final.vtu must byte-match the last saved state: {case_dir}")
    return Source(
        case_name, provenance, series_path.resolve(), summary, tuple(manifest)
    )


def sources(canonical_root: Path, exploratory_root: Path) -> tuple[Source, ...]:
    if "zero_u" in str(canonical_root) or "zero_u" in str(exploratory_root):
        raise ValueError("zero_u is excluded from this renderer")
    result = tuple(
        exact_source(
            canonical_root if provenance == "canonical" else exploratory_root,
            name,
            provenance,
        )
        for name, provenance in CASES
    )
    if tuple(source.case_name for source in result) != tuple(name for name, _ in CASES):
        raise AssertionError("exact source order changed")
    return result


def source_times(reader: Any) -> list[float]:
    reader.UpdatePipeline()
    values = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    return values or [0.0]


def arrays(reader: Any, association: str, time: float) -> set[str]:
    reader.UpdatePipeline(time)
    information = reader.GetDataInformation()
    data = (
        information.GetPointDataInformation()
        if association == "POINTS"
        else information.GetCellDataInformation()
    )
    return {
        data.GetArrayInformation(index).GetName()
        for index in range(data.GetNumberOfArrays())
    }


def pipeline_bounds(
    proxy: Any, time: float
) -> tuple[float, float, float, float, float, float]:
    """Return the exact geometry bounds after a pipeline filter is evaluated."""
    proxy.UpdatePipeline(time)
    result = tuple(float(value) for value in proxy.GetDataInformation().GetBounds())
    if len(result) != 6 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid pipeline bounds at time {time}: {result}")
    return result


def union(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return tuple(
        min(left[index], right[index])
        if index % 2 == 0
        else max(left[index], right[index])
        for index in range(6)
    )


def scan_union_camera(
    items: tuple[Source, ...],
) -> tuple[float, float, float, float, float, float]:
    result: tuple[float, float, float, float, float, float] | None = None
    for item in items:
        reset()
        reader = pvs.OpenDataFile(
            str(item.path), registrationName=f"scan {item.case_name}"
        )
        times = source_times(reader)
        if times != [row["time"] for row in item.manifest]:
            raise ValueError(f"ParaView/source manifest mismatch: {item.path}")
        deformed = pvs.WarpByVector(
            registrationName=f"scan deformed {item.case_name}", Input=reader
        )
        deformed.Vectors = ["POINTS", "Displacement"]
        target_shape = pvs.WarpByVector(
            registrationName=f"scan target {item.case_name}", Input=reader
        )
        target_shape.Vectors = ["POINTS", "TargetDisplacement"]
        for time in times:
            point_names, cell_names = (
                arrays(reader, "POINTS", time),
                arrays(reader, "CELLS", time),
            )
            if {"Displacement", "TargetDisplacement"} - point_names:
                raise KeyError(f"missing geometry vectors at {item.path}, step {time}")
            if "MuscleMask" not in cell_names:
                raise KeyError(
                    f"MuscleMask is required for material rendering: {item.path}"
                )
            current = union(
                pipeline_bounds(deformed, time),
                pipeline_bounds(target_shape, time),
            )
            result = current if result is None else union(result, current)
    if result is None:
        raise RuntimeError("no bounds scanned")
    return result


def camera(
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, list[float] | float]:
    xmin, xmax, ymin, ymax, _zmin, _zmax = bounds
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    width, height = xmax - xmin, ymax - ymin
    scale = 0.58 * max(height, width / (IMAGE_SIZE[0] / IMAGE_SIZE[1]), 1.0e-6)
    return {
        "position": [center_x, center_y, max(width, height, 1.0) * 3],
        "focal_point": [center_x, center_y, 0.0],
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": scale,
    }


def configure(
    view: Any, bounds: tuple[float, float, float, float, float, float]
) -> None:
    spec = camera(bounds)
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = spec["position"]
    view.CameraFocalPoint = spec["focal_point"]
    view.CameraViewUp = spec["view_up"]
    view.CameraParallelScale = spec["parallel_scale"]


def text(
    view: Any, value: str, position: str = "Upper Left Corner", size: int = 18
) -> None:
    source = pvs.Text(registrationName=value[:40])
    source.Text = value
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = position
    display.FontSize = size
    display.Bold = 1
    display.Color = [0.96, 0.97, 0.98]


def threshold(reader: Any, lower: float, upper: float, name: str) -> Any:
    selected = pvs.Threshold(registrationName=name, Input=reader)
    selected.Scalars = ["CELLS", "MuscleMask"]
    selected.ThresholdMethod = "Between"
    selected.LowerThreshold, selected.UpperThreshold = lower, upper
    return selected


def material(
    reader: Any,
    view: Any,
    lower: float,
    upper: float,
    color: tuple[float, float, float],
    name: str,
) -> None:
    warped = pvs.WarpByVector(
        registrationName=f"{name} deformed", Input=threshold(reader, lower, upper, name)
    )
    warped.Vectors = ["POINTS", "Displacement"]
    display = pvs.Show(warped, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    pvs.ColorBy(display, None)
    display.ColorArrayName = [None, ""]
    display.DiffuseColor, display.AmbientColor = color, color
    display.EdgeColor = EDGE_RGB
    display.LineWidth = 0.35
    display.Opacity = 1.0
    display.Ambient, display.Diffuse = 0.32, 0.68


def target(reader: Any, view: Any) -> None:
    displaced = pvs.WarpByVector(registrationName="target shape", Input=reader)
    displaced.Vectors = ["POINTS", "TargetDisplacement"]
    boundary = pvs.FeatureEdges(
        registrationName="target boundary", Input=pvs.ExtractSurface(Input=displaced)
    )
    boundary.BoundaryEdges, boundary.FeatureEdges = 1, 0
    boundary.ManifoldEdges, boundary.NonManifoldEdges = 0, 0
    display = pvs.Show(boundary, view, "GeometryRepresentation")
    display.Representation = "Wireframe"
    pvs.ColorBy(display, None)
    display.DiffuseColor, display.AmbientColor = TARGET_RGB, TARGET_RGB
    display.LineWidth, display.Opacity = 1.1, 0.72


def scene(
    reader: Any,
    view: Any,
    source: Source,
    bounds: tuple[float, float, float, float, float, float],
    *,
    with_time: bool,
) -> None:
    material(reader, view, 0.0, 0.5, FAT_RGB, "fat")
    material(reader, view, 0.5, 1.0, MUSCLE_RGB, "muscle")
    target(reader, view)
    label = f"{source.case_name} | {source.provenance}"
    text(view, label)
    text(
        view,
        "filled materials + thin triangle edges | pink: target",
        "Lower Left Corner",
        14,
    )
    if with_time:
        annotation = pvs.AnnotateTimeFilter(
            registrationName="saved optimization state", Input=reader
        )
        annotation.Format = "saved state = {time:.0f}"
        display = pvs.Show(annotation, view, "TextSourceRepresentation")
        display.WindowLocation, display.FontSize, display.Bold = (
            "Upper Right Corner",
            18,
            1,
        )
        display.Color = [0.96, 0.97, 0.98]
    configure(view, bounds)


def encode(frames: list[Path], video: Path) -> dict[str, Any]:
    ffmpeg, ffprobe = (shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames[0].parent / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,pix_fmt,nb_frames,r_frame_rate,width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream = probe["streams"][0]
    duration = len(frames) / FPS
    if (
        stream["codec_name"] != "h264"
        or stream["profile"] != "High"
        or stream["pix_fmt"] != "yuv420p"
        or int(stream["nb_frames"]) != len(frames)
        or stream["r_frame_rate"] != "30/1"
        or not math.isclose(
            float(probe["format"]["duration"]), duration, abs_tol=0.5 / FPS
        )
    ):
        raise ValueError(f"invalid exact-frame video: {probe}")
    return {
        **digest(video),
        "fps": FPS,
        "frame_count": len(frames),
        "duration_seconds": duration,
        "encoded_size": [int(stream["width"]), int(stream["height"])],
        "ffprobe": probe,
    }


def render_history(
    source: Source,
    output_root: Path,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    reset()
    output = output_root / source.output_name
    output.mkdir()
    reader = pvs.OpenDataFile(str(source.path), registrationName=source.case_name)
    times = source_times(reader)
    if times != [item["time"] for item in source.manifest]:
        raise ValueError(f"time mismatch for {source.path}")
    view, layout = (
        pvs.CreateView("RenderView"),
        pvs.CreateLayout(name=f"{source.case_name} evolution"),
    )
    layout.SetSize(*IMAGE_SIZE)
    if not layout.AssignView(0, view):
        raise RuntimeError("could not assign render view")
    scene(reader, view, source, bounds, with_time=True)
    animation = pvs.GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    animation.PlayMode = "Snap To TimeSteps"
    frames_dir = output / "frames"
    frames_dir.mkdir()
    pvs.SaveAnimation(
        str(frames_dir / "frame.png"),
        layout,
        animation,
        FrameWindow=[0, len(times) - 1],
        SuffixFormat="_%05d",
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) != len(source.manifest) or not all(
        frame.stat().st_size > 20_000 for frame in frames
    ):
        raise ValueError(f"invalid PNG sequence: {frames_dir}")
    png_manifest = tuple(
        {"step": step, **digest(frame)} for step, frame in enumerate(frames)
    )
    if len({item["sha256"] for item in png_manifest}) != len(png_manifest):
        raise ValueError(
            f"duplicate rendered PNGs in exact-state sequence: {frames_dir}"
        )
    final = output / "final-shape.png"
    shutil.copy2(frames[-1], final)
    if sha256(final) != png_manifest[-1]["sha256"]:
        raise ValueError("final still must byte-match the final PNG")
    video = encode(frames, output / "evolution.mp4")
    receipt = {
        "status": "ok",
        "source": digest(source.path),
        "case": source.case_name,
        "provenance": source.provenance,
        "source_timestep_values": times,
        "source_manifest": source.manifest,
        "png_manifest": png_manifest,
        "source_to_png_sha256_mapping": [
            {
                "step": source_row["step"],
                "source_sha256": source_row["sha256"],
                "png_sha256": png_row["sha256"],
            }
            for source_row, png_row in zip(source.manifest, png_manifest, strict=True)
        ],
        "one_saved_state_per_png": True,
        "no_interpolation_or_duplication": True,
        "final_still": {
            "path": "final-shape.png",
            "source_step": source.manifest[-1]["step"],
            "png_sha256": png_manifest[-1]["sha256"],
        },
        "rendering": {
            "filled_material_polygons": True,
            "material_mask": "MuscleMask",
            "fat_rgb": FAT_RGB,
            "muscle_rgb": MUSCLE_RGB,
            "thin_charcoal_triangle_edges": True,
            "metric_coloring": False,
            "metric_scalar_bars": False,
            "target": "thin pink outline",
        },
        "camera": {"shared_union_bounds": bounds, **camera(bounds)},
        "video": video,
    }
    write_json(output / "render-receipt.json", receipt)
    return receipt


def split_four(layout: Any) -> list[int]:
    layout.SplitHorizontal(0, 0.5)
    left, right = (
        int(layout.SMProxy.GetFirstChild(0)),
        int(layout.SMProxy.GetSecondChild(0)),
    )
    layout.SplitVertical(left, 0.5)
    layout.SplitVertical(right, 0.5)
    return [
        int(layout.SMProxy.GetFirstChild(left)),
        int(layout.SMProxy.GetSecondChild(left)),
        int(layout.SMProxy.GetFirstChild(right)),
        int(layout.SMProxy.GetSecondChild(right)),
    ]


def render_final_comparison(
    items: tuple[Source, ...],
    output: Path,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    reset()
    layout = pvs.CreateLayout(name="h=.20 final material comparison")
    layout.SetSize(*IMAGE_SIZE)
    slots = split_four(layout)
    rendered: list[str] = []
    for slot, item in zip(slots[:3], items, strict=True):
        # Read the manifest's last VTU directly.  A comparison layout cannot
        # have independent animation times for three histories of different
        # lengths; this preserves the exact final saved state in every tile.
        reader = pvs.OpenDataFile(
            item.manifest[-1]["path"], registrationName=f"final {item.case_name}"
        )
        reader.UpdatePipeline()
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(slot, view):
            raise RuntimeError("comparison layout assignment failed")
        scene(reader, view, item, bounds, with_time=False)
        rendered.append(item.case_name)
    missing = pvs.CreateView("RenderView")
    if not layout.AssignView(slots[3], missing):
        raise RuntimeError("missing-evidence tile assignment failed")
    configure(missing, bounds)
    text(missing, "nu = 0.35", "Upper Left Corner", 24)
    text(missing, "NO EXACT SAVED h=.20 RUN", "Lower Left Corner", 20)
    text(
        missing,
        "Geometry intentionally absent\n(no fabricated comparison)",
        "Upper Right Corner",
        16,
    )
    pvs.RenderAllViews()
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    if output.stat().st_size <= 20_000:
        raise ValueError("empty final comparison")
    return {
        "path": output.name,
        **digest(output),
        "tiles": [
            *rendered,
            "nu=.35: no exact saved h=.20 run; geometry intentionally absent",
        ],
        "shared_union_camera": {"bounds": bounds, **camera(bounds)},
    }


def square_polydata(matrix: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for x, y in np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) @ matrix.T:
        points.InsertNextPoint(float(x), float(y), 0.0)
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)
    for index in range(4):
        polygon.GetPointIds().SetId(index, index)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polygon)
    result = vtk.vtkPolyData()
    result.SetPoints(points)
    result.SetPolys(cells)
    return result


def show_polydata(
    polydata: vtk.vtkPolyData, view: Any, color: tuple[float, float, float], name: str
) -> None:
    producer = pvs.TrivialProducer(registrationName=name)
    producer.GetClientSideObject().SetOutput(polydata)
    display = pvs.Show(producer, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    pvs.ColorBy(display, None)
    display.DiffuseColor, display.AmbientColor, display.EdgeColor = (
        color,
        color,
        EDGE_RGB,
    )
    display.LineWidth, display.Opacity, display.Ambient, display.Diffuse = (
        1.2,
        0.85,
        0.35,
        0.65,
    )


def render_shared_square(shared: Source, output: Path) -> dict[str, Any]:
    convergence = shared.summary.get("inverse", {}).get("convergence", {})
    if convergence.get("practical_stationarity_gate") is not False:
        raise ValueError("shared square must be explicitly marked nonstationary")
    controls = np.asarray(
        np.load(shared.path.parent / "final-state.npz")["controls"], dtype=float
    )
    if controls.shape != (3,):
        raise ValueError(
            "shared activation square requires exactly one 3-DoF shared control"
        )
    ainv = np.array(
        [[1 + controls[0], controls[2]], [controls[2], 1 + controls[1]]], dtype=float
    )
    if not np.isfinite(ainv).all() or abs(np.linalg.det(ainv)) < 1.0e-12:
        raise ValueError("final shared Ainv is non-finite or singular")
    preferred_f = np.linalg.inv(ainv)
    all_vertices = np.vstack(
        (
            np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) @ preferred_f.T,
        )
    )
    lo, hi = all_vertices.min(axis=0), all_vertices.max(axis=0)
    padding = max(float(np.max(hi - lo)) * 0.12, 0.05)
    square_bounds = (
        lo[0] - padding,
        hi[0] + padding,
        lo[1] - padding,
        hi[1] + padding,
        0.0,
        0.0,
    )
    reset()
    layout = pvs.CreateLayout(name="shared activation preferred square")
    layout.SetSize(*IMAGE_SIZE)
    layout.SplitHorizontal(0, 0.5)
    left, right = (
        int(layout.SMProxy.GetFirstChild(0)),
        int(layout.SMProxy.GetSecondChild(0)),
    )
    for slot, matrix, heading, color in (
        (left, np.eye(2), "rest square", FAT_RGB),
        (right, preferred_f, "active-preferred square (A = Ainv^-1)", MUSCLE_RGB),
    ):
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(slot, view):
            raise RuntimeError("square layout assignment failed")
        configure(view, square_bounds)
        show_polydata(square_polydata(matrix), view, color, heading)
        text(view, heading)
        text(
            view,
            "nonstationary saved shared control; kinematic preferred shape only",
            "Lower Left Corner",
            14,
        )
        text(
            view,
            (
                f"a = [{controls[0]:+.4f}, {controls[1]:+.4f}, "
                f"{controls[2]:+.4f}]\n"
                f"det(Ainv) = {np.linalg.det(ainv):.6f}"
            ),
            "Lower Right Corner",
            14,
        )
    pvs.RenderAllViews()
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    if output.stat().st_size <= 20_000:
        raise ValueError("empty shared activation square")
    return {
        "path": output.name,
        **digest(output),
        "case": shared.case_name,
        "provenance": shared.provenance,
        "controls": controls.tolist(),
        "ainv": ainv.tolist(),
        "preferred_f": preferred_f.tolist(),
        "camera_bounds": square_bounds,
        "mapping": "Ainv = I + [[a_xx, a_xy], [a_xy, a_yy]]; Stable Neo-Hookean uses G = F @ Ainv; the zero-elastic-strain preferred affine map therefore has F = Ainv^-1.",
        "interpretation": "This is a unit square under the final saved shared control, not a solved pork equilibrium and not a convergence claim.",
        "nonstationary": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be empty: {output_root}")
    exact = sources(args.canonical_root.resolve(), args.exploratory_root.resolve())
    bounds = scan_union_camera(exact)
    output_root.mkdir(parents=True, exist_ok=True)
    histories = [render_history(item, output_root, bounds) for item in exact]
    comparison = render_final_comparison(
        exact, output_root / "final-comparison.png", bounds
    )
    shared_square = render_shared_square(
        exact[1], output_root / "shared-activation-square.png"
    )
    write_json(
        output_root / "render-receipt.json",
        {
            "status": "ok",
            "admissible_exact_cases": [item.case_name for item in exact],
            "excluded": [
                "h020-shared-release_zero_u",
                "all non-h=.20 or mismatched material/Poisson/geometry histories",
            ],
            "fps": FPS,
            "one_saved_state_per_video_frame": True,
            "shared_union_camera": {"bounds": bounds, **camera(bounds)},
            "render_contract": {
                "material_only": True,
                "filled_framework": True,
                "material_mask": "MuscleMask",
                "metric_scalar_coloring": False,
                "metric_scalar_bars": False,
                "fat_rgb": FAT_RGB,
                "muscle_rgb": MUSCLE_RGB,
                "target": "thin pink outline",
                "triangle_edges": "thin charcoal",
            },
            "software": {
                "pvpython": sys.executable,
                "paraview_version": paraview_version(),
                "ffmpeg": version("ffmpeg"),
                "ffprobe": version("ffprobe"),
            },
            "histories": histories,
            "final_comparison": comparison,
            "shared_activation_square": shared_square,
            "renderer_source": digest(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
