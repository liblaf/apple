"""Render exact inverse-evaluation histories with ParaView.

Run this file with ParaView's ``pvpython``.  It deliberately renders one PNG
for every source time step: no resampling, interpolation, dropped frames, or
duplicated stills are permitted.  ffmpeg only packages those PNGs at 30 FPS.
"""

from __future__ import annotations

# ruff: noqa: ARG001, C901, EM101, EM102, FBT003, PLR0912, PLR0915, SLF001, TRY003
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import paraview.simple as pvs

FPS = 30
PNG_SIZE = (1800, 1000)
HISTORY_NAMES = ("history.vtu.series", "inverse.vtkhdf")
ACTIVATION_ARRAYS = ("ActivationYY", "ActivationInv", "Activation")
DETERMINANT_ARRAYS = ("DetF", "DetG", "DetAinv")


@dataclass(frozen=True)
class Source:
    path: Path
    label: str
    case_name: str
    summary: dict[str, Any]


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


def paraview_version() -> str:
    from paraview import servermanager

    manager = servermanager.vtkSMProxyManager
    return ".".join(
        str(value)
        for value in (
            manager.GetVersionMajor(),
            manager.GetVersionMinor(),
            manager.GetVersionPatch(),
        )
    )


def reset_session() -> None:
    """Reset without allowing ParaView to replace our explicit cameras."""
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()


def sibling_summary(path: Path) -> dict[str, Any]:
    """Read the case summary when present; old/smoke histories need none."""
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"case summary must be a JSON object: {summary_path}")
    return payload


def discover(input_roots: list[Path]) -> list[Source]:
    found: list[Source] = []
    for input_root in input_roots:
        for name in HISTORY_NAMES:
            for path in sorted(input_root.rglob(name)):
                summary = sibling_summary(path)
                nested_case = summary.get("case")
                canonical_name = (
                    nested_case.get("name") if isinstance(nested_case, dict) else None
                )
                case_name = str(
                    canonical_name or summary.get("name") or path.parent.name
                )
                found.append(Source(path.resolve(), case_name, case_name, summary))
    if not found:
        expected = ", ".join(HISTORY_NAMES)
        raise FileNotFoundError(
            f"no inverse history beneath {input_roots}; expected {expected}"
        )
    return found


def summary_gate(summary: dict[str, Any]) -> bool:
    """Return the physical stationarity receipt; never use the old tail gate."""
    stationarity = summary.get("stationarity")
    if isinstance(stationarity, dict) and "passed" in stationarity:
        return bool(stationarity["passed"])
    inverse = summary.get("inverse")
    if isinstance(inverse, dict):
        convergence = inverse.get("convergence")
        if isinstance(convergence, dict):
            return bool(
                convergence.get(
                    "physical_stationarity_gate",
                    convergence.get("practical_stationarity_gate", False),
                )
            )
    return bool(summary.get("physical_stationarity_gate", False))


def summary_counts(summary: dict[str, Any]) -> tuple[int, int]:
    """Return valid and total inverse evaluations, preferring explicit receipts."""
    inverse = summary.get("inverse")
    nested_inverse = inverse if isinstance(inverse, dict) else {}
    total = next(
        (
            int(summary[key])
            for key in ("evaluations_and_frames", "frames", "evaluations")
            if isinstance(summary.get(key), int)
        ),
        int(nested_inverse.get("evaluations", summary.get("evaluations", 0))),
    )
    valid = next(
        (
            int(summary[key])
            for key in (
                "valid_inverse_frames",
                "valid_inverse_evaluations",
                "inverse_evaluation_success_count",
                "evaluation_success_count",
            )
            if isinstance(summary.get(key), int)
        ),
        total
        - max(
            int(
                summary.get(
                    "inverse_evaluation_failure_count",
                    nested_inverse.get("failures", {}).get("adjoint", 0),
                )
            ),
            int(
                summary.get(
                    "forward_failure_count",
                    nested_inverse.get("failures", {}).get("forward", 0),
                )
            ),
            int(nested_inverse.get("failures", {}).get("nonfinite", 0)),
        ),
    )
    return max(valid, 0), max(total, 0)


def infer_source_dimensions(sources: list[Source]) -> dict[Path, int]:
    dimensions: dict[Path, int] = {}
    for source in sources:
        reset_session()
        reader = pvs.OpenDataFile(
            str(source.path), registrationName=f"selection scan {source.case_name}"
        )
        reader.UpdatePipeline()
        times = source_times(reader)
        dimensions[source.path] = infer_dimension(reader, times[0])
    return dimensions


def deduplicate(sources: list[Source], dimensions: dict[Path, int]) -> list[Source]:
    """Keep one best complete history for each (dimension, case name)."""
    groups: dict[tuple[int, str], list[Source]] = {}
    for source in sources:
        groups.setdefault((dimensions[source.path], source.case_name), []).append(
            source
        )
    selected: list[Source] = []
    for (dimension, case_name), candidates in sorted(groups.items()):
        chosen = sorted(
            candidates,
            key=lambda source: (
                -int(summary_gate(source.summary)),
                -summary_counts(source.summary)[0],
                -summary_counts(source.summary)[1],
                str(source.path),
            ),
        )[0]
        selected.append(
            Source(
                chosen.path,
                f"{dimension}d__{case_name}",
                chosen.case_name,
                chosen.summary,
            )
        )
    return selected


def arrays(reader: Any, association: str, time_value: float) -> set[str]:
    reader.UpdatePipeline(time_value)
    info = reader.GetDataInformation()
    data = (
        info.GetPointDataInformation()
        if association == "POINTS"
        else info.GetCellDataInformation()
    )
    return {
        data.GetArrayInformation(index).GetName()
        for index in range(data.GetNumberOfArrays())
    }


def source_times(reader: Any) -> list[float]:
    values = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    # A non-temporal VTK-HDF dataset is still exactly one source state.
    return values or [0.0]


def infer_dimension(reader: Any, time_value: float) -> int:
    """Infer dimensionality from the first actual VTK frame, never its filename."""
    reader.UpdatePipeline(time_value)
    bounds = tuple(float(value) for value in reader.GetDataInformation().GetBounds())
    if len(bounds) != 6 or not all(math.isfinite(value) for value in bounds):
        raise ValueError(f"invalid first-frame bounds: {bounds}")
    x_span, y_span, z_span = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    scale = max(abs(value) for value in bounds) if bounds else 1.0
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError(f"degenerate first-frame x/y bounds: {bounds}")
    # The 2-D runner stores a flat z=0 mesh, while the pork cuboid has a
    # physical z extent.  Relative tolerance avoids classifying round-off as 3-D.
    return 3 if z_span > max(1.0e-12, scale * 1.0e-10) else 2


def camera_spec(
    dimension: int, bounds: tuple[float, float, float, float, float, float], views: int
) -> dict[str, Any]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    spans = [xmax - xmin, ymax - ymin, zmax - zmin]
    if not all(math.isfinite(value) and value >= 0 for value in spans):
        raise ValueError(f"invalid camera bounds: {bounds}")
    if dimension == 2:
        # SplitVertical produces stacked wide panels.  Fit both the physical
        # vertical span and the horizontal span without assuming L=1.
        panel_aspect = PNG_SIZE[0] / (PNG_SIZE[1] / views)
        scale = 0.58 * max(spans[1], spans[0] / panel_aspect, 1.0e-9)
        return {
            "position": [center[0], center[1], max(spans[0], spans[1], 1.0) * 3],
            "focal_point": [center[0], center[1], 0.0],
            "view_up": [0.0, 1.0, 0.0],
            "parallel_scale": scale,
        }
    diagonal = math.sqrt(sum(value * value for value in spans))
    distance = max(2.0 * diagonal, 1.0e-6)
    direction = np.asarray([1.25, 1.15, 1.25], dtype=float)
    direction /= np.linalg.norm(direction)
    return {
        "position": [
            center[index] + distance * float(direction[index]) for index in range(3)
        ],
        "focal_point": center,
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": max(0.58 * diagonal, 1.0e-9),
    }


def configure_view(
    view: Any,
    dimension: int,
    bounds: tuple[float, float, float, float, float, float],
    views: int,
) -> None:
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    camera = camera_spec(dimension, bounds, views)
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal_point"]
    view.CameraViewUp = camera["view_up"]
    view.CameraParallelScale = camera["parallel_scale"]


def add_text(view: Any, text: str) -> None:
    source = pvs.Text(registrationName=text[:48])
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 15
    display.Bold = 1
    display.Color = [0.96, 0.97, 0.98]


def add_time(view: Any, reader: Any) -> None:
    source = pvs.AnnotateTimeFilter(registrationName="Optimization step", Input=reader)
    source.Format = "optimization step = {time:.0f}"
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Right Corner"
    display.FontSize = 18
    display.Bold = 1
    display.Color = [0.96, 0.97, 0.98]


def show_target(reader: Any, view: Any, point_arrays: set[str], dimension: int) -> None:
    if "TargetDisplacement" not in point_arrays:
        return
    target = pvs.WarpByVector(registrationName="Target shape overlay", Input=reader)
    target.Vectors = ["POINTS", "TargetDisplacement"]
    surface = pvs.ExtractSurface(registrationName="Target surface", Input=target)
    if dimension == 2:
        overlay = pvs.FeatureEdges(
            registrationName="Target boundary outline", Input=surface
        )
        overlay.BoundaryEdges = 1
        overlay.FeatureEdges = 0
        overlay.ManifoldEdges = 0
        overlay.NonManifoldEdges = 0
    else:
        # A tet wireframe would cover the solution with every internal edge.
        overlay = surface
    display = pvs.Show(overlay, view, "GeometryRepresentation")
    display.Representation = "Wireframe"
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = [0.95, 0.22, 0.62]
    display.AmbientColor = [0.95, 0.22, 0.62]
    display.LineWidth = 1.2 if dimension == 2 else 0.8
    display.Opacity = 0.42 if dimension == 2 else 0.18


def muscle_only(reader: Any) -> Any:
    """Expose the internal 3-D muscle layer in the activation panel."""
    selected = pvs.Threshold(registrationName="Active muscle tetrahedra", Input=reader)
    selected.Scalars = ["CELLS", "Muscle"]
    selected.ThresholdMethod = "Between"
    selected.LowerThreshold = 0.5
    selected.UpperThreshold = 1.0
    return selected


def show_warped(
    reader: Any,
    view: Any,
    *,
    scalar: str | None,
    association: str,
    title: str,
    scalar_range: tuple[float, float] | None,
    use_magnitude: bool,
) -> None:
    warped = pvs.WarpByVector(registrationName=f"Deformed {title}", Input=reader)
    warped.Vectors = ["POINTS", "Displacement"]
    display = pvs.Show(warped, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.10, 0.11, 0.14]
    display.LineWidth = 0.25
    display.Ambient = 0.25
    display.Diffuse = 0.75
    if scalar is None:
        # Readers may mark a cell scalar active by default.  The geometry panel
        # must never inherit it: the target overlay needs a stable contrast.
        pvs.ColorBy(display, None)
        display.ColorArrayName = [None, ""]
        display.DiffuseColor = [0.78, 0.70, 0.56]
        display.AmbientColor = [0.78, 0.70, 0.56]
        return
    if use_magnitude:
        # ParaView's -1 component is a vector magnitude.  Match scan_ranges()
        # so the six-component 3-D activation has one honest shared scale.
        pvs.ColorBy(display, (association, scalar, "Magnitude"))
    else:
        pvs.ColorBy(display, (association, scalar))
    lut = pvs.GetColorTransferFunction(scalar)
    lut.VectorMode = "Magnitude" if use_magnitude else "Component"
    if not use_magnitude:
        lut.VectorComponent = 0
    lut.ApplyPreset("Cool to Warm", True)
    if scalar_range is None:
        raise AssertionError("a scalar view requires an exact temporal range")
    lut.RescaleTransferFunction(*scalar_range)
    display.SetScalarBarVisibility(view, True)
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Right Corner"


def split_three(layout: Any, count: int) -> list[int]:
    if count == 1:
        return [0]
    locations = [0]
    current = 0
    for _ in range(count - 1):
        layout.SplitVertical(current, 1.0 / (count - len(locations) + 1))
        first = int(layout.SMProxy.GetFirstChild(current))
        second = int(layout.SMProxy.GetSecondChild(current))
        if first < 0 or second < 0:
            raise RuntimeError("ParaView layout split failed")
        locations[-1] = first
        locations.append(second)
        current = second
    return locations


def encode(frames_dir: Path, frames: list[Path], output: Path) -> dict[str, Any]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
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
            str(frames_dir / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
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
                "stream=codec_name,pix_fmt,nb_frames,r_frame_rate,width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream = probe["streams"][0]
    duration = len(frames) / FPS
    half_frame = 0.5 / FPS
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or int(stream["nb_frames"]) != len(frames)
        or stream["r_frame_rate"] != "30/1"
        or not math.isclose(
            float(probe["format"]["duration"]), duration, abs_tol=half_frame
        )
    ):
        raise ValueError(f"invalid exact-frame video: {probe}")
    return {
        **digest(output),
        "fps": FPS,
        "frame_count": len(frames),
        "encoded_size": [int(stream["width"]), int(stream["height"])],
        "even_dimension_padding": "at most one pixel per axis",
        "duration_seconds": duration,
        "duration_tolerance_seconds": half_frame,
        "ffprobe": probe,
    }


def union_bounds(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return tuple(
        min(left[index], right[index])
        if index % 2 == 0
        else max(left[index], right[index])
        for index in range(6)
    )


def vector_displaced_bounds(
    reader: Any, name: str
) -> tuple[float, float, float, float, float, float]:
    info = reader.GetDataInformation()
    rest = tuple(float(value) for value in info.GetBounds())
    array = info.GetPointDataInformation().GetArrayInformation(name)
    if array is None or array.GetNumberOfComponents() < 3:
        raise KeyError(f"point vector {name!r} is unavailable for camera fitting")
    result = []
    for axis in range(3):
        lower, upper = (float(value) for value in array.GetComponentRange(axis))
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(f"non-finite {name} component {axis} range")
        result.extend((rest[2 * axis] + lower, rest[2 * axis + 1] + upper))
    return tuple(result)


def scan_ranges(
    sources: list[Source],
) -> tuple[
    dict[Path, int],
    dict[tuple[int, str], tuple[float, float]],
    dict[Path, tuple[float, float, float, float, float, float]],
]:
    """Return shared scalar ranges and per-case no-clipping camera bounds."""
    ranges: dict[tuple[int, str], tuple[float, float]] = {}
    dimensions: dict[Path, int] = {}
    camera_bounds: dict[Path, tuple[float, float, float, float, float, float]] = {}
    for source in sources:
        reset_session()
        reader = pvs.OpenDataFile(
            str(source.path), registrationName=f"range scan {source.label}"
        )
        reader.UpdatePipeline()
        times = source_times(reader)
        dimension = infer_dimension(reader, times[0])
        dimensions[source.path] = dimension
        names = arrays(reader, "CELLS", times[0])
        selected = [
            name for name in (*ACTIVATION_ARRAYS, *DETERMINANT_ARRAYS) if name in names
        ]
        bounds: tuple[float, float, float, float, float, float] | None = None
        for time_value in times:
            reader.UpdatePipeline(time_value)
            info = reader.GetDataInformation()
            rest = tuple(float(value) for value in info.GetBounds())
            if len(rest) != 6 or not all(math.isfinite(value) for value in rest):
                raise ValueError(f"invalid bounds at time {time_value}: {rest}")
            current = vector_displaced_bounds(reader, "Displacement")
            point_names = arrays(reader, "POINTS", time_value)
            if "TargetDisplacement" in point_names:
                current = union_bounds(
                    current, vector_displaced_bounds(reader, "TargetDisplacement")
                )
            current = union_bounds(current, rest)
            bounds = current if bounds is None else union_bounds(bounds, current)
            data = info.GetCellDataInformation()
            for name in selected:
                array = data.GetArrayInformation(name)
                if array is None:
                    raise KeyError(
                        f"array {name!r} disappeared at time {time_value}: {source.path}"
                    )
                component = (
                    -1
                    if name in ACTIVATION_ARRAYS and array.GetNumberOfComponents() > 1
                    else 0
                )
                lower, upper = (
                    float(value) for value in array.GetComponentRange(component)
                )
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    raise ValueError(f"non-finite range for {name} in {source.path}")
                key = (dimension, name)
                old = ranges.get(key)
                ranges[key] = (
                    (lower, upper)
                    if old is None
                    else (min(old[0], lower), max(old[1], upper))
                )
        if bounds is None:
            raise RuntimeError(f"no camera bounds scanned for {source.path}")
        camera_bounds[source.path] = bounds
    for key, (lower, upper) in list(ranges.items()):
        if lower == upper:
            padding = max(abs(lower) * 0.01, 1.0e-12)
            ranges[key] = (lower - padding, upper + padding)
        elif key[1] in DETERMINANT_ARRAYS:
            # Fixed zero makes inversions legible in all cases and frames.
            ranges[key] = (min(lower, 0.0), max(upper, 0.0))
    return dimensions, ranges, camera_bounds


def render(
    source: Source,
    output_root: Path,
    dimension: int,
    scalar_ranges: dict[tuple[int, str], tuple[float, float]],
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    reset_session()
    output = output_root / source.label
    output.mkdir(parents=True, exist_ok=False)
    reader = pvs.OpenDataFile(
        str(source.path), registrationName=f"{source.label} source"
    )
    reader.UpdatePipeline()
    times = source_times(reader)
    expected_times = [float(index) for index in range(len(times))]
    if times != expected_times:
        raise ValueError(
            f"source times must be exact consecutive optimizer steps: "
            f"expected {expected_times[:3]}...{expected_times[-3:]}, "
            f"got {times[:3]}...{times[-3:]}"
        )
    inverse = source.summary.get("inverse")
    nested_inverse = inverse if isinstance(inverse, dict) else {}
    refinement = source.summary.get("refinement", nested_inverse.get("refinement", {}))
    if not isinstance(refinement, dict):
        refinement = {}
    failures = nested_inverse.get("failures", {})
    if not isinstance(failures, dict):
        failures = {}
    paraview = source.summary.get("paraview")
    nested_paraview = paraview if isinstance(paraview, dict) else {}
    evaluations = nested_inverse.get("evaluations", source.summary.get("evaluations"))
    if evaluations is None:
        raise TypeError(f"summary lacks inverse.evaluations/evaluations: {source.path}")
    declared_counts = {"evaluations": evaluations}
    if "frames" in nested_paraview:
        declared_counts["paraview.frames"] = nested_paraview["frames"]
    for name, value in declared_counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer for {source.path}")
        if value != len(times):
            raise ValueError(
                f"{name}={value} does not match {len(times)} source steps: "
                f"{source.path}"
            )
    if infer_dimension(reader, times[0]) != dimension:
        raise ValueError(
            f"first-frame dimension changed since scalar-range scan: {source.path}"
        )
    if len(times) != len(set(times)):
        raise ValueError(f"duplicate source time values in {source.path}: {times}")
    point_arrays, cell_arrays = (
        arrays(reader, "POINTS", times[0]),
        arrays(reader, "CELLS", times[0]),
    )
    if "Displacement" not in point_arrays:
        raise KeyError(f"{source.path} has no point-vector Displacement array")
    activation = next((name for name in ACTIVATION_ARRAYS if name in cell_arrays), None)
    determinant = next(
        (name for name in DETERMINANT_ARRAYS if name in cell_arrays), None
    )
    activation_components = 0
    if activation is not None:
        cell_data = reader.GetDataInformation().GetCellDataInformation()
        activation_info = cell_data.GetArrayInformation(activation)
        if activation_info is None:
            raise KeyError(
                f"activation array disappeared at first frame: {source.path}"
            )
        activation_components = activation_info.GetNumberOfComponents()
    views = [("geometry", None, "POINTS", "deformed geometry")]
    if activation is not None:
        views.append(("activation", activation, "CELLS", activation))
    if determinant is not None:
        views.append(("determinant", determinant, "CELLS", determinant))
    layout = pvs.CreateLayout(name=f"Exact inverse evolution {source.label}")
    locations = split_three(layout, len(views))
    layout.SetSize(*PNG_SIZE)
    for location, (kind, scalar, association, title) in zip(
        locations, views, strict=True
    ):
        view = pvs.CreateView("RenderView")
        configure_view(view, dimension, bounds, len(views))
        if not layout.AssignView(location, view):
            raise RuntimeError("failed to assign ParaView render view")
        displayed_source = reader
        if kind == "activation" and dimension == 3:
            if "Muscle" not in cell_arrays:
                raise KeyError(
                    f"{source.path} has no cell-scalar Muscle array for the 3-D "
                    "activation view"
                )
            displayed_source = muscle_only(reader)
        show_warped(
            displayed_source,
            view,
            scalar=scalar,
            association=association,
            title=title,
            scalar_range=None if scalar is None else scalar_ranges[(dimension, scalar)],
            use_magnitude=scalar == activation and activation_components > 1,
        )
        if kind == "geometry":
            show_target(reader, view, point_arrays, dimension)
            add_time(view, reader)
        add_text(
            view,
            f"{source.label} | {kind} | stationarity={summary_gate(source.summary)} | exact source frames only",
        )
        # Filters and overlay representations can alter camera state on first
        # render; restore the deterministic dimension-specific camera last.
        configure_view(view, dimension, bounds, len(views))
        pvs.Render(view)
    scene = pvs.GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.PlayMode = "Snap To TimeSteps"
    actual = source_times(reader)
    if actual != times:
        raise ValueError(f"time values changed during scene setup: {actual} != {times}")
    frames_dir = output / "frames"
    frames_dir.mkdir()
    pvs.SaveAnimation(
        str(frames_dir / "frame.png"),
        layout,
        scene,
        FrameWindow=[0, len(times) - 1],
        SuffixFormat="_%05d",
        ImageResolution=list(PNG_SIZE),
        FontScaling="Do not scale fonts",
    )
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) != len(times):
        raise ValueError(
            f"source timestep count {len(times)} does not equal PNG count {len(frames)}"
        )
    if not all(frame.stat().st_size > 20_000 for frame in frames):
        raise ValueError("ParaView produced an empty/invalid PNG frame")
    video = encode(frames_dir, frames, output / "evolution.mp4")
    receipt = {
        "status": "ok",
        "paraview_version": paraview_version(),
        "source": digest(source.path),
        "dimension": dimension,
        "source_timestep_values": times,
        "source_timestep_count": len(times),
        "declared_source_counts": declared_counts,
        "physical_stationarity_gate": summary_gate(source.summary),
        "refinement_accepted_iterations": refinement.get("accepted_iterations"),
        "refinement_trial_forward_failure_count": refinement.get(
            "trial_forward_failures", failures.get("refinement_trial_forward")
        ),
        "png_frame_count": len(frames),
        "no_interpolation_or_duplication": True,
        "camera": {
            "deformed_and_target_bounds": list(bounds),
            **camera_spec(dimension, bounds, len(views)),
        },
        "views": [item[0] for item in views],
        "video": video,
    }
    write_json(output / "render-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="one input directory or a comma-separated list of input directories",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    input_roots = [
        Path(value).resolve() for value in args.input_root.split(",") if value
    ]
    if not input_roots:
        raise ValueError("--input-root must include at least one directory")
    for input_root in input_roots:
        if not input_root.is_dir():
            raise NotADirectoryError(input_root)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = discover(input_roots)
    candidate_dimensions = infer_source_dimensions(candidates)
    sources = deduplicate(candidates, candidate_dimensions)
    dimensions, scalar_ranges, camera_bounds = scan_ranges(sources)
    receipts = [
        render(
            source,
            output_root,
            dimensions[source.path],
            scalar_ranges,
            camera_bounds[source.path],
        )
        for source in sources
    ]
    write_json(
        output_root / "render-receipt.json",
        {"status": "ok", "fps": FPS, "case_count": len(receipts), "cases": receipts},
    )


if __name__ == "__main__":
    main()
