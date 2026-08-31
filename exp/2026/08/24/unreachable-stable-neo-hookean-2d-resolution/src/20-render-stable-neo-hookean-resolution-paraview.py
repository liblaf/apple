# Copyright (c) 2026 liblaf
from __future__ import annotations

# Executed by ParaView 6.1.1's pvpython, not the project interpreter.
# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, PLR0915, TRY003
import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_DESIGN = "exact-plane-strain-stable-neo-hookean-active-resolution-study"
FREE_VARIANT = "free"
CONTROL_VARIANTS = ("free", "tied", "regularized")
REQUIRED_POINT_ARRAYS = ("Displacement", "DisplacementY")
REQUIRED_CELL_ARRAYS = (
    "MaterialId",
    "YoungModulusMPa",
    "PoissonRatio",
    "DetF",
    "DetG",
    "DetAinv",
    "MinSingularAinv",
    "ActivationXX",
    "ActivationYY",
    "ActivationXY",
    "ActivationNorm",
    "MuscleMask",
)


@dataclass(frozen=True)
class Case:
    payload: dict[str, Any]
    root: Path

    @property
    def resolution(self) -> tuple[int, int]:
        raw = self.payload["resolution"]
        return (int(raw[0]), int(raw[1]))

    @property
    def variant(self) -> str:
        return str(self.payload["variant"])

    @property
    def identifier(self) -> str:
        nx, ny = self.resolution
        return f"{nx}x{ny}-{self.variant}"

    @property
    def directory(self) -> Path:
        return self.root / self.identifier

    def path(self, key: str) -> Path:
        paths = self.payload["paths"]
        return self.directory / str(paths[key])

    def admissible_path(self, kind: str) -> Path:
        return self.path(f"best_admissible_prefix_{kind}")

    @property
    def admissible_metrics(self) -> dict[str, Any]:
        value = self.payload["best_admissible_prefix"]
        if not isinstance(value, dict):
            raise TypeError(
                f"invalid best_admissible_prefix metrics: {self.identifier}"
            )
        return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


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


def validate_summary(path: Path) -> tuple[dict[str, Any], list[Case]]:
    summary = read_json(path)
    if summary.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("numerical schema changed")
    if summary.get("design") != EXPECTED_DESIGN:
        raise ValueError("numerical design changed")
    if summary.get("complete") is not True:
        raise ValueError("numerical run is incomplete")
    cases_raw = summary.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("numerical summary has no cases")
    root = path.parent.resolve()
    cases = [Case(payload=case, root=root) for case in cases_raw]
    identifiers = [case.identifier for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate case identifiers: {identifiers}")
    for case in cases:
        if not case.directory.is_dir():
            raise FileNotFoundError(case.directory)
        for key in (
            "series",
            "trace",
            "best_vtu",
            "best_admissible_prefix_vtu",
            "best_admissible_prefix_profile",
            "best_admissible_prefix_spectrum",
        ):
            path_value = case.path(key)
            if case.directory.resolve() not in path_value.resolve().parents:
                raise ValueError(f"case input escapes root: {path_value}")
            if not path_value.is_file():
                raise FileNotFoundError(path_value)
        if int(case.payload["evaluations"]) < 2:
            raise ValueError(f"not enough inverse evaluations: {case.identifier}")
        if case.payload.get("best_admissible_prefix_is_verified") is not True:
            raise ValueError(
                f"comparison state is not verified admissible: {case.identifier}"
            )
        metrics = case.payload.get("best_admissible_prefix")
        if (
            not isinstance(metrics, dict)
            or int(metrics.get("verified_admissible", 0)) != 1
        ):
            raise ValueError(
                f"comparison metrics are not verified admissible: {case.identifier}"
            )
        first_invalid = case.payload.get("first_invalid_step")
        if first_invalid is not None and not 0 <= int(first_invalid) < int(
            case.payload["evaluations"]
        ):
            raise ValueError(
                f"invalid first_invalid_step for {case.identifier}: {first_invalid}"
            )
    return summary, cases


def select_cases(cases: list[Case]) -> tuple[list[Case], list[Case], Case]:
    free = sorted(
        (case for case in cases if case.variant == FREE_VARIANT),
        key=lambda case: case.resolution,
    )
    if len(free) < 2:
        raise ValueError("need at least two free-control resolutions")
    resolutions = [case.resolution for case in free]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("duplicate free-control resolutions")
    median = free[len(free) // 2]
    by_variant = {
        case.variant: case for case in cases if case.resolution == median.resolution
    }
    if set(CONTROL_VARIANTS) - set(by_variant):
        raise ValueError(
            f"missing variants at median {median.resolution}: {set(CONTROL_VARIANTS) - set(by_variant)}"
        )
    controls = [by_variant[variant] for variant in CONTROL_VARIANTS]
    return free, controls, median


def configure_camera(view: Any, *, crop: bool = False) -> None:
    if crop:
        camera = {
            "position": [0.20, 0.05, 3.0],
            "focal_point": [0.20, 0.05, 0.0],
            "parallel_scale": 0.10,
        }
    else:
        camera = {
            "position": [0.50, 0.10, 3.0],
            "focal_point": [0.50, 0.10, 0.0],
            "parallel_scale": 0.16,
        }
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal_point"]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CenterOfRotation = camera["focal_point"]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = camera["parallel_scale"]


def configure_view(view: Any, *, crop: bool = False) -> None:
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    configure_camera(view, crop=crop)


def split_even(layout: Any, location: int, count: int) -> list[int]:
    if count == 1:
        return [location]
    # Stacked views preserve a useful wide aspect ratio for this long, thin strip.
    layout.SplitVertical(location, 1.0 / count)
    first = int(layout.SMProxy.GetFirstChild(location))
    second = int(layout.SMProxy.GetSecondChild(location))
    if first < 0 or second < 0:
        raise RuntimeError("ParaView layout split failed")
    return [first, *split_even(layout, second, count - 1)]


def add_text(view: Any, name: str, text: str, *, font_size: int = 16) -> None:
    source = pvs.Text(registrationName=name)
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = font_size
    display.Color = [0.96, 0.97, 0.98]
    display.Bold = 1


def add_time_annotation(reader: Any, view: Any, name: str) -> None:
    source = pvs.AnnotateTimeFilter(registrationName=name, Input=reader)
    source.Format = "inverse evaluation = {time:.0f}"
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Right Corner"
    display.FontSize = 18
    display.Color = [0.96, 0.97, 0.98]
    display.Bold = 1


def add_reference_lines(summary: dict[str, Any], view: Any) -> None:
    width, height = (float(value) for value in summary["geometry"]["domain"])
    target_y = height + float(summary["geometry"]["target"][1])
    rest = pvs.Line(
        registrationName="Reference top",
        Point1=[0.0, height, 0.0],
        Point2=[width, height, 0.0],
    )
    target = pvs.Line(
        registrationName="Requested free-top target",
        Point1=[0.0, target_y, 0.0],
        Point2=[width, target_y, 0.0],
    )
    for source, color, width_px in (
        (rest, [0.82, 0.84, 0.87], 1.6),
        (target, [0.90, 0.28, 0.62], 3.0),
    ):
        display = pvs.Show(source, view, "GeometryRepresentation")
        display.Representation = "Surface"
        display.ColorArrayName = [None, ""]
        display.DiffuseColor = color
        display.AmbientColor = color
        display.LineWidth = width_px
        display.RenderLinesAsTubes = 1


def save_screenshot(
    path: Path, layout: Any, resolution: tuple[int, int]
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveScreenshot(
        str(temporary),
        layout,
        ImageResolution=list(resolution),
        TransparentBackground=0,
        FontScaling="Do not scale fonts",
    )
    actual_resolution = png_size(temporary)
    # Multi-view layouts can gain a few rows through ParaView's viewport tiling.
    if (
        actual_resolution[0] != resolution[0]
        or abs(actual_resolution[1] - resolution[1]) > 16
        or temporary.stat().st_size < 20_000
    ):
        raise ValueError(f"invalid screenshot: {temporary}")
    temporary.replace(path)
    return {
        **identity(path),
        "width": actual_resolution[0],
        "height": actual_resolution[1],
        "requested_width": resolution[0],
        "requested_height": resolution[1],
    }


def save_state(path: Path) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveState(str(temporary))
    head = temporary.read_text(encoding="utf-8", errors="strict")[:2048]
    if "ServerManagerState" not in head:
        raise ValueError(f"invalid ParaView state: {temporary}")
    temporary.replace(path)
    return identity(path)


def encode_exact_frame_video(
    frames_dir: Path,
    frames: list[Path],
    output: Path,
    *,
    fps: int,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    """Encode ParaView PNG frames only; never synthesize intermediate time steps."""
    if fps < 1:
        raise ValueError("video fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
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
            str(fps),
            "-i",
            str(frames_dir / "evolution_%03d.png"),
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
                "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate",
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
    expected_duration = len(frames) / fps
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or int(stream["width"]) != resolution[0]
        or int(stream["height"]) != resolution[1]
        or int(stream["nb_frames"]) != len(frames)
        or stream["r_frame_rate"] != f"{fps}/1"
        or not math.isclose(
            float(probe["format"]["duration"]), expected_duration, abs_tol=1.0e-6
        )
    ):
        raise ValueError(f"unexpected video metadata: {probe}")
    return {
        **identity(output),
        "fps": fps,
        "frame_count": len(frames),
        "duration_seconds": expected_duration,
        "no_interpolation": True,
        "ffprobe": probe,
    }


def open_data(path: Path, name: str, *, temporal: bool) -> Any:
    reader = pvs.OpenDataFile(str(path.resolve()), registrationName=name)
    reader.PointArrayStatus = list(REQUIRED_POINT_ARRAYS)
    reader.CellArrayStatus = list(REQUIRED_CELL_ARRAYS)
    reader.UpdatePipeline()
    if temporal and not list(pvs.GetTimeKeeper().TimestepValues):
        raise ValueError(f"temporal reader exposes no time values: {path}")
    return reader


def validate_reader(reader: Any, case: Case, *, time: float | None = None) -> None:
    if time is None:
        reader.UpdatePipeline()
    else:
        reader.UpdatePipeline(time)
    info = reader.GetDataInformation()
    if info.GetNumberOfPoints() != int(case.payload["n_nodes"]):
        raise ValueError(f"point count changed for {case.identifier}")
    if info.GetNumberOfCells() != int(case.payload["n_triangles"]):
        raise ValueError(f"cell count changed for {case.identifier}")
    point_data = info.GetPointDataInformation()
    cell_data = info.GetCellDataInformation()
    for name in REQUIRED_POINT_ARRAYS:
        if point_data.GetArrayInformation(name) is None:
            raise KeyError(f"missing point array {name!r} in {case.identifier}")
    for name in REQUIRED_CELL_ARRAYS:
        if cell_data.GetArrayInformation(name) is None:
            raise KeyError(f"missing cell array {name!r} in {case.identifier}")
    displacement = point_data.GetArrayInformation("Displacement")
    if displacement.GetNumberOfComponents() != 3:
        raise ValueError(f"Displacement is not a 3-vector in {case.identifier}")


def array_range(
    reader: Any, association: str, name: str, *, time: float | None = None
) -> tuple[float, float]:
    if time is None:
        reader.UpdatePipeline()
    else:
        reader.UpdatePipeline(time)
    info = reader.GetDataInformation()
    data = (
        info.GetPointDataInformation()
        if association == "POINTS"
        else info.GetCellDataInformation()
    )
    array = data.GetArrayInformation(name)
    if array is None:
        raise KeyError(name)
    values = tuple(float(value) for value in array.GetComponentRange(0))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {name} range: {values}")
    return values


def validate_orientation_preserving(reader: Any, case: Case) -> None:
    validate_reader(reader, case)
    for name in ("DetF", "DetG", "DetAinv"):
        lower, _upper = array_range(reader, "CELLS", name)
        if lower <= 0.0:
            raise ValueError(
                "admissible-prefix input is not orientation-preserving: "
                f"{case.identifier} {name}={lower}"
            )


def union_ranges(
    ranges: list[tuple[float, float]], *, symmetric: bool
) -> tuple[float, float]:
    if not ranges:
        raise ValueError("no ranges")
    lower = min(value[0] for value in ranges)
    upper = max(value[1] for value in ranges)
    if symmetric:
        magnitude = max(abs(lower), abs(upper), 1.0e-12)
        return (-magnitude, magnitude)
    if lower == upper:
        pad = max(abs(lower) * 0.01, 1.0e-9)
        return (lower - pad, upper + pad)
    return (lower, upper)


def scan_evolution_ranges(
    case: Case,
) -> tuple[list[float], tuple[float, float], tuple[float, float]]:
    pvs.ResetSession()
    reader = open_data(
        case.path("series"), f"{case.identifier} temporal range scan", temporal=True
    )
    time_values = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    expected = [float(step) for step in range(int(case.payload["evaluations"]))]
    if time_values != expected:
        raise ValueError(
            f"{case.identifier} must export every numerical evaluation without interpolation"
        )
    displacement_ranges: list[tuple[float, float]] = []
    activation_ranges: list[tuple[float, float]] = []
    for time in time_values:
        validate_reader(reader, case, time=time)
        displacement_ranges.append(
            array_range(reader, "POINTS", "DisplacementY", time=time)
        )
        activation_ranges.append(
            array_range(reader, "CELLS", "ActivationYY", time=time)
        )
    pvs.ResetSession()
    return (
        time_values,
        union_ranges(displacement_ranges, symmetric=False),
        union_ranges(activation_ranges, symmetric=True),
    )


def display_deformed(
    reader: Any,
    view: Any,
    *,
    array: str,
    association: str,
    scalar_range: tuple[float, float],
    preset: str,
    edges: bool,
    scalar_title: str,
    threshold_muscle: bool = False,
) -> Any:
    source = reader
    if threshold_muscle:
        source = pvs.Threshold(registrationName=f"{array} muscle only", Input=reader)
        source.Scalars = ["CELLS", "MuscleMask"]
        source.LowerThreshold = 0.5
        source.UpperThreshold = 1.0
        source.AllScalars = 1
    warp = pvs.WarpByVector(
        registrationName=f"Warp reference mesh for {array}", Input=source
    )
    warp.Vectors = ["POINTS", "Displacement"]
    warp.ScaleFactor = 1.0
    display = pvs.Show(warp, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges" if edges else "Surface"
    display.EdgeColor = [0.12, 0.13, 0.16]
    display.LineWidth = 0.28
    display.Ambient = 0.24
    display.Diffuse = 0.74
    display.Specular = 0.12
    display.SpecularPower = 18.0
    pvs.ColorBy(display, (association, array))
    lut = pvs.GetColorTransferFunction(array)
    lut.ApplyPreset(preset, True)
    lut.RescaleTransferFunction(*scalar_range)
    display.SetScalarBarVisibility(view, True)
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = scalar_title
    bar.ComponentTitle = ""
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Right Corner"
    bar.ScalarBarLength = 0.31
    bar.ScalarBarThickness = 14
    bar.TitleColor = [0.96, 0.97, 0.98]
    bar.LabelColor = [0.96, 0.97, 0.98]
    bar.TitleFontSize = 13
    bar.LabelFontSize = 11
    return display


def render_setup(
    summary: dict[str, Any], case: Case, output_dir: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    reader = open_data(
        case.admissible_path("vtu"),
        f"{case.identifier} reference material setup",
        temporal=False,
    )
    validate_orientation_preserving(reader, case)
    view = pvs.CreateView("RenderView")
    configure_view(view)
    layout = pvs.CreateLayout(name="Stable Neo-Hookean 2D material setup")
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign setup view")
    resolution = (1800, 700)
    layout.SetSize(*resolution)
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.12, 0.13, 0.16]
    display.LineWidth = 0.30
    display.Ambient = 0.25
    display.Diffuse = 0.75
    pvs.ColorBy(display, ("CELLS", "MaterialId"))
    lut = pvs.GetColorTransferFunction("MaterialId")
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = ["0", "fat", "1", "SMAS", "2", "muscle"]
    lut.IndexedColors = [0.88, 0.80, 0.67, 0.34, 0.62, 0.80, 0.86, 0.28, 0.22]
    display.SetScalarBarVisibility(view, True)
    add_reference_lines(summary, view)
    add_text(
        view,
        "Material setup",
        "Reference configuration: fat / stiff SMAS / local muscle\n"
        "bottom + both sides fixed; magenta = requested +0.1 free-top target\n"
        "ParaView warps the reference mesh exactly once in deformation assets",
    )
    pvs.Render(view)
    configure_camera(view)
    pvs.Render(view)
    png = save_screenshot(output_dir / "setup-materials.png", layout, resolution)
    state = save_state(output_dir / "setup-materials.pvsm")
    return {"input": identity(case.admissible_path("vtu")), "png": png, "pvsm": state}


def render_free_resolution_geometry(
    summary: dict[str, Any],
    free: list[Case],
    output_dir: Path,
    displacement_range: tuple[float, float],
) -> dict[str, Any]:
    pvs.ResetSession()
    layout = pvs.CreateLayout(name="Free-control resolution geometry")
    locations = split_even(layout, 0, len(free))
    resolution = (1800, 1500)
    layout.SetSize(*resolution)
    inputs: list[dict[str, Any]] = []
    for location, case in zip(locations, free, strict=True):
        reader = open_data(
            case.admissible_path("vtu"),
            f"{case.identifier} admissible geometry",
            temporal=False,
        )
        validate_orientation_preserving(reader, case)
        view = pvs.CreateView("RenderView")
        configure_view(view)
        if not layout.AssignView(location, view):
            raise RuntimeError("failed to assign resolution geometry view")
        display_deformed(
            reader,
            view,
            array="DisplacementY",
            association="POINTS",
            scalar_range=displacement_range,
            preset="Cool to Warm",
            edges=True,
            scalar_title="vertical displacement",
        )
        add_reference_lines(summary, view)
        best = case.admissible_metrics
        add_text(
            view,
            case.identifier,
            f"Free controls, best admissible prefix | {case.resolution[0]} x {case.resolution[1]} | {case.payload['n_triangles']} triangles\n"
            f"top error RMS = {float(best['top_error_rms']):.5f} | high-pass RMS = {float(best['top_highpass_rms']):.6f}\n"
            "Edges show the actual control/mesh basis; shared colour range has no clipping",
            font_size=13,
        )
        pvs.Render(view)
        configure_camera(view)
        pvs.Render(view)
        inputs.append(identity(case.admissible_path("vtu")))
    png = save_screenshot(
        output_dir / "free-resolution-geometry.png", layout, resolution
    )
    state = save_state(output_dir / "free-resolution-geometry.pvsm")
    return {
        "inputs": inputs,
        "displacement_y_range": list(displacement_range),
        "png": png,
        "pvsm": state,
    }


def render_free_resolution_activation_xy(
    free: list[Case], output_dir: Path, activation_range: tuple[float, float]
) -> dict[str, Any]:
    pvs.ResetSession()
    layout = pvs.CreateLayout(name="Free-control signed ActivationXY resolution")
    locations = split_even(layout, 0, len(free))
    resolution = (1800, 1500)
    layout.SetSize(*resolution)
    inputs: list[dict[str, Any]] = []
    for location, case in zip(locations, free, strict=True):
        reader = open_data(
            case.admissible_path("vtu"),
            f"{case.identifier} signed ActivationXY",
            temporal=False,
        )
        validate_orientation_preserving(reader, case)
        view = pvs.CreateView("RenderView")
        configure_view(view, crop=True)
        if not layout.AssignView(location, view):
            raise RuntimeError("failed to assign activation view")
        display_deformed(
            reader,
            view,
            array="ActivationXY",
            association="CELLS",
            scalar_range=activation_range,
            preset="Cool to Warm",
            edges=True,
            scalar_title="signed ActivationXY",
            threshold_muscle=True,
        )
        best = case.admissible_metrics
        add_text(
            view,
            f"{case.identifier} signed",
            f"Free controls, best admissible prefix | {case.resolution[0]} x {case.resolution[1]} | muscle cells = {case.payload['n_muscle_triangles']}\n"
            f"activation jump RMS = {float(best['activation_neighbor_jump_rms']):.6f}\n"
            "Signed shear active-strain component; divergent colours identify positive and negative ActivationXY",
            font_size=13,
        )
        pvs.Render(view)
        configure_camera(view, crop=True)
        pvs.Render(view)
        inputs.append(identity(case.admissible_path("vtu")))
    png = save_screenshot(
        output_dir / "free-resolution-signed-activation-xy.png", layout, resolution
    )
    state = save_state(output_dir / "free-resolution-signed-activation-xy.pvsm")
    return {
        "inputs": inputs,
        "activation_xy_range": list(activation_range),
        "png": png,
        "pvsm": state,
    }


def render_control_comparison(
    summary: dict[str, Any],
    controls: list[Case],
    output_dir: Path,
    displacement_range: tuple[float, float],
) -> dict[str, Any]:
    pvs.ResetSession()
    layout = pvs.CreateLayout(name="Free tied regularized control comparison")
    locations = split_even(layout, 0, len(controls))
    resolution = (1800, 1500)
    layout.SetSize(*resolution)
    inputs: list[dict[str, Any]] = []
    for location, case in zip(locations, controls, strict=True):
        reader = open_data(
            case.admissible_path("vtu"),
            f"{case.identifier} control comparison",
            temporal=False,
        )
        validate_orientation_preserving(reader, case)
        view = pvs.CreateView("RenderView")
        configure_view(view)
        if not layout.AssignView(location, view):
            raise RuntimeError("failed to assign control view")
        display_deformed(
            reader,
            view,
            array="DisplacementY",
            association="POINTS",
            scalar_range=displacement_range,
            preset="Cool to Warm",
            edges=False,
            scalar_title="vertical displacement",
        )
        add_reference_lines(summary, view)
        best = case.admissible_metrics
        add_text(
            view,
            f"{case.identifier} controls",
            f"{case.variant} controls, best admissible prefix | {case.resolution[0]} x {case.resolution[1]}\n"
            f"top error RMS = {float(best['top_error_rms']):.5f} | high-pass RMS = {float(best['top_highpass_rms']):.6f}\n"
            f"control jump RMS = {float(best['activation_neighbor_jump_rms']):.6f}",
            font_size=13,
        )
        pvs.Render(view)
        configure_camera(view)
        pvs.Render(view)
        inputs.append(identity(case.admissible_path("vtu")))
    png = save_screenshot(
        output_dir / "free-tied-regularized-control-comparison.png", layout, resolution
    )
    state = save_state(output_dir / "free-tied-regularized-control-comparison.pvsm")
    return {
        "inputs": inputs,
        "displacement_y_range": list(displacement_range),
        "png": png,
        "pvsm": state,
    }


def render_admissibility_transition(case: Case, output_dir: Path) -> dict[str, Any]:
    """Contrast the last admissible prefix state with the algebraic global optimum."""
    pvs.ResetSession()
    prefix = case.admissible_path("vtu")
    global_best = case.path("best_vtu")
    ranges: list[tuple[float, float]] = []
    for path, name in (
        (prefix, "admissible-prefix detF probe"),
        (global_best, "global-best detF probe"),
    ):
        reader = open_data(path, f"{case.identifier} {name}", temporal=False)
        if path == prefix:
            validate_orientation_preserving(reader, case)
        else:
            validate_reader(reader, case)
        ranges.append(array_range(reader, "CELLS", "DetF"))
    det_f_range = union_ranges([*ranges, (0.0, 0.0)], symmetric=False)
    layout = pvs.CreateLayout(name="Admissibility transition evidence")
    locations = split_even(layout, 0, 2)
    resolution = (1800, 1000)
    layout.SetSize(*resolution)
    global_is_valid = bool(case.payload["global_best_is_orientation_preserving"])
    global_note = (
        "Global objective best remains orientation-preserving for this case."
        if global_is_valid
        else "INADMISSIBLE algebraic output: shown only as failure evidence, not a physical geometry."
    )
    states = (
        (
            prefix,
            "Admissible-prefix best",
            "Verified pre-inversion equilibrium state; suitable for physical comparison.",
        ),
        (
            global_best,
            "Global objective best",
            global_note,
        ),
    )
    inputs: list[dict[str, Any]] = []
    first_invalid = case.payload.get("first_invalid_step")
    comparison_step = int(case.payload["best_admissible_prefix_step"])
    for location, (path, title, note) in zip(locations, states, strict=True):
        reader = open_data(path, f"{case.identifier} {title}", temporal=False)
        validate_reader(reader, case)
        view = pvs.CreateView("RenderView")
        configure_view(view)
        if not layout.AssignView(location, view):
            raise RuntimeError("failed to assign admissibility-transition view")
        display_deformed(
            reader,
            view,
            array="DetF",
            association="CELLS",
            scalar_range=det_f_range,
            preset="Cool to Warm",
            edges=True,
            scalar_title="det F (negative = inverted)",
        )
        step_label = (
            f"verified comparison evaluation = {comparison_step}"
            if path == prefix
            else f"global objective best evaluation = {int(case.payload['best_step'])}"
        )
        add_text(
            view,
            f"{case.identifier} {title}",
            f"{title} | {case.resolution[0]} x {case.resolution[1]}\n"
            f"{step_label}\n"
            f"first invalid evaluation = {first_invalid if first_invalid is not None else 'none'}\n"
            f"{note}",
            font_size=14,
        )
        pvs.Render(view)
        configure_camera(view)
        pvs.Render(view)
        inputs.append(identity(path))
    png = save_screenshot(
        output_dir / "admissibility-transition.png", layout, resolution
    )
    state = save_state(output_dir / "admissibility-transition.pvsm")
    return {
        "inputs": inputs,
        "first_invalid_step": first_invalid,
        "verified_comparison_step": comparison_step,
        "det_f_range": list(det_f_range),
        "global_best_orientation_preserving": global_is_valid,
        "png": png,
        "pvsm": state,
    }


def read_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, list[float]]:
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(columns) - set(reader.fieldnames):
            raise ValueError(f"missing CSV columns in {path}")
        for row in reader:
            for column in columns:
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {column} in {path}")
                values[column].append(value)
    if not all(values.values()):
        raise ValueError(f"empty CSV: {path}")
    return values


def configure_chart(
    view: Any,
    *,
    title: str,
    x_title: str,
    y_title: str,
    y_range: tuple[float, float] | None = None,
    log_y: bool = False,
) -> None:
    # XYChartView has no RenderView background properties in ParaView 6.1.1.
    # Keep its white plotting surface, with explicit dark labels for export.
    view.ChartTitle = title
    view.ChartTitleColor = [0.08, 0.10, 0.13]
    view.BottomAxisTitle = x_title
    view.LeftAxisTitle = y_title
    for property_name in (
        "BottomAxisColor",
        "BottomAxisLabelColor",
        "BottomAxisTitleColor",
        "LeftAxisColor",
        "LeftAxisLabelColor",
        "LeftAxisTitleColor",
    ):
        setattr(view, property_name, [0.08, 0.10, 0.13])
    view.BottomAxisGridColor = [0.78, 0.80, 0.83]
    view.LeftAxisGridColor = [0.78, 0.80, 0.83]
    view.ShowLegend = 1
    view.LegendLocation = "Bottom Right"
    view.ShowBottomAxisGrid = 1
    view.ShowLeftAxisGrid = 1
    view.LeftAxisLogScale = int(log_y)
    if y_range is not None:
        view.LeftAxisUseCustomRange = 1
        view.LeftAxisRangeMinimum = float(y_range[0])
        view.LeftAxisRangeMaximum = float(y_range[1])


def show_csv_series(
    reader: Any,
    view: Any,
    *,
    x: str,
    visible: str,
    label: str,
    color: tuple[float, float, float],
) -> None:
    representation = pvs.Show(reader, view, "XYChartRepresentation")
    representation.UseIndexForXAxis = 0
    representation.XArrayName = x
    representation.SeriesVisibility = [visible, "1"]
    representation.SeriesLabel = [visible, label]
    representation.SeriesColor = [visible, *(str(component) for component in color)]
    representation.SeriesLineThickness = [visible, "2"]


def render_top_profiles(free: list[Case], output_dir: Path) -> dict[str, Any]:
    pvs.ResetSession()
    profile_data = [
        read_csv_columns(
            case.admissible_path("profile"),
            ("x", "uy", "target_uy", "highpass_uy"),
        )
        for case in free
    ]
    y_values = [
        value for data in profile_data for value in data["uy"] + data["target_uy"]
    ]
    highpass = [value for data in profile_data for value in data["highpass_uy"]]
    top_range = union_ranges([(min(y_values), max(y_values))], symmetric=False)
    highpass_range = union_ranges([(min(highpass), max(highpass))], symmetric=True)
    layout = pvs.CreateLayout(name="Top-profile and high-pass resolution evidence")
    locations = split_even(layout, 0, 2)
    top_view = pvs.CreateView("XYChartView")
    high_view = pvs.CreateView("XYChartView")
    configure_chart(
        top_view,
        title="Top-boundary displacement profiles (best admissible prefix)",
        x_title="x (model length)",
        y_title="u_y (model length)",
        y_range=top_range,
    )
    configure_chart(
        high_view,
        title="Top-profile high-pass residual, best admissible prefix (fixed physical Gaussian width = 0.02)",
        x_title="x (model length)",
        y_title="u_y - Gaussian(u_y)",
        y_range=highpass_range,
    )
    if not layout.AssignView(locations[0], top_view) or not layout.AssignView(
        locations[1], high_view
    ):
        raise RuntimeError("failed to assign profile charts")
    resolution = (1800, 1200)
    layout.SetSize(*resolution)
    colors = ((0.20, 0.70, 0.90), (0.96, 0.62, 0.20), (0.42, 0.82, 0.46))
    inputs: list[dict[str, Any]] = []
    for index, (case, color) in enumerate(zip(free, colors, strict=False)):
        reader = pvs.CSVReader(
            registrationName=f"{case.identifier} admissible-prefix profile",
            FileName=[str(case.admissible_path("profile"))],
        )
        reader.UpdatePipeline()
        show_csv_series(
            reader,
            top_view,
            x="x",
            visible="uy",
            label=f"{case.resolution[0]}x{case.resolution[1]} free u_y",
            color=color,
        )
        show_csv_series(
            reader,
            high_view,
            x="x",
            visible="highpass_uy",
            label=f"{case.resolution[0]}x{case.resolution[1]} high-pass",
            color=color,
        )
        if index == 0:
            target_reader = pvs.CSVReader(
                registrationName=f"{case.identifier} requested target profile",
                FileName=[str(case.admissible_path("profile"))],
            )
            target_reader.UpdatePipeline()
            show_csv_series(
                target_reader,
                top_view,
                x="x",
                visible="target_uy",
                label="requested uniform target",
                color=(0.90, 0.28, 0.62),
            )
        inputs.append(identity(case.admissible_path("profile")))
    pvs.Render(top_view)
    pvs.Render(high_view)
    png = save_screenshot(
        output_dir / "free-resolution-top-profiles.png", layout, resolution
    )
    state = save_state(output_dir / "free-resolution-top-profiles.pvsm")
    return {
        "inputs": inputs,
        "top_y_range": list(top_range),
        "highpass_y_range": list(highpass_range),
        "png": png,
        "pvsm": state,
    }


def render_spectra(free: list[Case], output_dir: Path) -> dict[str, Any]:
    pvs.ResetSession()
    spectrum_data = [
        read_csv_columns(
            case.admissible_path("spectrum"), ("cycles_per_unit_length", "power")
        )
        for case in free
    ]
    positive_frequency = [
        value
        for data in spectrum_data
        for value in data["cycles_per_unit_length"]
        if value > 0.0
    ]
    positive_power = [
        value for data in spectrum_data for value in data["power"] if value > 0.0
    ]
    if not positive_frequency or not positive_power:
        raise ValueError("spectrum has no positive frequencies/power")
    common_nyquist = min(max(data["cycles_per_unit_length"]) for data in spectrum_data)
    visible_power = [
        value
        for data in spectrum_data
        for f, value in zip(data["cycles_per_unit_length"], data["power"], strict=True)
        if f > 0.0 and f <= common_nyquist and value > 0.0
    ]
    if not visible_power:
        raise ValueError("no positive spectrum power in common Nyquist band")
    layout = pvs.CreateLayout(name="Free-control spatial spectra")
    view = pvs.CreateView("XYChartView")
    configure_chart(
        view,
        title="Spatial spectrum of mean-centred top displacement (best admissible prefix)",
        x_title="cycles per unit length (DC excluded; common Nyquist band)",
        y_title="power",
        y_range=(min(visible_power), max(visible_power)),
        log_y=True,
    )
    view.BottomAxisUseCustomRange = 1
    view.BottomAxisRangeMinimum = min(positive_frequency)
    view.BottomAxisRangeMaximum = common_nyquist
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign spectrum chart")
    resolution = (1800, 800)
    layout.SetSize(*resolution)
    colors = ((0.20, 0.70, 0.90), (0.96, 0.62, 0.20), (0.42, 0.82, 0.46))
    inputs: list[dict[str, Any]] = []
    for case, color in zip(free, colors, strict=False):
        reader = pvs.CSVReader(
            registrationName=f"{case.identifier} admissible-prefix spectrum",
            FileName=[str(case.admissible_path("spectrum"))],
        )
        reader.UpdatePipeline()
        show_csv_series(
            reader,
            view,
            x="cycles_per_unit_length",
            visible="power",
            label=f"{case.resolution[0]}x{case.resolution[1]} free",
            color=color,
        )
        inputs.append(identity(case.admissible_path("spectrum")))
    pvs.Render(view)
    png = save_screenshot(
        output_dir / "free-resolution-spatial-spectra.png", layout, resolution
    )
    state = save_state(output_dir / "free-resolution-spatial-spectra.pvsm")
    return {
        "inputs": inputs,
        "common_nyquist": common_nyquist,
        "png": png,
        "pvsm": state,
    }


def render_determinant_history(case: Case, output_dir: Path) -> dict[str, Any]:
    """Render the signed determinant trace and the exact first-invalid boundary."""
    pvs.ResetSession()
    trace_path = case.path("trace")
    columns = ("step", "min_det_f", "min_det_g", "min_det_ainv")
    trace = read_csv_columns(trace_path, columns)
    expected_steps = [float(step) for step in range(int(case.payload["evaluations"]))]
    if trace["step"] != expected_steps:
        raise ValueError(
            f"trace steps are not an exact inverse-evaluation sequence: {case.identifier}"
        )
    zero_path = output_dir / "determinant-history-zero-line.csv"
    with zero_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "zero"))
        writer.writerow((trace["step"][0], 0.0))
        writer.writerow((trace["step"][-1], 0.0))
    values = [value for name in columns[1:] for value in trace[name]]
    determinant_range = union_ranges(
        [(min(values), max(values)), (0.0, 0.0)], symmetric=False
    )
    first_invalid = case.payload.get("first_invalid_step")
    boundary = (
        f"first invalid evaluation = {first_invalid} (subsequent geometry is failure evidence)"
        if first_invalid is not None
        else "all stored evaluations remain orientation-preserving"
    )
    layout = pvs.CreateLayout(name="Determinant admissibility history")
    view = pvs.CreateView("XYChartView")
    configure_chart(
        view,
        title=f"Minimum determinants across inverse evaluations — {boundary}",
        x_title="inverse evaluation (exact saved step)",
        y_title="minimum determinant",
        y_range=determinant_range,
    )
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign determinant-history chart")
    resolution = (1800, 850)
    layout.SetSize(*resolution)
    for field, label, color in (
        ("min_det_f", "min det F (deformation)", (0.20, 0.70, 0.90)),
        ("min_det_g", "min det G (elastic)", (0.96, 0.62, 0.20)),
        ("min_det_ainv", "min det A^-1 (active inverse)", (0.84, 0.36, 0.58)),
    ):
        reader = pvs.CSVReader(
            registrationName=f"{case.identifier} {field} trace",
            FileName=[str(trace_path)],
        )
        reader.UpdatePipeline()
        show_csv_series(reader, view, x="step", visible=field, label=label, color=color)
    zero_reader = pvs.CSVReader(
        registrationName="zero determinant reference", FileName=[str(zero_path)]
    )
    zero_reader.UpdatePipeline()
    show_csv_series(
        zero_reader,
        view,
        x="step",
        visible="zero",
        label="orientation boundary (zero)",
        color=(0.12, 0.12, 0.12),
    )
    pvs.Render(view)
    png = save_screenshot(output_dir / "determinant-history.png", layout, resolution)
    state = save_state(output_dir / "determinant-history.pvsm")
    return {
        "trace": identity(trace_path),
        "zero_line": identity(zero_path),
        "first_invalid_step": first_invalid,
        "determinant_range": list(determinant_range),
        "png": png,
        "pvsm": state,
    }


def render_evolution(
    summary: dict[str, Any], case: Case, output_dir: Path, *, fps: int
) -> dict[str, Any]:
    time_values, displacement_range, activation_range = scan_evolution_ranges(case)
    pvs.ResetSession()
    reader = open_data(
        case.path("series"), f"{case.identifier} exact evolution", temporal=True
    )
    validate_reader(reader, case, time=time_values[0])
    shape_view = pvs.CreateView("RenderView")
    activation_view = pvs.CreateView("RenderView")
    configure_view(shape_view)
    configure_view(activation_view, crop=True)
    layout = pvs.CreateLayout(name="Exact Stable Neo-Hookean inverse evolution")
    locations = split_even(layout, 0, 2)
    if not layout.AssignView(locations[0], shape_view) or not layout.AssignView(
        locations[1], activation_view
    ):
        raise RuntimeError("failed to assign evolution views")
    resolution = (1800, 1200)
    layout.SetSize(*resolution)
    display_deformed(
        reader,
        shape_view,
        array="DisplacementY",
        association="POINTS",
        scalar_range=displacement_range,
        preset="Cool to Warm",
        edges=False,
        scalar_title="vertical displacement",
    )
    display_deformed(
        reader,
        activation_view,
        array="ActivationYY",
        association="CELLS",
        scalar_range=activation_range,
        preset="Cool to Warm",
        edges=True,
        scalar_title="signed ActivationYY",
        threshold_muscle=True,
    )
    add_reference_lines(summary, shape_view)
    first_invalid = case.payload.get("first_invalid_step")
    comparison_step = int(case.payload["best_admissible_prefix_step"])
    boundary = (
        f"VERIFIED COMPARISON = {comparison_step}; first orientation-invalid evaluation = {first_invalid}. Later frames are failure evidence only."
        if first_invalid is not None
        else f"VERIFIED COMPARISON = {comparison_step}; all stored evaluations remain orientation-preserving."
    )
    add_text(
        shape_view,
        "Evolution displacement",
        "Exact inverse history: deformed reference mesh, color = full-field u_y\nMagenta = requested uniform top target; shared no-clipping color range\n"
        + boundary,
        font_size=14,
    )
    add_text(
        activation_view,
        "Evolution activation",
        "Same evaluation: signed vertical active-strain component on muscle cells\nNo temporal interpolation; triangle edges identify the control basis\n"
        + boundary,
        font_size=14,
    )
    add_time_annotation(reader, shape_view, "Displacement evaluation")
    add_time_annotation(reader, activation_view, "Activation evaluation")
    scene = pvs.GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.PlayMode = "Snap To TimeSteps"
    actual_times = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    if actual_times != time_values:
        raise ValueError("timekeeper changed after animation setup")
    pvs.Render(shape_view)
    pvs.Render(activation_view)
    configure_camera(shape_view)
    configure_camera(activation_view, crop=True)
    pvs.Render(shape_view)
    pvs.Render(activation_view)

    frames_dir = output_dir / "evolution-frames"
    frames_dir.mkdir(parents=False, exist_ok=False)
    pvs.SaveAnimation(
        str(frames_dir / "evolution.png"),
        layout,
        scene,
        FrameWindow=[0, len(time_values) - 1],
        SuffixFormat="_{:03d}",
        ImageResolution=list(resolution),
        FontScaling="Do not scale fonts",
    )
    frames = sorted(frames_dir.glob("evolution_*.png"))
    if len(frames) != len(time_values):
        raise ValueError(
            f"ParaView wrote {len(frames)} frames for {len(time_values)} exact evaluations"
        )
    rendered_resolution = png_size(frames[0])
    if (
        rendered_resolution[0] != resolution[0]
        or abs(rendered_resolution[1] - resolution[1]) > 16
    ):
        raise ValueError(f"unexpected evolution frame size: {rendered_resolution}")
    for frame in frames:
        if png_size(frame) != rendered_resolution or frame.stat().st_size < 20_000:
            raise ValueError(f"invalid evolution frame: {frame}")
    stills: dict[str, Any] = {}
    for label, index in (
        ("initial", 0),
        ("middle", len(frames) // 2),
        ("final", len(frames) - 1),
    ):
        destination = output_dir / f"evolution-{label}.png"
        shutil.copy2(frames[index], destination)
        if png_size(destination) != rendered_resolution:
            raise ValueError(f"invalid copied evolution still: {destination}")
        stills[label] = {
            **identity(destination),
            "time": time_values[index],
            "source_frame": str(frames[index].resolve()),
            "width": rendered_resolution[0],
            "height": rendered_resolution[1],
        }
    state = save_state(output_dir / "evolution.pvsm")
    video = encode_exact_frame_video(
        frames_dir,
        frames,
        output_dir / "evolution.mp4",
        fps=fps,
        resolution=rendered_resolution,
    )
    step_by_step_video = encode_exact_frame_video(
        frames_dir,
        frames,
        output_dir / "evolution-step-by-step.mp4",
        fps=2,
        resolution=rendered_resolution,
    )
    return {
        "series": identity(case.path("series")),
        "time_values": time_values,
        "frame_count": len(frames),
        "no_interpolation": True,
        "first_invalid_step": first_invalid,
        "verified_comparison_step": comparison_step,
        "displacement_y_range": list(displacement_range),
        "activation_yy_range": list(activation_range),
        "requested_frame_width": resolution[0],
        "requested_frame_height": resolution[1],
        "rendered_frame_width": rendered_resolution[0],
        "rendered_frame_height": rendered_resolution[1],
        "stills": stills,
        "pvsm": state,
        "video": video,
        "step_by_step_video": step_by_step_video,
        "frames": {
            "directory": str(frames_dir.resolve()),
            "first_sha256": sha256(frames[0]),
            "last_sha256": sha256(frames[-1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    version = paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, found {version}"
        )
    summary_path = args.summary.resolve()
    summary, cases = validate_summary(summary_path)
    free, controls, median = select_cases(cases)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    # Physical comparisons use only the best orientation-preserving prefix state.
    best_displacement_ranges: list[tuple[float, float]] = []
    best_activation_ranges: list[tuple[float, float]] = []
    for case in [*free, *controls]:
        pvs.ResetSession()
        reader = open_data(
            case.admissible_path("vtu"),
            f"{case.identifier} admissible range probe",
            temporal=False,
        )
        validate_orientation_preserving(reader, case)
        best_displacement_ranges.append(array_range(reader, "POINTS", "DisplacementY"))
        best_activation_ranges.append(array_range(reader, "CELLS", "ActivationXY"))
    pvs.ResetSession()
    shared_displacement_range = union_ranges(best_displacement_ranges, symmetric=False)
    shared_activation_range = union_ranges(best_activation_ranges, symmetric=True)

    setup = render_setup(summary, median, output_dir)
    geometry = render_free_resolution_geometry(
        summary, free, output_dir, shared_displacement_range
    )
    activation = render_free_resolution_activation_xy(
        free, output_dir, shared_activation_range
    )
    controls_asset = render_control_comparison(
        summary, controls, output_dir, shared_displacement_range
    )
    transition = render_admissibility_transition(median, output_dir)
    profiles = render_top_profiles(free, output_dir)
    spectra = render_spectra(free, output_dir)
    determinant_history = render_determinant_history(median, output_dir)
    evolution = render_evolution(summary, median, output_dir, fps=args.fps)
    receipt = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "design": EXPECTED_DESIGN,
        "complete": True,
        "status": "ok",
        "paraview_version": version,
        "native_paraview_rendering": True,
        "video_note": "ffmpeg encodes only PNG frames rendered by ParaView",
        "summary": identity(summary_path),
        "free_resolution_cases": [case.identifier for case in free],
        "median_control_comparison_resolution": list(median.resolution),
        "shared_admissible_prefix_displacement_y_range": list(
            shared_displacement_range
        ),
        "shared_admissible_prefix_activation_xy_range": list(shared_activation_range),
        "setup": setup,
        "free_resolution_geometry": geometry,
        "free_resolution_signed_activation_xy": activation,
        "free_tied_regularized_control_comparison": controls_asset,
        "admissibility_transition": transition,
        "top_profiles": profiles,
        "spatial_spectra": spectra,
        "determinant_history": determinant_history,
        "evolution": evolution,
    }
    write_json(output_dir / "render-receipt.json", receipt)


if __name__ == "__main__":
    main()
