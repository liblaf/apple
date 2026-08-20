from __future__ import annotations

# This source is executed by ParaView's pvbatch, not by the project interpreter.
# ruff: noqa: EM101, EM102, FBT003, TRY003
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
CASE_ORDER = ("thin", "medium", "thick")
CAMERA = {
    "position": [1.48, 1.20, 1.55],
    "focal_point": [0.50, 0.00, 0.50],
    "view_up": [0.0, 1.0, 0.0],
    "parallel_scale": 0.79,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def configure_camera(view: Any) -> None:
    view.CameraPosition = CAMERA["position"]
    view.CameraFocalPoint = CAMERA["focal_point"]
    view.CameraViewUp = CAMERA["view_up"]
    view.CenterOfRotation = CAMERA["focal_point"]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = CAMERA["parallel_scale"]


def configure_view(name: str, resolution: tuple[int, int]) -> tuple[Any, Any]:
    view = pvs.CreateView("RenderView")
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 1
    configure_camera(view)
    layout = pvs.CreateLayout(name=name)
    if not layout.AssignView(0, view):
        raise RuntimeError(f"failed to assign render view for {name}")
    layout.SetSize(*resolution)
    return view, layout


def add_text(view: Any, name: str, text: str) -> None:
    label = pvs.Text(registrationName=name)
    label.Text = text
    display = pvs.Show(label, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 20
    display.Color = [1.0, 1.0, 1.0]
    display.Bold = 1


def add_scalar_bar(
    view: Any,
    display: Any,
    array_name: str,
    scalar_range: tuple[float, float],
    title: str,
    component_title: str,
) -> None:
    pvs.ColorBy(display, ("POINTS", array_name))
    lut = pvs.GetColorTransferFunction(array_name)
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(*scalar_range)
    display.SetScalarBarVisibility(view, True)
    scalar_bar = pvs.GetScalarBar(lut, view)
    scalar_bar.Title = title
    scalar_bar.ComponentTitle = component_title
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Lower Center"
    scalar_bar.ScalarBarLength = 0.48
    scalar_bar.ScalarBarThickness = 18
    scalar_bar.TitleColor = [1.0, 1.0, 1.0]
    scalar_bar.LabelColor = [1.0, 1.0, 1.0]
    scalar_bar.TitleFontSize = 15
    scalar_bar.LabelFontSize = 13


def save_pair(
    output_dir: Path,
    stem: str,
    layout: Any,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    png = output_dir / f"{stem}.png"
    state = output_dir / f"{stem}.pvsm"
    png_tmp = png.with_name(f".{png.stem}.tmp{png.suffix}")
    state_tmp = state.with_name(f".{state.stem}.tmp{state.suffix}")
    pvs.SaveScreenshot(
        str(png_tmp),
        layout,
        ImageResolution=list(resolution),
        TransparentBackground=0,
        FontScaling="Scale fonts proportionally",
    )
    pvs.SaveState(str(state_tmp))
    if png_size(png_tmp) != resolution or png_tmp.stat().st_size < 80_000:
        raise ValueError(f"invalid ParaView screenshot: {png_tmp}")
    if (
        "ServerManagerState"
        not in state_tmp.read_text(encoding="utf-8", errors="strict")[:2048]
    ):
        raise ValueError(f"invalid ParaView state: {state_tmp}")
    png_tmp.replace(png)
    state_tmp.replace(state)
    return {
        "png": {
            "relative_path": str(png.relative_to(output_dir.parent)),
            "size_bytes": png.stat().st_size,
            "sha256": sha256(png),
            "width": resolution[0],
            "height": resolution[1],
        },
        "pvsm": {
            "relative_path": str(state.relative_to(output_dir.parent)),
            "size_bytes": state.stat().st_size,
            "sha256": sha256(state),
        },
    }


def open_reader(path: Path, name: str) -> Any:
    reader = pvs.XMLUnstructuredGridReader(
        registrationName=name,
        FileName=[str(path.resolve())],
    )
    reader.PointArrayStatus = [
        "TopSurfaceMask",
        "BumpyMinusUniform",
        "BumpyMinusUniformY",
    ]
    reader.CellArrayStatus = [
        "ActivationMask",
        "ActivationInvXModulation",
    ]
    reader.UpdatePipeline()
    return reader


def array_range(path: Path, association: str, name: str) -> tuple[float, float]:
    pvs.ResetSession()
    reader = open_reader(path, f"range probe for {name}")
    info = reader.GetDataInformation()
    data_info = (
        info.GetPointDataInformation()
        if association == "POINTS"
        else info.GetCellDataInformation()
    )
    array_info = data_info.GetArrayInformation(name)
    if array_info is None:
        raise KeyError(f"missing {association} array {name!r} in {path}")
    result = tuple(float(value) for value in array_info.GetComponentRange(0))
    pvs.ResetSession()
    return result


def render_surface_case(
    case: dict[str, Any],
    path: Path,
    output_root: Path,
    resolution: tuple[int, int],
    warp_factor: float,
    color_range: tuple[float, float],
) -> dict[str, Any]:
    from paraview import servermanager

    label = str(case["label"])
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    reader = open_reader(path, f"{label}: corrected bumpy activation transfer")
    info = reader.GetDataInformation()
    if info.GetNumberOfPoints() != int(case["n_points"]):
        raise ValueError(f"point count changed for {label}")
    if info.GetNumberOfCells() != int(case["n_tets"]):
        raise ValueError(f"cell count changed for {label}")

    outer = pvs.ExtractSurface(registrationName="Rest outer surface", Input=reader)
    top = pvs.Threshold(registrationName="Top surface only", Input=outer)
    top.Scalars = ["POINTS", "TopSurfaceMask"]
    top.LowerThreshold = 0.5
    top.UpperThreshold = 1.0
    top.AllScalars = 1
    top.UpdatePipeline()
    fetched_top = servermanager.Fetch(top)
    if (
        fetched_top.GetNumberOfPoints() != 2401
        or fetched_top.GetNumberOfCells() != 4608
    ):
        raise ValueError(f"unexpected top grid for {label}")

    normalized = pvs.Transform(
        registrationName="Normalize top rest plane to y=0",
        Input=top,
    )
    normalized.Transform.Translate = [0.0, -float(case["total_height"]), 0.0]
    vertical = pvs.Calculator(
        registrationName="Vertical bumpy-minus-uniform vector",
        Input=normalized,
    )
    vertical.ResultArrayName = "VerticalBumpyMinusUniform"
    vertical.Function = "BumpyMinusUniformY*jHat"
    warped = pvs.WarpByVector(
        registrationName=f"Vertical response warped x{warp_factor:g}",
        Input=vertical,
    )
    warped.Vectors = ["POINTS", "VerticalBumpyMinusUniform"]
    warped.ScaleFactor = warp_factor

    view, layout = configure_view(f"{label} standalone response", resolution)
    display = pvs.Show(warped, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.10, 0.11, 0.14]
    display.LineWidth = 0.35
    display.Ambient = 0.22
    display.Diffuse = 0.78
    display.Specular = 0.15
    display.SpecularPower = 20.0
    add_scalar_bar(
        view,
        display,
        "BumpyMinusUniformY",
        color_range,
        "bumpy - uniform vertical displacement",
        "model length",
    )
    add_text(
        view,
        f"{label} annotation",
        (
            f"{label.capitalize()} top fat: thickness = "
            f"{float(case['top_fat_thickness']):.2f}\n"
            "Bumpy activation response minus uniform-activation response\n"
            f"Vertical warp x{warp_factor:g}; shared camera and color scale\n"
            f"Top RMS = {float(case['surface/top_induced_y_rms']):.3e}  |  "
            f"mode amplitude = {float(case['surface/top_modal_amplitude_abs']):.3e}"
        ),
    )
    pvs.Render(view)
    configure_camera(view)
    pvs.Render(view)

    case_dir = output_root / label
    case_dir.mkdir(parents=False, exist_ok=False)
    pair = save_pair(
        case_dir,
        f"21-{label}-bumpy-minus-uniform-paraview",
        layout,
        resolution,
    )
    pair.update(
        {
            "label": label,
            "input_path": str(path.resolve()),
            "input_size_bytes": path.stat().st_size,
            "input_sha256": sha256(path),
            "n_points": info.GetNumberOfPoints(),
            "n_tets": info.GetNumberOfCells(),
            "top_points": fetched_top.GetNumberOfPoints(),
            "top_triangles": fetched_top.GetNumberOfCells(),
            "top_fat_thickness": float(case["top_fat_thickness"]),
            "total_height": float(case["total_height"]),
            "surface_top_induced_y_rms": float(case["surface/top_induced_y_rms"]),
            "surface_top_modal_amplitude_abs": float(
                case["surface/top_modal_amplitude_abs"]
            ),
        }
    )
    return pair


def render_activation_source(
    thin_path: Path,
    output_root: Path,
    resolution: tuple[int, int],
    color_range: tuple[float, float],
) -> dict[str, Any]:
    from paraview import servermanager

    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    reader = open_reader(thin_path, "Shared bumpy muscle activation source")
    active = pvs.Threshold(registrationName="Active muscle tetrahedra", Input=reader)
    active.Scalars = ["CELLS", "ActivationMask"]
    active.LowerThreshold = 0.5
    active.UpperThreshold = 1.0
    active.UpdatePipeline()
    fetched_active = servermanager.Fetch(active)
    if fetched_active.GetNumberOfCells() != 27648:
        raise ValueError("active muscle cell count changed")
    source_surface = pvs.ExtractSurface(
        registrationName="Muscle-layer boundary",
        Input=active,
    )
    centered = pvs.Transform(
        registrationName="Center muscle layer around y=0",
        Input=source_surface,
    )
    centered.Transform.Translate = [0.0, -0.05, 0.0]

    view, layout = configure_view("shared activation source", resolution)
    display = pvs.Show(centered, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.10, 0.11, 0.14]
    display.LineWidth = 0.30
    display.Ambient = 0.24
    display.Diffuse = 0.76
    pvs.ColorBy(display, ("CELLS", "ActivationInvXModulation"))
    lut = pvs.GetColorTransferFunction("ActivationInvXModulation")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(*color_range)
    display.SetScalarBarVisibility(view, True)
    scalar_bar = pvs.GetScalarBar(lut, view)
    scalar_bar.Title = "ActivationInv x modulation"
    scalar_bar.ComponentTitle = "dimensionless"
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Lower Center"
    scalar_bar.ScalarBarLength = 0.48
    scalar_bar.ScalarBarThickness = 18
    scalar_bar.TitleColor = [1.0, 1.0, 1.0]
    scalar_bar.LabelColor = [1.0, 1.0, 1.0]
    scalar_bar.TitleFontSize = 15
    scalar_bar.LabelFontSize = 13
    add_text(
        view,
        "activation source annotation",
        (
            "Shared bumpy muscle-activation source (all thickness cases)\n"
            "Cell color: zero-mean ActivationInv x modulation; no warp\n"
            "Same active-cell IDs, x-z centroids, and activation hash"
        ),
    )
    pvs.Render(view)
    configure_camera(view)
    pvs.Render(view)

    source_dir = output_root / "source"
    source_dir.mkdir(parents=False, exist_ok=False)
    pair = save_pair(
        source_dir,
        "21-shared-bumpy-activation-source-paraview",
        layout,
        resolution,
    )
    pair.update(
        {
            "input_path": str(thin_path.resolve()),
            "input_size_bytes": thin_path.stat().st_size,
            "input_sha256": sha256(thin_path),
            "active_tets": fetched_active.GetNumberOfCells(),
        }
    )
    return pair


def validate_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary.get("status") != "ok" or summary.get("complete") is not True:
        raise ValueError("production simulation summary is not complete")
    cases = summary.get("cases")
    if not isinstance(cases, list) or [row.get("label") for row in cases] != list(
        CASE_ORDER
    ):
        raise ValueError("unexpected production case order")
    invariants = summary.get("shared_invariants")
    if not isinstance(invariants, dict):
        raise TypeError("missing shared-invariants record")
    for row in cases:
        for key in (
            "active_ids_sha256",
            "active_centers_xz_sha256",
            "bumpy_activation_sha256",
        ):
            if row.get(key) != invariants.get(key):
                raise ValueError(f"shared activation invariant failed: {key}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--renderer-receipt", required=True, type=Path)
    parser.add_argument("--warp-factor", type=float, default=40.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1800, 1350))
    args = parser.parse_args()

    version = paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, found {version}"
        )
    if not math.isfinite(args.warp_factor) or args.warp_factor <= 0.0:
        raise ValueError("warp factor must be finite and positive")
    resolution = tuple(args.resolution)
    if min(resolution) < 800:
        raise ValueError("render resolution is too small for review")
    input_root = args.input_root.resolve()
    summary_path = args.summary.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.renderer_receipt.resolve()
    if not input_root.is_dir() or not summary_path.is_file():
        raise FileNotFoundError("production input root or summary is missing")
    if output_root.exists():
        raise FileExistsError(f"output root must not already exist: {output_root}")
    output_root.mkdir(parents=True)

    summary = read_json(summary_path)
    cases = validate_summary(summary)
    paths = {
        label: input_root / label / f"10-{label}-bumpy-activation-transfer.vtu"
        for label in CASE_ORDER
    }
    if missing := [path for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"production VTUs are missing: {missing}")

    response_ranges = [
        array_range(path, "POINTS", "BumpyMinusUniformY") for path in paths.values()
    ]
    response_abs_max = max(abs(value) for pair in response_ranges for value in pair)
    response_abs_max = math.ceil(response_abs_max * 100_000.0) / 100_000.0
    response_color_range = (-response_abs_max, response_abs_max)
    source_range = array_range(paths["thin"], "CELLS", "ActivationInvXModulation")
    source_abs_max = max(abs(value) for value in source_range)
    source_abs_max = math.ceil(source_abs_max * 100.0) / 100.0
    source_color_range = (-source_abs_max, source_abs_max)

    rendered_cases = [
        render_surface_case(
            case,
            paths[str(case["label"])],
            output_root,
            resolution,
            args.warp_factor,
            response_color_range,
        )
        for case in cases
    ]
    activation_source = render_activation_source(
        paths["thin"],
        output_root,
        resolution,
        source_color_range,
    )
    receipt = {
        "schema_version": 1,
        "design": "fat-thickness-bumpy-activation-transfer-paraview-v1",
        "complete": True,
        "status": "ok",
        "native_paraview_rendering": True,
        "paraview_version": version,
        "resolution": list(resolution),
        "surface_visualization": {
            "response": "BumpyMinusUniformY",
            "vertical_warp_only": True,
            "warp_factor": args.warp_factor,
            "top_rest_plane_normalized_to_y": 0.0,
            "shared_response_color_range": list(response_color_range),
            "shared_camera": CAMERA,
        },
        "activation_visualization": {
            "response": "ActivationInvXModulation",
            "warp_factor": 0.0,
            "shared_source_color_range": list(source_color_range),
        },
        "summary_sha256": sha256(summary_path),
        "cases": rendered_cases,
        "activation_source": activation_source,
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
