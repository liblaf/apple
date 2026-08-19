from __future__ import annotations

# Executed by ParaView's pvbatch, not by the project Python.
# ruff: noqa: EM101, EM102, FBT003, PLR0915, TRY003
import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_DESIGN = "cuboid-fat-thickness-meeting-assets-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_contract(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("ParaView contract must be an object")
    if (
        value.get("schema_version") != EXPECTED_SCHEMA_VERSION
        or value.get("design") != EXPECTED_DESIGN
    ):
        raise ValueError("ParaView contract schema/design changed")
    if value.get("complete") is not True:
        raise ValueError("ParaView contract is incomplete")
    return value


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


def validate_case(case: dict[str, Any], input_root: Path) -> Path:
    path = Path(str(case["input_path"])).resolve()
    if input_root.resolve() not in path.parents:
        raise ValueError(f"input escapes pinned root: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    expected = {
        "size_bytes": int(case["input_size_bytes"]),
        "sha256": str(case["input_sha256"]),
    }
    if actual != expected:
        raise ValueError(f"input identity changed for {case['case_id']}")
    return path


def show_label(view: Any, case: dict[str, Any]) -> None:
    metrics = case["metrics"]
    label = pvs.Text(registrationName=f"{case['case_id']} label")
    label.Text = (
        f"Top fat thickness = {float(case['top_fat_thickness']):.2f}  |  "
        f"Bottom pressure = {float(case['pressure']):.2f}\n"
        f"p95-p05 = {float(metrics['p95_p05']):.5f}  |  "
        f"Laplacian RMS = {float(metrics['laplacian_rms']):.4f}\n"
        "White outline: rest shape  |  Color: vertical displacement"
    )
    display = pvs.Show(label, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 18
    display.Color = [1.0, 1.0, 1.0]
    display.Bold = 1


def configure_camera(view: Any, contract: dict[str, Any]) -> None:
    camera = contract["camera"]
    view.CameraPosition = [float(value) for value in camera["position"]]
    view.CameraFocalPoint = [float(value) for value in camera["focal_point"]]
    view.CameraViewUp = [float(value) for value in camera["view_up"]]
    view.CenterOfRotation = [float(value) for value in camera["focal_point"]]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = float(camera["parallel_scale"])


def render_case(
    case: dict[str, Any], contract: dict[str, Any], input_root: Path, output_root: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    input_path = validate_case(case, input_root)
    case_dir = output_root / str(case["case_id"])
    case_dir.mkdir(parents=False, exist_ok=False)

    reader = pvs.XMLUnstructuredGridReader(
        registrationName=f"{case['case_id']} pressure 0.60",
        FileName=[str(input_path)],
    )
    reader.PointArrayStatus = ["Displacement", "DisplacementNorm"]
    reader.CellArrayStatus = ["EffectiveYoungModulus", "SmasFraction", "FatFraction"]
    reader.UpdatePipeline()

    from paraview import servermanager

    fetched = servermanager.Fetch(reader)
    if fetched.GetNumberOfPoints() != int(case["n_points"]):
        raise ValueError(f"point count changed for {case['case_id']}")
    if fetched.GetNumberOfCells() != int(case["n_cells"]):
        raise ValueError(f"cell count changed for {case['case_id']}")
    displacement = fetched.GetPointData().GetArray("Displacement")
    if displacement is None or displacement.GetNumberOfComponents() != 3:
        raise KeyError(f"Displacement vector missing for {case['case_id']}")

    warp = pvs.WarpByVector(registrationName="Deformed shape", Input=reader)
    warp.Vectors = ["POINTS", "Displacement"]
    warp.ScaleFactor = 1.0
    deformed_surface = pvs.ExtractSurface(
        registrationName="Deformed outer surface", Input=warp
    )
    rest_surface = pvs.ExtractSurface(
        registrationName="Rest outer surface", Input=reader
    )
    rest_edges = pvs.FeatureEdges(
        registrationName="Rest feature-edge outline", Input=rest_surface
    )
    rest_edges.BoundaryEdges = 1
    rest_edges.FeatureEdges = 1
    rest_edges.ManifoldEdges = 0
    rest_edges.NonManifoldEdges = 1
    rest_edges.FeatureAngle = 30.0

    view = pvs.CreateView("RenderView")
    view.Background = [0.043, 0.051, 0.063]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    configure_camera(view, contract)
    layout = pvs.CreateLayout(name=f"{case['case_id']} standalone")
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign ParaView render view")
    resolution = [int(value) for value in contract["image_resolution"]]
    layout.SetSize(*resolution)

    display = pvs.Show(deformed_surface, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.12, 0.13, 0.15]
    display.LineWidth = 0.35
    display.Ambient = 0.20
    display.Diffuse = 0.78
    display.Specular = 0.18
    display.SpecularPower = 22.0
    pvs.ColorBy(display, ("POINTS", "Displacement", "Y"))
    lut = pvs.GetColorTransferFunction("Displacement")
    lut.ApplyPreset("Cool to Warm", True)
    lut.VectorMode = "Component"
    lut.VectorComponent = 1
    color_range = [float(value) for value in contract["shared_u_y_range"]]
    lut.RescaleTransferFunction(*color_range)
    display.SetScalarBarVisibility(view, True)
    scalar_bar = pvs.GetScalarBar(lut, view)
    scalar_bar.Title = "vertical displacement"
    scalar_bar.ComponentTitle = "model length"
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Lower Center"
    scalar_bar.ScalarBarLength = 0.46
    scalar_bar.ScalarBarThickness = 18
    scalar_bar.TitleColor = [1.0, 1.0, 1.0]
    scalar_bar.LabelColor = [1.0, 1.0, 1.0]
    scalar_bar.TitleFontSize = 15
    scalar_bar.LabelFontSize = 13

    outline_display = pvs.Show(rest_edges, view, "GeometryRepresentation")
    outline_display.Representation = "Surface"
    outline_display.ColorArrayName = [None, ""]
    outline_display.DiffuseColor = [0.96, 0.97, 0.98]
    outline_display.AmbientColor = [0.96, 0.97, 0.98]
    outline_display.LineWidth = 2.2
    outline_display.RenderLinesAsTubes = 1
    show_label(view, case)

    pvs.Render(view)
    configure_camera(view, contract)
    pvs.Render(view)

    stem = f"30-cuboid-{case['case_id']}-paraview"
    png = case_dir / f"{stem}.png"
    state = case_dir / f"{stem}.pvsm"
    png_tmp = png.with_name(f".{png.stem}.tmp{png.suffix}")
    state_tmp = state.with_name(f".{state.stem}.tmp{state.suffix}")
    pvs.SaveScreenshot(
        str(png_tmp),
        layout,
        ImageResolution=resolution,
        TransparentBackground=0,
        FontScaling="Scale fonts proportionally",
    )
    pvs.SaveState(str(state_tmp))
    if png_size(png_tmp) != tuple(resolution) or png_tmp.stat().st_size < 100_000:
        raise ValueError(f"invalid ParaView screenshot for {case['case_id']}")
    state_head = state_tmp.read_text(encoding="utf-8", errors="strict")[:2048]
    if "ServerManagerState" not in state_head:
        raise ValueError(f"invalid ParaView state for {case['case_id']}")
    png_tmp.replace(png)
    state_tmp.replace(state)
    return {
        "case_id": case["case_id"],
        "png": {
            "path": str(png.resolve()),
            "size_bytes": png.stat().st_size,
            "sha256": sha256(png),
            "width": resolution[0],
            "height": resolution[1],
        },
        "pvsm": {
            "path": str(state.resolve()),
            "size_bytes": state.stat().st_size,
            "sha256": sha256(state),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--renderer-receipt", required=True, type=Path)
    args = parser.parse_args()

    from paraview import servermanager

    manager = servermanager.vtkSMProxyManager
    version = ".".join(
        str(value)
        for value in (
            manager.GetVersionMajor(),
            manager.GetVersionMinor(),
            manager.GetVersionPatch(),
        )
    )
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, found {version}"
        )
    contract = read_contract(args.contract.resolve())
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not output_root.is_dir() or any(output_root.iterdir()):
        raise FileExistsError(f"output root must exist and be empty: {output_root}")

    outputs = [
        render_case(case, contract, input_root, output_root)
        for case in contract["cases"]
    ]
    receipt = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "design": EXPECTED_DESIGN,
        "complete": True,
        "status": "ok",
        "paraview_version": version,
        "native_paraview_rendering": True,
        "image_resolution": contract["image_resolution"],
        "shared_u_y_range": contract["shared_u_y_range"],
        "camera": contract["camera"],
        "outputs": outputs,
    }
    args.renderer_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
