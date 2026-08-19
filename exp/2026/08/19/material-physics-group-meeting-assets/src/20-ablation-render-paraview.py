from __future__ import annotations

# Run only with ParaView's pvbatch.
# ruff: noqa: EM101, EM102, FBT003, TRY003
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_VERSION = "6.1.1"
EXPECTED_DESIGN = "meeting-authoritative-homogeneous-vs-fat-floor-c020-2x2-step40"
CASE_ORDER = ("H0P0", "H0P1", "HFP0", "HFP1")
EXPECTED_POINTS = 15_299
EXPECTED_TRIANGLES = 29_899
BACKGROUND = (0.94, 0.94, 0.94)
TEXT_COLOR = (0.05, 0.05, 0.05)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _version() -> str:
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


def _validate_file(path: Path, identity: dict[str, Any]) -> None:
    expected = {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }
    actual = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    if actual != expected:
        raise ValueError(f"input identity changed: {path}")


def _new_view(camera: dict[str, Any]) -> Any:
    view = pvs.CreateView("RenderView")
    view.Background = list(BACKGROUND)
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    focus = [float(value) for value in camera["focus"]]
    direction = [float(value) for value in camera["direction"]]
    view.CameraFocalPoint = focus
    view.CameraPosition = [focus[index] + 0.30 * direction[index] for index in range(3)]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CenterOfRotation = focus
    view.CameraParallelScale = float(camera["parallel_scale"])
    return view


def _text(view: Any, title: str, subtitle: str) -> None:
    source = pvs.Text()
    source.Text = f"{title}\n{subtitle}"
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 18
    display.Color = list(TEXT_COLOR)
    display.Bold = 1


def _scalar_bar(lut: Any, view: Any, title: str) -> None:
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = list(TEXT_COLOR)
    bar.LabelColor = list(TEXT_COLOR)
    bar.TitleFontSize = 16
    bar.LabelFontSize = 14
    bar.WindowLocation = "Lower Right Corner"
    bar.Orientation = "Vertical"
    bar.ScalarBarLength = 0.30
    bar.ScalarBarThickness = 18


def _material_lut(field: str, limits: list[float], display: Any) -> Any:
    low, high = (float(value) for value in limits)
    lut = pvs.GetColorTransferFunction(field, display, separate=True)
    if field == "SkinYoungModulusMPa":
        lut.RGBPoints = [
            low,
            0.267004,
            0.004874,
            0.329415,
            high,
            0.993248,
            0.906157,
            0.143936,
        ]
    else:
        lut.RGBPoints = [
            low,
            0.267004,
            0.004874,
            0.329415,
            0.5 * (low + high),
            0.127568,
            0.566949,
            0.550556,
            high,
            0.993248,
            0.906157,
            0.143936,
        ]
    lut.ColorSpace = "Lab"
    lut.RescaleTransferFunction(low, high)
    lut.ScalarRangeInitialized = 1.0
    return lut


def _save(view: Any, output_dir: Path, stem: str, resolution: list[int]) -> None:
    png = output_dir / f"{stem}.png"
    state = output_dir / f"{stem}.pvsm"
    png_tmp = png.with_name(f".{png.stem}.tmp{png.suffix}")
    state_tmp = state.with_name(f".{state.stem}.tmp{state.suffix}")
    if any(path.exists() for path in (png, state, png_tmp, state_tmp)):
        raise FileExistsError(f"refusing to overwrite output for {stem}")
    pvs.Render(view)
    pvs.SaveScreenshot(
        str(png_tmp.resolve()),
        view,
        ImageResolution=resolution,
        TransparentBackground=0,
        # Keep annotation sizes stable at the requested meeting resolution. ParaView's
        # proportional mode scales from the small render-view canvas and makes the
        # labels dominate an 1800 x 1600 screenshot.
        FontScaling="Do not scale fonts",
    )
    pvs.SaveState(str(state_tmp.resolve()))
    if not png_tmp.is_file() or not state_tmp.is_file():
        raise RuntimeError(f"ParaView did not produce {stem}")
    png_tmp.replace(png)
    state_tmp.replace(state)


def _fetch_names(reader: Any) -> tuple[set[str], set[str], int, int]:
    from paraview import servermanager

    dataset = servermanager.Fetch(reader)
    point_data = dataset.GetPointData()
    cell_data = dataset.GetCellData()
    point_names = {
        str(point_data.GetArrayName(index))
        for index in range(point_data.GetNumberOfArrays())
    }
    cell_names = {
        str(cell_data.GetArrayName(index))
        for index in range(cell_data.GetNumberOfArrays())
    }
    return (
        point_names,
        cell_names,
        int(dataset.GetNumberOfPoints()),
        int(dataset.GetNumberOfCells()),
    )


def _render_material(
    contract: dict[str, Any], factor: dict[str, Any], output_dir: Path
) -> None:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    case = contract["cases"][factor["source_case"]]
    path = Path(str(case["material_path"]))
    _validate_file(path, case["material_identity"])
    reader = pvs.XMLPolyDataReader(
        registrationName=f"{factor['factor_id']} IsFace skin material",
        FileName=[str(path.resolve())],
    )
    reader.CellArrayStatus = [str(factor["field"])]
    reader.UpdatePipeline()
    _point, cell, n_points, n_cells = _fetch_names(reader)
    if n_points != EXPECTED_POINTS or n_cells != EXPECTED_TRIANGLES:
        raise ValueError(f"{factor['factor_id']} dimensions changed")
    if factor["field"] not in cell:
        raise KeyError(f"{factor['factor_id']} lacks {factor['field']}")
    view = _new_view(contract["camera"])
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    pvs.ColorBy(display, ("CELLS", str(factor["field"])))
    lut = _material_lut(str(factor["field"]), factor["range"], display)
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, True)
    _scalar_bar(lut, view, str(factor["scalar_title"]))
    _text(view, str(factor["title"]), str(factor["subtitle"]))
    _save(
        view,
        output_dir,
        f"20-ablation-material-{factor['factor_id']}",
        contract["image_resolution"],
    )


def _case_title(
    case_id: str, case: dict[str, Any], mode: str
) -> tuple[str, str]:
    if case_id == "H0P0":
        material = "E=.2 MPa | p000"
    elif case_id == "H0P1":
        material = "E=.2 MPa | c020"
    elif case_id == "HFP0":
        material = "E=.003 MPa where raw R>1 | p000"
    else:
        material = "E=.003 MPa where raw R>1 | c020"
    if mode == "point-error":
        return (
            f"{case_id} | {material}",
            f"step 40 | objective target RMS={float(case['target_rms_mm']):.3f} mm",
        )
    metrics = (
        f"step 40 | target RMS={float(case['target_rms_mm']):.3f} mm | "
        f"D={float(case['contraction_dihedral_rms_deg']):.2f} deg\n"
        f"L={float(case['residual_normal_laplacian_rms_mm']):.3f} mm | "
        f"folds={int(case['folded_skin_triangles'])} | inverted={int(case['inverted_tets'])}"
    )
    return f"{case_id} | {material}", metrics


def _apply_point_error_colors(
    display: Any, view: Any, contract: dict[str, Any]
) -> None:
    pvs.ColorBy(display, ("POINTS", "TargetPointErrorMM"))
    lut = pvs.GetColorTransferFunction("TargetPointErrorMM", display, separate=True)
    limit = float(contract["point_error_shared_limit_mm"])
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("invalid shared point-error limit")
    lut.RGBPoints = [
        0.0,
        0.267004,
        0.004874,
        0.329415,
        0.5 * limit,
        0.127568,
        0.566949,
        0.550556,
        limit,
        0.993248,
        0.906157,
        0.143936,
    ]
    lut.ColorSpace = "Lab"
    lut.RescaleTransferFunction(0.0, limit)
    lut.ScalarRangeInitialized = 1.0
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, True)
    _scalar_bar(lut, view, "point error (mm)")


def _render_case(
    contract: dict[str, Any], case_id: str, mode: str, output_dir: Path
) -> None:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    case = contract["cases"][case_id]
    path = Path(str(case["render_surface_path"]))
    _validate_file(path, case["render_surface_identity"])
    reader = pvs.XMLPolyDataReader(
        registrationName=f"{case_id} registered step-40 IsFace surface",
        FileName=[str(path.resolve())],
    )
    reader.PointArrayStatus = [
        "TargetNormalResidualMM",
        "TargetPointErrorMM",
        "DisplacementMM",
    ]
    reader.UpdatePipeline()
    point, _cell, n_points, n_cells = _fetch_names(reader)
    if n_points != EXPECTED_POINTS or n_cells != EXPECTED_TRIANGLES:
        raise ValueError(f"{case_id} surface dimensions changed")
    if not {
        "TargetNormalResidualMM",
        "TargetPointErrorMM",
        "DisplacementMM",
    } <= point:
        raise KeyError(f"{case_id} surface fields changed")
    view = _new_view(contract["camera"])
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    if mode == "geometry":
        display.ColorArrayName = [None, ""]
        display.DiffuseColor = [0.847, 0.706, 0.612]
        display.AmbientColor = [0.847, 0.706, 0.612]
        display.Ambient = 0.25
        display.Diffuse = 0.75
        display.Specular = 0.15
        display.SpecularPower = 20.0
    elif mode == "normal-residual":
        pvs.ColorBy(display, ("POINTS", "TargetNormalResidualMM"))
        lut = pvs.GetColorTransferFunction(
            "TargetNormalResidualMM", display, separate=True
        )
        limit = float(contract["normal_residual_shared_limit_mm"])
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("invalid shared normal-residual limit")
        lut.RGBPoints = [
            -limit,
            0.230,
            0.299,
            0.754,
            0.0,
            0.865,
            0.865,
            0.865,
            limit,
            0.706,
            0.016,
            0.150,
        ]
        lut.ColorSpace = "Lab"
        lut.RescaleTransferFunction(-limit, limit)
        lut.ScalarRangeInitialized = 1.0
        display.LookupTable = lut
        display.SetScalarBarVisibility(view, True)
        _scalar_bar(lut, view, "target-normal residual (mm)")
    elif mode == "point-error":
        _apply_point_error_colors(display, view, contract)
    else:
        raise ValueError(f"unsupported render mode: {mode}")
    title, subtitle = _case_title(case_id, case, mode)
    _text(view, title, subtitle)
    _save(
        view,
        output_dir,
        f"20-ablation-step40-{case_id.lower()}-{mode}",
        contract["image_resolution"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = _read_json(args.contract)
    if _version() != EXPECTED_VERSION:
        raise RuntimeError(f"expected ParaView {EXPECTED_VERSION}, got {_version()}")
    if contract.get("design") != EXPECTED_DESIGN or not contract.get("complete"):
        raise ValueError("meeting ablation contract changed")
    if contract.get("case_order") != list(CASE_ORDER):
        raise ValueError("meeting ablation case order changed")
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must exist and be empty")
    for factor in contract["material_factors"]:
        _render_material(contract, factor, args.output_dir)
    for case_id in CASE_ORDER:
        _render_case(contract, case_id, "geometry", args.output_dir)
    for case_id in CASE_ORDER:
        _render_case(contract, case_id, "normal-residual", args.output_dir)
    for case_id in CASE_ORDER:
        _render_case(contract, case_id, "point-error", args.output_dir)


if __name__ == "__main__":
    main()
