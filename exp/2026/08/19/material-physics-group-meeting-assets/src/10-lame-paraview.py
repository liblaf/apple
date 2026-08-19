from __future__ import annotations

# This file is executed by ParaView's pvbatch, not by the project interpreter.
# ruff: noqa: EM101, EM102, FBT003, TRY003
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_DESIGN = "meeting-lame-conversion-only-native-paraview-v1"
EXPECTED_CASES = ("old-3d", "corrected-plane-stress")
EXPECTED_ASSETS = (
    ("10-lame-old-3d-geometry", "old-3d", "geometry"),
    (
        "10-lame-corrected-plane-stress-geometry",
        "corrected-plane-stress",
        "geometry",
    ),
    ("10-lame-old-3d-area-strain", "old-3d", "area-strain"),
    (
        "10-lame-corrected-plane-stress-area-strain",
        "corrected-plane-stress",
        "area-strain",
    ),
)
EXPECTED_IMAGE_RESOLUTION = (1800, 1800)
EXPECTED_STRAIN_LIMIT_PERCENT = 7.322
EXPECTED_POINTS = 15_299
EXPECTED_TRIANGLES = 29_899


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contract(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("Lamé ParaView contract must be a JSON object")
    if value.get("schema_version") != 1 or value.get("design") != EXPECTED_DESIGN:
        raise ValueError("Lamé ParaView contract schema or design changed")
    if value.get("complete") is not True:
        raise ValueError("Lamé ParaView contract is incomplete")
    if value.get("case_order") != list(EXPECTED_CASES):
        raise ValueError("Lamé ParaView case order changed")
    if value.get("image_resolution") != list(EXPECTED_IMAGE_RESOLUTION):
        raise ValueError("Lamé ParaView resolution changed")
    if float(value.get("strain_limit_percent", -1.0)) != (
        EXPECTED_STRAIN_LIMIT_PERCENT
    ):
        raise ValueError("Lamé ParaView strain scale changed")
    expected_assets = [
        {"asset_id": asset_id, "case_id": case_id, "mode": mode}
        for asset_id, case_id, mode in EXPECTED_ASSETS
    ]
    actual_assets = [
        {
            "asset_id": item.get("asset_id"),
            "case_id": item.get("case_id"),
            "mode": item.get("mode"),
        }
        for item in value.get("assets", [])
    ]
    if actual_assets != expected_assets:
        raise ValueError("Lamé ParaView asset inventory changed")
    if value.get("layout") != "one-view-per-file-no-contact-sheet":
        raise ValueError("Lamé ParaView layout contract changed")
    return value


def _validate_version() -> str:
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
    return version


def _validate_file(path: Path, item: dict[str, Any], root: Path) -> None:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"ParaView input escapes input root: {resolved}")
    actual = {"size_bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}
    expected = {
        "size_bytes": int(item["size_bytes"]),
        "sha256": str(item["sha256"]),
    }
    if actual != expected:
        raise ValueError(f"ParaView input identity changed: {resolved}")


def _new_view(camera: dict[str, Any]) -> Any:
    view = pvs.CreateView("RenderView")
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    focus = [float(value) for value in camera["focus"]]
    direction = [float(value) for value in camera["direction"]]
    view.CameraFocalPoint = focus
    view.CameraPosition = [
        focus[index] + 0.30 * direction[index] for index in range(3)
    ]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CenterOfRotation = focus
    view.CameraParallelScale = float(camera["parallel_scale"])
    view.ViewSize = list(EXPECTED_IMAGE_RESOLUTION)
    return view


def _show_geometry(reader: Any, view: Any) -> None:
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    display.ColorArrayName = [None, ""]
    color = [0.847, 0.706, 0.612]
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.20
    display.Diffuse = 0.75
    display.Specular = 0.20
    display.SpecularPower = 22.0


def _show_area_strain(reader: Any, view: Any, limit: float) -> None:
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    pvs.ColorBy(display, ("CELLS", "AreaStrainPercent"))
    lut = pvs.GetColorTransferFunction("AreaStrainPercent")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(-limit, limit)
    display.SetScalarBarVisibility(view, True)
    scalar_bar = pvs.GetScalarBar(lut, view)
    scalar_bar.Title = "area strain"
    scalar_bar.ComponentTitle = "%"
    scalar_bar.TitleColor = [0.0, 0.0, 0.0]
    scalar_bar.LabelColor = [0.0, 0.0, 0.0]
    scalar_bar.TitleFontSize = 24
    scalar_bar.LabelFontSize = 20
    scalar_bar.WindowLocation = "Upper Right Corner"


def _render_asset(
    *,
    contract: dict[str, Any],
    asset: dict[str, Any],
    input_root: Path,
    output_root: Path,
) -> None:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    case_id = str(asset["case_id"])
    item = contract["inputs"][case_id]
    input_path = Path(str(item["path"]))
    _validate_file(input_path, item, input_root)
    reader = pvs.XMLPolyDataReader(
        registrationName=f"{case_id} IsFace", FileName=[str(input_path.resolve())]
    )
    reader.PointArrayStatus = [
        "GlobalPointId",
        "DisplacementMM",
        "TargetDisplacementMM",
    ]
    reader.CellArrayStatus = ["AreaRatio", "AreaStrainPercent"]
    reader.UpdatePipeline()

    from paraview import servermanager

    fetched = servermanager.Fetch(reader)
    if (
        fetched.GetNumberOfPoints() != EXPECTED_POINTS
        or fetched.GetNumberOfCells() != EXPECTED_TRIANGLES
    ):
        raise ValueError(f"{case_id} IsFace dimensions changed")
    cell_data = fetched.GetCellData()
    cell_names = {
        str(cell_data.GetArrayName(index))
        for index in range(cell_data.GetNumberOfArrays())
    }
    if not {"AreaRatio", "AreaStrainPercent"} <= cell_names:
        raise KeyError(f"{case_id} area arrays changed")

    view = _new_view(contract["camera"])
    mode = str(asset["mode"])
    if mode == "geometry":
        _show_geometry(reader, view)
    elif mode == "area-strain":
        _show_area_strain(reader, view, float(contract["strain_limit_percent"]))
    else:
        raise ValueError(f"unknown Lamé render mode: {mode}")
    pvs.Render(view)

    png = Path(str(asset["png_path"])).resolve()
    state = Path(str(asset["state_path"])).resolve()
    if png.parent != output_root.resolve() or state.parent != output_root.resolve():
        raise ValueError("Lamé output escapes output root")
    temporary_png = png.with_name(f".{png.stem}.tmp{png.suffix}")
    temporary_state = state.with_name(f".{state.stem}.tmp{state.suffix}")
    if any(
        path.exists() for path in (png, state, temporary_png, temporary_state)
    ):
        raise FileExistsError(f"refusing stale Lamé output for {asset['asset_id']}")
    pvs.SaveScreenshot(
        str(temporary_png),
        view,
        ImageResolution=list(EXPECTED_IMAGE_RESOLUTION),
        TransparentBackground=0,
        FontScaling="Scale fonts proportionally",
    )
    pvs.SaveState(str(temporary_state))
    if not temporary_png.is_file() or not temporary_state.is_file():
        raise RuntimeError(f"ParaView did not write {asset['asset_id']}")
    temporary_png.replace(png)
    temporary_state.replace(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _validate_version()
    contract = _read_contract(args.contract.resolve())
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not output_root.is_dir() or any(output_root.iterdir()):
        raise FileExistsError(
            f"Lamé output root must exist and be empty: {output_root}"
        )
    for asset in contract["assets"]:
        _render_asset(
            contract=contract,
            asset=asset,
            input_root=input_root,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
