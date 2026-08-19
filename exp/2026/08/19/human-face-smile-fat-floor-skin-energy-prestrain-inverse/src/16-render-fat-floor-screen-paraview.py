from __future__ import annotations

# This source is executed only by ParaView's pvbatch, not by the project Python.
# ruff: noqa: C901, EM101, EM102, FBT003, TRY003
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_DESIGN = "fat-floor-fixed-activation-paraview-screen-v1"
BRANCH_ORDER = ("P0", "P1")
STATE_ORDER = ("target", "old-e0", "new-efat-zero", "new-efat-old-seed")
VIEW_ORDER = ("front", "30-degree", "mouth", "eye-cheek+x")
MODE_ORDER = ("geometry", "normal-residual")
IMAGE_RESOLUTION = (4000, 3000)

# Direct pvbatch execution is blocked as well as wrapper execution.  After the
# frozen source has been reviewed, approval may change only this boolean.
PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False


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
        raise TypeError("ParaView contract must be a JSON object")
    if value.get("schema_version") != 1 or value.get("design") != EXPECTED_DESIGN:
        raise ValueError("ParaView contract schema or design changed")
    if value.get("complete") is not True:
        raise ValueError("ParaView contract is incomplete")
    if value.get("branch_order") != list(BRANCH_ORDER):
        raise ValueError("ParaView branch order changed")
    if value.get("state_order") != list(STATE_ORDER):
        raise ValueError("ParaView state order changed")
    if value.get("view_order") != list(VIEW_ORDER):
        raise ValueError("ParaView view order changed")
    if value.get("mode_order") != list(MODE_ORDER):
        raise ValueError("ParaView mode order changed")
    if value.get("image_resolution") != list(IMAGE_RESOLUTION):
        raise ValueError("ParaView image resolution changed")
    if value.get("renderer") != (
        "ParaView 6.1.1 native geometry and scalar rendering only; no PyVista rendering"
    ):
        raise ValueError("ParaView renderer authority changed")
    return value


def _split_even(
    layout: Any, location: int, count: int, *, horizontal: bool
) -> list[int]:
    if count == 1:
        return [location]
    fraction = 1.0 / count
    if horizontal:
        layout.SplitHorizontal(location, fraction)
    else:
        layout.SplitVertical(location, fraction)
    first = int(layout.SMProxy.GetFirstChild(location))
    second = int(layout.SMProxy.GetSecondChild(location))
    if first < 0 or second < 0:
        raise RuntimeError("ParaView layout split failed")
    return [first, *_split_even(layout, second, count - 1, horizontal=horizontal)]


def _grid_locations(layout: Any) -> list[list[int]]:
    rows = _split_even(layout, 0, len(VIEW_ORDER), horizontal=False)
    return [_split_even(layout, row, len(STATE_ORDER), horizontal=True) for row in rows]


def _validate_input(path: Path, item: dict[str, Any], input_root: Path) -> None:
    resolved = path.resolve()
    if input_root.resolve() not in resolved.parents:
        raise ValueError(f"ParaView input escapes the pinned input root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = {"size_bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}
    expected = {
        "size_bytes": int(item["size_bytes"]),
        "sha256": str(item["sha256"]),
    }
    if actual != expected:
        raise ValueError(f"ParaView input identity changed: {resolved}")


def _new_view(camera: dict[str, Any]) -> Any:
    view = pvs.CreateView("RenderView")
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    focus = [float(value) for value in camera["focus"]]
    direction = [float(value) for value in camera["direction"]]
    view.CameraFocalPoint = focus
    view.CameraPosition = [focus[index] + 0.30 * direction[index] for index in range(3)]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CenterOfRotation = focus
    view.CameraParallelScale = float(camera["parallel_scale"])
    return view


def _show_geometry(reader: Any, view: Any, state: str) -> Any:
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    display.ColorArrayName = [None, ""]
    color = [0.76, 0.78, 0.80] if state == "target" else [0.847, 0.706, 0.612]
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.20
    display.Diffuse = 0.75
    display.Specular = 0.20
    display.SpecularPower = 22.0
    return display


def _show_residual(
    reader: Any, view: Any, limit_mm: float, *, show_scalar_bar: bool
) -> Any:
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    pvs.ColorBy(display, ("POINTS", "TargetNormalResidualMM"))
    lut = pvs.GetColorTransferFunction("TargetNormalResidualMM")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(-limit_mm, limit_mm)
    display.SetScalarBarVisibility(view, show_scalar_bar)
    if show_scalar_bar:
        scalar_bar = pvs.GetScalarBar(lut, view)
        scalar_bar.Title = "target-normal residual"
        scalar_bar.ComponentTitle = "mm"
        scalar_bar.TitleColor = [0.0, 0.0, 0.0]
        scalar_bar.LabelColor = [0.0, 0.0, 0.0]
        scalar_bar.WindowLocation = "Upper Right Corner"
    return display


def _show_label(view: Any, text: str) -> None:
    source = pvs.Text()
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 9
    display.Color = [0.0, 0.0, 0.0]
    display.Bold = 1


def _render_plate(
    *,
    contract: dict[str, Any],
    branch: str,
    mode: str,
    input_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    layout = pvs.CreateLayout(name=f"fat-floor-{branch}-{mode}")
    locations = _grid_locations(layout)
    readers: dict[str, Any] = {}
    from paraview import servermanager

    for state in STATE_ORDER:
        item = contract["inputs"][branch][state]
        path = Path(str(item["path"]))
        _validate_input(path, item, input_root)
        reader = pvs.XMLPolyDataReader(
            registrationName=f"{branch} {state} IsFace",
            FileName=[str(path.resolve())],
        )
        reader.PointArrayStatus = [
            "TargetNormalResidualMM",
            "DisplacementMM",
            "TargetDisplacementMM",
            "ResidualDisplacementMM",
        ]
        reader.UpdatePipeline()
        fetched = servermanager.Fetch(reader)
        if (
            fetched.GetNumberOfPoints() != 15_299
            or fetched.GetNumberOfCells() != 29_899
        ):
            raise ValueError(f"{branch}/{state} ParaView dimensions changed")
        point_data = fetched.GetPointData()
        names = {
            str(point_data.GetArrayName(index))
            for index in range(point_data.GetNumberOfArrays())
        }
        required = {
            "TargetNormalResidualMM",
            "DisplacementMM",
            "TargetDisplacementMM",
            "ResidualDisplacementMM",
        }
        if not required <= names:
            raise KeyError(f"{branch}/{state} ParaView point arrays changed")
        readers[state] = reader

    limit_mm = float(contract["normal_residual_shared_limit_mm"])
    if not 0.25 <= limit_mm < 100.0:
        raise ValueError("invalid shared target-normal residual limit")
    for row, view_name in enumerate(VIEW_ORDER):
        camera = contract["views"][view_name]
        for column, state in enumerate(STATE_ORDER):
            view = _new_view(camera)
            if not layout.AssignView(locations[row][column], view):
                raise RuntimeError("ParaView failed to assign a view to the plate")
            if mode == "geometry":
                _show_geometry(readers[state], view, state)
            else:
                _show_residual(
                    readers[state],
                    view,
                    limit_mm,
                    show_scalar_bar=(row == 0 and column == len(STATE_ORDER) - 1),
                )
            item = contract["inputs"][branch][state]
            _show_label(
                view,
                f"{view_name} | {item['display_label']}\n"
                f"{item['material_label']}\n{item['metric_label']}",
            )
            pvs.Render(view)

    layout.SetSize(*IMAGE_RESOLUTION)
    stem = f"17-paraview-{branch.lower()}-{mode}"
    png = output_dir / f"{stem}.png"
    state_file = output_dir / f"{stem}.pvsm"
    png_temporary = png.with_name(f".{png.stem}.tmp{png.suffix}")
    state_temporary = state_file.with_name(f".{state_file.stem}.tmp{state_file.suffix}")
    if any(path.exists() for path in (png, state_file, png_temporary, state_temporary)):
        raise FileExistsError(f"refusing stale ParaView plate outputs for {stem}")
    pvs.RenderAllViews()
    pvs.SaveScreenshot(
        str(png_temporary.resolve()),
        layout,
        ImageResolution=list(IMAGE_RESOLUTION),
        TransparentBackground=0,
        FontScaling="Scale fonts proportionally",
    )
    pvs.SaveState(str(state_temporary.resolve()))
    if not png_temporary.is_file() or not state_temporary.is_file():
        raise RuntimeError(f"ParaView did not write {branch} {mode} outputs")
    png_temporary.replace(png)
    state_temporary.replace(state_file)
    return png, state_file


def main() -> None:
    if not PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(
            "NO-GO: direct ParaView rendering awaits static review and isolated source approval"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    contract = _read_contract(args.contract.resolve())
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise FileExistsError(
            f"ParaView output directory must exist and be empty: {output_dir}"
        )
    for branch in BRANCH_ORDER:
        for mode in MODE_ORDER:
            _render_plate(
                contract=contract,
                branch=branch,
                mode=mode,
                input_root=input_root,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
