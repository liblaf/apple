from __future__ import annotations

# This script is executed only by ParaView's pvbatch, not by the project Python.
# ruff: noqa: EM101, EM102, FBT003, TRY003, TRY004
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
CASE_ORDER = ("H0P0", "H0P1", "H1P1", "H1P0")
COHORT_ORDER = ("terminal", "baseline-fidelity", "common-tau")
VIEW_ORDER = ("front", "30-degree", "mouth", "eye-cheek+x")
MODES = ("geometry", "normal-residual")
IMAGE_RESOLUTION = (4000, 3000)

# Direct pvbatch execution is blocked too; the wrapper is not the only guard. A
# later isolated approval edit may change only this boolean after analysis review.
PARAVIEW_RENDERER_EXECUTION_APPROVED_AFTER_ANALYSIS_REVIEW = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_analysis(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict) or not bool(value.get("complete")):
        raise ValueError("analysis JSON is missing or incomplete")
    if value.get("case_order") != list(CASE_ORDER):
        raise ValueError("analysis case order changed")
    paraview = value.get("paraview")
    if not isinstance(paraview, dict):
        raise ValueError("analysis has no ParaView contract")
    if paraview.get("cohort_order") != list(COHORT_ORDER):
        raise ValueError("analysis cohort order changed")
    if paraview.get("view_order") != list(VIEW_ORDER):
        raise ValueError("analysis view order changed")
    if paraview.get("renderer") != (
        "ParaView 6.1.1 only; Matplotlib geometry render is prohibited"
    ):
        raise ValueError("analysis renderer contract changed")
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
    return [_split_even(layout, row, len(CASE_ORDER), horizontal=True) for row in rows]


def _validate_input(path: Path, identity: dict[str, Any], input_root: Path) -> None:
    resolved = path.resolve()
    if input_root.resolve() not in resolved.parents:
        raise ValueError(f"ParaView input escapes the analyzer output root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = {"size_bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}
    expected = {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
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
    view.CameraPosition = [focus[i] + 0.30 * direction[i] for i in range(3)]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CenterOfRotation = focus
    view.CameraParallelScale = float(camera["parallel_scale"])
    return view


def _show_geometry(reader: Any, view: Any) -> Any:
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = [0.847, 0.706, 0.612]
    display.AmbientColor = [0.847, 0.706, 0.612]
    display.Ambient = 0.25
    display.Diffuse = 0.75
    display.Specular = 0.15
    display.SpecularPower = 20.0
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
    display.FontSize = 10
    display.Color = [0.0, 0.0, 0.0]
    display.Bold = 1


def _render_one(
    *,
    analysis: dict[str, Any],
    cohort: str,
    mode: str,
    input_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    contract = analysis["paraview"]
    layout = pvs.CreateLayout(name=f"{cohort}-{mode}")
    locations = _grid_locations(layout)
    readers: dict[str, Any] = {}
    from paraview import servermanager

    for case_id in CASE_ORDER:
        item = contract["inputs"][cohort][case_id]
        path = Path(str(item["path"]))
        _validate_input(path, item, input_root)
        reader = pvs.XMLPolyDataReader(
            registrationName=f"{cohort} {case_id} selected skin",
            FileName=[str(path.resolve())],
        )
        reader.PointArrayStatus = ["TargetNormalResidualMM", "DisplacementMM"]
        reader.UpdatePipeline()
        fetched = servermanager.Fetch(reader)
        if (
            fetched.GetNumberOfPoints() != 15_299
            or fetched.GetNumberOfCells() != 29_899
        ):
            raise ValueError(f"{cohort}/{case_id} ParaView readback dimensions changed")
        point_data = fetched.GetPointData()
        names = {
            str(point_data.GetArrayName(index))
            for index in range(point_data.GetNumberOfArrays())
        }
        if not {"TargetNormalResidualMM", "DisplacementMM"} <= names:
            raise KeyError(f"{cohort}/{case_id} ParaView readback arrays changed")
        readers[case_id] = reader

    limit_mm = float(contract["normal_residual_shared_limit_mm"])
    if not 0.0 < limit_mm < 1.0e6:
        raise ValueError("invalid shared residual color limit")
    for row, view_name in enumerate(VIEW_ORDER):
        camera = contract["views"][view_name]
        for column, case_id in enumerate(CASE_ORDER):
            view = _new_view(camera)
            if not layout.AssignView(locations[row][column], view):
                raise RuntimeError("ParaView failed to assign a view to the grid")
            item = contract["inputs"][cohort][case_id]
            if mode == "geometry":
                _show_geometry(readers[case_id], view)
                metric_line = (
                    f"err={float(item['target_error_rms_mm']):.3f} mm | "
                    f"D={float(item['dihedral_rms_deg']):.2f} deg\n"
                    f"L={float(item['normal_laplacian_rms_mm']):.3f} mm | "
                    f"area={float(item['area_ratio_rms_error']):.3f}"
                )
            else:
                _show_residual(
                    readers[case_id],
                    view,
                    limit_mm,
                    show_scalar_bar=(row == 0 and column == len(CASE_ORDER) - 1),
                )
                metric_line = f"target-normal residual +/-{limit_mm:.2f} mm"
            status = str(item["selection_status"])
            _show_label(
                view,
                f"{view_name} | {case_id} | step {int(item['step'])} | {status}\n"
                f"{metric_line}",
            )
            pvs.Render(view)

    layout.SetSize(*IMAGE_RESOLUTION)
    png = output_dir / f"35-paraview-{cohort}-{mode}.png"
    state = output_dir / f"35-paraview-{cohort}-{mode}.pvsm"
    png_temporary = png.with_name(f".{png.stem}.tmp{png.suffix}")
    state_temporary = state.with_name(f".{state.stem}.tmp{state.suffix}")
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
        raise RuntimeError(f"ParaView did not write {cohort} {mode} outputs")
    png_temporary.replace(png)
    state_temporary.replace(state)
    return png, state


def main() -> None:
    if not PARAVIEW_RENDERER_EXECUTION_APPROVED_AFTER_ANALYSIS_REVIEW:
        raise RuntimeError(
            "NO-GO: direct ParaView rendering awaits completed analysis review and isolated source approval"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
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
    analysis = _read_analysis(args.analysis.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing nonempty ParaView output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for cohort in COHORT_ORDER:
        for mode in MODES:
            _render_one(
                analysis=analysis,
                cohort=cohort,
                mode=mode,
                input_root=args.input_root.resolve(),
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
