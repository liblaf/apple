from __future__ import annotations

# Run only with ParaView's pvbatch.
# ruff: noqa: C901, EM102, FBT003, PLR0915, SLF001, TRY003
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_VERSION = "6.1.1"
EXPECTED_DESIGN = "whole-anatomy-dominant-material-three-midplane-cross-sections"
BACKGROUND = (0.97, 0.97, 0.97)
TEXT = (0.05, 0.05, 0.05)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _fetch(proxy: Any) -> Any:
    from paraview import servermanager

    return servermanager.Fetch(proxy)


def _add_text(
    view: Any, text: str, location: str, font_size: int, *, bold: bool
) -> None:
    source = pvs.Text()
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = location
    display.FontSize = font_size
    display.Color = list(TEXT)
    display.Bold = int(bold)


def _render_view(contract: dict[str, Any], plane_id: str) -> dict[str, Any]:
    plane = contract["cross_sections"][plane_id]
    outputs = contract["outputs"]["views"][plane_id]
    input_spec = outputs["render_input"]
    input_path = Path(str(input_spec["path"])).resolve()
    if _identity(input_path) != input_spec["identity"]:
        raise ValueError(f"{plane_id} render-input identity changed")
    output_png = Path(str(outputs["png"])).resolve()
    output_pvsm = Path(str(outputs["pvsm"])).resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    reader = pvs.XMLPolyDataReader(
        registrationName=f"Prepared whole-anatomy {plane_id} cross-section",
        FileName=[str(input_path)],
    )
    reader.CellArrayStatus = ["DominantMaterial"]
    reader.UpdatePipeline()
    dataset = _fetch(reader)
    if dataset.GetNumberOfPoints() != int(plane["points"]):
        raise ValueError(f"{plane_id} point count changed")
    if dataset.GetNumberOfCells() != int(plane["cells"]):
        raise ValueError(f"{plane_id} cell count changed")
    category_array = dataset.GetCellData().GetArray("DominantMaterial")
    if category_array is None:
        raise KeyError(f"{plane_id} cross-section lost DominantMaterial")
    counts = {"Fat": 0, "Muscle": 0, "Aponeurosis": 0}
    for index in range(category_array.GetNumberOfTuples()):
        category = round(category_array.GetTuple1(index))
        if category not in (0, 1, 2):
            raise ValueError(f"{plane_id} category escapes [0,2]: {category}")
        counts[("Fat", "Muscle", "Aponeurosis")[category]] += 1
    if counts != plane["dominant_category_cell_counts"]:
        raise ValueError(f"{plane_id} category counts changed: {counts}")

    view = pvs.CreateView("RenderView")
    view.UseColorPaletteForBackground = 0
    view.Background = list(BACKGROUND)
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    focus = [float(value) for value in plane["camera_focus_m"]]
    normal = [float(value) for value in plane["normal"]]
    view.CameraFocalPoint = focus
    view.CameraPosition = [focus[i] + 0.30 * normal[i] for i in range(3)]
    view.CameraViewUp = [float(value) for value in plane["view_up"]]
    view.CenterOfRotation = focus
    view.CameraParallelScale = float(plane["camera_parallel_scale_m"])

    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface"
    pvs.ColorBy(display, ("CELLS", "DominantMaterial"))
    lut = pvs.GetColorTransferFunction("DominantMaterial", display, separate=True)
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = ["0", "Fat", "1", "Muscle", "2", "Aponeurosis"]
    lut.ActiveAnnotatedValues = ["0", "1", "2"]
    colors: list[float] = []
    for key in ("0", "1", "2"):
        colors.extend(
            float(value)
            for value in contract["categorical_view"]["materials"][key]["rgb"]
        )
    lut.IndexedColors = colors
    lut.ShowCategoricalColorsinDataRangeOnly = 1
    lut.RescaleTransferFunction(0.0, 2.0)
    lut.ScalarRangeInitialized = 1.0
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, True)
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = ""
    bar.ComponentTitle = ""
    bar.TitleColor = list(TEXT)
    bar.LabelColor = list(TEXT)
    bar.LabelFontSize = 18
    bar.WindowLocation = "Lower Right Corner"
    bar.Orientation = "Vertical"
    bar.ScalarBarLength = 0.23
    bar.ScalarBarThickness = 24

    outline = pvs.FeatureEdges(registrationName="Cross-section boundary", Input=reader)
    outline.BoundaryEdges = 1
    outline.FeatureEdges = 0
    outline.NonManifoldEdges = 0
    outline.ManifoldEdges = 0
    outline.UpdatePipeline()
    outline_display = pvs.Show(outline, view, "GeometryRepresentation")
    outline_display.Representation = "Surface"
    outline_display.ColorArrayName = [None, ""]
    outline_display.DiffuseColor = [0.12, 0.12, 0.12]
    outline_display.LineWidth = 1.5

    _add_text(
        view,
        f"WHOLE-ANATOMY VOLUME | {plane['name']}\n"
        "Dominant constituent (categorical view only)",
        "Upper Left Corner",
        24,
        bold=True,
    )
    _add_text(
        view,
        "FAT  |  Stable Neo-Hookean  |  E = 0.003 MPa, nu = 0.49\n"
        "MUSCLE  |  active Stable Neo-Hookean  |  E = 0.030 MPa, nu = 0.49\n"
        "APONEUROSIS  |  Stable Neo-Hookean  |  E = 0.10 MPa, nu = 0.35",
        "Upper Right Corner",
        18,
        bold=False,
    )
    _add_text(
        view,
        "Physics uses continuous fraction-weighted energies.\n"
        "FatFraction + MuscleFraction + AponeurosisFraction = 1 exactly.",
        "Lower Left Corner",
        17,
        bold=False,
    )

    resolution = [int(value) for value in contract["renderer"]["image_resolution"]]
    png_tmp = output_png.with_name(f".{output_png.stem}.tmp{output_png.suffix}")
    pvsm_tmp = output_pvsm.with_name(f".{output_pvsm.stem}.tmp{output_pvsm.suffix}")
    for path in (png_tmp, pvsm_tmp):
        if path.exists():
            path.unlink()
    pvs.Render(view)
    pvs.SaveScreenshot(
        str(png_tmp),
        view,
        ImageResolution=resolution,
        TransparentBackground=0,
        FontScaling="Do not scale fonts",
    )
    pvs.SaveState(str(pvsm_tmp))
    if not png_tmp.is_file() or not pvsm_tmp.is_file():
        raise RuntimeError(f"ParaView did not create {plane_id} PNG and PVSM")
    png_tmp.replace(output_png)
    pvsm_tmp.replace(output_pvsm)
    return {
        "cross_section": {**plane, "dominant_category_cell_counts": counts},
        "input": {"path": str(input_path), "identity": _identity(input_path)},
        "outputs": {
            "png": {"path": str(output_png), "identity": _identity(output_png)},
            "pvsm": {"path": str(output_pvsm), "identity": _identity(output_pvsm)},
        },
    }


def _render(contract_path: Path, receipt_path: Path) -> None:
    contract = _read_json(contract_path)
    if contract.get("design") != EXPECTED_DESIGN:
        raise ValueError(f"unexpected design: {contract.get('design')}")
    version = _version()
    if version != EXPECTED_VERSION:
        raise ValueError(f"ParaView version changed: {version}")
    views = {
        plane: _render_view(contract, plane)
        for plane in ("midsagittal", "coronal", "axial")
    }
    _write_json(
        receipt_path,
        {
            "schema_version": 2,
            "design": EXPECTED_DESIGN,
            "complete": True,
            "native_paraview_rendering": True,
            "paraview_version": version,
            "views": views,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    _render(args.contract.resolve(), args.receipt.resolve())


if __name__ == "__main__":
    main()
