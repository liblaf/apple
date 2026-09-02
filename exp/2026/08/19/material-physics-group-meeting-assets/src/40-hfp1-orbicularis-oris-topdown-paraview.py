"""Render the complete step-40 HFP1 Orbicularis oris with ParaView 6.1.1.

Inputs are compact provenance-checked tetra meshes from the saved endpoint.
This renderer runs no forward or inverse physics.
"""

from __future__ import annotations

# Run only with ParaView's pvbatch.
# ruff: noqa: C901, EM101, EM102, FBT003, SLF001, TRY003
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import paraview.simple as pvs

VERSION = "6.1.1"
DESIGN = "hfp1-step40-full-orbicularis-oris-superior-view"
N_POINTS = 3_248
N_TETS = 10_484
SELECTION = {
    "muscle_id": 254,
    "muscle_name": "Orbicularis oris001_Head_muscles_0",
    "predicate": "ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5",
    "spatial_crop": False,
    "cells": N_TETS,
    "points": N_POINTS,
}
ARRAYS = {
    "SourceCellId",
    "RestVolume",
    "MuscleFraction",
    "DetF",
    "DetAinv",
    "DetG",
    "DoubleInverted",
}
SCALARS = ("DetF", "DetAinv", "DetG")
BACKGROUND = (0.965, 0.965, 0.965)
TEXT = (0.04, 0.04, 0.04)
BLUE = (0.086, 0.286, 0.690)
WHITE = (1.0, 1.0, 1.0)
RED = (0.760, 0.020, 0.120)
MUSCLE_RED = (0.82, 0.08, 0.10)
MAGENTA = (0.98, 0.00, 0.72)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    result = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(result, dict):
        raise TypeError(f"{path} is not a JSON object")
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def fetch(proxy: Any) -> Any:
    from paraview import servermanager

    result = servermanager.Fetch(proxy)
    if result is None:
        raise RuntimeError("ParaView produced no dataset")
    return result


def cell_array(dataset: Any, name: str) -> np.ndarray:
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    array = dataset.GetCellData().GetArray(name)
    if array is None:
        raise KeyError(name)
    return np.asarray(vtk_to_numpy(array))


def points(dataset: Any) -> np.ndarray:
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    result = np.asarray(vtk_to_numpy(dataset.GetPoints().GetData()), dtype=np.float64)
    if result.shape != (N_POINTS, 3) or np.any(~np.isfinite(result)):
        raise ValueError(f"invalid compact point array: {result.shape}")
    return result


def connectivity(dataset: Any) -> np.ndarray:
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    if any(dataset.GetCellType(index) != 10 for index in range(N_TETS)):
        raise ValueError("compact muscle contains a non-tetrahedral cell")
    packed = np.asarray(vtk_to_numpy(dataset.GetCells().GetData()))
    if packed.size != 5 * N_TETS:
        raise ValueError(f"unexpected compact connectivity size: {packed.size}")
    packed = packed.reshape(-1, 5)
    if not np.all(packed[:, 0] == 4):
        raise ValueError("malformed compact tetrahedral connectivity")
    return packed[:, 1:].astype(np.int64, copy=False)


def validate_contract(contract: dict[str, Any], receipt_path: Path) -> None:
    required = {
        "schema_version",
        "design",
        "complete",
        "case",
        "inputs",
        "selection",
        "geometry",
        "metrics",
        "scalar_ranges",
        "camera",
        "image_resolution",
        "renderer",
        "outputs",
    }
    if set(contract) != required:
        raise ValueError(f"contract keys changed: {sorted(set(contract) ^ required)}")
    if contract["schema_version"] != 1 or contract["design"] != DESIGN:
        raise ValueError("contract schema or design changed")
    if contract["complete"] is not True or contract["selection"] != SELECTION:
        raise ValueError("contract is incomplete or selection is not the full muscle")
    expected_case = {
        "id": "HFP1",
        "source_case": "20-hfp1",
        "step": 40,
        "evaluations": 41,
        "inverse_converged": False,
        "stop_reason": "step_limit_smooth_decrease",
        "physics_rerun": False,
    }
    if contract["case"] != expected_case:
        raise ValueError(f"case contract changed: {contract['case']}")
    renderer = contract["renderer"]
    if renderer.get("version") != VERSION or renderer.get("authority") != (
        "native ParaView only; no PyVista pixel rendering"
    ):
        raise ValueError("renderer contract changed")
    expected_outputs = {
        "context_png",
        "context_pvsm",
        "determinants_png",
        "determinants_pvsm",
        "renderer_receipt",
    }
    if set(contract["outputs"]) != expected_outputs:
        raise ValueError("output schema changed")
    if Path(contract["outputs"]["renderer_receipt"]).resolve() != receipt_path:
        raise ValueError("CLI receipt path differs from the contract")


def validate_identity(spec: dict[str, Any], *, dimensions: bool) -> Path:
    keys = {"path", "identity"}
    if dimensions:
        keys |= {"points", "cells"}
    if set(spec) != keys:
        raise ValueError(f"input schema changed: {spec.keys()}")
    path = Path(str(spec["path"])).resolve()
    if not path.is_file() or identity(path) != spec["identity"]:
        raise ValueError(f"input identity changed: {path}")
    return path


def input_paths(contract: dict[str, Any]) -> dict[str, Path]:
    inputs = contract["inputs"]
    expected = {
        "source_endpoint",
        "source_summary",
        "reference",
        "deformed",
        "context_surface",
    }
    if set(inputs) != expected:
        raise ValueError("input schema changed")
    paths = {
        "source_endpoint": validate_identity(
            inputs["source_endpoint"], dimensions=True
        ),
        "source_summary": validate_identity(inputs["source_summary"], dimensions=False),
        "reference": validate_identity(inputs["reference"], dimensions=True),
        "deformed": validate_identity(inputs["deformed"], dimensions=True),
        "context_surface": validate_identity(
            inputs["context_surface"], dimensions=True
        ),
    }
    for key in ("reference", "deformed"):
        if (
            int(inputs[key]["points"]) != N_POINTS
            or int(inputs[key]["cells"]) != N_TETS
        ):
            raise ValueError(f"{key} dimensions changed")
    return paths


def reader(path: Path, name: str) -> Any:
    if path.suffix == ".vtu":
        proxy = pvs.XMLUnstructuredGridReader(
            registrationName=name, FileName=[str(path)]
        )
    elif path.suffix == ".vtp":
        proxy = pvs.XMLPolyDataReader(registrationName=name, FileName=[str(path)])
    else:
        raise ValueError(f"unsupported ParaView input: {path}")
    proxy.UpdatePipeline()
    return proxy


def bounds(array: np.ndarray) -> list[float]:
    low = array.min(axis=0)
    high = array.max(axis=0)
    return [
        float(low[0]),
        float(high[0]),
        float(low[1]),
        float(high[1]),
        float(low[2]),
        float(high[2]),
    ]


def validate_metric(
    expected: dict[str, Any], values: np.ndarray, volume: np.ndarray
) -> None:
    negative = values < 0.0
    actual = {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "negative_cells": int(negative.sum()),
        "negative_rest_volume_fraction": float(volume[negative].sum() / volume.sum()),
    }
    for key in ("minimum", "maximum", "negative_rest_volume_fraction"):
        if not math.isclose(
            float(expected[key]), actual[key], rel_tol=1e-11, abs_tol=1e-12
        ):
            raise ValueError(f"metric {key} changed: {actual[key]} != {expected[key]}")
    if int(expected["negative_cells"]) != actual["negative_cells"]:
        raise ValueError(f"negative-cell count changed: {actual['negative_cells']}")


def validate_volume(
    proxy: Any, expected_bounds: list[float], metrics: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    dataset = fetch(proxy)
    if dataset.GetNumberOfPoints() != N_POINTS or dataset.GetNumberOfCells() != N_TETS:
        raise ValueError("compact muscle topology changed")
    names = {
        str(dataset.GetCellData().GetArrayName(index))
        for index in range(dataset.GetCellData().GetNumberOfArrays())
    }
    if not names >= ARRAYS:
        raise KeyError(f"compact muscle lacks {sorted(ARRAYS - names)}")
    if not np.allclose(
        bounds(points(dataset)), expected_bounds, rtol=1e-11, atol=1e-12
    ):
        raise ValueError("compact muscle bounds changed")
    cells = connectivity(dataset)
    source_ids = cell_array(dataset, "SourceCellId").astype(np.int64)
    if source_ids.shape != (N_TETS,) or np.unique(source_ids).size != N_TETS:
        raise ValueError("SourceCellId no longer identifies every selected tet")
    volume = cell_array(dataset, "RestVolume").astype(np.float64)
    if volume.shape != (N_TETS,) or np.any(~np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("invalid rest-volume field")
    det_f = cell_array(dataset, "DetF").astype(np.float64)
    det_ainv = cell_array(dataset, "DetAinv").astype(np.float64)
    det_g = cell_array(dataset, "DetG").astype(np.float64)
    if not np.allclose(det_g, det_f * det_ainv, rtol=1e-12, atol=1e-12):
        raise ValueError("DetG no longer equals DetF * DetAinv")
    double = cell_array(dataset, "DoubleInverted").astype(bool)
    if not np.array_equal(double, (det_f < 0.0) & (det_ainv < 0.0)):
        raise ValueError("DoubleInverted field changed")
    if int(double.sum()) != int(metrics["double_inverted_cells"]):
        raise ValueError("double-inverted count changed")
    total = float(volume.sum())
    if not math.isclose(
        total, float(metrics["rest_volume_m3"]), rel_tol=1e-11, abs_tol=1e-15
    ):
        raise ValueError("selected rest volume changed")
    double_fraction = float(volume[double].sum() / total)
    if not math.isclose(
        double_fraction,
        float(metrics["double_inverted_rest_volume_fraction"]),
        rel_tol=1e-11,
        abs_tol=1e-12,
    ):
        raise ValueError("double-inverted rest-volume fraction changed")
    for name, values in zip(SCALARS, (det_f, det_ainv, det_g), strict=True):
        validate_metric(metrics[name], values, volume)
    return cells, source_ids


def validate_camera(camera: dict[str, Any]) -> None:
    keys = {
        "focus",
        "position",
        "view_up",
        "parallel_scale",
        "projection",
        "look_direction",
        "orientation",
    }
    if set(camera) != keys:
        raise ValueError("camera schema changed")
    focus = np.asarray(camera["focus"], dtype=float)
    position = np.asarray(camera["position"], dtype=float)
    up = np.asarray(camera["view_up"], dtype=float)
    look = np.asarray(camera["look_direction"], dtype=float)
    if any(value.shape != (3,) for value in (focus, position, up, look)):
        raise ValueError("camera vectors must have three components")
    offset = position - focus
    if not (offset[1] > 0 and np.allclose(offset[[0, 2]], 0.0, atol=1e-12)):
        raise ValueError("camera is not strictly above the head on +Y")
    if not np.allclose(look, (0.0, -1.0, 0.0), atol=1e-12):
        raise ValueError("camera does not look from +Y toward -Y")
    if not np.allclose(up, (0.0, 0.0, 1.0), atol=1e-12):
        raise ValueError("camera does not keep +Z toward image-up")
    if camera["projection"] != "parallel" or float(camera["parallel_scale"]) <= 0:
        raise ValueError("camera must use a positive parallel projection")


def configure_view(view: Any, camera: dict[str, Any]) -> None:
    validate_camera(camera)
    view.UseColorPaletteForBackground = 0
    view.Background = list(BACKGROUND)
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [float(value) for value in camera["focus"]]
    view.CameraPosition = [float(value) for value in camera["position"]]
    view.CameraViewUp = [float(value) for value in camera["view_up"]]
    view.CenterOfRotation = [float(value) for value in camera["focus"]]
    view.CameraParallelScale = float(camera["parallel_scale"])


def add_text(view: Any, text: str, *, size: int) -> None:
    source = pvs.Text(registrationName=text.splitlines()[0][:64])
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = size
    display.Color = list(TEXT)
    display.Bold = 1


def plain(display: Any, color: tuple[float, float, float]) -> None:
    pvs.ColorBy(display, None)
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = list(color)
    display.AmbientColor = list(color)


def show_reference(source: Any, view: Any) -> None:
    surface = pvs.ExtractSurface(
        registrationName="Reference full OO surface", Input=source
    )
    display = pvs.Show(surface, view, "GeometryRepresentation")
    display.Representation = "Wireframe"
    plain(display, (0.38, 0.38, 0.38))
    display.LineWidth = 1.1
    display.Opacity = 0.80


def show_double_inversion(source: Any, view: Any) -> None:
    selected = pvs.Threshold(registrationName="Double-inverted OO tets", Input=source)
    selected.Scalars = ["CELLS", "DoubleInverted"]
    selected.ThresholdMethod = "Between"
    selected.LowerThreshold = 0.5
    selected.UpperThreshold = 1.0
    selected.AllScalars = 1
    selected.UpdatePipeline()
    display = pvs.Show(selected, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    plain(display, MAGENTA)
    display.EdgeColor = [0.0, 0.0, 0.0]
    display.LineWidth = 1.3


def scalar_lut(display: Any, name: str, limits: list[float]) -> Any:
    low, high = (float(value) for value in limits)
    if not (math.isfinite(low) and math.isfinite(high) and low < 0 < high):
        raise ValueError(f"{name} range must strictly bracket zero")
    lut = pvs.GetColorTransferFunction(name, display, separate=True)
    lut.RGBPoints = [low, *BLUE, 0.0, *WHITE, high, *RED]
    lut.ColorSpace = "Lab"
    lut.RescaleTransferFunction(low, high)
    lut.ScalarRangeInitialized = 1.0
    return lut


def scalar_bar(view: Any, lut: Any, title: str) -> None:
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = list(TEXT)
    bar.LabelColor = list(TEXT)
    bar.TitleFontSize = 18
    bar.LabelFontSize = 15
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Right Corner"
    bar.ScalarBarLength = 0.48
    bar.ScalarBarThickness = 18


def output_paths(contract: dict[str, Any]) -> dict[str, Path]:
    result = {
        key: Path(str(value)).resolve() for key, value in contract["outputs"].items()
    }
    parents = {path.parent for path in result.values()}
    if len(parents) != 1:
        raise ValueError("all renderer outputs must share one directory")
    next(iter(parents)).mkdir(parents=True, exist_ok=True)
    return result


def save(target: Any, png: Path, pvsm: Path, resolution: list[int]) -> dict[str, Any]:
    if len(resolution) != 2 or any(int(value) <= 0 for value in resolution):
        raise ValueError(f"invalid image resolution: {resolution}")
    if png.exists() or pvsm.exists():
        raise FileExistsError(f"refusing to overwrite {png} or {pvsm}")
    png_tmp = png.with_name(f".{png.stem}.tmp{png.suffix}")
    pvsm_tmp = pvsm.with_name(f".{pvsm.stem}.tmp{pvsm.suffix}")
    if png_tmp.exists() or pvsm_tmp.exists():
        raise FileExistsError("stale temporary ParaView output exists")
    pvs.SaveScreenshot(
        str(png_tmp),
        target,
        ImageResolution=[int(value) for value in resolution],
        TransparentBackground=0,
        FontScaling="Do not scale fonts",
    )
    pvs.SaveState(str(pvsm_tmp))
    if not png_tmp.is_file() or png_tmp.stat().st_size <= 50_000:
        raise RuntimeError("ParaView did not create a substantive PNG")
    if not pvsm_tmp.is_file() or pvsm_tmp.stat().st_size <= 10_000:
        raise RuntimeError("ParaView did not create a substantive PVSM")
    png_tmp.replace(png)
    pvsm_tmp.replace(pvsm)
    return {
        "png": {"path": str(png), "identity": identity(png)},
        "pvsm": {"path": str(pvsm), "identity": identity(pvsm)},
    }


def render_context(
    contract: dict[str, Any], context_path: Path, muscle_path: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    context = reader(context_path, "HFP1 deformed face surface")
    muscle = reader(muscle_path, "Full deformed Orbicularis oris")
    view = pvs.CreateView("RenderView")
    configure_view(view, contract["camera"]["context"])
    skin = pvs.Show(context, view, "GeometryRepresentation")
    skin.Representation = "Surface"
    plain(skin, (0.74, 0.71, 0.68))
    skin.Opacity = 0.13
    skin.Specular = 0.08
    surface = pvs.ExtractSurface(
        registrationName="Full deformed OO surface", Input=muscle
    )
    display = pvs.Show(surface, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    plain(display, MUSCLE_RED)
    display.EdgeColor = [0.12, 0.01, 0.01]
    display.LineWidth = 0.45
    metrics = contract["metrics"]
    add_text(
        view,
        "HFP1 STEP 40 | FULL ORBICULARIS ORIS\n"
        "HEAD-SUPERIOR VIEW: +Y CAMERA -> -Y | +Z ANTERIOR/UP\n"
        f"{N_TETS:,} tets | {metrics['double_inverted_cells']} double-inverted | no crop | 1x deformation",
        size=22,
    )
    pvs.Render(view)
    outputs = output_paths(contract)
    return save(
        view,
        outputs["context_png"],
        outputs["context_pvsm"],
        contract["image_resolution"]["context"],
    )


def split_columns(layout: Any, location: int, count: int) -> list[int]:
    if count == 1:
        return [location]
    layout.SplitHorizontal(location, 1.0 / count)
    first = int(layout.SMProxy.GetFirstChild(location))
    second = int(layout.SMProxy.GetSecondChild(location))
    if first < 0 or second < 0:
        raise RuntimeError("ParaView layout split failed")
    return [first, *split_columns(layout, second, count - 1)]


def render_determinants(
    contract: dict[str, Any], ref_path: Path, def_path: Path
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    reference = reader(ref_path, "Reference full Orbicularis oris")
    deformed = reader(def_path, "Deformed full Orbicularis oris")
    layout = pvs.CreateLayout(name="Full Orbicularis determinant diagnostics")
    locations = split_columns(layout, 0, len(SCALARS))
    labels = {
        "DetF": "det(F): physical deformation",
        "DetAinv": "det(Ainv): activation map",
        "DetG": "det(G) = det(F) det(Ainv)",
    }
    metrics = contract["metrics"]
    for location, name in zip(locations, SCALARS, strict=True):
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(location, view):
            raise RuntimeError(f"failed to assign {name} view")
        configure_view(view, contract["camera"]["muscle"])
        show_reference(reference, view)
        surface = pvs.ExtractSurface(
            registrationName=f"Deformed full OO {name}", Input=deformed
        )
        display = pvs.Show(surface, view, "GeometryRepresentation")
        display.Representation = "Surface With Edges"
        display.EdgeColor = [0.08, 0.08, 0.08]
        display.LineWidth = 0.35
        display.Ambient = 0.22
        display.Diffuse = 0.78
        pvs.ColorBy(display, ("CELLS", name))
        lut = scalar_lut(display, name, contract["scalar_ranges"][name])
        display.LookupTable = lut
        display.SetScalarBarVisibility(view, True)
        scalar_bar(view, lut, name)
        show_double_inversion(deformed, view)
        add_text(
            view,
            f"{labels[name]}\n{name} < 0: {int(metrics[name]['negative_cells'])} / {N_TETS:,} tets\n"
            "gray wire: reference | magenta: double-inverted",
            size=18,
        )
        pvs.Render(view)
    resolution = contract["image_resolution"]["determinants"]
    layout.SetSize(int(resolution[0]), int(resolution[1]))
    pvs.RenderAllViews()
    outputs = output_paths(contract)
    return save(
        layout, outputs["determinants_png"], outputs["determinants_pvsm"], resolution
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    receipt_path = args.receipt.resolve()
    contract = read_json(contract_path)
    validate_contract(contract, receipt_path)
    version = paraview_version()
    if version != VERSION:
        raise ValueError(f"ParaView version changed: {version}")
    paths = input_paths(contract)
    reference = reader(paths["reference"], "Validate reference Orbicularis")
    deformed = reader(paths["deformed"], "Validate deformed Orbicularis")
    context = fetch(reader(paths["context_surface"], "Validate HFP1 face context"))
    context_spec = contract["inputs"]["context_surface"]
    if context.GetNumberOfPoints() != int(
        context_spec["points"]
    ) or context.GetNumberOfCells() != int(context_spec["cells"]):
        raise ValueError("context surface topology changed")
    ref_cells, ref_ids = validate_volume(
        reference, contract["geometry"]["reference_bounds_m"], contract["metrics"]
    )
    def_cells, def_ids = validate_volume(
        deformed, contract["geometry"]["deformed_bounds_m"], contract["metrics"]
    )
    if not np.array_equal(ref_cells, def_cells) or not np.array_equal(ref_ids, def_ids):
        raise ValueError("reference/deformed compact topology differs")
    for name in SCALARS:
        metric = contract["metrics"][name]
        if not np.allclose(
            contract["scalar_ranges"][name],
            [metric["minimum"], metric["maximum"]],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"{name} range is not the full-selection min/max")
    outputs = output_paths(contract)
    stale = [path for path in outputs.values() if path.exists()]
    if stale:
        raise FileExistsError(f"refusing stale renderer outputs: {stale}")
    context_outputs = render_context(
        contract, paths["context_surface"], paths["deformed"]
    )
    determinant_outputs = render_determinants(
        contract, paths["reference"], paths["deformed"]
    )
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "design": DESIGN,
            "complete": True,
            "native_paraview_rendering": True,
            "paraview_version": version,
            "contract": {
                "path": str(contract_path),
                "identity": identity(contract_path),
            },
            "selection": contract["selection"],
            "camera": contract["camera"],
            "metrics": contract["metrics"],
            "scalar_ranges": contract["scalar_ranges"],
            "outputs": {
                "context": context_outputs,
                "determinants": determinant_outputs,
            },
        },
    )


if __name__ == "__main__":
    main()
