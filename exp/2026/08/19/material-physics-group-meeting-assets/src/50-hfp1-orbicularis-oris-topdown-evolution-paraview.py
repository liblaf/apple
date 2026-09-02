"""Render the exact 41-state HFP1 Orbicularis oris history with ParaView."""

from __future__ import annotations

# Run only with ParaView's pvbatch.
# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, PLR0915, SLF001, TRY003
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import paraview.simple as pvs

VERSION = "6.1.1"
DESIGN = "hfp1-orbicularis-oris-initial-max-z-coplanar-section-evolution"
N_POINTS = 3_248
N_TETS = 10_484
N_FRAMES = 41
SCALARS = ("DetF", "DetAinv", "DetG")
REQUIRED_ARRAYS = {
    "SourceCellId",
    "RestVolume",
    "MuscleFraction",
    "DetF",
    "DetAinv",
    "DetG",
    "DoubleInverted",
}
REFERENCE_ARRAYS = {"SourceCellId", "RestVolume", "MuscleFraction"}
BLUE, WHITE, RED, MAGENTA = (
    (0.086, 0.286, 0.690),
    (1.0, 1.0, 1.0),
    (0.760, 0.02, 0.12),
    (0.98, 0.0, 0.72),
)


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

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
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

    dataset = servermanager.Fetch(proxy)
    if dataset is None:
        raise RuntimeError("ParaView produced no dataset")
    return dataset


def vtk_array(dataset: Any, name: str) -> np.ndarray:
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    array = dataset.GetCellData().GetArray(name)
    if array is None:
        raise KeyError(name)
    return np.asarray(vtk_to_numpy(array))


def validate_identity(spec: dict[str, Any]) -> Path:
    if not {"path", "identity"} <= set(spec):
        raise ValueError(f"identity schema changed: {sorted(spec)}")
    path = Path(str(spec["path"])).resolve()
    if not path.is_file() or identity(path) != spec["identity"]:
        raise ValueError(f"input identity changed: {path}")
    return path


def validate_contract(contract: dict[str, Any], receipt_path: Path) -> None:
    required = {
        "schema_version",
        "design",
        "complete",
        "case",
        "selection",
        "inputs",
        "frames",
        "geometry",
        "skin_section",
        "scalar_ranges",
        "camera",
        "render",
        "outputs",
    }
    if (
        set(contract) != required
        or contract["schema_version"] != 1
        or contract["design"] != DESIGN
        or contract["complete"] is not True
    ):
        raise ValueError("contract schema, design, or completion changed")
    selection = contract["selection"]
    if (
        not isinstance(selection, dict)
        or selection.get("muscle_id") != 254
        or selection.get("cells") != N_TETS
        or selection.get("points") != N_POINTS
        or selection.get("spatial_crop") is not False
    ):
        raise ValueError("contract is not the complete uncropped Orbicularis oris")
    render = contract["render"]
    expected_render = {
        "resolution",
        "fps",
        "frame_count",
        "no_interpolation_or_duplication",
        "no_deformation_exaggeration",
        "geometry_mode",
        "camera_scope",
        "skin_line_width_px",
    }
    if (
        set(render) != expected_render
        or render["resolution"] != [1200, 1800]
        or render["fps"] != 30
        or render["frame_count"] != N_FRAMES
        or render["no_interpolation_or_duplication"] is not True
        or render["no_deformation_exaggeration"] is not True
        or render["geometry_mode"]
        != "muscle-and-lip-skin-coplanar-initial-max-z-y-section"
        or render["camera_scope"] != "mouth-from-full-orbicularis-bounds"
        or render["skin_line_width_px"] != 2.0
    ):
        raise ValueError("render contract changed")
    outputs = contract["outputs"]
    if (
        set(outputs) != {"frames_dir", "pvsm", "renderer_receipt"}
        or Path(str(outputs["renderer_receipt"])).resolve() != receipt_path
    ):
        raise ValueError("output contract changed")


def validate_camera(camera: dict[str, Any]) -> None:
    expected = {
        "focus",
        "position",
        "view_up",
        "parallel_scale",
        "projection",
        "look_direction",
        "orientation",
    }
    if set(camera) != expected:
        raise ValueError("camera schema changed")
    focus, position, up, look = (
        np.asarray(camera[key], dtype=float)
        for key in ("focus", "position", "view_up", "look_direction")
    )
    if any(vector.shape != (3,) for vector in (focus, position, up, look)):
        raise ValueError("camera vectors must have three elements")
    if not (
        position[1] > focus[1]
        and np.allclose(position[[0, 2]], focus[[0, 2]], atol=1e-12)
    ):
        raise ValueError("camera is not on head-superior +Y axis")
    if not np.allclose(look, (0.0, -1.0, 0.0), atol=1e-12) or not np.allclose(
        up, (0.0, 0.0, 1.0), atol=1e-12
    ):
        raise ValueError("camera orientation is not +Y to -Y with +Z image-up")
    if camera["projection"] != "parallel" or float(camera["parallel_scale"]) <= 0:
        raise ValueError("camera must use parallel projection")


def configure(view: Any, camera: dict[str, Any]) -> None:
    validate_camera(camera)
    view.UseColorPaletteForBackground = 0
    view.Background = [0.965, 0.965, 0.965]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = camera["focus"]
    view.CameraPosition = camera["position"]
    view.CameraViewUp = camera["view_up"]
    view.CenterOfRotation = camera["focus"]
    view.CameraParallelScale = float(camera["parallel_scale"])


def plain(display: Any, color: tuple[float, float, float]) -> None:
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = color
    display.AmbientColor = color


def add_text(view: Any, text: str) -> Any:
    source = pvs.Text(registrationName="Evolution label")
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Left Corner"
    display.FontSize = 26
    display.Color = [0.04, 0.04, 0.04]
    display.Bold = 1
    display.Justification = "Left"
    display.VerticalJustification = "Top"
    return source


def split_rows(layout: Any, location: int, count: int) -> list[int]:
    if count == 1:
        return [location]
    layout.SplitVertical(location, 1.0 / count)
    top, bottom = (
        int(layout.SMProxy.GetFirstChild(location)),
        int(layout.SMProxy.GetSecondChild(location)),
    )
    if top < 0 or bottom < 0:
        raise RuntimeError("ParaView layout split failed")
    return [top, *split_rows(layout, bottom, count - 1)]


def fixed_lut(display: Any, name: str, limits: list[float]) -> Any:
    low, high = (float(value) for value in limits)
    if not (math.isfinite(low) and math.isfinite(high) and low < 0.0 < high):
        raise ValueError(f"{name} fixed range must bracket zero")
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
    bar.TitleColor = [0.04, 0.04, 0.04]
    bar.LabelColor = [0.04, 0.04, 0.04]
    bar.TitleFontSize, bar.LabelFontSize = 24, 20
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Right Corner"
    bar.ScalarBarLength, bar.ScalarBarThickness = 0.48, 18


def y_slice(source: Any, plane_y: float, name: str) -> Any:
    section = pvs.Slice(registrationName=name, Input=source)
    section.SliceType = "Plane"
    section.SliceType.Origin = [0.0, plane_y, 0.0]
    section.SliceType.Normal = [0.0, 1.0, 0.0]
    return section


def show_initial(initial_section: Any, view: Any) -> None:
    display = pvs.Show(initial_section, view, "GeometryRepresentation")
    display.Representation = "Wireframe"
    plain(display, (0.56, 0.56, 0.56))
    display.LineWidth, display.Opacity = 0.7, 0.22


def show_double_inversion(series: Any, view: Any) -> None:
    selected = pvs.Threshold(registrationName="Double-inverted OO tets", Input=series)
    selected.Scalars, selected.ThresholdMethod = ["CELLS", "DoubleInverted"], "Between"
    selected.LowerThreshold, selected.UpperThreshold, selected.AllScalars = 0.5, 1.0, 1
    display = pvs.Show(selected, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    plain(display, MAGENTA)
    display.EdgeColor, display.LineWidth = [0.0, 0.0, 0.0], 1.2


def validate_frame(
    dataset: Any, frame: dict[str, Any], ranges: dict[str, list[float]]
) -> dict[str, int]:
    if dataset.GetNumberOfPoints() != N_POINTS or dataset.GetNumberOfCells() != N_TETS:
        raise ValueError(f"step {frame['step']} compact topology changed")
    names = {
        str(dataset.GetCellData().GetArrayName(index))
        for index in range(dataset.GetCellData().GetNumberOfArrays())
    }
    if not names >= REQUIRED_ARRAYS:
        raise KeyError(f"step {frame['step']} lacks {sorted(REQUIRED_ARRAYS - names)}")
    rest = vtk_array(dataset, "RestVolume").astype(float)
    if rest.shape != (N_TETS,) or np.any(~np.isfinite(rest)) or np.any(rest <= 0):
        raise ValueError(f"step {frame['step']} rest volume invalid")
    counts: dict[str, int] = {}
    metric = frame["metrics"]
    for name in SCALARS:
        values = vtk_array(dataset, name).astype(float)
        if values.shape != (N_TETS,) or np.any(~np.isfinite(values)):
            raise ValueError(f"step {frame['step']} {name} invalid")
        if (
            values.min() < float(ranges[name][0]) - 1e-11
            or values.max() > float(ranges[name][1]) + 1e-11
        ):
            raise ValueError(f"step {frame['step']} lies outside fixed {name} range")
        count = int((values < 0.0).sum())
        counts[name] = count
        expected = metric[name]
        if (
            int(expected["negative_cells"]) != count
            or not math.isclose(
                float(expected["minimum"]),
                float(values.min()),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(expected["maximum"]),
                float(values.max()),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"step {frame['step']} {name} metrics changed")
    double = vtk_array(dataset, "DoubleInverted").astype(bool)
    if not np.array_equal(
        double,
        (vtk_array(dataset, "DetF") < 0.0) & (vtk_array(dataset, "DetAinv") < 0.0),
    ):
        raise ValueError(f"step {frame['step']} DoubleInverted changed")
    if int(metric["double_inverted_cells"]) != int(double.sum()):
        raise ValueError(f"step {frame['step']} double-inversion metric changed")
    return counts


def validate_muscle_section(dataset: Any, plane_y: float, step: int) -> dict[str, Any]:
    if dataset.GetNumberOfPoints() < 1 or dataset.GetNumberOfCells() < 1:
        raise ValueError(f"step {step} muscle section is empty")
    names = {
        str(dataset.GetCellData().GetArrayName(index))
        for index in range(dataset.GetCellData().GetNumberOfArrays())
    }
    if not names >= REQUIRED_ARRAYS:
        raise KeyError(
            f"step {step} muscle section lacks {sorted(REQUIRED_ARRAYS - names)}"
        )
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    points = np.asarray(vtk_to_numpy(dataset.GetPoints().GetData()))
    if not np.allclose(points[:, 1], plane_y, atol=1e-10, rtol=0.0):
        raise ValueError(f"step {step} muscle section is not at the fixed Y plane")
    source_ids = vtk_array(dataset, "SourceCellId").astype(np.int64)
    negative_tets = {
        name: int(
            np.unique(source_ids[vtk_array(dataset, name).astype(float) < 0.0]).size
        )
        for name in SCALARS
    }
    double = vtk_array(dataset, "DoubleInverted").astype(bool)
    if not np.array_equal(
        double,
        (vtk_array(dataset, "DetF") < 0.0) & (vtk_array(dataset, "DetAinv") < 0.0),
    ):
        raise ValueError(f"step {step} muscle-section DoubleInverted changed")
    return {
        "points": dataset.GetNumberOfPoints(),
        "cells": dataset.GetNumberOfCells(),
        "source_tets": int(np.unique(source_ids).size),
        "negative_tets": negative_tets,
        "double_inverted_tets": int(np.unique(source_ids[double]).size),
    }


def line_component_count(dataset: Any) -> int:
    parent = np.arange(dataset.GetNumberOfPoints(), dtype=np.int64)

    def find(point: int) -> int:
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = int(parent[point])
        return point

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    used: set[int] = set()
    for index in range(dataset.GetNumberOfCells()):
        cell = dataset.GetCell(index)
        points = [int(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
        if len(points) < 2:
            raise ValueError("skin section contains a malformed line")
        used.update(points)
        for left, right in itertools.pairwise(points):
            union(left, right)
    if used != set(range(dataset.GetNumberOfPoints())):
        raise ValueError("skin section line connectivity changed")
    return len({find(point) for point in used})


def camera_view_point_count(points: np.ndarray, camera: dict[str, Any]) -> int:
    focus = np.asarray(camera["focus"], dtype=float)
    scale = float(camera["parallel_scale"])
    panel_aspect = 1200 / (1800 / len(SCALARS))
    return int(
        (
            (points[:, 0] >= focus[0] - scale * panel_aspect)
            & (points[:, 0] <= focus[0] + scale * panel_aspect)
            & (points[:, 2] >= focus[2] - scale)
            & (points[:, 2] <= focus[2] + scale)
        ).sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    contract, receipt_path = read_json(args.contract.resolve()), args.receipt.resolve()
    validate_contract(contract, receipt_path)
    if paraview_version() != VERSION:
        raise ValueError(f"ParaView version changed: {paraview_version()}")
    inputs = contract["inputs"]
    if set(inputs) != {"reference", "series", "skin_section_series"}:
        raise ValueError("input schema changed")
    reference_path, series_path = (
        validate_identity(inputs["reference"]),
        validate_identity(inputs["series"]),
    )
    skin_series_path = validate_identity(inputs["skin_section_series"])
    if (
        set(inputs["reference"]) != {"path", "identity", "points", "cells"}
        or inputs["reference"].get("points") != N_POINTS
        or inputs["reference"].get("cells") != N_TETS
    ):
        raise ValueError("reference topology contract changed")
    if (
        set(inputs["series"]) != {"path", "identity", "frames", "steps"}
        or inputs["series"]["frames"] != N_FRAMES
        or inputs["series"]["steps"] != list(range(N_FRAMES))
    ):
        raise ValueError("series contract changed")
    if (
        set(inputs["skin_section_series"]) != {"path", "identity", "frames", "steps"}
        or inputs["skin_section_series"]["frames"] != N_FRAMES
        or inputs["skin_section_series"]["steps"] != list(range(N_FRAMES))
    ):
        raise ValueError("skin-section series contract changed")
    skin_section = contract["skin_section"]
    if set(skin_section) != {
        "plane_y",
        "anchor",
        "surface_selection",
        "frames",
    } or not math.isfinite(float(skin_section["plane_y"])):
        raise ValueError("skin-section metadata changed")
    anchor = skin_section["anchor"]
    surface_selection = skin_section["surface_selection"]
    geometry = contract["geometry"]
    if (
        set(anchor) != {"step", "definition", "global_point_id", "point_m"}
        or anchor["step"] != 0
        or anchor["global_point_id"] != 52_222
        or anchor["definition"]
        != "maximum Z point of the full selected Orbicularis at the initial saved frame; its Y coordinate fixes the section plane"
        or len(anchor["point_m"]) != 3
        or not all(math.isfinite(float(value)) for value in anchor["point_m"])
        or not math.isclose(
            float(anchor["point_m"][1]),
            float(skin_section["plane_y"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or set(surface_selection)
        != {"predicate", "points", "triangles", "nasal_geometry_included"}
        or surface_selection["predicate"]
        != "all three external-surface triangle vertices have IsLip == true"
        or surface_selection["points"] != 2_275
        or surface_selection["triangles"] != 4_296
        or surface_selection["nasal_geometry_included"] is not False
        or set(geometry)
        != {
            "reference_bounds_m",
            "all_muscle_state_union_bounds_m",
            "deformation_exaggeration",
        }
        or geometry["deformation_exaggeration"] != 1.0
    ):
        raise ValueError("initial maximum-Z mouth skin-section contract changed")
    skin_frames = skin_section["frames"]
    if not isinstance(skin_frames, list) or len(skin_frames) != N_FRAMES:
        raise ValueError("skin-section frame contract changed")
    frames = contract["frames"]
    if not isinstance(frames, list) or len(frames) != N_FRAMES:
        raise ValueError("frame contract changed")
    for step, frame in enumerate(frames):
        if (
            set(frame) != {"step", "path", "identity", "metrics"}
            or frame["step"] != step
        ):
            raise ValueError("frame sequence changed")
        frame_path = validate_identity(
            {"path": frame["path"], "identity": frame["identity"]}
        )
        if frame_path.suffix != ".vtu":
            raise ValueError(f"step {step} is not a VTU")
    for step, frame in enumerate(skin_frames):
        if (
            set(frame)
            != {
                "step",
                "path",
                "identity",
                "points",
                "cells",
                "components",
                "camera_view_points",
            }
            or frame["step"] != step
        ):
            raise ValueError("skin-section frame sequence changed")
        frame_path = validate_identity(
            {"path": frame["path"], "identity": frame["identity"]}
        )
        if (
            frame_path.suffix != ".vtp"
            or int(frame["points"]) <= 0
            or int(frame["cells"]) <= 0
            or int(frame["components"]) != 1
            or int(frame["camera_view_points"]) != int(frame["points"])
        ):
            raise ValueError(f"skin-section step {step} is invalid")
    series_manifest = read_json(series_path)
    if (
        set(series_manifest) != {"file-series-version", "files"}
        or series_manifest["file-series-version"] != "1.0"
    ):
        raise ValueError("VTU series manifest schema changed")
    manifest_frames = series_manifest["files"]
    if not isinstance(manifest_frames, list) or len(manifest_frames) != N_FRAMES:
        raise ValueError("VTU series manifest frame count changed")
    for step, (entry, frame) in enumerate(zip(manifest_frames, frames, strict=True)):
        if set(entry) != {"name", "time"} or float(entry["time"]) != float(step):
            raise ValueError(f"VTU series manifest step {step} changed")
        listed = (series_path.parent / str(entry["name"])).resolve()
        if listed != Path(str(frame["path"])).resolve():
            raise ValueError(f"VTU series manifest path {step} changed")
    skin_manifest = read_json(skin_series_path)
    if (
        set(skin_manifest) != {"file-series-version", "files"}
        or skin_manifest["file-series-version"] != "1.0"
        or not isinstance(skin_manifest["files"], list)
        or len(skin_manifest["files"]) != N_FRAMES
    ):
        raise ValueError("skin-section VTP series manifest changed")
    for step, (entry, frame) in enumerate(
        zip(skin_manifest["files"], skin_frames, strict=True)
    ):
        if set(entry) != {"name", "time"} or float(entry["time"]) != float(step):
            raise ValueError(f"skin-section VTP series manifest step {step} changed")
        listed = (skin_series_path.parent / str(entry["name"])).resolve()
        if listed != Path(str(frame["path"])).resolve():
            raise ValueError(f"skin-section VTP series manifest path {step} changed")
    ranges = contract["scalar_ranges"]
    if set(ranges) != set(SCALARS):
        raise ValueError("scalar range schema changed")
    for name in SCALARS:
        low, high = ranges[name]
        if not (
            math.isfinite(float(low))
            and math.isfinite(float(high))
            and float(low) < 0.0 < float(high)
        ):
            raise ValueError(f"invalid fixed {name} range")
    outputs = {
        name: Path(str(value)).resolve() for name, value in contract["outputs"].items()
    }
    frames_dir, pvsm = outputs["frames_dir"], outputs["pvsm"]
    if frames_dir.exists() or pvsm.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite renderer outputs")
    frames_dir.mkdir(parents=True)
    pvs._DisableFirstRenderCameraReset()
    reference = pvs.XMLUnstructuredGridReader(
        registrationName="Reference full Orbicularis oris",
        FileName=[str(reference_path)],
    )
    reference.UpdatePipeline()
    reference_dataset = fetch(reference)
    if (
        reference_dataset.GetNumberOfPoints() != N_POINTS
        or reference_dataset.GetNumberOfCells() != N_TETS
    ):
        raise ValueError("reference compact topology changed")
    reference_arrays = {
        str(reference_dataset.GetCellData().GetArrayName(index))
        for index in range(reference_dataset.GetCellData().GetNumberOfArrays())
    }
    if not reference_arrays >= REFERENCE_ARRAYS:
        raise KeyError(f"reference lacks {sorted(REFERENCE_ARRAYS - reference_arrays)}")
    series = pvs.OpenDataFile(str(series_path))
    if series is None:
        raise RuntimeError("ParaView cannot open the VTU series")
    skin_series = pvs.OpenDataFile(str(skin_series_path))
    if skin_series is None:
        raise RuntimeError("ParaView cannot open the skin-section VTP series")
    series.UpdatePipeline()
    values = [float(value) for value in series.TimestepValues]
    if values != [float(step) for step in range(N_FRAMES)]:
        raise ValueError(f"series TimestepValues changed: {values}")
    skin_series.UpdatePipeline()
    skin_values = [float(value) for value in skin_series.TimestepValues]
    if skin_values != values:
        raise ValueError(f"skin-section TimestepValues changed: {skin_values}")
    plane_y = float(skin_section["plane_y"])
    initial = pvs.XMLUnstructuredGridReader(
        registrationName="Initial saved full Orbicularis oris",
        FileName=[str(Path(str(frames[0]["path"])).resolve())],
    )
    initial.UpdatePipeline()
    validate_frame(fetch(initial), frames[0], ranges)
    initial_section = y_slice(initial, plane_y, "Initial OO max-Z-plane section")
    initial_section.UpdatePipeline()
    initial_cut = fetch(initial_section)
    if initial_cut.GetNumberOfPoints() < 1 or initial_cut.GetNumberOfCells() < 1:
        raise ValueError("initial muscle section is empty")
    muscle_section = y_slice(series, plane_y, "Deformed OO max-Z-plane section")
    layout = pvs.CreateLayout(name="Orbicularis max-Z-anchored coplanar section")
    locations = split_rows(layout, 0, len(SCALARS))
    labels = {
        "DetF": "det(F): physical deformation",
        "DetAinv": "det(Ainv): activation map",
        "DetG": "det(G) = det(F) det(Ainv)",
    }
    text_sources: dict[str, Any] = {}
    for location, name in zip(locations, SCALARS, strict=True):
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(location, view):
            raise RuntimeError("ParaView layout assignment failed")
        configure(view, contract["camera"])
        show_initial(initial_section, view)
        display = pvs.Show(muscle_section, view, "GeometryRepresentation")
        display.Representation, display.EdgeColor, display.LineWidth = (
            "Surface With Edges",
            [0.08, 0.08, 0.08],
            0.35,
        )
        pvs.ColorBy(display, ("CELLS", name))
        lut = fixed_lut(display, name, ranges[name])
        display.LookupTable = lut
        display.SetScalarBarVisibility(view, True)
        scalar_bar(view, lut, name)
        show_double_inversion(muscle_section, view)
        skin_display = pvs.Show(skin_series, view, "GeometryRepresentation")
        skin_display.Representation = "Wireframe"
        plain(skin_display, (0.0, 0.58, 0.58))
        skin_display.LineWidth = float(contract["render"]["skin_line_width_px"])
        text_sources[name] = add_text(view, labels[name])
    resolution = contract["render"]["resolution"]
    layout.SetSize(*resolution)
    rendered: list[dict[str, Any]] = []
    for step, frame in enumerate(frames):
        series.UpdatePipeline(float(step))
        counts = validate_frame(fetch(series), frame, ranges)
        muscle_section.UpdatePipeline(float(step))
        section_metrics = validate_muscle_section(fetch(muscle_section), plane_y, step)
        skin_series.UpdatePipeline(float(step))
        skin_dataset = fetch(skin_series)
        skin_frame = skin_frames[step]
        if skin_dataset.GetNumberOfPoints() != int(
            skin_frame["points"]
        ) or skin_dataset.GetNumberOfCells() != int(skin_frame["cells"]):
            raise ValueError(f"skin-section step {step} topology changed")
        if any(
            skin_dataset.GetCellType(index) not in {3, 4}
            for index in range(skin_dataset.GetNumberOfCells())
        ):
            raise ValueError(f"skin-section step {step} is not a polyline")
        from paraview.vtk.util.numpy_support import vtk_to_numpy

        skin_points = np.asarray(vtk_to_numpy(skin_dataset.GetPoints().GetData()))
        if not np.allclose(
            skin_points[:, 1], float(skin_section["plane_y"]), atol=1e-10, rtol=0.0
        ):
            raise ValueError(f"skin-section step {step} is not at the fixed Y plane")
        components = line_component_count(skin_dataset)
        camera_points = camera_view_point_count(skin_points, contract["camera"])
        if components != int(skin_frame["components"]) or camera_points != int(
            skin_frame["camera_view_points"]
        ):
            raise ValueError(f"skin-section step {step} visibility changed")
        for name, source in text_sources.items():
            source.Text = (
                f"{labels[name]} | STEP {step:02d} / 40\n"
                f"{name} < 0: {counts[name]:,} full | "
                f"{section_metrics['negative_tets'][name]:,} cut tets\n"
                "gray: dim initial cut | magenta: double-inverted cut\n"
                "teal: IsLip skin cut | same fixed max-Z-anchor plane"
            )
        pvs.GetAnimationScene().AnimationTime = float(step)
        pvs.RenderAllViews()
        target = frames_dir / f"frame-{step:03d}.png"
        temporary = frames_dir / f".{target.stem}.tmp.png"
        if temporary.exists() or target.exists():
            raise FileExistsError(target)
        pvs.SaveScreenshot(
            str(temporary),
            layout,
            ImageResolution=resolution,
            TransparentBackground=0,
            FontScaling="Do not scale fonts",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 50_000:
            raise RuntimeError(f"ParaView failed to render {target}")
        temporary.replace(target)
        rendered.append(
            {
                "step": step,
                "path": str(target),
                "identity": identity(target),
                "muscle_section": section_metrics,
                "skin_section": {
                    "points": skin_dataset.GetNumberOfPoints(),
                    "cells": skin_dataset.GetNumberOfCells(),
                    "components": components,
                    "camera_view_points": camera_points,
                },
            }
        )
    state_tmp = pvsm.with_name(f".{pvsm.stem}.tmp{pvsm.suffix}")
    pvs.SaveState(str(state_tmp))
    if not state_tmp.is_file() or state_tmp.stat().st_size <= 10_000:
        raise RuntimeError("ParaView did not write a substantive temporal PVSM")
    state_tmp.replace(pvsm)
    digest = hashlib.sha256()
    for item in rendered:
        digest.update(Path(item["path"]).name.encode())
        digest.update(item["identity"]["sha256"].encode())
    aggregate = digest.hexdigest()
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "design": DESIGN,
            "paraview_version": VERSION,
            "complete": True,
            "frames": rendered,
            "ordered_png_sha256": aggregate,
            "frame_count": N_FRAMES,
            "TimestepValues": values,
            "camera": contract["camera"],
            "scalar_ranges": ranges,
            "geometry_mode": "muscle-and-lip-skin-coplanar-initial-max-z-y-section",
            "section_plane_y": plane_y,
            "skin_line_width_px": float(contract["render"]["skin_line_width_px"]),
            "skin_section_anchor": anchor,
            "skin_surface_selection": surface_selection,
            "pvsm": {"path": str(pvsm), "identity": identity(pvsm)},
        },
    )


if __name__ == "__main__":
    main()
