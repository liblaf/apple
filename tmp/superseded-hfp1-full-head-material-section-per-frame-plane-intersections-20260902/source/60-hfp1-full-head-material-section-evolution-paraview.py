"""Render the HFP1 full-head filled material section with ParaView."""

from __future__ import annotations

# Run only with ParaView's pvbatch.
# ruff: noqa: C901, EM101, EM102, FBT003, PLR0915, SLF001, TRY003
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import paraview.simple as pvs

VERSION = "6.1.1"
DESIGN = "hfp1-full-head-dominant-material-fixed-y-section-evolution"
N_FRAMES = 41
REQUIRED_MATERIAL_ARRAYS = {
    "SourceCellId",
    "DominantMaterial",
    "FatFraction",
    "MuscleFraction",
    "AponeurosisFraction",
}
BACKGROUND = [0.965, 0.965, 0.965]
TEXT = [0.045, 0.045, 0.045]
EXPECTED_CAMERA_FOCUS = [
    1.4067350332694464,
    2.1730086794286745,
    0.08153530579422003,
]
EXPECTED_CAMERA_SCALE = 0.055


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def ordered_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
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


def vtk_points(dataset: Any) -> np.ndarray:
    from paraview.vtk.util.numpy_support import vtk_to_numpy

    if dataset.GetPoints() is None:
        raise ValueError("dataset has no points")
    return np.asarray(vtk_to_numpy(dataset.GetPoints().GetData()), dtype=float)


def validate_identity(spec: dict[str, Any]) -> Path:
    if set(spec) != {"path", "identity"}:
        raise ValueError(f"identity schema changed: {sorted(spec)}")
    path = Path(str(spec["path"])).resolve()
    if not path.is_file() or identity(path) != spec["identity"]:
        raise ValueError(f"input identity changed: {path}")
    return path


def validate_series(path: Path, frames: list[dict[str, Any]], key: str) -> None:
    manifest = read_json(path)
    if (
        set(manifest) != {"file-series-version", "files"}
        or manifest["file-series-version"] != "1.0"
        or not isinstance(manifest["files"], list)
        or len(manifest["files"]) != N_FRAMES
    ):
        raise ValueError(f"{key} series manifest changed")
    for step, (entry, frame) in enumerate(zip(manifest["files"], frames, strict=True)):
        if set(entry) != {"name", "time"} or float(entry["time"]) != float(step):
            raise ValueError(f"{key} series step {step} changed")
        listed = (path.parent / str(entry["name"])).resolve()
        expected = Path(str(frame[key]["path"])).resolve()
        if listed != expected or identity(listed) != frame[key]["identity"]:
            raise ValueError(f"{key} series frame {step} changed")


def validate_contract(contract: dict[str, Any], receipt_path: Path) -> None:
    required = {
        "schema_version",
        "design",
        "complete",
        "case",
        "plane",
        "topology",
        "materials",
        "material_view",
        "skin",
        "section_union_bounds_m",
        "camera",
        "render",
        "inputs",
        "frames",
        "outputs",
    }
    if (
        set(contract) != required
        or contract["schema_version"] != 1
        or contract["design"] != DESIGN
        or contract["complete"] is not True
    ):
        raise ValueError("contract schema, design, or completion changed")
    if contract["case"].get("steps") != list(range(N_FRAMES)):
        raise ValueError("source steps changed")
    plane = contract["plane"]
    if (
        plane.get("normal") != [0.0, 1.0, 0.0]
        or plane.get("fixed_for_all_frames") is not True
        or float(plane["origin"][1]) != 2.1730086794286745
        or plane.get("anchor", {}).get("global_point_id") != 52_222
    ):
        raise ValueError("fixed initial-max-Z plane changed")
    topology = contract["topology"]
    if (
        topology.get("full_head_points") != 228_660
        or topology.get("full_head_tetrahedra") != 1_146_517
        or topology.get("spatial_crop_before_section") is not False
    ):
        raise ValueError("full-head topology contract changed")
    materials = contract["materials"]
    if set(materials) != {"0", "1", "2"} or [
        materials[str(index)].get("name") for index in range(3)
    ] != ["Fat", "Muscle", "Aponeurosis"]:
        raise ValueError("material categories changed")
    material_view = contract["material_view"]
    if (
        material_view.get("field") != "DominantMaterial"
        or material_view.get("visualization_only") is not True
    ):
        raise ValueError("dominant-material view contract changed")
    render = contract["render"]
    if (
        render.get("resolution") != [1_200, 1_200]
        or render.get("fps") != 30
        or render.get("frame_count") != N_FRAMES
        or render.get("one_panel") is not True
        or render.get("determinant_metrics_rendered") is not False
        or render.get("material_representation") != "Surface With Edges"
        or render.get("cell_edges_rendered") is not True
        or render.get("internal_tet_section_edges_rendered") is not True
        or render.get("filled_material_polygons") is not True
        or render.get("full_head_section_before_camera") is not True
        or render.get("camera_crop_only") is not True
        or render.get("complete_section_in_view") is not False
        or render.get("cell_edge_rgb") != [0.12, 0.13, 0.15]
        or float(render.get("cell_edge_width_px", 0.0)) != 0.35
    ):
        raise ValueError("filled edge-and-camera render contract changed")
    validate_camera(contract["camera"], contract["section_union_bounds_m"])
    outputs = contract["outputs"]
    if Path(str(outputs["renderer_receipt"])).resolve() != receipt_path:
        raise ValueError("renderer receipt path changed")
    if not isinstance(contract["frames"], list) or len(contract["frames"]) != N_FRAMES:
        raise ValueError("frame contract changed")


def validate_camera(camera: dict[str, Any], section_bounds: list[float]) -> None:
    focus = np.asarray(camera["focus"], dtype=float)
    position = np.asarray(camera["position"], dtype=float)
    up = np.asarray(camera["view_up"], dtype=float)
    look = np.asarray(camera["look_direction"], dtype=float)
    focus_bounds = np.asarray(camera["focus_source_bounds_m"], dtype=float)
    if any(value.shape != (3,) for value in (focus, position, up, look)):
        raise ValueError("camera vectors must have three components")
    if focus_bounds.shape != (6,) or np.asarray(section_bounds).shape != (6,):
        raise ValueError("camera and section bounds must have six components")
    scale = float(camera["parallel_scale"])
    if not (
        position[1] > focus[1]
        and np.allclose(position[[0, 2]], focus[[0, 2]], atol=1e-12)
        and np.allclose(look, (0.0, -1.0, 0.0), atol=1e-12)
        and np.allclose(up, (0.0, 0.0, 1.0), atol=1e-12)
        and camera["projection"] == "parallel"
        and np.allclose(focus, EXPECTED_CAMERA_FOCUS, atol=1e-12, rtol=0.0)
        and scale == EXPECTED_CAMERA_SCALE
        and np.isclose(focus[0], 0.5 * (focus_bounds[0] + focus_bounds[1]))
        and np.isclose(focus[2], 0.5 * (focus_bounds[4] + focus_bounds[5]))
        and focus[0] - scale <= focus_bounds[0] <= focus_bounds[1] <= focus[0] + scale
        and focus[2] - scale <= focus_bounds[4] <= focus_bounds[5] <= focus[2] + scale
    ):
        raise ValueError("camera is not the fixed expanded Orbicularis crop")
    bounds = np.asarray(section_bounds, dtype=float)
    if not (
        bounds[0] < focus[0] - scale
        and bounds[1] > focus[0] + scale
        and bounds[4] < focus[2] - scale
    ):
        raise ValueError("full-head section no longer extends beyond the camera crop")


def configure_view(view: Any, camera: dict[str, Any]) -> None:
    view.UseColorPaletteForBackground = 0
    view.Background = BACKGROUND
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = camera["focus"]
    view.CameraPosition = camera["position"]
    view.CameraViewUp = camera["view_up"]
    view.CenterOfRotation = camera["focus"]
    view.CameraParallelScale = float(camera["parallel_scale"])


def plain(display: Any, color: list[float]) -> None:
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = color
    display.AmbientColor = color


def add_text(
    view: Any,
    text: str,
    location: str,
    font_size: int,
    *,
    bold: bool,
) -> Any:
    source = pvs.Text()
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = location
    display.FontSize = font_size
    display.Color = TEXT
    display.Bold = int(bold)
    display.Justification = "Left"
    display.VerticalJustification = "Top"
    return source


def configure_material_lut(display: Any, view: Any, materials: dict[str, Any]) -> Any:
    pvs.ColorBy(display, ("CELLS", "DominantMaterial"))
    lut = pvs.GetColorTransferFunction("DominantMaterial", display, separate=True)
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = ["0", "Fat", "1", "Muscle", "2", "Aponeurosis"]
    lut.ActiveAnnotatedValues = ["0", "1", "2"]
    colors: list[float] = []
    for index in range(3):
        colors.extend(float(value) for value in materials[str(index)]["rgb"])
    lut.IndexedColors = colors
    lut.ShowCategoricalColorsinDataRangeOnly = 1
    lut.RescaleTransferFunction(0.0, 2.0)
    lut.ScalarRangeInitialized = 1.0
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, True)
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = "Argmax volume material"
    bar.ComponentTitle = ""
    bar.TitleColor = TEXT
    bar.LabelColor = TEXT
    bar.TitleFontSize = 18
    bar.LabelFontSize = 17
    bar.WindowLocation = "Any Location"
    bar.Position = [0.74, 0.055]
    bar.Orientation = "Vertical"
    bar.ScalarBarLength = 0.23
    bar.ScalarBarThickness = 28
    return lut


def validate_material_frame(
    dataset: Any, spec: dict[str, Any], plane_y: float
) -> dict[str, Any]:
    if dataset.GetNumberOfPoints() != int(
        spec["points"]
    ) or dataset.GetNumberOfCells() != int(spec["cells"]):
        raise ValueError("material-section topology changed")
    names = {
        str(dataset.GetCellData().GetArrayName(index))
        for index in range(dataset.GetCellData().GetNumberOfArrays())
    }
    if not names >= REQUIRED_MATERIAL_ARRAYS:
        raise KeyError(
            f"material section lacks {sorted(REQUIRED_MATERIAL_ARRAYS - names)}"
        )
    dominant = np.asarray(vtk_array(dataset, "DominantMaterial"), dtype=np.int32)
    fractions = np.column_stack(
        [
            np.asarray(vtk_array(dataset, name), dtype=float)
            for name in (
                "FatFraction",
                "MuscleFraction",
                "AponeurosisFraction",
            )
        ]
    )
    fraction_sum_error = float(np.max(np.abs(fractions.sum(axis=1) - 1.0)))
    if (
        fractions.shape != (dataset.GetNumberOfCells(), 3)
        or np.any(~np.isfinite(fractions))
        or np.any((fractions < 0.0) | (fractions > 1.0))
        or fraction_sum_error > 1e-12
        or not np.array_equal(dominant, np.argmax(fractions, axis=1))
    ):
        raise ValueError("material categories are not argmax of continuous fractions")
    counts = {
        name: int(np.count_nonzero(dominant == index))
        for index, name in enumerate(("Fat", "Muscle", "Aponeurosis"))
    }
    if counts != spec["dominant_counts"]:
        raise ValueError(f"material-section categories changed: {counts}")
    source_ids = np.asarray(vtk_array(dataset, "SourceCellId"), dtype=np.int64)
    if np.unique(source_ids).size != dataset.GetNumberOfCells():
        raise ValueError("material section lost one-polygon-per-source-tet identity")
    points = vtk_points(dataset)
    if not np.allclose(points[:, 1], plane_y, atol=1e-10, rtol=0.0):
        raise ValueError("material section left the fixed plane")
    return {
        "points": len(points),
        "cells": len(dominant),
        "dominant_counts": counts,
        "argmax_of_continuous_fractions": True,
        "fraction_sum_max_abs_error": fraction_sum_error,
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
    for cell_index in range(dataset.GetNumberOfCells()):
        cell = dataset.GetCell(cell_index)
        if cell.GetCellType() not in {3, 4} or cell.GetNumberOfPoints() < 2:
            raise ValueError("skin section contains a non-line cell")
        ids = [int(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
        used.update(ids)
        for left, right in itertools.pairwise(ids):
            union(left, right)
    if used != set(range(dataset.GetNumberOfPoints())):
        raise ValueError("skin section contains unused points")
    return len({find(point) for point in used})


def validate_skin_frame(
    dataset: Any, spec: dict[str, Any], plane_y: float
) -> dict[str, int]:
    if dataset.GetNumberOfPoints() != int(
        spec["points"]
    ) or dataset.GetNumberOfCells() != int(spec["cells"]):
        raise ValueError("skin-section topology changed")
    points = vtk_points(dataset)
    if not np.allclose(points[:, 1], plane_y, atol=1e-10, rtol=0.0):
        raise ValueError("skin section left the fixed plane")
    components = line_component_count(dataset)
    if components != int(spec["components"]):
        raise ValueError("skin-section component count changed")
    return {
        "points": dataset.GetNumberOfPoints(),
        "cells": dataset.GetNumberOfCells(),
        "components": components,
    }


def render(contract_path: Path, receipt_path: Path) -> None:
    contract = read_json(contract_path)
    validate_contract(contract, receipt_path)
    if paraview_version() != VERSION:
        raise ValueError(f"ParaView version changed: {paraview_version()}")
    frames = contract["frames"]
    material_series_path = validate_identity(contract["inputs"]["material_series"])
    skin_series_path = validate_identity(contract["inputs"]["skin_series"])
    validate_series(material_series_path, frames, "material_section")
    validate_series(skin_series_path, frames, "skin_section")
    outputs = contract["outputs"]
    frames_dir = Path(str(outputs["frames_dir"])).resolve()
    pvsm = Path(str(outputs["pvsm"])).resolve()
    if frames_dir.exists() or pvsm.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite renderer outputs")
    frames_dir.mkdir(parents=True)

    pvs._DisableFirstRenderCameraReset()
    material_series = pvs.OpenDataFile(str(material_series_path))
    skin_series = pvs.OpenDataFile(str(skin_series_path))
    if material_series is None or skin_series is None:
        raise RuntimeError("ParaView cannot open the temporal VTP series")
    material_series.UpdatePipeline()
    skin_series.UpdatePipeline()
    values = [float(value) for value in material_series.TimestepValues]
    skin_values = [float(value) for value in skin_series.TimestepValues]
    if values != [float(step) for step in range(N_FRAMES)] or skin_values != values:
        raise ValueError("temporal series values changed")

    layout = pvs.CreateLayout(name="HFP1 full-head filled material section")
    view = pvs.CreateView("RenderView")
    if not layout.AssignView(0, view):
        raise RuntimeError("ParaView layout assignment failed")
    layout.SetSize(*contract["render"]["resolution"])
    configure_view(view, contract["camera"])

    material_display = pvs.Show(material_series, view, "GeometryRepresentation")
    material_display.Representation = contract["render"]["material_representation"]
    material_display.Opacity = 1.0
    material_display.EdgeColor = contract["render"]["cell_edge_rgb"]
    material_display.LineWidth = float(contract["render"]["cell_edge_width_px"])
    material_display.Ambient = 1.0
    material_display.Diffuse = 0.0
    configure_material_lut(material_display, view, contract["materials"])

    skin_display = pvs.Show(skin_series, view, "GeometryRepresentation")
    skin_display.Representation = "Wireframe"
    plain(skin_display, [float(value) for value in contract["skin"]["rgb"]])
    skin_display.LineWidth = float(contract["skin"]["line_width_px"])

    title = add_text(
        view,
        "HFP1 FULL-HEAD MATERIAL SECTION | STEP 00 / 40\n"
        "fixed Y plane | full head cut first | camera-only mouth crop\n"
        "fills = argmax continuous fractions | edges = tet cuts | teal = skin",
        "Upper Left Corner",
        22,
        bold=True,
    )

    plane_y = float(contract["plane"]["origin"][1])
    rendered: list[dict[str, Any]] = []
    for step, frame in enumerate(frames):
        pvs.GetAnimationScene().AnimationTime = float(step)
        material_series.UpdatePipeline(float(step))
        skin_series.UpdatePipeline(float(step))
        material_metrics = validate_material_frame(
            fetch(material_series),
            frame["material_section"],
            plane_y,
        )
        skin_metrics = validate_skin_frame(
            fetch(skin_series),
            frame["skin_section"],
            plane_y,
        )
        title.Text = (
            f"HFP1 FULL-HEAD MATERIAL SECTION | STEP {step:02d} / 40\n"
            "fixed Y plane | full head cut first | camera-only mouth crop\n"
            "fills = argmax continuous fractions | edges = tet cuts | teal = skin"
        )
        pvs.Render(view)
        target = frames_dir / f"frame-{step:03d}.png"
        temporary = frames_dir / f".{target.stem}.tmp.png"
        if target.exists() or temporary.exists():
            raise FileExistsError(target)
        pvs.SaveScreenshot(
            str(temporary),
            view,
            ImageResolution=contract["render"]["resolution"],
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
                "material_section": material_metrics,
                "skin_section": skin_metrics,
            }
        )

    temporary_state = pvsm.with_name(f".{pvsm.stem}.tmp{pvsm.suffix}")
    if temporary_state.exists():
        raise FileExistsError(temporary_state)
    pvs.SaveState(str(temporary_state))
    if not temporary_state.is_file() or temporary_state.stat().st_size <= 10_000:
        raise RuntimeError("ParaView did not write a substantive temporal state")
    temporary_state.replace(pvsm)
    png_paths = [Path(item["path"]) for item in rendered]
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "design": DESIGN,
            "complete": True,
            "paraview_version": VERSION,
            "frame_count": N_FRAMES,
            "TimestepValues": values,
            "frames": rendered,
            "ordered_png_sha256": ordered_digest(png_paths),
            "camera": contract["camera"],
            "plane": contract["plane"],
            "material_representation": contract["render"]["material_representation"],
            "filled_material_surface": True,
            "cell_edges_rendered": True,
            "internal_tet_section_edges_rendered": True,
            "cell_edge_rgb": contract["render"]["cell_edge_rgb"],
            "cell_edge_width_px": contract["render"]["cell_edge_width_px"],
            "determinant_metrics_rendered": False,
            "skin": contract["skin"],
            "pvsm": {"path": str(pvsm), "identity": identity(pvsm)},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    render(args.contract.resolve(), args.receipt.resolve())


if __name__ == "__main__":
    main()
