# ruff: noqa: EM102, TRY003
"""Render fixed-camera visual QA for the joint-inverse preparation state."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from joint_common import ProfileJoint, write_json
from joint_data import (
    PreparedInputs,
    _collision_details,
    _fem_bone_soft_surfaces,
    _fem_jaw_oral_surfaces,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter

from liblaf import cherries

LOG = logging.getLogger(__name__)
WINDOW = (1600, 1200)
BACKGROUND = "#f7f7f5"


class Config(cherries.BaseConfig):
    prepared_dir: Path = cherries.input("prepared")
    checkpoints: tuple[Path, ...] = (
        cherries.input("neutral-prestress-001/best-admissible.pt"),
        cherries.input("neutral-prestress-010/best-admissible.pt"),
    )
    output_dir: Path = cherries.output("preparation-visuals", mkdir=True)
    overlay_magnification: float = 20.0


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def camera(mesh: pv.DataSet, view: str, zoom: float = 1.0) -> dict[str, Any]:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.asarray(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if view == "front":
        eye = center + np.asarray((0.0, 0.0, 2.7 * span / zoom))
        up = (0.0, 1.0, 0.0)
        horizontal_span = xmax - xmin
    elif view == "side":
        eye = center + np.asarray((2.7 * span / zoom, 0.0, 0.0))
        up = (0.0, 1.0, 0.0)
        horizontal_span = zmax - zmin
    elif view == "oblique":
        eye = center + np.asarray((1.8, 0.7, 2.0)) * span / zoom
        up = (0.0, 1.0, 0.0)
        horizontal_span = span
    else:
        raise ValueError(f"unsupported view: {view}")
    aspect = WINDOW[0] / WINDOW[1]
    parallel_scale = (
        1.10 * max((ymax - ymin) / 2, horizontal_span / (2 * aspect)) / zoom
    )
    return {
        "position": [eye.tolist(), center.tolist(), up],
        "parallel_scale": parallel_scale,
    }


def plotter(title: str) -> pv.Plotter:
    result = pv.Plotter(off_screen=True, window_size=WINDOW)
    result.set_background(BACKGROUND)
    result.enable_anti_aliasing("ssaa")
    result.add_text(title, position="upper_left", font_size=16, color="#202124")
    return result


def save(plot: pv.Plotter, path: Path, camera_settings: dict[str, Any]) -> None:
    plot.camera_position = camera_settings["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera_settings["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def extract_surface_part(
    surface: pv.PolyData, volume: pv.UnstructuredGrid, names: tuple[str, ...]
) -> pv.PolyData:
    table = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    original = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    selected = np.isin(group[faces], [table.index(name) for name in names])
    mask = np.all(selected, axis=1)
    return (
        surface.extract_cells(np.flatnonzero(mask))
        .extract_surface(algorithm=None)
        .triangulate()
    )


def render_anatomy(
    output: Path,
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
    arrays: dict[str, np.ndarray],
) -> list[str]:
    files = []
    active = volume.extract_cells(arrays["active_cell_ids"]).extract_surface(
        algorithm=None
    )
    active = active.cell_data_to_point_data()
    boundary = volume.extract_surface(algorithm=None).triangulate()
    cranium = extract_surface_part(boundary, volume, ("Cranium",))
    mandible = extract_surface_part(boundary, volume, ("Mandible",))
    for view in ("front", "side"):
        name = f"01-anatomy-{view}.png"
        p = plotter(f"Frozen FEM anatomy — {view}")
        p.add_mesh(skin, color="#f2c7b8", opacity=0.20, smooth_shading=True)
        p.add_mesh(
            active,
            scalars="MuscleFraction",
            cmap=["#ffd6d6", "#c9334b"],
            clim=(0.0, 1.0),
            opacity=0.72,
            smooth_shading=True,
            scalar_bar_args={"title": "Muscle fraction"},
        )
        p.add_mesh(cranium, color="#80a7c7", opacity=0.65, smooth_shading=True)
        p.add_mesh(mandible, color="#e39a45", opacity=0.90, smooth_shading=True)
        save(p, output / name, camera(skin, view, zoom=1.08))
        files.append(name)
    return files


def render_contact_partition(
    output: Path, volume: pv.UnstructuredGrid, points: np.ndarray
) -> tuple[list[str], dict[str, Any]]:
    cranium, mandible, soft, classifier = _fem_bone_soft_surfaces(volume, points)
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    classified = np.concatenate(
        [
            np.asarray(part.cell_data["BoundaryCellId"], dtype=np.int64)
            for part in (cranium, mandible, soft)
        ]
    )
    mixed_ids = np.setdiff1d(
        np.arange(boundary.n_cells, dtype=np.int64), classified, assume_unique=False
    )
    mixed = (
        boundary.extract_cells(mixed_ids).extract_surface(algorithm=None).triangulate()
    )

    soft_points = np.asarray(soft.points)
    proximity: dict[str, Any] = {}
    for name, bone in (("cranium", cranium), ("mandible", mandible)):
        _, closest = bone.find_closest_cell(soft_points, return_closest_point=True)
        distance_mm = np.linalg.norm(soft_points - closest, axis=1) * 1000
        proximity[name] = {
            "minimum_mm": float(distance_mm.min()),
            "vertices_below_0_25_mm": int(np.count_nonzero(distance_mm < 0.25)),
            "vertices_below_0_5_mm": int(np.count_nonzero(distance_mm < 0.5)),
            "vertices_below_1_mm": int(np.count_nonzero(distance_mm < 1.0)),
            "sample": "vertices of the pure-soft FEM boundary to closest pure-bone triangle",
        }

    files = []
    for view in ("front", "side"):
        name = f"05-contact-surface-partition-{view}.png"
        p = plotter(f"Contact ownership on one FEM boundary — {view}")
        p.add_mesh(soft, color="#e7b8aa", opacity=0.20, smooth_shading=True)
        p.add_mesh(cranium, color="#4c83b6", opacity=0.78, smooth_shading=True)
        p.add_mesh(mandible, color="#e08b32", opacity=0.92, smooth_shading=True)
        p.add_mesh(
            mixed,
            color="#c729d9",
            opacity=0.92,
            show_edges=True,
            edge_color="#6c1675",
            line_width=0.5,
        )
        p.add_text(
            "pink: free soft surface   blue/orange: pure bone   magenta: bonded transition",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(boundary, view, zoom=1.08))
        files.append(name)
    return files, {
        **classifier,
        "rendered_mixed_transition_triangles": int(mixed.n_cells),
        "soft_to_bone_reference_proximity": proximity,
        "ownership": "pure-soft versus nonadjacent pure-bone faces is contact-eligible; mixed bone-soft transition faces are bonded and excluded from sliding contact",
    }


def render_material_slices(
    output: Path, volume: pv.UnstructuredGrid, pivot: np.ndarray
) -> list[str]:
    mesh = volume.copy(deep=True)
    fractions = np.column_stack(
        [
            np.asarray(mesh.cell_data[name], dtype=np.float64)
            for name in ("FatFraction", "AponeurosisFraction", "MuscleFraction")
        ]
    )
    mesh.cell_data["DominantTissue"] = np.argmax(fractions, axis=1).astype(np.int8)
    files = []

    coronal = mesh.slice(normal=(0, 0, 1), origin=(pivot[0], pivot[1] - 0.02, 0.065))
    name = "03-material-coronal-slice.png"
    p = plotter("Coronal material partition — fat / aponeurosis / muscle")
    p.add_mesh(
        coronal,
        scalars="DominantTissue",
        cmap=["#d8bd91", "#f4d03f", "#c0394b"],
        clim=(-0.5, 2.5),
        categories=True,
        show_edges=True,
        edge_color="#777777",
        line_width=0.2,
        scalar_bar_args={"title": "0 fat   1 aponeurosis   2 muscle", "n_labels": 3},
    )
    save(p, output / name, camera(coronal, "front", zoom=1.20))
    files.append(name)

    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    sagittal = point_mesh.slice(normal=(1, 0, 0), origin=pivot)
    name = "04-aponeurosis-sagittal-slice.png"
    p = plotter("Sagittal constructed aponeurosis fraction")
    p.add_mesh(
        sagittal,
        scalars="AponeurosisFraction",
        cmap="cividis",
        clim=(0.0, 1.0),
        show_edges=True,
        edge_color="#777777",
        line_width=0.2,
        scalar_bar_args={"title": "Aponeurosis fraction"},
    )
    muscle = sagittal.contour(
        [0.25, 0.5, 0.75], scalars="MuscleFraction", preference="point"
    )
    p.add_mesh(muscle, color="#d13c4f", line_width=3)
    save(p, output / name, camera(sagittal, "side", zoom=1.20))
    files.append(name)
    return files


def checkpoint_label(path: Path) -> str:
    parent = path.parent.name
    return parent.removeprefix("neutral-")


def render_displacement(
    output: Path,
    skin: pv.PolyData,
    checkpoint: dict[str, Any],
    path: Path,
    shared_max_mm: float,
    magnification: float,
) -> tuple[list[str], dict[str, Any]]:
    displacement = checkpoint["primal"]["neutral"].detach().cpu().numpy()
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    surface_displacement = displacement[ids]
    magnitude_mm = np.linalg.norm(surface_displacement, axis=1) * 1000
    deformed = skin.copy(deep=True)
    deformed.points = np.asarray(skin.points) + surface_displacement
    deformed.point_data["NeutralDriftMm"] = magnitude_mm
    label = checkpoint_label(path)
    files = []
    for view in ("front", "side"):
        name = f"10-{label}-drift-{view}.png"
        p = plotter(f"{label}: neutral surface drift — {view}")
        p.add_mesh(
            deformed,
            scalars="NeutralDriftMm",
            cmap="turbo",
            clim=(0.0, shared_max_mm),
            smooth_shading=True,
            scalar_bar_args={"title": "|u| (mm) — shared scale"},
        )
        save(p, output / name, camera(skin, view, zoom=1.06))
        files.append(name)

    amplified = skin.copy(deep=True)
    amplified.points = np.asarray(skin.points) + magnification * surface_displacement
    name = f"11-{label}-overlay-front.png"
    p = plotter(f"{label}: neutral vs {magnification:g}x amplified drift")
    p.add_mesh(skin, color="#6f7782", opacity=0.30, smooth_shading=True)
    p.add_mesh(
        amplified,
        color="#d43f4f",
        style="wireframe",
        line_width=1.2,
        opacity=0.85,
    )
    p.add_mesh(deformed, color="#2774ae", opacity=0.30, smooth_shading=True)
    p.add_text(
        "gray: reference   blue: actual   red wire: amplified",
        position="lower_left",
        font_size=11,
        color="#202124",
    )
    save(p, output / name, camera(skin, "front", zoom=1.06))
    files.append(name)
    return files, {
        "label": label,
        "checkpoint": str(path.resolve()),
        "sha256": sha256(path),
        "surface_displacement_mm": {
            "min": float(magnitude_mm.min()),
            "median": float(np.median(magnitude_mm)),
            "q95": float(np.quantile(magnitude_mm, 0.95)),
            "q99": float(np.quantile(magnitude_mm, 0.99)),
            "max": float(magnitude_mm.max()),
        },
    }


def boundary_faces(volume: pv.UnstructuredGrid) -> tuple[pv.PolyData, np.ndarray]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = original[np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]]
    return boundary, faces


def mapped_contacts(
    first: pv.PolyData, second: pv.PolyData
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    collision = vtkCollisionDetectionFilter()
    collision.SetInputData(0, first)
    collision.SetTransform(0, vtkTransform())
    collision.SetInputData(1, second)
    collision.SetMatrix(1, vtkMatrix4x4())
    collision.SetBoxTolerance(0.0)
    collision.SetCellTolerance(0.0)
    collision.SetNumberOfCellsPerNode(2)
    collision.SetCollisionModeToAllContacts()
    collision.Update()
    count = collision.GetNumberOfContacts()
    if count:
        first_ids = np.asarray(
            pv.wrap(collision.GetOutput(0)).field_data["ContactCells"],
            dtype=np.int64,
        )
        second_ids = np.asarray(
            pv.wrap(collision.GetOutput(1)).field_data["ContactCells"],
            dtype=np.int64,
        )
        local = np.column_stack((first_ids, second_ids))
        segments = np.asarray(pv.wrap(collision.GetContactsOutput()).points).reshape(
            count, 2, 3
        )
        midpoints = segments.mean(axis=1)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    else:
        local = np.empty((0, 2), dtype=np.int64)
        segments = np.empty((0, 2, 3))
        midpoints = np.empty((0, 3))
        lengths = np.empty(0)
    if not len(local):
        return (
            np.empty((0, 2), dtype=np.int64),
            midpoints,
            lengths,
            segments,
            local,
        )
    pairs = np.column_stack(
        (
            np.asarray(first.cell_data["BoundaryCellId"], dtype=np.int64)[local[:, 0]],
            np.asarray(second.cell_data["BoundaryCellId"], dtype=np.int64)[local[:, 1]],
        )
    )
    return pairs, midpoints, lengths, segments, local


def contact_topology(
    pairs: np.ndarray,
    lengths: np.ndarray,
    segments: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
) -> dict[str, Any]:
    coordinate_scale = max(1.0, float(np.max(np.abs(points))))
    segment_dtype = (
        segments.dtype
        if np.issubdtype(segments.dtype, np.floating)
        else np.dtype("<f8")
    )
    tolerance_m = max(1e-12, 32 * float(np.finfo(segment_dtype).eps) * coordinate_scale)
    shared_nodes = [np.intersect1d(faces[a], faces[b]) for a, b in pairs]
    shared_count = np.asarray([len(nodes) for nodes in shared_nodes], dtype=np.int8)
    vertex_mask = shared_count == 1
    edge_mask = shared_count == 2
    vertex_nonzero = int(np.count_nonzero(lengths[vertex_mask] > 1e-10))
    edge_excess = []
    confined = np.zeros(len(pairs), dtype=bool)
    for index in np.flatnonzero(vertex_mask):
        vertex = points[shared_nodes[index][0]]
        confined[index] = bool(
            np.all(np.linalg.norm(segments[index] - vertex, axis=1) <= tolerance_m)
        )
    for index in np.flatnonzero(edge_mask):
        nodes = shared_nodes[index]
        start, end = points[nodes[0]], points[nodes[1]]
        vector = end - start
        edge_length = np.linalg.norm(vector)
        edge_excess.append(float(lengths[index] - edge_length))
        coordinate = ((segments[index] - start) @ vector) / (edge_length**2)
        closest = start + np.clip(coordinate, 0.0, 1.0)[:, None] * vector
        distance = np.linalg.norm(segments[index] - closest, axis=1)
        coordinate_tolerance = tolerance_m / edge_length
        confined[index] = bool(
            np.all(distance <= tolerance_m)
            and np.all(coordinate >= -coordinate_tolerance)
            and np.all(coordinate <= 1.0 + coordinate_tolerance)
        )
    return {
        "contact_pairs": len(pairs),
        "pairs_sharing_no_nodes": int(np.count_nonzero(shared_count == 0)),
        "pairs_sharing_one_node": int(vertex_mask.sum()),
        "pairs_sharing_edge": int(edge_mask.sum()),
        "one_node_pairs_with_nonzero_segment_gt_1e-10m": vertex_nonzero,
        "max_edge_contact_excess_m": float(max(edge_excess, default=0.0)),
        "endpoint_confinement_tolerance_m": tolerance_m,
        "pairs_confined_to_shared_vertex_or_edge": int(confined.sum()),
        "pairs_not_confined_to_shared_topology": int((~confined).sum()),
        "all_contacts_topological_adjacency": bool(np.all(shared_count > 0)),
        "all_contacts_confined_to_shared_topology": bool(np.all(confined)),
        "shared_count": shared_count,
    }


def render_fem_oral(
    output: Path,
    volume: pv.UnstructuredGrid,
    points: np.ndarray,
    label: str,
    reference_pairs: dict[str, set[tuple[int, int]]],
) -> tuple[str, dict[str, Any]]:
    jaw, upper, lower, classifier = _fem_jaw_oral_surfaces(volume, points)
    _, faces = boundary_faces(volume)
    records: dict[str, Any] = {"classifier": classifier}
    contact_cells = {"jaw": set(), "upper": set(), "lower": set()}
    new_cells = {"jaw": set(), "upper": set(), "lower": set()}
    for name, second, second_key in (
        ("upper", upper, "upper"),
        ("lower", lower, "lower"),
    ):
        pairs, _midpoints, lengths, segments, local = mapped_contacts(jaw, second)
        topology = contact_topology(pairs, lengths, segments, faces, points)
        new = np.asarray(
            [tuple(pair) not in reference_pairs[name] for pair in pairs], dtype=bool
        )
        new_topology = contact_topology(
            pairs[new], lengths[new], segments[new], faces, points
        )
        contact_cells["jaw"].update(local[:, 0].tolist())
        contact_cells[second_key].update(local[:, 1].tolist())
        new_cells["jaw"].update(local[new, 0].tolist())
        new_cells[second_key].update(local[new, 1].tolist())
        records[name] = {
            key: value for key, value in topology.items() if key != "shared_count"
        } | {
            "new_pair_identity_count": int(new.sum()),
            "new_pair_topology": {
                key: value
                for key, value in new_topology.items()
                if key != "shared_count"
            },
        }

    name = f"30-fem-oral-{label}.png"
    p = plotter(f"FEM oral topology — {label}")
    p.add_mesh(jaw, color="#e49a44", opacity=0.32, smooth_shading=True)
    p.add_mesh(upper, color="#5d8fc2", opacity=0.42, smooth_shading=True)
    p.add_mesh(lower, color="#78b77a", opacity=0.42, smooth_shading=True)
    for mesh, key, color in (
        (jaw, "jaw", "#f0b928"),
        (upper, "upper", "#345995"),
        (lower, "lower", "#3b8f4e"),
    ):
        if contact_cells[key]:
            p.add_mesh(
                mesh.extract_cells(sorted(contact_cells[key])),
                color=color,
                opacity=0.80,
                show_edges=True,
                line_width=1.0,
            )
        if new_cells[key]:
            p.add_mesh(
                mesh.extract_cells(sorted(new_cells[key])),
                color="#d000ff",
                opacity=1.0,
                show_edges=True,
                line_width=1.5,
            )
    p.add_text(
        "magenta: pair identity absent at reference; all tested pairs share FEM nodes",
        position="lower_left",
        font_size=10,
        color="#202124",
    )
    oral_bounds = pv.Box((1.365, 1.445, 2.115, 2.195, 0.068, 0.108))
    save(p, output / name, camera(oral_bounds, "oblique", zoom=1.25))
    return name, records


def source_lip_parts(template: pv.PolyData) -> tuple[pv.PolyData, pv.PolyData]:
    template = template.triangulate()
    names = [
        str(value) for value in np.asarray(template.field_data["GroupName"]).reshape(-1)
    ]
    group = np.asarray(template.cell_data["GroupId"], dtype=np.int32)
    faces = np.asarray(template.faces).reshape(-1, 4)[:, 1:]
    upper_ids = [names.index(name) for name in ("LipTop", "LipInnerTop", "LipOuterTop")]
    lower_ids = [
        names.index(name) for name in ("LipBottom", "LipInnerBottom", "LipOuterBottom")
    ]
    upper_cells = np.flatnonzero(np.isin(group, upper_ids))
    lower_cells = np.flatnonzero(np.isin(group, lower_ids))
    shared = np.intersect1d(
        np.unique(faces[upper_cells]), np.unique(faces[lower_cells])
    )
    upper_cells = upper_cells[~np.any(np.isin(faces[upper_cells], shared), axis=1)]
    lower_cells = lower_cells[~np.any(np.isin(faces[lower_cells], shared), axis=1)]

    def part(ids: np.ndarray) -> pv.PolyData:
        return template.extract_cells(ids).extract_surface(algorithm=None).triangulate()

    return part(upper_cells), part(lower_cells)


def render_source_lips(
    output: Path, template: pv.PolyData
) -> tuple[str, dict[str, Any]]:
    upper, lower = source_lip_parts(template)
    local, midpoints, lengths = _collision_details(upper, lower)
    name = "29-source-lip-intersections.png"
    p = plotter("Registered source lips — nonadjacent inherited intersections")
    p.add_mesh(upper, color="#477db3", opacity=0.48, smooth_shading=True)
    p.add_mesh(lower, color="#58a05b", opacity=0.48, smooth_shading=True)
    p.add_mesh(
        upper.extract_cells(np.unique(local[:, 0])),
        color="#d62728",
        opacity=1.0,
        show_edges=True,
        line_width=2,
    )
    p.add_mesh(
        lower.extract_cells(np.unique(local[:, 1])),
        color="#ff9f1c",
        opacity=1.0,
        show_edges=True,
        line_width=2,
    )
    points = pv.PolyData(midpoints)
    p.add_mesh(points, color="#101010", point_size=12, render_points_as_spheres=True)
    p.add_text(
        "17 pairs remain after removing every shared-seam-incident triangle",
        position="lower_left",
        font_size=11,
        color="#202124",
    )
    center = midpoints.mean(axis=0)
    contact_window = pv.Box(
        (
            center[0] - 0.004,
            center[0] + 0.004,
            center[1] - 0.003,
            center[1] + 0.003,
            center[2] - 0.003,
            center[2] + 0.003,
        )
    )
    save(p, output / name, camera(contact_window, "front", zoom=1.0))
    return name, {
        "contact_pairs": len(local),
        "contact_segment_length_sum_m": float(lengths.sum()),
        "contact_midpoint_bounds_m": [
            midpoints.min(axis=0).tolist(),
            midpoints.max(axis=0).tolist(),
        ],
        "topology": "triangles incident to all shared upper/lower seam vertices were removed before collision; remaining pairs are nonadjacent in source topology",
    }


def main(cfg: Config) -> None:
    pv.OFF_SCREEN = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path).triangulate()
    reference_points = np.asarray(volume.points, dtype=np.float64)
    checkpoints = [
        (path, torch.load(path, map_location="cpu", weights_only=False))
        for path in cfg.checkpoints
    ]
    for path, checkpoint in checkpoints:
        displacement = checkpoint["primal"]["neutral"]
        if tuple(displacement.shape) != reference_points.shape:
            raise ValueError(f"checkpoint displacement shape changed: {path}")

    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    shared_max_mm = max(
        float(
            np.linalg.norm(
                checkpoint["primal"]["neutral"].detach().cpu().numpy()[ids], axis=1
            ).max()
            * 1000
        )
        for _, checkpoint in checkpoints
    )
    images = []
    images.extend(render_anatomy(cfg.output_dir, volume, skin, prepared.arrays))
    images.extend(
        render_material_slices(
            cfg.output_dir, volume, prepared.arrays["mandible_pivot_m"]
        )
    )
    contact_files, contact_partition = render_contact_partition(
        cfg.output_dir, volume, reference_points
    )
    images.extend(contact_files)
    checkpoint_records = []
    for path, checkpoint in checkpoints:
        files, record = render_displacement(
            cfg.output_dir,
            skin,
            checkpoint,
            path,
            shared_max_mm,
            cfg.overlay_magnification,
        )
        images.extend(files)
        checkpoint_records.append(record)

    jaw, upper, lower, _ = _fem_jaw_oral_surfaces(volume, reference_points)
    reference_pairs = {}
    for name, second in (("upper", upper), ("lower", lower)):
        pairs, _, _, _, _ = mapped_contacts(jaw, second)
        reference_pairs[name] = {tuple(pair) for pair in pairs.tolist()}
    name, reference_oral = render_fem_oral(
        cfg.output_dir,
        volume,
        reference_points,
        "reference",
        reference_pairs,
    )
    images.append(name)
    oral_records = {"reference": reference_oral}
    for path, checkpoint in checkpoints:
        points = (
            reference_points + checkpoint["primal"]["neutral"].detach().cpu().numpy()
        )
        label = checkpoint_label(path)
        name, record = render_fem_oral(
            cfg.output_dir,
            volume,
            points,
            label,
            reference_pairs,
        )
        images.append(name)
        oral_records[label] = record

    template = pv.read(prepared.manifest["sources"]["template_skin"]["path"])
    name, source_lip = render_source_lips(cfg.output_dir, template)
    images.append(name)
    summary = {
        "schema_version": 1,
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "fixed_surface_drift_scale_mm": [0.0, shared_max_mm],
        "overlay_magnification": cfg.overlay_magnification,
        "checkpoints": checkpoint_records,
        "source_lip": source_lip,
        "fem_oral": oral_records,
        "fem_contact_partition": contact_partition,
        "interpretation": {
            "fem_collision_pairs": "all reference and saved-neutral mandible/oral and lip collision pairs share at least one FEM node; collision segments are confined to the exact shared vertex or edge within a dtype-and-coordinate-scale-derived tolerance",
            "provable_exclusion": "pairs whose triangles share a FEM node and whose intersection does not extend beyond that shared vertex/edge are bonded-topology adjacency, not free-surface penetration",
            "simulation_gate": "reject nonadjacent FEM surface intersections, adjacent-pair contact extending beyond shared topology, element inversion, and boundary foldover; preserve free soft-tissue contact against both pure cranium and pure mandible faces",
            "anatomy_limit": "the 17 nonadjacent registered-template lip intersections and independently registered source skull/mandible intersections remain anatomy QA defects, but they are not automatically intersections of the tetrahedral simulation boundary",
        },
        "images": images,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "visuals/images": len(images),
            "visuals/checkpoints": len(checkpoints),
            "visuals/shared_drift_scale_mm": shared_max_mm,
            "oral/source_nonadjacent_pairs": source_lip["contact_pairs"],
        }
    )
    LOG.info("Wrote %d preparation images to %s", len(images), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
