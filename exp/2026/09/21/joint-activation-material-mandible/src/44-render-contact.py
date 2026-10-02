"""Render the actual IPC soft-tissue/bone contact state and gradient."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_contact import build_owned_contact
from joint_data import PreparedInputs

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)
WINDOW = (1600, 1200)
BACKGROUND = "#f7f7f5"


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    contact_spec: Path = GROUP / "data/contact/config.json"
    contact_validation: Path = (
        GROUP / "data/contact-initial-validation-002/summary.json"
    )
    checkpoint: Path | None = None
    output_dir: Path = cherries.output("contact-visuals", mkdir=True)


def load_stable_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"checkpoint changed during read: {path}"
    assert len(payload) == before.st_size
    return (
        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False),
        hashlib.sha256(payload).hexdigest(),
    )


def camera(mesh: pv.DataSet, view: str, zoom: float = 1.0) -> dict[str, Any]:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.asarray(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if view == "front":
        eye = center + np.asarray((0.0, 0.0, 2.7 * span))
        horizontal_span = xmax - xmin
    elif view == "side":
        eye = center + np.asarray((2.7 * span, 0.0, 0.0))
        horizontal_span = zmax - zmin
    else:
        raise ValueError(view)
    aspect = WINDOW[0] / WINDOW[1]
    parallel_scale = (
        1.10 * max((ymax - ymin) / 2, horizontal_span / (2 * aspect)) / zoom
    )
    return {
        "position": [eye.tolist(), center.tolist(), (0.0, 1.0, 0.0)],
        "parallel_scale": parallel_scale,
    }


def plotter(title: str) -> pv.Plotter:
    result = pv.Plotter(off_screen=True, window_size=WINDOW)
    result.set_background(BACKGROUND)
    result.enable_anti_aliasing("ssaa")
    result.add_text(title, position="upper_left", font_size=16, color="#202124")
    return result


def save(plot: pv.Plotter, path: Path, settings: dict[str, Any]) -> None:
    plot.camera_position = settings["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = settings["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def scalar_bar(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.24,
        "width": 0.07,
        "height": 0.50,
    }


def boundary_partition(
    volume: pv.UnstructuredGrid, points: np.ndarray
) -> dict[str, pv.PolyData]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    cranium_id, mandible_id = names.index("Cranium"), names.index("Mandible")
    is_cranium = labels[faces] == cranium_id
    is_mandible = labels[faces] == mandible_id
    masks = {
        "cranium": np.all(is_cranium, axis=1),
        "mandible": np.all(is_mandible, axis=1),
        "soft": np.all(~(is_cranium | is_mandible), axis=1),
    }
    masks["bonded"] = ~(masks["cranium"] | masks["mandible"] | masks["soft"])
    result = {}
    for name, mask in masks.items():
        result[name] = (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )
    return result


def _surface_distance(points: np.ndarray, surface: pv.PolyData) -> np.ndarray:
    _, closest = surface.find_closest_cell(points, return_closest_point=True)
    return np.linalg.norm(points - np.asarray(closest), axis=1)


def _distance_quantiles(points: np.ndarray, surface: pv.PolyData) -> dict[str, float]:
    distance = _surface_distance(points, surface)
    return {
        "minimum_m": float(distance.min()),
        "median_m": float(np.median(distance)),
        "q95_m": float(np.quantile(distance, 0.95)),
        "q99_m": float(np.quantile(distance, 0.99)),
        "maximum_m": float(distance.max()),
        "within_1um_fraction": float(np.mean(distance <= 1.0e-6)),
        "within_0p1mm_fraction": float(np.mean(distance <= 1.0e-4)),
    }


def collider_scope_audit(
    volume: pv.UnstructuredGrid, prepared: PreparedInputs
) -> dict[str, Any]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    local_faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    faces = original[local_faces]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    historical = prepared.arrays["historical_fixed_node_ids"]
    result: dict[str, Any] = {}
    for bone, source_key in (
        ("cranium", "cranium_surface"),
        ("mandible", "mandible_surface"),
    ):
        mask = np.all(labels[faces] == names.index(bone.title()), axis=1)
        collider_ids = np.unique(faces[mask])
        support = prepared.arrays[f"{bone}_node_ids"]
        missing_support = np.setdiff1d(collider_ids, support)
        assert not len(missing_support)
        fem_surface = (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )
        source = pv.read(prepared.manifest["sources"][source_key]["path"]).triangulate()
        result[bone] = {
            "pure_boundary_triangles": int(mask.sum()),
            "collider_vertices": len(collider_ids),
            "support_nodes": len(support),
            "collider_vertices_outside_support": len(missing_support),
            "support_nodes_not_on_pure_collider": len(
                np.setdiff1d(support, collider_ids)
            ),
            "collider_vertices_in_historical_IsFixed": len(
                np.intersect1d(collider_ids, historical)
            ),
            "collider_vertices_added_by_recovered_group_support": len(
                np.setdiff1d(collider_ids, historical)
            ),
            "runtime_boundary_ownership": (
                "fixed recovered cranium support"
                if bone == "cranium"
                else "differentiable rigid mandible support"
            ),
            "source_surface": {
                "path": prepared.manifest["sources"][source_key]["path"],
                "points": source.n_points,
                "triangles": source.n_cells,
                "fem_collider_vertex_to_source_surface": _distance_quantiles(
                    np.asarray(fem_surface.points), source
                ),
                "source_vertex_to_fem_collider_surface": _distance_quantiles(
                    np.asarray(source.points), fem_surface
                ),
            },
        }
    result["scope"] = (
        "complete partition of the labeled face-subvolume FEM boundary under the pure-face ownership rule; not complete anatomical coverage of the independently registered source bones"
    )
    result["source_comparison_limit"] = (
        "nearest vertex-to-surface distances quantify geometric coverage only; they do not establish bijective correspondence or anatomical completeness"
    )
    return result


def _mask_receipt(
    mask: np.ndarray,
    points: np.ndarray,
    area_weight: np.ndarray,
    *,
    source_distance: np.ndarray,
    collider_distance: np.ndarray,
    mixed_distance: np.ndarray,
) -> dict[str, Any]:
    selected = points[mask]
    return {
        "vertices": int(mask.sum()),
        "vertex_fraction": float(mask.mean()),
        "area_m2": float(area_weight[mask].sum()),
        "area_fraction": float(area_weight[mask].sum() / area_weight.sum()),
        "bounds_m": (
            [selected.min(axis=0).tolist(), selected.max(axis=0).tolist()]
            if len(selected)
            else None
        ),
        "source_distance_max_m": (
            float(source_distance[mask].max()) if mask.any() else None
        ),
        "fem_collider_distance_min_m": (
            float(collider_distance[mask].min()) if mask.any() else None
        ),
        "fem_collider_distance_max_m": (
            float(collider_distance[mask].max()) if mask.any() else None
        ),
        "mixed_distance_min_m": (
            float(mixed_distance[mask].min()) if mask.any() else None
        ),
    }


def collider_relevance_audit(
    volume: pv.UnstructuredGrid, prepared: PreparedInputs
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    local_faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    faces = original[local_faces]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    is_cranium = labels[faces] == names.index("Cranium")
    is_mandible = labels[faces] == names.index("Mandible")
    masks = {
        "cranium": np.all(is_cranium, axis=1),
        "mandible": np.all(is_mandible, axis=1),
        "soft": np.all(~(is_cranium | is_mandible), axis=1),
    }
    masks["mixed"] = ~(masks["cranium"] | masks["mandible"] | masks["soft"])

    def subset(mask: np.ndarray) -> pv.PolyData:
        return (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )

    surfaces = {name: subset(mask) for name, mask in masks.items()}
    soft = surfaces["soft"]
    soft_points = np.asarray(soft.points)
    soft_faces = np.asarray(soft.faces).reshape(-1, 4)[:, 1:]
    xyz = soft_points[soft_faces]
    triangle_area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    area_weight = np.zeros(soft.n_points)
    np.add.at(
        area_weight,
        soft_faces.reshape(-1),
        np.repeat(triangle_area / 3, 3),
    )
    mixed_distance = _surface_distance(soft_points, surfaces["mixed"])
    summary: dict[str, Any] = {
        "definition": "pure-soft FEM boundary vertices near an independently registered source bone but farther from the declared pure-face FEM bone collider",
        "soft_boundary": {
            "vertices": soft.n_points,
            "triangles": soft.n_cells,
            "area_m2": float(triangle_area.sum()),
        },
        "primary_issue_definition": "source distance <=1 mm, pure FEM bone collider distance >2 mm, and bonded mixed-face distance >1 mm",
    }
    render_data: dict[str, dict[str, Any]] = {}
    for bone, source_key in (
        ("cranium", "cranium_surface"),
        ("mandible", "mandible_surface"),
    ):
        fem = surfaces[bone]
        source = pv.read(prepared.manifest["sources"][source_key]["path"]).triangulate()
        source_distance = _surface_distance(soft_points, source)
        collider_distance = _surface_distance(soft_points, fem)
        rows: dict[str, Any] = {}
        for near_m in (0.0005, 0.001):
            for far_m in (0.001, 0.002):
                candidate = (source_distance <= near_m) & (collider_distance > far_m)
                bonded = candidate & (mixed_distance <= near_m)
                unaccounted = candidate & (mixed_distance > near_m)
                rows[f"source_le_{near_m * 1000:g}mm_fem_gt_{far_m * 1000:g}mm"] = {
                    "all": _mask_receipt(
                        candidate,
                        soft_points,
                        area_weight,
                        source_distance=source_distance,
                        collider_distance=collider_distance,
                        mixed_distance=mixed_distance,
                    ),
                    "bonded_mixed_within_source_threshold": _mask_receipt(
                        bonded,
                        soft_points,
                        area_weight,
                        source_distance=source_distance,
                        collider_distance=collider_distance,
                        mixed_distance=mixed_distance,
                    ),
                    "unaccounted_after_mixed": _mask_receipt(
                        unaccounted,
                        soft_points,
                        area_weight,
                        source_distance=source_distance,
                        collider_distance=collider_distance,
                        mixed_distance=mixed_distance,
                    ),
                }
        source_to_fem = _surface_distance(np.asarray(source.points), fem)
        source_to_soft = _surface_distance(np.asarray(source.points), soft)
        source_far = source_to_fem > 0.002
        summary[bone] = {
            "thresholds": rows,
            "source_frame_alignment": {
                "evidence": "median FEM-collider vertex to source-surface distance is near machine precision and q95 is below 5 micrometres",
                "fem_collider_vertex_to_source_surface": _distance_quantiles(
                    np.asarray(fem.points), source
                ),
                "passed_same_world_frame_check": bool(
                    np.median(_surface_distance(np.asarray(fem.points), source))
                    < 1.0e-9
                ),
            },
            "source_vertex_sampling_outside_fem_collider": {
                "source_vertices": source.n_points,
                "farther_than_2mm_from_fem_collider": int(source_far.sum()),
                "also_within_1mm_of_pure_soft_domain": int(
                    np.sum(source_far & (source_to_soft <= 0.001))
                ),
                "also_farther_than_2mm_from_pure_soft_domain": int(
                    np.sum(source_far & (source_to_soft > 0.002))
                ),
                "interpretation": "the latter samples are outside both collider and simulated pure-soft boundary and primarily measure crop/domain extent rather than a local contact hole",
            },
        }
        issue = (
            (source_distance <= 0.001)
            & (collider_distance > 0.002)
            & (mixed_distance > 0.001)
        )
        assert issue.any(), bone
        render_data[bone] = {
            "soft": soft,
            "fem": fem,
            "mixed": surfaces["mixed"],
            "source": source,
            "issue": issue,
            "collider_distance_mm": 1000.0 * collider_distance,
            "receipt": rows["source_le_1mm_fem_gt_2mm"]["unaccounted_after_mixed"],
        }
    summary["interpretation"] = (
        "primary issue vertices are adjacent to the simulated soft-tissue domain and are therefore relevance flags for the frozen numerical collider; source vertices far from both FEM collider and soft boundary are crop/domain differences, not direct evidence of an omitted contact patch"
    )
    return summary, render_data


def render_collider_relevance(
    output: Path, render_data: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for bone, data in render_data.items():
        issue_points = np.asarray(data["soft"].points)[data["issue"]]
        gap_mm = data["collider_distance_mm"][data["issue"]]
        bounds = expanded_bounds(issue_points)
        focus = pv.Box(bounds)
        for view in ("front", "side"):
            name = f"05-{bone}-collider-relevance-{view}.png"
            p = plotter(f"{bone.title()} collider relevance flags — {view}")
            p.add_mesh(
                clipped(data["source"], bounds),
                color="#2f7d4a",
                opacity=0.30,
                style="wireframe",
                line_width=0.7,
            )
            p.add_mesh(
                clipped(data["fem"], bounds),
                color="#4c83b6" if bone == "cranium" else "#df8c2f",
                opacity=0.50,
                show_edges=True,
                edge_color="#5e6670",
                line_width=0.30,
            )
            p.add_mesh(clipped(data["soft"], bounds), color="#ead0c7", opacity=0.14)
            p.add_mesh(
                clipped(data["mixed"], bounds),
                color="#d52bd0",
                opacity=0.25,
                style="wireframe",
                line_width=0.7,
            )
            cloud = pv.PolyData(issue_points)
            cloud.point_data["ColliderGapMm"] = gap_mm
            spheres = cloud.glyph(
                geom=pv.Sphere(radius=0.00035, theta_resolution=16, phi_resolution=16),
                scale=False,
                orient=False,
            )
            p.add_mesh(
                spheres,
                scalars="ColliderGapMm",
                cmap="turbo",
                clim=(2.0, float(gap_mm.max())),
                scalar_bar_args=scalar_bar("distance to FEM collider (mm)"),
            )
            p.add_text(
                "markers: soft <=1 mm from source, >2 mm from pure collider, >1 mm from bonded mixed faces",
                position="lower_left",
                font_size=9,
                color="#202124",
            )
            save(p, output / name, camera(focus, view))
            receipt = data["receipt"]
            assets.append(
                {
                    "filename": name,
                    "caption": f"{view.title()} localization of {receipt['vertices']} pure-soft boundary vertices ({100 * receipt['area_fraction']:.4g}% of pure-soft vertex-area weight) within 1 mm of the registered source {bone} but more than 2 mm from the pure FEM {bone} collider and more than 1 mm from bonded mixed faces. Green wire is source bone, solid color is the FEM collider, magenta is bonded transition topology, and marker color is collider gap in mm.",
                }
            )
    return assets


def contact_records(
    contact: Any,
    state: Any,
    displacement: torch.Tensor,
    volume: pv.UnstructuredGrid,
) -> list[dict[str, Any]]:
    positions = (contact.vertices + displacement[contact.indices]).numpy(force=True)
    edges = contact.collision_mesh.edges
    faces = contact.collision_mesh.faces
    global_ids = contact.indices.numpy(force=True)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    bone_ids = {"cranium": names.index("Cranium"), "mandible": names.index("Mandible")}
    result = []
    for collection in (
        "vv_collisions",
        "ev_collisions",
        "ee_collisions",
        "fv_collisions",
    ):
        for collision in getattr(state.collisions, collection):
            local_ids = np.asarray(collision.vertex_ids(edges, faces), dtype=np.int64)
            local_ids = local_ids[local_ids >= 0]
            stencil = np.asarray(collision.dof(positions, edges, faces))
            xyz = stencil.reshape(-1, 3)
            coefficients = np.asarray(
                collision.compute_coefficients(stencil), dtype=np.float64
            ).reshape(-1)
            positive = coefficients > 0
            negative = coefficients < 0
            assert positive.any()
            assert negative.any()
            point_a = (xyz[positive] * coefficients[positive, None]).sum(
                axis=0
            ) / coefficients[positive].sum()
            point_b = (xyz[negative] * (-coefficients[negative, None])).sum(axis=0) / (
                -coefficients[negative].sum()
            )
            stencil_labels = labels[global_ids[local_ids]]
            targets = [
                name for name, value in bone_ids.items() if value in stencil_labels
            ]
            assert len(targets) == 1, (collection, targets, stencil_labels)
            result.append(
                {
                    "kind": collection.removesuffix("_collisions"),
                    "bone": targets[0],
                    "gap_m": float(np.linalg.norm(point_a - point_b)),
                    "location_m": ((point_a + point_b) / 2).tolist(),
                }
            )
    return result


def expanded_bounds(points: np.ndarray) -> tuple[float, ...]:
    lower = points.min(axis=0) - np.asarray((0.008, 0.012, 0.008))
    upper = points.max(axis=0) + np.asarray((0.008, 0.012, 0.008))
    return (
        lower[0],
        upper[0],
        lower[1],
        upper[1],
        lower[2],
        upper[2],
    )


def clipped(mesh: pv.PolyData, bounds: tuple[float, ...]) -> pv.PolyData:
    return (
        mesh.clip_box(bounds, invert=False)
        .extract_surface(algorithm=None)
        .triangulate()
    )


def render_locations(
    output: Path,
    surfaces: dict[str, pv.PolyData],
    records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    points = np.asarray([record["location_m"] for record in records])
    target = np.asarray([0 if record["bone"] == "cranium" else 1 for record in records])
    bounds = expanded_bounds(points)
    focus = pv.Box(bounds)
    assets = []
    for view in ("front", "side"):
        name = f"01-active-contact-locations-{view}.png"
        p = plotter(f"IPC active soft-tissue/bone locations — {view}")
        for key, color in (("cranium", "#4c83b6"), ("mandible", "#df8c2f")):
            p.add_mesh(
                clipped(surfaces[key], bounds),
                color=color,
                opacity=0.42,
                show_edges=True,
                edge_color="#5e6670",
                line_width=0.35,
            )
        p.add_mesh(
            clipped(surfaces["soft"], bounds),
            color="#ead0c7",
            opacity=0.15,
        )
        p.add_mesh(
            clipped(surfaces["bonded"], bounds),
            color="#d52bd0",
            opacity=0.30,
            style="wireframe",
            line_width=0.8,
        )
        for value, color in (
            (0, "#1155cc"),
            (1, "#e65100"),
        ):
            cloud = pv.PolyData(points[target == value])
            spheres = cloud.glyph(
                geom=pv.Sphere(radius=0.00045, theta_resolution=16, phi_resolution=16),
                scale=False,
                orient=False,
            )
            p.add_mesh(
                spheres,
                color=color,
            )
        p.add_text(
            "blue: cranium contact   orange: mandible contact   magenta wire: bonded",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(focus, view))
        assets.append(
            {
                "filename": name,
                "caption": f"{view.title()} cutaway of runtime IPC active locations: blue targets cranium, orange targets mandible, and magenta is excluded bonded attachment topology.",
            }
        )
    return assets


def force_summary(
    force_n: np.ndarray, volume: pv.UnstructuredGrid
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    magnitude = np.linalg.norm(force_n, axis=1)
    cutoff = max(float(magnitude.max()) * 1e-12, 1e-18)
    summary: dict[str, Any] = {}
    active: dict[str, np.ndarray] = {}
    for name in ("cranium", "mandible"):
        ids = np.flatnonzero(
            (labels == names.index(name.title())) & (magnitude > cutoff)
        )
        values = magnitude[ids]
        resultant = force_n[ids].sum(axis=0)
        active[name] = ids
        summary[name] = {
            "active_bone_nodes": len(ids),
            "resultant_vector_N": resultant.tolist(),
            "resultant_magnitude_N": float(np.linalg.norm(resultant)),
            "sum_nodal_magnitudes_N": float(values.sum()),
            "rms_active_nodal_magnitude_N": float(np.sqrt(np.mean(values**2))),
            "maximum_nodal_magnitude_N": float(values.max()),
        }
    global_resultant = force_n.sum(axis=0)
    summary["definitions"] = {
        "force": "negative IPC energy gradient; model MPa*m^2 multiplied by 1e6 gives N",
        "resultant_magnitude_N": "norm of the vector sum over active nodes carrying the selected bone label",
        "sum_nodal_magnitudes_N": "sum of individual nodal force magnitudes; not a net resultant",
        "rms_active_nodal_magnitude_N": "root mean square of magnitudes over nonzero selected bone nodes",
        "pressure_limit": "nodal locations and forces are not physical pressure without a validated nodal/patch area division",
        "global_action_reaction_resultant_N": global_resultant.tolist(),
        "global_action_reaction_resultant_magnitude_N": float(
            np.linalg.norm(global_resultant)
        ),
    }
    return summary, active


def render_forces(
    output: Path,
    surfaces: dict[str, pv.PolyData],
    points: np.ndarray,
    force_n: np.ndarray,
    active: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    contact_points = np.asarray([record["location_m"] for record in records])
    bounds = expanded_bounds(contact_points)
    focus = pv.Box(bounds)
    all_ids = np.concatenate([active["cranium"], active["mandible"]])
    magnitude = np.linalg.norm(force_n[all_ids], axis=1)
    clim = (float(magnitude.min()), float(magnitude.max()))
    assets = []
    for view in ("front", "side"):
        name = f"02-bone-contact-force-{view}.png"
        p = plotter(f"IPC bone-side nodal contact force — {view}")
        for key, color in (("cranium", "#779abb"), ("mandible", "#d89a57")):
            p.add_mesh(
                clipped(surfaces[key], bounds),
                color=color,
                opacity=0.40,
                show_edges=True,
                edge_color="#616a73",
                line_width=0.35,
            )
        p.add_mesh(
            clipped(surfaces["bonded"], bounds),
            color="#d52bd0",
            opacity=0.25,
            style="wireframe",
            line_width=0.8,
        )
        cloud = pv.PolyData(points[all_ids])
        cloud.point_data["ForceVectorN"] = force_n[all_ids]
        cloud.point_data["ForceMagnitudeN"] = magnitude
        spheres = cloud.glyph(
            geom=pv.Sphere(radius=0.00055, theta_resolution=20, phi_resolution=20),
            scale=False,
            orient=False,
        )
        p.add_mesh(
            spheres,
            scalars="ForceMagnitudeN",
            cmap="viridis",
            clim=clim,
            log_scale=True,
            scalar_bar_args={
                "title": "|f| (N, log)",
                "vertical": True,
                "position_x": 0.86,
                "position_y": 0.24,
                "width": 0.07,
                "height": 0.50,
            },
        )
        arrows = cloud.glyph(orient="ForceVectorN", scale="ForceMagnitudeN", factor=4.0)
        p.add_mesh(arrows, color="#222222", opacity=0.72)
        p.add_text(
            "negative IPC gradient; arrows share 4 m/N scale; magenta is bonded",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(focus, view))
        assets.append(
            {
                "filename": name,
                "caption": f"{view.title()} cutaway of cranium and mandible nodal contact-force magnitudes in newtons on a shared log color scale; arrows show force direction and magenta marks bonded attachment faces.",
            }
        )
    return assets


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def render_gap_histogram(
    output: Path, records: list[dict[str, Any]], dhat_m: float
) -> tuple[dict[str, str], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    bins = np.linspace(0, dhat_m * 1000, 21)
    gap_summary = {}
    for bone, color in (("cranium", "#2868a5"), ("mandible", "#dc711b")):
        values = np.asarray(
            [record["gap_m"] * 1000 for record in records if record["bone"] == bone]
        )
        ax.hist(
            values,
            bins=bins,
            alpha=0.72,
            color=color,
            label=f"{bone} (n={len(values)})",
        )
        gap_summary[bone] = {"count": len(values), "gap_mm": quantiles(values)}
    ax.axvline(dhat_m * 1000, color="#222222", linestyle="--", label="dhat")
    ax.set(xlabel="Active IPC gap (mm)", ylabel="Collision stencil count")
    ax.set_title("Reference IPC active-gap distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    name = "03-active-gap-histogram.png"
    fig.savefig(output / name, facecolor="white")
    plt.close(fig)
    return (
        {
            "filename": name,
            "caption": "Active IPC stencil gaps split by target bone; the dashed line is dhat=0.1 mm. Counts describe collision-set representatives, not anatomical contact area.",
        },
        gap_summary,
    )


def render_force_totals(output: Path, summary: dict[str, Any]) -> dict[str, str]:
    labels = ("resultant", "sum magnitudes", "RMS active-node")
    keys = (
        "resultant_magnitude_N",
        "sum_nodal_magnitudes_N",
        "rms_active_nodal_magnitude_N",
    )
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for offset, bone, color in (
        (-width / 2, "cranium", "#2868a5"),
        (width / 2, "mandible", "#dc711b"),
    ):
        ax.bar(
            x + offset,
            [summary[bone][key] for key in keys],
            width,
            label=bone,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Bone-side nodal contact force (N, log scale)")
    ax.set_title("Reference IPC bone-force summaries")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    name = "04-bone-force-summary.png"
    fig.savefig(output / name, facecolor="white")
    plt.close(fig)
    return {
        "filename": name,
        "caption": "Per-bone force summaries. Resultant is the norm of the vector sum, sum magnitudes is non-cancelling, and RMS is over active bone nodes; none is pressure.",
    }


def write_report(output: Path, summary: dict[str, Any]) -> None:
    runtime = summary["runtime_contact"]
    force = summary["bone_force_N"]
    collider = summary["collider_scope_audit"]
    relevance = summary["collider_relevance_audit"]
    assets = summary["assets"]
    lines = [
        "# IPC contact visual audit",
        "",
        f"State: `{summary['state']['label']}`. Frozen native-filter preflight recorded "
        f"{summary['preflight']['active_contact_count']} active pairs with minimum active "
        f"gap {summary['preflight']['minimum_active_gap_mm']:.5f} mm. The rendered runtime "
        f"reconstruction contains {runtime['active_contact_count']} collision-set representatives; "
        "the energy, minimum gap, and nodal gradient are the physical diagnostics.",
        "",
    ]
    for asset in assets:
        lines.extend(
            [
                f"## {asset['filename']}",
                "",
                f"![{asset['filename']}]({asset['filename']})",
                "",
                asset["caption"],
                "",
            ]
        )
    lines.extend(
        [
            "## Force values",
            "",
            f"Cranium: resultant {force['cranium']['resultant_magnitude_N']:.6g} N, "
            f"sum magnitudes {force['cranium']['sum_nodal_magnitudes_N']:.6g} N, "
            f"RMS {force['cranium']['rms_active_nodal_magnitude_N']:.6g} N.",
            "",
            f"Mandible: resultant {force['mandible']['resultant_magnitude_N']:.6g} N, "
            f"sum magnitudes {force['mandible']['sum_nodal_magnitudes_N']:.6g} N, "
            f"RMS {force['mandible']['rms_active_nodal_magnitude_N']:.6g} N.",
            "",
            "Force is the negative IPC energy gradient. Model MPa*m^2 is multiplied by "
            "1e6 to report newtons. Locations and nodal forces are not physical pressure "
            "without a validated nodal or patch-area division.",
            "",
            "## Collider ownership and coverage",
            "",
            f"All {collider['cranium']['collider_vertices']:,} pure-cranium collider "
            f"vertices belong to the recovered fixed support; all "
            f"{collider['mandible']['collider_vertices']:,} pure-mandible collider "
            "vertices belong to the differentiable rigid-jaw support. No collider "
            "vertex lies outside its runtime support.",
            "",
            "This is a complete partition of the selected labeled FEM face-subvolume "
            "boundary under the pure-face ownership rule. It is not complete anatomical "
            "coverage of the independently registered source cranium or mandible.",
            "",
            "Conditioning on the actual simulated pure-soft boundary leaves "
            f"{relevance['cranium']['thresholds']['source_le_1mm_fem_gt_2mm']['unaccounted_after_mixed']['vertices']} "
            "cranium-near and "
            f"{relevance['mandible']['thresholds']['source_le_1mm_fem_gt_2mm']['unaccounted_after_mixed']['vertices']} "
            "mandible-near vertices within 1 mm of the registered source bone, more than "
            "2 mm from the pure FEM collider, and more than 1 mm from bonded mixed faces. "
            "These are relevance flags for the frozen numerical collider, not proof of "
            "anatomical contact or penetration.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def main(cfg: Config) -> None:  # noqa: PLR0915
    pv.OFF_SCREEN = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    config = json.loads(cfg.contact_spec.read_text())
    validation = json.loads(cfg.contact_validation.read_text())
    assert validation["success"] is True
    assert validation["contact_spec_sha256"] == sha256(cfg.contact_spec)
    assert validation["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )
    fixed = np.unique(
        np.concatenate(
            [
                prepared.arrays[name]
                for name in (
                    "cranium_node_ids",
                    "mandible_node_ids",
                    "historical_fixed_node_ids",
                )
            ]
        )
    )
    contact, surface_map = build_owned_contact(volume, fixed, config)
    collider_scope = collider_scope_audit(volume, prepared)
    collider_relevance, collider_relevance_render = collider_relevance_audit(
        volume, prepared
    )
    if cfg.checkpoint is None:
        displacement = torch.zeros((volume.n_points, 3), dtype=torch.float64)
        state_label = "reference"
        checkpoint = None
    else:
        checkpoint_value, checkpoint_hash = load_stable_checkpoint(cfg.checkpoint)
        displacement = checkpoint_value["primal"]["neutral"].detach().cpu()
        assert tuple(displacement.shape) == (volume.n_points, 3)
        state_label = cfg.checkpoint.parent.name
        checkpoint = {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": checkpoint_hash,
        }
    state = contact.state_at(displacement)
    diagnostics = contact.diagnostics(state, displacement)
    assert diagnostics["contact_numerically_valid"] is True
    if cfg.checkpoint is None:
        expected = validation["reference"]
        assert math.isclose(
            diagnostics["barrier_energy"], expected["barrier_energy"], rel_tol=1e-12
        )
        assert math.isclose(
            diagnostics["minimum_active_distance_m"],
            expected["minimum_active_distance_m"],
            rel_tol=1e-12,
        )
    records = contact_records(contact, state, displacement, volume)
    assert len(records) == diagnostics["active_contact_count"]
    points = np.asarray(volume.points) + displacement.numpy(force=True)
    surfaces = boundary_partition(volume, points)
    gradient = torch.zeros_like(displacement)
    contact.grad(state, displacement, gradient)
    force_n = -gradient.numpy(force=True) * 1e6
    bone_force, active_nodes = force_summary(force_n, volume)
    noncancelling = float(np.linalg.norm(force_n, axis=1).sum())
    assert bone_force["definitions"][
        "global_action_reaction_resultant_magnitude_N"
    ] <= max(noncancelling * 1e-10, 1e-15)

    assets = []
    assets.extend(render_locations(cfg.output_dir, surfaces, records))
    assets.extend(
        render_forces(
            cfg.output_dir,
            surfaces,
            points,
            force_n,
            active_nodes,
            records,
        )
    )
    asset, gap_summary = render_gap_histogram(cfg.output_dir, records, config["dhat_m"])
    assets.append(asset)
    assets.append(render_force_totals(cfg.output_dir, bone_force))
    assets.extend(render_collider_relevance(cfg.output_dir, collider_relevance_render))
    pair_types = {
        kind: sum(record["kind"] == kind for record in records)
        for kind in ("vv", "ev", "ee", "fv")
    }
    per_bone_runtime = {
        bone: sum(record["bone"] == bone for record in records)
        for bone in ("cranium", "mandible")
    }
    summary = {
        "schema": "joint-contact-visual-audit-v1",
        "state": {"label": state_label, "checkpoint": checkpoint},
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "contact_spec_sha256": sha256(cfg.contact_spec),
        "contact_validation_sha256": sha256(cfg.contact_validation),
        "preflight": {
            "success": validation["success"],
            "active_contact_count": validation["reference"]["active_contact_count"],
            "minimum_active_gap_mm": validation["reference"][
                "minimum_active_distance_m"
            ]
            * 1000,
            "barrier_energy_MPa_m3": validation["reference"]["barrier_energy"],
        },
        "runtime_contact": {
            **diagnostics,
            "active_contact_count_by_bone": per_bone_runtime,
            "collision_stencil_types": pair_types,
            "representation_note": "IMPROVED_MAX_APPROX may choose a different number of degenerate collision-set representatives across LBVH rebuilds; energy, minimum distance, force gradient, and geometry locations are the quantitative diagnostics",
        },
        "gap_summary": gap_summary,
        "bone_force_N": bone_force,
        "surface_map": surface_map,
        "collider_scope_audit": collider_scope,
        "collider_relevance_audit": collider_relevance,
        "interpretation": {
            "contact": "free pure-soft versus pure-cranium/pure-mandible IPC only",
            "bonded": "mixed bone-soft transition faces are rendered magenta and excluded from collision",
            "source_geometry": "registered-template lip defects are intentionally absent from this exact-FEM contact-force view",
            "bone_coverage": "complete only for the selected labeled FEM face-subvolume boundary under the pure-face rule; not independently registered anatomical source-bone coverage",
            "pressure_limit": "force locations and nodal magnitudes are not pressure without validated area division",
        },
        "assets": assets,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    write_report(cfg.output_dir, summary)
    cherries.log_metrics(
        {
            "contact/runtime_active_count": diagnostics["active_contact_count"],
            "contact/preflight_active_count": validation["reference"][
                "active_contact_count"
            ],
            "contact/minimum_active_gap_mm": diagnostics["minimum_active_distance_m"]
            * 1000,
            "contact/cranium_resultant_N": bone_force["cranium"][
                "resultant_magnitude_N"
            ],
            "contact/mandible_resultant_N": bone_force["mandible"][
                "resultant_magnitude_N"
            ],
        }
    )
    LOG.info("Wrote %d contact visuals to %s", len(assets), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
