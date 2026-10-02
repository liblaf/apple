"""Audit complete registered source bones against the FEM pure-soft boundary."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

from liblaf import cherries

LOG = logging.getLogger(__name__)
WINDOW = (1600, 1200)
BACKGROUND = "#f7f7f5"


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    checkpoint: Path = GROUP / "data/neutral-convergence-010-contact/terminal.pt"
    output_dir: Path = cherries.output("source-bone-contact-audit", mkdir=True)
    attachment_neighborhood_m: float = 0.001
    source_near_m: float = 0.001
    collider_far_m: float = 0.002


def load_stable_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"checkpoint changed during read: {path}"
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return value, hashlib.sha256(payload).hexdigest()


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
    scale = 1.10 * max((ymax - ymin) / 2, horizontal_span / (2 * aspect)) / zoom
    return {
        "position": [eye.tolist(), center.tolist(), (0.0, 1.0, 0.0)],
        "parallel_scale": scale,
    }


def plotter(title: str) -> pv.Plotter:
    result = pv.Plotter(off_screen=True, window_size=WINDOW)
    result.set_background(BACKGROUND)
    result.enable_anti_aliasing("ssaa")
    result.add_text(title, position="upper_left", font_size=15, color="#202124")
    return result


def save(plot: pv.Plotter, path: Path, settings: dict[str, Any]) -> None:
    plot.camera_position = settings["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = settings["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def expanded_bounds(points: np.ndarray) -> tuple[float, ...]:
    padding = np.asarray((0.006, 0.008, 0.006))
    lower, upper = points.min(axis=0) - padding, points.max(axis=0) + padding
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


def surface_distance(points: np.ndarray, surface: pv.PolyData) -> np.ndarray:
    _, closest = surface.find_closest_cell(points, return_closest_point=True)
    return np.linalg.norm(points - np.asarray(closest), axis=1)


def signed_clearance(points: np.ndarray, closed_surface: pv.PolyData) -> np.ndarray:
    assert closed_surface.is_manifold
    assert closed_surface.n_open_edges == 0
    distance = surface_distance(points, closed_surface)
    selector = vtkSelectEnclosedPoints()
    selector.SetInputData(pv.PolyData(points))
    selector.SetSurfaceData(closed_surface)
    selector.SetTolerance(0.0)
    selector.CheckSurfaceOn()
    selector.Update()
    inside = np.asarray(
        pv.wrap(selector.GetOutput()).point_data["SelectedPoints"], dtype=bool
    )
    return np.where(inside, -distance, distance)


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def boundary_partition(
    volume: pv.UnstructuredGrid, points: np.ndarray
) -> dict[str, pv.PolyData]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    global_ids = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.point_data["GlobalPointId"] = global_ids
    boundary.points = points[global_ids]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    local_faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    faces = global_ids[local_faces]
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
    return {
        name: boundary.extract_cells(np.flatnonzero(mask))
        .extract_surface(algorithm=None)
        .triangulate()
        for name, mask in masks.items()
    }


def source_tolerance(source: pv.PolyData, *points: np.ndarray) -> float:
    dtype = np.asarray(source.points).dtype
    assert np.issubdtype(dtype, np.floating)
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(source.points))),
        *(float(np.max(np.abs(value))) for value in points),
    )
    return max(1.0e-12, 32.0 * float(np.finfo(dtype).eps) * coordinate_scale)


def issue_vertices(
    surfaces: dict[str, pv.PolyData],
    source: pv.PolyData,
    source_near_m: float,
    collider_far_m: float,
    attachment_neighborhood_m: float,
    bone: str,
) -> tuple[np.ndarray, np.ndarray]:
    soft = surfaces["soft"]
    points = np.asarray(soft.points)
    issue = (
        (surface_distance(points, source) <= source_near_m)
        & (surface_distance(points, surfaces[bone]) > collider_far_m)
        & (surface_distance(points, surfaces["mixed"]) > attachment_neighborhood_m)
    )
    assert issue.any()
    global_ids = np.asarray(soft.point_data["GlobalPointId"], dtype=np.int64)
    return issue, global_ids[issue]


def collision_receipt(
    soft: pv.PolyData,
    mixed: pv.PolyData,
    source: pv.PolyData,
    issue_global_ids: np.ndarray,
    attachment_neighborhood_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs, segments, midpoints, lengths = _collision_geometry(soft, source)
    tolerance = source_tolerance(source, np.asarray(soft.points), segments)
    samples = np.concatenate((segments, midpoints[:, None, :]), axis=1)
    mixed_distance = surface_distance(samples.reshape(-1, 3), mixed).reshape(-1, 3)
    max_mixed_distance = mixed_distance.max(axis=1)
    coincidence = max_mixed_distance <= tolerance
    attachment_neighborhood = (~coincidence) & (
        max_mixed_distance <= attachment_neighborhood_m
    )
    free = max_mixed_distance > attachment_neighborhood_m
    soft_faces = np.asarray(soft.faces).reshape(-1, 4)[:, 1:]
    soft_global = np.asarray(soft.point_data["GlobalPointId"], dtype=np.int64)
    issue_face = np.any(np.isin(soft_global[soft_faces], issue_global_ids), axis=1)
    issue_pair = issue_face[pairs[:, 0]]
    result = {
        "raw_separate_mesh_intersection_pairs": len(pairs),
        "unique_soft_triangles": len(np.unique(pairs[:, 0])),
        "unique_source_bone_triangles": len(np.unique(pairs[:, 1])),
        "source_coordinate_dtype": str(np.asarray(source.points).dtype),
        "geometric_coincidence_tolerance_m": tolerance,
        "shared_geometric_attachment_coincidences": int(coincidence.sum()),
        "not_confined_to_bonded_geometry": int((~coincidence).sum()),
        "not_confined_but_within_1mm_attachment_neighborhood": int(
            attachment_neighborhood.sum()
        ),
        "farther_than_1mm_from_bonded_geometry": int(free.sum()),
        "pairs_incident_to_primary_issue_vertices": int(issue_pair.sum()),
        "issue_pairs_shared_geometric_coincidence": int(
            np.sum(issue_pair & coincidence)
        ),
        "issue_pairs_not_confined_to_bonded_geometry": int(
            np.sum(issue_pair & ~coincidence)
        ),
        "issue_pairs_farther_than_1mm_from_bonded_geometry": int(
            np.sum(issue_pair & free)
        ),
        "intersection_segment_length_m": quantiles(lengths),
        "intersection_segment_length_sum_m": float(lengths.sum()),
        "maximum_segment_sample_distance_to_bonded_mixed_surface_m": quantiles(
            max_mixed_distance
        ),
        "midpoint_bounds_m": [
            midpoints.min(axis=0).tolist(),
            midpoints.max(axis=0).tolist(),
        ],
        "interpretation": "the meshes have no shared topology; coincidence means the complete collision segment and midpoint are confined to the deformed FEM bonded mixed-face geometry within a source-dtype tolerance",
        "segment_length_limit": "intersection-segment length is a diagnostic proxy, not penetration depth or contact area",
    }
    render = {
        "pairs": pairs,
        "midpoints": midpoints,
        "coincidence": coincidence,
        "attachment_neighborhood": attachment_neighborhood,
        "free": free,
        "issue_pair": issue_pair,
    }
    return result, render


def compare_pairs(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, int]:
    reference_pairs = {tuple(value) for value in reference["pairs"].tolist()}
    current_pairs = {tuple(value) for value in current["pairs"].tolist()}
    return {
        "retained_pairs": len(reference_pairs & current_pairs),
        "new_pairs": len(current_pairs - reference_pairs),
        "disappeared_pairs": len(reference_pairs - current_pairs),
    }


def render_intersections(
    output: Path,
    state: str,
    bone: str,
    view: str,
    surfaces: dict[str, pv.PolyData],
    source: pv.PolyData,
    render: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, str]:
    midpoints = render["midpoints"]
    bounds = expanded_bounds(midpoints)
    name = f"{state}-{bone}-source-intersections-{view}.png"
    p = plotter(f"Source {bone} / FEM soft intersections — {state} — {view}")
    p.add_mesh(
        clipped(source, bounds),
        color="#3a7f54",
        opacity=0.28,
        style="wireframe",
        line_width=0.6,
    )
    p.add_mesh(clipped(surfaces["soft"], bounds), color="#ead0c7", opacity=0.14)
    p.add_mesh(
        clipped(surfaces["mixed"], bounds),
        color="#c62bc4",
        opacity=0.28,
        style="wireframe",
        line_width=0.7,
    )
    issue_points = midpoints[render["issue_pair"]]
    if len(issue_points):
        halo = pv.PolyData(issue_points).glyph(
            geom=pv.Sphere(radius=0.00032, theta_resolution=12, phi_resolution=12),
            scale=False,
            orient=False,
        )
        p.add_mesh(halo, color="#171717", opacity=0.65)
    for key, color, radius in (
        ("coincidence", "#1b9e77", 0.00018),
        ("attachment_neighborhood", "#e6ab02", 0.00020),
        ("free", "#d73027", 0.00022),
    ):
        points = midpoints[render[key]]
        if len(points):
            glyph = pv.PolyData(points).glyph(
                geom=pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12),
                scale=False,
                orient=False,
            )
            p.add_mesh(glyph, color=color)
    p.add_text(
        "green: bonded coincidence   amber: <=1 mm bonded neighborhood   red: >1 mm   black halo: 411/182 cohort",
        position="lower_left",
        font_size=9,
        color="#202124",
    )
    save(p, output / name, settings)
    return {
        "filename": name,
        "caption": f"{view.title()} localization of complete registered source-{bone}/pure-soft FEM triangle intersections in the {state} state. Green collision geometry is confined to bonded mixed faces within the float32 source-coordinate tolerance; amber is unconfined but within 1 mm of bonded geometry; red is farther than 1 mm. Black halos identify pairs incident to the fixed 411/182 source-near relevance cohort.",
    }


def write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Complete source-bone versus FEM soft-surface audit",
        "",
        "The independently registered source cranium and mandible are closed manifold "
        "surfaces in the same world frame as the FEM fixture. Signed clearance is "
        "negative inside the watertight source bone and positive outside.",
        "",
    ]
    for bone in ("cranium", "mandible"):
        lines.extend([f"## {bone.title()}", ""])
        for state in ("reference", "current_neutral"):
            record = summary["bones"][bone][state]["intersections"]
            clearance = summary["bones"][bone][state]["flagged_clearance_mm"]
            lines.extend(
                [
                    f"{state}: {record['raw_separate_mesh_intersection_pairs']} raw pairs; "
                    f"{record['shared_geometric_attachment_coincidences']} confined bonded "
                    f"coincidences; {record['not_confined_to_bonded_geometry']} unconfined; "
                    f"{record['farther_than_1mm_from_bonded_geometry']} farther than 1 mm "
                    f"from bonded geometry. The fixed relevance cohort has signed-clearance "
                    f"range {clearance['minimum']:.6g}--{clearance['maximum']:.6g} mm.",
                    "",
                ]
            )
    for asset in summary["assets"]:
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
            "## Admission conclusion",
            "",
            summary["admission"]["conclusion"],
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def main(cfg: Config) -> None:
    pv.OFF_SCREEN = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    reference_points = np.asarray(volume.points, dtype=np.float64)
    checkpoint, checkpoint_hash = load_stable_checkpoint(cfg.checkpoint)
    assert checkpoint["stage"] == "neutral"
    displacement = checkpoint["primal"]["neutral"].detach().cpu().numpy(force=True)
    assert displacement.dtype == np.float64
    assert displacement.shape == reference_points.shape
    assert np.isfinite(displacement).all()
    assert (
        np.linalg.norm(displacement[prepared.arrays["cranium_node_ids"]], axis=1).max()
        == 0.0
    )
    assert (
        np.linalg.norm(displacement[prepared.arrays["mandible_node_ids"]], axis=1).max()
        <= 2.0e-18
    )
    current_points = reference_points + displacement
    states = {
        "reference": boundary_partition(volume, reference_points),
        "current_neutral": boundary_partition(volume, current_points),
    }

    assets: list[dict[str, str]] = []
    bones: dict[str, Any] = {}
    for bone, source_key, view in (
        ("cranium", "cranium_surface", "front"),
        ("mandible", "mandible_surface", "side"),
    ):
        source_path = Path(prepared.manifest["sources"][source_key]["path"])
        source = pv.read(source_path).triangulate()
        assert source.is_manifold
        assert source.n_open_edges == 0
        reference_issue, issue_global_ids = issue_vertices(
            states["reference"],
            source,
            cfg.source_near_m,
            cfg.collider_far_m,
            cfg.attachment_neighborhood_m,
            bone,
        )
        reference_soft_global = np.asarray(
            states["reference"]["soft"].point_data["GlobalPointId"], dtype=np.int64
        )
        assert np.array_equal(reference_soft_global[reference_issue], issue_global_ids)
        state_receipts: dict[str, Any] = {}
        raw_for_comparison: dict[str, Any] = {}
        all_midpoints: list[np.ndarray] = []
        for state, surfaces in states.items():
            soft_global = np.asarray(
                surfaces["soft"].point_data["GlobalPointId"], dtype=np.int64
            )
            lookup = {int(value): index for index, value in enumerate(soft_global)}
            issue_local = np.asarray(
                [lookup[int(value)] for value in issue_global_ids], dtype=np.int64
            )
            clearance_m = signed_clearance(
                np.asarray(surfaces["soft"].points)[issue_local], source
            )
            intersections, render = collision_receipt(
                surfaces["soft"],
                surfaces["mixed"],
                source,
                issue_global_ids,
                cfg.attachment_neighborhood_m,
            )
            raw_for_comparison[state] = {"pairs": render["pairs"]}
            state_receipts[state] = {
                "flagged_vertices": len(issue_global_ids),
                "flagged_global_point_ids": issue_global_ids.tolist(),
                "flagged_clearance_mm": quantiles(1000.0 * clearance_m),
                "flagged_inside_source_bone": int(np.sum(clearance_m < 0.0)),
                "flagged_on_source_bone_within_tolerance": int(
                    np.sum(
                        np.abs(clearance_m)
                        <= source_tolerance(source, reference_points)
                    )
                ),
                "intersections": intersections,
            }
            all_midpoints.append(render["midpoints"])
            raw_for_comparison[state]["render"] = render
        pair_change = compare_pairs(
            raw_for_comparison["reference"], raw_for_comparison["current_neutral"]
        )
        focus = pv.PolyData(np.vstack(all_midpoints))
        settings = camera(focus, view)
        for state, surfaces in states.items():
            assets.append(
                render_intersections(
                    cfg.output_dir,
                    state,
                    bone,
                    view,
                    surfaces,
                    source,
                    raw_for_comparison[state]["render"],
                    settings,
                )
            )
        bones[bone] = {
            "source": {
                "path": str(source_path.resolve()),
                "sha256": sha256(source_path),
                "points": source.n_points,
                "triangles": source.n_cells,
                "manifold": bool(source.is_manifold),
                "open_edges": source.n_open_edges,
                "signed_clearance_convention": "negative inside the watertight source bone, positive outside; sign determined by closed-surface containment rather than input normal orientation",
            },
            **state_receipts,
            "pair_change": pair_change,
        }

    reference_free = sum(
        bones[bone]["reference"]["intersections"]["not_confined_to_bonded_geometry"]
        for bone in bones
    )
    current_free = sum(
        bones[bone]["current_neutral"]["intersections"][
            "not_confined_to_bonded_geometry"
        ]
        for bone in bones
    )
    summary = {
        "schema": "joint-source-bone-contact-audit-v1",
        "purpose": "read-only comparison of the pure-soft FEM boundary with complete registered source bones; active mechanics unchanged",
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "checkpoint": {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "update": checkpoint["update"],
            "neutral_converged": bool(checkpoint["neutral_converged"]),
        },
        "thresholds": {
            "source_near_m": cfg.source_near_m,
            "declared_fem_collider_far_m": cfg.collider_far_m,
            "bonded_attachment_neighborhood_m": cfg.attachment_neighborhood_m,
            "geometric_coincidence": "32 source-coordinate epsilons times world coordinate scale; evaluated on both segment endpoints and midpoint",
        },
        "bones": bones,
        "assets": assets,
        "admission": {
            "complete_source_bones_collision_free_at_reference": reference_free == 0,
            "complete_source_bones_collision_free_at_current_neutral": current_free
            == 0,
            "reference_unconfined_intersection_pairs": reference_free,
            "current_neutral_unconfined_intersection_pairs": current_free,
            "conclusion": "Complete source-bone surfaces cannot be added directly as collision obstacles: both the reference and current neutral pure-soft FEM surfaces have source-bone triangle intersections not confined to the bonded mixed-face geometry. A reviewed binding/exclusion map plus repair or collision-free offset/remeshing of remaining free patches is required before replacing or augmenting the frozen FEM colliders.",
            "modeling_limit": "the intersections demonstrate incompatibility of the complete source surfaces with immediate collision use; they do not by themselves prove that every overlap is an anatomical defect or that every source-near gap requires contact",
            "active_mechanics_changed": False,
        },
        "anatomical_validation": False,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    write_report(cfg.output_dir, summary)
    cherries.log_metrics(
        {
            "source_contact/reference_unconfined_pairs": reference_free,
            "source_contact/current_unconfined_pairs": current_free,
            "source_contact/assets": len(assets),
        }
    )
    LOG.info("Wrote source-bone contact audit to %s", cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
