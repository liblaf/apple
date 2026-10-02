"""Locally untangle the collision-free full-skull initialization candidate."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_contact import OwnedContact
from joint_data import PreparedInputs, _collision_geometry
from joint_full_skull_contact import (
    CONTACT_SCHEMA,
    build_full_skull_contact,
    load_full_skull_geometry,
)
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    candidate_summary: Path = (
        GROUP / "data/full-skull-initialization-candidate-001/summary.json"
    )
    volume_margin: float = 1e-4
    clearance_m: float = 1e-5
    clearance_neighborhood_m: float = 5e-4
    coordinate_scale_m: float = 1e-4
    surface_motion_weight: float = 100.0
    max_iterations: int = 1000
    output_dir: Path = cherries.output("full-skull-initialization-repair", mkdir=True)


def _geometry_module() -> Any:
    path = Path(__file__).with_name("17-audit-source-bone-contact.py")
    spec = importlib.util.spec_from_file_location("source_bone_geometry_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bone(arrays: Any, name: str) -> pv.PolyData:
    faces = np.asarray(arrays[f"{name}_faces"])
    bone = pv.PolyData(
        np.asarray(arrays[f"{name}_points_m"]),
        np.column_stack((np.full(len(faces), 3), faces)),
    )
    result = bone.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    assert np.array_equal(result.points, bone.points)
    assert np.array_equal(result.faces, bone.faces)
    return result


def _determinants(
    points: np.ndarray, reference: np.ndarray, tets: np.ndarray
) -> np.ndarray:
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    ds = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds) / np.linalg.det(dm)


def _ipc_diagnostics(contact: OwnedContact, displacement: np.ndarray) -> dict[str, Any]:
    value = torch.from_numpy(np.ascontiguousarray(displacement))
    state = contact.state_at(value)
    zero = torch.zeros_like(value)
    fraction = float(contact.max_step_size(state, value, zero))
    receipt = contact.diagnostics(state, value)
    assert fraction == 1.0
    assert receipt["contact_numerically_valid"] is True
    assert receipt["minimum_active_distance_m"] is None or (
        receipt["minimum_active_distance_m"] > 0
    )
    return {**receipt, "zero_increment_ccd_fraction": fraction}


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    assert 0 < cfg.volume_margin < 0.01
    assert cfg.clearance_m > 0
    assert cfg.clearance_neighborhood_m > cfg.clearance_m
    assert cfg.coordinate_scale_m > 0
    assert cfg.surface_motion_weight > 1
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)

    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    source = json.loads(cfg.candidate_summary.read_text())
    assert source["schema"] == "joint-full-skull-initialization-candidate-v1"
    assert source["geometric_candidate_passed"] is False
    candidate_path = Path(source["candidate"]["path"])
    assert sha256(candidate_path) == source["candidate"]["sha256"]
    audit = json.loads(cfg.geometry_audit.read_text())
    geometry_path = Path(audit["geometry"]["path"])
    assert sha256(geometry_path) == audit["geometry"]["sha256"]
    assert source["geometry_sha256"] == sha256(geometry_path)

    with np.load(geometry_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(candidate_path) as archive:
        source_displacement = np.asarray(
            archive["initial_displacement_m"], dtype=np.float64
        )
    reference = np.asarray(arrays["fem_reference_points_m"], dtype=np.float64)
    assert source_displacement.shape == reference.shape
    candidate = reference + source_displacement
    volume = pv.read(prepared.volume_path)
    assert np.array_equal(reference, np.asarray(volume.points))
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    det_dm = np.linalg.det(dm)
    assert np.all(det_dm > 0)

    source_j = _determinants(candidate, reference, tets)
    bad = np.flatnonzero((source_j < 0.25) | (source_j > 2.0))
    assert len(bad) == 57
    assert int(np.sum(source_j <= 0)) == 13
    fixed_mask = np.zeros(len(reference), dtype=bool)
    fixed_ids = np.asarray(arrays["fixed_global_ids"], dtype=np.int64)
    fixed_mask[fixed_ids] = True
    soft_mask = np.zeros(len(reference), dtype=bool)
    soft_ids = np.asarray(arrays["soft_global_ids"], dtype=np.int64)
    soft_mask[soft_ids] = True
    observation_ids = np.asarray(
        prepared.arrays["observation_node_ids"], dtype=np.int64
    )
    observation_mask = np.zeros(len(reference), dtype=bool)
    observation_mask[observation_ids] = True

    seed_nodes = np.unique(tets[bad])
    seed_mask = np.zeros(len(reference), dtype=bool)
    seed_mask[seed_nodes] = True
    seed_incident = np.flatnonzero(np.any(seed_mask[tets], axis=1))
    ring_nodes = np.unique(tets[seed_incident])
    # Release every nonfixed bad-cell vertex. Add only adjacent interior nodes;
    # unrelated soft-surface vertices remain exactly at candidate-001.
    movable = np.unique(
        np.concatenate(
            (
                seed_nodes[~fixed_mask[seed_nodes]],
                ring_nodes[~fixed_mask[ring_nodes] & ~soft_mask[ring_nodes]],
            )
        )
    )
    assert not np.any(fixed_mask[movable])
    assert not np.any(observation_mask[movable])
    variable_index = np.full(len(reference), -1, dtype=np.int64)
    variable_index[movable] = np.arange(len(movable))
    constrained_ids = np.flatnonzero(np.any(variable_index[tets] >= 0, axis=1))
    constrained_tets = tets[constrained_ids]
    constrained_det_dm = det_dm[constrained_ids]
    local_index = variable_index[constrained_tets]
    scale = cfg.coordinate_scale_m

    def local_positions(value: np.ndarray) -> np.ndarray:
        result = candidate[constrained_tets].copy()
        valid = local_index >= 0
        result[valid] += scale * value.reshape(-1, 3)[local_index[valid]]
        return result

    def determinant_constraints(value: np.ndarray) -> np.ndarray:
        points = local_positions(value)
        ds = np.transpose(points[:, 1:] - points[:, :1], (0, 2, 1))
        return np.linalg.det(ds) / constrained_det_dm

    def determinant_jacobian(value: np.ndarray) -> np.ndarray:
        points = local_positions(value)
        e1 = points[:, 1] - points[:, 0]
        e2 = points[:, 2] - points[:, 0]
        e3 = points[:, 3] - points[:, 0]
        gradient = np.empty((len(points), 4, 3))
        gradient[:, 1] = np.cross(e2, e3) / constrained_det_dm[:, None]
        gradient[:, 2] = np.cross(e3, e1) / constrained_det_dm[:, None]
        gradient[:, 3] = np.cross(e1, e2) / constrained_det_dm[:, None]
        gradient[:, 0] = -gradient[:, 1:].sum(axis=1)
        result = np.zeros((len(points), 3 * len(movable)))
        for corner in range(4):
            rows = np.flatnonzero(local_index[:, corner] >= 0)
            columns = local_index[rows, corner]
            for axis in range(3):
                result[rows, 3 * columns + axis] += scale * gradient[rows, corner, axis]
        return result

    bones = {name: _bone(arrays, name) for name in ("cranium", "mandible")}
    geometry = _geometry_module()
    initial_surface = pv.PolyData(
        candidate[soft_ids],
        np.column_stack((np.full(len(arrays["soft_faces"]), 3), arrays["soft_faces"])),
    )
    for bone in bones.values():
        pairs, _, _, _ = _collision_geometry(initial_surface, bone)
        assert not len(pairs)
        assert float(geometry.signed_clearance(candidate[soft_ids], bone).min()) > 0

    rows: list[np.ndarray] = []
    lower: list[float] = []
    movable_soft_local = np.flatnonzero(soft_mask[movable])
    movable_soft = movable[movable_soft_local]
    for bone in bones.values():
        _, closest = bone.find_closest_cell(
            candidate[movable_soft], return_closest_point=True
        )
        direction = candidate[movable_soft] - np.asarray(closest)
        distance = np.linalg.norm(direction, axis=1)
        near = distance < cfg.clearance_neighborhood_m
        assert np.all(distance[near] > 0)
        for local, delta, norm in zip(
            movable_soft_local[near], direction[near], distance[near], strict=True
        ):
            row = np.zeros(3 * len(movable))
            row[3 * local : 3 * local + 3] = delta / norm
            rows.append(row)
            lower.append((cfg.clearance_m - norm) / scale)
    plane_matrix = np.stack(rows)
    plane_lower = np.asarray(lower)
    assert float((-plane_lower).min()) >= -1e-10

    coordinate_weight = np.repeat(
        np.where(soft_mask[movable], cfg.surface_motion_weight, 1.0), 3
    )

    def objective(value: np.ndarray) -> float:
        return float(0.5 * np.dot(coordinate_weight * value, value))

    def objective_gradient(value: np.ndarray) -> np.ndarray:
        return coordinate_weight * value

    nonlinear = NonlinearConstraint(
        determinant_constraints,
        0.25 + cfg.volume_margin,
        2.0 - cfg.volume_margin,
        jac=determinant_jacobian,
    )
    clearance = LinearConstraint(plane_matrix, plane_lower, np.inf)
    result = minimize(
        objective,
        np.zeros(3 * len(movable)),
        jac=objective_gradient,
        constraints=(nonlinear, clearance),
        bounds=Bounds(-10.0, 10.0),
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": cfg.max_iterations, "disp": True},
    )
    assert result.success, result.message
    repaired = candidate.copy()
    repaired[movable] += scale * result.x.reshape(-1, 3)
    displacement = repaired - reference
    determinant = _determinants(repaired, reference, tets)
    assert float(determinant.min()) >= 0.25
    assert float(determinant.max()) <= 2.0
    assert not np.any(displacement[fixed_ids])
    assert np.array_equal(repaired[observation_ids], candidate[observation_ids])

    soft_faces = np.asarray(arrays["soft_faces"])
    repaired_surface = pv.PolyData(
        repaired[soft_ids],
        np.column_stack((np.full(len(soft_faces), 3), soft_faces)),
    )
    contacts = {}
    for name, bone in bones.items():
        pairs, _, _, _ = _collision_geometry(repaired_surface, bone)
        signed = geometry.signed_clearance(repaired[soft_ids], bone)
        contacts[name] = {
            "intersection_pairs": len(pairs),
            "minimum_node_signed_distance_m": float(signed.min()),
        }
        assert not len(pairs)
        assert float(signed.min()) > 0

    observation_weight = np.asarray(
        prepared.arrays["observation_weight_normalized"], dtype=np.float64
    )
    observation_rms = float(
        np.sqrt(
            np.sum(
                observation_weight * np.sum(displacement[observation_ids] ** 2, axis=1)
            )
        )
    )
    all_soft_rms = float(np.sqrt(np.mean(np.sum(displacement[soft_ids] ** 2, axis=1))))
    assert observation_rms <= 0.00025

    contact_config = {
        "schema": CONTACT_SCHEMA,
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 0.0001,
        "stiffness_mpa": 0.01,
    }
    bound_geometry = load_full_skull_geometry(geometry_path, cfg.geometry_audit)
    adapter = build_full_skull_contact(bound_geometry, contact_config)
    full_displacement = np.concatenate(
        (
            displacement,
            np.zeros(
                (
                    bound_geometry.cranium_node_count
                    + bound_geometry.mandible_node_count,
                    3,
                ),
                dtype=np.float64,
            ),
        )
    )
    ipc = _ipc_diagnostics(adapter.collision, full_displacement)

    candidate_output = cfg.output_dir / "candidate.npz"
    np.savez_compressed(candidate_output, initial_displacement_m=displacement)
    displacement_hash = (
        __import__("hashlib")
        .sha256(np.ascontiguousarray(displacement).tobytes())
        .hexdigest()
    )
    admission = {
        "schema": "joint-full-skull-contact-admission-v2",
        "success": True,
        "scope": "collision-and-volume-valid initialization only; not equilibrium or final-launch admission",
        "geometry_sha256": sha256(geometry_path),
        "geometry_audit_sha256": sha256(cfg.geometry_audit),
        "initialization_path": str(candidate_output.resolve()),
        "initialization_sha256": sha256(candidate_output),
        "initialization_displacement_sha256": displacement_hash,
        "initialization_array_key": "initial_displacement_m",
        "complete_source_triangles_retained": True,
        "source_coordinates_changed": False,
        "excluded_source_triangles": 0,
        "initialization_intersection_free": True,
        "fixed_nodes_unchanged": True,
        "detF_min": float(determinant.min()),
        "detF_max": float(determinant.max()),
        "observation_surface_rms_m": observation_rms,
        "ipc_initialization": ipc,
        "equilibrium_converged": False,
        "final_launch_ready": False,
    }
    write_json(cfg.output_dir / "admission.json", admission)
    summary = {
        "schema": "joint-full-skull-initialization-repair-v1",
        "success": True,
        "status": "initialization_geometry_admitted_not_equilibrium",
        "source_candidate_summary": str(cfg.candidate_summary.resolve()),
        "source_candidate_summary_sha256": sha256(cfg.candidate_summary),
        "source_candidate_sha256": sha256(candidate_path),
        "geometry_audit_sha256": sha256(cfg.geometry_audit),
        "geometry_sha256": sha256(geometry_path),
        "candidate": {
            "path": str(candidate_output.resolve()),
            "sha256": sha256(candidate_output),
            "initial_displacement_sha256": displacement_hash,
        },
        "admission": {
            "path": str((cfg.output_dir / "admission.json").resolve()),
            "sha256": sha256(cfg.output_dir / "admission.json"),
        },
        "method": {
            "name": "local constrained SLSQP volume projection",
            "coordinate_scale_m": scale,
            "surface_motion_weight": cfg.surface_motion_weight,
            "detF_internal_bounds": [
                0.25 + cfg.volume_margin,
                2.0 - cfg.volume_margin,
            ],
            "clearance_halfspace_m": cfg.clearance_m,
            "clearance_neighborhood_m": cfg.clearance_neighborhood_m,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit),
        },
        "classification": {
            "source_out_of_range_tetrahedra": len(bad),
            "source_inverted_tetrahedra": int(np.sum(source_j <= 0)),
            "seed_nodes": len(seed_nodes),
            "one_ring_nodes": len(ring_nodes),
            "movable_nodes": len(movable),
            "movable_soft_boundary_nodes": int(soft_mask[movable].sum()),
            "constrained_incident_tetrahedra": len(constrained_ids),
            "fixed_nodes_moved": False,
            "observation_nodes_moved_by_repair": False,
        },
        "metrics": {
            "detF_min": float(determinant.min()),
            "detF_max": float(determinant.max()),
            "inverted_tetrahedra": int(np.sum(determinant <= 0)),
            "outside_detF_0p25_2": int(
                np.sum((determinant < 0.25) | (determinant > 2.0))
            ),
            "maximum_repair_from_candidate_m": float(
                np.linalg.norm(repaired - candidate, axis=1).max()
            ),
            "observation_surface_rms_m": observation_rms,
            "all_soft_boundary_rms_m": all_soft_rms,
            "maximum_nodal_displacement_m": float(
                np.linalg.norm(displacement, axis=1).max()
            ),
            "contacts": contacts,
            "ipc": ipc,
        },
        "complete_source_triangles_retained": True,
        "source_coordinates_changed": False,
        "equilibrium_converged": False,
        "final_launch_ready": False,
        "implementation_sha256": {
            key: value
            for key, value in provenance["sources"].items()
            if key.startswith("experiment/")
            and Path(key).name
            in {
                "17-audit-source-bone-contact.py",
                "51-audit-full-skull-initialization.py",
                "52-prepare-full-skull-initialization.py",
                "54-repair-full-skull-initialization.py",
                "joint_full_skull_contact.py",
            }
        },
    }
    write_json(cfg.output_dir / "summary.json", summary)
    LOG.info("full-skull initialization repair: %s", summary["metrics"])
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
