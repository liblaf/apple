"""Search small rigid mandible poses that separate unchanged source bones."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry, _rotation_matrix

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    bone_pair_audit: Path = GROUP / "data/full-skull-bone-bone-audit-001/summary.json"
    prepared_dir: Path = GROUP / "data/prepared"
    maximum_translation_m: float = 0.002
    translation_scan_step_m: float = 5e-5
    bisection_steps: int = 20
    clearance_target_m: float = 1e-5
    output_dir: Path = cherries.output("full-skull-rigid-separation", mkdir=True)


def _geometry_module() -> Any:
    path = Path(__file__).with_name("17-audit-source-bone-contact.py")
    spec = importlib.util.spec_from_file_location("source_bone_geometry", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    return pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))


def _transform(
    points: np.ndarray, pivot: np.ndarray, pose_rad_m: np.ndarray
) -> np.ndarray:
    return (
        (points - pivot) @ _rotation_matrix(pose_rad_m[:3]).T + pivot + pose_rad_m[3:]
    )


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    assert cfg.maximum_translation_m == 0.002
    assert cfg.translation_scan_step_m == 5e-5
    assert cfg.bisection_steps == 20
    assert cfg.clearance_target_m == 1e-5
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)
    audit = json.loads(cfg.geometry_audit.read_text())
    pair_audit = json.loads(cfg.bone_pair_audit.read_text())
    assert pair_audit["schema"] == "joint-full-skull-bone-pair-audit-v1"
    assert pair_audit["raw_intersection_pairs"] == 128
    assert pair_audit["ipc_cross_bone_intersection"] is True
    geometry_path = Path(audit["geometry"]["path"])
    assert sha256(geometry_path) == audit["geometry"]["sha256"]
    assert pair_audit["geometry_sha256"] == sha256(geometry_path)
    with np.load(geometry_path) as archive:
        a = {key: archive[key] for key in archive.files}
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    assert np.array_equal(prepared.arrays["mandible_pivot_m"], a["mandible_pivot_m"])

    cranium = np.asarray(a["cranium_points_m"], dtype=np.float64)
    mandible = np.asarray(a["mandible_points_m"], dtype=np.float64)
    cranium_faces = np.asarray(a["cranium_faces"], dtype=np.int32)
    mandible_faces = np.asarray(a["mandible_faces"], dtype=np.int32)
    pivot = np.asarray(a["mandible_pivot_m"], dtype=np.float64)
    nc = len(cranium)
    faces = np.concatenate((cranium_faces, mandible_faces + nc))
    reference = np.concatenate((cranium, mandible))
    mesh = ipctk.CollisionMesh(reference, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.concatenate(
            (
                np.zeros(nc, dtype=np.int32),
                np.ones(len(mandible), dtype=np.int32),
            )
        )
    )
    mesh.init_adjacencies()
    broad_phase = ipctk.LBVH()

    def positions(pose: np.ndarray) -> np.ndarray:
        return np.concatenate((cranium, _transform(mandible, pivot, pose)))

    def intersects(pose: np.ndarray) -> bool:
        return bool(ipctk.has_intersections(mesh, positions(pose), broad_phase))

    zero = np.zeros(6)
    assert intersects(zero)

    def translation_threshold(direction: np.ndarray) -> float | None:
        direction = direction / np.linalg.norm(direction)
        magnitudes = np.arange(
            0,
            cfg.maximum_translation_m + cfg.translation_scan_step_m / 2,
            cfg.translation_scan_step_m,
        )
        flags = [
            intersects(np.concatenate((np.zeros(3), magnitude * direction)))
            for magnitude in magnitudes
        ]
        first = next((index for index, flag in enumerate(flags) if not flag), None)
        if first is None:
            return None
        # The sampled ray must remain separated after its first clear endpoint;
        # otherwise a single bisection interval would hide re-entry.
        assert not any(flags[first:])
        lower = float(magnitudes[first - 1])
        upper = float(magnitudes[first])
        for _ in range(cfg.bisection_steps):
            middle = (lower + upper) / 2
            pose = np.concatenate((np.zeros(3), middle * direction))
            if intersects(pose):
                lower = middle
            else:
                upper = middle
        return upper

    # Search normalized translations d=(a,-1,c). This is a numerical endpoint
    # search, not an anatomical jaw-motion model or a proof over all six DOFs.
    levels = ((np.linspace(-0.6, 0.6, 25), np.linspace(-0.6, 0.6, 25)),)
    rows = []
    best: dict[str, Any] | None = None
    for x_values, z_values in levels:
        for x_slope in x_values:
            for z_slope in z_values:
                direction = np.asarray([x_slope, -1.0, z_slope])
                direction /= np.linalg.norm(direction)
                threshold = translation_threshold(direction)
                if threshold is None:
                    continue
                row = {
                    "x_slope": float(x_slope),
                    "z_slope": float(z_slope),
                    "direction": direction.tolist(),
                    "pair_free_threshold_m": threshold,
                }
                rows.append(row)
                if best is None or threshold < best["pair_free_threshold_m"]:
                    best = row
    assert best is not None
    for spacing, radius in ((0.01, 0.05), (0.005, 0.03)):
        center_x = best["x_slope"]
        center_z = best["z_slope"]
        values = np.arange(-radius, radius + spacing / 2, spacing)
        for dx in values:
            for dz in values:
                x_slope = center_x + dx
                z_slope = center_z + dz
                direction = np.asarray([x_slope, -1.0, z_slope])
                direction /= np.linalg.norm(direction)
                threshold = translation_threshold(direction)
                if threshold is None:
                    continue
                row = {
                    "x_slope": float(x_slope),
                    "z_slope": float(z_slope),
                    "direction": direction.tolist(),
                    "pair_free_threshold_m": threshold,
                }
                rows.append(row)
                if threshold < best["pair_free_threshold_m"]:
                    best = row

    geometry = _geometry_module()
    cranium_mesh = _polydata(cranium, cranium_faces)

    def exact_metrics(pose: np.ndarray) -> dict[str, Any]:
        moved = _transform(mandible, pivot, pose)
        mandible_mesh = _polydata(moved, mandible_faces)
        pairs, _, _, lengths = _collision_geometry(cranium_mesh, mandible_mesh)
        cranium_signed = geometry.signed_clearance(cranium, mandible_mesh)
        mandible_signed = geometry.signed_clearance(moved, cranium_mesh)
        motion = np.linalg.norm(moved - mandible, axis=1)
        return {
            "pose_rad_m": pose.tolist(),
            "ipc_has_intersections": intersects(pose),
            "raw_intersection_pairs": len(pairs),
            "intersection_segment_length_sum_m": float(lengths.sum()),
            "cranium_against_mandible_minimum_signed_distance_m": float(
                cranium_signed.min()
            ),
            "mandible_against_cranium_minimum_signed_distance_m": float(
                mandible_signed.min()
            ),
            "mandible_maximum_motion_m": float(motion.max()),
            "mandible_rms_motion_m": float(np.sqrt(np.mean(motion**2))),
        }

    best_direction = np.asarray(best["direction"])
    pair_free_pose = np.concatenate(
        (np.zeros(3), best["pair_free_threshold_m"] * best_direction)
    )
    pair_free = exact_metrics(pair_free_pose)
    assert pair_free["ipc_has_intersections"] is False
    assert pair_free["raw_intersection_pairs"] == 0

    # Add an explicit 10 micrometre bidirectional vertex-clearance margin.
    lower = best["pair_free_threshold_m"]
    upper = lower + 0.0002
    target_internal = cfg.clearance_target_m + 1e-7

    def clearance_at(magnitude: float) -> float:
        pose = np.concatenate((np.zeros(3), magnitude * best_direction))
        metrics = exact_metrics(pose)
        assert metrics["ipc_has_intersections"] is False
        assert metrics["raw_intersection_pairs"] == 0
        return min(
            metrics["cranium_against_mandible_minimum_signed_distance_m"],
            metrics["mandible_against_cranium_minimum_signed_distance_m"],
        )

    assert clearance_at(upper) >= target_internal
    for _ in range(cfg.bisection_steps):
        middle = (lower + upper) / 2
        if clearance_at(middle) >= target_internal:
            upper = middle
        else:
            lower = middle
    clearance_pose = np.concatenate((np.zeros(3), upper * best_direction))
    clearance_candidate = exact_metrics(clearance_pose)
    minimum_clearance = min(
        clearance_candidate["cranium_against_mandible_minimum_signed_distance_m"],
        clearance_candidate["mandible_against_cranium_minimum_signed_distance_m"],
    )
    assert clearance_candidate["ipc_has_intersections"] is False
    assert clearance_candidate["raw_intersection_pairs"] == 0
    assert minimum_clearance >= cfg.clearance_target_m

    pure_y_threshold = translation_threshold(np.asarray([0.0, -1.0, 0.0]))
    assert pure_y_threshold is not None
    frame = np.asarray(prepared.arrays["mandible_frame_world"], dtype=np.float64)
    hinge_rows = []
    for degrees in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        pose = np.concatenate((np.deg2rad(degrees) * frame[:, 0], np.zeros(3)))
        hinge_rows.append({"degrees": degrees, **exact_metrics(pose)})
    assert all(row["raw_intersection_pairs"] > 0 for row in hinge_rows)

    np.savez_compressed(
        cfg.output_dir / "poses.npz",
        pair_free_pose_rad_m=pair_free_pose,
        clearance_candidate_pose_rad_m=clearance_pose,
        clearance_candidate_mandible_points_m=_transform(
            mandible, pivot, clearance_pose
        ),
    )
    result = {
        "schema": "joint-full-skull-rigid-separation-search-v1",
        "success": True,
        "status": "rigid_endpoint_candidate_found_not_anatomy_or_physics_admission",
        "geometry_sha256": sha256(geometry_path),
        "geometry_audit_sha256": sha256(cfg.geometry_audit),
        "bone_pair_audit_sha256": sha256(cfg.bone_pair_audit),
        "coordinates_or_topology_changed": False,
        "search": {
            "parameterization": "normalized pure translations d=(a,-1,c), zero rotation",
            "maximum_translation_m": cfg.maximum_translation_m,
            "coarse_slopes": "a,c in [-0.6,0.6], step 0.05",
            "local_refinements": [
                {"radius": 0.05, "step": 0.01},
                {"radius": 0.03, "step": 0.005},
            ],
            "sampled_directions": len(rows),
            "translation_ray_scan_step_m": cfg.translation_scan_step_m,
            "bisection_steps": cfg.bisection_steps,
            "minimality_scope": "smallest pair-free magnitude among sampled pure-translation rays; not a global six-DoF minimum",
        },
        "reference": exact_metrics(zero),
        "best_sampled_pair_free_endpoint": pair_free,
        "clearance_target_m": cfg.clearance_target_m,
        "clearance_candidate": clearance_candidate,
        "pure_negative_world_y_pair_free_threshold_m": pure_y_threshold,
        "landmark_hinge_samples": hinge_rows,
        "interpretation": {
            "candidate_scope": "complete unchanged source-bone endpoint geometry only",
            "soft_bone_initialization_reused": False,
            "soft_tissue_reinitialization_required": True,
            "equilibrium_required": True,
            "path_ccd_certified": False,
            "path_ccd_reason": "registered zero pose already intersects, so collision-free-start CCD is undefined",
            "anatomical_correction": False,
            "automatic_registration_change": False,
            "final_launch_ready": False,
        },
        "poses_npz": {
            "path": str((cfg.output_dir / "poses.npz").resolve()),
            "sha256": sha256(cfg.output_dir / "poses.npz"),
        },
        "implementation_sha256": {
            key: value
            for key, value in provenance["sources"].items()
            if key.startswith("experiment/")
            and Path(key).name
            in {
                "17-audit-source-bone-contact.py",
                "59-audit-full-skull-bone-pairs.py",
                "61-search-full-skull-rigid-separation.py",
                "joint_data.py",
            }
        },
    }
    write_json(cfg.output_dir / "summary.json", result)
    LOG.info("Full-skull rigid separation: %s", result)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
