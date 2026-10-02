"""CPU validation of bone-bone CCD and a declared jaw proposal-domain probe."""

from __future__ import annotations

import logging
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_rigid_bone_collision import RigidBoneCollision, linear_bone_step

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    output_dir: Path = GROUP / "data/rigid-bone-ccd-validation-001"


def synthetic_checks() -> dict:
    start = np.asarray(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [-1, -1, 1], [1, -1, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = ipctk.CollisionMesh(start, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    )
    mesh.init_adjacencies()
    stationary = linear_bone_step(mesh, start, start)
    safe = start.copy()
    safe[3:, 2] += 0.25
    safe_receipt = linear_bone_step(mesh, start, safe)
    crossing = start.copy()
    crossing[3:, 2] = -1
    crossing_receipt = linear_bone_step(mesh, start, crossing)
    assert stationary["numerically_admissible"]
    assert safe_receipt["numerically_admissible"]
    assert not crossing_receipt["end_intersects"]
    assert not crossing_receipt["numerically_admissible"]
    assert 0 < crossing_receipt["collision_free_fraction"] < 1
    invalid_start = start.copy()
    invalid_start[3:, 2] = [-1, 1, 1]
    rejected_reason = None
    try:
        linear_bone_step(mesh, invalid_start, safe)
    except AssertionError as error:
        rejected_reason = str(error)
    assert rejected_reason is not None
    assert "intersection-free start" in rejected_reason
    return {
        "stationary": stationary,
        "safe_motion": safe_receipt,
        "crossing_with_disjoint_endpoints": crossing_receipt,
        "intersecting_start_rejected": True,
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    synthetic = synthetic_checks()
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    volume = pv.read(prepared.volume_path)
    guard = RigidBoneCollision(volume, prepared.arrays["mandible_pivot_m"])
    zero = np.zeros(6, dtype=np.float64)
    seed_pose = zero.copy()
    seed_pose[0] = np.deg2rad(0.01)
    end_pose = zero.copy()
    end_pose[0] = np.deg2rad(0.02)
    displacement = np.zeros_like(guard.points)
    displacement[guard.indices] = guard.positions(seed_pose) - guard.vertices
    seeded = guard.from_displacement(displacement, end_pose)
    from_poses = guard.from_poses(seed_pose, end_pose)
    assert seeded == from_poses
    assert seeded["numerically_admissible"]
    invalid_displacement = displacement.copy()
    invalid_displacement[guard.indices[~guard.mandible], 0] = 0.001
    moving_cranium_rejected = False
    try:
        guard.from_displacement(invalid_displacement, end_pose)
    except AssertionError:
        moving_cranium_rejected = True
    assert moving_cranium_rejected
    scales = np.asarray([*np.deg2rad([10.0] * 3), *([0.005] * 3)])
    rows = [
        {"label": "zero", "pose_rad_m": zero.tolist(), **guard.from_poses(zero, zero)}
    ]
    for coordinate in range(6):
        for sign in (-1, 1):
            pose = zero.copy()
            pose[coordinate] = sign * scales[coordinate]
            row = {
                "label": f"coordinate_{coordinate}_{sign:+d}",
                "pose_rad_m": pose.tolist(),
                **guard.from_poses(zero, pose),
            }
            rows.append(row)
            LOG.info(
                "%s: collision-free fraction %.8g",
                row["label"],
                row["collision_free_fraction"],
            )
    summary = {
        "schema": "joint-rigid-bone-ccd-validation-v1",
        "success": True,
        "synthetic_checks": synthetic,
        "seed_adapter_checks": {
            "nonzero_seed_pose_rad_m": seed_pose.tolist(),
            "nonzero_end_pose_rad_m": end_pose.tolist(),
            "receipt": seeded,
            "pose_and_displacement_routes_identical": True,
            "moving_cranium_seed_rejected": True,
        },
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "sources": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_rigid_bone_collision.py"),
                Path(__file__).with_name("joint_data.py"),
            )
        },
        "mapping": guard.mapping,
        "reference": guard.reference_receipt,
        "probes": rows,
        "probe_count": len(rows),
        "accepted_probe_count": sum(row["numerically_admissible"] for row in rows),
        "scope": "CPU collision geometry only; no forward equilibrium or anatomical domain validation",
        "whole_pose_box_validated": False,
        "bone_bone_energy_added": False,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)
    LOG.info(
        "Validated linear CCD; %d/%d sampled bone motions admissible",
        summary["accepted_probe_count"],
        len(rows),
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
