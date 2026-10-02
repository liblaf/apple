"""Independently audit the saved forward endpoint against complete source bones."""

from __future__ import annotations

import json
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
import torch
from joint_common import ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_full_skull_contact import build_full_skull_contact, load_full_skull_geometry

from liblaf import cherries


class Config(cherries.BaseConfig):
    run_dir: Path
    output_dir: Path


def main(cfg: Config) -> None:  # noqa: PLR0915 - keep the independent audit sequence together
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    protocol_path = cfg.run_dir / "protocol.json"
    summary_path = cfg.run_dir / "summary.json"
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(summary_path.read_text())
    checkpoint = result["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    assert sha256(checkpoint_path) == checkpoint["sha256"]
    inputs = protocol["inputs"]
    prepared = PreparedInputs.load(
        Path(inputs["prepared_npz"]), Path(inputs["prepared_manifest"])
    )
    geometry_receipt = inputs["geometry"]["geometry"]
    geometry = load_full_skull_geometry(
        Path(geometry_receipt["geometry_path"]),
        Path(geometry_receipt["audit_path"]),
    )
    assert geometry.geometry_sha256 == geometry_receipt["geometry_sha256"]
    assert protocol["mechanics"]["jaw_pose_rad_m"] == [0.0] * 6
    adapter = build_full_skull_contact(geometry, protocol["mechanics"]["contact"])
    with np.load(checkpoint_path) as archive:
        displacement = archive["displacement_m"]
    assert displacement.shape == (geometry.fem_node_count, 3)
    assert np.isfinite(displacement).all()
    fixed_error = float(np.abs(displacement[geometry.fixed_global_ids]).max())
    # A zero rigid pose evaluates (X - pivot) + pivot - X, so the solver's
    # prescribed neutral displacement can contain coordinate-scale roundoff.
    fixed_tolerance = float(
        8 * np.finfo(np.float64).eps * np.abs(geometry.fem_reference_points_m).max()
    )
    assert fixed_error <= fixed_tolerance
    u = adapter.extend_seed(torch.from_numpy(displacement), torch.zeros(6))
    collision = adapter.collision
    state = collision.state_at(u)
    positions = (collision.vertices + u[collision.indices]).numpy()
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
    )
    contact = collision.diagnostics(state, u)
    weights = np.array(
        [state.collisions[i].weight for i in range(len(state.collisions))]
    )
    mesh = pv.read(prepared.volume_path)
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    reference = np.asarray(mesh.points)
    current = reference + displacement
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    ds = current[cells[:, 1:]] - current[cells[:, :1]]
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    nu = protocol["mechanics"]["poisson_ratios"]
    assert all(nu[name] == 0.49 for name in ("fat", "aponeurosis", "muscle"))
    assert nu["skin_range"] == [0.49, 0.49]
    skin_path = Path(inputs["skin_field_path"])
    assert sha256(skin_path) == inputs["skin_field_sha256"]
    with np.load(skin_path) as skin:
        assert np.all(skin["nu"] == 0.49)
    receipt = {
        "schema": "joint-forward-terminal-independent-audit-v1",
        "success": not intersects and contact["contact_numerically_valid"],
        "run_dir": str(cfg.run_dir.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "summary_sha256": sha256(summary_path),
        "checkpoint": checkpoint,
        "poisson_ratios": nu,
        "fixed_displacement_max_m": fixed_error,
        "fixed_displacement_roundoff_tolerance_m": fixed_tolerance,
        "complete_source_triangles_retained": True,
        "soft_bone_intersections": intersects,
        "contact": contact,
        "negative_contact_weights": int(np.count_nonzero(weights < 0)),
        "tetrahedra": len(detf),
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
        "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
        "force_converged_in_runner": result["success"],
        "runner_exact_force_norm_N": result["final_free_force_norm"] * 1e6,
        "force_threshold_N": result["force_threshold"] * 1e6,
        "scope": "Independent CPU endpoint intersection, contact-energy, fixed-boundary, material-input and Jacobian audit. Force is independently recomputed by the runner, not recomputed here. Soft-soft and bone-bone intersections and bonded mixed transition faces are outside this contact selection. This is not a mechanical-stability proof.",
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir / "summary.json")
    cherries.log_metrics(
        {
            "soft_bone_intersections": int(intersects),
            "inverted_tetrahedra": receipt["inverted_tetrahedra"],
            "detF_min": receipt["detF_min"],
            "negative_contact_weights": receipt["negative_contact_weights"],
        }
    )
    assert receipt["success"]
    assert receipt["negative_contact_weights"] == 0


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint())
