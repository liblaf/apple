"""Independently audit a saved clearance-repaired constitutive reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
NEUTRAL = ROOT / "exp/2026/09/22/neutral-newton"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(HERE), str(SOLVERS), str(JOINT / "src"), str(NEUTRAL / "src")]

from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import configure_cuda  # noqa: E402
from smile_collision import (  # noqa: E402
    audit_collision_state,
    audit_required_collision,
)


class Config(cherries.BaseConfig):
    reference_dir: Path = GROUP / "data/reference-clearance-001"
    screen_factor: float = 2.0


def _record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _input(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    assert path.is_file(), f"{label} is missing: {path}"
    assert sha256(path) == record["sha256"], f"{label} SHA-256 mismatch: {path}"
    return path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mesh_connectivity(mesh: pv.DataSet) -> dict[str, str]:
    if isinstance(mesh, pv.UnstructuredGrid):
        cells = np.asarray(mesh.cells, dtype=np.int64)
        celltypes = np.asarray(mesh.celltypes, dtype=np.uint8)
        return {
            "kind": "unstructured-grid",
            "cells_sha256": sha256_bytes(cells.tobytes()),
            "celltypes_sha256": sha256_bytes(celltypes.tobytes()),
        }
    if isinstance(mesh, pv.PolyData):
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return {"kind": "poly-data", "faces_sha256": sha256_bytes(faces.tobytes())}
    raise TypeError(type(mesh))


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _tetrahedra(mesh: pv.DataSet) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    return cells[:, 1:]


def _original_constitutive_meshes(repair: dict[str, Any]) -> tuple[Path, Path]:
    """Follow the immutable frozen manifest chain to the pre-repair meshes."""
    manifest = _json(_input(repair["inputs"]["neutral_manifest"], "neutral manifest"))
    protocol = _json(_input(manifest["sources"]["protocol"], "frozen protocol"))
    inputs = protocol["inputs"]
    _input(
        {"path": inputs["prepared_npz"], "sha256": inputs["prepared_npz_sha256"]},
        "prepared inputs",
    )
    prepared = _json(
        _input(
            {
                "path": inputs["prepared_manifest"],
                "sha256": inputs["prepared_manifest_sha256"],
            },
            "prepared manifest",
        )
    )
    return (
        _input(prepared["sources"]["volume"], "original constitutive volume"),
        _input(prepared["sources"]["skin"], "original constitutive skin"),
    )


def _tet_metrics(
    original: np.ndarray, repaired: np.ndarray, tets: np.ndarray
) -> dict[str, Any]:
    dm = np.transpose(original[tets[:, 1:]] - original[tets[:, :1]], (0, 2, 1))
    ds = np.transpose(repaired[tets[:, 1:]] - repaired[tets[:, :1]], (0, 2, 1))
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    assert np.isfinite(detf).all()
    return {
        "method": "float64 det(repaired edge matrix) / det(original edge matrix)",
        "detF_min": float(detf.min()),
        "detF_p001": float(np.quantile(detf, 0.001)),
        "detF_max": float(detf.max()),
        "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
    }


def _strict_ipc_audit(physics: Any, target: float, screen: float) -> dict[str, Any]:
    """Run a fresh IPC query on rebuilt repaired geometry at a larger radius."""
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    original = collision.potential
    old_minimum = collision.min_distance
    old_inflation = collision.inflation_radius
    stiffness = float(physics.contact_definition["config"]["stiffness_mpa"])
    assert stiffness > 0
    collision.potential = ipctk.BarrierPotential(
        type(original.barrier)(), screen, stiffness, collision.use_physical_barrier
    )
    collision.min_distance = target
    collision.inflation_radius = screen
    try:
        pose = torch.zeros(
            6,
            device=model.dof_map.fixed_values.device,
            dtype=model.dof_map.fixed_values.dtype,
        )
        zero = torch.zeros_like(physics.points_t)
        state = audit_collision_state(physics, zero, pose)
        full = physics.full_skull.extend_seed(zero, pose)
        diagnostic = collision.diagnostics(collision.state_at(full), full)
    finally:
        collision.potential = original
        collision.min_distance = old_minimum
        collision.inflation_radius = old_inflation
    active = diagnostic["minimum_active_distance_m"]
    lower_bound = screen if active is None else float(active)
    return {
        "method": "independent rebuilt IPC collision query at enlarged activation radius",
        "target_dhat_m": target,
        "screen_radius_m": screen,
        "screen_stiffness_mpa": stiffness,
        "active_contact_count": int(diagnostic["active_contact_count"]),
        "active_minimum_distance_m": active,
        "minimum_distance_lower_bound_m": lower_bound,
        "intersections_zero": bool(state["soft_rigid_intersection_free"]),
        "contact_numerically_valid": bool(diagnostic["contact_numerically_valid"]),
        "meets_target": bool(
            state["soft_rigid_intersection_free"]
            and diagnostic["contact_numerically_valid"]
            and lower_bound >= target
        ),
    }


def main(cfg: Config) -> None:
    reference_dir = cfg.reference_dir.resolve()
    output = reference_dir / "independent-audit.json"
    assert not output.exists(), output
    assert cfg.screen_factor > 1
    repair = _json(reference_dir / "receipt.json")
    assert repair["schema"] == "reference-clearance-repair-v2"
    assert repair["success"]
    target = float(repair["requested_dhat_m"])
    assert target >= 1e-4
    archive = _input(repair["archive"], "repair archive")
    volume_path = _input(repair["meshes"]["volume"], "repaired volume")
    skin_path = _input(repair["meshes"]["skin"], "repaired skin")
    with np.load(archive, allow_pickle=False) as npz:
        original_points = np.asarray(npz["reference_points_m"], dtype=np.float64)
        repaired_points = np.asarray(npz["repaired_points_m"], dtype=np.float64)
        displacement = np.asarray(npz["displacement_m"], dtype=np.float64)
    assert repaired_points.shape == original_points.shape == displacement.shape
    assert np.array_equal(repaired_points, original_points + displacement)
    repaired_volume = pv.read(volume_path)
    repaired_skin = pv.read(skin_path)
    assert np.array_equal(np.asarray(repaired_volume.points), repaired_points)

    # This reconstructs a fresh FEM/IPC model from the saved repair, without a
    # forward solve or a saved displacement state.
    from reference_rebase import build_rebased_physics

    configure_cuda()
    physics, _ = build_rebased_physics(reference_dir)
    assert np.array_equal(np.asarray(physics.points), repaired_points)
    original_volume_path, original_skin_path = _original_constitutive_meshes(repair)
    original_volume = pv.read(original_volume_path)
    original_skin = pv.read(original_skin_path)
    assert np.array_equal(np.asarray(original_volume.points), original_points)
    assert _mesh_connectivity(original_volume) == _mesh_connectivity(repaired_volume)
    assert _mesh_connectivity(original_skin) == _mesh_connectivity(repaired_skin)
    for name in ("GlobalPointId",):
        assert np.array_equal(
            original_skin.point_data[name], repaired_skin.point_data[name]
        )
    assert np.array_equal(
        np.asarray(repaired_skin.points),
        repaired_points[
            np.asarray(repaired_skin.point_data["GlobalPointId"], dtype=np.int64)
        ],
    )
    fixed = np.asarray(physics.full_skull.geometry.fixed_global_ids, dtype=np.int64)
    assert np.array_equal(displacement[fixed], np.zeros((len(fixed), 3)))
    tet = _tet_metrics(original_points, repaired_points, _tetrahedra(original_volume))
    assert tet["inverted_tetrahedra"] == 0
    coverage = audit_required_collision(physics)
    assert coverage["coverage"] == {
        "complete_source_bones": True,
        "excluded_source_triangles": 0,
        "soft_cranium": True,
        "soft_mandible": True,
        "soft_eyes": True,
        "soft_soft": False,
        "rigid_rigid": False,
    }
    ipc = _strict_ipc_audit(physics, target, target * cfg.screen_factor)
    assert ipc["meets_target"], ipc
    receipt = {
        "schema": "reference-clearance-independent-audit-v1",
        "scope": "Saved repaired reference audit; no forward equilibrium solve.",
        "repair": _record(reference_dir / "receipt.json"),
        "input_archive": _record(archive),
        "constitutive_volume": _record(volume_path),
        "constitutive_skin": _record(skin_path),
        "original_constitutive": {
            "volume": _record(original_volume_path),
            "skin": _record(original_skin_path),
        },
        "connectivity_preserved": True,
        "fixed_coordinates_unchanged": True,
        "rigid_geometry": {
            "full_skull": physics.base.full_skull.geometry.binding_receipt(),
            "eyes": physics.eyes.binding_receipt(),
            "source_coordinates_changed": False,
        },
        "tetrahedra": tet,
        "collision_coverage": coverage,
        "independent_ipc": ipc,
        "rebuilt_from_saved_reference": True,
    }
    write_json(output, receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "audit/reference_detF_min": tet["detF_min"],
            "audit/reference_inversions": tet["inverted_tetrahedra"],
            "audit/reference_clearance_m": ipc["minimum_distance_lower_bound_m"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
