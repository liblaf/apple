# ruff: noqa: PLR0915
"""Bind a clearance-repaired mesh as the constitutive FEM reference.

This rebuilds the complete FEM and IPC model from repaired coordinates.  It is
deliberately separate from a seed: at ``u = 0`` both bulk and membrane
potentials, the collision rest vertices, and the fixed-DOF map use the repaired
configuration.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pyvista as pv
import torch
from joint_common import sha256
from joint_data import PreparedInputs
from joint_frozen_neutral import load_script
from joint_rigid_eye_contact import RigidEyeJointPhysics, build_eye_collision_physics
from profile_input_binding import (
    CURRENT_OWNED_ADJOINT_SHA256,
    bind_frozen_neutral_load,
)

SCHEMA = "reference-configuration-rebase-v1"
_GEOMETRY_FIELDS = frozenset(
    {
        "dV",
        "dhdX",
        "metric_map",
        "rest_edge_01",
        "rest_edge_02",
        "rest_metric_sqrt_det",
    }
)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    assert path.is_file(), path
    return {"path": str(path), "sha256": sha256(path)}


def _array_sha256(value: np.ndarray) -> str:
    """Hash array layout and bytes so receipt checks do not depend on NumPy repr."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells)
    assert cells.size % 5 == 0
    packed = cells.reshape(-1, 5)
    assert np.all(packed[:, 0] == 4)
    return np.ascontiguousarray(packed[:, 1:], dtype=np.int64)


def _signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    edges = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    volume = np.linalg.det(edges) / 6.0
    assert np.isfinite(volume).all()
    return volume


def _material_identity(
    old: dict[str, dict[str, torch.Tensor]],
    new: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    assert old.keys() == new.keys()
    preserved: dict[str, list[str]] = {}
    rebuilt: dict[str, list[str]] = {}
    for name, old_fields in old.items():
        new_fields = new[name]
        assert old_fields.keys() == new_fields.keys(), name
        for field, value in old_fields.items():
            if field in _GEOMETRY_FIELDS:
                continue
            assert torch.equal(value, new_fields[field]), f"{name}.{field} changed"
        preserved[name] = sorted(old_fields.keys() - _GEOMETRY_FIELDS)
        rebuilt[name] = sorted(old_fields.keys() & _GEOMETRY_FIELDS)
    return {
        "moduli_activation_and_prescribed_fields_preserved_exactly": True,
        "preserved_fields": preserved,
        "rebuilt_geometry_fields": rebuilt,
    }


def _write_repaired_meshes(
    physics: RigidEyeJointPhysics, repaired: np.ndarray, output_dir: Path
) -> tuple[Path, Path, np.ndarray]:
    volume = physics.mesh.copy(deep=True)
    skin = physics.skin.copy(deep=True)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert ids.ndim == 1
    assert ids.size == skin.n_points
    volume.points = repaired
    skin.points = repaired[ids]
    assert np.array_equal(np.asarray(volume.points), repaired)
    assert np.array_equal(np.asarray(skin.points), repaired[ids])
    volume_path = output_dir / "rebased-reference-volume.vtu"
    skin_path = output_dir / "rebased-reference-skin.vtp"
    volume.save(volume_path)
    skin.save(skin_path)
    return volume_path, skin_path, ids


def _rebuilt_geometry_receipt(
    physics: Any, materials: dict[str, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    """Check material geometry arrays against the repaired mesh-derived fields."""
    bulk: dict[str, dict[str, float]] = {}
    for name in ("fat", "muscle", "aponeurosis"):
        fields = materials[name]
        assert {"dhdX", "dV"}.issubset(fields)
        dhdx = fields["dhdX"].detach().cpu().numpy()
        dv = fields["dV"].detach().cpu().numpy()
        assert np.isfinite(dhdx).all()
        assert np.isfinite(dv).all()
        fraction = np.asarray(physics.mesh.cell_data[name.title() + "Fraction"])
        expected_volume = physics.volumes * fraction
        assert np.allclose(dv.sum(axis=1), expected_volume, rtol=0.0, atol=1e-18)
        bulk[name] = {
            "dhdX_finite": True,
            "dV_sum_matches_fraction_weighted_rebased_tet_volume": True,
        }
    skin = materials["skin"]
    required = {
        "metric_map",
        "rest_edge_01",
        "rest_edge_02",
        "rest_metric_sqrt_det",
    }
    assert required.issubset(skin)
    for name in required:
        assert np.isfinite(skin[name].detach().cpu().numpy()).all(), name
    rest_area = skin["rest_metric_sqrt_det"].detach().cpu().numpy() / 2.0
    assert np.allclose(rest_area, physics.skin_area, rtol=0.0, atol=1e-18)
    return {
        "bulk": bulk,
        "skin": {
            "metric_and_rest_edges_finite": True,
            "rest_metric_area_matches_rebased_skin_area": True,
        },
    }


def _zero_contact_receipt(physics: RigidEyeJointPhysics) -> dict[str, Any]:
    """Prove that the rebuilt reference has no active IPC force at ``u=0``."""
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    pose = torch.zeros(
        6,
        device=model.dof_map.fixed_values.device,
        dtype=model.dof_map.fixed_values.dtype,
    )
    fixed_values = physics.boundary(pose).detach().clone()
    model.dof_map.fixed_values = fixed_values
    state = model.init()
    assert torch.count_nonzero(model.dof_map.to_free(state.u)) == 0
    assert torch.equal(state.u.flatten()[model.dof_map.fixed_indices], fixed_values)
    roundoff_limit = float(
        8
        * np.finfo(np.float64).eps
        * np.abs(physics.full_skull.full_reference_points_m).max()
    )
    maximum_displacement = float(state.u.abs().max())
    assert maximum_displacement <= roundoff_limit
    state.collision = collision.state_at(state.u)
    diagnostics = collision.diagnostics(state.collision, state.u)
    gradient = torch.zeros_like(state.u)
    collision.grad(state.collision, state.u, gradient)
    force_norm = float(torch.linalg.vector_norm(gradient))
    assert diagnostics["active_contact_count"] == 0, diagnostics
    assert diagnostics["barrier_energy"] == 0.0, diagnostics
    assert force_norm == 0.0
    return {
        "free_displacement_zero": True,
        "maximum_zero_pose_displacement_m": maximum_displacement,
        "zero_pose_roundoff_limit_m": roundoff_limit,
        "fixed_dof_values_match_zero_pose_boundary": True,
        "maximum_fixed_dof_displacement_m": float(fixed_values.abs().max()),
        "active_contact_count": diagnostics["active_contact_count"],
        "barrier_energy_mpa_m3": diagnostics["barrier_energy"],
        "contact_force_norm_mpa_m2": force_norm,
        "contact_force_norm_n": force_norm * 1e6,
        "dhat_m": diagnostics["dhat_m"],
    }


def rebase_reference(
    physics: RigidEyeJointPhysics,
    baseline: dict[str, dict[str, torch.Tensor]],
    reference_dir: Path,
    output_dir: Path,
) -> tuple[RigidEyeJointPhysics, dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Rebuild ``physics`` with the repaired coordinates as its actual reference.

    ``reference_dir`` is the output of ``50-repair-reference.py``.  The repair
    archive is intentionally a strict coordinate contract, never a displacement
    seed for the old constitutive model.
    """
    reference_dir = reference_dir.resolve()
    output_dir = output_dir.resolve()
    repair_path = reference_dir / "reference-clearance.npz"
    repair_receipt_path = reference_dir / "receipt.json"
    repair_receipt = _record(repair_receipt_path)
    repair_info = json.loads(repair_receipt_path.read_text())
    assert repair_info["schema"] == "reference-clearance-repair-v2"
    assert repair_info["success"] is True
    assert Path(repair_info["archive"]["path"]).resolve() == repair_path
    assert repair_info["archive"]["sha256"] == sha256(repair_path)

    with np.load(repair_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "reference_points_m",
            "repaired_points_m",
            "displacement_m",
        }
        reference = np.asarray(archive["reference_points_m"])
        repaired = np.asarray(archive["repaired_points_m"])
        displacement = np.asarray(archive["displacement_m"])
    for name, value in {
        "reference_points_m": reference,
        "repaired_points_m": repaired,
        "displacement_m": displacement,
    }.items():
        assert value.dtype == np.float64, name
        assert value.ndim == 2, name
        assert value.shape[1] == 3, name
        assert np.isfinite(value).all(), name
    reference = np.ascontiguousarray(reference)
    repaired = np.ascontiguousarray(repaired)
    displacement = np.ascontiguousarray(displacement)
    assert (
        reference.shape == repaired.shape == displacement.shape == physics.points.shape
    )
    assert np.array_equal(repaired - reference, displacement)
    assert np.array_equal(reference, np.asarray(physics.points, dtype=np.float64))

    fixed = np.asarray(physics.full_skull.geometry.fixed_global_ids, dtype=np.int64)
    assert fixed.ndim == 1
    assert fixed.size
    assert np.array_equal(repaired[fixed], reference[fixed])
    volume_path, skin_path, skin_ids = _write_repaired_meshes(
        physics, repaired, output_dir
    )
    original_volume = physics.mesh
    tets = _tetrahedra(original_volume)
    repaired_mesh = pv.read(volume_path)
    assert np.array_equal(_tetrahedra(repaired_mesh), tets)
    original_volumes = _signed_volumes(reference, tets)
    repaired_volumes = _signed_volumes(repaired, tets)
    assert np.all(original_volumes > 0)
    assert np.all(repaired_volumes > 0)

    # Recreate the runner's original constructor with only the FEM/skin files
    # and its audited FEM reference replaced.  The complete source bones and
    # rigid eyes are retained from their existing hash-bound objects.
    neutral = physics.neutral
    protocol = json.loads(
        Path(neutral.manifest["sources"]["protocol"]["path"]).read_text()
    )
    inputs = protocol["inputs"]
    prepared = PreparedInputs.load(
        Path(inputs["prepared_npz"]), Path(inputs["prepared_manifest"])
    )
    runner = load_script("68-run-simple-skin-forward.py")
    geometry = attrs.evolve(
        physics.base.full_skull.geometry,
        fem_reference_points_m=repaired.copy(),
    )
    canonical = runner.research_informed_material_config()["materials"]
    mechanics = protocol["mechanics"]
    runtime_arrays = dict(neutral.arrays)
    runtime_arrays["target_displacement_m"] = neutral.arrays[
        "target_total_displacement_m"
    ]
    rebased_base = runner.FullSkullJointPhysics(
        volume_path,
        skin_path,
        runtime_arrays,
        bulk_young_mpa={
            name: canonical[name]["young_mpa"] for name in runner.BULK_TISSUES
        },
        bulk_nu={
            name: mechanics["poisson_ratios"][name] for name in runner.BULK_TISSUES
        },
        skin_young_mpa=canonical["skin"]["reference_map"]["young_mpa"],
        skin_nu=0.49,
        thickness_m=canonical["skin"]["thickness_m"],
        full_skull_geometry=geometry,
        full_skull_admission=physics.base.full_skull_admission,
        full_skull_contact_config=physics.base.contact_definition["config"],
        rtol=0.0,
        atol=neutral.manifest["force_threshold_code"],
        max_steps=protocol["solver"]["max_steps"],
        forward_method="pncg",
        adjoint_rtol=1e-7,
    )
    skin_fields, _ = runner.load_skin_field(
        Path(inputs["skin_field_path"]),
        Path(inputs["skin_field_manifest_path"]),
        prepared=prepared,
        expected_triangles=rebased_base.skin_tri,
    )
    rebased_baseline = runner.heterogeneous_materials(rebased_base, skin_fields)
    rebased_base.runtime.forward.model.set_materials(rebased_baseline)
    rebased = RigidEyeJointPhysics(
        rebased_base, physics.eyes, rebased_baseline, neutral=neutral
    )
    material_receipt = _material_identity(baseline, rebased_baseline)
    assert np.array_equal(rebased.points, repaired)
    assert np.array_equal(rebased.full_skull.geometry.fem_reference_points_m, repaired)
    assert np.array_equal(rebased.skin.points, repaired[skin_ids])
    assert np.array_equal(rebased.full_skull.eyes.points_m, physics.eyes.points_m)
    assert np.isfinite(rebased_base.dm_inv).all()
    assert np.isfinite(rebased_base.volumes).all()
    assert np.all(rebased_base.volumes > 0)
    assert np.isfinite(rebased_base.skin_area).all()
    assert np.all(rebased_base.skin_area > 0)
    geometry_receipt = _rebuilt_geometry_receipt(rebased_base, rebased_baseline)

    zero_state = _zero_contact_receipt(rebased)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "constitutive_volume": _record(volume_path),
        "constitutive_skin": _record(skin_path),
        "repair": _record(repair_path),
        "repair_receipt": repair_receipt,
        "original_constitutive": {
            "volume": _record(prepared.volume_path),
            "skin": _record(prepared.skin_path),
            "reference_points_sha256": _array_sha256(reference),
        },
        "coordinate_contract": {
            "definition": "repaired_points_m is the constitutive FEM and IPC reference at u=0",
            "repaired_points_sha256": _array_sha256(repaired),
            "displacement_sha256": _array_sha256(displacement),
            "reference_plus_displacement_exact": True,
            "fem_reference_rebased": True,
        },
        "connectivity": {
            "tetrahedra_sha256": _array_sha256(tets),
            "skin_global_point_ids_sha256": _array_sha256(skin_ids),
            "skin_faces_sha256": _array_sha256(
                np.asarray(physics.skin.faces).reshape(-1, 4)[:, 1:]
            ),
            "volume_cells_unchanged": True,
            "skin_faces_unchanged": True,
        },
        "fixed_nodes": {
            "global_ids_sha256": _array_sha256(fixed),
            "count": int(fixed.size),
            "maximum_coordinate_change_m": float(np.abs(displacement[fixed]).max()),
            "coordinates_unchanged": True,
        },
        "tetrahedra": {
            "original_min_signed_volume_m3": float(original_volumes.min()),
            "rebased_min_signed_volume_m3": float(repaired_volumes.min()),
            "rebased_max_signed_volume_m3": float(repaired_volumes.max()),
            "positive_reference_volumes": True,
        },
        "skin": {
            "rest_tangent_metrics_rebuilt": True,
            "rest_vertex_coordinates_equal_repaired_volume": True,
            **geometry_receipt["skin"],
        },
        "bulk": {
            "dhdX_and_dV_rebuilt": True,
            "per_material": geometry_receipt["bulk"],
        },
        "contact": {
            "reference_vertices_rebuilt": True,
            "soft_rigid_pair_filter_preserved": True,
            "definition": rebased.contact_definition,
        },
        "rigid_geometry": {
            "full_skull": rebased.base.full_skull.geometry.binding_receipt(),
            "eyes": rebased.eyes.binding_receipt(),
            "source_coordinates_changed": False,
        },
        "materials": material_receipt,
        "zero_state": zero_state,
    }
    receipt_path = output_dir / "reference-rebase.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # The protocol holds this same mapping inline; retain the file record for
    # reviewers without adding a fifth required protocol field.
    assert _record(receipt_path)["sha256"] == sha256(receipt_path)
    return rebased, rebased_baseline, receipt


def build_rebased_physics(
    reference_dir: Path,
    *,
    inverse: bool = False,
) -> tuple[RigidEyeJointPhysics, dict[str, dict[str, torch.Tensor]]]:
    """Rebuild the repaired configuration for an independent audit.

    The only temporary artifacts are private files deleted before this function
    returns.  The returned model is the same fully rebuilt FEM/contact model as
    :func:`rebase_reference`, with no saved solver state or displacement seed.

    Set ``inverse=True`` only for the custom joint-equilibrium implicit path.
    It binds the reviewed current ``_AdjointProblem`` rather than recording the
    public DifferentiableForward module as unused.
    """
    reference_dir = reference_dir.resolve()
    repair = json.loads((reference_dir / "receipt.json").read_text())
    assert repair["schema"] == "reference-clearance-repair-v2"
    neutral_dir = Path(repair["inputs"]["neutral_manifest"]["path"]).parent
    eyes_dir = Path(repair["inputs"]["eyes_manifest"]["path"]).parent
    with tempfile.TemporaryDirectory(prefix="new-neutral-rebase-audit-") as tmp:
        temp_dir = Path(tmp)
        with bind_frozen_neutral_load(
            neutral_dir,
            temp_dir,
            allow_pncg_curvature_clamps=True,
            allow_isfixed_boundary=True,
            **(
                {"current_owned_adjoint_sha256": CURRENT_OWNED_ADJOINT_SHA256}
                if inverse
                else {"unused_inverse_sha256": (CURRENT_OWNED_ADJOINT_SHA256)}
            ),
        ) as neutral:
            physics, baseline = build_eye_collision_physics(neutral, eyes_dir)
            rebased, rebased_baseline, _ = rebase_reference(
                physics, baseline, reference_dir, temp_dir
            )
    return rebased, rebased_baseline


__all__ = ["SCHEMA", "build_rebased_physics", "rebase_reference"]
