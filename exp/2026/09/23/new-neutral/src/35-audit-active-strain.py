"""Independently audit the saved active-strain neutral endpoint on the CPU."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-active-strain-001"
    output_name: str = "independent-audit.json"


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


def _verify_archive_sources(run_dir: Path, receipt: dict[str, Any]) -> dict[str, int]:
    """Verify every saved source against both recorded archive manifests."""
    active = _json(run_dir / "active-strain-source-provenance.json")
    for relative, expected in active.items():
        archived = run_dir / "sources/new-neutral" / relative
        assert sha256(archived) == expected, archived

    provenance = _json(run_dir / "provenance.json")["sources"]
    for relative, expected in provenance.items():
        archived = run_dir / "sources" / relative
        assert sha256(archived) == expected, archived

    binding = _json(run_dir / "profile-input-binding.json")
    receipt["source_archives_verified"] = True
    drift = {}
    for relative, expected in active.items():
        current = HERE / relative
        if current.is_file():
            observed = sha256(current)
            if observed != expected:
                drift[relative] = {
                    "archived_execution_sha256": expected,
                    "current_worktree_sha256": observed,
                }
    receipt["current_new_neutral_source_drift"] = drift
    return {
        "active_strain": len(active),
        "runtime_archive": len(provenance),
        "runtime_binding": len(binding["runtime_changes"]),
    }


def _original_constitutive_volume(protocol: dict[str, Any]) -> Path:
    """Recover the original volume used by the zero-displacement cold seed."""
    frozen_manifest = _input(
        protocol["fixture"]["neutral_manifest"], "neutral manifest"
    )
    frozen = _json(frozen_manifest)
    source_protocol = _input(frozen["sources"]["protocol"], "frozen source protocol")
    original_protocol = _json(source_protocol)
    inputs = original_protocol["inputs"]
    _input(
        {"path": inputs["prepared_npz"], "sha256": inputs["prepared_npz_sha256"]},
        "prepared inputs",
    )
    prepared_manifest = _input(
        {
            "path": inputs["prepared_manifest"],
            "sha256": inputs["prepared_manifest_sha256"],
        },
        "prepared manifest",
    )
    return _input(_json(prepared_manifest)["sources"]["volume"], "constitutive volume")


def _constitutive_volume(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Select the declared reference while retaining the original input chain."""
    original = _original_constitutive_volume(protocol)
    configuration = protocol.get("reference_configuration")
    if configuration is None:
        assert protocol["fixture"]["fem_reference_rebased"] is False
        return original, {
            "fem_reference_rebased": False,
            "original_volume": _record(original),
        }
    assert protocol["fixture"]["fem_reference_rebased"] is True
    assert configuration["schema"] == "reference-configuration-rebase-v1"
    repaired = _input(
        configuration["constitutive_volume"], "repaired constitutive volume"
    )
    _input(configuration["constitutive_skin"], "repaired constitutive skin")
    _input(configuration["repair"], "reference repair archive")
    repair_receipt_path = _input(
        configuration["repair_receipt"], "reference repair receipt"
    )
    repair_receipt = _json(repair_receipt_path)
    original_record = configuration["original_constitutive"]["volume"]
    assert _input(original_record, "rebase original constitutive volume") == original
    return repaired, {
        "fem_reference_rebased": True,
        "original_volume": _record(original),
        "reference_configuration": {
            name: configuration[name]
            for name in (
                "schema",
                "constitutive_volume",
                "constitutive_skin",
                "repair",
                "repair_receipt",
            )
        },
        "repair_receipt": _record(repair_receipt_path),
        "repair_receipt_schema": repair_receipt["schema"],
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _isfixed_boundary(
    protocol: dict[str, Any],
    constitutive: Path,
    binding: dict[str, Any],
    runtime_boundary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify the saved rebuilt model binds exactly the source IsFixed IDs."""
    original = _original_constitutive_volume(protocol)
    original_mesh = pv.read(original)
    constitutive_mesh = pv.read(constitutive)
    original_mask = np.asarray(original_mesh.point_data["IsFixed"], dtype=bool)
    constitutive_mask = np.asarray(constitutive_mesh.point_data["IsFixed"], dtype=bool)
    assert original_mask.ndim == constitutive_mask.ndim == 1
    assert np.array_equal(original_mask, constitutive_mask)
    expected_ids = np.flatnonzero(original_mask).astype("<i8", copy=False)
    correction = binding.get("isfixed_boundary_correction")
    if correction is None:
        return {
            "isfixed_boundary_correction_enabled": False,
            "isfixed_equality_verified": False,
            "reason": "Historical run did not opt into the corrected IsFixed boundary policy.",
            "source_isfixed_count": int(expected_ids.size),
            "source_isfixed_ids_sha256": _array_sha256(expected_ids),
        }

    assert correction["enabled"] is True
    assert correction["physics_unchanged"] is False
    assert correction["source_diffs_recorded"] is True
    assert (
        correction["scope"]
        == "IsFixed is the sole FEM clamp; jaw = IsFixed intersect Mandible; "
        "runtime geometry fixed IDs are rebound; historical neutral equilibrium "
        "is invalidated."
    )
    configuration = protocol["reference_configuration"]
    fixed = configuration["fixed_nodes"]
    # These are the same original FEM IDs. Rebase hashes dtype, shape and
    # bytes; the runtime boundary receipt hashes bytes only.
    rebase_digest = hashlib.sha256()
    rebase_digest.update(expected_ids.dtype.str.encode())
    rebase_digest.update(np.asarray(expected_ids.shape, dtype=np.int64).tobytes())
    rebase_digest.update(expected_ids.tobytes())
    assert fixed["global_ids_sha256"] == rebase_digest.hexdigest()
    assert fixed["count"] == int(expected_ids.size)
    assert fixed["coordinates_unchanged"] is True
    assert float(fixed["maximum_coordinate_change_m"]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(constitutive_mesh.point_data["FixedMask"], dtype=bool),
        np.repeat(original_mask[:, None], 3, axis=1),
    )
    assert runtime_boundary is not None
    assert runtime_boundary["policy"] == (
        "Only IsFixed prescribes original FEM nodes; Mandible selects rigid motion "
        "within that set."
    )
    assert runtime_boundary["runtime_dofs_verified_against_isfixed"] is True
    assert runtime_boundary["runtime_fixed_count"] == int(expected_ids.size)
    assert runtime_boundary["runtime_fixed_ids_sha256"] == _array_sha256(expected_ids)
    assert runtime_boundary["historical_equilibrium_reused_as_converged"] is False
    return {
        "isfixed_boundary_correction_enabled": True,
        "isfixed_equality_verified": True,
        "source_isfixed_count": int(expected_ids.size),
        "source_isfixed_ids_sha256": _array_sha256(expected_ids),
        "repaired_constitutive_isfixed_equals_original": True,
        "saved_fixedmask_equals_isfixed": True,
        "rebase_fixed_id_layout_and_bytes_hash_verified": True,
        "rebased_reference_fixed_global_nodes": fixed,
        "runtime_dof_boundary": runtime_boundary,
        "jaw_policy": "IsFixed intersect Mandible",
        "historical_equilibrium_invalidated": True,
    }


def _detf(volume: Path, endpoint: Path) -> tuple[dict[str, Any], np.ndarray]:
    mesh = pv.read(volume)
    reference = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    with np.load(endpoint, allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    assert displacement.shape == reference.shape
    assert np.isfinite(displacement).all()
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    deformed = reference + displacement
    ds = deformed[cells[:, 1:]] - deformed[cells[:, :1]]
    determinant = np.linalg.det(ds) / np.linalg.det(dm)
    assert np.isfinite(determinant).all()
    inverted = np.flatnonzero(determinant <= 0)
    return {
        "method": "float64 CPU det(deformed edge matrix) / det(constitutive edge matrix)",
        "tetrahedra": len(determinant),
        "detF_min": float(determinant.min()),
        "detF_p001": float(np.quantile(determinant, 0.001)),
        "detF_max": float(determinant.max()),
        "inverted_tetrahedra": len(inverted),
        "inverted_tetrahedron_ids": inverted.tolist(),
    }, displacement


def _active_strain(run_dir: Path) -> dict[str, Any]:
    mapping = _json(run_dir / "active-strain-mapping.json")
    fields_record = mapping["fields"]
    validation_record = mapping["derivative_validation"]
    assert _input(fields_record, "active-strain fields") == (
        run_dir / "active-strain-fields.npz"
    )
    _input(validation_record, "active-strain derivative validation")
    validation = _json(Path(validation_record["path"]))
    assert validation["success"] is True
    assert mapping["additive_stress_fields_in_installed_model"] is False
    installed = mapping["installed_potentials"]
    assert installed["fat"] == "StableNeoHookeanActive"
    assert installed["muscle"] == "StableNeoHookeanActive"
    assert installed["aponeurosis"] == "StableNeoHookeanActive"
    assert installed["skin"] == "StableNeoHookeanActiveMembrane"

    with np.load(run_dir / "active-strain-fields.npz", allow_pickle=False) as fields:
        B = np.asarray(fields["skin_activation_inverse"], dtype=np.float64)
        tension = np.asarray(fields["source_skin_tension_mpa_m"], dtype=np.float64)
        mu = np.asarray(fields["skin_mu_mpa"], dtype=np.float64)
        thickness = np.asarray(fields["skin_thickness_m"], dtype=np.float64)
    assert B.shape == tension.shape
    assert B.shape[1:] == (2, 2)
    target = np.eye(2)[None] + tension / (thickness * mu)[:, None, None]
    error = B @ np.swapaxes(B, -1, -2) - target
    relative = float(np.linalg.norm(error) / np.linalg.norm(target))
    eigenvalues = np.linalg.eigvalsh(B)
    assert np.isfinite(B).all()
    assert np.all(eigenvalues > 0)
    assert relative < 1e-12, relative
    equivalence = mapping["full_face_seed_equivalence"]
    assert equivalence["force_relative_error"] < 1e-10
    assert equivalence["hvp_relative_error"] < 1e-10
    return {
        "mapping_receipt": _record(run_dir / "active-strain-mapping.json"),
        "fields": fields_record,
        "derivative_validation": validation_record,
        "installed_potentials": installed,
        "additive_stress_fields_in_installed_model": False,
        "skin_BBt_relation": "B B^T = I + T / (h mu)",
        "skin_BBt_relative_error": relative,
        "skin_B_principal_min": float(eigenvalues.min()),
        "skin_B_principal_max": float(eigenvalues.max()),
        "full_face_seed_equivalence": equivalence,
    }


def _continuation(
    protocol: dict[str, Any], run_dir: Path, constitutive: Path
) -> dict[str, Any] | None:
    """Verify that an optional Newton continuation retains its parent state."""
    continuation = protocol.get("continuation")
    if continuation is None:
        return None
    required = {
        name: continuation[name]
        for name in (
            "parent_protocol",
            "parent_summary",
            "parent_endpoint",
            "parent_stiffness",
        )
    }
    parent_dir = Path(continuation["parent_directory"]).resolve()
    assert parent_dir.is_dir(), parent_dir
    parent_paths = {
        name: _input(record, f"continuation {name}")
        for name, record in required.items()
    }
    assert all(path.parent == parent_dir for path in parent_paths.values())
    parent_protocol = _json(parent_paths["parent_protocol"])
    parent_summary = _json(parent_paths["parent_summary"])
    parent_stiffness = _json(parent_paths["parent_stiffness"])
    assert parent_summary["protocol"] == parent_protocol
    assert parent_protocol["schema"] == protocol["schema"]

    with np.load(parent_paths["parent_endpoint"], allow_pickle=False) as parent:
        parent_endpoint = np.asarray(parent["displacement_m"], dtype=np.float64)
    seed = _input(protocol["fixture"]["seed"], "continuation seed")
    with np.load(seed, allow_pickle=False) as current:
        current_seed = np.asarray(current["displacement_m"], dtype=np.float64)
    assert np.array_equal(current_seed, parent_endpoint)

    parent_active = _json(parent_dir / "active-strain-mapping.json")
    current_active = _json(run_dir / "active-strain-mapping.json")
    assert (
        parent_active["installed_potentials"] == current_active["installed_potentials"]
    )
    parent_fields = _input(parent_active["fields"], "parent active-strain fields")
    current_fields = _input(current_active["fields"], "current active-strain fields")
    with (
        np.load(parent_fields, allow_pickle=False) as parent,
        np.load(current_fields, allow_pickle=False) as current,
    ):
        assert set(parent.files) == set(current.files)
        assert all(np.array_equal(parent[name], current[name]) for name in parent.files)

    parent_constitutive, _ = _constitutive_volume(parent_protocol)
    assert np.array_equal(
        np.asarray(pv.read(parent_constitutive).points),
        np.asarray(pv.read(constitutive).points),
    )
    parent_configuration = parent_protocol["reference_configuration"]
    current_configuration = protocol["reference_configuration"]
    parent_skin = _input(
        parent_configuration["constitutive_skin"], "parent constitutive skin"
    )
    current_skin = _input(
        current_configuration["constitutive_skin"], "current constitutive skin"
    )
    assert np.array_equal(
        np.asarray(pv.read(parent_skin).points),
        np.asarray(pv.read(current_skin).points),
    )

    parent_policy = parent_protocol["contact_stiffness_policy"]
    policy = protocol["contact_stiffness_policy"]
    for name in ("tolerance_anchor_force", "effective_force_tolerance"):
        assert policy[name] == parent_policy[name]
    parent_terminal = float(parent_summary["result"]["terminal_stiffness_mpa"])
    assert parent_terminal == float(parent_stiffness["final_stiffness"])
    assert float(policy["initial_stiffness_mpa"]) == parent_terminal
    assert float(policy["maximum_stiffness_mpa"]) == float(
        parent_policy["maximum_stiffness_mpa"]
    )
    return {
        "parent_directory": str(parent_dir),
        "parent_inputs": {name: _record(path) for name, path in parent_paths.items()},
        "seed_equals_parent_endpoint": True,
        "active_strain_fields_equal": True,
        "constitutive_reference_coordinates_equal": True,
        "tolerance_anchor_force_preserved": policy["tolerance_anchor_force"],
        "effective_force_tolerance_preserved": policy["effective_force_tolerance"],
        "initial_stiffness_mpa": policy["initial_stiffness_mpa"],
        "maximum_stiffness_mpa": policy["maximum_stiffness_mpa"],
    }


def main(cfg: Config) -> None:
    run_dir = cfg.run_dir.resolve()
    output = run_dir / cfg.output_name
    assert not output.exists(), output
    required = {
        name: run_dir / name
        for name in (
            "summary.json",
            "protocol.json",
            "endpoint.npz",
            "status.json",
            "active-strain-mapping.json",
            "active-strain-fields.npz",
            "active-strain-source-provenance.json",
            "profile-input-binding.json",
            "forward-only-guard.json",
            "provenance.json",
        )
    }
    assert all(path.is_file() for path in required.values()), required
    summary = _json(required["summary.json"])
    protocol = _json(required["protocol.json"])
    result = summary["result"]
    assert protocol == summary["protocol"]
    assert protocol["schema"] == "natural-reference-active-strain-hybrid-v1"
    assert _json(required["status.json"])["running"] is False
    for label in ("eyes_manifest", "seed", "seed_receipt", "runtime_binding"):
        _input(protocol["fixture"].get(label, protocol.get(label)), label)

    guard = _json(required["forward-only-guard.json"])
    assert guard["completed_without_inverse_calls"] is True
    assert sorted(guard["guarded_entrypoints"]) == [
        "adjoint_solve",
        "forward",
        "receipt",
        "step",
    ]
    constitutive, reference_provenance = _constitutive_volume(protocol)
    geometry, displacement = _detf(constitutive, required["endpoint.npz"])
    assert geometry["inverted_tetrahedra"] == result["geometry"]["inverted_tetrahedra"]
    assert np.isclose(geometry["detF_min"], result["geometry"]["detF_min"])
    active = _active_strain(run_dir)
    runtime_boundary = protocol.get("fixed_boundary")
    fixed_boundary_path = run_dir / "fixed-boundary.json"
    if runtime_boundary is None:
        assert not fixed_boundary_path.exists()
    else:
        assert fixed_boundary_path.is_file()
        assert runtime_boundary == _json(fixed_boundary_path)
    boundary = _isfixed_boundary(
        protocol,
        constitutive,
        _json(required["profile-input-binding.json"]),
        runtime_boundary,
    )
    continuation = _continuation(protocol, run_dir, constitutive)
    source_counts = _verify_archive_sources(run_dir, active)

    force = float(result["terminal_force"])
    tolerance = float(protocol["contact_stiffness_policy"]["effective_force_tolerance"])
    collision = result["collision"]
    force_converged = force <= tolerance
    contact_feasible = bool(
        collision["state_feasible"] and collision["minimum_gap_gate"]
    )
    expected_valid = bool(
        result["success"]
        and force_converged
        and geometry["inverted_tetrahedra"] == 0
        and contact_feasible
    )
    assert bool(result["valid_forward"]) == expected_valid
    receipt = {
        "schema": "new-neutral-active-strain-independent-audit-v1",
        "scope": "Saved-result CPU and provenance audit; it does not rerun the solver.",
        "run_inputs": {name: _record(path) for name, path in required.items()},
        "constitutive_volume": _record(constitutive),
        "reference_provenance": reference_provenance,
        "endpoint": {
            **_record(required["endpoint.npz"]),
            "shape": list(displacement.shape),
            "finite": bool(np.isfinite(displacement).all()),
        },
        "active_strain": active,
        "fixed_boundary_record": (
            _record(fixed_boundary_path) if fixed_boundary_path.is_file() else None
        ),
        "isfixed_boundary": boundary,
        "continuation": continuation,
        "source_archive_counts": source_counts,
        "forward_only_guard": guard,
        "geometry": geometry,
        "solver_gates": {
            "solver_success": bool(result["success"]),
            "terminal_force": force,
            "force_tolerance": tolerance,
            "force_converged": force_converged,
            "contact_feasible": contact_feasible,
            "minimum_active_distance_m": collision["minimum_active_distance_m"],
            "minimum_required_distance_m": collision["minimum_required_distance_m"],
            "reported_valid_forward": bool(result["valid_forward"]),
            "recomputed_valid_forward": expected_valid,
        },
    }
    write_json(output, receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "audit/inverted_tetrahedra": geometry["inverted_tetrahedra"],
            "audit/detF_min": geometry["detF_min"],
            "audit/force_converged": float(force_converged),
            "audit/contact_feasible": float(contact_feasible),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
