"""Verify the corrected FEM constraints on the actual face and saved endpoints."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT / "src")]
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import configure_cuda
from joint_rigid_eye_contact import build_eye_collision_physics
from neutral_active_strain import install_active_strain
from profile_input_binding import (
    CURRENT_OWNED_ADJOINT_SHA256,
    bind_frozen_neutral_load,
)
from reference_rebase import rebase_reference

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/isfixed-runtime-audit-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    old_neutral_dir: Path = GROUP / "data/forward-repaired-reference-005"
    old_trial_dir: Path = GROUP / "data/inverse-mouthopen-rigid-trial-001"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main(cfg: Config) -> None:
    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    sources = output / "sources"
    sources.mkdir()
    for path in (
        Path(__file__),
        GROUP / "src/reference_rebase.py",
        JOINT / "src/joint_physics.py",
        JOINT / "src/joint_full_skull_contact.py",
        SOLVERS / "profile_input_binding.py",
    ):
        shutil.copy2(path, sources / path.name)
    configure_cuda()
    ipctk.set_num_threads(4)
    with bind_frozen_neutral_load(
        cfg.neutral_dir,
        output,
        allow_pncg_curvature_clamps=True,
        allow_isfixed_boundary=True,
        unused_inverse_sha256=CURRENT_OWNED_ADJOINT_SHA256,
    ) as neutral:
        physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    source_boundary = dict(physics.fixed_boundary_receipt)
    rebased_dir = output / "rebased"
    rebased_dir.mkdir()
    physics, baseline, reference = rebase_reference(
        physics, baseline, cfg.reference_dir, rebased_dir
    )
    model = physics.runtime.forward.model
    n = len(physics.points)
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    fixed = np.flatnonzero(is_fixed)
    lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    cran = np.asarray(neutral.arrays["cranium_node_ids"])
    mand = np.asarray(neutral.arrays["mandible_node_ids"])
    old_fixed = np.union1d(fixed, np.union1d(cran, mand))
    released = np.setdiff1d(old_fixed, fixed)
    prescribed = np.zeros((model.n_points, 3), dtype=bool)
    prescribed[:n] = is_fixed[:, None]
    prescribed[n:] = True
    actual_fixed = model.dof_map.fixed_indices.cpu().numpy()
    actual_free = model.dof_map.free_indices.cpu().numpy()
    np.testing.assert_array_equal(actual_fixed, np.flatnonzero(prescribed))
    np.testing.assert_array_equal(actual_free, np.flatnonzero(~prescribed))
    np.testing.assert_array_equal(physics.full_skull.geometry.fixed_global_ids, fixed)
    np.testing.assert_array_equal(physics.mesh.point_data["FixedMask"], prescribed[:n])
    np.testing.assert_array_equal(
        physics.jaw_t.cpu().numpy(), np.intersect1d(fixed, mand)
    )
    np.testing.assert_array_equal(physics.arrays["mandible_node_ids"], mand)
    np.testing.assert_array_equal(physics.arrays["cranium_node_ids"], cran)
    assert not np.any(lip & is_fixed)
    assert np.all(~prescribed[released])
    pose = torch.tensor([0.01, -0.02, 0.005, 0.001, -0.0005, 0.0002])
    model.dof_map.fixed_values = physics.boundary(pose)
    prescribed_u = model.dof_map.to_full(torch.zeros(model.n_free))
    nonmand = np.setdiff1d(fixed, mand)
    assert torch.count_nonzero(prescribed_u[torch.as_tensor(nonmand)]) == 0
    assert torch.count_nonzero(prescribed_u[torch.as_tensor(released)]) == 0
    assert bool(
        (torch.linalg.vector_norm(prescribed_u[physics.jaw_t], dim=1) > 0).all()
    )

    spec = importlib.util.spec_from_file_location(
        "reference_audit", GROUP / "src/55-audit-reference.py"
    )
    assert spec is not None
    assert spec.loader is not None
    reference_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_audit)
    LOG.info("Checking reference clearance with corrected fixed DOFs")
    reference_contact = reference_audit._strict_ipc_audit(physics, 1e-4, 2e-4)  # noqa: SLF001
    assert reference_contact["meets_target"], reference_contact
    baseline, activation, _ = install_active_strain(model)
    model.set_materials(baseline)
    old_free_mask = np.ones((model.n_points, 3), dtype=bool)
    old_free_mask[old_fixed] = False
    old_free_mask[n:] = False
    old_free = torch.as_tensor(np.flatnonzero(old_free_mask))
    released_t = torch.as_tensor(released)
    force_audits = []
    for name, directory in (
        ("old_neutral", cfg.old_neutral_dir),
        ("old_mouthopen_trial", cfg.old_trial_dir),
    ):
        LOG.info("Evaluating saved %s under corrected boundary", name)
        with np.load(directory / "endpoint.npz", allow_pickle=False) as archive:
            u = torch.as_tensor(archive["displacement_m"].copy())
            q = (
                torch.as_tensor(archive["activation_inv"].copy())
                if name == "old_mouthopen_trial"
                else torch.zeros((len(physics.active_t), 6))
            )
            jaw = (
                torch.as_tensor(archive["pose_rad_m"].copy())
                if name == "old_mouthopen_trial"
                else torch.zeros(6)
            )
        materials = {key: dict(fields) for key, fields in baseline.items()}
        materials["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, physics.active_t, q)
        model.set_materials(materials)
        potential = model.collision.potential
        model.collision.potential = ipctk.BarrierPotential(
            type(potential.barrier)(),
            potential.dhat,
            0.3386,
            model.collision.use_physical_barrier,
        )
        model.dof_map.fixed_values = physics.boundary(jaw)
        full = physics.full_skull.extend_seed(u[:n], jaw)
        np.testing.assert_allclose(
            full.flatten()[model.dof_map.fixed_indices].cpu().numpy(),
            model.dof_map.fixed_values.cpu().numpy(),
            rtol=0,
            atol=2e-15,
        )
        state = model.State(u=full, collision=model.collision.state_at(full))
        gradient = model.grad(state)
        old_force = float(torch.linalg.vector_norm(gradient.flatten()[old_free])) * 1e6
        new_force = (
            float(torch.linalg.vector_norm(model.dof_map.to_free_grad(gradient))) * 1e6
        )
        extra_force = float(torch.linalg.vector_norm(gradient[released_t])) * 1e6
        assert np.isclose(new_force**2, old_force**2 + extra_force**2, rtol=1e-12)
        force_audits.append(
            {
                "name": name,
                "endpoint": record(directory / "endpoint.npz"),
                "old_union_free_force_n": old_force,
                "isfixed_free_force_n": new_force,
                "force_on_released_nodes_n": extra_force,
                "absolute_force_tolerance_n": 0.01,
                "equilibrium_inherited": False,
            }
        )
    arrays_path = output / "boundary-arrays.npz"
    np.savez_compressed(
        arrays_path,
        isfixed_node_ids=fixed,
        released_node_ids=released,
        fixed_dof_indices=actual_fixed,
        free_dof_indices=actual_free,
        fixed_mandible_node_ids=physics.jaw_t.cpu().numpy(),
    )
    receipt = {
        "schema": "isfixed-runtime-audit-v1",
        "success": True,
        "scope": "Actual corrected FEM/IPC construction and saved-state force audit; no new equilibrium solved.",
        "source_boundary": source_boundary,
        "runtime_boundary": physics.fixed_boundary_receipt,
        "counts": {
            "fem_nodes": n,
            "isfixed_nodes": len(fixed),
            "legacy_fixed_nodes": len(old_fixed),
            "released_nodes": len(released),
            "released_cranium_nodes": len(np.intersect1d(released, cran)),
            "released_mandible_nodes": len(np.intersect1d(released, mand)),
            "lip_nodes": int(lip.sum()),
            "fixed_lip_nodes": int((lip & is_fixed).sum()),
            "released_lip_nodes": int(lip[released].sum()),
            "appended_rigid_nodes": model.n_points - n,
        },
        "checks": {
            "runtime_dofs_exactly_match_isfixed_plus_appended_rigid_nodes": True,
            "jaw_prescription_uses_only_isfixed_mandible_intersection": True,
            "original_anatomy_arrays_preserved": True,
            "runtime_geometry_support_matches_isfixed": True,
            "all_lip_nodes_free": True,
        },
        "reference": reference,
        "reference_contact": reference_contact,
        "active_strain": activation,
        "saved_state_forces": force_audits,
        "arrays": record(arrays_path),
        "sources": {path.name: record(path) for path in sorted(sources.iterdir())},
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output / "receipt.json")
    cherries.log_output(arrays_path)
    cherries.log_metrics({"boundary/released_vertices": len(released)})
    LOG.info("IsFixed runtime audit passed: %s", receipt["counts"])
    LOG.info("Saved-state forces: %s", force_audits)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
