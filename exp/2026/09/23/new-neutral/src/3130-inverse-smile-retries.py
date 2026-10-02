# Copyright (c) 2026 liblaf
# ruff: noqa: C901, E402, PLR0912, PLR0915, FBT001, FBT003, PT018
"""Smile continuation with bounded rejection of known numerical trial failures."""

from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import ipctk
import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)
POSE_SCALE = torch.tensor([math.pi / 18.0] * 3 + [0.01] * 3, dtype=torch.float64)
MAX_ROTATION_INCREMENT_DEG = 1.0
MAX_TRANSLATION_INCREMENT_M = 0.001


def known_projection_failure(error: Exception, trial_dir: Path) -> bool:
    """Reject only the archived bounded-dual line-search failure."""
    message = "Bounded dual Newton line search unresolved"
    path = trial_dir / "summary.json"
    if (
        not isinstance(error, AssertionError)
        or str(error) != message
        or not path.is_file()
    ):
        return False
    summary = json.loads(path.read_text())
    return summary.get("status") == "joint_projection_failed" and summary.get(
        "failure"
    ) == {
        "stage": "bounded_dual",
        "type": "AssertionError",
        "message": message,
    }


def known_adjoint_failure(error: Exception, receipt: dict, tolerance: float) -> bool:
    """Recognize sparse solve failure without swallowing other assertions."""
    if not isinstance(error, AssertionError) or len(error.args) != 1:
        return False
    if (
        error.args[0] is not receipt
        or receipt.get("method")
        != "hybrid_fem_ipc_free_csr_with_optional_relative_shift"
    ):
        return False
    if not receipt.get("attempts") or not isinstance(
        receipt.get("solver_success"), bool
    ):
        return False
    operator_error = receipt.get("operator_relative_error")
    residuals = [
        receipt.get(name)
        for name in ("shifted_relative_residual", "native_shifted_relative_residual")
    ]
    if (
        not isinstance(operator_error, (int, float))
        or not math.isfinite(operator_error)
        or operator_error > 1e-10
    ):
        return False
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in residuals
    ):
        return False
    return receipt["solver_success"] is False or max(residuals) > tolerance


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/inverse-smile-coupled-001"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    blendshape_dir: Path = GROUP / "data/blendshapes-isfixed-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    expression_name: str = "Smile"
    initialization_checkpoint: Path | None = None
    initialization_refinement: Path | None = None
    seed_method: str = "coupled_tangent"
    exclude_fully_fixed_tets: bool = True
    maximum_inverted_tetrahedra: int = 100
    maximum_inverted_rest_volume_fraction: float = 1e-4
    max_rotation_increment_deg: float | None = None
    max_translation_increment_m: float | None = None
    predictor_relative_shift: float = 1e-5
    predictor_rtol: float = 1e-7
    maximum_iterations: int = 50
    learning_rate: float = 0.002
    pose_learning_rate: float = 0.1
    forward_atol: float = 1e-8
    internal_forward_atol: float | None = 1e-9
    adjoint_rtol: float = 1e-7
    adjoint_relative_shift: float = 1e-5
    gradient_check_epsilons: tuple[float, ...] = (1e-4, 1e-5, 1e-6)
    max_newton_steps: int = 3000
    max_backtracks: int = 12
    wall_seconds: float | None = None
    deadline_iso_utc: str | None = None
    no_contact_linear_max_steps: int = 3000
    initializer_wall_seconds: float = 3600
    ipc_threads: int = 4
    resume: bool = False
    continue_optimizer_state: bool = False
    q_only_iterations: int = 0
    project_pose_at_inversion_limit: bool = False
    projection_activation_threshold: float = 0.05
    projection_determinant_margin: float = 1e-6
    projection_probe_epsilon: float = 1e-4
    projection_feasible_witness: bool = False
    projection_witness_policy: Literal["zero_pose", "attainable_optimum"] = "zero_pose"
    projection_analytic_determinants: bool = False
    projection_affine_residual: bool = False
    projection_bounded_joint: bool = True
    projection_descent_fraction: float | None = 0.1
    adaptive_trial_alpha: bool = False
    initial_trial_alpha: float = 1.0
    minimum_trial_alpha: float = 0.0
    maximum_projection_failures: int = 3
    maximum_forward_failures: int = 3
    maximum_adjoint_failures: int = 1
    convergence_patience: int = 0
    convergence_loss_rtol: float = 1e-6
    convergence_gradient_rtol: float = 1e-3
    convergence_gradient_atol: float = 1e-8
    objective_mode: Literal["l2", "l2-normal-smooth"] = "l2-normal-smooth"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_expression_inputs(cfg: Config) -> dict:
    """CPU-only, hash-bound check of the selected target and neutral origin."""
    target_path = cfg.blendshape_dir / "blendshapes.npz"
    neutral_endpoint = cfg.neutral_dir / "endpoint.npz"
    repaired_reference = cfg.reference_dir / "reference-clearance.npz"
    with np.load(target_path, allow_pickle=False) as target:
        names = [str(name) for name in target["expression_names"]]
        index = names.index(cfg.expression_name)
        skin_ids = np.asarray(target["skin_global_ids"], dtype=np.int64)
        new_neutral = np.asarray(target["new_neutral_points_m"], dtype=np.float64)
        target_points = np.asarray(target["target_points_m"][index], dtype=np.float64)
        source_delta = np.asarray(
            target["expression_displacement_m"][index], dtype=np.float64
        )
    with np.load(neutral_endpoint, allow_pickle=False) as endpoint:
        neutral_u = np.asarray(endpoint["displacement_m"], dtype=np.float64)
    with np.load(repaired_reference, allow_pickle=False) as reference:
        repaired_points = np.asarray(reference["repaired_points_m"], dtype=np.float64)
    assert skin_ids.ndim == 1
    assert (
        new_neutral.shape
        == target_points.shape
        == source_delta.shape
        == (
            len(skin_ids),
            3,
        )
    )
    np.testing.assert_array_equal(
        repaired_points[skin_ids] + neutral_u[skin_ids], new_neutral
    )
    np.testing.assert_array_equal(target_points, new_neutral + source_delta)

    def closure(path: Path) -> dict:
        item = record(path)
        item["project_relative_path"] = str(path.resolve().relative_to(ROOT))
        return item

    return {
        "schema": "smile-input-preflight-v1",
        "expression_name": cfg.expression_name,
        "expression_index": index,
        "expression_count": len(names),
        "inputs": {
            "blendshapes": closure(target_path),
            "neutral_endpoint": closure(neutral_endpoint),
            "repaired_reference": closure(repaired_reference),
        },
        "neutral_origin_exact": True,
        "target_delta_exact": True,
        "skin_vertices": len(skin_ids),
    }


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_torch(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def _bounded_pose(
    normalized: torch.Tensor,
    proposed_delta: torch.Tensor,
    *,
    max_rotation_deg: float | None = MAX_ROTATION_INCREMENT_DEG,
    max_translation_m: float | None = MAX_TRANSLATION_INCREMENT_M,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one physical SE(3) increment bounded on SO(3) and translation."""
    from mouthopen_pose_path import pose_waypoints

    assert normalized.shape == proposed_delta.shape == (6,)
    scale = POSE_SCALE.to(device=normalized.device, dtype=normalized.dtype)
    old = (normalized * scale).detach().cpu().numpy()
    requested = ((normalized + proposed_delta) * scale).detach().cpu().numpy()
    if max_rotation_deg is None and max_translation_m is None:
        from scipy.spatial.transform import Rotation

        angle = float(
            (
                Rotation.from_rotvec(requested[:3])
                * Rotation.from_rotvec(old[:3]).inv()
            ).magnitude()
        )
        assert angle < math.pi
        return normalized + proposed_delta, {
            "rotation_deg": math.degrees(angle),
            "translation_m": float(np.linalg.norm(requested[3:] - old[3:])),
        }
    assert max_rotation_deg is not None and max_translation_m is not None
    path, receipt = pose_waypoints(
        old,
        requested,
        max_rotation_deg=max_rotation_deg,
        max_translation_m=max_translation_m,
    )
    new = path[min(1, len(path) - 1)]
    result = (
        torch.as_tensor(new, dtype=normalized.dtype, device=normalized.device) / scale
    )
    increment = (
        receipt["steps"][0]
        if receipt["steps"]
        else {"rotation_deg": 0.0, "translation_m": 0.0}
    )
    assert increment["rotation_deg"] <= max_rotation_deg + 1e-10
    assert increment["translation_m"] <= max_translation_m + 1e-12
    return result, increment


def adam_update(
    moments: list[torch.Tensor],
    gradients: tuple[torch.Tensor, torch.Tensor],
    *,
    optimizer_step: int,
    learning_rate: float,
    pose_learning_rate: float,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return the unchanged Adam proposal at its explicit global step."""
    assert optimizer_step >= 1
    assert len(moments) == 4
    gq, gp = gradients
    mq, vq, mp, vp = moments
    assert mq.shape == vq.shape == gq.shape
    assert mp.shape == vp.shape == gp.shape
    proposed_moments = [
        0.9 * mq + 0.1 * gq,
        0.999 * vq + 0.001 * gq.square(),
        0.9 * mp + 0.1 * gp,
        0.999 * vp + 0.001 * gp.square(),
    ]
    next_mq, next_vq, next_mp, next_vp = proposed_moments
    dq = (
        -learning_rate
        * (next_mq / (1 - 0.9**optimizer_step))
        / ((next_vq / (1 - 0.999**optimizer_step)).sqrt() + 1e-12)
    )
    dp = (
        -pose_learning_rate
        * (next_mp / (1 - 0.9**optimizer_step))
        / ((next_vp / (1 - 0.999**optimizer_step)).sqrt() + 1e-12)
    )
    return proposed_moments, dq, dp


def main(cfg: Config) -> None:
    from mesh_step_scale import mean_rest_edge_length
    from mouthopen_block_optimizer import block_adam_update
    from mouthopen_convergence import gradient_metrics, stationarity_candidate
    from mouthopen_rigid_seed import prepare_rigid_seed
    from mouthopen_runtime import install_mouthopen_hybrid_runtime

    output = cfg.output_dir.resolve()
    assert cfg.maximum_iterations >= 0
    assert cfg.q_only_iterations >= 0
    assert 0 < cfg.initial_trial_alpha <= 1
    assert 0 <= cfg.minimum_trial_alpha < cfg.initial_trial_alpha
    assert cfg.convergence_patience >= 0
    assert cfg.maximum_projection_failures == 3
    assert cfg.maximum_forward_failures == 3
    assert cfg.maximum_adjoint_failures == 1
    assert not (cfg.resume and cfg.continue_optimizer_state)
    internal_forward_atol = (
        cfg.forward_atol
        if cfg.internal_forward_atol is None
        else cfg.internal_forward_atol
    )
    assert 0 < internal_forward_atol <= cfg.forward_atol
    if cfg.initialization_refinement is not None:
        assert cfg.continue_optimizer_state
        assert not cfg.resume
    assert (
        cfg.projection_witness_policy == "zero_pose" or cfg.projection_feasible_witness
    )
    assert not cfg.projection_feasible_witness or (
        cfg.project_pose_at_inversion_limit
        and cfg.projection_descent_fraction is not None
    )
    if cfg.projection_bounded_joint:
        assert not cfg.projection_affine_residual
        assert not cfg.project_pose_at_inversion_limit
        assert not cfg.projection_feasible_witness
        assert cfg.projection_descent_fraction is not None
        assert cfg.q_only_iterations == 0
        assert cfg.max_rotation_increment_deg is None
        assert cfg.max_translation_increment_m is None
        assert cfg.seed_method == "coupled_tangent"
    if cfg.projection_affine_residual:
        assert cfg.projection_analytic_determinants
        assert cfg.projection_feasible_witness
        assert cfg.projection_witness_policy == "attainable_optimum"
        assert cfg.project_pose_at_inversion_limit
        assert cfg.projection_descent_fraction is not None
        assert cfg.q_only_iterations == 0
        assert cfg.max_rotation_increment_deg is None
        assert cfg.max_translation_increment_m is None
        assert cfg.seed_method == "coupled_tangent"
    if cfg.projection_descent_fraction is not None:
        assert 0 < cfg.projection_descent_fraction <= 1
        assert cfg.max_rotation_increment_deg is None
        assert cfg.max_translation_increment_m is None
    assert cfg.resume or not output.exists(), output
    output.mkdir(parents=True, exist_ok=cfg.resume)
    write_json(output / "input-preflight.json", verify_expression_inputs(cfg))
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)

    neutral_audit = json.loads((cfg.neutral_dir / "independent-audit.json").read_text())
    for name in ("endpoint.npz", "protocol.json", "summary.json"):
        assert (
            sha256(cfg.neutral_dir / name)
            == neutral_audit["run_inputs"][name]["sha256"]
        )
    transfer = json.loads((cfg.blendshape_dir / "manifest.json").read_text())
    assert transfer["transfer_success"]
    target_path = cfg.blendshape_dir / "blendshapes.npz"
    assert sha256(target_path) == transfer["artifacts"]["blendshapes.npz"]["sha256"]

    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.concatenate(
        (
            np.repeat(is_fixed, 3),
            np.ones((model.dof_map.n_full - len(is_fixed) * 3), dtype=bool),
        )
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
    )
    baseline, strain_receipt, _ = install_active_strain(model)
    tetrahedron_policy = {"excluded_tetrahedra": 0, "method": "original complete FEM"}
    if cfg.exclude_fully_fixed_tets:
        from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra

        # All-fixed cells must contribute exactly zero to the free equations.
        # Prove this on the actual neutral, including a free Hessian product.
        with np.load(cfg.neutral_dir / "endpoint.npz", allow_pickle=False) as archive:
            probe_u = torch.zeros_like(physics.runtime.forward.state.u)
            probe_u[: len(is_fixed)] = torch.as_tensor(
                archive["displacement_m"][: len(is_fixed)],
                device=probe_u.device,
                dtype=probe_u.dtype,
            )
        model.dof_map.fixed_values = physics.boundary(
            torch.zeros(6, device=probe_u.device, dtype=probe_u.dtype)
        )
        probe_u = model.dof_map.to_full(model.dof_map.to_free(probe_u))
        generator = torch.Generator(device=probe_u.device).manual_seed(20260930)
        probe_direction = model.dof_map.to_full_grad(
            torch.randn(
                (len(model.dof_map.free_indices),),
                device=probe_u.device,
                dtype=probe_u.dtype,
                generator=generator,
            )
        )

        def probe_free_equations():
            state = model.State(u=probe_u)
            state.collision = model.collision.state_at(probe_u)
            return (
                model.dof_map.to_free_grad(model.grad(state)).detach().clone(),
                model.dof_map.to_free_grad(model.hess_prod(state, probe_direction))
                .detach()
                .clone(),
            )

        original_probe = probe_free_equations()
        tetrahedron_policy = exclude_fully_fixed_tetrahedra(physics)
        baseline = model.get_materials()
        retained_probe = probe_free_equations()
        proof = {}
        for name, before, after in zip(
            ("free_gradient", "free_hessian_product"),
            original_probe,
            retained_probe,
            strict=True,
        ):
            difference = float(torch.linalg.vector_norm(after - before))
            scale = float(torch.linalg.vector_norm(before))
            proof[name] = {
                "absolute_error": difference,
                "relative_error": difference / max(scale, 1e-30),
            }
            torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-15)
        tetrahedron_policy["neutral_free_equation_proof"] = proof
        write_json(output / "tetrahedron-policy.json", tetrahedron_policy)

    inversion_policy = {
        "maximum_inverted_tetrahedra": cfg.maximum_inverted_tetrahedra,
        "maximum_inverted_rest_volume_fraction": cfg.maximum_inverted_rest_volume_fraction,
        "orientation_floor": None,
        "scope": "retained mechanical tetrahedra",
    }
    assert cfg.maximum_inverted_tetrahedra >= 0
    assert 0 <= cfg.maximum_inverted_rest_volume_fraction <= 1

    def geometry_metrics(u: torch.Tensor) -> dict:
        if cfg.exclude_fully_fixed_tets:
            from mouthopen_tet_policy import geometry_metrics as retained_metrics

            return retained_metrics(physics, u)
        return physics.metrics(u)

    def geometry_allowed(geometry: dict) -> bool:
        return (
            geometry["inverted_tetrahedra"] <= cfg.maximum_inverted_tetrahedra
            and geometry.get("inverted_rest_volume_fraction", 0.0)
            <= cfg.maximum_inverted_rest_volume_fraction
        )

    def bounded_pose(pose: torch.Tensor, delta: torch.Tensor):
        return _bounded_pose(
            pose,
            delta,
            max_rotation_deg=cfg.max_rotation_increment_deg,
            max_translation_m=cfg.max_translation_increment_m,
        )

    with np.load(
        cfg.neutral_dir / "active-strain-fields.npz", allow_pickle=False
    ) as saved_strain:
        np.testing.assert_array_equal(
            baseline["skin"]["activation_inv"].cpu().numpy(),
            saved_strain["skin_activation_inverse"],
        )
        np.testing.assert_array_equal(
            baseline["skin"]["mu"].cpu().numpy(), saved_strain["skin_mu_mpa"]
        )
        np.testing.assert_array_equal(
            baseline["skin"]["thickness"].cpu().numpy(),
            saved_strain["skin_thickness_m"],
        )
    strain_receipt["neutral_skin_prestretch_mu_thickness_preserved_exactly"] = True
    model.set_materials(baseline)
    kappa = float(
        json.loads((cfg.neutral_dir / "stiffness.json").read_text())["final_stiffness"]
    )
    collision = model.collision
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=internal_forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        adjoint_relative_shift=cfg.adjoint_relative_shift,
        newton_max_steps=cfg.max_newton_steps,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    with np.load(target_path, allow_pickle=False) as data:
        target_index = list(data["expression_names"]).index(cfg.expression_name)
        skin_ids = data["skin_global_ids"].copy()
        tri = data["skin_triangles"].copy()
        neutral_points = data["new_neutral_points_m"].copy()
        target_points = data["target_points_m"][target_index].copy()
    with np.load(cfg.neutral_dir / "endpoint.npz") as data:
        neutral_u = torch.as_tensor(data["displacement_m"].copy())
    np.testing.assert_array_equal(
        physics.points[skin_ids] + neutral_u.cpu().numpy()[skin_ids], neutral_points
    )
    ids = torch.as_tensor(skin_ids, dtype=torch.int64)
    active_ids = physics.base.active_t
    active_source_ids = (
        physics.base.retained_active_cell_ids
        if cfg.exclude_fully_fixed_tets
        else active_ids.cpu().numpy()
    )
    if cfg.initialization_checkpoint is None:
        initialization_dir = output / "neutral-initialization"
        initialization_dir.mkdir(exist_ok=cfg.resume)
        initialization_checkpoint = initialization_dir / "endpoint.pt"
        zero_pose = torch.zeros(6, dtype=neutral_u.dtype, device="cuda")
        full_seed = torch.zeros_like(runtime.forward.state.u)
        full_seed[: len(neutral_u)] = neutral_u.to(device="cuda")
        model.dof_map.fixed_values = physics.boundary(zero_pose)
        full_seed = model.dof_map.to_full(model.dof_map.to_free(full_seed))
        initial_activation = (
            baseline["muscle"]["activation_inv"][active_ids].detach().clone()
        )
        if not cfg.resume:
            save_torch(
                initialization_checkpoint,
                {
                    "activation_inv": initial_activation.cpu(),
                    "pose_rad_m": zero_pose.cpu(),
                    "displacement_m": full_seed.cpu(),
                },
            )
            save_npz(
                initialization_checkpoint.with_suffix(".npz"),
                activation_inv=initial_activation.cpu().numpy(),
                pose_rad_m=zero_pose.cpu().numpy(),
                displacement_m=full_seed.cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
        initialization_method = (
            "verified corrected neutral equilibrium with zero jaw pose"
        )
    else:
        initialization_checkpoint = cfg.initialization_checkpoint.resolve()
        initialization_method = "saved endpoint checkpoint; force, contact, and tetrahedra revalidated before optimization"
    initialization_endpoint = initialization_checkpoint.with_name("endpoint.npz")
    assert initialization_checkpoint.is_file()
    assert initialization_endpoint.is_file()
    with np.load(initialization_endpoint, allow_pickle=False) as data:
        initialization_active_ids = np.asarray(data["active_cell_ids"])
    np.testing.assert_array_equal(initialization_active_ids, active_source_ids)
    target = torch.as_tensor(target_points - physics.points[skin_ids])
    delta = torch.as_tensor(target_points - neutral_points)
    xyz = neutral_points[tri]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    weights_t = torch.as_tensor(weights)
    scale2 = (weights_t[:, None] * delta.square()).sum()
    assert float(scale2) > 0

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return result

    def physical_pose(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == (6,)
        return value * POSE_SCALE.to(device=value.device, dtype=value.dtype)

    fit_objective = None
    if cfg.objective_mode == "l2-normal-smooth":
        from mouthopen_fit_objective import build_mouthopen_fit_objective

        fit_objective = build_mouthopen_fit_objective(
            reference_points=physics.points,
            skin_ids=skin_ids,
            triangles=tri,
            target_points=target_points,
            area_reference_points=neutral_points,
            vertex_weights=weights,
            scale2=float(scale2),
            active_tets=np.asarray(physics.base.tets)[active_source_ids],
            muscle_ids=np.asarray(physics.mesh.cell_data["MuscleId"])[
                active_source_ids
            ],
            physical_volumes=np.asarray(physics.base.volumes)[active_source_ids]
            * np.asarray(physics.mesh.cell_data["MuscleFraction"])[active_source_ids],
            device=target.device,
            dtype=target.dtype,
        )
        fit_objective.protocol["muscle_label_source"] = (
            "original mesh cell_data.MuscleId in saved active_cell_ids order"
        )
        fit_objective.protocol["mode"] = cfg.objective_mode
        write_json(output / "objective-terms.json", fit_objective.protocol)

    def objective(u: torch.Tensor) -> torch.Tensor:
        if fit_objective is not None:
            return fit_objective.surface(u)
        return (weights_t[:, None] * (u[ids] - target).square()).sum() / scale2

    control_objective = fit_objective.regularizer if fit_objective is not None else None

    protocol = None
    if cfg.resume:
        protocol = json.loads((output / "protocol.json").read_text())
        assert protocol["schema"] == "new-neutral-expression-rigid6-inverse-v1"
        assert protocol["expression_name"] == cfg.expression_name
        assert protocol["sources"]["blendshapes"] == record(target_path)
        assert protocol["sources"]["neutral_endpoint"] == record(
            cfg.neutral_dir / "endpoint.npz"
        )
        assert protocol["sources"]["reference_repair"] == record(
            cfg.reference_dir / "reference-clearance.npz"
        )
        assert protocol["initialization"]["checkpoint"] == record(
            initialization_checkpoint
        )
        assert protocol["initialization"]["endpoint"] == record(initialization_endpoint)
        assert protocol["config"].get("objective_mode", "l2") == cfg.objective_mode

    seed_attempt = len(list((output / "initialization").glob("attempt-*")))

    def evaluate(
        value: torch.Tensor,
        pose_value: torch.Tensor,
        initial: torch.Tensor,
        gradient: bool,
        *,
        source_q: torch.Tensor | None = None,
        source_pose: torch.Tensor | None = None,
    ) -> dict:
        nonlocal seed_attempt
        if source_q is not None:
            assert source_pose is not None
            seed_dir = output / "initialization" / f"attempt-{seed_attempt:05d}"
            seed_attempt += 1
            try:
                seed_function = prepare_rigid_seed
                if cfg.seed_method == "collision_carry":
                    from mouthopen_collision_seed import prepare_collision_seed

                    seed_function = prepare_collision_seed
                elif cfg.seed_method == "coupled_tangent":
                    from mouthopen_coupled_seed import prepare_coupled_seed

                    seed_function = prepare_coupled_seed
                else:
                    assert cfg.seed_method == "legacy_contact_off"
                initial, seed_receipt = seed_function(
                    physics,
                    materials,
                    source_q,
                    value,
                    physical_pose(source_pose),
                    physical_pose(pose_value),
                    initial,
                    seed_dir,
                    forward_atol=cfg.forward_atol,
                    max_newton_steps=cfg.max_newton_steps,
                    off_wall_seconds=cfg.initializer_wall_seconds,
                    no_contact_linear_max_steps=cfg.no_contact_linear_max_steps,
                    deadline=deadline,
                    predictor_relative_shift=cfg.predictor_relative_shift,
                    predictor_rtol=cfg.predictor_rtol,
                )
            except ForwardConvergenceError as error:
                summary = seed_dir / "summary.json"
                row = {
                    "success": False,
                    "output_dir": str(seed_dir.resolve()),
                    "summary": record(summary) if summary.is_file() else None,
                    "error_receipt": error.receipt,
                }
                append(output / "predictors.jsonl", row)
                error.seed_dir = str(seed_dir.resolve())
                raise
            seed_receipt["output_dir"] = str(seed_dir.resolve())
            append(output / "predictors.jsonl", seed_receipt)
            seed_geometry = geometry_metrics(initial[: len(physics.points)])
            if not geometry_allowed(seed_geometry):
                message = "predictor exceeds declared inversion allowance"
                raise ForwardConvergenceError(message, receipt=seed_geometry)
        value = value.detach().requires_grad_(gradient)
        pose_value = pose_value.detach().requires_grad_(gradient)
        u = runtime.solve(
            materials(value),
            physics.boundary(physical_pose(pose_value)),
            initial.detach(),
            key=cfg.expression_name,
        )
        geometry = geometry_metrics(u[: len(physics.points)])
        if not geometry_allowed(geometry):
            message = (
                f"{cfg.expression_name} candidate exceeds declared inversion allowance"
            )
            raise ForwardConvergenceError(
                message,
                receipt={
                    "geometry": geometry,
                    "forward": copy.deepcopy(runtime.last_forward),
                },
            )
        loss = objective(u)
        if control_objective is not None:
            loss = loss + control_objective(value)
        components = {
            "position_loss": float(
                (weights_t[:, None] * (u[ids] - target).square()).sum() / scale2
            )
        }
        if fit_objective is not None:
            components = {
                key: float(component.detach())
                for key, component in fit_objective.components(u, value).items()
            }
            assert math.isclose(float(loss), components["loss"], rel_tol=1e-13)
        forward = copy.deepcopy(runtime.last_forward)
        grads = torch.autograd.grad(loss, (value, pose_value)) if gradient else None
        if gradient:
            runtime.last_adjoint["sparse_solver"] = copy.deepcopy(
                runtime.last_sparse_adjoint
            )
        return {
            "q": value.detach(),
            "pose": pose_value.detach(),
            "u": u.detach(),
            "loss": float(loss),
            "objective_components": components,
            "grads": grads,
            "forward": forward,
            "adjoint": copy.deepcopy(runtime.last_adjoint) if gradient else None,
        }

    def metric(
        candidate: dict,
        local_iteration: int,
        optimizer_steps: dict[str, int],
        optimizer_phase: str,
        elapsed: float,
        **extra: object,
    ) -> dict:
        assert set(optimizer_steps) == {"q", "pose"}
        assert optimizer_phase in {"initial", "q_only", "joint"}
        u = candidate["u"][: len(physics.points)]
        geometry = geometry_metrics(u)
        forward = candidate["forward"]
        terminal_gates = forward.get("terminal_gates", {})
        contact_valid = all(
            terminal_gates.get(name, False)
            for name in (
                "contact_numerically_valid",
                "no_intersections",
                "minimum_active_gap_at_least_buffer",
            )
        )
        pose_m = physical_pose(candidate["pose"]).detach().cpu()
        return {
            "iteration": local_iteration,
            "local_iteration": local_iteration,
            "optimizer_step": max(optimizer_steps.values()),
            "optimizer_steps": dict(optimizer_steps),
            "optimizer_phase": optimizer_phase,
            "loss": candidate["loss"],
            "objective_mode": cfg.objective_mode,
            "loss_components": candidate["objective_components"],
            "fit_rms_mm": math.sqrt(
                candidate["objective_components"]["position_loss"] * float(scale2)
            )
            * 1000,
            "positional_fit_rms_mm": math.sqrt(
                candidate["objective_components"]["position_loss"] * float(scale2)
            )
            * 1000,
            "pose_rad_m": pose_m.tolist(),
            "pose_rotation_degrees": float(
                torch.linalg.vector_norm(pose_m[:3]) * 180 / math.pi
            ),
            "pose_translation_mm": float(torch.linalg.vector_norm(pose_m[3:]) * 1000),
            "force_norm_n": forward["grad_norm"] * 1e6,
            "force_threshold_n": cfg.forward_atol * 1e6,
            "internal_force_threshold_n": internal_forward_atol * 1e6,
            "forward_converged": forward["success"],
            "contact_valid": contact_valid,
            "geometry": geometry,
            "inversion_free": geometry["inverted_tetrahedra"] == 0,
            "geometry_within_declared_allowance": geometry_allowed(geometry),
            "valid_forward": geometry_allowed(geometry)
            and forward["success"]
            and contact_valid,
            "inverse_converged": False,
            "gradient": gradient_metrics(candidate["grads"]),
            "adjoint_relative_shift": cfg.adjoint_relative_shift,
            "adjoint_gradient_type": (
                "damped_approximation"
                if cfg.adjoint_relative_shift > 0
                else "unshifted_implicit"
            ),
            "elapsed_seconds": elapsed,
            "activation_rms": float(candidate["q"].square().mean().sqrt()),
            "activation_max_abs": float(candidate["q"].abs().max()),
            **extra,
        }

    initialization_refinement = None
    moments: list[torch.Tensor]
    history: list[dict] = []
    start_iteration = 0
    optimizer_steps = {"q": 0, "pose": 0}
    optimizer_state_source = "fresh_zero_moments"
    if cfg.resume:
        state = torch.load(output / "checkpoint.pt", weights_only=False)
        q = state["activation_inv"].to(device="cuda")
        pose_z = state["pose_normalized"].to(device="cuda")
        seed = state["displacement_m"].to(device="cuda")
        moments = [value.to(device="cuda") for value in state["moments"]]
        assert len(moments) == 4
        start_iteration = int(state.get("local_iteration", state["iteration"]))
        legacy_step = int(state.get("optimizer_step", state["iteration"]))
        optimizer_steps = dict(
            state.get("optimizer_steps", {"q": legacy_step, "pose": legacy_step})
        )
        assert set(optimizer_steps) == {"q", "pose"}
        assert all(
            isinstance(value, int) and value >= 0 for value in optimizer_steps.values()
        )
        optimizer_state_source = "same_run_checkpoint"
        history = [
            json.loads(line)
            for line in (output / "progress.jsonl").read_text().splitlines()
        ]
    else:
        state = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=False
        )
        assert {"activation_inv", "pose_rad_m", "displacement_m"} <= state.keys()
        q = state["activation_inv"].to(device="cuda")
        initial_pose = state["pose_rad_m"].to(device="cuda")
        seed = state["displacement_m"].to(device="cuda")
        assert q.shape == (len(active_ids), 6) and initial_pose.shape == (6,)
        assert seed.shape == physics.full_skull.full_reference_points_m.shape
        assert bool(torch.isfinite(q).all())
        assert bool(torch.isfinite(initial_pose).all())
        assert bool(torch.isfinite(seed).all())
        with np.load(initialization_endpoint, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["activation_inv"], q.cpu().numpy())
            np.testing.assert_array_equal(
                data["pose_rad_m"], initial_pose.cpu().numpy()
            )
            np.testing.assert_array_equal(data["displacement_m"], seed.cpu().numpy())
        if cfg.initialization_refinement is not None:
            refinement_path = cfg.initialization_refinement.resolve()
            refinement = json.loads(refinement_path.read_text())
            refinement_protocol_path = refinement_path.with_name("protocol.json")
            refinement_protocol = json.loads(refinement_protocol_path.read_text())
            assert refinement["refinement_succeeded"]
            assert refinement["candidate_kind"] == "completed_physical_corrector"
            assert refinement["final"]["original_acceptance_gates_met"]
            assert refinement["final"]["internal_force_target_met"]
            assert refinement["source_checkpoint"] == record(initialization_checkpoint)
            assert refinement_protocol["internal_force_target"] <= internal_forward_atol
            for item in refinement_protocol["source_refs"].values():
                assert record(Path(item["path"])) == item
            refined_endpoint = Path(refinement["endpoint"]["path"])
            assert record(refined_endpoint) == refinement["endpoint"]
            with np.load(refined_endpoint, allow_pickle=False) as data:
                for key in ("activation_inv", "pose_rad_m", "pose_normalized"):
                    np.testing.assert_array_equal(data[key], state[key].numpy())
                np.testing.assert_array_equal(
                    data["active_cell_ids"], active_source_ids
                )
                seed = torch.as_tensor(data["displacement_m"].copy(), device="cuda")
            initialization_refinement = {
                "result": record(refinement_path),
                "protocol": record(refinement_protocol_path),
                "endpoint": record(refined_endpoint),
                "original_fit_rms_mm": refinement["initial"]["fit_rms_mm"],
                "refined_fit_rms_mm": refinement["final"]["fit_rms_mm"],
                "internal_force_target": refinement_protocol["internal_force_target"],
                "optimizer_state_unchanged": True,
            }
        pose_z = initial_pose / POSE_SCALE.to(device="cuda", dtype=initial_pose.dtype)
        if cfg.continue_optimizer_state:
            # Preserve the actual optimized coordinates: divide/multiply
            # roundoff can change the sign of an almost singular boundary tet.
            assert "pose_normalized" in state
            pose_z = state["pose_normalized"].to(device="cuda")
            torch.testing.assert_close(
                physical_pose(pose_z), initial_pose, rtol=0, atol=0
            )
            assert "moments" in state
            assert "iteration" in state
            moments = [value.to(device="cuda") for value in state["moments"]]
            assert len(moments) == 4
            legacy_step = int(state.get("optimizer_step", state["iteration"]))
            optimizer_steps = dict(
                state.get("optimizer_steps", {"q": legacy_step, "pose": legacy_step})
            )
            assert set(optimizer_steps) == {"q", "pose"}
            assert all(
                isinstance(value, int) and value >= 0
                for value in optimizer_steps.values()
            )
            optimizer_state_source = (
                "initialization_checkpoint_optimizer_step"
                if "optimizer_step" in state
                else "initialization_checkpoint_legacy_iteration"
            )
        else:
            moments = [
                torch.zeros_like(q),
                torch.zeros_like(q),
                torch.zeros_like(pose_z),
                torch.zeros_like(pose_z),
            ]
        assert all(bool(torch.isfinite(value).all()) for value in moments)
        assert moments[0].shape == moments[1].shape == q.shape
        assert moments[2].shape == moments[3].shape == pose_z.shape

    def checkpoint(candidate: dict, row: dict, optimizer_steps: dict[str, int]) -> None:
        pose_m = physical_pose(candidate["pose"])
        save_npz(
            output / "endpoint.npz",
            displacement_m=candidate["u"].cpu().numpy(),
            activation_inv=candidate["q"].cpu().numpy(),
            active_cell_ids=active_source_ids,
            pose_rad_m=pose_m.cpu().numpy(),
        )
        save_torch(
            output / "checkpoint.pt",
            {
                "iteration": row["iteration"],
                "local_iteration": row["local_iteration"],
                "optimizer_step": max(optimizer_steps.values()),
                "optimizer_steps": dict(optimizer_steps),
                "activation_inv": candidate["q"].cpu(),
                "pose_normalized": candidate["pose"].cpu(),
                "pose_rad_m": pose_m.cpu(),
                "displacement_m": candidate["u"].cpu(),
                "moments": [value.cpu() for value in moments],
            },
        )
        write_json(
            output / "summary.json",
            {
                "status": "running",
                "initial": history[0],
                "final": row,
                "inverse_converged": False,
                "endpoint": record(output / "endpoint.npz"),
            },
        )

    if not cfg.resume:
        rendering = output / "rendering.npz"
        geometry = physics.full_skull.geometry
        save_npz(
            rendering,
            full_reference_points_m=physics.full_skull.full_reference_points_m,
            skin_global_ids=skin_ids,
            skin_triangles=tri,
            cranium_global_ids=geometry.cranium_global_ids,
            cranium_triangles=geometry.cranium_faces,
            mandible_global_ids=geometry.mandible_global_ids,
            mandible_triangles=geometry.mandible_faces,
            eye_global_ids=physics.full_skull.eye_global_ids,
            eye_triangles=physics.eyes.triangles,
        )
        source_records = {}
        source_dir = output / "sources"
        for directory, label in (
            (GROUP / "src", "new-neutral"),
            (SOLVERS, "solver-performance"),
            (JOINT, "joint"),
            (ROOT / "src/liblaf/apple", "apple"),
        ):
            shutil.copytree(
                directory,
                source_dir / label,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for path in source_dir.rglob("*.py"):
            source_records[str(path.relative_to(source_dir))] = sha256(path)
        if cfg.continue_optimizer_state:
            parent_protocol = json.loads(
                (initialization_checkpoint.parent / "protocol.json").read_text()
            )
            assert fit_objective is not None
            assert parent_protocol["objective_terms"] == fit_objective.protocol
        write_json(
            output / "protocol.json",
            {
                "schema": "new-neutral-expression-rigid6-inverse-v1",
                "config": cfg.model_dump(mode="json"),
                "expression_name": cfg.expression_name,
                "expression_index": target_index,
                "initialization": {
                    "method": initialization_method,
                    "checkpoint": record(initialization_checkpoint),
                    "endpoint": record(initialization_endpoint),
                    "refinement": initialization_refinement,
                    "objective_initialization": (
                        {
                            "kind": "continued_same_objective",
                            "source_protocol": record(
                                initialization_checkpoint.parent / "protocol.json"
                            ),
                            "previous_objective": json.loads(
                                (
                                    initialization_checkpoint.parent / "protocol.json"
                                ).read_text()
                            )["objective"],
                            "optimizer_moments": "retained",
                            "same_state_new_objective": False,
                            "reason": "User requested continuation with bounded numerical trial rejection; objective and physical gates retained",
                        }
                        if cfg.continue_optimizer_state
                        else {
                            "kind": "fresh_objective",
                            "objective_mode": cfg.objective_mode,
                            "optimizer_moments": "zero",
                            "neutral_q_and_pose": "zero",
                        }
                    ),
                    "optimizer_state": {
                        "continued": cfg.continue_optimizer_state,
                        "source": optimizer_state_source,
                        "initial_steps": dict(optimizer_steps),
                    },
                },
                "sources": {
                    "blendshapes": record(target_path),
                    "blendshape_manifest": record(cfg.blendshape_dir / "manifest.json"),
                    "neutral_endpoint": record(cfg.neutral_dir / "endpoint.npz"),
                    "neutral_summary": record(cfg.neutral_dir / "summary.json"),
                    "reference_repair": record(
                        cfg.reference_dir / "reference-clearance.npz"
                    ),
                },
                "rendering": {
                    "archive": record(rendering),
                    "layout": "Full runtime coordinates; each mesh uses named global IDs and local zero-based triangles.",
                },
                "parameterization": {
                    "activation": (
                        "B = I + symmetric(q); six independent dimensionless components per active muscle tetrahedron; unrestricted total activation parameters without positivity, rank or magnitude penalty; smoothness is specified in objective_terms; numerical joint increment projection is recorded separately"
                        if cfg.projection_bounded_joint
                        else "B = I + symmetric(q); six independent dimensionless components per active muscle tetrahedron; no projection, clipping, positivity, rank, magnitude or smoothness penalty"
                    ),
                    "active_cells": len(active_ids),
                    "activation_dofs": q.numel(),
                    "jaw_dofs": 6,
                    "jaw": "world rotation-vector radians and world translation metres about existing mandible pivot; no total-pose bound",
                    "normalization_scales": [math.pi / 18.0] * 3 + [0.01] * 3,
                    "adjacent_proposed_increment": {
                        "rotation_degrees": cfg.max_rotation_increment_deg,
                        "translation_m": cfg.max_translation_increment_m,
                        "rotation_interpolation": "SO(3) geodesic",
                    },
                    "jaw_pivot_m": physics.pivot_t.cpu().tolist(),
                    "packing": "xx, yy, zz, xy, yz, xz",
                },
                "optimizer_policy": {
                    "q_only_accepted_iterations": cfg.q_only_iterations,
                    "q_only": "updates q Adam state only; pose moments and counter remain frozen",
                    "joint": "updates q and pose Adam blocks with separate bias-correction counters",
                    "projection_descent_fraction": cfg.projection_descent_fraction,
                    "projection_analytic_determinants": cfg.projection_analytic_determinants,
                    "projection_feasible_witness": cfg.projection_feasible_witness,
                    "projection_witness_policy": cfg.projection_witness_policy,
                    "projection_affine_residual": cfg.projection_affine_residual,
                    "projection_bounded_joint": cfg.projection_bounded_joint,
                    "bounded_joint_policy": (
                        "joint determinant adjoints and separate residual response cached per outer iteration; each alpha projects actual q and pose increments together in Adam inverse metric with strain increment box 0.01 and metric trust ratio 2; original physical count/volume gates allow inversion identities to change"
                        if cfg.projection_bounded_joint
                        else None
                    ),
                    "affine_projection_policy": (
                        "cache once per joint outer iteration; solve actual pose increment at every alpha; separate residual intercept; full positive-cell postcheck"
                        if cfg.projection_affine_residual
                        else None
                    ),
                    "projection_gradient_model": "current adjoint; descent proposal still requires nonlinear Armijo and physical gates",
                },
                "objective": (
                    "neutral-skin-area-weighted L2 position error plus target surface-normal matching plus same-muscle tensor activation smoothness"
                    if fit_objective is not None
                    else "neutral-skin-area-weighted squared position error divided by weighted squared configured-expression target motion; no regularization"
                ),
                **(
                    {"objective_terms": fit_objective.protocol}
                    if fit_objective is not None
                    else {}
                ),
                "materials": strain_receipt,
                "tetrahedron_policy": tetrahedron_policy,
                "inversion_policy": inversion_policy,
                "boundary_policy": "original FEM IsFixed only; rigid obstacle nodes prescribed; runtime fixed and free DOF arrays verified",
                "seed_method": cfg.seed_method,
                "ipc_stiffness_mpa": kappa,
                "ipc_policy": "freeze inherited terminal adaptive stiffness for a consistent differentiable objective",
                "force_contract": {
                    "rtol": 1e-3,
                    "atol": cfg.forward_atol,
                    "effective_threshold": cfg.forward_atol,
                    "internal_solver_stopping_target": internal_forward_atol,
                    "relative_anchor": "original neutral; inherited effective force threshold; each trial uses the fixed absolute gate",
                },
                "adjoint_contract": {
                    "relative_shift": cfg.adjoint_relative_shift,
                    "shift_definition": "lambda = relative_shift * mean(abs(diag(H_ff)))",
                    "system": "(H_ff + lambda I) p = -L_f",
                    "approximate_gradient": cfg.adjoint_relative_shift > 0,
                    "relative_tolerance": cfg.adjoint_rtol,
                    "physical_forward_unchanged": True,
                },
                "source_sha256": source_records,
                "commit_enabled": False,
            },
        )
    else:
        assert protocol is not None

    assert cfg.wall_seconds is None or cfg.wall_seconds > 0
    started = time.perf_counter()
    deadline = None if cfg.wall_seconds is None else started + cfg.wall_seconds
    if cfg.deadline_iso_utc is not None:
        absolute_deadline = datetime.fromisoformat(cfg.deadline_iso_utc)
        assert absolute_deadline.tzinfo is not None
        remaining = (absolute_deadline - datetime.now(UTC)).total_seconds()
        assert remaining > 0, f"Smile deadline has passed: {cfg.deadline_iso_utc}"
        absolute_monotonic_deadline = time.perf_counter() + remaining
        deadline = (
            absolute_monotonic_deadline
            if deadline is None
            else min(deadline, absolute_monotonic_deadline)
        )
    runtime.deadline = deadline
    from expression_gradient_check import check_joint_pullback

    try:
        current = evaluate(q, pose_z, seed, True)
        if cfg.seed_method == "coupled_tangent":
            from mouthopen_coupled_seed import prepare_coupled_seed

            identity_seed, identity_receipt = prepare_coupled_seed(
                physics,
                materials,
                current["q"],
                current["q"],
                physical_pose(current["pose"]),
                physical_pose(current["pose"]),
                current["u"],
                output
                / ("resume-zero-update-check" if cfg.resume else "zero-update-check"),
                deadline=deadline,
                predictor_relative_shift=cfg.predictor_relative_shift,
                predictor_rtol=cfg.predictor_rtol,
            )
            torch.testing.assert_close(identity_seed, current["u"], rtol=0, atol=0)
            assert identity_receipt["predictor"]["rhs_norm"] == 0
            assert identity_receipt["predictor"]["maximum_free_displacement_m"] == 0
        check = check_joint_pullback(
            physics,
            runtime,
            materials,
            physical_pose,
            current["q"],
            current["pose"],
            current["u"],
            grads=current["grads"],
            objective=objective,
            control_objective=control_objective,
            epsilons=cfg.gradient_check_epsilons,
            expression_name=cfg.expression_name,
        )
        write_json(
            output
            / ("resume-gradient-check.json" if cfg.resume else "gradient-check.json"),
            check,
        )
        assert min(row["relative_error"] for row in check["q"].values()) < 1e-3, check[
            "q"
        ]
        for coordinate in check["jaw_coordinates"].values():
            assert (
                min(
                    coordinate[f"epsilon_{epsilon:.0e}"]["relative_error"]
                    for epsilon in cfg.gradient_check_epsilons
                )
                < 1e-3
            ), coordinate
    except Exception as error:
        write_json(
            output / "summary.json",
            {
                "status": "initialization_failed",
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
                "inverse_converged": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "receipt": getattr(error, "receipt", None),
                },
            },
        )
        cherries.log_output(output / "summary.json")
        raise
    if not cfg.resume:
        row = metric(
            current,
            0,
            optimizer_steps,
            "initial",
            time.perf_counter() - started,
        )
        history.append(row)
        append(output / "progress.jsonl", row)
        checkpoint(current, row, optimizer_steps)
        write_json(output / "initial-adjoint.json", current["adjoint"])

    status = "finite_budget_exhausted"
    trial_alpha_start = cfg.initial_trial_alpha
    if cfg.adaptive_trial_alpha and cfg.resume and "alpha" in history[-1]:
        trial_alpha_start = min(1.0, 2 * history[-1]["alpha"])
    previous_phase = history[-1].get("optimizer_phase", "joint")
    for iteration in range(start_iteration + 1, cfg.maximum_iterations + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            status = "time_budget_exhausted"
            break
        pose_m = physical_pose(current["pose"])
        LOG.info(
            "Iteration %d start: loss %.7g, rotation %.5f deg, translation %.5f mm",
            iteration,
            current["loss"],
            float(torch.linalg.vector_norm(pose_m[:3]) * 180 / math.pi),
            float(torch.linalg.vector_norm(pose_m[3:]) * 1000),
        )
        optimizer_phase = "q_only" if iteration <= cfg.q_only_iterations else "joint"
        if previous_phase == "q_only" and optimizer_phase == "joint":
            trial_alpha_start = cfg.initial_trial_alpha
        gq, gp = current["grads"]
        proposed_moments, next_optimizer_steps, dq, dp = block_adam_update(
            moments,
            (gq, gp),
            q_optimizer_step=optimizer_steps["q"],
            pose_optimizer_step=optimizer_steps["pose"],
            update_q=True,
            update_pose=optimizer_phase == "joint",
            learning_rate=cfg.learning_rate,
            pose_learning_rate=cfg.pose_learning_rate,
        )
        if optimizer_phase == "q_only":
            proposed_pose = current["pose"]
            increment = {"rotation_deg": 0.0, "translation_m": 0.0}
        else:
            proposed_pose, increment = bounded_pose(current["pose"], dp)
        if optimizer_phase == "joint" and cfg.seed_method == "collision_carry":
            from mouthopen_collision_seed import fixed_pose_geometry

            # A jaw feasibility limit should not also shrink the strain update.
            requested_delta = proposed_pose - current["pose"]
            for pose_backtrack in range(30):
                proposed_pose, increment = bounded_pose(
                    current["pose"], requested_delta * (0.5**pose_backtrack)
                )
                pose_geometry = fixed_pose_geometry(
                    physics, physical_pose(proposed_pose)
                )
                if pose_geometry["detF_min"] >= 0.1:
                    break
            else:
                # At an active jaw constraint, continue the independent strain
                # block using the already admitted pose instead of aborting.
                proposed_pose = current["pose"].detach().clone()
                pose_geometry = fixed_pose_geometry(
                    physics, physical_pose(proposed_pose)
                )
                assert pose_geometry["detF_min"] >= 0.1, pose_geometry
                increment = {
                    "rotation_deg": 0.0,
                    "translation_m": 0.0,
                    "pose_quality_constraint_active": True,
                }
            increment["fixed_pose_backtracks"] = pose_backtrack
            increment["fixed_pose_detF_min"] = pose_geometry["detF_min"]
        if not cfg.projection_bounded_joint:
            dp = proposed_pose - current["pose"]
        directional = float((gq * dq).sum() + (gp * dp).sum())
        if directional >= 0:
            assert not cfg.projection_bounded_joint, (
                "Bounded joint Adam center is not descending; explicit diagnosis required"
            )
            dq = -cfg.learning_rate * gq / (gq.abs() + 1e-12)
            if optimizer_phase == "joint":
                proposed_pose, increment = bounded_pose(
                    current["pose"], -cfg.pose_learning_rate * gp.sign()
                )
            else:
                proposed_pose = current["pose"]
                increment = {"rotation_deg": 0.0, "translation_m": 0.0}
            dp = proposed_pose - current["pose"]
            directional = float((gq * dq).sum() + (gp * dp).sum())
        assert directional < 0
        projection_receipt = None
        affine_cache = None
        joint_cache = None
        if cfg.projection_bounded_joint:
            from mouthopen_joint_projection import build_joint_projection_cache

            assert optimizer_phase == "joint"
            try:
                joint_cache = build_joint_projection_cache(
                    physics,
                    materials,
                    physical_pose,
                    current["q"],
                    current["pose"],
                    current["u"],
                    dq,
                    dp,
                    gq,
                    gp,
                    proposed_moments,
                    next_optimizer_steps,
                    output / "joint-projections" / f"{iteration:05d}" / "model",
                    learning_rate=cfg.learning_rate,
                    pose_learning_rate=cfg.pose_learning_rate,
                    deadline=deadline,
                )
            except Exception as error:
                summary = json.loads((output / "summary.json").read_text())
                summary["status"] = "joint_projection_cache_failed"
                summary["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "iteration": iteration,
                }
                write_json(output / "summary.json", summary)
                raise
        if cfg.projection_affine_residual:
            from mouthopen_affine_tangent import build_affine_tangent_cache

            assert optimizer_phase == "joint"
            assert cfg.projection_descent_fraction is not None
            try:
                affine_cache = build_affine_tangent_cache(
                    physics,
                    materials,
                    physical_pose,
                    current["q"],
                    dq,
                    current["pose"],
                    dp,
                    current["u"],
                    output / "affine-projections" / f"{iteration:05d}" / "model",
                    pose_gradient=gp,
                    strain_slope=float((gq * dq).sum()),
                    original_target=cfg.projection_descent_fraction * directional,
                    epsilon=cfg.projection_probe_epsilon,
                    deadline=deadline,
                )
            except Exception as error:
                summary = json.loads((output / "summary.json").read_text())
                summary["status"] = "affine_tangent_cache_failed"
                summary["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "iteration": iteration,
                }
                write_json(output / "summary.json", summary)
                raise
        if (
            not cfg.projection_affine_residual
            and cfg.project_pose_at_inversion_limit
            and optimizer_phase == "joint"
            and history[-1]["geometry"]["inverted_tetrahedra"]
            >= max(1, cfg.maximum_inverted_tetrahedra - 5)
        ):
            from mouthopen_constrained_direction import project_coupled_pose

            descent_target = (
                cfg.projection_descent_fraction * directional
                if cfg.projection_descent_fraction is not None
                else None
            )
            try:
                dp, projection_receipt = project_coupled_pose(
                    physics,
                    materials,
                    physical_pose,
                    current["q"],
                    current["q"] + dq,
                    current["pose"],
                    dp,
                    current["u"],
                    output / "pose-projections" / f"{iteration:05d}",
                    activation_threshold=cfg.projection_activation_threshold,
                    margin=cfg.projection_determinant_margin,
                    epsilon=cfg.projection_probe_epsilon,
                    deadline=deadline,
                    pose_gradient=gp if descent_target is not None else None,
                    strain_directional=(
                        float((gq * dq).sum()) if descent_target is not None else None
                    ),
                    descent_target=descent_target,
                    analytic_determinants=cfg.projection_analytic_determinants,
                    feasible_witness=cfg.projection_feasible_witness,
                    witness_policy=cfg.projection_witness_policy,
                )
            except Exception as error:
                summary = json.loads((output / "summary.json").read_text())
                summary["status"] = "pose_projection_failed"
                summary["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "iteration": iteration,
                }
                write_json(output / "summary.json", summary)
                raise
            proposed_pose, increment = bounded_pose(current["pose"], dp)
            dp = proposed_pose - current["pose"]
            directional = float((gq * dq).sum() + (gp * dp).sum())
            if descent_target is not None:
                chosen_target = projection_receipt["projection"]["descent"][
                    "joint_directional_upper_bound"
                ]
                assert directional <= chosen_target + 1e-11
            if directional >= 0:
                status = "projected_direction_not_descent"
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "success": False,
                        "failure": status,
                        "directional": directional,
                    },
                )
                break
        accepted = None
        failure_counts = {"projection": 0, "forward": 0, "adjoint": 0}
        for trial in range(cfg.max_backtracks + 1):
            alpha = trial_alpha_start * 0.5**trial
            if alpha < cfg.minimum_trial_alpha:
                status = "line_search_below_declared_resolution"
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": status,
                        "minimum_trial_alpha": cfg.minimum_trial_alpha,
                    },
                )
                break
            if deadline is not None and time.perf_counter() >= deadline:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": "declared forward wall budget exhausted before trial",
                    },
                )
                status = "time_budget_exhausted"
                break
            trial_q = current["q"] + alpha * dq
            if joint_cache is not None:
                from mouthopen_joint_direction import JointTrustRegionInfeasibleError
                from mouthopen_joint_projection import project_joint_increment

                joint_trial_dir = (
                    output
                    / "joint-projections"
                    / f"{iteration:05d}"
                    / f"trial-{trial:02d}"
                )
                try:
                    actual_dq_np, actual_dp_np, joint_receipt = project_joint_increment(
                        joint_cache,
                        alpha,
                        joint_trial_dir,
                        margin=cfg.projection_determinant_margin,
                        strain_limit=0.01,
                        descent_fraction=cfg.projection_descent_fraction,
                        trust_ratio=2.0,
                        deadline=deadline,
                    )
                    actual_dq = torch.as_tensor(
                        actual_dq_np,
                        device=current["q"].device,
                        dtype=current["q"].dtype,
                    )
                    actual_dp = torch.as_tensor(
                        actual_dp_np,
                        device=current["pose"].device,
                        dtype=current["pose"].dtype,
                    )
                    trial_q = current["q"] + actual_dq
                    trial_pose, trial_increment = bounded_pose(
                        current["pose"], actual_dp
                    )
                    torch.testing.assert_close(
                        trial_pose, current["pose"] + actual_dp, rtol=0, atol=0
                    )
                    actual_slope = float(
                        (gq * (trial_q - current["q"])).sum()
                        + (gp * (trial_pose - current["pose"])).sum()
                    )
                    assert actual_slope < 0
                    assert (
                        actual_slope <= joint_receipt["chosen_increment_target"] + 1e-11
                    )
                    joint_record = record(joint_trial_dir / "summary.json")
                    projection_receipt = {
                        "active_cell_count": joint_receipt["active_cell_count"],
                        "minimum_linearized_projected_J": joint_receipt[
                            "minimum_linearized_projected_J"
                        ],
                        "projection": joint_receipt["projection"],
                        "bounded_joint_trial": joint_record,
                    }
                    append(
                        output / "joint-projections.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "actual_q_and_pose_increments_already_scaled": True,
                            "actual_joint_increment_slope": actual_slope,
                            "chosen_increment_target": joint_receipt[
                                "chosen_increment_target"
                            ],
                            "receipt": joint_record,
                        },
                    )
                except JointTrustRegionInfeasibleError as error:
                    append(
                        output / "trials.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "success": False,
                            "failure": "joint_projection_outside_declared_trust_region",
                            "message": str(error),
                            "receipt": record(joint_trial_dir / "summary.json"),
                            "physical_candidate_evaluated": False,
                        },
                    )
                    continue
                except Exception as error:
                    failure = {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "type": type(error).__name__,
                        "message": str(error),
                        "receipt": record(joint_trial_dir / "summary.json")
                        if (joint_trial_dir / "summary.json").is_file()
                        else None,
                    }
                    append(
                        output / "trials.jsonl",
                        {
                            **failure,
                            "success": False,
                            "failure": "joint_projection_failed",
                        },
                    )
                    if known_projection_failure(error, joint_trial_dir):
                        failure_counts["projection"] += 1
                        append(
                            output / "recoverable-failures.jsonl",
                            {
                                **failure,
                                "phase": "joint_projection",
                                "count": failure_counts["projection"],
                                "limit": cfg.maximum_projection_failures,
                                "next_alpha": alpha * 0.5,
                                "state_and_optimizer_committed": False,
                            },
                        )
                        if (
                            failure_counts["projection"]
                            <= cfg.maximum_projection_failures
                        ):
                            continue
                        status = "joint_projection_retry_exhausted"
                        break
                    summary = json.loads((output / "summary.json").read_text())
                    summary["status"] = "joint_projection_failed"
                    summary["failure"] = failure
                    write_json(output / "summary.json", summary)
                    raise
            elif affine_cache is not None:
                from mouthopen_affine_direction import project_affine_increment

                affine_trial_dir = (
                    output
                    / "affine-projections"
                    / f"{iteration:05d}"
                    / f"trial-{trial:02d}"
                )
                try:
                    actual_dp_np, affine_receipt = project_affine_increment(
                        affine_cache,
                        alpha,
                        affine_trial_dir,
                        margin=cfg.projection_determinant_margin,
                        activation_threshold=cfg.projection_activation_threshold,
                    )
                    actual_dp = torch.as_tensor(
                        actual_dp_np,
                        device=current["pose"].device,
                        dtype=current["pose"].dtype,
                    )
                    trial_pose, trial_increment = bounded_pose(
                        current["pose"], actual_dp
                    )
                    torch.testing.assert_close(
                        trial_pose, current["pose"] + actual_dp, rtol=0, atol=0
                    )
                    affine_actual_slope = float(
                        (gq * (alpha * dq)).sum()
                        + (gp * (trial_pose - current["pose"])).sum()
                    )
                    assert affine_actual_slope < 0
                    assert (
                        affine_actual_slope
                        <= affine_receipt["chosen_increment_target"] + 1e-11
                    )
                    affine_record = record(affine_trial_dir / "summary.json")
                    projection_receipt = {
                        "active_cell_count": len(
                            affine_receipt["constraint_passes"][-1][
                                "active_original_ids"
                            ]
                        ),
                        "minimum_linearized_projected_J": min(
                            affine_receipt["minimum_seed_positive_J"],
                            affine_receipt["minimum_affine_positive_J"],
                        ),
                        "projection": affine_receipt["projection"],
                        "affine_trial": affine_record,
                    }
                    append(
                        output / "affine-projections.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "actual_pose_increment_already_scaled": True,
                            "actual_joint_increment_slope": affine_actual_slope,
                            "chosen_increment_target": affine_receipt[
                                "chosen_increment_target"
                            ],
                            "receipt": affine_record,
                        },
                    )
                except Exception as error:
                    failure = {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "type": type(error).__name__,
                        "message": str(error),
                        "receipt": (
                            record(affine_trial_dir / "summary.json")
                            if (affine_trial_dir / "summary.json").is_file()
                            else None
                        ),
                    }
                    append(
                        output / "trials.jsonl",
                        {
                            **failure,
                            "success": False,
                            "failure": "affine_projection_failed",
                        },
                    )
                    summary = json.loads((output / "summary.json").read_text())
                    summary["status"] = "affine_projection_failed"
                    summary["failure"] = failure
                    write_json(output / "summary.json", summary)
                    raise
            elif optimizer_phase == "joint":
                trial_pose, trial_increment = bounded_pose(current["pose"], alpha * dp)
            else:
                trial_pose = current["pose"]
                trial_increment = {"rotation_deg": 0.0, "translation_m": 0.0}
            trial_directional = float(
                (gq * (trial_q - current["q"])).sum()
                + (gp * (trial_pose - current["pose"])).sum()
            )
            if joint_cache is not None:
                assert trial_directional < 0
                assert (
                    trial_directional
                    <= joint_receipt["chosen_increment_target"] + 1e-11
                )
            if affine_cache is not None:
                assert trial_directional < 0
                assert (
                    trial_directional
                    <= affine_receipt["chosen_increment_target"] + 1e-11
                )
            if trial_directional >= 0:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": "non_descent_capped_trial",
                        "directional": trial_directional,
                        "proposed_increment": trial_increment,
                    },
                )
                continue
            try:
                candidate = evaluate(
                    trial_q,
                    trial_pose,
                    current["u"],
                    False,
                    source_q=current["q"],
                    source_pose=current["pose"],
                )
            except ForwardConvergenceError as error:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": str(error),
                        "receipt": error.receipt,
                        "initializer_output_dir": getattr(error, "seed_dir", None),
                    },
                )
                if deadline is not None and time.perf_counter() >= deadline:
                    status = "time_budget_exhausted"
                    break
                failure_counts["forward"] += 1
                append(
                    output / "recoverable-failures.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "phase": "trial_forward",
                        "failure": str(error),
                        "receipt": error.receipt,
                        "count": failure_counts["forward"],
                        "limit": cfg.maximum_forward_failures,
                        "next_alpha": alpha * 0.5,
                        "state_and_optimizer_committed": False,
                    },
                )
                if failure_counts["forward"] > cfg.maximum_forward_failures:
                    status = "forward_retry_exhausted"
                    break
                continue
            passed = candidate["loss"] <= current["loss"] + 1e-4 * trial_directional
            append(
                output / "trials.jsonl",
                {
                    "iteration": iteration,
                    "trial": trial,
                    "alpha": alpha,
                    "success": True,
                    "accepted": passed,
                    "loss": candidate["loss"],
                    "forward": candidate["forward"],
                    "proposed_increment": trial_increment,
                    "directional": trial_directional,
                },
            )
            if passed:
                try:
                    accepted = evaluate(
                        candidate["q"], candidate["pose"], candidate["u"], True
                    )
                except ForwardConvergenceError as error:
                    append(
                        output / "trials.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "success": False,
                            "failure": str(error),
                            "receipt": error.receipt,
                            "phase": "accepted_gradient_re_evaluation",
                        },
                    )
                    if deadline is not None and time.perf_counter() >= deadline:
                        status = "time_budget_exhausted"
                        break
                    failure_counts["forward"] += 1
                    append(
                        output / "recoverable-failures.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "phase": "accepted_gradient_forward",
                            "failure": str(error),
                            "receipt": error.receipt,
                            "count": failure_counts["forward"],
                            "limit": cfg.maximum_forward_failures,
                            "next_alpha": alpha * 0.5,
                            "state_and_optimizer_committed": False,
                        },
                    )
                    if failure_counts["forward"] > cfg.maximum_forward_failures:
                        status = "forward_retry_exhausted"
                        break
                    continue
                except Exception as error:
                    if known_adjoint_failure(
                        error, runtime.last_sparse_adjoint, cfg.adjoint_rtol
                    ):
                        failure_counts["adjoint"] += 1
                        append(
                            output / "recoverable-failures.jsonl",
                            {
                                "iteration": iteration,
                                "trial": trial,
                                "alpha": alpha,
                                "phase": "accepted_gradient_adjoint",
                                "type": type(error).__name__,
                                "message": str(error),
                                "sparse_adjoint": copy.deepcopy(
                                    runtime.last_sparse_adjoint
                                ),
                                "candidate_forward": candidate["forward"],
                                "count": failure_counts["adjoint"],
                                "limit": cfg.maximum_adjoint_failures,
                                "next_alpha": alpha * 0.5,
                                "state_and_optimizer_committed": False,
                            },
                        )
                        runtime.drop_warm_adjoint(cfg.expression_name)
                        if failure_counts["adjoint"] <= cfg.maximum_adjoint_failures:
                            continue
                        status = "adjoint_retry_exhausted"
                        break
                    summary = json.loads((output / "summary.json").read_text())
                    summary["status"] = "accepted_gradient_failed"
                    summary["failure"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "iteration": iteration,
                        "trial": trial,
                        "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
                    }
                    write_json(output / "summary.json", summary)
                    cherries.log_output(output / "summary.json")
                    raise
                assert math.isfinite(accepted["loss"])
                if accepted["loss"] > current["loss"] + 1e-4 * trial_directional:
                    append(
                        output / "trials.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "success": True,
                            "accepted": False,
                            "phase": "gradient_re_evaluation_armijo",
                            "loss": accepted["loss"],
                            "armijo_bound": current["loss"] + 1e-4 * trial_directional,
                            "state_and_optimizer_committed": False,
                        },
                    )
                    accepted = None
                    continue
                break
        if status == "time_budget_exhausted":
            break
        if accepted is None:
            if status not in (
                "time_budget_exhausted",
                "line_search_below_declared_resolution",
                "joint_projection_retry_exhausted",
                "forward_retry_exhausted",
                "adjoint_retry_exhausted",
            ):
                status = "line_search_stalled"
            break
        current = accepted
        moments = proposed_moments
        optimizer_steps = next_optimizer_steps
        row = metric(
            current,
            iteration,
            optimizer_steps,
            optimizer_phase,
            time.perf_counter() - started,
            alpha=alpha,
            line_search_backtracks=trial,
            trial_alpha_start=trial_alpha_start,
            pose_projection=(
                None
                if projection_receipt is None or joint_cache is not None
                else {
                    "active_cell_count": projection_receipt["active_cell_count"],
                    "minimum_linearized_projected_J": projection_receipt[
                        "minimum_linearized_projected_J"
                    ],
                    "projection": projection_receipt["projection"],
                    **(
                        {"affine_trial": projection_receipt["affine_trial"]}
                        if "affine_trial" in projection_receipt
                        else {}
                    ),
                }
            ),
            joint_projection=projection_receipt if joint_cache is not None else None,
            proposed_increment=increment,
            accepted_increment=trial_increment,
            adjoint_relative_residual=current["adjoint"]["relative_residual"],
        )
        history.append(row)
        if cfg.convergence_patience:
            row["convergence_monitor"] = stationarity_candidate(
                history,
                patience=cfg.convergence_patience,
                loss_rtol=cfg.convergence_loss_rtol,
                gradient_rtol=cfg.convergence_gradient_rtol,
                gradient_atol=cfg.convergence_gradient_atol,
            )
        append(output / "progress.jsonl", row)
        checkpoint(current, row, optimizer_steps)
        cherries.set_step(iteration)
        cherries.log_metrics(
            {
                key: row[key]
                for key in (
                    "loss",
                    "fit_rms_mm",
                    "pose_rotation_degrees",
                    "pose_translation_mm",
                    "force_norm_n",
                    "activation_rms",
                )
            }
        )
        if fit_objective is not None:
            cherries.log_metrics({"loss_components": row["loss_components"]})
        LOG.info(
            "Accepted %d: fit %.6f mm, rotation %.5f deg, translation %.5f mm, force %.6g N, inverted %d",
            iteration,
            row["fit_rms_mm"],
            row["pose_rotation_degrees"],
            row["pose_translation_mm"],
            row["force_norm_n"],
            row["geometry"]["inverted_tetrahedra"],
        )
        if cfg.adaptive_trial_alpha:
            trial_alpha_start = min(1.0, 2 * alpha)
        previous_phase = optimizer_phase
        if (
            cfg.convergence_patience
            and row["convergence_monitor"]["candidate_requires_independent_audit"]
        ):
            status = "stationarity_candidate_requires_audit"
            break
    summary = json.loads((output / "summary.json").read_text())
    summary["status"] = status
    summary["inverse_converged"] = False
    write_json(output / "summary.json", summary)
    cherries.log_output(output / "summary.json")
    cherries.log_output(output / "progress.jsonl")
    cherries.log_output(output / "endpoint.npz")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
