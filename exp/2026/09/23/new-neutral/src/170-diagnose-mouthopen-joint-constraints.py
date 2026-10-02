"""Test a frozen joint strain and pose determinant projection."""

# ruff: noqa: C901, E402, PLR0912, PLR0915, SLF001, TRY300
from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import ipctk
import numpy as np
import torch
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from joint_common import ProfileJoint, sha256, write_json
from joint_coupled_predictor import audit_coupled_motion, rotation_sagitta
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from mouthopen_block_optimizer import block_adam_update
from mouthopen_coupled_seed import _damped_equilibrium_tangent, _mandible_arc_radius
from mouthopen_pose_projection import (
    determinant_ratio,
)
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-012"
    output_dir: Path = GROUP / "data/mouthopen-joint-constraints-001"
    maximum_determinant_rows: int = 16
    maximum_seed_attempts: int = 3
    relative_shift: float = 1e-5
    epsilon: float = 0.00005
    determinant_margin: float = 1e-6
    activation_threshold: float = 0.05
    descent_fraction: float = 0.1
    candidate_alpha: float = 0.125
    case_wall_seconds: float = 600.0
    internal_atol: float = 1e-9
    ipc_threads: int = 4


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict) -> Path:
    path = Path(item["path"]).resolve()
    assert record(path) == item, path
    return path


def main(cfg: Config) -> None:
    assert cfg.relative_shift == 1e-5
    assert 5 <= cfg.maximum_determinant_rows <= 16
    assert 1 <= cfg.maximum_seed_attempts <= 3
    assert cfg.candidate_alpha == 0.125
    assert cfg.internal_atol == 1e-9
    assert 0 < cfg.epsilon < 1
    assert 0 < cfg.determinant_margin < cfg.activation_threshold
    assert 0 < cfg.descent_fraction <= 1
    assert 0 < cfg.candidate_alpha <= 1
    assert 0 < cfg.case_wall_seconds <= 600
    source, output = cfg.source_run.resolve(), cfg.output_dir.resolve()
    assert not output.exists(), output
    refs = {
        name: record(source / name)
        for name in (
            "summary.json",
            "protocol.json",
            "endpoint.npz",
            "checkpoint.pt",
            "independent-audit.json",
        )
    }
    protocol = json.loads((source / "protocol.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    audit = json.loads((source / "independent-audit.json").read_text())
    assert summary["status"] != "running"
    assert summary["endpoint"] == refs["endpoint.npz"]
    assert audit["valid_forward"]
    for item in audit["inputs"].values():
        bound(item)
    for key in ("summary", "protocol", "endpoint"):
        assert (
            audit["inputs"][key]
            == refs[key + (".npz" if key == "endpoint" else ".json")]
        )
    assert protocol["force_contract"]["atol"] == 1e-8
    policy = protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] == 100
    assert policy["maximum_inverted_rest_volume_fraction"] == 0.0001
    assert policy["orientation_floor"] is None
    checkpoint = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer_steps"] == {"q": 268, "pose": 268}
    assert checkpoint["iteration"] == summary["final"]["iteration"]
    assert checkpoint["optimizer_steps"] == summary["final"]["optimizer_steps"]
    with np.load(source / "endpoint.npz", allow_pickle=False) as saved:
        for key in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        active_ids = saved["active_cell_ids"].copy()
    reference_path = bound(protocol["sources"]["reference_repair"])
    neutral_path = bound(protocol["sources"]["neutral_endpoint"])
    target_path = bound(protocol["sources"]["blendshapes"])
    model_dir = source / "affine-projections/00002/model"
    trial_dir = source / "affine-projections/00002/trial-01"
    cached_summary = json.loads((model_dir / "summary.json").read_text())
    assert cached_summary["status"] == "model_ready_for_trial_projection"
    assert cached_summary["relative_shift"] == cfg.relative_shift
    assert cached_summary["epsilon"] == cfg.epsilon == 5e-5
    cache_path = bound(cached_summary["cache"])
    assert cache_path == model_dir / "cache/coefficients.npz"
    cached_trial = json.loads((trial_dir / "summary.json").read_text())
    assert cached_trial["status"] == "certified_linear_increment"
    assert cached_trial["alpha"] == cfg.candidate_alpha
    assert cached_trial["margin"] == cfg.determinant_margin
    assert cached_trial["cache"] == cached_summary["cache"]
    with np.load(model_dir / "source-and-response.npz") as saved:
        for key in ("activation_inv", "pose_normalized", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        dq_np = saved["dq"].copy()
    calibration_refs = {
        "model_summary": record(model_dir / "summary.json"),
        "model_source_state": record(model_dir / "source-and-response.npz"),
        "coefficients": record(cache_path),
        "trial_summary": record(trial_dir / "summary.json"),
        "trial_input_receipt": record(trial_dir / "input-receipt.json"),
        "trial_constraints": record(trial_dir / "constraints-00.npz"),
        "trial_predictions": record(trial_dir / "predictions.npz"),
    }
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    physics, _ = build_rebased_physics(reference_path.parent, inverse=True)
    model = physics.runtime.forward.model
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.r_[
        np.repeat(isfixed, 3),
        np.ones(model.dof_map.n_full - 3 * len(isfixed), dtype=bool),
    ]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu(), np.flatnonzero(~expected_fixed)
    )
    assert not isfixed[np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)].any()
    install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    expected_exclusion = dict(protocol["tetrahedron_policy"])
    expected_exclusion.pop("neutral_free_equation_proof")
    assert exclusion == expected_exclusion
    baseline = model.get_materials()
    with np.load(
        neutral_path.parent / "active-strain-fields.npz", allow_pickle=False
    ) as saved:
        for key, saved_key in (
            ("activation_inv", "skin_activation_inverse"),
            ("mu", "skin_mu_mpa"),
            ("thickness", "skin_thickness_m"),
        ):
            np.testing.assert_array_equal(baseline["skin"][key].cpu(), saved[saved_key])
    np.testing.assert_array_equal(active_ids, physics.base.retained_active_cell_ids)
    q = checkpoint["activation_inv"].clone().cuda()
    pose = checkpoint["pose_normalized"].clone().cuda()
    old_u = checkpoint["displacement_m"].clone().cuda()
    scales = torch.tensor(
        [math.pi / 18] * 3 + [0.01] * 3, device=pose.device, dtype=pose.dtype
    )
    torch.testing.assert_close(
        pose * scales, checkpoint["pose_rad_m"].cuda(), rtol=0, atol=0
    )
    fixed = physics.boundary(pose * scales).detach().clone()
    torch.testing.assert_close(
        old_u.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=5e-16
    )
    moments = [value.clone().cuda() for value in checkpoint["moments"]]
    assert len(moments) == 4

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, physics.base.active_t, value)
        return result

    old_materials = materials(q)
    kappa = float(protocol["ipc_stiffness_mpa"])
    assert kappa == 1.3544
    collision = model.collision
    assert collision is not None
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    with np.load(target_path, allow_pickle=False) as data:
        index = list(data["expression_names"]).index("MouthOpen")
        skin_ids = data["skin_global_ids"].copy()
        triangles = data["skin_triangles"].copy()
        neutral = data["new_neutral_points_m"].copy()
        target = data["target_points_m"][index].copy()
    xyz = neutral[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()
    scale2 = float(np.sum(weights * np.sum((target - neutral) ** 2, axis=1)))
    assert scale2 > 0
    target_u = torch.as_tensor(target - physics.points[skin_ids], device=q.device)
    weights_t = torch.as_tensor(weights, device=q.device)

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (weights_t[:, None] * (u[skin_ids] - target_u).square()).sum() / scale2

    reference = np.asarray(physics.points)
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)
    retained_tets = np.asarray(physics.base.tets)[retained_ids]
    old_np = old_u.cpu().numpy()
    old_all_j = determinant_ratio(reference, retained_tets, old_np)
    active = (old_all_j > 0) & (old_all_j <= cfg.activation_threshold)
    assert active.any()
    output.mkdir(parents=True)
    source_manifest = {}
    for label, source_root in (
        ("new-neutral", GROUP / "src"),
        ("joint", JOINT),
        ("solvers", SOLVERS),
        ("apple", ROOT / "src/liblaf/apple"),
    ):
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(source_root)
            destination = output / "sources" / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            assert sha256(path) == sha256(destination)
            source_manifest[str(Path(label) / relative)] = record(destination)
    with np.load(cache_path) as saved:
        np.testing.assert_array_equal(saved["original_ids"], retained_ids)
        np.testing.assert_array_equal(saved["old_j"], old_all_j)
        residual_delta = saved["residual_delta_j"].copy()
        cached_qdj = saved["q_delta_j"].copy()
        cached_pose_a = saved["pose_jacobian"].copy()
    evidence_dir = GROUP / "data/mouthopen-saved-defect-seed-001"
    evidence_refs = {
        name: record(evidence_dir / name)
        for name in ("summary.json", "protocol.json", "corrected.npz")
    }
    evidence = json.loads((evidence_dir / "summary.json").read_text())
    assert evidence["status"] == "candidate_rejected_diagnostic"
    evidence_protocol = json.loads((evidence_dir / "protocol.json").read_text())
    assert evidence_protocol["source_refs"] == refs
    write_json(
        output / "protocol.json",
        {
            "schema": "mouthopen-frozen-joint-constraints-v1",
            "config": cfg.model_dump(mode="json"),
            "source_refs": refs,
            "calibration_refs": calibration_refs,
            "prior_corrector_evidence": evidence_refs,
            "script": record(Path(__file__)),
            "source_snapshot": source_manifest,
            "tetrahedron_policy": exclusion,
            "isfixed_exact": True,
            "skin_fields_exact": True,
            "ipc_stiffness_mpa": kappa,
            "original_acceptance_force": 1e-8,
            "internal_force_target": cfg.internal_atol,
            "inversion_policy": policy,
            "source_optimizer_steps": checkpoint["optimizer_steps"],
            "direction_policy": "Fresh objective and determinant adjoints through one frozen shifted source graph. Project actual joint q/normalized-pose increment in D^-1 metric, D=block_rate/(sqrt(proposed_bias_corrected_v)+1e-12), center alpha times fresh source Adam. Protect seed and residual affine determinant forecasts and one certified descent halfspace. Add only new actual-seed violations, at most16 determinant rows and3 seeds. Never adopt optimizer or physical state.",
            "trust_policy": "Reject if metric correction exceeds twice baseline metric norm or maxabs strain increment exceeds max(.01, twice baseline maxabs strain increment). No clipping and no extra pose cap.",
            "budget_scope": "600 seconds for one unchanged-source forward graph, objective/determinant adjoints, CPU projections, up to3 actual seeds and one nonlinear corrector; physical rebuild excluded. Diagnostic serialization included. Failure metrics may finish after budget, but no new solve starts.",
            "inverse_adopted": False,
        },
    )
    positive = old_all_j > 0
    initial_ids = set(retained_ids[positive & (old_all_j < 1e-4)].tolist())
    initial_ids.update((18514, 155249, 598977, 656201, 688235))
    lookup = {int(value): i for i, value in enumerate(retained_ids)}
    assert all(positive[lookup[value]] for value in initial_ids)
    assert len(initial_ids) <= cfg.maximum_determinant_rows
    selected = sorted(initial_ids)
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.internal_atol,
        adjoint_rtol=1e-7,
        adjoint_relative_shift=cfg.relative_shift,
        newton_max_steps=3000,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    started = time.perf_counter()
    deadline = started + cfg.case_wall_seconds
    runtime.deadline = deadline
    row = {
        "status": "running",
        "inverse_adopted": False,
        "attempts": [],
        "gradients": [],
    }
    stage = "source_graph"
    write_json(output / "summary.json", row)

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            message = "Declared joint diagnostic budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    def save_state(
        path: Path, u: torch.Tensor, target_q: torch.Tensor, target_pose: torch.Tensor
    ) -> np.ndarray:
        value = u.detach().cpu().numpy()
        j = determinant_ratio(reference, retained_tets, value)
        np.savez_compressed(
            path,
            displacement_m=value,
            activation_inv=target_q.detach().cpu().numpy(),
            pose_normalized=target_pose.detach().cpu().numpy(),
            pose_rad_m=(target_pose.detach() * scales).cpu().numpy(),
            original_retained_ids=retained_ids,
            old_det_f=old_all_j,
            det_f=j,
            observed_affine_det_f=j + residual_delta,
            newly_inverted_original_ids=retained_ids[positive & (j <= 0)],
            sign_changed_original_ids=retained_ids[positive != (j > 0)],
        )
        return j

    try:
        check_budget()
        restore_source()
        source_q = q.detach().clone().requires_grad_()
        source_pose = pose.detach().clone().requires_grad_()
        source_u = runtime.solve(
            materials(source_q),
            physics.boundary(source_pose * scales),
            old_u,
            key="FrozenJointDeterminants",
        )
        torch.testing.assert_close(source_u.detach(), old_u, rtol=0, atol=0)
        source_forward = copy.deepcopy(runtime.last_forward)
        assert source_forward["pncg"]["steps"] == 0
        assert source_forward["newton"]["steps"] == 0
        row["source_forward"] = source_forward
        gq, gp = torch.autograd.grad(
            objective(source_u), (source_q, source_pose), retain_graph=True
        )
        row["objective_adjoint"] = copy.deepcopy(runtime.last_adjoint)
        row["objective_sparse_adjoint"] = copy.deepcopy(runtime.last_sparse_adjoint)
        write_json(output / "objective-adjoint.json", row["objective_sparse_adjoint"])
        assert runtime.last_sparse_adjoint["shifted_relative_residual"] <= 1e-7
        assert runtime.last_sparse_adjoint["native_shifted_relative_residual"] <= 1e-7
        rates = (
            float(protocol["config"]["learning_rate"]),
            float(protocol["config"]["pose_learning_rate"]),
        )
        assert rates == (0.002, 0.1)
        proposed, counts, fresh_dq, fresh_dp = block_adam_update(
            moments,
            (gq, gp),
            q_optimizer_step=268,
            pose_optimizer_step=268,
            update_q=True,
            update_pose=True,
            learning_rate=rates[0],
            pose_learning_rate=rates[1],
        )
        assert counts == {"q": 269, "pose": 269}
        cached_delta_difference = fresh_dq.detach().cpu().numpy() - dq_np
        row["fresh_dq_vs_cached"] = {
            "maximum_absolute_difference": float(np.max(abs(cached_delta_difference))),
            "l2_difference": float(np.linalg.norm(cached_delta_difference)),
        }
        dq_metric = rates[0] / ((proposed[1] / (1 - 0.999**269)).sqrt() + 1e-12)
        dp_metric = rates[1] / ((proposed[3] / (1 - 0.999**269)).sqrt() + 1e-12)
        metric = torch.cat((dq_metric.flatten(), dp_metric.flatten())).detach()
        x0 = (
            cfg.candidate_alpha
            * torch.cat((fresh_dq.flatten(), fresh_dp.flatten())).detach()
        )
        gradient = torch.cat((gq.flatten(), gp.flatten())).detach()
        baseline_slope = float(gradient @ x0)
        assert baseline_slope < 0
        descent_target = cfg.descent_fraction * baseline_slope
        assert bool(torch.isfinite(metric).all())
        assert bool((metric > 0).all())
        baseline_norm = float(torch.linalg.vector_norm(x0 / metric.sqrt()))
        q_limit = max(0.01, 2 * float(x0[: q.numel()].abs().max()))
        np.savez_compressed(
            output / "objective-and-adam.npz",
            gq=gq.cpu().numpy(),
            gp=gp.cpu().numpy(),
            dq=fresh_dq.cpu().numpy(),
            dp=fresh_dp.cpu().numpy(),
            x0=x0.cpu().numpy(),
            diagonal_inverse_metric=metric.cpu().numpy(),
            cached_dq=dq_np,
            source_q=q.cpu().numpy(),
            source_pose=pose.cpu().numpy(),
        )
        row.update(
            baseline_slope=baseline_slope,
            descent_target=descent_target,
            baseline_metric_norm=baseline_norm,
            strain_absolute_limit=q_limit,
        )
        gradients = {}
        comparisons = {}
        cached_dq_tensor = torch.as_tensor(dq_np, device=q.device, dtype=q.dtype)
        points_t = torch.as_tensor(reference, device=q.device, dtype=q.dtype)

        def add_gradient(original_id: int) -> None:
            check_budget()
            restore_source()
            index = lookup[original_id]
            tet = retained_tets[index]
            vertex_ids = torch.as_tensor(tet, device=q.device, dtype=torch.long)
            rest = points_t[vertex_ids]
            deformed = rest + source_u[vertex_ids]
            rest_matrix = (rest[1:] - rest[0]).T
            value = torch.linalg.det((deformed[1:] - deformed[0]).T) / torch.linalg.det(
                rest_matrix
            )
            np.testing.assert_allclose(
                float(value.detach()), old_all_j[index], rtol=1e-9, atol=1e-12
            )
            runtime.drop_warm_adjoint("FrozenJointDeterminants")
            jq, jp = torch.autograd.grad(
                value, (source_q, source_pose), retain_graph=True
            )
            full = torch.cat((jq.flatten(), jp.flatten())).detach()
            gradients[original_id] = full
            comparison = {
                "original_id": original_id,
                "source_J": old_all_j[index],
                "q_gradient_dot_cached_dq": float((jq * cached_dq_tensor).sum()),
                "cached_q_directional": float(cached_qdj[index]),
                "pose_gradient": jp.detach().cpu().tolist(),
                "cached_pose_jacobian": cached_pose_a[index].tolist(),
                "q_directional_difference": float((jq * cached_dq_tensor).sum())
                - float(cached_qdj[index]),
                "pose_maximum_difference": float(
                    np.max(abs(jp.detach().cpu().numpy() - cached_pose_a[index]))
                ),
                "adjoint": copy.deepcopy(runtime.last_adjoint),
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
            }
            q_comparison_scale = max(
                abs(comparison["q_gradient_dot_cached_dq"]),
                abs(comparison["cached_q_directional"]),
            )
            pose_comparison_scale = np.maximum(
                abs(jp.detach().cpu().numpy()), abs(cached_pose_a[index])
            )
            q_threshold = 1e-7 + 5e-3 * q_comparison_scale
            pose_threshold = 1e-7 + 5e-3 * pose_comparison_scale
            comparison.update(
                comparison_model="Same relative-shift1e-5 derivative; cached finite tangent epsilon5e-5. No claim of unshifted derivative accuracy.",
                comparison_absolute_tolerance=1e-7,
                comparison_relative_tolerance=5e-3,
                q_comparison_scale=q_comparison_scale,
                q_comparison_threshold=q_threshold,
                pose_comparison_scale=pose_comparison_scale.tolist(),
                pose_comparison_threshold=pose_threshold.tolist(),
                q_comparison_passed=abs(comparison["q_directional_difference"])
                <= q_threshold,
                pose_comparison_passed=bool(
                    np.all(
                        abs(jp.detach().cpu().numpy() - cached_pose_a[index])
                        <= pose_threshold
                    )
                ),
            )
            comparisons[original_id] = comparison
            np.savez_compressed(
                output / f"gradient-{original_id}.npz",
                q_gradient=jq.detach().cpu().numpy(),
                pose_gradient=jp.detach().cpu().numpy(),
            )
            write_json(output / f"gradient-{original_id}.json", comparison)
            row["gradients"].append(comparison)
            write_json(output / "summary.json", row)
            assert comparison["sparse_adjoint"]["shifted_relative_residual"] <= 1e-7
            assert (
                comparison["sparse_adjoint"]["native_shifted_relative_residual"] <= 1e-7
            )
            assert comparison["q_comparison_passed"], (
                "q directional derivative comparison failed"
            )
            assert comparison["pose_comparison_passed"], (
                "pose derivative comparison failed"
            )

        for original_id in selected:
            stage = f"determinant_adjoint_{original_id}"
            add_gradient(original_id)
        admitted = False
        for attempt in range(cfg.maximum_seed_attempts):
            stage = f"joint_projection_{attempt}"
            check_budget()
            directory = output / f"attempt-{attempt:02d}"
            directory.mkdir()
            matrix = torch.stack([gradients[value] for value in selected] + [-gradient])
            indices = np.asarray([lookup[value] for value in selected])
            lower = np.r_[
                cfg.determinant_margin
                - old_all_j[indices]
                - np.minimum(residual_delta[indices], 0),
                -descent_target,
            ]
            gram = (matrix * metric) @ matrix.T
            centered = (matrix @ x0).cpu().numpy()
            np.savez_compressed(
                directory / "qp-inputs.npz",
                original_ids=np.asarray(selected),
                lower=lower,
                gram=gram.cpu().numpy(),
                center_rows=centered,
                original_descent_target=descent_target,
            )
            write_json(
                directory / "input-receipt.json",
                {
                    "qp_inputs": record(directory / "qp-inputs.npz"),
                    "objective_and_metric": record(output / "objective-and-adam.npz"),
                    "determinant_gradients": {
                        str(value): record(output / f"gradient-{value}.npz")
                        for value in selected
                    },
                    "source_cache": record(cache_path),
                    "row_order": [*selected, "negative_objective_gradient"],
                },
            )
            correction_multipliers, certificate = solve_dual(
                gram.cpu().numpy(),
                lower - centered,
                receipt_path=directory / "qp-certificate.json",
            )
            write_json(directory / "qp-certificate.json", certificate)
            multipliers_t = torch.as_tensor(
                correction_multipliers, device=q.device, dtype=q.dtype
            )
            increment = x0 + metric * (matrix.T @ multipliers_t)
            actual_rows = (matrix @ increment).cpu().numpy()
            actual_slack = actual_rows - lower
            actual_slope = float(gradient @ increment)
            correction_norm = float(
                torch.linalg.vector_norm((increment - x0) / metric.sqrt())
            )
            delta_q = increment[: q.numel()].reshape_as(q)
            delta_pose = increment[q.numel() :].reshape_as(pose)
            np.savez_compressed(
                directory / "joint-increment.npz",
                delta_q=delta_q.cpu().numpy(),
                delta_pose=delta_pose.cpu().numpy(),
                multipliers=correction_multipliers,
                actual_rows=actual_rows,
                actual_slack=actual_slack,
            )
            attempt_row = {
                "attempt": attempt,
                "selected_ids": selected.copy(),
                "actual_slope": actual_slope,
                "descent_target": descent_target,
                "minimum_linear_slack": float(actual_slack.min()),
                "correction_metric_norm": correction_norm,
                "trust_norm_limit": 2 * baseline_norm,
                "maximum_absolute_delta_q": float(delta_q.abs().max()),
                "strain_absolute_limit": q_limit,
            }
            row["attempts"].append(attempt_row)
            write_json(directory / "summary.json", attempt_row)
            assert actual_slack.min() >= -1e-10
            assert actual_slope <= descent_target + 1e-12
            assert correction_norm <= 2 * baseline_norm
            assert float(delta_q.abs().max()) <= q_limit
            next_q, next_pose = q + delta_q, pose + delta_pose
            next_fixed = physics.boundary(next_pose * scales).detach().clone()
            restore_source()
            check_budget()
            stage = f"actual_seed_{attempt}"
            seed, seed_receipt = _damped_equilibrium_tangent(
                physics,
                old_materials,
                materials(next_q),
                old_u,
                next_fixed,
                deadline=deadline,
                predictor_relative_shift=cfg.relative_shift,
                predictor_rtol=1e-7,
                material_changed=True,
            )
            seed_j = save_state(directory / "seed.npz", seed, next_q, next_pose)
            write_json(directory / "predictor.json", seed_receipt)
            angle = float(
                (
                    Rotation.from_rotvec((next_pose * scales).cpu().numpy()[:3])
                    * Rotation.from_rotvec((pose * scales).cpu().numpy()[:3]).inv()
                ).magnitude()
            )
            assert 0 <= angle <= math.pi
            pivot = torch.as_tensor(physics.pivot_t, device=q.device, dtype=q.dtype)
            radius, _ = _mandible_arc_radius(
                physics, collision, pivot, fixed, next_fixed
            )
            motion = audit_coupled_motion(
                collision,
                old_u,
                seed,
                rotation_margin_m=rotation_sagitta(radius, angle),
            )
            seed_geometry = geometry_metrics(physics, seed)
            observed_affine = seed_j + residual_delta
            violated = retained_ids[
                positive & ((seed_j <= 0) | (observed_affine <= 0))
            ].tolist()
            attempt_row.update(
                seed_geometry=seed_geometry,
                motion=motion,
                predictor=seed_receipt,
                minimum_positive_seed_J=float(seed_j[positive].min()),
                minimum_positive_observed_affine_J=float(
                    observed_affine[positive].min()
                ),
                strict_model_margin_minimum_slack=float(
                    np.minimum(seed_j, observed_affine)[positive].min()
                    - cfg.determinant_margin
                ),
                violated_original_positive_ids=violated,
            )
            write_json(directory / "summary.json", attempt_row)
            write_json(output / "summary.json", row)
            assert motion["admitted"]
            if violated:
                assert not set(violated).intersection(selected), (
                    "Already-constrained determinant fails actual positivity; no identical reprojection"
                )
                additions = sorted(set(violated) - set(selected))
                assert len(selected) + len(additions) <= cfg.maximum_determinant_rows
                assert attempt + 1 < cfg.maximum_seed_attempts, (
                    "Declared actual-seed attempt budget exhausted"
                )
                for original_id in additions:
                    stage = f"added_determinant_adjoint_{original_id}"
                    add_gradient(original_id)
                selected.extend(additions)
                continue
            assert seed_geometry["inverted_tetrahedra"] <= 100
            assert seed_geometry["inverted_rest_volume_fraction"] <= 1e-4
            admitted = True
            break
        assert admitted
        stage = "single_nonlinear_corrector"
        check_budget()
        restore_source()
        if hasattr(runtime, "last_failed_displacement"):
            del runtime.last_failed_displacement
        try:
            corrected = runtime.primal(
                materials(next_q), next_fixed, seed.detach().clone()
            )
        except ForwardConvergenceError as error:
            row["corrector_failure"] = {"message": str(error), "receipt": error.receipt}
            assert hasattr(runtime, "last_failed_displacement")
            corrected = runtime.last_failed_displacement.detach().clone()
        corrected_j = save_state(output / "corrected.npz", corrected, next_q, next_pose)
        row["forward"] = copy.deepcopy(runtime.last_forward)
        runtime.deadline = None
        stage = "fresh_corrected_metrics"
        model.set_materials(materials(next_q))
        model.dof_map.fixed_values = next_fixed.detach().clone()
        state = model.State(u=corrected.detach().clone())
        state.collision = collision.state_at(state.u)
        force = float(
            torch.linalg.vector_norm(
                FeasibleExpressionProblem(model=model, collision_step_safety=0.9).grad(
                    state
                )
            )
        )
        contact = runtime._contact_gate(state)
        geometry = geometry_metrics(physics, corrected)
        loss, old_loss = float(objective(corrected)), float(objective(old_u))
        armijo = old_loss + 1e-4 * actual_slope
        gates = (
            force <= 1e-8
            and geometry["inverted_tetrahedra"] <= 100
            and geometry["inverted_rest_volume_fraction"] <= 1e-4
            and contact["receipt"]["contact_numerically_valid"]
            and contact["no_intersections"]
            and contact["minimum_active_gap_at_least_buffer"]
        )
        success = (
            "corrector_failure" not in row
            and gates
            and force <= cfg.internal_atol
            and loss < old_loss
            and loss <= armijo
        )
        row.update(
            status="candidate_passed_diagnostic"
            if success
            else "candidate_rejected_diagnostic",
            success=success,
            raw_force=force,
            contact=contact,
            geometry=geometry,
            loss=loss,
            source_loss=old_loss,
            fit_rms_mm=math.sqrt(loss * scale2) * 1000,
            source_fit_rms_mm=math.sqrt(old_loss * scale2) * 1000,
            armijo_threshold=armijo,
            armijo_passed=loss <= armijo,
            original_acceptance_gates_met=gates,
            internal_force_target_met=force <= cfg.internal_atol,
            newly_inverted_original_ids=retained_ids[
                positive & (corrected_j <= 0)
            ].tolist(),
        )
        check_budget()
    except (ForwardConvergenceError, RuntimeError, AssertionError, ValueError) as error:
        row["status"] = "diagnostic_failed"
        row["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ForwardConvergenceError):
            row["failure"]["receipt"] = error.receipt
        LOG.exception("Frozen joint diagnostic failed at %s", stage)
    finally:
        runtime.deadline = None
        restore_source()
        for name, item in refs.items():
            assert record(source / name) == item
        for item in (*calibration_refs.values(), *evidence_refs.values()):
            bound(item)
        for original, value in zip(checkpoint["moments"], moments, strict=True):
            torch.testing.assert_close(original, value.cpu(), rtol=0, atol=0)
        row["source_hashes_unchanged"] = True
        row["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", row)
        cherries.log_output(output / "protocol.json")
        cherries.log_output(output / "summary.json")
    assert row["status"] != "diagnostic_failed", row


def solve_dual(
    gram: np.ndarray, deficit: np.ndarray, *, receipt_path: Path | None = None
) -> tuple[np.ndarray, dict]:
    """Certify a projection dual after row and right-hand-side normalization."""
    receipt = {
        "status": "running",
        "solver": "HiGHS feasibility and SLSQP nonnegative dual",
    }

    def persist() -> None:
        if receipt_path is not None:
            write_json(receipt_path, receipt)

    try:
        gram = np.asarray(gram, dtype=np.float64)
        deficit = np.asarray(deficit, dtype=np.float64)
        assert np.isfinite(gram).all()
        assert np.isfinite(deficit).all()
        np.testing.assert_allclose(gram, gram.T, rtol=1e-12, atol=1e-18)
        scale = np.sqrt(np.diag(gram))
        assert np.all(scale > 0)
        assert np.isfinite(scale).all()
        normalized = gram / scale[:, None] / scale[None, :]
        normalized = (normalized + normalized.T) / 2
        h = deficit / scale
        rhs_scale = float(np.max(abs(h)))
        eigenvalues = np.linalg.eigvalsh(normalized)
        receipt.update(
            row_scales=scale.tolist(),
            rhs_scale=rhs_scale,
            normalized_deficit=h.tolist(),
            normalized_gram_eigenvalues=eigenvalues.tolist(),
        )
        persist()
        assert eigenvalues.min() >= -1e-10
        if rhs_scale == 0:
            receipt.update(
                status="certified",
                certified=True,
                method="zero_deficit",
                multipliers=np.zeros(len(h)).tolist(),
                original_unit_slack=np.zeros(len(h)).tolist(),
            )
            persist()
            return np.zeros(len(h)), receipt
        hhat = h / rhs_scale
        lp = linprog(
            np.zeros(len(h)),
            A_ub=-normalized,
            b_ub=-hhat,
            bounds=[(None, None)] * len(h),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        receipt.update(lp_status=int(lp.status), lp_message=lp.message)
        persist()
        assert lp.success, (
            f"Joint row-space feasibility unresolved: {lp.status}: {lp.message}"
        )
        witness_slack = (normalized @ lp.x - hhat) * rhs_scale * scale
        receipt.update(
            lp_normalized_witness=lp.x.tolist(),
            lp_original_unit_slack=witness_slack.tolist(),
        )
        persist()
        assert witness_slack.min() >= -1e-10
        result = minimize(
            lambda v: 0.5 * v @ normalized @ v - hhat @ v,
            np.zeros(len(h)),
            jac=lambda v: normalized @ v - hhat,
            bounds=[(0, None)] * len(h),
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        lam = np.asarray(result.x)
        receipt.update(
            solver_status=int(result.status),
            solver_message=str(result.message),
            solver_iterations=int(result.nit),
            raw_normalized_multipliers=lam.tolist(),
        )
        persist()
        assert result.success, result.message

        def certified(value: np.ndarray) -> bool:
            slack = normalized @ value - hhat
            return bool(
                np.isfinite(value).all()
                and np.min(value) >= 0
                and np.min(slack * rhs_scale * scale) >= -1e-10
                and np.min(slack) >= -1e-8
                and np.max(abs(value * slack))
                <= 1e-8 * max(1, float(abs(hhat @ value)))
            )

        active = np.flatnonzero(lam > max(1e-12, 1e-10 * float(lam.max())))
        polished = lam.copy()
        if len(active):
            polished[:] = 0
            polished[active] = np.linalg.lstsq(
                normalized[np.ix_(active, active)], hhat[active], rcond=1e-12
            )[0]
        used_polish = certified(polished)
        if used_polish:
            lam = polished
        slack = normalized @ lam - hhat
        multipliers = lam * rhs_scale / scale
        receipt.update(
            active_set_polish_used=used_polish,
            multipliers=multipliers.tolist(),
            minimum_dual_multiplier=float(multipliers.min()),
            original_unit_slack=(slack * rhs_scale * scale).tolist(),
            normalized_stationarity_minimum=float(slack.min()),
            normalized_maximum_complementarity=float(np.max(abs(lam * slack))),
            original_dual_primal_gap=float(lam @ slack) * rhs_scale**2,
        )
        persist()
        assert certified(lam), "Joint dual KKT certificate failed"
        receipt.update(status="certified", certified=True)
        persist()
        return multipliers, receipt
    except (AssertionError, ValueError, RuntimeError) as error:
        receipt.update(
            status="failed",
            certified=False,
            failure={"type": type(error).__name__, "message": str(error)},
        )
        persist()
        raise


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
