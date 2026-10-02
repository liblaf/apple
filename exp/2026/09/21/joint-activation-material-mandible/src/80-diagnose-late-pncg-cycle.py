"""Diagnose the late run-009 PNCG cycle from a frozen accepted checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.solvers.optim.pncg._direction import dai_kou, dai_kou_plus


class Config(cherries.BaseConfig):
    protocol: Path = GROUP / "data/simple-skin-forward-009/protocol.json"
    checkpoint: Path = GROUP / "data/simple-skin-forward-009/checkpoint-step-08000.npz"
    trace: Path = GROUP / "data/simple-skin-forward-009/trace.jsonl"
    steps_per_arm: int = 20
    output_dir: Path = GROUP / "data/late-pncg-cycle-diagnostic-002"


class ExactCurvatureForwardProblem(ForwardProblem):
    """Use the exact model HVP only for PNCG's directional curvature."""

    def hess_quad(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        return torch.dot(direction, self.hess_prod(state, direction))


def load_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("simple_forward_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denominator) if float(denominator) else math.nan


def evaluate(
    problem: ForwardProblem,
    state: Any,
    params: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    problem.update(state, params)
    return float(problem.fun(state)), problem.grad(state)


def directional_checks(
    problem: ForwardProblem,
    state: Any,
    params: torch.Tensor,
    *,
    damping_factor: float,
) -> dict[str, Any]:
    central_energy, gradient = evaluate(problem, state, params)
    hess_diag = problem.hess_diag(state)
    diagonal_mean = torch.mean(torch.abs(hess_diag))
    preconditioner = torch.reciprocal(
        torch.abs(hess_diag) + damping_factor * diagonal_mean
    )
    direction = -preconditioner * gradient
    slope = torch.dot(gradient, direction)
    raw_hess_quad = problem.hess_quad(state, direction)
    exact_hp = problem.hess_prod(state, direction)
    exact_hess_quad = torch.dot(direction, exact_hp)
    damping_quad = damping_factor * diagonal_mean * torch.dot(direction, direction)
    damped_hess_quad = raw_hess_quad + damping_quad
    direction_inf = torch.linalg.vector_norm(direction, ord=torch.inf)
    rows = []
    for displacement_m in (1e-8, 3e-8, 1e-7, 3e-7, 1e-6):
        alpha = direction.new_tensor(displacement_m) / direction_inf
        plus_energy, plus_gradient = evaluate(
            problem, state, params + alpha * direction
        )
        minus_energy, minus_gradient = evaluate(
            problem, state, params - alpha * direction
        )
        energy_fd = (plus_energy - minus_energy) / (2 * float(alpha))
        hp_fd = (plus_gradient - minus_gradient) / (2 * alpha)
        rows.append(
            {
                "displacement_inf_m": displacement_m,
                "alpha": float(alpha),
                "energy_directional_fd": energy_fd,
                "energy_gradient_relative_error": abs(energy_fd - float(slope))
                / max(abs(energy_fd), abs(float(slope)), 1e-300),
                "exact_hvp_fd_relative_error": float(
                    torch.linalg.vector_norm(hp_fd - exact_hp)
                    / torch.linalg.vector_norm(exact_hp)
                ),
                "fd_hess_quad": float(torch.dot(direction, hp_fd)),
            }
        )
    profile = []
    for displacement_m in (0.0, 2.5e-7, 5e-7, 1e-6, 2e-6):
        alpha = direction.new_tensor(displacement_m) / direction_inf
        energy, _ = evaluate(problem, state, params + alpha * direction)
        profile.append(
            {
                "displacement_inf_m": displacement_m,
                "alpha": float(alpha),
                "actual_energy_change": energy - central_energy,
                "raw_quadratic_prediction": float(
                    alpha * slope + 0.5 * alpha**2 * raw_hess_quad
                ),
                "damped_quadratic_prediction": float(
                    alpha * slope + 0.5 * alpha**2 * damped_hess_quad
                ),
            }
        )
    evaluate(problem, state, params)
    return {
        "central_energy": central_energy,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "damping_factor": damping_factor,
        "diagonal_abs_mean": float(diagonal_mean),
        "direction_inf_norm": float(direction_inf),
        "slope": float(slope),
        "raw_hess_quad": float(raw_hess_quad),
        "exact_hess_quad": float(exact_hess_quad),
        "damping_quad": float(damping_quad),
        "damped_hess_quad": float(damped_hess_quad),
        "raw_vs_exact_hess_quad_relative_difference": abs(
            float(raw_hess_quad - exact_hess_quad)
        )
        / max(abs(float(raw_hess_quad)), abs(float(exact_hess_quad)), 1e-300),
        "finite_differences": rows,
        "line_profile": profile,
    }


def run_arm(
    runner: Any,
    problem: ForwardProblem,
    state: Any,
    initial_params: torch.Tensor,
    *,
    armijo: float,
    steps: int,
) -> dict[str, Any]:
    problem.update(state, initial_params)
    optimizer = runner.StrictPncg(
        hess_damping=runner.StrictPncg.HessianDamping(initial=0.001),
        line_search=runner.StrictLineSearch(
            armijo=armijo, max_steps=60, max_step_norm=0.0005
        ),
    )
    opt_state = optimizer.init(problem, state, initial_params)
    rows = []
    previous_preconditioner = None
    for _ in range(steps):
        gradient = problem.grad(state)
        hess_diag = problem.hess_diag(state)
        factor_before = float(opt_state.hess_damping_state.factor)
        diagonal_mean = torch.mean(torch.abs(hess_diag))
        preconditioner = torch.reciprocal(
            torch.abs(hess_diag) + factor_before * diagonal_mean
        )
        restart = opt_state.step == 0 or not opt_state.line_search_state.ok
        if restart:
            beta_raw = beta_plus = 0.0
        else:
            beta_raw = float(
                dai_kou(
                    g=gradient,
                    g_prev=opt_state.grad,
                    P=preconditioner,
                    p_prev=opt_state.direction,
                )
            )
            beta_plus = float(
                dai_kou_plus(
                    g=gradient,
                    g_prev=opt_state.grad,
                    P=preconditioner,
                    p_prev=opt_state.direction,
                )
            )
        steepest = -preconditioner * gradient
        expected_direction = (
            steepest if restart else steepest + beta_plus * opt_state.direction
        )
        if float(torch.dot(expected_direction, gradient)) >= 0:
            expected_direction = steepest
        p_variation = (
            None
            if previous_preconditioner is None
            else float(
                torch.linalg.vector_norm(preconditioner - previous_preconditioner)
                / torch.linalg.vector_norm(previous_preconditioner)
            )
        )
        direction_angle = cosine(expected_direction, steepest)
        previous_angle = (
            None
            if opt_state.direction is None
            else cosine(expected_direction, opt_state.direction)
        )
        energy_before = float(problem.fun(state))
        optimizer.step(problem, state, opt_state)
        accepted_gradient = problem.grad(state)
        alpha = float(opt_state.line_search_state.alpha)
        damped_hess_quad = float(
            opt_state.hess_quad
            + factor_before
            * diagonal_mean
            * torch.dot(opt_state.direction, opt_state.direction)
        )
        predicted_decrease = float(
            -alpha * opt_state.slope - 0.5 * alpha**2 * damped_hess_quad
        )
        actual_decrease = energy_before - float(opt_state.fun)
        rows.append(
            {
                "step": opt_state.step,
                "force_before": float(torch.linalg.vector_norm(gradient)),
                "force_after": float(torch.linalg.vector_norm(accepted_gradient)),
                "beta_raw": beta_raw,
                "beta_plus": beta_plus,
                "direction_cosine_preconditioned_steepest": direction_angle,
                "direction_cosine_previous": previous_angle,
                "preconditioner_relative_change": p_variation,
                "direction_inf_norm": float(
                    torch.linalg.vector_norm(opt_state.direction, ord=torch.inf)
                ),
                "accepted_displacement_inf_m": alpha
                * float(torch.linalg.vector_norm(opt_state.direction, ord=torch.inf)),
                "slope": float(opt_state.slope),
                "raw_hess_quad": float(opt_state.hess_quad),
                "damped_hess_quad": damped_hess_quad,
                "alpha": alpha,
                "line_search_step": int(opt_state.line_search_state.step),
                "actual_decrease": actual_decrease,
                "predicted_decrease": predicted_decrease,
                "actual_to_predicted_decrease": actual_decrease
                / max(predicted_decrease, 1e-300),
                "armijo_decrease_ratio": actual_decrease
                / max(-alpha * float(opt_state.slope), 1e-300),
                "damping_factor_before": factor_before,
                "damping_factor_after": float(opt_state.hess_damping_state.factor),
            }
        )
        previous_preconditioner = preconditioner
    return {
        "armijo": armijo,
        "steps": rows,
        "initial_force": rows[0]["force_before"],
        "terminal_force": rows[-1]["force_after"],
        "minimum_force": min(row["force_after"] for row in rows),
        "maximum_beta_plus": max(row["beta_plus"] for row in rows),
        "minimum_direction_cosine_preconditioned_steepest": min(
            row["direction_cosine_preconditioned_steepest"] for row in rows
        ),
    }


def main(cfg: Config) -> None:
    if cfg.steps_per_arm != 20:
        message = "the declared bounded comparison uses exactly 20 steps per arm"
        raise ValueError(message)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    protocol = json.loads(cfg.protocol.read_text())
    archived_runner = (
        cfg.protocol.parent / "sources/experiment/68-run-simple-skin-forward.py"
    )
    runner = load_runner(archived_runner)
    runner.configure_cuda()
    checkpoint_receipt = json.loads(cfg.checkpoint.with_suffix(".json").read_text())
    assert sha256(cfg.checkpoint) == checkpoint_receipt["sha256"]
    inputs = protocol["inputs"]
    prepared = runner.PreparedInputs.load(
        Path(inputs["prepared_npz"]), Path(inputs["prepared_manifest"])
    )
    geometry_receipt = inputs["geometry"]["geometry"]
    geometry = runner.load_full_skull_geometry(
        Path(geometry_receipt["geometry_path"]),
        Path(geometry_receipt["audit_path"]),
    )
    admission = json.loads(Path(inputs["admission_path"]).read_text())
    canonical = runner.research_informed_material_config()["materials"]
    common_nu = float(protocol["mechanics"]["poisson_ratios"]["fat"])
    contact = protocol["mechanics"]["contact"]
    physics = runner.FullSkullJointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={
            name: canonical[name]["young_mpa"] for name in runner.BULK_TISSUES
        },
        bulk_nu=dict.fromkeys(runner.BULK_TISSUES, common_nu),
        skin_young_mpa=canonical["skin"]["reference_map"]["young_mpa"],
        skin_nu=common_nu,
        thickness_m=canonical["skin"]["thickness_m"],
        full_skull_geometry=geometry,
        full_skull_admission=admission,
        full_skull_contact_config=contact,
        rtol=protocol["solver"]["rtol"],
        atol=protocol["solver"]["atol"],
        max_steps=protocol["solver"]["max_steps"],
        forward_method="pncg",
        adjoint_rtol=1e-7,
    )
    skin, _ = runner.load_skin_field(
        Path(inputs["skin_field_path"]),
        Path(inputs["skin_field_manifest_path"]),
        prepared=prepared,
        expected_triangles=physics.skin_tri,
    )
    model = physics.runtime.forward.model
    model.set_materials(runner.heterogeneous_materials(physics, skin))
    pose = torch.zeros(6, dtype=torch.float64, device="cuda")
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    with np.load(cfg.checkpoint, allow_pickle=False) as archive:
        fem_u = np.asarray(archive["displacement_m"], dtype=np.float64)
    full_u = physics.full_skull.extend_seed(torch.as_tensor(fem_u, device="cuda"), pose)
    params = model.dof_map.to_free(full_u).detach().clone()
    state = physics.runtime.forward.state
    problem = ForwardProblem(model)
    problem.update(state, params)

    trace_text = cfg.trace.read_text()
    trace_rows = [json.loads(line) for line in trace_text.splitlines()]
    next_row = next(row for row in trace_rows if row.get("step") == 8001)
    late_damping = float(next_row["hessian_damping_factor"])
    derivative = directional_checks(problem, state, params, damping_factor=late_damping)
    exact_problem = ExactCurvatureForwardProblem(model)
    exact_problem.update(state, params)
    arm = run_arm(
        runner,
        exact_problem,
        state,
        params,
        armijo=0.25,
        steps=cfg.steps_per_arm,
    )
    summary = {
        "schema": "joint-late-pncg-cycle-diagnostic-v2",
        "success": True,
        "status": "completed_bounded_checkpoint_clone_diagnostic",
        "scope": "one frozen derivative check and one 20-step exact-HVP-curvature, Armijo-0.25 cloned PNCG arm; active runs009/010 were not modified",
        "inputs": {
            "protocol": str(cfg.protocol.resolve()),
            "protocol_sha256": sha256(cfg.protocol),
            "trace": str(cfg.trace.resolve()),
            "trace_sha256_at_read": hashlib.sha256(trace_text.encode()).hexdigest(),
            "trace_bytes_at_read": len(trace_text.encode()),
            "checkpoint": str(cfg.checkpoint.resolve()),
            "checkpoint_sha256": sha256(cfg.checkpoint),
            "archived_runner": str(archived_runner.resolve()),
            "archived_runner_sha256": sha256(archived_runner),
        },
        "late_trace_reference": next_row,
        "directional_derivatives": derivative,
        "cloned_exact_curvature_armijo_0p25": arm,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
