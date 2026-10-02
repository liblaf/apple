"""Benchmark exact-HVP Newton-CG from a frozen full-face warm state."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import SharedFieldParameters, activation_stresses_mpa
from joint_physics import BULK_NAMES, JointPhysics

from liblaf import cherries
from liblaf.apple.inverse._diff_forward import _AdjointProblem
from liblaf.apple.solvers.linalg.cupy import CupyCG

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    initial_checkpoint: Path = GROUP / "data/neutral-prestress-010/best-admissible.pt"
    contact_spec: Path = GROUP / "data/contact/config.json"
    base_equilibrium: Path = (
        GROUP / "data/face-gradient-validation-contact/base-equilibrium.pt"
    )
    pncg_checks: Path = GROUP / "data/face-gradient-validation-contact/checks.json"
    output_dir: Path = cherries.output("refinement-benchmark", mkdir=True)
    perturbation_step: float = 0.003
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    linear_rtol: float = 1e-8
    linear_max_iterations: int = 10000
    max_newton_steps: int = 5
    armijo_coefficient: float = 1e-4
    max_line_search_steps: int = 40


def make_state(model: Any, displacement: torch.Tensor) -> Any:
    state = model.State(u=displacement.detach().clone())
    if model.collision is not None:
        state.collision = model.collision.state_at(state.u)
    return state


def contact_receipt(model: Any, state: Any) -> dict[str, Any] | None:
    if model.collision is None:
        return None
    return model.collision.diagnostics(state.collision, state.u)


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.perturbation_step == 0.003
    assert cfg.forward_rtol == 1e-6
    assert cfg.forward_atol == 1e-12
    assert cfg.adjoint_rtol == 1e-7
    assert cfg.linear_rtol in {1e-8, 1e-3}
    assert cfg.max_newton_steps == 5
    assert 0 < cfg.armijo_coefficient < 1
    assert cfg.max_line_search_steps == 40

    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    configure_cuda()
    initial = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=False)
    base_u = torch.load(
        cfg.base_equilibrium, map_location="cpu", weights_only=False
    ).to(device="cuda")
    contact_config = json.loads(cfg.contact_spec.read_text())
    materials_spec = initial["materials"]["materials"]
    skin = materials_spec["skin"]
    physics = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials_spec[name]["young_mpa"] for name in BULK_NAMES},
        bulk_nu={name: materials_spec[name]["poisson"] for name in BULK_NAMES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=10000,
        contact_config=contact_config,
    )
    assert base_u.shape == (len(physics.points), 3)
    shared = SharedFieldParameters(initial["materials"])
    with torch.no_grad():
        shared.coefficients.copy_(initial["shared_coefficients"])
        direction = torch.zeros_like(shared.coefficients)
        direction[:6] = direction.new_tensor([0.4, -0.3, 0.2, 0.1, -0.2, 0.3])
        shared.coefficients.add_(cfg.perturbation_step * direction)
    q = torch.zeros((len(physics.ids), 6))
    q[:, :3] = 0.01
    jaw = torch.zeros(6)
    model = physics.runtime.forward.model
    perturbed_materials = physics.materials(
        shared.bulk_stresses_mpa(),
        shared.skin_resultant_n_per_m(),
        shared.skin_stiffness_multiplier(),
        activation_stresses_mpa(q, shared.activation_reference_mpa),
    )
    fixed_values = physics.boundary(jaw).detach().clone()

    # Matched reference solve: same model, perturbation, frozen base seed, and
    # convergence tolerances as the corresponding 09 plus-side solve.
    pncg_started = time.perf_counter()
    pncg_u = physics.runtime.primal(perturbed_materials, fixed_values, base_u)
    torch.cuda.synchronize()
    pncg_wall_seconds = time.perf_counter() - pncg_started
    pncg_receipt_local = copy.deepcopy(physics.runtime.last_forward)
    pncg_fit_loss = float(physics.fit_loss(pncg_u, 0))
    pncg_metrics = physics.metrics(pncg_u, target_index=0)
    torch.save(pncg_u.detach().cpu(), output / "pncg-displacement.pt")

    # Restore the exact perturbed model while starting Newton independently from
    # the same frozen base state, never from the PNCG result.
    model.set_materials(perturbed_materials)
    model.dof_map.fixed_values = fixed_values
    # Re-impose current fixed values exactly on the frozen warm state.
    free = model.dof_map.to_free(base_u).detach().clone()
    state = make_state(model, model.dof_map.to_full(free))
    problem = physics.runtime.forward.problem
    initial_gradient = problem.grad(state)
    initial_gradient_norm = float(torch.linalg.vector_norm(initial_gradient))
    force_threshold = max(cfg.forward_atol, cfg.forward_rtol * initial_gradient_norm)
    assert math.isfinite(initial_gradient_norm)
    assert initial_gradient_norm > 0
    initial_energy = float(problem.fun(state))
    initial_contact = contact_receipt(model, state)
    assert initial_contact is not None
    assert initial_contact["contact_numerically_valid"], initial_contact

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    converged = initial_gradient_norm <= force_threshold
    failure: dict[str, Any] | None = None
    for iteration in range(cfg.max_newton_steps):
        gradient = problem.grad(state)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        if gradient_norm <= force_threshold:
            converged = True
            break
        energy = float(problem.fun(state))
        system = _AdjointProblem(b=-gradient, model=model, model_state=state)
        linear_started = time.perf_counter()
        solution = CupyCG(
            maxiter=cfg.linear_max_iterations,
            rtol=cfg.linear_rtol,
            atol=0.0,
        ).solve(system, torch.zeros_like(gradient))
        torch.cuda.synchronize()
        linear_seconds = time.perf_counter() - linear_started
        direction_free = solution.params.detach()
        linear_residual = float(
            torch.linalg.vector_norm(system.matvec(direction_free) - system.b)
        )
        linear_relative_residual = linear_residual / gradient_norm
        slope = float(torch.dot(gradient, direction_free))
        linear_ok = bool(
            solution.success
            and torch.isfinite(direction_free).all()
            and math.isfinite(linear_relative_residual)
            and linear_relative_residual <= cfg.linear_rtol * 1.05
            and math.isfinite(slope)
            and slope < 0
        )
        if not linear_ok:
            failure = {
                "stage": "linear_solve_or_descent",
                "iteration": iteration,
                "linear_result": str(solution.result),
                "linear_relative_residual": linear_relative_residual,
                "slope": slope,
            }
            break

        ccd_fraction = float(problem.max_step_size(state, direction_free))
        if not math.isfinite(ccd_fraction) or not 0 < ccd_fraction <= 1:
            failure = {
                "stage": "ccd",
                "iteration": iteration,
                "ccd_fraction": ccd_fraction,
            }
            break
        # Use a safety margin only when CCD is active; preserve a full Newton step
        # when the entire proposed path is already collision-free.
        alpha = 1.0 if ccd_fraction == 1.0 else 0.99 * ccd_fraction
        accepted = False
        trial_receipts: list[dict[str, Any]] = []
        for line_step in range(cfg.max_line_search_steps + 1):
            candidate_free = free + alpha * direction_free
            candidate = make_state(model, model.dof_map.to_full(candidate_free))
            candidate_energy = float(problem.fun(candidate))
            armijo_rhs = energy + cfg.armijo_coefficient * alpha * slope
            candidate_contact = contact_receipt(model, candidate)
            valid_contact = bool(
                candidate_contact is not None
                and candidate_contact["contact_numerically_valid"]
            )
            trial_receipts.append(
                {
                    "line_step": line_step,
                    "alpha": alpha,
                    "energy": candidate_energy,
                    "armijo_rhs": armijo_rhs,
                    "finite": math.isfinite(candidate_energy),
                    "contact_valid": valid_contact,
                }
            )
            if (
                math.isfinite(candidate_energy)
                and candidate_energy <= armijo_rhs
                and valid_contact
            ):
                free = candidate_free.detach().clone()
                state = candidate
                accepted = True
                break
            alpha *= 0.5
        record = {
            "iteration": iteration,
            "energy": energy,
            "gradient_norm": gradient_norm,
            "force_threshold": force_threshold,
            "linear_result": str(solution.result),
            "linear_relative_residual": linear_relative_residual,
            "linear_seconds": linear_seconds,
            "slope": slope,
            "ccd_fraction": ccd_fraction,
            "accepted": accepted,
            "accepted_alpha": alpha if accepted else None,
            "line_search_steps": line_step,
            "trials": trial_receipts,
        }
        records.append(record)
        write_json(output / "trace.json", records)
        if not accepted:
            failure = {
                "stage": "armijo",
                "iteration": iteration,
                "trials": trial_receipts,
            }
            break

    terminal_gradient = problem.grad(state)
    terminal_gradient_norm = float(torch.linalg.vector_norm(terminal_gradient))
    converged = terminal_gradient_norm <= force_threshold
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    terminal_contact = contact_receipt(model, state)
    assert terminal_contact is not None
    terminal_metrics = physics.metrics(state.u, target_index=0)
    terminal_fit_loss = float(physics.fit_loss(state.u, 0))
    torch.save(state.u.detach().cpu(), output / "terminal-displacement.pt")

    pncg_rows = json.loads(cfg.pncg_checks.read_text())
    matches = [
        row
        for row in pncg_rows
        if row["name"] == "fat" and row["step"] == cfg.perturbation_step
    ]
    assert len(matches) == 1, matches
    pncg = copy.deepcopy(matches[0]["plus_forward"])
    pncg_comparison: dict[str, Any] = {
        "source": str(cfg.pncg_checks.resolve()),
        "source_sha256": sha256(cfg.pncg_checks),
        "side": "plus",
        "receipt": pncg,
        "speed_ratio_pncg_over_newton": pncg_wall_seconds / elapsed,
        "nonlinear_step_ratio_pncg_over_newton": (pncg["steps"] / max(1, len(records))),
        "matched_local_run": {
            "wall_seconds": pncg_wall_seconds,
            "receipt": pncg_receipt_local,
            "fit_loss": pncg_fit_loss,
            "metrics": pncg_metrics,
            "displacement_path": str((output / "pncg-displacement.pt").resolve()),
        },
        "solution_comparison_available": True,
        "solution_displacement_l2_m": float(torch.linalg.vector_norm(state.u - pncg_u)),
        "solution_displacement_inf_m": float((state.u - pncg_u).abs().max()),
        "newton_fit_loss": terminal_fit_loss,
        "pncg_fit_loss": pncg_fit_loss,
        "fit_loss_absolute_difference": abs(terminal_fit_loss - pncg_fit_loss),
    }

    success = bool(
        failure is None
        and converged
        and terminal_contact["contact_numerically_valid"]
        and terminal_metrics["inverted_tetrahedra"] == 0
    )
    receipt = {
        "schema": "joint-newton-refinement-benchmark-v1",
        "success": success,
        "status": "converged" if success else "failed",
        "scope": (
            "one full-face fat-baseline plus perturbation at h=0.003 from the "
            "frozen q=0.01 contact equilibrium; benchmark, not solver adoption"
        ),
        "contention": (
            "run concurrently with the remaining 09 directional validation and "
            "an unrelated user GPU process; wall timing is contended"
        ),
        "operator": (
            "exact FEM plus IPC barrier Hessian-vector products through "
            "_AdjointProblem; absolute Hessian diagonal is preconditioning only"
        ),
        "linear_solver": {
            "implementation": "CupyCG",
            "rtol": cfg.linear_rtol,
            "atol": 0.0,
            "max_iterations": cfg.linear_max_iterations,
            "failure_policy": "fail visibly; no MINRES or PNCG fallback",
        },
        "nonlinear_solver": {
            "max_newton_steps": cfg.max_newton_steps,
            "armijo_coefficient": cfg.armijo_coefficient,
            "max_line_search_steps": cfg.max_line_search_steps,
            "ccd_safety": "0.99 times restrictive CCD fraction; 1 for unrestricted path",
        },
        "force_criterion": {
            "rtol": cfg.forward_rtol,
            "atol": cfg.forward_atol,
            "reference": "initial warm-state free-force norm, fixed for this run",
            "initial_gradient_norm": initial_gradient_norm,
            "threshold": force_threshold,
            "terminal_gradient_norm": terminal_gradient_norm,
            "met": converged,
        },
        "initial_energy": initial_energy,
        "terminal_energy": float(problem.fun(state)),
        "terminal_fit_loss": terminal_fit_loss,
        "elapsed_seconds": elapsed,
        "newton_steps": len(records),
        "failure": failure,
        "initial_contact": initial_contact,
        "terminal_contact": terminal_contact,
        "terminal_metrics": terminal_metrics,
        "pncg_comparison": pncg_comparison,
        "provenance": {
            "prepared_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
            "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
            "initial_checkpoint_sha256": sha256(cfg.initial_checkpoint),
            "contact_spec_sha256": sha256(cfg.contact_spec),
            "base_equilibrium_sha256": sha256(cfg.base_equilibrium),
        },
        "trace": records,
    }
    write_json(output / "summary.json", receipt)
    cherries.log_metrics(
        {
            "refinement/elapsed_seconds": elapsed,
            "refinement/newton_steps": len(records),
            "refinement/terminal_force_norm": terminal_gradient_norm,
            "refinement/pncg_over_newton_time": pncg_wall_seconds / elapsed,
        }
    )
    cherries.log_output(output)
    assert success, receipt
    COMPLETED = True


if __name__ == "__main__":
    os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
    os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
