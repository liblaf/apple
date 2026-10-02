# ruff: noqa: E402, PLR0915, PT018
"""Run the legacy strict PNCG algorithm on the current neutral reference model."""

from __future__ import annotations

import copy
import faulthandler
import importlib.util
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries

PATH = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "neutral_newton_driver", PATH.with_name("10-run-neutral.py")
)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

from joint_equilibrium import StrictLineSearch, StrictPncg
from joint_expression_equilibrium import AcceptedForcePncg

LOG = logging.getLogger(__name__)


class Config(base.Config):
    output_dir: Path = base.GROUP / "data/forward-pncg-200-001"
    comparison_dir: Path = base.GROUP / "data/forward-contact-200-001"
    max_steps: int = 200
    line_search_armijo: float = 0.25
    line_search_max_backtracks: int = 60
    max_step_norm_m: float = 0.0005
    hessian_damping_initial: float = 0.001
    restart_interval_steps: int = 200
    collision_step_safety: float = 0.95


def main(cfg: Config) -> None:
    assert cfg.resume_dir is None
    assert cfg.max_steps > 0 and cfg.checkpoint_every > 0 and cfg.ipc_threads > 0
    assert cfg.atol == 1e-8 and cfg.rtol == 1e-3
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = base.archive_sources(cfg.output_dir)
    for directory, name in (
        (base.GROUP / "src", "neutral-newton"),
        (base.SOLVERS / "src", "solver-performance"),
    ):
        shutil.copytree(
            directory,
            cfg.output_dir / "sources" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    provenance["sources"] = {
        str(path.relative_to(cfg.output_dir / "sources")): base.sha256(path)
        for path in sorted((cfg.output_dir / "sources").rglob("*.py"))
    }
    base.write_json(cfg.output_dir / "provenance.json", provenance)
    comparison = json.loads((cfg.comparison_dir / "summary.json").read_text())
    reference = json.loads((cfg.comparison_dir / "protocol.json").read_text())
    assert (
        base.sha256(cfg.comparison_dir / "protocol.json")
        == comparison["protocol"]["sha256"]
    )
    base.configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    neutral = base.load_current_binding(cfg.neutral_dir, cfg.output_dir)
    diagonal_policy = base.install_exact_bulk_diagonal()
    physics, baseline = base.build_eye_collision_physics(neutral, cfg.eyes_dir)
    materials = base.verify_materials(baseline)
    assert materials == reference["materials"]
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    pose = torch.zeros(6)
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    model.collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=1e-10, max_iterations=100000
    )
    coverage = base.audit_required_collision(physics)
    assert coverage == reference["collision"]
    seed_path = cfg.seed_dir / "seed.npz"
    seed_receipt = json.loads((cfg.seed_dir / "summary.json").read_text())
    assert seed_receipt["success"] and not seed_receipt["prior_equilibrium_used"]
    assert not seed_receipt["fem_reference_rebased"]
    assert (
        seed_receipt["seed"]["sha256"]
        == base.sha256(seed_path)
        == reference["inputs"]["seed"]["sha256"]
    )
    with np.load(seed_path, allow_pickle=False) as saved:
        seed_np = saved["displacement_m"].copy()
    assert seed_np.shape == physics.points.shape and np.isfinite(seed_np).all()
    full = physics.full_skull.extend_seed(torch.as_tensor(seed_np), pose)
    projected = model.dof_map.to_full(model.dof_map.to_free(full))
    roundoff = float((projected - full).abs().max())
    assert roundoff <= 8 * np.finfo(np.float64).eps * np.abs(physics.points).max()
    seed = projected[: len(physics.points)].detach().clone()
    with np.load(cfg.comparison_dir / "initial.npz", allow_pickle=False) as saved:
        assert np.array_equal(seed.cpu().numpy(), saved["displacement_m"])
    state = model.State(u=projected.detach().clone())
    state.collision = model.collision.state_at(state.u)
    initial_collision = base.audit_collision_state(physics, seed, pose)
    assert initial_collision["state_feasible"]
    delegate = base.FeasibleExpressionProblem(
        model=model, collision_step_safety=cfg.collision_step_safety
    )
    hessian = base.HessianProblem(delegate, cfg.hessian_backend)
    problem = base.CachedProblem(hessian, exact_curvature=True)
    diagonal_validation = base.verify_exact_diagonal(problem, state)
    g0 = float(torch.linalg.vector_norm(problem.grad(state)))
    threshold = max(cfg.atol, cfg.rtol * g0)
    np.testing.assert_allclose(g0, comparison["initial_grad_norm"], rtol=1e-14, atol=0)
    optimizer = AcceptedForcePncg(
        criteria=StrictPncg.ConvergenceCriteria(
            max_steps=cfg.max_steps,
            atol_primary=cfg.atol,
            rtol_primary=cfg.rtol,
            atol_secondary=cfg.atol,
            rtol_secondary=cfg.rtol,
        ),
        hess_damping=StrictPncg.HessianDamping(initial=cfg.hessian_damping_initial),
        line_search=StrictLineSearch(
            armijo=cfg.line_search_armijo,
            max_steps=cfg.line_search_max_backtracks,
            max_step_norm=cfg.max_step_norm_m,
        ),
    )
    optimizer.restart_interval = cfg.restart_interval_steps
    opt_state = optimizer.init(problem, state, model.dof_map.to_free(state.u))
    protocol = copy.deepcopy(reference)
    protocol.update(
        {
            "schema": "single-reference-neutral-legacy-pncg-v1",
            "config": cfg.model_dump(mode="json"),
            "comparison": base.record(cfg.comparison_dir / "summary.json"),
            "resume": None,
            "materials": materials,
            "collision": coverage,
            "initial_collision": initial_collision,
            "initial_geometry": physics.metrics(seed),
            "diagonal_policy": diagonal_policy,
            "diagonal_validation": diagonal_validation,
            "solver": {
                "method": "legacy strict PNCG, accepted-state force criterion",
                "implementation": "joint_expression_equilibrium.AcceptedForcePncg -> face_physics.StrictPncg -> liblaf.apple.solvers.optim.Pncg",
                "algorithm": "Dai-Kou+ nonlinear CG; descent restart; adaptive diagonal damping",
                "legacy_configuration_source": base.record(
                    base.JOINT / "src/87-run-eye-neutral-forward.py"
                ),
                "atol": cfg.atol,
                "rtol": cfg.rtol,
                "initial_grad_norm": g0,
                "effective_grad_threshold": threshold,
                "max_steps": cfg.max_steps,
                "convergence_rule": "true free-coordinate L2 gradient at accepted configuration <= max(atol, rtol*initial_gradient)",
                "preconditioner": "1 / (abs(diag(H)) + damping_factor * mean(abs(diag(H))))",
                "damping": {
                    "initial": optimizer.hess_damping.initial,
                    "minimum": optimizer.hess_damping.factor_min,
                    "maximum": optimizer.hess_damping.factor_max,
                },
                "restart_interval_steps": cfg.restart_interval_steps,
                "line_search": {
                    "armijo": cfg.line_search_armijo,
                    "factor": 0.5,
                    "max_backtracks": cfg.line_search_max_backtracks,
                    "max_attempts": cfg.line_search_max_backtracks + 1,
                    "max_coordinate_displacement_m": cfg.max_step_norm_m,
                    "initial_alpha": "min(infinity-norm cap, -g.p / damped_pHp); legacy curvature handling",
                },
                "ccd": {
                    "tolerance_m": 1e-10,
                    "max_iterations": 100000,
                    "safety": cfg.collision_step_safety,
                    "min_distance_m": model.collision.min_distance,
                },
                "physical_derivatives": "current exact unprojected diagonal and p.H.p from the same physical HVP as the current Newton run",
                "scope": "old PNCG direction, damping and line search; current material/reference/derivative contract and force threshold",
                "forward_solve_count": 1,
            },
            "hessian_backend": {
                "name": cfg.hessian_backend,
                "initial_setup": hessian.report(),
            },
            "runtime": {
                "torch": torch.__version__,
                "ipctk": ipctk.__version__,
                "gpu": torch.cuda.get_device_name(),
                "dtype": str(state.u.dtype),
            },
        }
    )
    protocol["inputs"]["current_runtime_binding"] = base.record(
        cfg.output_dir / "model-binding.json"
    )
    base.write_json(cfg.output_dir / "protocol.json", base.jsonable(protocol))
    base.save_checkpoint(cfg.output_dir / "initial.npz", seed)
    problem.counts.clear()
    torch.cuda.synchronize()
    started = time.perf_counter()
    failure = None
    previous_step = None
    result = "not_started"
    converged = False
    terminated = False
    faulthandler.dump_traceback_later(60, repeat=True)
    with torch.no_grad():
        for iteration in range(cfg.max_steps + 1):
            assert opt_state.step == iteration
            force = float(torch.linalg.vector_norm(problem.grad(state)))
            energy = float(problem.fun(state))
            assert np.isfinite(force) and np.isfinite(energy)
            row = {
                "accepted": True,
                "iteration": iteration,
                "grad_norm": force,
                "energy": energy,
                "elapsed_seconds": time.perf_counter() - started,
                "pncg": previous_step,
            }
            with (cfg.output_dir / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            base.write_json(
                cfg.output_dir / "status.json",
                {
                    "running": not terminated
                    and force > threshold
                    and iteration < cfg.max_steps,
                    **row,
                    "effective_grad_threshold": threshold,
                },
            )
            cherries.set_step(iteration)
            cherries.log_metrics(
                {
                    "forward/grad_norm": force,
                    "forward/energy": energy,
                    "forward/threshold": threshold,
                }
            )
            LOG.info(
                "Accepted PNCG %d: gradient %.8g (target %.8g), energy %.9g",
                iteration,
                force,
                threshold,
                energy,
            )
            if iteration % cfg.checkpoint_every == 0:
                base.save_checkpoint(
                    cfg.output_dir / f"checkpoint-{iteration:03d}.npz",
                    state.u[: len(physics.points)],
                )
            if force <= threshold:
                converged = True
                break
            if terminated or iteration == cfg.max_steps:
                failure = {
                    "reason": "PNCG iteration budget exhausted"
                    if iteration == cfg.max_steps
                    else "PNCG stopped before force gate",
                    "result": result,
                }
                break
            damping_before = opt_state.hess_damping_state.factor
            step_started = time.perf_counter()
            try:
                optimizer.step(problem, state, opt_state)
            except base.ForwardConvergenceError as error:
                failure = {
                    "reason": str(error),
                    "receipt": base.jsonable(error.receipt),
                }
                base.save_checkpoint(
                    cfg.output_dir / "failed-trial.npz", state.u[: len(physics.points)]
                )
                # StrictLineSearch rejects before params advances. Persist the last accepted state.
                problem.update(state, opt_state.params)
                break
            terminated, result_code = optimizer.terminate(problem, state, opt_state)
            result = str(result_code)
            line = opt_state.line_search_state
            assert line.ok
            previous_step = {
                "alpha": float(line.alpha),
                "backtracks": int(line.step),
                "line_search_trials": int(line.step) + 1,
                "energy_before": float(line.f0),
                "energy_after": float(line.f_alpha),
                "slope": float(opt_state.slope),
                "physical_pHp": float(opt_state.hess_quad),
                "damping_before": damping_before,
                "damping_after": opt_state.hess_damping_state.factor,
                "step_seconds": time.perf_counter() - step_started,
                "max_coordinate_displacement_m": float(
                    (line.alpha * opt_state.direction).abs().max()
                ),
            }
    faulthandler.cancel_dump_traceback_later()
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    final_force = float(torch.linalg.vector_norm(delegate.grad(state)))
    final_energy = float(delegate.fun(state))
    displacement = state.u[: len(physics.points)].detach().clone()
    geometry = physics.metrics(displacement)
    geometry["skin_rms_mm"] = geometry["surface_motion_rms_mm"]
    collision = base.audit_collision_state(physics, displacement, pose)
    success = (
        converged
        and final_force <= threshold
        and geometry["inverted_tetrahedra"] == 0
        and collision["state_feasible"]
    )
    terminal = base.save_checkpoint(cfg.output_dir / "terminal.npz", displacement)
    summary = {
        "schema": "single-reference-neutral-legacy-pncg-result-v1",
        "success": bool(success),
        "status": "converged_valid_neutral"
        if success
        else (
            "force_converged_geometry_invalid"
            if converged
            else "forward_did_not_converge"
        ),
        "method": "legacy strict PNCG",
        "failure": failure,
        "optimizer_result": result,
        "forward_solve_count": 1,
        "accepted_steps": int(opt_state.step),
        "max_steps": cfg.max_steps,
        "initial_grad_norm": g0,
        "effective_grad_threshold": threshold,
        "final_grad_norm": final_force,
        "final_energy": final_energy,
        "forward_seconds": forward_seconds,
        "absolute_tolerance_met": final_force <= cfg.atol,
        "relative_tolerance_met": final_force <= cfg.rtol * g0,
        "geometry": geometry,
        "collision": collision,
        "operation_counts": dict(problem.counts),
        "hessian_backend": hessian.report(),
        "rejected_contact_trials": delegate.rejected_contact_trials,
        "terminal": terminal,
        "protocol": base.record(cfg.output_dir / "protocol.json"),
        "adoption": "saved separately; selected neutral and Newton driver unchanged",
    }
    base.write_json(cfg.output_dir / "summary.json", base.jsonable(summary))
    base.write_json(
        cfg.output_dir / "status.json", {"running": False, **base.jsonable(summary)}
    )
    cherries.log_output(cfg.output_dir)
    LOG.info("Legacy PNCG forward result: %s", json.dumps(base.jsonable(summary)))
    if not success:
        message = f"Legacy PNCG forward saved without a valid equilibrium: {summary['status']}"
        raise RuntimeError(message)


if __name__ == "__main__":
    cherries.main(main, profile=base.ProfileJoint)
