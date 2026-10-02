# ruff: noqa: C901, E402, PLR0912, PLR0915, PT018
"""Run one matched skin/contact ablation with adaptive IPC on the natural face."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
NEUTRAL = ROOT / "exp/2026/09/22/neutral-newton"
sys.path[:0] = [str(JOINT / "src"), str(NEUTRAL / "src")]

spec = importlib.util.spec_from_file_location(
    "neutral_adaptive_benchmark", HERE / "10-benchmark.py"
)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)

import accelerated_solvers
import hybrid_first_solver
from accelerated_solvers import CachedProblem
from adaptive_ipc_stiffness import AdaptiveIPCStiffness
from assembled_fem_hvp import AssembledFemHvp
from gpu_free_sparse_hessian import GpuFreeSparseHessian
from hybrid_hessian import HybridHessian
from inverse_timing import install_inverse_timing
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from joint_materials import StableNeoHookeanMembrane, StableNeoHookeanStress
from joint_physics import moduli
from joint_rigid_eye_contact import build_eye_collision_physics
from mesh_step_scale import mean_rest_edge_length
from profile_input_binding import bind_frozen_neutral_load
from smile_collision import audit_collision_state, audit_required_collision


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/neutral-ablation-full-001"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    seed_dir: Path = NEUTRAL / "data/reference-seed-002"
    case: Literal["full", "no_collision", "no_skin", "neither"] = "full"
    epsilon_scale: float = 1e-6
    maximum_stiffness_multiplier: float = 100.0
    ipc_threads: int = 4
    cuda_sync: bool = True


def _sha256(path: Path) -> str:
    return benchmark.sha256(path)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _gpu_processes() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _copy_neutral_sources(output_dir: Path) -> dict[str, Any]:
    target = output_dir / "sources" / "neutral-newton"
    shutil.copytree(
        NEUTRAL / "src", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    receipt = {
        "source_root": str((NEUTRAL / "src").resolve()),
        "sources": {
            str(path.relative_to(target)): _sha256(path)
            for path in sorted(target.rglob("*.py"))
        },
    }
    benchmark.write_json(output_dir / "neutral-newton-source-provenance.json", receipt)
    return receipt


def _verify_materials(baseline: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    expected = {"fat": 0.0112, "muscle": 0.012, "aponeurosis": 1.693}
    result: dict[str, float] = {}
    for name, young in expected.items():
        mu, la = moduli(young, 0.49)
        assert torch.allclose(
            baseline[name]["mu"], torch.full_like(baseline[name]["mu"], mu)
        )
        assert torch.allclose(
            baseline[name]["lmbda"], torch.full_like(baseline[name]["lmbda"], la)
        )
        assert not bool(torch.count_nonzero(baseline[name]["active_stress"])), name
        result[name] = young
    skin = baseline["skin"]
    skin_young = skin["mu"] * (2 * 1.49)
    assert bool(torch.count_nonzero(skin["baseline_stress"]))
    result["skin_maximum"] = float(skin_young.max())
    return result


def _set_stiffness(collision: Any, stiffness: float) -> None:
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(),
        potential.dhat,
        stiffness,
        collision.use_physical_barrier,
    )


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.ipc_threads > 0
    assert cfg.epsilon_scale > 0 and cfg.maximum_stiffness_multiplier >= 1
    cfg.output_dir.mkdir(parents=True)
    benchmark.archive_benchmark_sources(
        benchmark.Config.model_construct(output_dir=cfg.output_dir, source_root=ROOT)
    )
    neutral_sources = _copy_neutral_sources(cfg.output_dir)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    assert ipctk.get_num_threads() == cfg.ipc_threads
    ipctk_version = importlib.metadata.version("ipctk")
    assert ipctk.__version__ == ipctk_version

    seed_path = cfg.seed_dir / "seed.npz"
    seed_receipt_path = cfg.seed_dir / "summary.json"
    seed_receipt = json.loads(seed_receipt_path.read_text())
    assert seed_receipt["success"] and seed_receipt["prior_equilibrium_used"] is False
    assert seed_receipt["fem_reference_rebased"] is False
    assert seed_receipt["seed"]["sha256"] == _sha256(seed_path)
    with np.load(seed_path, allow_pickle=False) as archive:
        seed_np = archive["displacement_m"].copy()

    setup_started = time.perf_counter()
    with bind_frozen_neutral_load(
        cfg.neutral_dir, cfg.output_dir, allow_pncg_curvature_clamps=True
    ) as neutral:
        physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    young = _verify_materials(baseline)
    stiffest_material = max(young, key=young.__getitem__)
    kappa_anchor = 0.1 * young[stiffest_material]
    kappa_initial = kappa_anchor
    collision_enabled = cfg.case in {"full", "no_skin"}
    skin_enabled = cfg.case in {"full", "no_collision"}
    pose = torch.zeros(
        6,
        device=model.dof_map.fixed_values.device,
        dtype=model.dof_map.fixed_values.dtype,
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    collision = model.collision
    assert collision is not None
    collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=1e-10, max_iterations=100000
    )
    collision.min_distance = 1e-8
    _set_stiffness(collision, kappa_anchor)
    assert seed_np.shape == physics.points.shape and np.isfinite(seed_np).all()
    seed = torch.as_tensor(seed_np, device=pose.device, dtype=pose.dtype)
    full_seed = physics.full_skull.extend_seed(seed, pose)
    projected = model.dof_map.to_full(model.dof_map.to_free(full_seed))
    seed = projected[: len(physics.points)].detach().clone()
    seed_collision = audit_collision_state(physics, seed, pose)
    assert seed_collision["state_feasible"], seed_collision
    coverage = audit_required_collision(physics)
    state = model.State(u=projected.detach().clone())
    state.collision = collision.state_at(state.u)
    anchor_problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    # Every arm uses the full model's force at the identical repaired seed and
    # kappa0 to define one common absolute force gate before removing terms.
    anchor_force = float(torch.linalg.vector_norm(anchor_problem.grad(state)))
    target_force = max(1e-8, 1e-3 * anchor_force)
    potentials = model.warp_model.__wrapped__.potentials
    assert set(potentials) == {"fat", "muscle", "aponeurosis", "skin"}
    if not skin_enabled:
        assert isinstance(potentials["skin"], StableNeoHookeanMembrane)
        del potentials["skin"]
    if not collision_enabled:
        model.collision = None
        state.collision = None
    delegate = (
        FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
        if collision_enabled
        else ForwardProblem(model=model)
    )
    problem = CachedProblem(delegate, exact_curvature=False)
    direction = torch.full_like(model.dof_map.to_free(state.u), 1e-6)
    prewarm_started = time.perf_counter()
    with torch.no_grad():
        delegate.fun(state)
        initial_gradient = delegate.grad(state)
        delegate.hess_diag(state)
        delegate.hess_prod(state, direction)
        delegate.hess_quad(state, direction)
        delegate.max_step_size(state, direction)
    torch.cuda.synchronize()
    prewarm_seconds = time.perf_counter() - prewarm_started
    initial_force = float(torch.linalg.vector_norm(initial_gradient))
    problem.invalidate()
    if state.collision is not None:
        state.collision.hess = None
    controller = (
        AdaptiveIPCStiffness(
            collision,
            initial_stiffness=kappa_initial,
            enabled=True,
            epsilon_scale=cfg.epsilon_scale,
            max_stiffness=kappa_initial * cfg.maximum_stiffness_multiplier,
        )
        if collision_enabled
        else None
    )

    def stiffness_value() -> float | None:
        return controller.current_stiffness if controller is not None else None

    def stiffness_receipt() -> dict[str, Any]:
        if controller is not None:
            return controller.receipt()
        return {
            "enabled": False,
            "disabled_reason": "contact component removed from model",
            "initial_stiffness": None,
            "final_stiffness": None,
            "observations": [],
            "events": [],
        }

    edge_mean = mean_rest_edge_length(model, physics.points)
    max_step = 0.5 * edge_mean
    boundary_roundoff = float((projected - full_seed).abs().max())
    boundary_roundoff_limit = float(
        8 * np.finfo(np.float64).eps * np.abs(physics.points).max()
    )
    assert boundary_roundoff <= boundary_roundoff_limit, (
        boundary_roundoff,
        boundary_roundoff_limit,
    )
    setup_seconds = time.perf_counter() - setup_started - prewarm_seconds

    protocol = {
        "schema": "natural-reference-component-ablation-v1",
        "config": cfg.model_dump(mode="json"),
        "ablation": {
            "case": cfg.case,
            "collision_enabled": collision_enabled,
            "skin_enabled": skin_enabled,
            "skin_removal": "entire membrane potential including prescribed baseline tension",
            "collision_removal": "no barrier energy, derivatives, broad phase, or CCD during solve",
            "active_potentials": list(potentials),
            "reduced_problem_validity": "force gate and positive tetrahedra; contact gate only when contact is enabled",
        },
        "initial_force": initial_force,
        "fixture": {
            "name": "repaired constitutive-reference natural face",
            "neutral_manifest": _record(cfg.neutral_dir / "manifest.json"),
            "eyes_manifest": _record(cfg.eyes_dir / "manifest.json"),
            "seed": _record(seed_path),
            "seed_receipt": _record(seed_receipt_path),
            "prior_equilibrium_used": False,
            "fem_reference_rebased": False,
            "activation": "all bulk active_stress tensors verified zero",
            "jaw_rotation_rad": 0.0,
        },
        "coverage": coverage,
        "seed_collision": seed_collision,
        "boundary_projection": {
            "maximum_roundoff_m": boundary_roundoff,
            "roundoff_limit_m": boundary_roundoff_limit,
        },
        "materials": {
            "young_moduli_mpa": young,
            "stiffest_material": stiffest_material,
        },
        "contact_stiffness_policy": {
            "anchor_rule": "0.1 * maximum deformable-material Young modulus",
            "anchor_stiffness_mpa": kappa_anchor,
            "initial_stiffness_mpa": kappa_initial,
            "maximum_stiffness_mpa": kappa_initial * cfg.maximum_stiffness_multiplier,
            "epsilon_scale": cfg.epsilon_scale,
            "mode": "adaptive" if collision_enabled else "disabled",
            "tolerance_anchor_force": anchor_force,
            "effective_force_tolerance": target_force,
            "tolerance_rule": "max(1e-8, 1e-3 * full-model free-force norm at anchor kappa before component removal)",
        },
        "solver": {
            "method": "per-contribution-clamped PNCG then exact sparse Newton-CG",
            "pncg": {
                "damping": 0,
                "backtracking": False,
                "curvature": "ipctk GN diagonal and per-contact GN quadratic contributions clamped before summation",
                "window_steps": 20,
                "required_poor_comparisons": 2,
                "minimum_force_reduction": 0.1,
            },
            "newton": {
                "max_steps": 100,
                "linear_rtol": 1e-3,
                "linear_max_steps": 1000,
                "preconditioner": "abs_jacobi",
                "initial_shift": 0.0,
                "shift": "mean signed Hessian diagonal then x10; 8 attempts",
                "armijo": 1e-4,
                "backtracking": "x0.5, 8 trials",
            },
            "max_coordinate_displacement_m": max_step,
            "ccd": {
                "tolerance_m": 1e-10,
                "max_iterations": 100000,
                "safety": 0.9,
                "min_distance_m": collision.min_distance,
            },
        },
        "sources": {"neutral_newton": neutral_sources},
        "model_setup_seconds": setup_seconds,
        "operator_prewarm_seconds": prewarm_seconds,
        "prewarm_scope": "one untimed physical evaluation at the cold seed; no displacement update and no sparse Newton setup",
        "runtime_binding": _record(cfg.output_dir / "profile-input-binding.json"),
        "versions": {
            "ipctk": ipctk_version,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "gpu": torch.cuda.get_device_name(0),
        "pid": os.getpid(),
        "compute_processes_before": _gpu_processes(),
        "timing": "Hierarchical CUDA-synchronized timing includes trace work, adaptive updates, and first sparse Newton setup. The uninstrumented operator prewarm and model construction are excluded; profiling perturbs overlap.",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": True, "stage": "forward"}
    )

    installed = install_inverse_timing(model, cuda_sync=cfg.cuda_sync)
    timer = installed.timer
    for potential_type, label in (
        (StableNeoHookeanStress, "bulk"),
        (StableNeoHookeanMembrane, "skin"),
    ):
        for method in ("fun", "grad", "hess_diag", "hess_prod", "hess_quad"):
            timer.patch(potential_type, method, f"{label}/{method}", sync=True)
    for owner, attribute, label, sync in (
        (type(collision), "hess_quad", "collision/hess_quad", True),
        (
            ipctk.BarrierPotential,
            "gauss_newton_hessian_diagonal",
            "ipc/gauss_newton_hessian_diagonal",
            False,
        ),
        (
            type(collision),
            "raw_hess_quad_terms",
            "ipc/per_contact_gauss_newton_batch",
            False,
        ),
        (hybrid_first_solver, "run_pncg_phase", "pncg", True),
        (hybrid_first_solver, "safeguarded_newton", "newton", True),
        (accelerated_solvers, "pcg", "pcg", True),
        (AdaptiveIPCStiffness, "after_update", "contact_stiffness/update", True),
        (HybridHessian, "prepare", "hessian/cache_prepare", False),
        (HybridHessian, "apply", "hessian/spmv", True),
        (HybridHessian, "diagonal", "hessian/diagonal", True),
        (AssembledFemHvp, "__init__", "hessian/fem_constructor", True),
        (AssembledFemHvp, "setup", "hessian/fem_numeric", True),
        (GpuFreeSparseHessian, "__init__", "hessian/csr_constructor", True),
        (GpuFreeSparseHessian, "setup", "hessian/csr_refresh", True),
    ):
        timer.patch(owner, attribute, label, sync=sync)
    assert installed.missing == ([] if collision_enabled else ["collision"]), (
        installed.missing,
        timer.missing,
    )
    assert not timer.missing, timer.missing
    original_step = accelerated_solvers.safeguarded_newton_step
    original_diagonal = hybrid_first_solver.SparseNewtonProblem.hess_diag
    original_update = controller.after_update if controller is not None else None
    hvp_validation: list[dict[str, float]] = []
    newton_steps = 0
    pncg_steps = 0
    result: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (cfg.output_dir / "trace.jsonl").open("x") as trace:

        def record(row: dict[str, Any]) -> None:
            nonlocal pncg_steps
            if row.get("kind") == "pncg":
                pncg_steps = int(row["step"])
            trace.write(
                json.dumps(
                    benchmark.jsonable(
                        {
                            **row,
                            "stiffness_mpa": stiffness_value(),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                )
                + "\n"
            )
            trace.flush()

        def update_stiffness(
            current_problem: Any, current_state: Any, **kwargs: Any
        ) -> Any:
            nonlocal pncg_steps
            assert original_update is not None
            event = original_update(current_problem, current_state, **kwargs)
            if event["phase"] == "pncg":
                pncg_steps = int(event["step"])
            if event["stiffness_changed"]:
                record({"kind": "stiffness_change", **event})
            return event

        def checked_diagonal(current_problem: Any, current_state: Any) -> torch.Tensor:
            diagonal = original_diagonal(current_problem, current_state)
            if not hvp_validation:
                direction = torch.randn(
                    diagonal.shape,
                    device=diagonal.device,
                    dtype=diagonal.dtype,
                    generator=torch.Generator(device=diagonal.device).manual_seed(
                        20260923
                    ),
                )
                reference = current_problem.delegate.hess_prod(current_state, direction)
                actual = current_problem.hessian.apply(current_state, direction)
                relative = float(
                    torch.linalg.vector_norm(actual - reference)
                    / torch.linalg.vector_norm(reference)
                )
                assert relative < 1e-10, relative
                hvp_validation.append(
                    {
                        "stiffness_mpa": stiffness_value(),
                        "relative_error": relative,
                    }
                )
            return diagonal

        def traced_step(current_problem: Any, current_state: Any, **kwargs: Any) -> Any:
            nonlocal newton_steps
            current_state, receipt = original_step(
                current_problem, current_state, **kwargs
            )
            newton_steps += 1
            record(
                {
                    "kind": "newton",
                    "step": newton_steps,
                    "force": float(
                        torch.linalg.vector_norm(current_problem.grad(current_state))
                    ),
                    "energy": float(current_problem.fun(current_state)),
                    "accepted_step": receipt,
                }
            )
            return current_state, receipt

        if controller is not None:
            controller.after_update = update_stiffness
        accelerated_solvers.safeguarded_newton_step = traced_step
        hybrid_first_solver.SparseNewtonProblem.hess_diag = checked_diagonal
        try:
            with timer.scope("forward"):
                state, receipt = hybrid_first_solver.hybrid_first(
                    problem,
                    state,
                    atol=target_force,
                    max_step_norm=max_step,
                    linear_rtol=1e-3,
                    max_newton_steps=100,
                    callback=record,
                    adaptive_stiffness=controller,
                )
            result.update(success=True, forward_receipt=receipt)
        except ForwardConvergenceError as error:
            result.update(
                success=False,
                failure=str(error),
                failure_receipt=benchmark.jsonable(error.receipt),
            )
        finally:
            torch.cuda.synchronize()
            result["forward_wall_seconds"] = time.perf_counter() - started
            result["hvp_validation"] = hvp_validation
            result["stiffness"] = stiffness_receipt()
            result["pncg_steps"] = pncg_steps
            result["newton_steps"] = newton_steps
            result["initial_force"] = initial_force
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            timing = timer.report()

            def labels(tree: dict[str, Any]) -> set[str]:
                return set(tree).union(
                    *(labels(node["children"]) for node in tree.values())
                )

            measured_labels = labels(timing["tree"])
            if not collision_enabled:
                assert not any(
                    label.startswith(("collision/", "ipc/", "contact_stiffness/"))
                    for label in measured_labels
                ), measured_labels
            if not skin_enabled:
                assert not any(label.startswith("skin/") for label in measured_labels)
            result["disabled_component_calls_verified_absent"] = True
            benchmark.write_json(cfg.output_dir / "timing.json", timing)
            benchmark.write_json(cfg.output_dir / "stiffness.json", stiffness_receipt())
            installed.uninstall()
            if controller is not None:
                del controller.after_update
            accelerated_solvers.safeguarded_newton_step = original_step
            hybrid_first_solver.SparseNewtonProblem.hess_diag = original_diagonal

    displacement = state.u[: len(physics.points)].detach().clone()
    result["terminal_force"] = float(torch.linalg.vector_norm(problem.grad(state)))
    result["terminal_energy"] = float(problem.fun(state))
    result["terminal_stiffness_mpa"] = stiffness_value()
    result["geometry"] = benchmark.jsonable(physics.metrics(displacement))
    # The disabled geometry is still audited AFTER timing. This is a diagnostic
    # of compatibility with the full anatomy, not part of the reduced solve.
    model.collision = collision
    try:
        result["collision"] = benchmark.jsonable(
            audit_collision_state(physics, displacement, pose)
        )
    finally:
        if not collision_enabled:
            model.collision = None
    result["full_contact_valid"] = result["collision"]["state_feasible"]
    result["valid_forward"] = bool(
        result["success"]
        and result["terminal_force"] <= target_force
        and result["geometry"]["inverted_tetrahedra"] == 0
        and (not collision_enabled or result["full_contact_valid"])
    )
    result["compute_processes_after"] = _gpu_processes()
    np.savez_compressed(
        cfg.output_dir / "endpoint.npz",
        displacement_m=displacement.detach().cpu().numpy(),
    )
    benchmark.write_json(
        cfg.output_dir / "summary.json",
        {"protocol": protocol, "result": benchmark.jsonable(result)},
    )
    benchmark.write_json(
        cfg.output_dir / "status.json",
        {
            "running": False,
            "success": result["success"],
            "valid_forward": result["valid_forward"],
        },
    )
    cherries.log_metrics(
        {
            "forward/success": float(result["success"]),
            "forward/valid": float(result["valid_forward"]),
            "forward/seconds": result["forward_wall_seconds"],
            "forward/terminal_force": result["terminal_force"],
        }
    )
    if collision_enabled:
        cherries.log_metrics(
            {"forward/stiffness_mpa": result["terminal_stiffness_mpa"]}
        )
    for name in (
        "protocol.json",
        "summary.json",
        "timing.json",
        "stiffness.json",
        "trace.jsonl",
        "endpoint.npz",
    ):
        cherries.log_output(cfg.output_dir / name)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
