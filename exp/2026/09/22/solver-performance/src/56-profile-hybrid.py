# ruff: noqa: E402, PLR0915, PT018
"""Profile one undamped PNCG-first, sparse-Newton cold Smile forward solve."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "hybrid_profile_cold", HERE / "43-compare-cold-forward.py"
)
assert spec is not None and spec.loader is not None
cold = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cold
spec.loader.exec_module(cold)
benchmark = cold.benchmark

import accelerated_solvers
import hybrid_first_solver
from assembled_fem_hvp import AssembledFemHvp
from gpu_free_sparse_hessian import GpuFreeSparseHessian
from hybrid_hessian import HybridHessian
from inverse_timing import install_inverse_timing
from profile_input_binding import bind_frozen_neutral_load


class Config(cold.Config):
    checkpoint: Path = (
        cold.EXPERIMENT
        / "data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt"
    )
    neutral_checkpoint: Path = (
        cold.EXPERIMENT
        / "data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt"
    )
    output_dir: Path = cold.EXPERIMENT / "data/hybrid-first-profile-001"
    newton_switch_atol: float = 0.0
    cuda_sync: bool = True
    finalize_only: bool = False


def finalize_saved_profile(cfg: Config) -> None:
    """Validate an already-saved endpoint without rerunning the forward solve."""
    protocol = json.loads((cfg.output_dir / "protocol.json").read_text())
    result = json.loads((cfg.output_dir / "solver-result.json").read_text())
    endpoint = torch.load(
        cfg.output_dir / "endpoint.pt", map_location="cpu", weights_only=False
    )
    assert benchmark.sha256(cfg.source_root / "uv.lock") == protocol["uv_lock_sha256"]
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    cold.configure_cuda()
    runner = cold.load_runner()
    inputs_manifest = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    neutral_dir = Path(inputs_manifest["parent_frozen_neutral"]["directory"])
    validation_dir = cfg.output_dir / "endpoint-validation"
    validation_dir.mkdir(exist_ok=True)
    with bind_frozen_neutral_load(
        neutral_dir, validation_dir, allow_pncg_curvature_clamps=True
    ):
        fitter = cold.build_fitter(
            cfg, runner, "hybrid_first", validation_dir / "scratch"
        )
    q = endpoint["activation"].to(device="cuda", dtype=torch.float64)
    jaw = endpoint["jaw_normalized"].to(device="cuda", dtype=torch.float64)
    saved = endpoint["displacement_m"].to(device="cuda", dtype=torch.float64)
    assert benchmark.tensor_sha256(q) == protocol["activation_sha256"]
    physics, model = fitter.physics, fitter.runtime.forward.model
    pose = runner.hinge_pose(jaw, fitter.hinge_axis)
    fem_count = physics.full_skull.geometry.fem_node_count
    displacement = saved[:fem_count]
    reconstructed_full = physics.full_skull.extend_seed(displacement, pose)
    if len(saved) == model.n_points:
        torch.testing.assert_close(saved, reconstructed_full, rtol=0, atol=1e-14)
    model.set_materials(
        physics.expression_materials(
            skin_multiplier=torch.ones((), device="cuda"),
            active_stress=runner.activation_stresses_mpa(q, runner.REFERENCE_MPA),
        )
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    state = model.State(u=reconstructed_full)
    state.collision = model.collision.state_at(state.u)
    problem = cold.FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    result["terminal_force"] = float(torch.linalg.vector_norm(problem.grad(state)))
    result["shape"] = benchmark.jsonable(physics.metrics(displacement, target_index=12))
    result["collision"] = benchmark.jsonable(
        cold.audit_collision_state(physics, displacement, pose)
    )
    result["gpu_after_endpoint_validation"] = benchmark.gpu_snapshot()
    result["endpoint_validation"] = (
        "Reconstructed saved full collision state from FEM nodes; independently reevaluated free force, inversion and contact. No forward iteration rerun."
    )
    if result["success"]:
        assert result["terminal_force"] <= cfg.forward_atol
        assert (
            result["shape"]["inverted_tetrahedra"] == 0
            and result["collision"]["state_feasible"]
        )
    benchmark.write_json(
        cfg.output_dir / "summary.json", {"protocol": protocol, "result": result}
    )
    benchmark.write_json(
        cfg.output_dir / "status.json",
        {
            "running": False,
            "success": result["success"],
            "endpoint_validation_completed": True,
        },
    )
    for name in ("summary.json", "timing.json", "trace.jsonl"):
        cherries.log_output(cfg.output_dir / name)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("success", "forward_wall_seconds", "terminal_force")
            }
        ),
        flush=True,
    )


def main(cfg: Config) -> None:
    if cfg.finalize_only:
        finalize_saved_profile(cfg)
        return
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.wall_seconds is None
    assert cfg.forward_atol == 1e-8 and cfg.linear_rtol == 1e-3
    assert cfg.max_newton_steps == 100 and cfg.newton_switch_atol == 0
    cfg.output_dir.mkdir(parents=True)
    benchmark.archive_benchmark_sources(cfg)
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    assert ipctk.get_num_threads() == cfg.ipc_threads
    cold.configure_cuda()
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    neutral = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["expression"] == "Smile" and checkpoint["accepted_steps"] == 16
    assert neutral["accepted_steps"] == 0
    runner = cold.load_runner()
    setup_started = time.perf_counter()
    inputs_manifest = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    neutral_dir = Path(inputs_manifest["parent_frozen_neutral"]["directory"])
    with bind_frozen_neutral_load(
        neutral_dir, cfg.output_dir, allow_pncg_curvature_clamps=True
    ):
        fitter = cold.build_fitter(
            cfg, runner, "hybrid_first", cfg.output_dir / "scratch"
        )
    model_setup_seconds = time.perf_counter() - setup_started
    q = checkpoint["activation"].to(device="cuda", dtype=torch.float64).detach().clone()
    jaw = (
        checkpoint["jaw_normalized"]
        .to(device="cuda", dtype=torch.float64)
        .detach()
        .clone()
    )
    seed = (
        neutral["displacement_m"]
        .to(device="cuda", dtype=torch.float64)
        .detach()
        .clone()
    )
    seed_jaw = torch.zeros_like(jaw)
    coverage = cold.audit_required_collision(fitter.physics)
    seed_collision = cold.audit_collision_state(
        fitter.physics, seed, runner.hinge_pose(seed_jaw, fitter.hinge_axis)
    )
    assert seed_collision["state_feasible"]
    warm_started = time.perf_counter()
    initial_force = cold.warm_operators(
        fitter=fitter, runner=runner, q=q, jaw=jaw, seed=seed
    )
    prewarm_seconds = time.perf_counter() - warm_started
    runtime, model = fitter.runtime, fitter.runtime.forward.model
    protocol = {
        "schema": "hybrid-first-cold-profile-v1",
        "config": cfg.model_dump(mode="json"),
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "activation_sha256": benchmark.tensor_sha256(q),
        "jaw_sha256": benchmark.tensor_sha256(jaw),
        "seed_displacement_sha256": benchmark.tensor_sha256(seed),
        "initial_force": initial_force,
        "coverage": coverage,
        "seed_collision": seed_collision,
        "model_setup_seconds": model_setup_seconds,
        "operator_prewarm_seconds": prewarm_seconds,
        "scope": "One cold displacement start with saved active stress and fixed prestress; full bone/eyeball contact. No inverse update or adjoint. Operator JIT prewarm and model construction excluded; first Newton sparse topology construction included.",
        "pncg": {
            "damping": 0,
            "backtracking": False,
            "preconditioner": "abs_jacobi",
            "contact_curvature": "ipctk_gauss_newton_clamped_per_contact",
            "material_curvature": "clamped_per_cell_quadrature_or_membrane_triangle",
            "window_steps": 20,
            "window_statistic": "median",
            "minimum_reduction": 0.1,
            "required_poor_comparisons": 2,
            "earliest_stall_handoff_step": 60,
            "fixed_relative_handoff": False,
            "step_cap": None,
            "ccd_safety": 0.9,
        },
        "newton": runtime.newton_parameters,
        "newton_hessian": "exact FEM plus IPC, cached unshifted GPU free CSR; exact assembled diagonal",
        "versions": {
            "ipctk": importlib.metadata.version("ipctk"),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "ipctk_source": "official PyPI wheel; project uv.lock archived by hash",
        "uv_lock_sha256": benchmark.sha256(cfg.source_root / "uv.lock"),
        "runtime_binding": {
            "path": str(cfg.output_dir / "profile-input-binding.json"),
            "sha256": benchmark.sha256(cfg.output_dir / "profile-input-binding.json"),
        },
        "host": platform.node(),
        "pid": os.getpid(),
        "gpu_device": torch.cuda.get_device_name(0),
        "compute_processes_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "gpu_before": benchmark.gpu_snapshot(),
        "timing": "Hierarchical wall time with explicit CUDA synchronization at Python scopes; additive accounting uses exclusive time. Profiling changes overlap. Trace evaluation overhead is separately scoped. All linear retries are included. Per-contact GN curvature is timed as one batch, without wrapping thousands of individual native calls.",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": True, "stage": "forward"}
    )
    installed = install_inverse_timing(model, cuda_sync=cfg.cuda_sync)
    timer = installed.timer
    timer.patch(type(model.collision), "hess_quad", "collision/hess_quad")
    timer.patch(
        ipctk.BarrierPotential,
        "gauss_newton_hessian_diagonal",
        "ipc/gauss_newton_hessian_diagonal",
        sync=False,
    )
    timer.patch(
        type(model.collision),
        "raw_hess_quad_terms",
        "ipc/per_contact_gauss_newton_batch",
        sync=False,
    )
    timer.patch(hybrid_first_solver, "run_pncg_phase", "pncg")
    timer.patch(hybrid_first_solver, "safeguarded_newton", "newton")
    timer.patch(accelerated_solvers, "pcg", "pcg")
    timer.patch(HybridHessian, "prepare", "hessian/cache_prepare", sync=False)
    timer.patch(HybridHessian, "apply", "hessian/spmv")
    timer.patch(HybridHessian, "diagonal", "hessian/diagonal")
    timer.patch(AssembledFemHvp, "__init__", "hessian/fem_constructor")
    timer.patch(AssembledFemHvp, "setup", "hessian/fem_numeric")
    timer.patch(GpuFreeSparseHessian, "__init__", "hessian/csr_constructor")
    timer.patch(GpuFreeSparseHessian, "setup", "hessian/csr_refresh")
    assert not installed.missing and not timer.missing, (
        installed.missing,
        timer.missing,
    )
    original_step = accelerated_solvers.safeguarded_newton_step
    original_diagonal = hybrid_first_solver.SparseNewtonProblem.hess_diag
    validated = False
    hvp_validation: dict[str, float] = {}
    newton_steps = 0
    torch.cuda.reset_peak_memory_stats()
    solve_started = time.perf_counter()
    with (cfg.output_dir / "trace.jsonl").open("x") as trace:

        def record(row: dict) -> None:
            row = {**row, "elapsed_seconds": time.perf_counter() - solve_started}
            with timer.scope("trace/write", sync=False):
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if (
                    row.get("kind") != "pncg"
                    or row.get("step", row.get("pncg_steps", 0)) % 20 == 0
                ):
                    print(json.dumps(row), flush=True)

        def checked_diagonal(problem: Any, state: Any) -> torch.Tensor:
            nonlocal validated
            result = original_diagonal(problem, state)
            if not validated:
                with timer.scope("validation/handoff_hvp"):
                    generator = torch.Generator(device=state.u.device).manual_seed(
                        20260922
                    )
                    direction = torch.randn(
                        result.shape,
                        device=result.device,
                        dtype=result.dtype,
                        generator=generator,
                    )
                    reference = problem.delegate.hess_prod(state, direction)
                    actual = problem.hessian.apply(state, direction)
                    relative = float(
                        torch.linalg.vector_norm(actual - reference)
                        / torch.linalg.vector_norm(reference)
                    )
                    assert relative < 1e-10, relative
                    hvp_validation["handoff_relative_error"] = relative
                    validated = True
            return result

        def traced_step(problem: Any, state: Any, **kwargs: Any) -> Any:
            nonlocal newton_steps
            with timer.scope("newton/step"):
                state, receipt = original_step(problem, state, **kwargs)
            newton_steps += 1
            with timer.scope("trace/newton_state"):
                force = float(torch.linalg.vector_norm(problem.grad(state)))
                energy = float(problem.fun(state))
                record(
                    {
                        "kind": "newton",
                        "step": newton_steps,
                        "force": force,
                        "energy": energy,
                        "accepted_step": receipt,
                    }
                )
            return state, receipt

        runtime.trace_callback = record
        accelerated_solvers.safeguarded_newton_step = traced_step
        hybrid_first_solver.SparseNewtonProblem.hess_diag = checked_diagonal
        result: dict[str, Any] = {}
        try:
            with timer.scope("forward"):
                displacement = fitter.solve(
                    q, jaw, seed, seed_jaw, "profile/hybrid_first"
                )
            result["success"] = True
        except cold.ForwardConvergenceError as error:
            displacement = (
                runtime.forward.state.u[
                    : fitter.physics.full_skull.geometry.fem_node_count
                ]
                .detach()
                .clone()
            )
            result.update(
                success=False,
                failure=str(error),
                failure_receipt=benchmark.jsonable(error.receipt),
            )
        finally:
            torch.cuda.synchronize()
            result["forward_wall_seconds"] = time.perf_counter() - solve_started
            result["forward"] = benchmark.jsonable(runtime.last_forward)
            result["hvp_validation"] = hvp_validation
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            benchmark.write_json(cfg.output_dir / "timing.json", timer.report())
            installed.uninstall()
            accelerated_solvers.safeguarded_newton_step = original_step
            hybrid_first_solver.SparseNewtonProblem.hess_diag = original_diagonal
    torch.save(
        {
            "displacement_m": displacement.detach().cpu(),
            "activation": q.cpu(),
            "jaw_normalized": jaw.cpu(),
            "converged": result["success"],
        },
        cfg.output_dir / "endpoint.pt",
    )
    benchmark.write_json(cfg.output_dir / "solver-result.json", result)
    benchmark.write_json(
        cfg.output_dir / "status.json",
        {
            "running": False,
            "stage": "validating_endpoint",
            "solver_success": result["success"],
        },
    )
    result["terminal_force"] = float(
        torch.linalg.vector_norm(runtime.last_problem.grad(runtime.forward.state))
    )
    result["shape"] = benchmark.jsonable(
        fitter.physics.metrics(displacement, target_index=12)
    )
    result["collision"] = benchmark.jsonable(
        cold.audit_collision_state(
            fitter.physics, displacement, runner.hinge_pose(jaw, fitter.hinge_axis)
        )
    )
    result["gpu_after"] = benchmark.gpu_snapshot()
    result["compute_processes_after"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if result["success"]:
        assert result["terminal_force"] <= cfg.forward_atol
        assert (
            result["shape"]["inverted_tetrahedra"] == 0
            and result["collision"]["state_feasible"]
        )
    torch.save(
        {
            "displacement_m": displacement.detach().cpu(),
            "activation": q.cpu(),
            "jaw_normalized": jaw.cpu(),
            "converged": result["success"],
        },
        cfg.output_dir / "endpoint.pt",
    )
    benchmark.write_json(
        cfg.output_dir / "summary.json", {"protocol": protocol, "result": result}
    )
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": False, "success": result["success"]}
    )
    cherries.log_metrics(
        {
            "hybrid/success": float(result["success"]),
            "hybrid/seconds": result["forward_wall_seconds"],
            "hybrid/terminal_force": result["terminal_force"],
        }
    )
    for name in ("protocol.json", "summary.json", "timing.json", "trace.jsonl"):
        cherries.log_output(cfg.output_dir / name)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("success", "forward_wall_seconds", "terminal_force")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
