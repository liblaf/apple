# ruff: noqa: C901, E402, PLR0915, PT018
"""One sequential cold Smile arm with fixed or adaptive physical IPC stiffness."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import numpy as np
import torch

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "adaptive_ipc_profile_base", Path(__file__).with_name("56-profile-hybrid.py")
)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
cold, benchmark = base.cold, base.benchmark

import accelerated_solvers
import hybrid_first_solver
from adaptive_ipc_stiffness import AdaptiveIPCStiffness
from assembled_fem_hvp import AssembledFemHvp
from gpu_free_sparse_hessian import GpuFreeSparseHessian
from hybrid_hessian import HybridHessian
from inverse_timing import install_inverse_timing
from joint_fields import BULK_TISSUES, research_informed_material_config
from profile_input_binding import bind_frozen_neutral_load


class Config(base.Config):
    output_dir: Path = cold.EXPERIMENT / "data/adaptive-ipc-fixed-001"
    mode: Literal["fixed", "adaptive"] = "fixed"
    stiffness_multiplier: float = 1.0
    epsilon_scale: float = 1e-6
    maximum_stiffness_multiplier: float = 100.0


def gpu_processes() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists() and not cfg.finalize_only
    assert cfg.forward_atol == 1e-8 and cfg.linear_rtol == 1e-3
    assert cfg.max_newton_steps == 100 and cfg.wall_seconds is None
    assert cfg.stiffness_multiplier > 0 and cfg.maximum_stiffness_multiplier >= 1
    cfg.output_dir.mkdir(parents=True)
    benchmark.archive_benchmark_sources(cfg)
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    cold.configure_cuda()
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    neutral = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["expression"] == "Smile" and checkpoint["accepted_steps"] == 16
    assert neutral["accepted_steps"] == 0
    runner = cold.load_runner()
    manifest = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    setup_started = time.perf_counter()
    with bind_frozen_neutral_load(
        Path(manifest["parent_frozen_neutral"]["directory"]),
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
    ):
        fitter = cold.build_fitter(
            cfg, runner, "hybrid_first", cfg.output_dir / "scratch"
        )
    runtime, physics = fitter.runtime, fitter.physics
    model = runtime.forward.model
    neutral_manifest = json.loads(
        (
            Path(manifest["parent_frozen_neutral"]["directory"]) / "manifest.json"
        ).read_text()
    )
    frozen_protocol = json.loads(
        Path(neutral_manifest["sources"]["protocol"]["path"]).read_text()
    )
    original_stiffness = float(frozen_protocol["mechanics"]["contact"]["stiffness_mpa"])
    canonical = research_informed_material_config()["materials"]
    young_moduli = {name: float(canonical[name]["young_mpa"]) for name in BULK_TISSUES}
    with np.load(neutral_manifest["sources"]["skin_field"]["path"]) as skin_field:
        young_moduli["skin_maximum"] = float(np.max(skin_field["E_mpa"]))
    stiffest_material = max(young_moduli, key=young_moduli.__getitem__)
    maximum_young = young_moduli[stiffest_material]
    initial_stiffness = 0.1 * maximum_young * cfg.stiffness_multiplier
    potential = model.collision.potential
    model.collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(),
        potential.dhat,
        initial_stiffness,
        model.collision.use_physical_barrier,
    )
    controller = AdaptiveIPCStiffness(
        model.collision,
        initial_stiffness=initial_stiffness,
        enabled=cfg.mode == "adaptive",
        epsilon_scale=cfg.epsilon_scale,
        max_stiffness=initial_stiffness * cfg.maximum_stiffness_multiplier,
    )
    setup_seconds = time.perf_counter() - setup_started
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
    coverage = cold.audit_required_collision(physics)
    seed_collision = cold.audit_collision_state(
        physics, seed, runner.hinge_pose(seed_jaw, fitter.hinge_axis)
    )
    assert seed_collision["state_feasible"]
    warm_started = time.perf_counter()
    initial_force = cold.warm_operators(
        fitter=fitter, runner=runner, q=q, jaw=jaw, seed=seed
    )
    protocol = {
        "schema": "adaptive-ipc-cold-smile-v1",
        "config": cfg.model_dump(mode="json"),
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "activation_sha256": benchmark.tensor_sha256(q),
        "jaw_sha256": benchmark.tensor_sha256(jaw),
        "seed_displacement_sha256": benchmark.tensor_sha256(seed),
        "initial_force": initial_force,
        "coverage": coverage,
        "seed_collision": seed_collision,
        "model_setup_seconds": setup_seconds,
        "operator_prewarm_seconds": time.perf_counter() - warm_started,
        "contact_stiffness_policy": {
            "mode": cfg.mode,
            "original_stiffness_mpa": original_stiffness,
            "selection_rule": "0.1 * maximum Young modulus over all deformable materials, then optional comparison multiplier",
            "young_moduli_mpa": young_moduli,
            "stiffest_material": stiffest_material,
            "maximum_young_mpa": maximum_young,
            "rigid_obstacles": "bones and eyes have prescribed motion and no elastic Young modulus",
            "initial_stiffness_mpa": controller.current_stiffness,
            "epsilon_scale": cfg.epsilon_scale,
            "maximum_stiffness_mpa": controller.current_stiffness
            * cfg.maximum_stiffness_multiplier,
            "scope": "Explicit physical contact-parameter intervention after immutable input validation. Energy, force, and Hessian all use current stiffness; no inverse or adjoint is run.",
        },
        "pncg": {
            "damping": 0,
            "backtracking": False,
            "window_steps": 20,
            "required_poor_comparisons": 2,
            "minimum_reduction": 0.1,
            "curvature": "per-contribution clamped",
            "stiffness_change": "restart direction and force windows",
        },
        "newton": runtime.newton_parameters,
        "versions": {
            "ipctk": base.importlib.metadata.version("ipctk"),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "ipctk_source": "official PyPI",
        "uv_lock_sha256": benchmark.sha256(cfg.source_root / "uv.lock"),
        "runtime_binding_sha256": benchmark.sha256(
            cfg.output_dir / "profile-input-binding.json"
        ),
        "host": platform.node(),
        "pid": os.getpid(),
        "gpu_device": torch.cuda.get_device_name(0),
        "compute_processes_before": gpu_processes(),
        "timing": "One fresh process per arm, sequential. CUDA-synchronized hierarchical timings include tracing and first sparse setup/validation; operator prewarm and model construction excluded. Persistent kernel caches may be warm.",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": True, "stage": "forward"}
    )
    installed = install_inverse_timing(model, cuda_sync=cfg.cuda_sync)
    timer = installed.timer
    for owner, attribute, label, sync in (
        (type(model.collision), "hess_quad", "collision/hess_quad", True),
        (
            ipctk.BarrierPotential,
            "gauss_newton_hessian_diagonal",
            "ipc/gauss_newton_hessian_diagonal",
            False,
        ),
        (
            type(model.collision),
            "raw_hess_quad_terms",
            "ipc/per_contact_gauss_newton_batch",
            False,
        ),
        (hybrid_first_solver, "run_pncg_phase", "pncg", True),
        (hybrid_first_solver, "safeguarded_newton", "newton", True),
        (accelerated_solvers, "pcg", "pcg", True),
        (HybridHessian, "prepare", "hessian/cache_prepare", False),
        (HybridHessian, "apply", "hessian/spmv", True),
        (HybridHessian, "diagonal", "hessian/diagonal", True),
        (AssembledFemHvp, "__init__", "hessian/fem_constructor", True),
        (AssembledFemHvp, "setup", "hessian/fem_numeric", True),
        (GpuFreeSparseHessian, "__init__", "hessian/csr_constructor", True),
        (GpuFreeSparseHessian, "setup", "hessian/csr_refresh", True),
    ):
        timer.patch(owner, attribute, label, sync=sync)
    assert not installed.missing and not timer.missing
    original_hybrid = hybrid_first_solver.hybrid_first
    original_step = accelerated_solvers.safeguarded_newton_step
    original_diagonal = hybrid_first_solver.SparseNewtonProblem.hess_diag
    original_update = controller.after_update
    hvp_validation: list[dict[str, float]] = []
    validated_stiffness: float | None = None
    newton_steps = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    solve_started = time.perf_counter()
    with (cfg.output_dir / "trace.jsonl").open("x") as trace:

        def record(row: dict) -> None:
            row = {
                **row,
                "stiffness_mpa": controller.current_stiffness,
                "elapsed_seconds": time.perf_counter() - solve_started,
            }
            with timer.scope("trace/write", sync=False):
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if (
                    row["kind"] in {"initial", "stiffness_change"}
                    or row.get("step", 0) % 20 == 0
                ):
                    print(json.dumps(row), flush=True)

        def adaptive_update(problem: Any, state: Any, **kwargs: Any) -> Any:
            with timer.scope("contact_stiffness/update"):
                event = original_update(problem, state, **kwargs)
            record(
                {
                    "kind": "stiffness_change"
                    if event["stiffness_changed"]
                    else "contact_update",
                    **event,
                }
            )
            return event

        def adaptive_hybrid(problem: Any, state: Any, **kwargs: Any) -> Any:
            return original_hybrid(
                problem, state, adaptive_stiffness=controller, **kwargs
            )

        def checked_diagonal(problem: Any, state: Any) -> torch.Tensor:
            nonlocal validated_stiffness
            diagonal = original_diagonal(problem, state)
            stiffness = controller.current_stiffness
            if validated_stiffness != stiffness:
                with timer.scope("validation/handoff_hvp"):
                    generator = torch.Generator(device=state.u.device).manual_seed(
                        20260922
                    )
                    direction = torch.randn(
                        diagonal.shape,
                        device=diagonal.device,
                        dtype=diagonal.dtype,
                        generator=generator,
                    )
                    reference = problem.delegate.hess_prod(state, direction)
                    actual = problem.hessian.apply(state, direction)
                    relative = float(
                        torch.linalg.vector_norm(actual - reference)
                        / torch.linalg.vector_norm(reference)
                    )
                    assert relative < 1e-10, relative
                    hvp_validation.append(
                        {"stiffness_mpa": stiffness, "relative_error": relative}
                    )
                    validated_stiffness = stiffness
            return diagonal

        def traced_step(problem: Any, state: Any, **kwargs: Any) -> Any:
            nonlocal newton_steps
            with timer.scope("newton/step"):
                state, receipt = original_step(problem, state, **kwargs)
            newton_steps += 1
            with timer.scope("trace/newton_state"):
                record(
                    {
                        "kind": "newton",
                        "step": newton_steps,
                        "force": float(torch.linalg.vector_norm(problem.grad(state))),
                        "energy": float(problem.fun(state)),
                        "accepted_step": receipt,
                        "observation": "after accepted Newton step, before adaptive stiffness update",
                    }
                )
            return state, receipt

        runtime.trace_callback = record
        controller.after_update = adaptive_update
        hybrid_first_solver.hybrid_first = adaptive_hybrid
        accelerated_solvers.safeguarded_newton_step = traced_step
        hybrid_first_solver.SparseNewtonProblem.hess_diag = checked_diagonal
        result: dict[str, Any] = {}
        try:
            with timer.scope("forward"):
                displacement = fitter.solve(
                    q, jaw, seed, seed_jaw, f"adaptive_ipc/{cfg.mode}"
                )
            result["success"] = True
        except cold.ForwardConvergenceError as error:
            displacement = (
                runtime.forward.state.u[: physics.full_skull.geometry.fem_node_count]
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
            result["stiffness"] = controller.receipt()
            result["hvp_validation"] = hvp_validation
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            benchmark.write_json(cfg.output_dir / "timing.json", timer.report())
            benchmark.write_json(
                cfg.output_dir / "stiffness.json", controller.receipt()
            )
            installed.uninstall()
            controller.after_update = original_update
            hybrid_first_solver.hybrid_first = original_hybrid
            accelerated_solvers.safeguarded_newton_step = original_step
            hybrid_first_solver.SparseNewtonProblem.hess_diag = original_diagonal
    result["terminal_force"] = float(
        torch.linalg.vector_norm(runtime.last_problem.grad(runtime.forward.state))
    )
    result["terminal_stiffness_mpa"] = controller.current_stiffness
    result["shape"] = benchmark.jsonable(physics.metrics(displacement, target_index=12))
    result["collision"] = benchmark.jsonable(
        cold.audit_collision_state(
            physics, displacement, runner.hinge_pose(jaw, fitter.hinge_axis)
        )
    )
    result["compute_processes_after"] = gpu_processes()
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
            "contact_stiffness_mpa": result["terminal_stiffness_mpa"],
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
            "forward/success": float(result["success"]),
            "forward/seconds": result["forward_wall_seconds"],
            "forward/terminal_force": result["terminal_force"],
            "forward/stiffness_mpa": result["terminal_stiffness_mpa"],
        }
    )
    for name in (
        "protocol.json",
        "summary.json",
        "timing.json",
        "stiffness.json",
        "trace.jsonl",
    ):
        cherries.log_output(cfg.output_dir / name)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "success",
                    "forward_wall_seconds",
                    "terminal_force",
                    "terminal_stiffness_mpa",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
