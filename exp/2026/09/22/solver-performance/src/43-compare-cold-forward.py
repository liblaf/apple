# ruff: noqa: E402, PLR0915, PT018
"""One cold-start Smile forward comparison: Newton-CG-only versus hybrid."""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
JOINT = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(JOINT / "src")]

from accelerated_solvers import accelerate_runtime
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from remote_paths import install_loader_path_relocation
from smile_collision import audit_collision_state, audit_required_collision


def load_benchmark() -> Any:
    path = Path(__file__).with_name("10-benchmark.py")
    spec = importlib.util.spec_from_file_location("cold_forward_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark()


class Config(cherries.BaseConfig):
    checkpoint: Path
    neutral_checkpoint: Path
    output_dir: Path = EXPERIMENT / "data/cold-forward-comparison-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    inputs_dir: Path = JOINT / "data/expression-inputs-002"
    forward_atol: float = 1e-8
    linear_rtol: float = 1e-3
    newton_switch_atol: float = 1e-7
    max_newton_steps: int = 100
    wall_seconds: float | None = None
    ipc_threads: int = 8


def load_runner() -> Any:
    path = JOINT / "src/93-fit-expressions.py"
    spec = importlib.util.spec_from_file_location("cold_forward_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synchronized_time(operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def warm_operators(
    *, fitter: Any, runner: Any, q: torch.Tensor, jaw: torch.Tensor, seed: torch.Tensor
) -> float:
    """Compile fixed-model operators at the identical cold seed, without solve."""
    runtime, physics = fitter.runtime, fitter.physics
    model = runtime.forward.model
    target_pose = runner.hinge_pose(jaw, fitter.hinge_axis)
    zero_pose = torch.zeros_like(target_pose)
    model.set_materials(
        physics.expression_materials(
            skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
            active_stress=runner.activation_stresses_mpa(
                q.detach(), runner.REFERENCE_MPA
            ),
        )
    )
    model.dof_map.fixed_values = physics.boundary(target_pose).detach().clone()
    full_seed = physics.full_skull.extend_seed(seed.detach(), zero_pose)
    state = model.State(u=model.dof_map.to_full(model.dof_map.to_free(full_seed)))
    state.collision = model.collision.state_at(state.u)
    problem = FeasibleExpressionProblem(
        model=model, collision_step_safety=runtime.collision_step_safety
    )
    direction = torch.full_like(model.dof_map.to_free(state.u), 1e-6)
    with torch.no_grad():
        problem.fun(state)
        gradient = problem.grad(state)
        problem.hess_diag(state)
        problem.hess_prod(state, direction)
        problem.hess_quad(state, direction)
        problem.max_step_size(state, direction)
    torch.cuda.synchronize()
    return float(torch.linalg.vector_norm(gradient))


def build_fitter(cfg: Config, runner: Any, method: str, directory: Path) -> Any:
    fit_cfg = runner.Config(
        _cli_parse_args=False,
        output_dir=directory,
        inputs_dir=cfg.inputs_dir,
        calibration_source=None,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=1e-7,
        ipc_threads=cfg.ipc_threads,
        learning_rate=0.3,
        magnitude_weight=0.0,
        jaw_weight=0.0,
        outer_step_policy="full_adam",
        pose_first=False,
        pose_collision=True,
    )
    fitter = runner.Fitter(fit_cfg)
    runtime = accelerate_runtime(
        fitter.runtime,
        method,
        rest_points=fitter.physics.points,
        wall_seconds=cfg.wall_seconds,
        linear_rtol=cfg.linear_rtol,
        max_newton_steps=cfg.max_newton_steps,
        newton_switch_atol=cfg.newton_switch_atol,
        shift_policy="reset",
    )
    fitter.runtime = runtime
    fitter.physics.runtime = runtime
    return fitter


def run_one(
    cfg: Config,
    *,
    runner: Any,
    method: str,
    q_cpu: torch.Tensor,
    jaw_cpu: torch.Tensor,
    seed_cpu: torch.Tensor,
    target_displacement_cpu: torch.Tensor,
) -> dict[str, Any]:
    directory = cfg.output_dir / "scratch" / method
    fitter = build_fitter(cfg, runner, method, directory)
    q = q_cpu.to(device="cuda", dtype=torch.float64).detach().clone()
    jaw = jaw_cpu.to(device="cuda", dtype=torch.float64).detach().clone()
    seed = seed_cpu.to(device="cuda", dtype=torch.float64).detach().clone()
    seed_jaw = torch.zeros_like(jaw)
    initial = {
        "activation_sha256": benchmark.tensor_sha256(q),
        "jaw_sha256": benchmark.tensor_sha256(jaw),
        "seed_displacement_sha256": benchmark.tensor_sha256(seed),
        "seed_jaw_sha256": benchmark.tensor_sha256(seed_jaw),
        "target_jaw_normalized": float(jaw[0]),
    }
    coverage = audit_required_collision(fitter.physics)
    seed_collision = audit_collision_state(
        fitter.physics, seed, runner.hinge_pose(seed_jaw, fitter.hinge_axis)
    )
    assert seed_collision["state_feasible"]
    prewarm_force_norm = warm_operators(
        fitter=fitter, runner=runner, q=q, jaw=jaw, seed=seed
    )
    before = benchmark.gpu_snapshot()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        displacement, seconds = synchronized_time(
            lambda: fitter.solve(q, jaw, seed, seed_jaw, f"cold-forward/{method}")
        )
        forward = copy.deepcopy(fitter.runtime.last_forward)
        shape = fitter.physics.metrics(displacement.detach(), target_index=12)
        collision = audit_collision_state(
            fitter.physics,
            displacement.detach(),
            runner.hinge_pose(jaw, fitter.hinge_axis),
        )
        valid = (
            forward["success"]
            and forward["grad_norm"] <= cfg.forward_atol
            and shape["inverted_tetrahedra"] == 0
            and collision["state_feasible"]
        )
        output = cfg.output_dir / "outputs" / f"{method}.pt"
        output.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "displacement_m": displacement.detach().cpu(),
                "activation": q.detach().cpu(),
                "jaw_normalized": jaw.detach().cpu(),
            },
            output,
        )
        delta_mm = 1000 * (displacement.detach().cpu() - target_displacement_cpu)
        endpoint_reference = {
            "saved_target_displacement_sha256": benchmark.tensor_sha256(
                target_displacement_cpu
            ),
            "skin_weighted_rms_mm": float(
                (
                    fitter.weights.detach().cpu()
                    * delta_mm[fitter.obs.detach().cpu()].square().sum(-1)
                )
                .sum()
                .sqrt()
            ),
            "maximum_node_mm": float(torch.linalg.vector_norm(delta_mm, dim=-1).max()),
            "scope": "comparison to saved target checkpoint endpoint only; it was not used as the seed",
        }
        initial_after = {
            "activation_sha256": benchmark.tensor_sha256(q),
            "jaw_sha256": benchmark.tensor_sha256(jaw),
            "seed_displacement_sha256": benchmark.tensor_sha256(seed),
            "seed_jaw_sha256": benchmark.tensor_sha256(seed_jaw),
        }
        assert initial_after == {
            key: value
            for key, value in initial.items()
            if key != "target_jaw_normalized"
        }
        return {
            "method": method,
            "success": bool(valid),
            "forward_wall_seconds": seconds,
            "initial": initial,
            "initial_after": initial_after,
            "prewarm_force_norm": prewarm_force_norm,
            "prewarm": "fixed-model energy, gradient, Hessian diagonal/product/quadratic, and CCD max-step; CUDA synchronized and excluded from timing",
            "forward": benchmark.jsonable(forward),
            "operation_counts": benchmark.operation_counts(forward),
            "shape": benchmark.jsonable(shape),
            "collision": benchmark.jsonable(collision),
            "coverage": coverage,
            "saved_target_endpoint": endpoint_reference,
            "output_path": str(output),
            "output_sha256": benchmark.sha256(output),
            "displacement_sha256": benchmark.tensor_sha256(displacement),
            "gpu": {
                "before": before,
                "after": benchmark.gpu_snapshot(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }
    except ForwardConvergenceError as error:
        torch.cuda.synchronize()
        return {
            "method": method,
            "success": False,
            "expected_failure": True,
            "forward_wall_seconds": time.perf_counter() - started,
            "initial": initial,
            "prewarm_force_norm": prewarm_force_norm,
            "forward": benchmark.jsonable(copy.deepcopy(fitter.runtime.last_forward)),
            "operation_counts": benchmark.operation_counts(fitter.runtime.last_forward),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "receipt": benchmark.jsonable(
                    getattr(error, "receipt", fitter.runtime.last_forward)
                ),
            },
            "gpu": {"before": before, "after": benchmark.gpu_snapshot()},
        }


def compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    weights: torch.Tensor,
    obs: torch.Tensor,
) -> dict[str, Any]:
    if not reference["success"] or not candidate["success"]:
        return {"comparable": False, "reason": "one_or_both_forwards_failed"}
    original = torch.load(
        reference["output_path"], map_location="cpu", weights_only=False
    )["displacement_m"]
    other = torch.load(
        candidate["output_path"], map_location="cpu", weights_only=False
    )["displacement_m"]
    delta_mm = 1000 * (other - original)
    rms = float((weights * delta_mm[obs].square().sum(-1)).sum().sqrt())
    maximum = float(torch.linalg.vector_norm(delta_mm, dim=-1).max())
    return {
        "comparable": True,
        "skin_weighted_rms_mm": rms,
        "maximum_node_mm": maximum,
        "forward_wall_speedup": reference["forward_wall_seconds"]
        / candidate["forward_wall_seconds"],
    }


def main(cfg: Config) -> None:
    assert cfg.checkpoint.is_file() and cfg.neutral_checkpoint.is_file()
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.forward_atol == 1e-8
    assert cfg.linear_rtol == 1e-3
    assert cfg.newton_switch_atol == 1e-7
    assert cfg.max_newton_steps == 100
    assert cfg.wall_seconds is None or cfg.wall_seconds > 0
    assert cfg.ipc_threads == 8
    cfg.output_dir.mkdir(parents=True)
    benchmark.archive_benchmark_sources(cfg)
    install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    assert int(ipctk.get_num_threads()) == cfg.ipc_threads
    configure_cuda()
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    neutral = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["expression"] == "Smile" and checkpoint["expression_index"] == 12
    q_cpu, jaw_cpu = (
        checkpoint["activation"].clone(),
        checkpoint["jaw_normalized"].clone(),
    )
    seed_cpu = neutral["displacement_m"].clone()
    target_displacement_cpu = checkpoint["displacement_m"].clone()
    assert q_cpu.shape[-1] == 6 and jaw_cpu.shape == (1,)
    assert seed_cpu.ndim == 2 and seed_cpu.shape[-1] == 3
    assert checkpoint["accepted_steps"] == 16
    assert checkpoint["metrics"]["shape"]["inverted_tetrahedra"] == 0
    assert neutral["accepted_steps"] == 0
    assert not bool(torch.count_nonzero(neutral["activation"]))
    assert not bool(torch.count_nonzero(neutral["jaw_normalized"]))
    assert neutral["metrics"]["shape"]["inverted_tetrahedra"] == 0
    protocol = {
        "schema": "cold-smile-forward-newton-versus-hybrid-v1",
        "checkpoint": {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": benchmark.sha256(cfg.checkpoint),
        },
        "neutral_checkpoint": {
            "path": str(cfg.neutral_checkpoint.resolve()),
            "sha256": benchmark.sha256(cfg.neutral_checkpoint),
        },
        "inputs": {
            name: benchmark.sha256(cfg.inputs_dir / name)
            for name in ("manifest.json", "state.npz")
        },
        "initial": {
            "activation_sha256": benchmark.tensor_sha256(q_cpu),
            "jaw_sha256": benchmark.tensor_sha256(jaw_cpu),
            "seed_displacement_sha256": benchmark.tensor_sha256(seed_cpu),
            "seed_jaw_sha256": benchmark.tensor_sha256(torch.zeros_like(jaw_cpu)),
            "target_jaw_normalized": float(jaw_cpu[0]),
        },
        "saved_target_displacement_sha256": benchmark.tensor_sha256(
            target_displacement_cpu
        ),
        "methods": ["newton_diag", "hybrid_diag"],
        "config": cfg.model_dump(mode="json"),
        "scope": "saved Smile activation/jaw with no Adam step, contact-valid shared-neutral displacement at zero seed jaw, fresh sequential runtimes; no warm displacement, adjoint, or outer update",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    runner = load_runner()
    rows: list[dict[str, Any]] = []
    reference = None
    for method in ("newton_diag", "hybrid_diag"):
        benchmark.write_json(
            cfg.output_dir / "status.json",
            {
                "running": True,
                "current_method": method,
                "completed_methods": [item["method"] for item in rows],
            },
        )
        row = run_one(
            cfg,
            runner=runner,
            method=method,
            q_cpu=q_cpu,
            jaw_cpu=jaw_cpu,
            seed_cpu=seed_cpu,
            target_displacement_cpu=target_displacement_cpu,
        )
        assert row["initial"] == protocol["initial"]
        if reference is None:
            reference = row
        else:
            assert reference is not None
            probe = build_fitter(
                cfg, runner, method, cfg.output_dir / "scratch" / "comparison"
            )
            row["comparison_to_newton_diag"] = compare(
                reference, row, probe.weights.cpu(), probe.obs.cpu()
            )
            del probe
        rows.append(row)
        benchmark.write_json(cfg.output_dir / "results.json", rows)
        benchmark.write_json(
            cfg.output_dir / "status.json",
            {
                "running": True,
                "current_method": None,
                "completed_methods": [item["method"] for item in rows],
            },
        )
        print(
            json.dumps(
                {
                    "method": method,
                    "success": row["success"],
                    "seconds": row["forward_wall_seconds"],
                }
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema": protocol["schema"],
        "success": all(row["success"] for row in rows),
        "protocol": protocol,
        "results": rows,
        "source_sha256": benchmark.sha256(Path(__file__)),
    }
    benchmark.write_json(cfg.output_dir / "summary.json", summary)
    benchmark.write_json(
        cfg.output_dir / "status.json",
        {
            "running": False,
            "completed_methods": [item["method"] for item in rows],
            "success": summary["success"],
        },
    )
    cherries.log_metrics(
        {
            "cold_forward/success": float(summary["success"]),
            "cold_forward/arms": len(rows),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
