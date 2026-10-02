"""Fixed-state IPC thread-count operation benchmark; no equilibrium solve."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
BENCHMARK = EXPERIMENT / "src/10-benchmark.py"
spec = importlib.util.spec_from_file_location("solver_performance_benchmark", BENCHMARK)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Config(benchmark.Config):
    output: Path = cherries.output("ipc-thread-results.json", mkdir=True)
    repeats: int = 5
    thread_counts: str = "default,1,2,4,8"


def timed(operation: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def fixture(cfg: Config, runner: Any) -> tuple[Any, Any, torch.Tensor, Any]:
    _inputs, physics, checkpoint, q_cpu, jaw_cpu = benchmark.make_fixture(
        cfg,
        "contact_expression",
        {"contact_expression": {"copy": str(cfg.run006_initial)}},
    )
    runtime = physics.runtime
    q = q_cpu.to(device="cuda", dtype=torch.float64)
    jaw = jaw_cpu.to(device="cuda", dtype=torch.float64)
    seed = checkpoint["displacement_m"].to(device="cuda", dtype=torch.float64)
    seed_jaw = checkpoint["jaw_normalized"].to(device="cuda", dtype=torch.float64)
    pose = runner.hinge_pose(
        jaw,
        torch.as_tensor(physics.arrays["mandible_frame_world"][:, 0], device="cuda"),
    )
    seed_pose = runner.hinge_pose(
        seed_jaw,
        torch.as_tensor(physics.arrays["mandible_frame_world"][:, 0], device="cuda"),
    )
    model = runtime.forward.model
    model.set_materials(
        physics.expression_materials(
            skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
            active_stress=runner.activation_stresses_mpa(q, runner.REFERENCE_MPA),
        )
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    full_seed = physics.full_skull.extend_seed(seed.detach(), seed_pose.detach())
    state = model.State(u=model.dof_map.to_full(model.dof_map.to_free(full_seed)))
    assert model.collision is not None
    state.collision = model.collision.state_at(state.u)
    problem = benchmark.FeasibleExpressionProblem(
        model=model, collision_step_safety=0.95
    )
    free = model.dof_map.to_free(state.u)
    direction = 1e-6 * torch.sin(
        torch.arange(free.numel(), device=free.device, dtype=free.dtype)
    )
    return model, problem, direction, state


def relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value - reference)
        / max(float(torch.linalg.vector_norm(reference)), 1e-300)
    )


def main(cfg: Config) -> None:
    assert cfg.repeats >= 2
    benchmark.configure_cuda()
    from remote_paths import install_loader_path_relocation

    install_loader_path_relocation(source_root=cfg.source_root)
    runner = benchmark.load_runner()
    _model, problem, direction, state = fixture(cfg, runner)
    default_threads = int(ipctk.get_num_threads())
    names = [item.strip() for item in cfg.thread_counts.split(",")]
    assert names[0] == "default"
    assert len(set(names)) == len(names)
    requested = [default_threads if item == "default" else int(item) for item in names]
    with torch.no_grad():
        reference = {
            "fun": problem.fun(state).detach().clone(),
            "grad": problem.grad(state).detach().clone(),
            "diag": problem.hess_diag(state).detach().clone(),
            "quad": problem.hess_quad(state, direction).detach().clone(),
        }
        state.collision.hess = None
        reference["hvp"] = problem.hess_prod(state, direction).detach().clone()
    rows = []
    for label, threads in zip(names, requested, strict=True):
        ipctk.set_num_threads(threads)
        timings: dict[str, list[float]] = {
            name: []
            for name in (
                "fun",
                "grad",
                "hess_diag",
                "hess_quad",
                "hess_prod_first",
                "hess_prod_cached",
                "ccd",
            )
        }
        values: dict[str, torch.Tensor] = {}
        for _repeat in range(cfg.repeats):
            values["fun"], timings["fun"] = (
                problem.fun(state),
                timings["fun"] + [timed(lambda: problem.fun(state))],
            )
            values["grad"], timings["grad"] = (
                problem.grad(state),
                timings["grad"] + [timed(lambda: problem.grad(state))],
            )
            values["diag"], timings["hess_diag"] = (
                problem.hess_diag(state),
                timings["hess_diag"] + [timed(lambda: problem.hess_diag(state))],
            )
            values["quad"], timings["hess_quad"] = (
                problem.hess_quad(state, direction),
                timings["hess_quad"]
                + [timed(lambda: problem.hess_quad(state, direction))],
            )
            state.collision.hess = None
            values["hvp"], timings["hess_prod_first"] = (
                problem.hess_prod(state, direction),
                timings["hess_prod_first"]
                + [
                    timed(
                        lambda: (
                            setattr(state.collision, "hess", None),
                            problem.hess_prod(state, direction),
                        )
                    )
                ],
            )
            timings["hess_prod_cached"].append(
                timed(lambda: problem.hess_prod(state, direction))
            )
            timings["ccd"].append(
                timed(lambda: problem.max_step_size(state, direction))
            )
        checks = {
            "energy_absolute": abs(float(values["fun"] - reference["fun"])),
            "gradient_relative": relative(values["grad"], reference["grad"]),
            "diag_relative": relative(values["diag"], reference["diag"]),
            "quad_relative": abs(float(values["quad"] - reference["quad"]))
            / max(abs(float(reference["quad"])), 1e-300),
            "hvp_relative": relative(values["hvp"], reference["hvp"]),
        }
        assert checks["energy_absolute"] <= 1e-12, checks
        assert max(checks.values()) <= 1e-10, checks
        rows.append(
            {
                "label": label,
                "ipctk_threads": threads,
                "checks": checks,
                "seconds": {
                    key: {"samples": values, "median": sorted(values)[len(values) // 2]}
                    for key, values in timings.items()
                },
            }
        )
    ipctk.set_num_threads(default_threads)
    result = {
        "schema": "ipc-thread-fixed-state-v1",
        "default_threads": default_threads,
        "repeats": cfg.repeats,
        "state": "fixed contact-expression MouthOpen initial proposal; no equilibrium",
        "rows": rows,
    }
    cfg.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            f"ipc_threads/{row['label']}/hess_prod_cached_seconds": row["seconds"][
                "hess_prod_cached"
            ]["median"]
            for row in rows
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
