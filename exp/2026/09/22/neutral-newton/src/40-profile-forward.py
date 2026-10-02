# ruff: noqa: C901, E402, PLR0912, PLR0915, PT018
"""Profile matched checkpoint windows without extending the neutral trajectory."""

from __future__ import annotations

import contextlib
import cProfile
import io
import json
import logging
import pstats
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import scipy.sparse
import torch

from liblaf import cherries
from liblaf.apple.forward.hessian._contact import GpuContactHessian
from liblaf.apple.forward.hessian._problem import HessianBackend, HessianProblem

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
SOLVERS = GROUP.parent / "solver-performance"
sys.path[:0] = [str(JOINT / "src"), str(SOLVERS / "src")]

import accelerated_solvers as solver
from inverse_timing import InverseTimer
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from joint_rigid_eye_contact import build_eye_collision_physics
from model_binding import install_exact_bulk_diagonal, load_current_binding

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/profile-001"
    run_dir: Path = GROUP / "data/forward-003"
    cold_run_dir: Path = GROUP / "data/forward-002"
    steps: int = 10
    repeats: int = 3
    ipc_threads: int = 4
    hessian_backend: HessianBackend = "matrix_free"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def gpu_snapshot() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv"],
        text=True,
    ) + subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        text=True,
    )


def install_hooks(model: Any) -> InverseTimer:
    timer = InverseTimer(cuda_sync=True)
    timer.patch(solver, "safeguarded_newton_step", "newton")
    timer.patch(solver, "pcg", "pcg")
    timer.patch(HessianProblem, "_prepare", "backend/prepare")
    timer.patch(GpuContactHessian, "hess_prod", "backend/contact_hvp")
    timer.patch(torch.sparse, "mm", "backend/sparse_matvec")
    for cls, prefix, methods in (
        (
            solver.CachedProblem,
            "problem",
            ("fun", "grad", "hess_diag", "hess_prod", "max_step_size", "update"),
        ),
        (
            type(model),
            "model",
            ("fun", "grad", "hess_diag", "hess_prod", "max_step_size", "update"),
        ),
        (type(model.warp_model), "fem", ("fun", "grad", "hess_diag", "hess_prod")),
        (
            type(model.collision),
            "contact",
            ("fun", "grad", "hess_diag", "hess_prod", "max_step_size", "update"),
        ),
    ):
        for method in methods:
            timer.patch(cls, method, f"{prefix}/{method}")
    for cls, method, label in (
        (ipctk.Candidates, "build", "ipc/broad_phase"),
        (ipctk.NormalCollisions, "build", "ipc/contact_set"),
        (ipctk.BarrierPotential, "hessian", "ipc/hessian_assembly"),
        (ipctk.BarrierPotential, "gradient", "ipc/gradient"),
        (ipctk.NormalCollisions, "compute_minimum_distance", "ipc/minimum_distance"),
        (ipctk.Candidates, "compute_collision_free_stepsize", "ipc/ccd"),
        (scipy.sparse.csc_matrix, "_matmul_vector", "cpu/contact_spmv"),
    ):
        timer.patch(cls, method, label, sync=False)
    assert not timer.missing, timer.missing
    return timer


def flatten(tree: dict) -> dict:
    totals = defaultdict(
        lambda: {"count": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0}
    )

    def walk(nodes: dict) -> None:
        for name, node in nodes.items():
            for field in ("count", "inclusive_seconds", "exclusive_seconds"):
                totals[name][field] += node[field]
            walk(node["children"])

    walk(tree)
    return dict(totals)


def replay(
    physics: Any,
    seed: np.ndarray,
    start: int,
    steps: int,
    settings: dict,
    expected: list[dict],
    mode: str,
    output: Path,
    backend: HessianBackend = "matrix_free",
) -> dict:
    model = physics.runtime.forward.model
    pose = torch.zeros(6)
    full = physics.full_skull.extend_seed(torch.as_tensor(seed), pose)
    full = model.dof_map.to_full(model.dof_map.to_free(full))
    assert np.array_equal(full[: len(seed)].detach().cpu().numpy(), seed)
    state = model.State(u=full.detach().clone())
    state.collision = model.collision.state_at(state.u)
    delegate = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    hessian = HessianProblem(delegate, backend)
    problem = solver.CachedProblem(hessian, exact_curvature=True)
    initial_grad = float(torch.linalg.vector_norm(problem.grad(state)))
    initial_energy = float(problem.fun(state))
    np.testing.assert_allclose(
        initial_grad, expected[start]["grad_norm"], rtol=1e-8, atol=1e-15
    )
    np.testing.assert_allclose(
        initial_energy, expected[start]["energy"], rtol=1e-8, atol=1e-20
    )
    # The production driver performs exact-diagonal checks before iterations 0 and 100.
    if start in (0, 100):
        problem.hess_diag(state)
    operator_errors = []
    if backend != "matrix_free":
        # Validate against the original exact matrix-free + CPU IPC operator.
        generator = torch.Generator(device=state.u.device).manual_seed(20260922)
        for _ in range(3):
            vector = torch.randn(
                (model.n_free,),
                device=state.u.device,
                dtype=state.u.dtype,
                generator=generator,
            )
            reference = delegate.hess_prod(state, vector)
            candidate = hessian.hess_prod(state, vector)
            error = float(
                torch.linalg.vector_norm(candidate - reference)
                / torch.linalg.vector_norm(reference)
            )
            assert error < 1e-10, error
            operator_errors.append(error)
    backend_before = hessian.report()
    if start not in (0, 100):
        # Preflight paid only one-time construction; the first timed state still
        # refreshes physical values and constructs its contact Hessian.
        state.collision.hess = None
        hessian.invalidate()
    problem.counts.clear()
    original_pcg = solver.pcg
    reference_residuals = []
    if mode == "validation":

        def audited_pcg(
            matvec: Any, precondition: Any, rhs: torch.Tensor, **kwargs: Any
        ) -> tuple:
            direction, info = original_pcg(matvec, precondition, rhs, **kwargs)
            assert len(matvec.__defaults__) == 1
            shift = matvec.__defaults__[0]
            residual = float(
                torch.linalg.vector_norm(
                    delegate.hess_prod(state, direction) + shift * direction - rhs
                )
                / torch.linalg.vector_norm(rhs)
            )
            assert residual <= kwargs["rtol"], residual
            reference_residuals.append(residual)
            return direction, {**info, "reference_true_relative_residual": residual}

        solver.pcg = audited_pcg
    profiler = cProfile.Profile() if mode == "cprofile" else None
    timer = install_hooks(model) if mode == "synchronized" else None
    rows = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with (
            torch.no_grad(),
            timer.scope("window") if timer else contextlib.nullcontext(),
        ):
            if profiler:
                profiler.enable()
            for iteration in range(start, start + steps):
                gradient = problem.grad(state)
                # Retain the production driver's accepted-state energy evaluation.
                float(problem.fun(state))
                state, receipt = solver.safeguarded_newton_step(
                    problem,
                    state,
                    atol=settings["effective_grad_threshold"],
                    linear_rtol=1e-3,
                    linear_max_steps=1000,
                    max_step_norm=settings["max_coordinate_displacement_m"],
                    armijo=1e-4,
                    max_shift_attempts=8,
                    max_backtracking_trials=8,
                    backtracking_factor=0.5,
                    preconditioner="diag",
                    shift_policy="reset",
                    shift_scale_policy="mean_abs",
                    gradient=gradient,
                )
                rows.append({"iteration": iteration + 1, "newton": receipt})
            final_grad = float(torch.linalg.vector_norm(problem.grad(state)))
            final_energy = float(problem.fun(state))
            torch.cuda.synchronize()
            if profiler:
                profiler.disable()
    finally:
        if profiler:
            profiler.disable()
        if timer:
            timer.restore()
        solver.pcg = original_pcg
    seconds = time.perf_counter() - started
    target = expected[start + steps]
    np.testing.assert_allclose(final_grad, target["grad_norm"], rtol=1e-8, atol=1e-15)
    np.testing.assert_allclose(final_energy, target["energy"], rtol=1e-8, atol=1e-20)
    for row in rows:
        actual, saved = row["newton"], expected[row["iteration"]]["newton"]
        assert actual["linear"]["steps"] == saved["linear"]["steps"]
        assert len(actual["regularization_retries"]) == len(
            saved["regularization_retries"]
        )
        assert actual["line_search_trials"] == saved["line_search_trials"]
    if profiler:
        profiler.dump_stats(str(output.with_suffix(".prof")))
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("tottime").print_stats(50)
        stats.sort_stats("cumulative").print_stats(50)
        output.with_suffix(".txt").write_text(stream.getvalue())
    result = {
        "mode": mode,
        "start_iteration": start,
        "steps": steps,
        "wall_seconds": seconds,
        "seconds_per_step": seconds / steps,
        "operation_counts": dict(problem.counts),
        "final_gradient": final_grad,
        "final_energy": final_energy,
        "gradient_replay_absolute_error": abs(final_grad - target["grad_norm"]),
        "energy_replay_absolute_error": abs(final_energy - target["energy"]),
        "matched_saved_pcg_iterations_and_shift_retry_counts": True,
        "accepted_pcg_iterations": sum(r["newton"]["linear"]["steps"] for r in rows),
        "rejected_linear_systems": sum(
            len(r["newton"]["regularization_retries"]) for r in rows
        ),
        "trace": rows,
        "hessian_backend": backend,
        "backend_before_timing": backend_before,
        "backend_after_timing": hessian.report(),
        "operator_relative_errors": operator_errors,
        "reference_true_relative_residuals": reference_residuals,
    }
    if timer:
        result["timing"] = timer.report()
        result["timing_totals"] = flatten(result["timing"]["tree"])
        total_exclusive = sum(
            row["exclusive_seconds"] for row in result["timing_totals"].values()
        )
        assert np.isclose(
            total_exclusive, result["timing"]["tree"]["window"]["inclusive_seconds"]
        )
    write_json(output.with_suffix(".json"), result)
    return result


def main(cfg: Config) -> None:
    assert cfg.steps == 10 and cfg.repeats >= 1 and cfg.ipc_threads == 4
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((cfg.run_dir / "summary.json").read_text())
    protocol = json.loads((cfg.run_dir / "protocol.json").read_text())
    assert sha256(cfg.run_dir / "protocol.json") == summary["protocol"]["sha256"]
    expected = [
        json.loads(line)
        for line in (cfg.run_dir / "trace.jsonl").read_text().splitlines()
    ]
    source_manifest = json.loads((cfg.run_dir / "provenance.json").read_text())[
        "sources"
    ]
    source_roots = {
        "experiment": JOINT / "src",
        "apple": ROOT / "src/liblaf/apple",
        "tensor-reference": ROOT / "exp/2026/09/07/tensor-active-stress/src",
        "solver-performance": SOLVERS / "src",
    }
    for name, digest in source_manifest.items():
        family, relative = name.split("/", 1)
        assert sha256(cfg.run_dir / "sources" / name) == digest
        if family in source_roots:
            assert sha256(source_roots[family] / relative) == digest, name
        elif relative == "model_binding.py":
            assert sha256(GROUP / "src/model_binding.py") == digest
    (cfg.output_dir / "gpu-before.txt").write_text(gpu_snapshot())
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    setup_started = time.perf_counter()
    neutral = load_current_binding(
        Path(protocol["config"]["neutral_dir"]), cfg.output_dir
    )
    install_exact_bulk_diagonal()
    physics, baseline = build_eye_collision_physics(
        neutral, Path(protocol["config"]["eyes_dir"])
    )
    model = physics.runtime.forward.model
    model.set_materials(baseline)
    model.dof_map.fixed_values = physics.boundary(torch.zeros(6)).detach().clone()
    model.collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=1e-10, max_iterations=100000
    )
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    shutil.copy2(__file__, cfg.output_dir / Path(__file__).name)
    shutil.copy2(
        SOLVERS / "src/inverse_timing.py", cfg.output_dir / "inverse_timing.py"
    )
    backend_sources = ROOT / "src/liblaf/apple/forward/hessian"
    shutil.copytree(
        backend_sources,
        cfg.output_dir / "hessian-sources",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    receipt = {
        "schema": "neutral-matched-window-profile-v1",
        "config": cfg.model_dump(mode="json"),
        "production_summary": record(cfg.run_dir / "summary.json"),
        "production_trace": record(cfg.run_dir / "trace.jsonl"),
        "production_sources": record(cfg.run_dir / "provenance.json"),
        "baseline_numerical_sources_verified_unchanged": True,
        "backend_sources": {
            str(p.relative_to(backend_sources)): record(p)
            for p in sorted(backend_sources.glob("*.py"))
        },
        "model_setup_seconds_excluded": setup_seconds,
        "scope": "Replays 0..10, 100..110, 190..200; no extension or selected-neutral update. Imports, setup, contact state construction, start gradient, checkpoint I/O, logging and Comet upload excluded. End gradient and energy included.",
        "timing_limits": "Shared GPU load; synchronized nested instrumentation perturbs overlap. Unsynchronized baseline repetitions are performance observations; profile percentages describe instrumented workload. Python cProfile is separate from synchronized profiling.",
        "runtime": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "ipc_threads": ipctk.get_num_threads(),
        },
        "windows": [],
    }
    for start, checkpoint in (
        (0, cfg.cold_run_dir / "initial.npz"),
        (100, cfg.run_dir / "initial.npz"),
        (190, cfg.run_dir / "checkpoint-190.npz"),
    ):
        with np.load(checkpoint, allow_pickle=False) as archive:
            seed = archive["displacement_m"].copy()
        directory = cfg.output_dir / f"window-{start:03d}"
        directory.mkdir()
        # Warm all numerical operations before measuring this window from a fresh state.
        replay(
            physics,
            seed,
            start,
            1,
            protocol["solver"],
            expected,
            "warmup",
            directory / "warmup",
            cfg.hessian_backend,
        )
        runs = [
            replay(
                physics,
                seed,
                start,
                cfg.steps,
                protocol["solver"],
                expected,
                "baseline",
                directory / f"baseline-{repeat}",
                cfg.hessian_backend,
            )
            for repeat in range(cfg.repeats)
        ]
        runs.extend(
            replay(
                physics,
                seed,
                start,
                cfg.steps,
                protocol["solver"],
                expected,
                mode,
                directory / mode,
                cfg.hessian_backend,
            )
            for mode in ("validation", "cprofile", "synchronized")
        )
        baseline_times = [
            row["wall_seconds"] for row in runs if row["mode"] == "baseline"
        ]
        window = {
            "start_iteration": start,
            "checkpoint": record(checkpoint),
            "baseline_median_seconds": float(np.median(baseline_times)),
            "baseline_min_seconds": min(baseline_times),
            "baseline_max_seconds": max(baseline_times),
            "runs": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("trace", "timing")
                }
                for row in runs
            ],
        }
        receipt["windows"].append(window)
        write_json(cfg.output_dir / "progress.json", receipt)
        LOG.info(
            "Window %d..%d: baseline median %.3f s, synchronized %.3f s",
            start,
            start + cfg.steps,
            np.median(baseline_times),
            runs[-1]["wall_seconds"],
        )
        cherries.set_step(start)
        cherries.log_metrics(
            {
                "profile/baseline_seconds": float(np.median(baseline_times)),
                "profile/synchronized_seconds": runs[-1]["wall_seconds"],
            }
        )
    (cfg.output_dir / "gpu-after.txt").write_text(gpu_snapshot())
    receipt["success"] = True
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
