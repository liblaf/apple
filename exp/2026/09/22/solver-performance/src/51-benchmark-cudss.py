# ruff: noqa: E402, PLR0915, SLF001
"""Exact free-space cuDSS versus PCG on the two frozen Smile states."""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import resource
import sys
import threading
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "hessian_benchmark", HERE / "49-benchmark-hessian-representations.py"
)
assert spec and spec.loader
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
cold, benchmark = previous.cold, previous.benchmark
from accelerated_solvers import pcg
from adjoint_tolerance_common import build_fixed_state_context
from assembled_fem_hvp import AssembledFemHvp
from cudss_direct import CudssDirect
from free_sparse_hessian import FreeSparseHessian
from gpu_contact import install_gpu_contact


class Config(previous.Config):
    output_dir: Path = cold.EXPERIMENT / "data/cudss-comparison-001"
    hessian_baseline: Path = (
        cold.EXPERIMENT / "data/hessian-representations-001/summary.json"
    )
    factor_repeats: int = 3
    solve_repeats: int = 5
    unshifted_probe: bool = True
    direct_rtol: float = 1e-7


class MemoryInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class SampledGpuPeak:
    """Sample whole-device NVML used bytes every 10 ms; not an exact allocator peak."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0
        self.stop = threading.Event()
        self.peak = 0
        self.samples = 0
        self.before = self.read()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def read(self) -> int:
        info = MemoryInfo()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) == 0
        self.samples += 1
        self.peak = max(self.peak, int(info.used))
        return int(info.used)

    def poll(self) -> None:
        while not self.stop.wait(0.01):
            self.read()

    def __enter__(self) -> SampledGpuPeak:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.read()
        self.stop.set()
        self.thread.join()
        assert self.lib.nvmlShutdown() == 0

    def record(self) -> dict:
        return {
            "device_before_bytes": self.before,
            "sampled_peak_device_bytes": self.peak,
            "sampled_peak_delta_bytes": self.peak - self.before,
            "samples": self.samples,
            "interval_seconds": 0.01,
        }


def timed(operation: Any) -> tuple[Any, float]:
    return previous.timed(operation)


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.factor_repeats >= 3 and cfg.solve_repeats >= 3
    base = json.loads(cfg.hessian_baseline.read_text())
    original = json.loads(cfg.baseline.read_text())
    assert benchmark.sha256(cfg.checkpoint) == base["protocol"]["checkpoint_sha256"]
    assert (
        benchmark.sha256(cfg.neutral_checkpoint)
        == base["protocol"]["neutral_checkpoint_sha256"]
    )
    for name, digest in original["protocol"]["inputs"].items():
        assert benchmark.sha256(cfg.inputs_dir / name) == digest
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    old_sources = json.loads(
        (cfg.hessian_baseline.parent / "provenance.json").read_text()
    )["sources"]
    changes = [
        name
        for name, digest in old_sources.items()
        if name.startswith(("apple/", "joint-experiment/", "tensor-reference/"))
        and provenance["sources"][name] != digest
    ]
    assert not changes, changes
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    cold.configure_cuda()
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "schema": "cudss-frozen-v1",
        "scope": "Sequential single-GPU frozen linear systems, exact FEM plus IPC contact restricted to free DOFs. No forward/inverse optimization. Common Newton shifts from previous benchmark. Saved equilibrium also tested with unshifted LDLT as a separate residual probe, not an adjoint benchmark.",
        "baseline_sha256": benchmark.sha256(cfg.hessian_baseline),
        "physical_source_changes": changes,
        "gpu": base["protocol"]["gpu"],
        "cudss_library": os.environ.get("CUDSS_LIBRARY"),
        "timing": "CUDA synchronized wall time. Separate FEM construction, free CSR construction, symbolic analysis, first factorization, repeated numeric factorizations, triangular solves. PCG uses same assembled full free CSR with original diagonal and common shift. No overlapping GPU work.",
        "memory": "cuDSS symbolic estimates plus whole-device NVML sampling every10ms. Torch allocator counters exclude cuDSS; NVML observed peak is sampled, not exact.",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    context = build_fixed_state_context(
        checkpoint_path=cfg.checkpoint,
        inputs_dir=cfg.inputs_dir,
        output_dir=cfg.output_dir / "fitter",
        ipc_threads=cfg.ipc_threads,
    )
    model = context.fitter.runtime.forward.model
    model.set_materials(context.materials)
    model.dof_map.fixed_values = context.fixed_values
    neutral = torch.load(
        cfg.neutral_checkpoint, map_location="cuda", weights_only=False
    )
    pose = context.runner.hinge_pose(
        torch.zeros_like(context.jaw), context.fitter.hinge_axis
    )
    cold_u = context.fitter.physics.full_skull.extend_seed(
        neutral["displacement_m"], pose
    )
    handle = install_gpu_contact(model)
    results = []
    fem = None
    try:
        with torch.no_grad():
            for state_index, (state_name, full_u) in enumerate(
                (
                    ("cold_loaded_neutral", cold_u),
                    ("saved_smile", context.full_displacement),
                )
            ):
                baseline = base["results"][state_index]
                assert baseline["state"] == state_name
                state = model.State(
                    u=model.dof_map.to_full(model.dof_map.to_free(full_u)).detach()
                )
                assert (
                    benchmark.tensor_sha256(state.u) == baseline["displacement_sha256"]
                )
                state.collision, contact_state_seconds = timed(
                    lambda: model.collision.state_at(state.u)
                )
                handle.adapter.invalidate()
                _, contact_assembly_seconds = timed(
                    lambda: handle.adapter._assemble(state.collision, state.u)
                )
                rhs = -model.dof_map.to_free_grad(model.grad(state))
                force = float(torch.linalg.vector_norm(rhs))
                assert (
                    abs(force - baseline["force_l2"])
                    <= 1e-8 * baseline["force_l2"] + 1e-20
                )
                diagonal = model.dof_map.to_free_hess_diag(model.hess_diag(state)).abs()

                def reference(vector: torch.Tensor) -> torch.Tensor:
                    full = model.dof_map.to_full_grad(vector)
                    output = torch.zeros_like(full)
                    model.warp_model.hess_prod(state.u, full, output)
                    handle.adapter.original_hess_prod(
                        state.collision, state.u, full, output
                    )
                    return model.dof_map.to_free_grad(output)

                if fem is None:
                    fem, fem_seconds = timed(lambda: AssembledFemHvp(model, state))
                else:
                    _, fem_seconds = timed(lambda: fem.setup(state))
                row = {
                    "state": state_name,
                    "force_l2": force,
                    "free_dofs": model.n_free,
                    "displacement_sha256": benchmark.tensor_sha256(state.u),
                    "rhs_sha256": benchmark.tensor_sha256(rhs),
                    "contact_state_seconds": contact_state_seconds,
                    "contact_assembly_seconds": contact_assembly_seconds,
                    "fem_seconds": fem_seconds,
                    "fem_metadata": dict(fem.metadata),
                    "systems": [],
                }
                shifts = [("matched_newton", baseline["shift"])]
                if (
                    state_name == "saved_smile"
                    and cfg.unshifted_probe
                    and baseline["shift"] != 0
                ):
                    shifts.append(("unshifted_residual_probe", 0.0))
                for system_name, shift in shifts:
                    print(
                        json.dumps(
                            {
                                "state": state_name,
                                "system": system_name,
                                "shift": shift,
                                "stage": "assemble_free",
                            }
                        ),
                        flush=True,
                    )
                    setup_rss_before = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    free, free_seconds = timed(
                        lambda: FreeSparseHessian(model, state, fem, shift=shift)
                    )
                    setup_rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    shift_seconds = 0.0
                    generator = torch.Generator(device="cuda").manual_seed(20260922)
                    checks = []
                    for vector in [
                        torch.randn(
                            rhs.shape,
                            dtype=rhs.dtype,
                            device=rhs.device,
                            generator=generator,
                        )
                        for _ in range(2)
                    ] + [rhs / torch.linalg.vector_norm(rhs)]:
                        actual = torch.sparse.mm(free.matrix, vector[:, None])[:, 0]
                        expected = reference(vector) + shift * vector
                        error = float(
                            torch.linalg.vector_norm(actual - expected)
                            / torch.linalg.vector_norm(expected)
                        )
                        assert error < 1e-10, error
                        checks.append(error)
                    system = {
                        "system": system_name,
                        "shift": shift,
                        "free_matrix_seconds": free_seconds,
                        "shifted_reconstruction_seconds": shift_seconds,
                        "matrix_metadata": free.metadata,
                        "cpu_process_max_rss_kib_before_setup": setup_rss_before,
                        "cpu_process_max_rss_kib_after_setup": setup_rss_after,
                        "operator_relative_errors": checks,
                        "pcg": [],
                        "direct": [],
                    }
                    if system_name == "matched_newton":
                        for repeat in range(cfg.cg_repeats):
                            (solution, info), seconds = timed(
                                lambda: pcg(
                                    lambda p: torch.sparse.mm(free.matrix, p[:, None])[
                                        :, 0
                                    ],
                                    lambda r: r / (diagonal + shift),
                                    rhs,
                                    rtol=cfg.linear_rtol,
                                )
                            )
                            residual = float(
                                torch.linalg.vector_norm(
                                    reference(solution) + shift * solution - rhs
                                )
                                / torch.linalg.vector_norm(rhs)
                            )
                            assert residual <= 1.05 * cfg.linear_rtol
                            system["pcg"].append(
                                {
                                    "seconds": seconds,
                                    "reference_true_relative_residual": residual,
                                    **info,
                                }
                            )
                        print(
                            json.dumps(
                                {
                                    "state": state_name,
                                    "system": system_name,
                                    "pcg": system["pcg"],
                                }
                            ),
                            flush=True,
                        )
                    for mode in (
                        ("spd", "symmetric")
                        if system_name == "matched_newton"
                        else ("symmetric",)
                    ):
                        benchmark.write_json(
                            cfg.output_dir / "status.json",
                            {
                                "running": True,
                                "state": state_name,
                                "system": system_name,
                                "mode": mode,
                                "stage": "analysis",
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "state": state_name,
                                    "system": system_name,
                                    "mode": mode,
                                    "stage": "analysis",
                                }
                            ),
                            flush=True,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        solver = CudssDirect(free.lower, mode=mode)
                        record = {"mode": mode, "factorizations": [], "solves": []}
                        try:
                            with SampledGpuPeak() as peak:
                                analysis, seconds = timed(solver.analyze)
                            assert analysis["memory_estimates_status"] == 0, analysis
                            record["analysis"] = analysis
                            record["analysis_seconds"] = seconds
                            record["analysis_memory"] = peak.record()
                            print(
                                json.dumps(
                                    {
                                        "state": state_name,
                                        "mode": mode,
                                        "analysis_seconds": seconds,
                                        "analysis": analysis,
                                    }
                                ),
                                flush=True,
                            )
                            available, _ = torch.cuda.mem_get_info()
                            estimates = analysis.get("memory_estimates", [])
                            predicted = int(estimates[1]) if len(estimates) >= 2 else 0
                            if predicted > available - 1024**3:
                                record["success"] = False
                                record["reason"] = (
                                    "symbolic estimated peak exceeds available GPU memory with1GiBheadroom"
                                )
                            else:
                                record["success"] = True
                                for repeat in range(cfg.factor_repeats):
                                    benchmark.write_json(
                                        cfg.output_dir / "status.json",
                                        {
                                            "running": True,
                                            "state": state_name,
                                            "system": system_name,
                                            "mode": mode,
                                            "stage": "factorization",
                                            "repeat": repeat,
                                        },
                                    )
                                    with SampledGpuPeak() as peak:
                                        info, seconds = timed(solver.factorize)
                                    assert info["info_status"] == 0, info
                                    record["factorizations"].append(
                                        {
                                            "seconds": seconds,
                                            "memory": peak.record(),
                                            "info": info,
                                        }
                                    )
                                    print(
                                        json.dumps(
                                            {
                                                "state": state_name,
                                                "system": system_name,
                                                "mode": mode,
                                                "factor_repeat": repeat,
                                                "seconds": seconds,
                                                "info": info,
                                                "memory": peak.record(),
                                            }
                                        ),
                                        flush=True,
                                    )
                                    if int(info.get("info", 0)) != 0:
                                        record["success"] = False
                                        record["reason"] = (
                                            "factorization returned nonzero cuDSS info"
                                        )
                                        break
                                    solution, solve_seconds = timed(
                                        lambda: solver.solve(rhs)
                                    )
                                    residual = float(
                                        torch.linalg.vector_norm(
                                            reference(solution) + shift * solution - rhs
                                        )
                                        / torch.linalg.vector_norm(rhs)
                                    )
                                    record["solves"].append(
                                        {
                                            "kind": "after_factorization",
                                            "seconds": solve_seconds,
                                            "reference_true_relative_residual": residual,
                                        }
                                    )
                                    assert residual <= cfg.direct_rtol, (
                                        mode,
                                        residual,
                                        record,
                                    )
                                if record["success"]:
                                    for _ in range(cfg.solve_repeats):
                                        solution, seconds = timed(
                                            lambda: solver.solve(rhs)
                                        )
                                        residual = float(
                                            torch.linalg.vector_norm(
                                                reference(solution)
                                                + shift * solution
                                                - rhs
                                            )
                                            / torch.linalg.vector_norm(rhs)
                                        )
                                        assert residual <= cfg.direct_rtol, residual
                                        record["solves"].append(
                                            {
                                                "kind": "reuse_factors",
                                                "seconds": seconds,
                                                "reference_true_relative_residual": residual,
                                            }
                                        )
                            record["process_max_rss_kib"] = resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                        finally:
                            solver.close()
                        system["direct"].append(record)
                        benchmark.write_json(
                            cfg.output_dir / f"{state_name}-{system_name}.json",
                            {**row, "current_system": system},
                        )
                        del solver
                    row["systems"].append(system)
                    del free
                    gc.collect()
                results.append(row)
                benchmark.write_json(cfg.output_dir / f"{state_name}.json", row)
            benchmark.write_json(
                cfg.output_dir / "summary.json",
                {"schema": "cudss-frozen-v1", "protocol": protocol, "results": results},
            )
            benchmark.write_json(
                cfg.output_dir / "status.json", {"running": False, "success": True}
            )
            cherries.log_metric("states_completed", len(results))
            print(
                json.dumps({"success": True, "states_completed": len(results)}),
                flush=True,
            )
    finally:
        handle.uninstall()


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
