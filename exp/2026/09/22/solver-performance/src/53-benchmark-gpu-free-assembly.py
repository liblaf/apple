# ruff: noqa: B023, C901, E402, PLR0912, PLR0915, PLW0108, PT018, SLF001
"""Compare CPU-merged and GPU-resident exact free-space Hessian assembly."""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "hessian_benchmark", HERE / "49-benchmark-hessian-representations.py"
)
assert spec is not None and spec.loader is not None
previous = importlib.util.module_from_spec(spec)
__import__("sys").modules[spec.name] = previous
spec.loader.exec_module(previous)
cold, benchmark = previous.cold, previous.benchmark
from accelerated_solvers import pcg
from adjoint_tolerance_common import build_fixed_state_context
from assembled_fem_hvp import AssembledFemHvp
from cudss_direct import CudssDirect
from free_sparse_hessian import FreeSparseHessian
from gpu_contact import install_gpu_contact
from gpu_free_sparse_hessian import GpuFreeSparseHessian


class Config(previous.Config):
    output_dir: Path = cold.EXPERIMENT / "data/gpu-free-assembly-001"
    hessian_baseline: Path = (
        cold.EXPERIMENT / "data/hessian-representations-001/summary.json"
    )
    setup_repeats: int = 3
    pcg_repeats: int = 3
    direct_rtol: float = 1e-7


def timed(operation: Any) -> tuple[Any, float]:
    return previous.timed(operation)


def memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated": torch.cuda.memory_allocated(),
        "torch_reserved": torch.cuda.memory_reserved(),
        "torch_peak_allocated": torch.cuda.max_memory_allocated(),
        "cuda_free": free,
        "cuda_total": total,
    }


def sparse_apply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, vector[:, None])[:, 0]


def lower_apply(lower: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric matrix represented by diagonal-inclusive lower CSR."""
    coo = lower.to_sparse_coo()
    rows, cols = coo.indices()
    diagonal = torch.zeros_like(vector)
    mask = rows == cols
    diagonal.index_add_(0, rows[mask], coo.values()[mask])
    return (
        sparse_apply(lower, vector)
        + sparse_apply(lower.transpose(0, 1), vector)
        - diagonal * vector
    )


def vectors(rhs: torch.Tensor) -> list[torch.Tensor]:
    generator = torch.Generator(device=rhs.device).manual_seed(20260922)
    random = [
        torch.randn(rhs.shape, dtype=rhs.dtype, device=rhs.device, generator=generator)
        for _ in range(2)
    ]
    return [*random, rhs / torch.linalg.vector_norm(rhs)]


def check_representation(
    assembly: Any, reference: Any, rhs: torch.Tensor, shift: float
) -> dict[str, list[float]]:
    matrix_errors, lower_errors = [], []
    for vector in vectors(rhs):
        expected = reference(vector) + shift * vector
        matrix_error = float(
            torch.linalg.vector_norm(sparse_apply(assembly.matrix, vector) - expected)
            / torch.linalg.vector_norm(expected)
        )
        lower_error = float(
            torch.linalg.vector_norm(lower_apply(assembly.lower, vector) - expected)
            / torch.linalg.vector_norm(expected)
        )
        assert matrix_error < 1e-10, matrix_error
        assert lower_error < 1e-10, lower_error
        matrix_errors.append(matrix_error)
        lower_errors.append(lower_error)
    return {
        "matrix_relative_errors": matrix_errors,
        "lower_relative_errors": lower_errors,
    }


def pcg_receipts(
    assembly: Any,
    diagonal: torch.Tensor,
    rhs: torch.Tensor,
    reference: Any,
    shift: float,
    repeats: int,
) -> list[dict[str, Any]]:
    receipts = []
    for repeat in range(repeats):
        (solution, info), seconds = timed(
            lambda: pcg(
                lambda p: sparse_apply(assembly.matrix, p),
                lambda r: r / (diagonal + shift),
                rhs,
                rtol=1e-3,
            )
        )
        residual = float(
            torch.linalg.vector_norm(reference(solution) + shift * solution - rhs)
            / torch.linalg.vector_norm(rhs)
        )
        assert residual <= 1.05e-3, residual
        receipts.append(
            {
                "repeat": repeat,
                "seconds": seconds,
                "reference_true_relative_residual": residual,
                **info,
            }
        )
    return receipts


def direct_receipt(
    solver: CudssDirect,
    lower: torch.Tensor,
    rhs: torch.Tensor,
    reference: Any,
    shift: float,
    *,
    reuse: bool,
) -> dict[str, Any]:
    if reuse:
        solver.update_values(lower.values().contiguous())
        factor, factor_seconds = timed(lambda: solver.refactorize())
    else:
        analysis, analysis_seconds = timed(solver.analyze)
        assert analysis["memory_estimates_status"] == 0, analysis
        factor, factor_seconds = timed(solver.factorize)
    assert factor["info_status"] == 0 and factor["info"] == 0, factor
    solution, solve_seconds = timed(lambda: solver.solve(rhs))
    residual = float(
        torch.linalg.vector_norm(reference(solution) + shift * solution - rhs)
        / torch.linalg.vector_norm(rhs)
    )
    assert residual <= 1e-7, residual
    receipt = {
        "success": True,
        "status": "success",
        "factorizations": [{"seconds": factor_seconds, "info": factor}],
        "solves": [
            {"seconds": solve_seconds, "reference_true_relative_residual": residual}
        ],
        "reused_analysis": reuse,
    }
    if not reuse:
        receipt.update({"analysis": analysis, "analysis_seconds": analysis_seconds})
    return receipt


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.setup_repeats >= 3 and cfg.pcg_repeats >= 3 and cfg.linear_rtol == 1e-3
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
        "schema": "gpu-free-assembly-v1",
        "config": cfg.model_dump(mode="json"),
        "scope": "Two frozen Newton linear systems only. Exact FEM plus exact owned IPC Hessian, restricted to free DOFs. CPU SciPy merge/restriction is compared with GPU-resident cached FEM/contact union assembly. No forward solve, inverse update, or production replacement.",
        "timing": "CUDA-synchronized wall timing; constructors, every numeric refresh, contact state/evaluation, and matrixfree reference validation are separate. Scalar-Jacobi PCG uses the same RHS, diagonal, shift and rtol 1e-3 in both arms.",
        "validation": "Matrices and diagonal-inclusive lower storage are independently compared with the original matrixfree FEM plus CPU IPC HVP. cuDSS Cholesky is an additional optimized-lower residual check, not the main timing comparison.",
        "physical_source_changes": changes,
        "baseline_sha256": benchmark.sha256(cfg.hessian_baseline),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "processes_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "torch_version": torch.__version__,
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
    states = (
        ("cold_loaded_neutral", cold_u),
        ("saved_smile", context.full_displacement),
    )
    handle = install_gpu_contact(model)
    results: list[dict[str, Any]] = []
    fem = None
    optimized = None
    direct = None
    direct_pattern: str | None = None
    cold_state_data: tuple[Any, torch.Tensor, torch.Tensor, Any, float] | None = None
    try:
        with torch.no_grad():
            for index, (state_name, full_u) in enumerate(states):
                baseline = base["results"][index]
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
                _, contact_cpu_assembly_seconds = timed(
                    lambda: handle.adapter._assemble(state.collision, state.u)
                )
                rhs = -model.dof_map.to_free_grad(model.grad(state))
                force = float(torch.linalg.vector_norm(rhs))
                assert (
                    abs(force - baseline["force_l2"])
                    <= 1e-8 * baseline["force_l2"] + 1e-20
                )
                diagonal = model.dof_map.to_free_hess_diag(model.hess_diag(state)).abs()
                shift = float(baseline["shift"])

                def reference(
                    vector: torch.Tensor, current: Any = state
                ) -> torch.Tensor:
                    full = model.dof_map.to_full_grad(vector)
                    output = torch.zeros_like(full)
                    model.warp_model.hess_prod(current.u, full, output)
                    handle.adapter.original_hess_prod(
                        current.collision, current.u, full, output
                    )
                    return model.dof_map.to_free_grad(output)

                if fem is None:
                    fem, fem_constructor_seconds = timed(
                        lambda: AssembledFemHvp(model, state)
                    )
                else:
                    _, fem_constructor_seconds = timed(lambda: fem.setup(state))
                row: dict[str, Any] = {
                    "state": state_name,
                    "free_dofs": model.n_free,
                    "shift": shift,
                    "force_l2": force,
                    "displacement_sha256": benchmark.tensor_sha256(state.u),
                    "rhs_sha256": benchmark.tensor_sha256(rhs),
                    "contact_state_seconds": contact_state_seconds,
                    "contact_cpu_assembly_seconds": contact_cpu_assembly_seconds,
                    "fem_preparation_seconds": fem_constructor_seconds,
                    "old": {},
                    "gpu": {},
                }
                _, matrixfree_reference_hvp_seconds = timed(
                    lambda: reference(vectors(rhs)[0])
                )
                row["matrixfree_cpu_ipc_hvp_seconds"] = matrixfree_reference_hvp_seconds
                benchmark.write_json(
                    cfg.output_dir / "status.json",
                    {"running": True, "state": state_name, "stage": "old_constructor"},
                )
                torch.cuda.reset_peak_memory_stats()
                old, old_constructor_seconds = timed(
                    lambda: FreeSparseHessian(model, state, fem, shift=shift)
                )
                old_first_metadata = copy.deepcopy(old.metadata)
                old_setup_samples = []
                for _ in range(cfg.setup_repeats):

                    def old_setup(
                        assembly: Any = old,
                        current: Any = state,
                        current_fem: Any = fem,
                        current_shift: float = shift,
                    ) -> None:
                        assembly.setup(current, current_fem, shift=current_shift)

                    _, seconds = timed(old_setup)
                    old_setup_samples.append(seconds)
                old_checks = check_representation(old, reference, rhs, shift)
                row["old"] = {
                    "constructor_seconds": old_constructor_seconds,
                    "first_state_setup_seconds": old_constructor_seconds,
                    "first_state_setup_metadata": old_first_metadata,
                    "setup_refresh_seconds": old_setup_samples,
                    "metadata": copy.deepcopy(old.metadata),
                    "persistent_bytes": old.persistent_bytes,
                    "torch_memory": memory(),
                    **old_checks,
                    "pcg": pcg_receipts(
                        old, diagonal, rhs, reference, shift, cfg.pcg_repeats
                    ),
                }
                benchmark.write_json(cfg.output_dir / f"{state_name}-old.json", row)

                torch.cuda.reset_peak_memory_stats()
                if optimized is None:
                    optimized, gpu_constructor_seconds = timed(
                        lambda: GpuFreeSparseHessian(model, state, fem, shift=shift)
                    )
                    gpu_constructor = gpu_constructor_seconds
                else:
                    _, gpu_constructor_seconds = timed(
                        lambda: optimized.setup(state, fem, shift=shift)
                    )
                    gpu_constructor = None
                gpu_first_metadata = copy.deepcopy(optimized.metadata)
                gpu_setup_samples = []
                for _ in range(cfg.setup_repeats):

                    def gpu_setup(
                        assembly: Any = optimized,
                        current: Any = state,
                        current_fem: Any = fem,
                        current_shift: float = shift,
                    ) -> None:
                        assembly.setup(current, current_fem, shift=current_shift)

                    _, seconds = timed(gpu_setup)
                    gpu_setup_samples.append(seconds)
                gpu_checks = check_representation(optimized, reference, rhs, shift)
                pattern_diagnostic = {
                    "old_physical_pattern_hash": old.metadata[
                        "physical_hessian_pattern_hash"
                    ],
                    "gpu_physical_pattern_hash": optimized.metadata[
                        "physical_hessian_pattern_hash"
                    ],
                    "equal": old.metadata["physical_hessian_pattern_hash"]
                    == optimized.metadata["physical_hessian_pattern_hash"],
                    "old_free_nnz": old.metadata["free_nnz"],
                    "gpu_free_nnz": optimized.metadata["free_nnz"],
                }
                gpu_receipt = {
                    "constructor_seconds": gpu_constructor,
                    "first_state_setup_seconds": gpu_constructor_seconds,
                    "first_state_setup_metadata": gpu_first_metadata,
                    "setup_refresh_seconds": gpu_setup_samples,
                    "metadata": copy.deepcopy(optimized.metadata),
                    "persistent_bytes": optimized.persistent_bytes,
                    "torch_memory": memory(),
                    "pattern_diagnostic_against_old": pattern_diagnostic,
                    **gpu_checks,
                    "pcg": pcg_receipts(
                        optimized, diagonal, rhs, reference, shift, cfg.pcg_repeats
                    ),
                }
                pattern = optimized.metadata["matrix_pattern_hash"]
                if direct is None:
                    direct = CudssDirect(optimized.lower, mode="spd")
                    gpu_receipt["cudss_spd"] = direct_receipt(
                        direct, optimized.lower, rhs, reference, shift, reuse=False
                    )
                    gpu_receipt["cudss_pattern_reused"] = False
                elif pattern == direct_pattern:
                    gpu_receipt["cudss_spd"] = direct_receipt(
                        direct, optimized.lower, rhs, reference, shift, reuse=True
                    )
                    gpu_receipt["cudss_pattern_reused"] = True
                else:
                    direct.close()
                    direct = CudssDirect(optimized.lower, mode="spd")
                    gpu_receipt["cudss_spd"] = direct_receipt(
                        direct, optimized.lower, rhs, reference, shift, reuse=False
                    )
                    gpu_receipt["cudss_pattern_reused"] = False
                direct_pattern = pattern
                row["gpu"] = gpu_receipt
                results.append(row)
                benchmark.write_json(cfg.output_dir / f"{state_name}.json", row)
                print(
                    json.dumps(
                        {
                            "state": state_name,
                            "old_setup_seconds": old_setup_samples,
                            "gpu_setup_seconds": gpu_setup_samples,
                            "gpu_pattern": pattern,
                        }
                    ),
                    flush=True,
                )
                if index == 0:
                    cold_state_data = (state, rhs, diagonal, reference, shift)
                del old
                gc.collect()

            assert cold_state_data is not None and optimized is not None
            state, rhs, _diagonal, reference, shift = cold_state_data
            back, back_seconds = timed(lambda: optimized.setup(state, fem, shift=shift))
            assert back is None
            back_checks = check_representation(optimized, reference, rhs, shift)
            transition_back_to_cold = {
                "setup_seconds": back_seconds,
                "metadata": copy.deepcopy(optimized.metadata),
                **back_checks,
            }
            assert (
                transition_back_to_cold["metadata"]["matrix_pattern_hash"]
                == results[0]["gpu"]["metadata"]["matrix_pattern_hash"]
            )
            summary = {
                "schema": "gpu-free-assembly-v1",
                "protocol": protocol,
                "results": results,
                "transition_back_to_cold": transition_back_to_cold,
            }
            benchmark.write_json(cfg.output_dir / "summary.json", summary)
            benchmark.write_json(
                cfg.output_dir / "status.json", {"running": False, "success": True}
            )
            cherries.log_metric("states_completed", len(results))
            print(
                json.dumps({"success": True, "states_completed": len(results)}),
                flush=True,
            )
    finally:
        if direct is not None:
            direct.close()
        handle.uninstall()


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
