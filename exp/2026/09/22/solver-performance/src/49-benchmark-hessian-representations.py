# ruff: noqa: E402, PLR0915, SLF001
"""Sequential exact frozen-state HVP representation comparison on one GPU."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import statistics
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
    "cold_forward", HERE / "43-compare-cold-forward.py"
)
assert spec is not None and spec.loader is not None
cold = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cold
spec.loader.exec_module(cold)
benchmark = cold.benchmark
from accelerated_solvers import LinearRejection, pcg
from adjoint_tolerance_common import build_fixed_state_context
from assembled_fem_hvp import AssembledFemHvp
from cached_fem_hvp import CachedFemHvp
from gpu_contact import install_gpu_contact


class Config(cold.Config):
    checkpoint: Path = (
        cold.EXPERIMENT
        / "data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt"
    )
    neutral_checkpoint: Path = (
        cold.EXPERIMENT
        / "data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt"
    )
    output_dir: Path = cold.EXPERIMENT / "data/hessian-representations-001"
    baseline: Path = cold.EXPERIMENT / "data/cold-forward-comparison-001/summary.json"
    hvp_repeats: int = 30
    cg_repeats: int = 3
    setup_repeats: int = 3


def stats(values: list[float]) -> dict:
    return {
        "samples": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def timed(operation: Any) -> tuple[Any, float]:
    return cold.synchronized_time(operation)


def digest_csr(matrix: Any) -> str:
    h = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        h.update(array.tobytes())
    return h.hexdigest()


def memory() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated": torch.cuda.memory_allocated(),
        "torch_reserved": torch.cuda.memory_reserved(),
        "cuda_free": free,
        "cuda_total": total,
    }


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.linear_rtol == 1e-3 and cfg.ipc_threads == 8
    assert cfg.hvp_repeats >= 20 and cfg.cg_repeats >= 3
    baseline = json.loads(cfg.baseline.read_text())
    assert (
        benchmark.sha256(cfg.checkpoint) == baseline["protocol"]["checkpoint"]["sha256"]
    )
    assert (
        benchmark.sha256(cfg.neutral_checkpoint)
        == baseline["protocol"]["neutral_checkpoint"]["sha256"]
    )
    for name, digest in baseline["protocol"]["inputs"].items():
        assert benchmark.sha256(cfg.inputs_dir / name) == digest
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    old_sources = json.loads((cfg.baseline.parent / "provenance.json").read_text())[
        "sources"
    ]
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
        "scope": "Frozen-state exact Newton-system microbenchmark; no forward or inverse optimization. Active stress, fixed neutral prestress, full bone and eye collision. Scalar diagonal PCG, common reference-selected shift, zero initial CG. Each variant runs sequentially on the same GPU.",
        "contact": "One exact CPU IPC CSR per owned state; shared GPU CSR for three FEM variants, CPU contact retained as separate current-route control.",
        "timing": "CUDA synchronized wall time. Construction first call includes JIT if any; setup refresh and repeated applications follow prewarm. Numeric refresh includes implementation allocations, material copies, and index transfers. CG timing includes free/full mapping, contact, diagonal preconditioning, scalar synchronization and convergence verification. Common contact setup reported separately.",
        "memory_scope": "Explicit persistent bytes are owned extra operator buffers. CUDA meminfo covers runtime; torch allocator metrics omit Warp/native allocations. Sparse CPU topology is reported separately.",
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
        "physical_source_changes": changes,
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "seed": 20260922,
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
    try:
        with torch.no_grad():
            for state_name, full_u in (
                ("cold_loaded_neutral", cold_u),
                ("saved_smile", context.full_displacement),
            ):
                state = model.State(
                    u=model.dof_map.to_full(model.dof_map.to_free(full_u)).detach()
                )
                state.collision, contact_state_seconds = timed(
                    lambda: model.collision.state_at(state.u)
                )
                handle.adapter.invalidate()
                gradient = model.dof_map.to_free_grad(model.grad(state))
                rhs = -gradient
                assert float(torch.linalg.vector_norm(rhs)) > 0
                # Assemble contact before hess_diag so its one-time work is visible.
                _, contact_assembly_seconds = timed(
                    lambda: handle.adapter._assemble(state.collision, state.u)
                )
                _, contact_upload_seconds = timed(
                    lambda: handle.adapter._upload(
                        state.collision, state.u, state.u.device, state.u.dtype
                    )
                )
                diagonal, diagonal_seconds = timed(
                    lambda: model.dof_map.to_free_hess_diag(
                        model.hess_diag(state)
                    ).abs()
                )
                assert torch.isfinite(diagonal).all() and float(diagonal.min()) > 0
                generator = torch.Generator(device="cuda").manual_seed(
                    cfg.seed if hasattr(cfg, "seed") else 20260922
                )
                vectors = [
                    torch.randn(
                        rhs.shape,
                        dtype=rhs.dtype,
                        device=rhs.device,
                        generator=generator,
                    )
                    for _ in range(2)
                ]
                vectors.append(rhs / torch.linalg.vector_norm(rhs))
                row = {
                    "state": state_name,
                    "full_points": model.n_points,
                    "free_dofs": model.n_free,
                    "displacement_sha256": benchmark.tensor_sha256(state.u),
                    "rhs_sha256": benchmark.tensor_sha256(rhs),
                    "diagonal_sha256": benchmark.tensor_sha256(diagonal),
                    "force_l2": float(torch.linalg.vector_norm(rhs)),
                    "contact_state_seconds": contact_state_seconds,
                    "contact_assembly_seconds": contact_assembly_seconds,
                    "contact_upload_seconds": contact_upload_seconds,
                    "diagonal_seconds": diagonal_seconds,
                    "contact_csr_sha256": digest_csr(state.collision.hess),
                    "contact_csr_nnz": state.collision.hess.nnz,
                    "variants": [],
                }
                print(
                    json.dumps(
                        {
                            "state": state_name,
                            "force_l2": row["force_l2"],
                            "stage": "reference_and_shift",
                        }
                    ),
                    flush=True,
                )

                def make_operator(cache: Any, gpu_contact: bool) -> Any:
                    contact = (
                        handle.adapter.hess_prod
                        if gpu_contact
                        else handle.adapter.original_hess_prod
                    )

                    def apply(vector: torch.Tensor) -> torch.Tensor:
                        full = model.dof_map.to_full_grad(vector)
                        if cache is None:
                            output = torch.zeros_like(full)
                            model.warp_model.hess_prod(state.u, full, output)
                        else:
                            output = cache.apply(full)
                        contact(state.collision, state.u, full, output)
                        return model.dof_map.to_free_grad(output)

                    return apply

                reference = make_operator(None, False)
                control = make_operator(None, True)
                ref_values, reference_prewarm_seconds = timed(
                    lambda: [reference(v) for v in vectors]
                )
                row["reference_prewarm_seconds"] = reference_prewarm_seconds
                shift = 0.0
                calibration = []
                for _ in range(10):
                    started = time.perf_counter()
                    try:
                        reference_solution, cg = pcg(
                            lambda p: control(p) + shift * p,
                            lambda r: r / (diagonal + shift),
                            rhs,
                            rtol=cfg.linear_rtol,
                        )
                        torch.cuda.synchronize()
                        calibration.append(
                            {
                                "shift": shift,
                                "seconds": time.perf_counter() - started,
                                "success": True,
                                **cg,
                            }
                        )
                        break
                    except LinearRejection as error:
                        torch.cuda.synchronize()
                        calibration.append(
                            {
                                "shift": shift,
                                "seconds": time.perf_counter() - started,
                                "success": False,
                                "error": str(error),
                            }
                        )
                        shift = (
                            float(diagonal.mean()) * 1e-6 if shift == 0 else 10 * shift
                        )
                else:
                    raise RuntimeError(
                        f"reference shift selection failed: {calibration}"
                    )
                row["shift"] = shift
                row["shift_calibration"] = calibration
                for name, constructor, gpu_contact in (
                    ("current_cpu_contact", None, False),
                    ("current_gpu_contact", None, True),
                    ("cached_fem_gpu_contact", CachedFemHvp, True),
                    ("sparse_fem_gpu_contact", AssembledFemHvp, True),
                ):
                    benchmark.write_json(
                        cfg.output_dir / "status.json",
                        {"running": True, "state": state_name, "variant": name},
                    )
                    print(
                        json.dumps(
                            {"state": state_name, "variant": name, "stage": "start"}
                        ),
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    before = memory()
                    torch.cuda.reset_peak_memory_stats()
                    cache, construction_seconds = (
                        timed(lambda: constructor(model, state))
                        if constructor
                        else (None, 0.0)
                    )
                    apply = make_operator(cache, gpu_contact)
                    # First apply compiles/cache-initializes the candidate as needed.
                    _, prewarm_seconds = timed(lambda: apply(vectors[0]))
                    setup_samples = (
                        [
                            timed(lambda: cache.setup(state))[1]
                            for _ in range(cfg.setup_repeats)
                        ]
                        if cache
                        else [0.0]
                    )
                    checks = []
                    values = []
                    for vector, expected in zip(vectors, ref_values, strict=True):
                        actual = apply(vector)
                        delta = actual - expected
                        relative = float(
                            torch.linalg.vector_norm(delta)
                            / torch.linalg.vector_norm(expected)
                        )
                        checks.append(
                            {
                                "relative_l2": relative,
                                "max_abs": float(delta.abs().max()),
                            }
                        )
                        assert relative < 1e-10, (name, state_name, checks)
                        values.append(actual)
                    lhs, right = (
                        torch.dot(vectors[0], values[1]),
                        torch.dot(vectors[1], values[0]),
                    )
                    symmetry = float(
                        (lhs - right).abs()
                        / (
                            torch.linalg.vector_norm(vectors[0])
                            * torch.linalg.vector_norm(values[1])
                            + torch.linalg.vector_norm(vectors[1])
                            * torch.linalg.vector_norm(values[0])
                        )
                    )
                    assert symmetry < 1e-10, symmetry
                    hvp_samples = [
                        timed(lambda: apply(vectors[0]))[1]
                        for _ in range(cfg.hvp_repeats)
                    ]
                    cg_samples = []
                    for repeat in range(cfg.cg_repeats):
                        calls = [0]

                        def shifted(p: torch.Tensor) -> torch.Tensor:
                            calls[0] += 1
                            return apply(p) + shift * p

                        (solution, info), seconds = timed(
                            lambda: pcg(
                                shifted,
                                lambda r: r / (diagonal + shift),
                                rhs,
                                rtol=cfg.linear_rtol,
                            )
                        )
                        true_residual = float(
                            torch.linalg.vector_norm(
                                reference(solution) + shift * solution - rhs
                            )
                            / torch.linalg.vector_norm(rhs)
                        )
                        assert true_residual <= 1.05 * cfg.linear_rtol
                        solution_difference = float(
                            torch.linalg.vector_norm(solution - reference_solution)
                            / torch.linalg.vector_norm(reference_solution)
                        )
                        cg_samples.append(
                            {
                                "repeat": repeat,
                                "seconds": seconds,
                                "hvp_calls": calls[0],
                                "reference_true_relative_residual": true_residual,
                                "solution_relative_difference": solution_difference,
                                **info,
                            }
                        )
                        print(
                            json.dumps(
                                {"state": state_name, "variant": name, **cg_samples[-1]}
                            ),
                            flush=True,
                        )
                    variant = {
                        "name": name,
                        "construction_first_call_seconds": construction_seconds,
                        "first_apply_prewarm_seconds": prewarm_seconds,
                        "numeric_setup_seconds": stats(setup_samples),
                        "hvp_seconds": stats(hvp_samples),
                        "cg": cg_samples,
                        "cg_seconds": stats([item["seconds"] for item in cg_samples]),
                        "accuracy": checks,
                        "symmetry_error": symmetry,
                        "extra_persistent_bytes": cache.persistent_bytes
                        if cache
                        else 0,
                        "metadata": cache.metadata if cache else {},
                        "memory_before": before,
                        "memory_after": memory(),
                        "peak_torch_allocated": torch.cuda.max_memory_allocated(),
                    }
                    row["variants"].append(variant)
                    benchmark.write_json(cfg.output_dir / f"{state_name}.json", row)
                    del apply, shifted, cache, values, actual, solution
                    gc.collect()
                assert digest_csr(state.collision.hess) == row["contact_csr_sha256"]
                results.append(row)
                handle.adapter.invalidate()
                del reference, control, ref_values, reference_solution, state
                gc.collect()
    finally:
        handle.uninstall()
    summary = {
        "schema": "frozen-hessian-representations-v1",
        "protocol": protocol,
        "model_initialization_seconds": context.initialization_seconds,
        "results": results,
    }
    benchmark.write_json(cfg.output_dir / "summary.json", summary)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": False, "success": True}
    )
    cherries.log_metric("states_completed", len(results))
    print(json.dumps({"success": True, "states": len(results)}), flush=True)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
