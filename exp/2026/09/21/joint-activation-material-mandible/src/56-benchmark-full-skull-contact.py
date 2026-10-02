"""CPU-only IPC microbenchmark on the complete-source, invalid-volume candidate.

The candidate is collision-free against the complete source bones but contains
inverted tetrahedra.  This script times contact operations only and cannot serve
as a mechanics, equilibrium, or production-admission receipt.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_full_skull_contact import (
    build_full_skull_contact,
    load_full_skull_geometry,
)

from liblaf import cherries

COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    geometry: Path = GROUP / "data/full-skull-initialization-audit-001/geometry.npz"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    candidate: Path = (
        GROUP / "data/full-skull-initialization-candidate-001/candidate.npz"
    )
    candidate_summary: Path = (
        GROUP / "data/full-skull-initialization-candidate-001/summary.json"
    )
    adapter_validation: Path = (
        GROUP / "data/full-skull-contact-adapter-validation-002/summary.json"
    )
    output_dir: Path = cherries.output(
        "full-skull-contact-microbenchmark-invalid-candidate", mkdir=True
    )
    repeats: int = 5
    warmups: int = 1
    dhat_m: float = 1e-4
    stiffness_mpa: float = 0.01
    direction_scale: float = 0.01


def timed(operation: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def distribution(values: list[float]) -> dict[str, float]:
    assert len(values) == 5
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "population_stdev": statistics.pstdev(values),
    }


def run_operations(
    collision: Any, displacement: torch.Tensor, direction: torch.Tensor
) -> dict[str, Any]:
    state, state_seconds = timed(lambda: collision.state_at(displacement))
    energy, energy_seconds = timed(lambda: collision.fun(state, displacement))
    gradient = torch.zeros_like(displacement)
    _, gradient_seconds = timed(lambda: collision.grad(state, displacement, gradient))
    hessian_state = collision.state_at(displacement)
    hessian_direction = torch.zeros_like(displacement)
    _, first_hvp_seconds = timed(
        lambda: collision.hess_prod(
            hessian_state, displacement, direction, hessian_direction
        )
    )
    cached_hessian_direction = torch.zeros_like(displacement)
    _, cached_hvp_seconds = timed(
        lambda: collision.hess_prod(
            hessian_state,
            displacement,
            direction,
            cached_hessian_direction,
        )
    )
    ccd_state = collision.state_at(displacement)
    ccd_fraction, ccd_seconds = timed(
        lambda: collision.max_step_size(ccd_state, displacement, direction)
    )
    diagnostics = collision.diagnostics(state, displacement)
    assert diagnostics["contact_numerically_valid"] is True
    assert diagnostics["active_contact_count"] > 0
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian_direction).all()
    assert torch.equal(hessian_direction, cached_hessian_direction)
    return {
        "seconds": {
            "state_at": state_seconds,
            "energy": energy_seconds,
            "gradient": gradient_seconds,
            "first_hvp_assembly_and_matvec": first_hvp_seconds,
            "cached_hvp_matvec": cached_hvp_seconds,
            "ccd_max_step_size": ccd_seconds,
        },
        "energy": float(energy),
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "hvp_norm": float(torch.linalg.vector_norm(hessian_direction)),
        "ccd_fraction": float(ccd_fraction),
        "diagnostics": diagnostics,
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.repeats == 5
    assert cfg.warmups == 1
    assert cfg.dhat_m == 1e-4
    assert cfg.stiffness_mpa == 0.01
    assert cfg.direction_scale == 0.01
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.get_default_device().type != "cuda"

    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    candidate_summary = json.loads(cfg.candidate_summary.read_text())
    assert candidate_summary["schema"] == "joint-full-skull-initialization-candidate-v1"
    assert candidate_summary["candidate"]["sha256"] == sha256(cfg.candidate)
    assert candidate_summary["geometric_candidate_passed"] is False
    assert candidate_summary["full_skull_contact_admitted"] is False
    assert candidate_summary["equilibrium_converged"] is False
    final_candidate_row = candidate_summary["trace"][-1]
    assert final_candidate_row["inverted_tetrahedra"] == 13
    assert final_candidate_row["detF_min"] < 0
    assert all(
        bone["intersection_pairs"] == 0
        for bone in final_candidate_row["contacts"].values()
    )
    adapter_validation = json.loads(cfg.adapter_validation.read_text())
    assert adapter_validation["schema"] == (
        "joint-full-skull-contact-adapter-validation-v1"
    )
    assert adapter_validation["success"] is True
    assert adapter_validation["audited_geometry_contact_admitted"] is False

    geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    assert geometry.geometry_sha256 == candidate_summary["geometry_sha256"]
    assert geometry.audit_sha256 == candidate_summary["geometry_audit_sha256"]
    config = {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": cfg.dhat_m,
        "stiffness_mpa": cfg.stiffness_mpa,
    }
    adapter, cold_build_seconds = timed(
        lambda: build_full_skull_contact(geometry, config)
    )
    with np.load(cfg.candidate) as arrays:
        assert set(arrays.files) == {"initial_displacement_m"}
        fem_displacement_np = np.asarray(arrays["initial_displacement_m"])
    assert fem_displacement_np.shape == (geometry.fem_node_count, 3)
    assert fem_displacement_np.dtype == np.float64
    fem_displacement = torch.from_numpy(fem_displacement_np.copy())
    zero_pose = torch.zeros(6)
    displacement = adapter.extend_seed(fem_displacement, zero_pose)
    direction = cfg.direction_scale * displacement
    assert float(torch.linalg.vector_norm(direction)) > 0

    warmups = [
        run_operations(adapter.collision, displacement, direction)
        for _ in range(cfg.warmups)
    ]
    write_json(output / "warmups.json", warmups)
    rows = []
    for repeat in range(cfg.repeats):
        row = run_operations(adapter.collision, displacement, direction)
        row["repeat"] = repeat
        rows.append(row)
        write_json(output / "repeats.json", rows)

    operation_names = rows[0]["seconds"]
    timings = {
        name: distribution([row["seconds"][name] for row in rows])
        for name in operation_names
    }
    success = all(
        row["diagnostics"]["contact_numerically_valid"]
        and row["diagnostics"]["active_contact_count"] > 0
        for row in rows
    )
    summary = {
        "schema": "joint-full-skull-contact-microbenchmark-v1",
        "success": success,
        "status": (
            "passed_contact_only_invalid_volume_diagnostic"
            if success
            else "failed_contact_only_invalid_volume_diagnostic"
        ),
        "scope": {
            "complete_source_bones": True,
            "all_source_triangles_retained": True,
            "source_coordinates_changed": False,
            "excluded_source_triangles": 0,
            "contact_operations_only": True,
            "fem_equilibrium_run": False,
            "forward_or_adjoint_timing": False,
            "production_admission": False,
            "candidate_inverted_tetrahedra": 13,
            "claim_limit": (
                "contact-only CPU timing on a collision-free but mechanically "
                "invalid FEM displacement; it is not full-model overhead"
            ),
        },
        "protocol": {
            "warmups": cfg.warmups,
            "repeats": cfg.repeats,
            "cpu_only": True,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "direction": "0.01 times the complete candidate displacement",
            "direction_norm": float(torch.linalg.vector_norm(direction)),
            "contact_config": config,
        },
        "cold_contact_build_seconds": cold_build_seconds,
        "warmed_contact_operations_seconds": timings,
        "contact_definition": adapter.contact_definition,
        "representative_diagnostics": rows[-1]["diagnostics"],
        "geometry": geometry.binding_receipt(),
        "candidate": {
            "path": str(cfg.candidate.resolve()),
            "sha256": sha256(cfg.candidate),
            "summary_path": str(cfg.candidate_summary.resolve()),
            "summary_sha256": sha256(cfg.candidate_summary),
            "final_geometry_metrics": final_candidate_row,
        },
        "adapter_validation": {
            "path": str(cfg.adapter_validation.resolve()),
            "sha256": sha256(cfg.adapter_validation),
            "status": adapter_validation["status"],
        },
        "hardware_software": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "process_affinity_cpu_count": len(os.sched_getaffinity(0)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "ipctk": ipctk.__version__,
            "maximum_resident_set_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_contact.py"),
                Path(__file__).with_name("joint_full_skull_contact.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "full_skull_contact_microbenchmark/success": float(success),
            "full_skull_contact_microbenchmark/state_at_median_seconds": timings[
                "state_at"
            ]["median"],
            "full_skull_contact_microbenchmark/first_hvp_median_seconds": timings[
                "first_hvp_assembly_and_matvec"
            ]["median"],
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
