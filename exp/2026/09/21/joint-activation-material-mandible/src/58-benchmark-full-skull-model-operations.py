"""Matched fixed-state model-operation timing with complete-source contact.

This diagnostic uses a hash-bound collision- and volume-valid initialization.
It exercises the wrapper's extended Model/DofMap and real GPU/CPU IPC paths, but
never runs equilibrium and never claims final-run admission.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_full_skull_contact import (
    build_full_skull_contact,
    extend_dof_map,
    load_admitted_initialization,
    load_full_skull_geometry,
)
from joint_physics import JointPhysics
from joint_spatial_fields import SpatialSharedFieldParameters

from liblaf import cherries
from liblaf.apple.forward import Model

COMPLETED = False
ARM_OFF = "off"
ARM_FULL = "complete_source_contact"
ORDER = (ARM_OFF, ARM_FULL, ARM_FULL, ARM_OFF) * 2 + (ARM_OFF, ARM_FULL)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    checkpoint: Path = (
        GROUP
        / "data/neutral-convergence-025-contact-spatial80-metric-bfgs-002/terminal.pt"
    )
    geometry: Path = GROUP / "data/full-skull-initialization-audit-001/geometry.npz"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    candidate: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/candidate.npz"
    )
    candidate_summary: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/summary.json"
    )
    admission: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )
    output_dir: Path = cherries.output(
        "full-skull-model-operation-benchmark", mkdir=True
    )
    repeats: int = 5
    warmups: int = 1
    direction_scale: float = 0.01


def sync_time(operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = processes.splitlines() if processes else []
    return {
        "gpu": gpu,
        "processes": rows,
        "other_python_processes": [
            row
            for row in rows
            if not row.startswith(f"{os.getpid()},") and "python" in row.lower()
        ],
    }


def distribution(values: list[float]) -> dict[str, float]:
    assert len(values) == 5
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "population_stdev": statistics.pstdev(values),
    }


def make_state(model: Model, u: torch.Tensor) -> Model.State:
    state = model.State(u=u.detach().clone())
    if model.collision is not None:
        state.collision = model.collision.state_at(state.u)
    return state


def run_operations(
    arm: str, model: Model, u: torch.Tensor, direction: torch.Tensor
) -> dict[str, Any]:
    occupancy_before = gpu_snapshot()
    state, state_seconds = sync_time(lambda: make_state(model, u))
    energy, energy_seconds = sync_time(lambda: model.fun(state))
    gradient, gradient_seconds = sync_time(lambda: model.grad(state))
    hessian_state = make_state(model, u)
    first_hvp, first_hvp_seconds = sync_time(
        lambda: model.hess_prod(hessian_state, direction)
    )
    cached_hvp, cached_hvp_seconds = sync_time(
        lambda: model.hess_prod(hessian_state, direction)
    )
    ccd_state = make_state(model, u)
    ccd_fraction, ccd_seconds = sync_time(
        lambda: model.max_step_size(ccd_state, direction)
    )
    assert torch.isfinite(energy)
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(first_hvp).all()
    hvp_repeat_relative_error = float(
        torch.linalg.vector_norm(first_hvp - cached_hvp)
        / torch.linalg.vector_norm(first_hvp).clamp_min(1e-30)
    )
    assert hvp_repeat_relative_error <= 1e-12
    contact = (
        model.collision.diagnostics(state.collision, state.u)
        if model.collision is not None
        else None
    )
    if contact is not None:
        assert contact["contact_numerically_valid"] is True
    return {
        "arm": arm,
        "seconds": {
            "state_build": state_seconds,
            "full_model_energy": energy_seconds,
            "full_model_gradient": gradient_seconds,
            "full_model_first_hvp": first_hvp_seconds,
            "full_model_cached_hvp": cached_hvp_seconds,
            "full_model_ccd": ccd_seconds,
        },
        "energy": float(energy),
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "hvp_norm": float(torch.linalg.vector_norm(first_hvp)),
        "hvp_repeat_relative_error": hvp_repeat_relative_error,
        "ccd_fraction": float(ccd_fraction),
        "contact": contact,
        "occupancy_before": occupancy_before,
        "occupancy_after": gpu_snapshot(),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.repeats == 5
    assert cfg.warmups == 1
    assert cfg.direction_scale == 0.01
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    candidate_summary = json.loads(cfg.candidate_summary.read_text())
    assert candidate_summary["schema"] == "joint-full-skull-initialization-repair-v1"
    assert candidate_summary["success"] is True
    assert candidate_summary["equilibrium_converged"] is False
    assert candidate_summary["final_launch_ready"] is False
    assert candidate_summary["candidate"]["sha256"] == sha256(cfg.candidate)
    assert candidate_summary["admission"]["sha256"] == sha256(cfg.admission)
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["protocol"]["shared_basis"] == "spatial80"

    geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    assert candidate_summary["geometry_sha256"] == geometry.geometry_sha256
    assert candidate_summary["geometry_audit_sha256"] == geometry.audit_sha256
    admission = json.loads(cfg.admission.read_text())
    assert Path(admission["initialization_path"]).resolve() == cfg.candidate.resolve()
    fem_u_array = load_admitted_initialization(admission, geometry)

    configure_cuda()
    startup_occupancy = gpu_snapshot()
    material_config = research_informed_material_config()
    materials = material_config["materials"]
    skin = materials["skin"]
    base = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        contact_config=None,
    )
    basis = checkpoint["protocol"]["spatial_basis"]
    shared = SpatialSharedFieldParameters(
        Path(basis["basis_path"]),
        Path(basis["audit_summary_path"]),
        len(base.tets),
        material_config=material_config,
        device="cuda",
    )
    with torch.no_grad():
        shared.coefficients.copy_(checkpoint["shared_coefficients"])
    original = base.runtime.forward.model
    original.set_materials(
        base.materials(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
        )
    )
    candidate_metrics = base.metrics(torch.from_numpy(fem_u_array))
    assert candidate_metrics["inverted_tetrahedra"] == 0
    assert candidate_metrics["detF_min"] >= 0.25
    assert candidate_metrics["detF_max"] <= 2.0
    assert abs(candidate_metrics["detF_min"] - admission["detF_min"]) <= 1e-12
    assert abs(candidate_metrics["detF_max"] - admission["detF_max"]) <= 1e-12
    contact_config = {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 1e-4,
        "stiffness_mpa": 0.01,
    }
    adapter = build_full_skull_contact(geometry, contact_config)
    models = {
        ARM_OFF: Model(
            dof_map=extend_dof_map(original.dof_map, geometry),
            warp_model=original.warp_model,
            collision=None,
            device=original.device,
        ),
        ARM_FULL: Model(
            dof_map=extend_dof_map(original.dof_map, geometry),
            warp_model=original.warp_model,
            collision=adapter.collision,
            device=original.device,
        ),
    }
    fem_u = torch.as_tensor(fem_u_array.copy(), device="cuda")
    u = adapter.extend_seed(fem_u, torch.zeros(6))
    direction = cfg.direction_scale * u
    assert float(torch.linalg.vector_norm(direction)) > 0

    warmups = [run_operations(arm, models[arm], u, direction) for arm in models]
    write_json(output / "warmups.json", warmups)
    rows = []
    counts = {ARM_OFF: 0, ARM_FULL: 0}
    for sequence, arm in enumerate(ORDER):
        row = run_operations(arm, models[arm], u, direction)
        row["repeat"] = counts[arm]
        row["sequence"] = sequence
        rows.append(row)
        counts[arm] += 1
        write_json(output / "repeats.json", rows)

    operation_names = rows[0]["seconds"]
    warmed = {
        arm: {
            name: distribution(
                [row["seconds"][name] for row in rows if row["arm"] == arm]
            )
            for name in operation_names
        }
        for arm in models
    }
    ratios = {
        name: warmed[ARM_FULL][name]["median"] / warmed[ARM_OFF][name]["median"]
        for name in operation_names
    }
    success = all(
        row["contact"] is None or row["contact"]["contact_numerically_valid"] is True
        for row in rows
    )
    timing_contended = bool(startup_occupancy["other_python_processes"])
    summary = {
        "schema": "joint-full-skull-fixed-state-model-operation-benchmark-v1",
        "success": success,
        "status": (
            "passed_contended_valid_initialization_kernel_diagnostic"
            if timing_contended
            else "passed_uncontended_valid_initialization_kernel_diagnostic"
        ),
        "scope": {
            "complete_source_contact": True,
            "same_extended_model_and_dof_map": True,
            "fixed_state_only": True,
            "equilibrium_run": False,
            "candidate_volume_valid": True,
            "candidate_metrics": candidate_metrics,
            "soft_bone_initialization_admitted": True,
            "bone_bone_contact": False,
            "equilibrium_converged": False,
            "final_launch_ready": False,
            "production_admission": False,
            "claim_limit": "fixed-state kernel-operation overhead only; not equilibrium, full solve, bone-bone, or final-run evidence",
        },
        "protocol": {
            "warmups_per_arm": cfg.warmups,
            "repeats_per_arm": cfg.repeats,
            "order": list(ORDER),
            "cuda_synchronized": True,
            "direction": "0.01 times the complete candidate displacement",
            "startup_occupancy": startup_occupancy,
        },
        "warmed_seconds": warmed,
        "complete_source_over_off_median_ratios": ratios,
        "representative_contact": next(
            row["contact"] for row in rows if row["arm"] == ARM_FULL
        ),
        "contact_definition": adapter.contact_definition,
        "checkpoint_sha256": sha256(cfg.checkpoint),
        "candidate_sha256": sha256(cfg.candidate),
        "candidate_summary_sha256": sha256(cfg.candidate_summary),
        "admission_sha256": sha256(cfg.admission),
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_contact.py"),
                Path(__file__).with_name("joint_full_skull_contact.py"),
                Path(__file__).with_name("joint_physics.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "full_skull_model_operations/success": float(success),
            "full_skull_model_operations/gradient_ratio": ratios["full_model_gradient"],
            "full_skull_model_operations/first_hvp_ratio": ratios[
                "full_model_first_hvp"
            ],
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
