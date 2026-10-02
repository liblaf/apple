"""Matched collision on/off timing for the legacy partial-FEM IPC model.

This benchmark deliberately does not claim to measure full source-skull contact.
Both arms start from one frozen accepted displacement and re-equilibrate after a
predeclared one-percent increase in the skin baseline resultant.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_physics import JointPhysics
from joint_spatial_fields import SpatialSharedFieldParameters, spatial_field_config

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
ARM_OFF = "off"
ARM_LEGACY = "legacy_partial_fem_ipc"
ARM_ORDER = (ARM_OFF, ARM_LEGACY, ARM_LEGACY, ARM_OFF) * 2 + (
    ARM_OFF,
    ARM_LEGACY,
)


class BenchmarkContractError(RuntimeError):
    """The host does not expose an identity field required by the receipt."""


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    checkpoint: Path = (
        GROUP
        / "data/neutral-convergence-025-contact-spatial80-metric-bfgs-002/terminal.pt"
    )
    contact_spec: Path = GROUP / "data/contact/config.json"
    contact_validation: Path = GROUP / "data/contact-validation/summary.json"
    output_dir: Path = cherries.output("collision-benchmark-legacy", mkdir=True)
    skin_resultant_multiplier: float = 1.01
    repeats_per_arm: int = 5
    warmups_per_arm: int = 1
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12


def tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().to(torch.float64).numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def synchronized_time(operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "maximum_allocated_bytes": torch.cuda.max_memory_allocated(),
        "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def process_thread_count() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1])
    raise BenchmarkContractError


def gpu_occupancy_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    processes = process_query.splitlines() if process_query else []
    other_python = [
        row
        for row in processes
        if not row.startswith(f"{os.getpid()},") and "python" in row.lower()
    ]
    return {
        "perf_counter_seconds": time.perf_counter(),
        "gpu_query": gpu_query,
        "compute_processes": processes,
        "other_python_compute_processes": other_python,
    }


def identity() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unavailable",
    )
    nvidia_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contention = gpu_occupancy_snapshot()
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "process_affinity_cpu_count": len(os.sched_getaffinity(0)),
        "process_thread_count": process_thread_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "warp": wp.__version__,
            "ipctk": ipctk.__version__,
            "numpy": np.__version__,
            "pyvista": pv.__version__,
        },
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "nvidia_smi": nvidia_query,
        },
        "contention_snapshot": {
            "query": "nvidia-smi compute applications at benchmark startup",
            **contention,
            "timing_environment": (
                "contended_by_other_python_compute_process"
                if contention["other_python_compute_processes"]
                else "no_other_python_compute_process_detected"
            ),
        },
    }


def make_physics(
    cfg: Config,
    prepared: PreparedInputs,
    material_config: dict[str, Any],
    contact_config: dict[str, Any] | None,
) -> JointPhysics:
    materials = material_config["materials"]
    skin = materials["skin"]
    return JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
        contact_config=contact_config,
        forward_method="newton_cg",
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
    )


def make_shared(
    checkpoint: dict[str, Any], physics: JointPhysics
) -> SpatialSharedFieldParameters:
    basis = checkpoint["protocol"]["spatial_basis"]
    shared = SpatialSharedFieldParameters(
        Path(basis["basis_path"]),
        Path(basis["audit_summary_path"]),
        len(physics.tets),
        material_config=research_informed_material_config(),
        device="cuda",
    )
    assert shared.basis_receipt() == basis
    with torch.no_grad():
        shared.coefficients.copy_(checkpoint["shared_coefficients"])
        shared.skin_baseline_coordinate.mul_(1.01)
    return shared


def objective(
    physics: JointPhysics,
    shared: SpatialSharedFieldParameters,
    displacement: torch.Tensor,
    *,
    prior_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    surface = (
        (
            physics.weights_t[:, None] * displacement[physics.observation_t].square()
        ).sum()
        * 1e6
        / 0.25**2
    )
    centers = displacement[physics.muscle_tets_t].mean(dim=1)
    muscle = (physics.muscle_mass_t[:, None] * centers.square()).sum() * 1e6 / 0.5**2
    regularizers = shared.regularizers()
    weighted_prior = prior_weight * regularizers["prior_total"]
    weighted_roughness = (
        0.5 * smoothness_weight * regularizers["bulk_spatial_roughness"]
    )
    total = surface + muscle + weighted_prior + weighted_roughness
    return total, {
        "surface_loss": float(surface.detach()),
        "muscle_loss": float(muscle.detach()),
        "weighted_prior": float(weighted_prior.detach()),
        "weighted_spatial_roughness": float(weighted_roughness.detach()),
        "total": float(total.detach()),
    }


def forward_counts(receipt: dict[str, Any]) -> dict[str, int]:
    trace = receipt["trace"]
    return {
        "newton_steps": int(receipt["steps"]),
        "linear_matvec_count": sum(int(row["linear_matvec_count"]) for row in trace),
        "line_search_trial_count": sum(len(row["trials"]) for row in trace),
    }


def time_contact_operations(
    physics: JointPhysics, displacement: torch.Tensor, seed: torch.Tensor
) -> dict[str, Any]:
    collision = physics.runtime.forward.model.collision
    assert collision is not None
    u = displacement.detach()
    state, state_seconds = synchronized_time(lambda: collision.state_at(u))
    energy, energy_seconds = synchronized_time(lambda: collision.fun(state, u))
    gradient = torch.zeros_like(u)
    _, gradient_seconds = synchronized_time(lambda: collision.grad(state, u, gradient))
    hessian_state = collision.state_at(u)
    direction = u - seed
    assert float(torch.linalg.vector_norm(direction)) > 0.0, (
        "the one-percent perturbation produced no displacement"
    )
    hessian_direction = torch.zeros_like(u)
    _, first_hvp_seconds = synchronized_time(
        lambda: collision.hess_prod(hessian_state, u, direction, hessian_direction)
    )
    cached_hessian_direction = torch.zeros_like(u)
    _, cached_hvp_seconds = synchronized_time(
        lambda: collision.hess_prod(
            hessian_state, u, direction, cached_hessian_direction
        )
    )
    ccd_state = collision.state_at(u)
    ccd_fraction, ccd_seconds = synchronized_time(
        lambda: collision.max_step_size(ccd_state, u, direction)
    )
    diagnostics = collision.diagnostics(state, u)
    assert diagnostics["contact_numerically_valid"]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian_direction).all()
    assert torch.isfinite(cached_hessian_direction).all()
    assert torch.allclose(hessian_direction, cached_hessian_direction, rtol=0, atol=0)
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
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "ccd_fraction": float(ccd_fraction),
        "diagnostics": diagnostics,
    }


def run_once(
    *,
    arm: str,
    repeat: int,
    measured: bool,
    physics: JointPhysics,
    shared: SpatialSharedFieldParameters,
    checkpoint: dict[str, Any],
    seed: torch.Tensor,
    prior_weight: float,
    smoothness_weight: float,
) -> dict[str, Any]:
    assert arm in {ARM_OFF, ARM_LEGACY}
    with torch.no_grad():
        shared.coefficients.copy_(checkpoint["shared_coefficients"])
        shared.skin_baseline_coordinate.mul_(1.01)
    shared.coefficients.grad = None
    physics.runtime.warm_adjoints.clear()
    occupancy_before = gpu_occupancy_snapshot()
    torch.cuda.reset_peak_memory_stats()
    memory_before = cuda_memory()
    pose = torch.zeros(6, device="cuda", dtype=torch.float64)
    key = f"{arm}/{'repeat' if measured else 'warmup'}/{repeat}"
    displacement, forward_wall = synchronized_time(
        lambda: physics.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
            pose,
            seed.detach().clone(),
            key=key,
        )
    )
    forward = copy.deepcopy(physics.runtime.last_forward)
    assert forward["success"] is True
    assert forward["forward_solver"] == {
        "method": "newton_cg",
        "newton_linear_rtol": 1e-3,
        "newton_max_steps": 12,
        "fallback": None,
    }
    value, terms = objective(
        physics,
        shared,
        displacement,
        prior_weight=prior_weight,
        smoothness_weight=smoothness_weight,
    )
    gradient, adjoint_wall = synchronized_time(
        lambda: torch.autograd.grad(value, shared.coefficients)[0]
    )
    adjoint = copy.deepcopy(physics.runtime.last_adjoint)
    assert adjoint["success"] is True
    assert adjoint["relative_residual"] <= 1.05e-7
    assert torch.isfinite(displacement).all()
    assert torch.isfinite(gradient).all()
    metrics = physics.metrics(displacement)
    assert metrics["inverted_tetrahedra"] == 0
    assert metrics["detF_min"] >= 0.25
    assert metrics["detF_max"] <= 2.0
    assert metrics["skin_area_ratio_min"] >= 0.25
    if arm == ARM_LEGACY:
        assert forward["contact"]["contact_numerically_valid"] is True
        contact_operations = time_contact_operations(physics, displacement, seed)
    else:
        assert "contact" not in forward
        contact_operations = None
    memory_after = cuda_memory()
    occupancy_after = gpu_occupancy_snapshot()
    return {
        "arm": arm,
        "repeat": repeat,
        "measured": measured,
        "seed_sha256": tensor_sha256(seed),
        "coefficient_sha256": tensor_sha256(shared.coefficients),
        "displacement_sha256": tensor_sha256(displacement),
        "gradient_sha256": tensor_sha256(gradient),
        "skin_resultant_n_per_m": float(shared.skin_resultant_n_per_m()[0, 0]),
        "objective": terms,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "forward_wall_seconds": forward_wall,
        "forward_receipt_seconds": float(forward["seconds"]),
        "forward_counts": forward_counts(forward),
        "initial_force_norm": float(forward["initial_gradient_norm"]),
        "terminal_force_norm": float(forward["grad_norm"]),
        "adjoint_autograd_wall_seconds": adjoint_wall,
        "adjoint_receipt_seconds": float(adjoint["seconds"]),
        "adjoint_wrapper_and_direct_seconds": adjoint_wall - float(adjoint["seconds"]),
        "adjoint_iteration_count": None,
        "adjoint_iteration_count_status": (
            "unavailable from the owned adjoint solver receipt"
        ),
        "adjoint_relative_residual": float(adjoint["relative_residual"]),
        "metrics": metrics,
        "contact": forward.get("contact"),
        "contact_operations": contact_operations,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "process_thread_count": process_thread_count(),
        "gpu_occupancy_before": occupancy_before,
        "gpu_occupancy_after": occupancy_after,
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


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(rows) == 5
    result = {
        "repeat_count": len(rows),
        "forward_wall_seconds": distribution(
            [row["forward_wall_seconds"] for row in rows]
        ),
        "forward_receipt_seconds": distribution(
            [row["forward_receipt_seconds"] for row in rows]
        ),
        "adjoint_autograd_wall_seconds": distribution(
            [row["adjoint_autograd_wall_seconds"] for row in rows]
        ),
        "adjoint_receipt_seconds": distribution(
            [row["adjoint_receipt_seconds"] for row in rows]
        ),
        "initial_force_norm": distribution([row["initial_force_norm"] for row in rows]),
        "newton_steps": [row["forward_counts"]["newton_steps"] for row in rows],
        "linear_matvec_count": [
            row["forward_counts"]["linear_matvec_count"] for row in rows
        ],
        "line_search_trial_count": [
            row["forward_counts"]["line_search_trial_count"] for row in rows
        ],
        "maximum_adjoint_relative_residual": max(
            row["adjoint_relative_residual"] for row in rows
        ),
    }
    if rows[0]["contact_operations"] is not None:
        names = rows[0]["contact_operations"]["seconds"]
        result["contact_operations_seconds"] = {
            name: distribution(
                [row["contact_operations"]["seconds"][name] for row in rows]
            )
            for name in names
        }
    return result


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.skin_resultant_multiplier == 1.01
    assert cfg.repeats_per_arm == 5
    assert cfg.warmups_per_arm == 1
    assert len(ARM_ORDER) == 2 * cfg.repeats_per_arm
    assert ARM_ORDER.count(ARM_OFF) == cfg.repeats_per_arm
    assert ARM_ORDER.count(ARM_LEGACY) == cfg.repeats_per_arm
    assert cfg.forward_rtol == 1e-6
    assert cfg.forward_atol == 1e-12
    assert cfg.adjoint_rtol == 1e-7
    assert cfg.max_forward_steps == 10000
    assert cfg.newton_linear_rtol == 1e-3
    assert cfg.newton_max_steps == 12

    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == "joint-inverse-checkpoint-v1"
    assert checkpoint["stage"] == "neutral"
    assert checkpoint["protocol"]["shared_basis"] == "spatial80"
    assert checkpoint["materials"] == spatial_field_config()
    assert checkpoint["shared_coefficients"].shape == (80,)
    assert checkpoint["protocol"]["input_arrays_sha256"] == sha256(
        cfg.prepared_dir / "inputs.npz"
    )
    assert checkpoint["protocol"]["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )
    assert checkpoint["protocol"]["contact"]["spec_sha256"] == sha256(cfg.contact_spec)
    contact_validation = json.loads(cfg.contact_validation.read_text())
    assert contact_validation["schema"] == "joint-contact-validation-v1"
    assert contact_validation["success"] is True
    assert contact_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)
    contact_config = json.loads(cfg.contact_spec.read_text())
    assert contact_config["schema"] == "joint-bone-contact-v1"
    assert contact_config["enabled"] is True
    assert contact_config["surface_selection"] == "pure-soft-vs-pure-bone"
    assert checkpoint["neutral_converged"] is False
    assert checkpoint["preparation_complete"] is False
    prior_weight = float(checkpoint["protocol"]["config"]["prior_weight"])
    smoothness_weight = float(
        checkpoint["protocol"]["shared_field"]["spatial_smoothness"]["weight"]
    )
    assert smoothness_weight == 100.0

    configure_cuda()
    hardware = identity()
    timing_contended = bool(
        hardware["contention_snapshot"]["other_python_compute_processes"]
    )
    material_config = research_informed_material_config()
    arms: dict[str, dict[str, Any]] = {}
    for arm, arm_contact in (
        (ARM_OFF, None),
        (ARM_LEGACY, contact_config),
    ):
        memory_before = cuda_memory()
        physics, physics_seconds = synchronized_time(
            lambda arm_contact=arm_contact: make_physics(
                cfg, prepared, material_config, arm_contact
            )
        )
        shared, shared_seconds = synchronized_time(
            lambda physics=physics: make_shared(checkpoint, physics)
        )
        assert isinstance(physics, JointPhysics)
        assert isinstance(shared, SpatialSharedFieldParameters)
        arms[arm] = {
            "physics": physics,
            "shared": shared,
            "cold_setup": {
                "physics_seconds": physics_seconds,
                "spatial_field_seconds": shared_seconds,
                "total_seconds": physics_seconds + shared_seconds,
                "memory_before": memory_before,
                "memory_after": cuda_memory(),
                "contact_surface_map": physics.contact_definition,
            },
        }

    seed = checkpoint["primal"]["neutral"].to(device="cuda", dtype=torch.float64)
    seed_sha = tensor_sha256(seed)
    coefficient = checkpoint["shared_coefficients"].clone()
    coefficient[78] *= cfg.skin_resultant_multiplier
    coefficient_sha = tensor_sha256(coefficient)
    original_skin_resultant = float(
        checkpoint["metrics"]["material"]["skin_resultant_n_per_m"]
    )
    perturbed_skin_resultant = cfg.skin_resultant_multiplier * original_skin_resultant
    assert original_skin_resultant == 20.15
    assert math.isclose(perturbed_skin_resultant, 20.3515, rel_tol=0, abs_tol=1e-12)

    warmups = []
    for index, arm in enumerate((ARM_OFF, ARM_LEGACY)):
        warmups.append(
            run_once(
                arm=arm,
                repeat=index,
                measured=False,
                physics=arms[arm]["physics"],
                shared=arms[arm]["shared"],
                checkpoint=checkpoint,
                seed=seed,
                prior_weight=prior_weight,
                smoothness_weight=smoothness_weight,
            )
        )
        write_json(output / "warmups.json", warmups)

    rows: list[dict[str, Any]] = []
    counts = {ARM_OFF: 0, ARM_LEGACY: 0}
    for sequence_index, arm in enumerate(ARM_ORDER):
        repeat = counts[arm]
        row = run_once(
            arm=arm,
            repeat=repeat,
            measured=True,
            physics=arms[arm]["physics"],
            shared=arms[arm]["shared"],
            checkpoint=checkpoint,
            seed=seed,
            prior_weight=prior_weight,
            smoothness_weight=smoothness_weight,
        )
        row["sequence_index"] = sequence_index
        rows.append(row)
        counts[arm] += 1
        write_json(output / "repeats.json", rows)
        LOG.info(
            "%s repeat %d: forward %.4fs adjoint %.4fs",
            arm,
            repeat,
            row["forward_wall_seconds"],
            row["adjoint_autograd_wall_seconds"],
        )

    by_arm = {
        arm: summarize_arm([row for row in rows if row["arm"] == arm])
        for arm in (ARM_OFF, ARM_LEGACY)
    }
    workload_identity = {
        "all_seed_hashes_match": all(row["seed_sha256"] == seed_sha for row in rows),
        "all_coefficient_hashes_match": all(
            row["coefficient_sha256"] == coefficient_sha for row in rows
        ),
        "unique_displacement_hashes_by_arm": {
            arm: sorted(
                {row["displacement_sha256"] for row in rows if row["arm"] == arm}
            )
            for arm in (ARM_OFF, ARM_LEGACY)
        },
        "unique_gradient_hashes_by_arm": {
            arm: sorted({row["gradient_sha256"] for row in rows if row["arm"] == arm})
            for arm in (ARM_OFF, ARM_LEGACY)
        },
    }
    workload_identity["deterministic_per_arm"] = all(
        len(workload_identity[name][arm]) == 1
        for name in (
            "unique_displacement_hashes_by_arm",
            "unique_gradient_hashes_by_arm",
        )
        for arm in (ARM_OFF, ARM_LEGACY)
    )
    ratios = {
        key: by_arm[ARM_LEGACY][key]["median"] / by_arm[ARM_OFF][key]["median"]
        for key in (
            "forward_wall_seconds",
            "forward_receipt_seconds",
            "adjoint_autograd_wall_seconds",
            "adjoint_receipt_seconds",
        )
    }
    success = all(
        row["metrics"]["inverted_tetrahedra"] == 0
        and row["adjoint_relative_residual"] <= 1.05e-7
        and (
            row["contact"] is None
            or row["contact"]["contact_numerically_valid"] is True
        )
        for row in rows
    ) and all(
        workload_identity[name]
        for name in (
            "all_seed_hashes_match",
            "all_coefficient_hashes_match",
        )
    )
    summary = {
        "schema": "joint-collision-performance-benchmark-v1",
        "success": success,
        "status": (
            (
                "passed_contended_legacy_partial_fem_on_off_benchmark"
                if timing_contended
                else "passed_uncontended_legacy_partial_fem_on_off_benchmark"
            )
            if success
            else "failed_legacy_partial_fem_on_off_benchmark"
        ),
        "scope": {
            "measured_contact": "legacy_partial_fem_ipc",
            "legacy_surface_selection": "pure FEM soft boundary versus pure FEM bone boundary",
            "full_source_skull_contact": "pending; not implemented or measured by this artifact",
            "claim_limit": (
                "legacy partial-FEM timing is not evidence of full source-skull "
                "contact overhead"
            ),
            "timing_environment": (
                "contended; preserve ratios and variability, then repeat once idle"
                if timing_contended
                else "no other Python GPU compute process detected at startup"
            ),
        },
        "workload": {
            "checkpoint": str(cfg.checkpoint.resolve()),
            "checkpoint_sha256": sha256(cfg.checkpoint),
            "checkpoint_update": checkpoint["update"],
            "checkpoint_converged": checkpoint["neutral_converged"],
            "checkpoint_preparation_complete": checkpoint["preparation_complete"],
            "seed_sha256": seed_sha,
            "coefficient_sha256_after_perturbation": coefficient_sha,
            "perturbation": "multiply only Spatial80 coordinate 78 by 1.01",
            "skin_resultant_multiplier": cfg.skin_resultant_multiplier,
            "original_skin_resultant_n_per_m": original_skin_resultant,
            "perturbed_skin_resultant_n_per_m": perturbed_skin_resultant,
            "all_other_coefficients_fixed": True,
            "pose_rad_m": [0.0] * 6,
            "activation": None,
            "objective": (
                "neutral surface + muscle + prior_weight*prior_total + "
                "0.5*100*bulk_spatial_roughness"
            ),
            "prior_weight": prior_weight,
            "spatial_smoothness_weight": smoothness_weight,
            "spatial_smoothness_factor": 0.5,
        },
        "protocol": {
            "warmups_per_arm": cfg.warmups_per_arm,
            "repeats_per_arm": cfg.repeats_per_arm,
            "measured_order": list(ARM_ORDER),
            "seed_reset_each_repeat": True,
            "adjoint_warm_start_cleared_each_repeat": True,
            "cuda_synchronized_wall_timing": True,
            "cold_setup_excluded_from_warmed_statistics": True,
            "forward": {
                "method": "newton_cg",
                "rtol": cfg.forward_rtol,
                "atol": cfg.forward_atol,
                "max_steps": cfg.max_forward_steps,
                "linear_rtol": cfg.newton_linear_rtol,
                "newton_max_steps": cfg.newton_max_steps,
            },
            "adjoint_rtol": cfg.adjoint_rtol,
        },
        "cold_setup": {arm: value["cold_setup"] for arm, value in arms.items()},
        "warmed": by_arm,
        "legacy_contact_over_off_median_ratios": ratios,
        "workload_identity": workload_identity,
        "hardware_software": hardware,
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "contact_spec_sha256": sha256(cfg.contact_spec),
        "contact_validation_sha256": sha256(cfg.contact_validation),
        "spatial_basis": checkpoint["protocol"]["spatial_basis"],
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_contact.py"),
                Path(__file__).with_name("joint_equilibrium.py"),
                Path(__file__).with_name("joint_newton.py"),
                Path(__file__).with_name("joint_physics.py"),
                Path(__file__).with_name("joint_spatial_fields.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "collision_benchmark/success": float(success),
            "collision_benchmark/legacy_forward_ratio": ratios["forward_wall_seconds"],
            "collision_benchmark/legacy_adjoint_ratio": ratios[
                "adjoint_autograd_wall_seconds"
            ],
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
