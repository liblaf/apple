"""Matched full-forward and adjoint timing for complete-source soft-bone IPC.

Both arms start from the same admitted, volume-valid FEM displacement and use the
same frozen Spatial80 coefficients at zero jaw pose. Complete-source contact keeps
all registered source-bone triangles, but excludes source bone-bone pairs by the
declared soft-versus-bone collision policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import (
    Equilibrium,
    ForwardConvergenceError,
    configure_cuda,
    rigid_displacement,
)
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_full_skull_contact import (
    FullSkullGeometry,
    FullSkullJointPhysics,
    extend_dof_map,
    load_admitted_initialization,
    load_full_skull_geometry,
)
from joint_physics import JointPhysics
from joint_spatial_fields import SpatialSharedFieldParameters, spatial_field_config

from liblaf import cherries
from liblaf.apple.forward import Forward, Model

LOG = logging.getLogger(__name__)
COMPLETED = False
ARM_OFF = "off"
ARM_FULL_SOURCE = "complete_source_soft_bone_ipc"
ARM_ORDER = (ARM_OFF, ARM_FULL_SOURCE, ARM_FULL_SOURCE, ARM_OFF) * 2 + (
    ARM_OFF,
    ARM_FULL_SOURCE,
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
    geometry: Path = GROUP / "data/full-skull-initialization-audit-001/geometry.npz"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    initialization: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/candidate.npz"
    )
    admission: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )
    adapter_validation: Path = (
        GROUP / "data/full-skull-contact-adapter-validation-003/summary.json"
    )
    output_dir: Path = cherries.output("full-skull-forward-benchmark", mkdir=True)
    repeats_per_arm: int = 5
    warmups_per_arm: int = 1
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12
    run_mode: Literal["matched", "complete_source_first_attempt"] = "matched"


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


class FullSkullNoContactPhysics(JointPhysics):
    """No-contact control with the same appended fixed source-bone DOFs."""

    def __init__(
        self,
        *args: Any,
        full_skull_geometry: FullSkullGeometry,
        **kwargs: Any,
    ) -> None:
        assert kwargs.get("contact_config") is None
        super().__init__(*args, **kwargs)
        if not np.array_equal(self.points, full_skull_geometry.fem_reference_points_m):
            msg = "control FEM points differ from the full-skull geometry"
            raise ValueError(msg)
        if not np.array_equal(
            np.sort(
                np.asarray(self.mesh.point_data["FixedMask"]).any(axis=1).nonzero()[0]
            ),
            np.sort(full_skull_geometry.fixed_global_ids),
        ):
            msg = "control fixed-node set differs from the full-skull geometry"
            raise ValueError(msg)
        original = self.runtime.forward.model
        model = Model(
            dof_map=extend_dof_map(original.dof_map, full_skull_geometry),
            warp_model=original.warp_model,
            collision=None,
            device=original.device,
        )
        tolerances = self.runtime.tolerances
        solver = self.runtime.forward_solver
        self.runtime = Equilibrium(
            Forward(model),
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            adjoint_rtol=tolerances["adjoint_rtol"],
            max_steps=tolerances["max_steps"],
            forward_method=solver["method"],
            newton_linear_rtol=solver["newton_linear_rtol"],
            newton_max_steps=solver["newton_max_steps"],
        )
        self.full_skull_geometry = full_skull_geometry

    def _extend_seed(
        self, fem_displacement: torch.Tensor, pose: torch.Tensor
    ) -> torch.Tensor:
        geometry = self.full_skull_geometry
        if fem_displacement.shape != (geometry.fem_node_count, 3):
            msg = "control seed must contain exactly the original FEM nodes"
            raise ValueError(msg)
        cranium = fem_displacement.new_zeros((geometry.cranium_node_count, 3))
        source_mandible = torch.as_tensor(
            geometry.mandible_points_m,
            device=fem_displacement.device,
            dtype=fem_displacement.dtype,
        )
        pivot = torch.as_tensor(
            geometry.mandible_pivot_m,
            device=fem_displacement.device,
            dtype=fem_displacement.dtype,
        )
        mandible = rigid_displacement(source_mandible, pivot, pose)
        return torch.cat((fem_displacement, cranium, mandible))

    def boundary(self, pose: torch.Tensor) -> torch.Tensor:
        geometry = self.full_skull_geometry
        result = self.points_t.new_zeros((geometry.full_node_count, 3))
        pivot = torch.as_tensor(
            geometry.mandible_pivot_m,
            device=self.points_t.device,
            dtype=self.points_t.dtype,
        )
        result[self.jaw_t] = rigid_displacement(self.points_t[self.jaw_t], pivot, pose)
        source_points = torch.as_tensor(
            geometry.mandible_points_m,
            device=self.points_t.device,
            dtype=self.points_t.dtype,
        )
        source_ids = torch.as_tensor(
            geometry.mandible_global_ids,
            device=self.points_t.device,
            dtype=torch.long,
        )
        result[source_ids] = rigid_displacement(source_points, pivot, pose)
        return result.flatten()[self.runtime.forward.model.dof_map.fixed_indices]

    def solve(
        self,
        bulk_stress: torch.Tensor,
        skin_resultant_n_m: torch.Tensor,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor | None,
        pose: torch.Tensor,
        seed: torch.Tensor,
        *,
        seed_pose: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        full = self.runtime.solve(
            self.materials(
                bulk_stress, skin_resultant_n_m, skin_multiplier, active_stress
            ),
            self.boundary(pose),
            self._extend_seed(seed, seed_pose),
            key=key,
        )
        return full[: self.full_skull_geometry.fem_node_count]


def make_physics(
    cfg: Config,
    prepared: PreparedInputs,
    material_config: dict[str, Any],
    *,
    full_source: bool,
    geometry: Any,
    admission: dict[str, Any],
    contact_config: dict[str, Any],
) -> JointPhysics:
    materials = material_config["materials"]
    skin = materials["skin"]
    cls = FullSkullJointPhysics if full_source else FullSkullNoContactPhysics
    extra = (
        {
            "full_skull_geometry": geometry,
            "full_skull_admission": admission,
            "full_skull_contact_config": contact_config,
        }
        if full_source
        else {"full_skull_geometry": geometry}
    )
    return cls(
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
        contact_config=None,
        forward_method="newton_cg",
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
        **extra,
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
    forward_phase_receipt: Path | None = None,
) -> dict[str, Any]:
    assert arm in {ARM_OFF, ARM_FULL_SOURCE}
    with torch.no_grad():
        shared.coefficients.copy_(checkpoint["shared_coefficients"])
    shared.coefficients.grad = None
    physics.runtime.warm_adjoints.clear()
    occupancy_before = gpu_occupancy_snapshot()
    torch.cuda.reset_peak_memory_stats()
    memory_before = cuda_memory()
    pose = torch.zeros(6, device="cuda", dtype=torch.float64)
    key = f"{arm}/{'repeat' if measured else 'warmup'}/{repeat}"

    def solve() -> torch.Tensor:
        arguments = (
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
            pose,
            seed.detach().clone(),
        )
        if isinstance(physics, (FullSkullJointPhysics, FullSkullNoContactPhysics)):
            return physics.solve(
                *arguments,
                seed_pose=pose,
                key=key,
            )
        return physics.solve(*arguments, key=key)

    displacement, forward_wall = synchronized_time(solve)
    forward = copy.deepcopy(physics.runtime.last_forward)
    if forward_phase_receipt is not None:
        write_json(
            forward_phase_receipt,
            {
                "phase": "forward_complete_before_adjoint",
                "arm": arm,
                "repeat": repeat,
                "measured": measured,
                "forward_wall_seconds": forward_wall,
                "forward": forward,
            },
        )
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
    if arm == ARM_FULL_SOURCE:
        assert forward["contact"]["contact_numerically_valid"] is True
        assert isinstance(physics, FullSkullJointPhysics)
    else:
        assert "contact" not in forward
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
    return result


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.repeats_per_arm == 5
    assert cfg.warmups_per_arm == 1
    assert len(ARM_ORDER) == 2 * cfg.repeats_per_arm
    assert ARM_ORDER.count(ARM_OFF) == cfg.repeats_per_arm
    assert ARM_ORDER.count(ARM_FULL_SOURCE) == cfg.repeats_per_arm
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
    prior_weight = float(checkpoint["protocol"]["config"]["prior_weight"])
    smoothness_weight = float(
        checkpoint["protocol"]["shared_field"]["spatial_smoothness"]["weight"]
    )
    assert smoothness_weight == 100.0

    geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    admission = json.loads(cfg.admission.read_text())
    assert admission["schema"] == "joint-full-skull-contact-admission-v2"
    assert admission["success"] is True
    assert admission["equilibrium_converged"] is False
    assert admission["final_launch_ready"] is False
    assert admission["initialization_sha256"] == sha256(cfg.initialization)
    seed_array = load_admitted_initialization(admission, geometry)
    seed_cpu = torch.as_tensor(seed_array.copy())
    assert seed_cpu.dtype == torch.float64
    assert seed_cpu.shape == (geometry.fem_node_count, 3)
    assert (
        hashlib.sha256(np.ascontiguousarray(seed_array).tobytes()).hexdigest()
        == admission["initialization_displacement_sha256"]
    )
    adapter_validation = json.loads(cfg.adapter_validation.read_text())
    assert (
        adapter_validation["schema"] == "joint-full-skull-contact-adapter-validation-v1"
    )
    assert adapter_validation["success"] is True
    assert adapter_validation["audited_geometry_contact_admitted"] is False
    assert adapter_validation["soft_bone_initialization_admitted"] is True
    assert adapter_validation["bone_bone_contact_validated"] is False
    assert adapter_validation["final_launch_ready"] is False
    assert adapter_validation["initialization_admission"]["sha256"] == sha256(
        cfg.admission
    )

    contact_config = {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 1e-4,
        "stiffness_mpa": 0.01,
    }
    configure_cuda()
    hardware = identity()
    timing_contended = bool(
        hardware["contention_snapshot"]["other_python_compute_processes"]
    )
    seed = seed_cpu.to(device="cuda", dtype=torch.float64)
    seed_sha = tensor_sha256(seed)
    coefficient_sha = tensor_sha256(checkpoint["shared_coefficients"])
    material_config = research_informed_material_config()
    arms: dict[str, dict[str, Any]] = {}
    arm_definitions = (
        ((ARM_OFF, False), (ARM_FULL_SOURCE, True))
        if cfg.run_mode == "matched"
        else ((ARM_FULL_SOURCE, True),)
    )
    for arm, full_source in arm_definitions:
        memory_before = cuda_memory()
        physics, physics_seconds = synchronized_time(
            lambda full_source=full_source: make_physics(
                cfg,
                prepared,
                material_config,
                full_source=full_source,
                geometry=geometry,
                admission=admission,
                contact_config=contact_config,
            )
        )
        shared, shared_seconds = synchronized_time(
            lambda physics=physics: make_shared(checkpoint, physics)
        )
        assert isinstance(physics, JointPhysics)
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

    first_attempts: dict[str, dict[str, Any]] = {}
    for arm, _full_source in arm_definitions:
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            first_attempts[arm] = run_once(
                arm=arm,
                repeat=0,
                measured=False,
                physics=arms[arm]["physics"],
                shared=arms[arm]["shared"],
                checkpoint=checkpoint,
                seed=seed,
                prior_weight=prior_weight,
                smoothness_weight=smoothness_weight,
                forward_phase_receipt=output / f"{arm}-forward-receipt.json",
            )
        except ForwardConvergenceError as error:
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
            first_attempts[arm] = {
                "arm": arm,
                "success": False,
                "wall_seconds": wall_seconds,
                "forward": copy.deepcopy(error.receipt),
                "seed_sha256": seed_sha,
                "coefficient_sha256": coefficient_sha,
            }
            write_json(
                output / f"{arm}-forward-receipt.json",
                {
                    "phase": "forward_failed_before_adjoint",
                    "arm": arm,
                    "forward_wall_seconds": wall_seconds,
                    "forward": copy.deepcopy(error.receipt),
                },
            )
            write_json(output / "first-attempts.json", first_attempts)
            write_json(
                output / "summary.json",
                {
                    "schema": "joint-full-skull-forward-performance-benchmark-v1",
                    "success": False,
                    "status": "failed_first_equilibrium_attempt",
                    "failed_arm": arm,
                    "first_attempts": first_attempts,
                    "admission_sha256": sha256(cfg.admission),
                    "adapter_validation_sha256": sha256(cfg.adapter_validation),
                    "checkpoint_sha256": sha256(cfg.checkpoint),
                    "source_sha256": sha256(Path(__file__)),
                },
            )
            return
        assert first_attempts[arm]["forward_counts"]["newton_steps"] >= 0
        arms[arm]["equilibrium_seed"] = (
            arms[arm]["physics"]
            .runtime.forward.state.u[: geometry.fem_node_count]
            .detach()
            .clone()
        )
        write_json(output / "first-attempts.json", first_attempts)

    if cfg.run_mode == "complete_source_first_attempt":
        attempt = first_attempts[ARM_FULL_SOURCE]
        write_json(
            output / "summary.json",
            {
                "schema": "joint-full-skull-first-equilibrium-attempt-v1",
                "success": True,
                "status": "passed_complete_source_first_equilibrium_and_adjoint_attempt",
                "scope": {
                    "measured_contact": "pure FEM soft boundary versus complete registered source bones",
                    "complete_source_triangles_retained": True,
                    "source_bone_bone_contact": False,
                    "zero_jaw_pose_only": True,
                    "matched_on_off_ratio": False,
                    "final_launch_ready": False,
                },
                "first_attempt": attempt,
                "cold_setup": arms[ARM_FULL_SOURCE]["cold_setup"],
                "hardware_software": hardware,
                "full_skull_binding": arms[ARM_FULL_SOURCE][
                    "physics"
                ].full_skull_receipt(),
                "admission_sha256": sha256(cfg.admission),
                "adapter_validation_sha256": sha256(cfg.adapter_validation),
                "checkpoint_sha256": sha256(cfg.checkpoint),
                "initialization_sha256": sha256(cfg.initialization),
                "source_sha256": sha256(Path(__file__)),
            },
        )
        COMPLETED = True
        return

    warmups = []
    for index, arm in enumerate((ARM_OFF, ARM_FULL_SOURCE)):
        warmups.append(
            run_once(
                arm=arm,
                repeat=index,
                measured=False,
                physics=arms[arm]["physics"],
                shared=arms[arm]["shared"],
                checkpoint=checkpoint,
                seed=arms[arm]["equilibrium_seed"],
                prior_weight=prior_weight,
                smoothness_weight=smoothness_weight,
            )
        )
        write_json(output / "warmups.json", warmups)

    rows: list[dict[str, Any]] = []
    counts = {ARM_OFF: 0, ARM_FULL_SOURCE: 0}
    for sequence_index, arm in enumerate(ARM_ORDER):
        repeat = counts[arm]
        row = run_once(
            arm=arm,
            repeat=repeat,
            measured=True,
            physics=arms[arm]["physics"],
            shared=arms[arm]["shared"],
            checkpoint=checkpoint,
            seed=arms[arm]["equilibrium_seed"],
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
        for arm in (ARM_OFF, ARM_FULL_SOURCE)
    }
    ratios = {
        key: by_arm[ARM_FULL_SOURCE][key]["median"] / by_arm[ARM_OFF][key]["median"]
        for key in (
            "forward_wall_seconds",
            "forward_receipt_seconds",
            "adjoint_autograd_wall_seconds",
            "adjoint_receipt_seconds",
        )
    }
    success = all(
        row["metrics"]["inverted_tetrahedra"] == 0
        and row["metrics"]["detF_min"] >= 0.25
        and row["metrics"]["detF_max"] <= 2.0
        and row["adjoint_relative_residual"] <= 1.05e-7
        and (
            row["contact"] is None
            or row["contact"]["contact_numerically_valid"] is True
        )
        for row in [*first_attempts.values(), *warmups, *rows]
    )
    summary = {
        "schema": "joint-full-skull-forward-performance-benchmark-v1",
        "success": success,
        "status": (
            (
                "passed_contended_complete_source_soft_bone_on_off_benchmark"
                if timing_contended
                else "passed_uncontended_complete_source_soft_bone_on_off_benchmark"
            )
            if success
            else "failed_complete_source_soft_bone_on_off_benchmark"
        ),
        "scope": {
            "measured_contact": "pure FEM soft boundary versus complete registered source bones",
            "complete_source_triangles_retained": True,
            "source_bone_bone_contact": False,
            "zero_jaw_pose_only": True,
            "final_launch_ready": False,
            "claim_limit": (
                "soft-versus-complete-source-bone neutral solve overhead only; "
                "the source cranium-mandible domain is separately inadmissible"
            ),
            "timing_environment": (
                "contended; repeat once idle"
                if timing_contended
                else "no other Python GPU compute process detected at startup"
            ),
        },
        "workload": {
            "checkpoint": str(cfg.checkpoint.resolve()),
            "checkpoint_sha256": sha256(cfg.checkpoint),
            "checkpoint_update": checkpoint["update"],
            "initialization": str(cfg.initialization.resolve()),
            "initialization_sha256": sha256(cfg.initialization),
            "initialization_displacement_sha256": seed_sha,
            "coefficient_sha256": coefficient_sha,
            "all_coefficients_unchanged": True,
            "skin_resultant_n_per_m": float(
                checkpoint["metrics"]["material"]["skin_resultant_n_per_m"]
            ),
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
            "first_attempt_same_admitted_seed": True,
            "first_attempt_requires_convergence_before_warm_repeats": True,
            "warmups_per_arm": cfg.warmups_per_arm,
            "repeats_per_arm": cfg.repeats_per_arm,
            "measured_order": list(ARM_ORDER),
            "warmed_seed": "each arm's converged first-attempt equilibrium",
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
        "first_attempts": first_attempts,
        "cold_setup": {arm: value["cold_setup"] for arm, value in arms.items()},
        "warmed": by_arm,
        "complete_source_over_off_median_ratios": ratios,
        "hardware_software": hardware,
        "full_skull_binding": arms[ARM_FULL_SOURCE]["physics"].full_skull_receipt(),
        "admission_sha256": sha256(cfg.admission),
        "adapter_validation_sha256": sha256(cfg.adapter_validation),
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "spatial_basis": checkpoint["protocol"]["spatial_basis"],
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_contact.py"),
                Path(__file__).with_name("joint_equilibrium.py"),
                Path(__file__).with_name("joint_full_skull_contact.py"),
                Path(__file__).with_name("joint_newton.py"),
                Path(__file__).with_name("joint_physics.py"),
                Path(__file__).with_name("joint_spatial_fields.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "full_skull_benchmark/success": float(success),
            "full_skull_benchmark/forward_ratio": ratios["forward_wall_seconds"],
            "full_skull_benchmark/adjoint_ratio": ratios[
                "adjoint_autograd_wall_seconds"
            ],
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
