"""Diagnose curvature at the failed complete-skull collision-off cold state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_full_skull_contact import (
    extend_dof_map,
    load_admitted_initialization,
    load_full_skull_geometry,
)
from joint_physics import JointPhysics
from joint_spatial_fields import SpatialSharedFieldParameters, spatial_field_config

from liblaf import cherries
from liblaf.apple.forward import Forward, Model

COMPLETED = False


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
    admission: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )
    output_dir: Path = cherries.output("collision-off-curvature-diagnostic", mkdir=True)
    lanczos_steps: int = 30
    symmetry_trials: int = 3
    diagnostic_wall_cap_seconds: float = 120.0
    random_seed: int = 64001
    ablation_witness: Path | None = None


def tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().to(torch.float64).numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


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


def state(model: Model, displacement: torch.Tensor) -> Model.State:
    return model.State(u=displacement.detach().clone())


def force_receipt(
    model: Model,
    forward: Forward,
    displacement: torch.Tensor,
    materials: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    model.set_materials(materials)
    current = state(model, displacement)
    energy = forward.problem.fun(current)
    gradient = forward.problem.grad(current)
    assert torch.isfinite(energy)
    assert torch.isfinite(gradient).all()
    return {
        "energy": float(energy),
        "free_force_norm": float(torch.linalg.vector_norm(gradient)),
        "free_force_inf": float(torch.linalg.vector_norm(gradient, ord=float("inf"))),
    }


def normalized_random(
    size: int, *, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    value = torch.randn(size, generator=generator, device=device, dtype=torch.float64)
    value /= torch.linalg.vector_norm(value)
    return value


def symmetry_checks(
    problem: Any,
    current: Model.State,
    *,
    trials: int,
    generator: torch.Generator,
) -> list[dict[str, float]]:
    rows = []
    for trial in range(trials):
        left = normalized_random(
            problem.model.n_free, generator=generator, device=problem.device
        )
        right = normalized_random(
            problem.model.n_free, generator=generator, device=problem.device
        )
        h_left = problem.hess_prod(current, left)
        h_right = problem.hess_prod(current, right)
        left_h_right = float(torch.dot(left, h_right))
        right_h_left = float(torch.dot(right, h_left))
        scale = max(abs(left_h_right), abs(right_h_left), 1e-30)
        rows.append(
            {
                "trial": trial,
                "left_h_right": left_h_right,
                "right_h_left": right_h_left,
                "relative_error": abs(left_h_right - right_h_left) / scale,
            }
        )
    return rows


def lanczos(  # noqa: C901, PLR0915
    problem: Any,
    current: Model.State,
    sqrt_preconditioner: torch.Tensor,
    *,
    steps: int,
    wall_cap_seconds: float,
    generator: torch.Generator,
    witness_path: Path,
) -> dict[str, Any]:
    """Lanczos on M^(1/2) H M^(1/2), with full reorthogonalization."""
    started = time.perf_counter()
    matvec_count = 0

    def product(value: torch.Tensor) -> torch.Tensor:
        nonlocal matvec_count
        matvec_count += 1
        return sqrt_preconditioner * problem.hess_prod(
            current, sqrt_preconditioner * value
        )

    basis: list[torch.Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    q = normalized_random(
        problem.model.n_free, generator=generator, device=problem.device
    )
    previous: torch.Tensor | None = None
    previous_beta = 0.0
    stopped_by_wall_cap = False
    breakdown = False
    for iteration in range(steps):
        if time.perf_counter() - started >= wall_cap_seconds:
            stopped_by_wall_cap = True
            break
        basis.append(q)
        value = product(q)
        alpha = float(torch.dot(q, value))
        assert math.isfinite(alpha)
        residual = value - alpha * q
        if previous is not None:
            residual -= previous_beta * previous
        # Reorthogonalize twice: this is cheap for the declared 30-vector cap.
        for _pass in range(2):
            for vector in basis:
                residual -= torch.dot(vector, residual) * vector
        beta = float(torch.linalg.vector_norm(residual))
        assert math.isfinite(beta)
        alphas.append(alpha)
        if beta <= 1e-14:
            breakdown = True
            break
        if iteration + 1 < steps:
            betas.append(beta)
        previous, q = q, residual / beta
        previous_beta = beta

    torch.cuda.synchronize()
    iteration_elapsed = time.perf_counter() - started
    count = len(alphas)
    assert count > 0
    assert len(betas) >= count - 1
    betas = betas[: count - 1]
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if betas:
        off = np.asarray(betas, dtype=np.float64)
        tridiagonal += np.diag(off, 1) + np.diag(off, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    lowest = float(eigenvalues[0])
    coefficients = torch.as_tensor(
        eigenvectors[:, 0], device=problem.device, dtype=torch.float64
    )
    ritz = torch.zeros_like(basis[0])
    for coefficient, vector in zip(coefficients, basis, strict=True):
        ritz += coefficient * vector
    ritz /= torch.linalg.vector_norm(ritz)
    transformed_h_ritz = product(ritz)
    ritz_residual = float(
        torch.linalg.vector_norm(transformed_h_ritz - lowest * ritz)
        / max(float(torch.linalg.vector_norm(transformed_h_ritz)), abs(lowest), 1e-30)
    )
    witness = sqrt_preconditioner * ritz
    h_witness = problem.hess_prod(current, witness)
    matvec_count += 1
    repeated = problem.hess_prod(current, witness)
    matvec_count += 1
    quadratic = float(torch.dot(witness, h_witness))
    repeated_quadratic = float(torch.dot(witness, repeated))
    witness_norm = float(torch.linalg.vector_norm(witness))
    h_witness_norm = float(torch.linalg.vector_norm(h_witness))
    relative_scale = quadratic / max(witness_norm * h_witness_norm, 1e-30)
    rayleigh_quotient = quadratic / max(witness_norm**2, 1e-30)
    transformed_rayleigh = float(torch.dot(ritz, transformed_h_ritz))
    repeat_relative_error = abs(quadratic - repeated_quadratic) / max(
        abs(quadratic), abs(repeated_quadratic), 1e-30
    )
    verified_negative = bool(
        quadratic < 0 and relative_scale < -1e-8 and repeat_relative_error <= 1e-10
    )
    witness_artifact = None
    if verified_negative:
        np.savez_compressed(
            witness_path,
            free_direction=witness.detach().cpu().to(torch.float64).numpy(),
        )
        witness_artifact = {
            "path": str(witness_path.resolve()),
            "sha256": sha256(witness_path),
            "array": "free_direction",
            "coordinates": "original free FEM DOFs; appended source nodes are fixed",
        }
    elapsed = time.perf_counter() - started
    return {
        "declared_steps": steps,
        "completed_steps": count,
        "matvec_count": matvec_count,
        "elapsed_seconds": elapsed,
        "iteration_elapsed_seconds": iteration_elapsed,
        "wall_cap_seconds": wall_cap_seconds,
        "stopped_by_wall_cap": stopped_by_wall_cap,
        "breakdown": breakdown,
        "lowest_ritz_value": lowest,
        "largest_ritz_value": float(eigenvalues[-1]),
        "tridiagonal_alpha": alphas,
        "tridiagonal_beta": betas,
        "ritz_residual_relative": ritz_residual,
        "transformed_rayleigh": transformed_rayleigh,
        "explicit_witness": {
            "quadratic_v_h_v": quadratic,
            "repeated_quadratic_v_h_v": repeated_quadratic,
            "witness_norm": witness_norm,
            "h_witness_norm": h_witness_norm,
            "rayleigh_quotient": rayleigh_quotient,
            "quadratic_relative_scale": relative_scale,
            "repeat_relative_error": repeat_relative_error,
            "verified_negative_curvature": verified_negative,
            "witness_sha256": tensor_sha256(witness),
            "artifact": witness_artifact,
        },
        "interpretation": (
            "explicit negative-curvature witness"
            if verified_negative
            else "no robust negative witness found; this does not prove SPD"
        ),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.lanczos_steps == 30
    assert cfg.symmetry_trials == 3
    assert cfg.diagnostic_wall_cap_seconds == 120.0
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
    assert checkpoint["update"] == 111
    assert checkpoint["protocol"]["shared_basis"] == "spatial80"
    assert checkpoint["materials"] == spatial_field_config()
    geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    admission = json.loads(cfg.admission.read_text())
    candidate_array = load_admitted_initialization(admission, geometry)
    configure_cuda()
    occupancy_start = gpu_snapshot()

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
    model = Model(
        dof_map=extend_dof_map(original.dof_map, geometry),
        warp_model=original.warp_model,
        collision=None,
        device=original.device,
    )
    forward = Forward(model)
    candidate_fem = torch.as_tensor(candidate_array.copy(), device="cuda")
    appended_zero = candidate_fem.new_zeros(
        (geometry.cranium_node_count + geometry.mandible_node_count, 3)
    )
    candidate = torch.cat((candidate_fem, appended_zero))
    reference = torch.zeros_like(candidate)
    assert torch.count_nonzero(candidate.flatten()[model.dof_map.fixed_indices]) == 0

    loaded_materials = base.materials(
        shared.bulk_stresses_mpa(),
        shared.skin_resultant_n_per_m(),
        shared.skin_stiffness_multiplier(),
        None,
    )
    passive_materials = base.materials(
        torch.zeros((3, 3, 3), device="cuda", dtype=torch.float64),
        torch.zeros((2, 2), device="cuda", dtype=torch.float64),
        torch.ones((), device="cuda", dtype=torch.float64),
        None,
    )
    zero_baseline_fitted_stiffness = base.materials(
        torch.zeros((3, 3, 3), device="cuda", dtype=torch.float64),
        torch.zeros((2, 2), device="cuda", dtype=torch.float64),
        shared.skin_stiffness_multiplier(),
        None,
    )
    forces = {
        "passive_reference_zero_displacement": force_receipt(
            model, forward, reference, passive_materials
        ),
        "loaded_reference_zero_displacement": force_receipt(
            model, forward, reference, loaded_materials
        ),
        "zero_baseline_fitted_stiffness_reference": force_receipt(
            model, forward, reference, zero_baseline_fitted_stiffness
        ),
        "passive_repaired_candidate": force_receipt(
            model, forward, candidate, passive_materials
        ),
        "loaded_repaired_candidate": force_receipt(
            model, forward, candidate, loaded_materials
        ),
        "zero_baseline_fitted_stiffness_repaired": force_receipt(
            model, forward, candidate, zero_baseline_fitted_stiffness
        ),
    }
    assert math.isclose(
        forces["loaded_repaired_candidate"]["free_force_norm"],
        6.784123355571406e-6,
        rel_tol=1e-10,
        abs_tol=1e-15,
    )

    if cfg.ablation_witness is not None:
        with np.load(cfg.ablation_witness) as archive:
            assert set(archive.files) == {"free_direction"}
            witness = torch.as_tensor(archive["free_direction"].copy(), device="cuda")
        assert witness.shape == (model.n_free,)
        witness_norm = float(torch.linalg.vector_norm(witness))
        cases = {}
        for geometry_name, displacement in (
            ("reference", reference),
            ("repaired", candidate),
        ):
            for baseline_name, case_materials in (
                ("zero_baseline", zero_baseline_fitted_stiffness),
                ("fitted_baseline", loaded_materials),
            ):
                model.set_materials(case_materials)
                current = state(model, displacement)
                h_witness = forward.problem.hess_prod(current, witness)
                repeated = forward.problem.hess_prod(current, witness)
                quadratic = float(torch.dot(witness, h_witness))
                repeated_quadratic = float(torch.dot(witness, repeated))
                cases[f"{geometry_name}_{baseline_name}"] = {
                    "quadratic_v_h_v": quadratic,
                    "repeated_quadratic_v_h_v": repeated_quadratic,
                    "rayleigh_quotient": quadratic / witness_norm**2,
                    "quadratic_relative_scale": quadratic
                    / max(
                        witness_norm * float(torch.linalg.vector_norm(h_witness)),
                        1e-30,
                    ),
                    "repeat_relative_error": abs(quadratic - repeated_quadratic)
                    / max(abs(quadratic), abs(repeated_quadratic), 1e-30),
                    "negative_on_saved_mode": quadratic < 0,
                }
        curvature_delta = {
            geometry_name: cases[f"{geometry_name}_fitted_baseline"]["quadratic_v_h_v"]
            - cases[f"{geometry_name}_zero_baseline"]["quadratic_v_h_v"]
            for geometry_name in ("reference", "repaired")
        }
        summary = {
            "schema": "joint-collision-off-prestress-curvature-ablation-v1",
            "success": all(
                row["repeat_relative_error"] <= 1e-10 for row in cases.values()
            ),
            "scope": {
                "collision": False,
                "equilibrium_solve": False,
                "fixed_skin_stiffness_multiplier": float(
                    shared.skin_stiffness_multiplier()
                ),
                "ablation": "toggle all fitted bulk and skin baseline stresses at fixed geometry and stiffness",
                "claim_limit": "cross-evaluation of the saved loaded/repaired negative mode; not a minimum eigenvalue search",
            },
            "saved_witness": {
                "path": str(cfg.ablation_witness.resolve()),
                "sha256": sha256(cfg.ablation_witness),
                "tensor_sha256": tensor_sha256(witness),
                "norm": witness_norm,
            },
            "forces": forces,
            "curvature_cases": cases,
            "fitted_baseline_curvature_delta_at_fixed_geometry": curvature_delta,
            "occupancy": {"start": occupancy_start, "end": gpu_snapshot()},
            "inputs": {
                "checkpoint_sha256": sha256(cfg.checkpoint),
                "admission_sha256": sha256(cfg.admission),
                "geometry_sha256": geometry.geometry_sha256,
                "material_config": material_config,
                "spatial_basis": basis,
            },
            "source_hashes": {
                str(path.resolve()): sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("joint_materials.py"),
                    Path(__file__).with_name("joint_physics.py"),
                    Path(__file__).with_name("joint_spatial_fields.py"),
                )
            },
        }
        write_json(output / "summary.json", summary)
        cherries.log_metrics(
            {
                "prestress_curvature_ablation/success": float(summary["success"]),
                "prestress_curvature_ablation/reference_delta": curvature_delta[
                    "reference"
                ],
                "prestress_curvature_ablation/repaired_delta": curvature_delta[
                    "repaired"
                ],
            }
        )
        COMPLETED = bool(summary["success"])
        return

    model.set_materials(loaded_materials)
    current = state(model, candidate)
    diagonal = forward.problem.hess_diag(current)
    nonfinite_diagonal_count = int(torch.count_nonzero(~torch.isfinite(diagonal)))
    assert nonfinite_diagonal_count == 0
    absolute_diagonal = diagonal.abs()
    zero_diagonal_count = int(torch.count_nonzero(absolute_diagonal == 0))
    assert zero_diagonal_count == 0
    sqrt_preconditioner = absolute_diagonal.rsqrt()
    assert torch.isfinite(sqrt_preconditioner).all()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(cfg.random_seed)
    symmetry = symmetry_checks(
        forward.problem,
        current,
        trials=cfg.symmetry_trials,
        generator=generator,
    )
    maximum_symmetry_error = max(row["relative_error"] for row in symmetry)
    assert maximum_symmetry_error <= 1e-8

    lanczos_result = lanczos(
        forward.problem,
        current,
        sqrt_preconditioner,
        steps=cfg.lanczos_steps,
        wall_cap_seconds=cfg.diagnostic_wall_cap_seconds,
        generator=generator,
        witness_path=output / "negative-curvature-witness.npz",
    )
    assert lanczos_result["elapsed_seconds"] <= (cfg.diagnostic_wall_cap_seconds + 5.0)
    success = bool(
        maximum_symmetry_error <= 1e-8
        and not lanczos_result["stopped_by_wall_cap"]
        and lanczos_result["completed_steps"] == cfg.lanczos_steps
    )
    summary = {
        "schema": "joint-collision-off-curvature-diagnostic-v1",
        "success": success,
        "status": "passed_bounded_fixed_state_curvature_diagnostic"
        if success
        else "failed_bounded_fixed_state_curvature_diagnostic",
        "scope": {
            "collision": False,
            "equilibrium_solve": False,
            "state": "candidate002 repaired displacement with frozen Spatial80 update111 coefficients",
            "claim_limit": (
                "a negative explicit vHv proves local negative curvature; "
                "nonnegative sampled/Ritz values do not prove SPD"
            ),
        },
        "forces": forces,
        "hessian_diagonal": {
            "minimum": float(diagonal.min()),
            "maximum": float(diagonal.max()),
            "negative_count": int(torch.count_nonzero(diagonal < 0)),
            "zero_count": zero_diagonal_count,
            "nonfinite_count": nonfinite_diagonal_count,
            "minimum_absolute": float(absolute_diagonal.min()),
            "maximum_absolute": float(absolute_diagonal.max()),
            "absolute_dynamic_range": float(
                absolute_diagonal.max() / absolute_diagonal.min()
            ),
            "preconditioner": "same reciprocal-absolute-diagonal convention as joint_newton.NewtonSystem",
            "free_dof_count": model.n_free,
            "appended_source_dofs": "all fixed; original free indices preserved",
        },
        "symmetry": {
            "trials": symmetry,
            "maximum_relative_error": maximum_symmetry_error,
        },
        "lanczos": lanczos_result,
        "occupancy": {
            "start": occupancy_start,
            "end": gpu_snapshot(),
        },
        "inputs": {
            "checkpoint": str(cfg.checkpoint.resolve()),
            "checkpoint_sha256": sha256(cfg.checkpoint),
            "shared_coefficients_sha256": tensor_sha256(
                checkpoint["shared_coefficients"]
            ),
            "admission": str(cfg.admission.resolve()),
            "admission_sha256": sha256(cfg.admission),
            "candidate_sha256": admission["initialization_sha256"],
            "candidate_displacement_sha256": admission[
                "initialization_displacement_sha256"
            ],
            "geometry_sha256": geometry.geometry_sha256,
            "geometry_audit_sha256": geometry.audit_sha256,
            "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
            "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
            "material_config": material_config,
            "spatial_basis": basis,
        },
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("joint_full_skull_contact.py"),
                Path(__file__).with_name("joint_materials.py"),
                Path(__file__).with_name("joint_newton.py"),
                Path(__file__).with_name("joint_physics.py"),
                Path(__file__).with_name("joint_spatial_fields.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "collision_off_curvature/success": float(success),
            "collision_off_curvature/lowest_ritz": lanczos_result["lowest_ritz_value"],
            "collision_off_curvature/negative_witness": float(
                lanczos_result["explicit_witness"]["verified_negative_curvature"]
            ),
            "collision_off_curvature/passive_reference_force": forces[
                "passive_reference_zero_displacement"
            ]["free_force_norm"],
            "collision_off_curvature/loaded_candidate_force": forces[
                "loaded_repaired_candidate"
            ]["free_force_norm"],
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
