# ruff: noqa: C901, E402, PLR0912, PLR0915, PT018
"""Projected-Newton variant of new-neutral/src/30-forward-active-strain.py.

Copied from exp/2026/09/23/new-neutral/src/30-forward-active-strain.py.  Changes:
optional per-element PSD projection of the sparse Newton search matrix
(``projection``), an optional Newton-only path (``skip_pncg``), and the exact
sparse-HVP gate recorded instead of asserted when the search matrix is
deliberately projected.  Physics, tolerances and acceptance gates are unchanged.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.inverse import DifferentiableForward

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[4]
NEW_NEUTRAL = ROOT / "exp/2026/09/23/new-neutral"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
NEUTRAL = ROOT / "exp/2026/09/22/neutral-newton"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [
    str(HERE),
    str(NEW_NEUTRAL / "src"),
    str(SOLVERS),
    str(JOINT / "src"),
    str(NEUTRAL / "src"),
]

spec = importlib.util.spec_from_file_location(
    "neutral_adaptive_benchmark", SOLVERS / "10-benchmark.py"
)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)

import accelerated_solvers
import hybrid_first_solver
import projected_hessian
from accelerated_solvers import CachedProblem
from adaptive_ipc_stiffness import AdaptiveIPCStiffness
from assembled_fem_hvp import AssembledFemHvp
from gpu_free_sparse_hessian import GpuFreeSparseHessian
from hybrid_hessian import HybridHessian
from inverse_timing import install_inverse_timing
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from joint_physics import moduli
from joint_rigid_eye_contact import build_eye_collision_physics
from mesh_step_scale import mean_rest_edge_length
from neutral_active_strain import install_active_strain, verify_equivalence
from profile_input_binding import bind_frozen_neutral_load
from smile_collision import audit_collision_state, audit_required_collision


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/forward-active-strain-001"
    validation_file: Path = NEW_NEUTRAL / "data/active-strain-check.json"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    seed_dir: Path = NEUTRAL / "data/reference-seed-002"
    reference_dir: Path | None = None
    resume_dir: Path | None = None
    max_newton_steps: int = 100
    checkpoint_steps: int = 100
    newton_shift_policy: Literal["reset", "reuse"] = "reset"
    reuse_shift_force_ratio: float = 3.0
    mode: Literal["fixed", "adaptive"] = "adaptive"
    fixed_stiffness_multiplier: float = 1.0
    epsilon_scale: float = 1e-6
    maximum_stiffness_multiplier: float = 100.0
    ipc_threads: int = 4
    cuda_sync: bool = True
    projection: Literal["none", "clamp", "abs"] = "clamp"
    skip_pncg: bool = False
    cell_batch: int = 16384
    seed_endpoint: Path | None = None
    target_force_override: float | None = None


def _sha256(path: Path) -> str:
    return benchmark.sha256(path)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _gpu_processes() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _copy_neutral_sources(output_dir: Path) -> dict[str, Any]:
    target = output_dir / "sources" / "neutral-newton"
    shutil.copytree(
        NEUTRAL / "src", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    receipt = {
        "source_root": str((NEUTRAL / "src").resolve()),
        "sources": {
            str(path.relative_to(target)): _sha256(path)
            for path in sorted(target.rglob("*.py"))
        },
    }
    benchmark.write_json(output_dir / "neutral-newton-source-provenance.json", receipt)
    return receipt


def _verify_materials(baseline: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    expected = {"fat": 0.0112, "muscle": 0.012, "aponeurosis": 1.693}
    result: dict[str, float] = {}
    for name, young in expected.items():
        mu, la = moduli(young, 0.49)
        assert torch.allclose(
            baseline[name]["mu"], torch.full_like(baseline[name]["mu"], mu)
        )
        assert torch.allclose(
            baseline[name]["lmbda"], torch.full_like(baseline[name]["lmbda"], la)
        )
        assert not bool(torch.count_nonzero(baseline[name]["active_stress"])), name
        result[name] = young
    skin = baseline["skin"]
    skin_young = skin["mu"] * (2 * 1.49)
    assert bool(torch.count_nonzero(skin["baseline_stress"]))
    result["skin_maximum"] = float(skin_young.max())
    return result


def _set_stiffness(collision: Any, stiffness: float) -> None:
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(),
        potential.dhat,
        stiffness,
        collision.use_physical_barrier,
    )


def main(cfg: Config) -> None:
    assert all(
        getattr(DifferentiableForward, name) is forbidden_inverse
        for name in ("forward", "step", "adjoint_solve", "receipt")
    )
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.ipc_threads > 0 and cfg.fixed_stiffness_multiplier > 0
    assert cfg.epsilon_scale > 0 and cfg.maximum_stiffness_multiplier >= 1
    assert cfg.max_newton_steps > 0 and cfg.checkpoint_steps > 0
    projection_stats = projected_hessian.install(
        cfg.projection, cell_batch=cfg.cell_batch
    )
    parent = None
    parent_protocol = None
    parent_stiffness = None
    continuation = None
    if cfg.resume_dir is not None:
        assert cfg.reference_dir is not None
        parent_protocol = json.loads((cfg.resume_dir / "protocol.json").read_text())
        parent = json.loads((cfg.resume_dir / "summary.json").read_text())["result"]
        parent_stiffness = json.loads((cfg.resume_dir / "stiffness.json").read_text())
        audit = json.loads((cfg.resume_dir / "independent-audit.json").read_text())
        for name in (
            "protocol.json",
            "summary.json",
            "endpoint.npz",
            "active-strain-fields.npz",
        ):
            assert _sha256(cfg.resume_dir / name) == audit["run_inputs"][name]["sha256"]
        assert not parent["success"] and parent["geometry"]["inverted_tetrahedra"] == 0
        assert parent["collision"]["state_feasible"]
        for name in (
            "neutral_dir",
            "eyes_dir",
            "reference_dir",
            "mode",
            "epsilon_scale",
        ):
            assert str(getattr(cfg, name)) == str(parent_protocol["config"][name]), name
        continuation = {
            "parent_directory": str(cfg.resume_dir.resolve()),
            **{
                f"parent_{name}": _record(cfg.resume_dir / filename)
                for name, filename in (
                    ("protocol", "protocol.json"),
                    ("summary", "summary.json"),
                    ("endpoint", "endpoint.npz"),
                    ("stiffness", "stiffness.json"),
                    ("audit", "independent-audit.json"),
                    ("trace", "trace.jsonl"),
                )
            },
            "method": f"Continue the existing Newton phase with {cfg.newton_shift_policy} shift policy",
        }
    cfg.output_dir.mkdir(parents=True)
    validation = json.loads(cfg.validation_file.read_text())
    assert validation["success"]
    shutil.copy2(cfg.validation_file, cfg.output_dir / "active-strain-check.json")
    guard_receipt = {
        "guarded_entrypoints": ["forward", "step", "adjoint_solve", "receipt"],
        "reviewed_inverse_sha256": "0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
        "completed_without_inverse_calls": False,
    }
    benchmark.write_json(cfg.output_dir / "forward-only-guard.json", guard_receipt)
    benchmark.archive_benchmark_sources(
        benchmark.Config.model_construct(output_dir=cfg.output_dir, source_root=ROOT)
    )
    neutral_sources = _copy_neutral_sources(cfg.output_dir)
    shutil.copytree(
        NEW_NEUTRAL / "src",
        cfg.output_dir / "sources/new-neutral",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        HERE,
        cfg.output_dir / "sources/projected-newton-neutral",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    active_sources = {
        str(path.relative_to(HERE)): _sha256(path) for path in sorted(HERE.glob("*.py"))
    }
    benchmark.write_json(
        cfg.output_dir / "active-strain-source-provenance.json", active_sources
    )
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    assert ipctk.get_num_threads() == cfg.ipc_threads
    ipctk_version = importlib.metadata.version("ipctk")
    assert ipctk.__version__ == ipctk_version

    reference_receipt = None
    if cfg.reference_dir is None:
        seed_path = cfg.seed_dir / "seed.npz"
        seed_receipt_path = cfg.seed_dir / "summary.json"
        seed_receipt = json.loads(seed_receipt_path.read_text())
        assert (
            seed_receipt["success"] and seed_receipt["prior_equilibrium_used"] is False
        )
        assert seed_receipt["fem_reference_rebased"] is False
        assert seed_receipt["seed"]["sha256"] == _sha256(seed_path)
        with np.load(seed_path, allow_pickle=False) as archive:
            seed_np = archive["displacement_m"].copy()

    setup_started = time.perf_counter()
    with bind_frozen_neutral_load(
        cfg.neutral_dir,
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
        allow_isfixed_boundary=True,
        unused_inverse_sha256="0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
    ) as neutral:
        physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    if cfg.reference_dir is not None:
        from reference_rebase import rebase_reference

        physics, baseline, reference_receipt = rebase_reference(
            physics, baseline, cfg.reference_dir, cfg.output_dir
        )
        seed_np = np.zeros_like(physics.points)
        if cfg.seed_endpoint is not None:
            with np.load(cfg.seed_endpoint, allow_pickle=False) as archive:
                seed_np = archive["displacement_m"].copy()
        if cfg.resume_dir is not None:
            with np.load(
                cfg.resume_dir / "endpoint.npz", allow_pickle=False
            ) as archive:
                seed_np = archive["displacement_m"].copy()
        seed_path = cfg.output_dir / "seed.npz"
        seed_receipt_path = cfg.output_dir / "seed-receipt.json"
        np.savez_compressed(seed_path, displacement_m=seed_np)
        seed_receipt = {
            "success": True,
            "prior_equilibrium_used": False,
            "prior_endpoint_used": cfg.resume_dir is not None
            or cfg.seed_endpoint is not None,
            "fem_reference_rebased": True,
            "origin": "saved unconverged endpoint"
            if cfg.resume_dir
            else f"saved endpoint {cfg.seed_endpoint}"
            if cfg.seed_endpoint
            else "zero displacement from clearance-repaired constitutive reference",
            "seed": _record(seed_path),
        }
        benchmark.write_json(seed_receipt_path, seed_receipt)
    model = physics.runtime.forward.model
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    prescribed = np.ones((model.n_points, 3), dtype=bool)
    prescribed[: len(physics.points)] = is_fixed[:, None]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(prescribed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~prescribed)
    )
    np.testing.assert_array_equal(
        physics.full_skull.geometry.fixed_global_ids, np.flatnonzero(is_fixed)
    )
    boundary_receipt = {
        **physics.fixed_boundary_receipt,
        "runtime_dofs_verified_against_isfixed": True,
        "free_dofs": model.n_free,
        "fixed_dofs": model.n_fixed,
        "fixed_lip_vertices": int(
            (is_fixed & np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)).sum()
        ),
    }
    benchmark.write_json(cfg.output_dir / "fixed-boundary.json", boundary_receipt)
    model.set_materials(baseline)
    young = _verify_materials(baseline)
    baseline, strain_receipt, strain_context = install_active_strain(model)
    model.set_materials(baseline)
    np.savez_compressed(
        cfg.output_dir / "active-strain-fields.npz", **strain_context["arrays"]
    )
    strain_receipt["fields"] = _record(cfg.output_dir / "active-strain-fields.npz")
    strain_receipt["derivative_validation"] = _record(
        cfg.output_dir / "active-strain-check.json"
    )
    stiffest_material = max(young, key=young.__getitem__)
    kappa_anchor = 0.1 * young[stiffest_material]
    kappa_initial = kappa_anchor * cfg.fixed_stiffness_multiplier
    kappa_maximum = kappa_initial * cfg.maximum_stiffness_multiplier
    if parent is not None:
        assert parent_protocol is not None and parent_stiffness is not None
        assert (
            parent_protocol["contact_stiffness_policy"]["anchor_stiffness_mpa"]
            == kappa_anchor
        )
        kappa_initial = parent["terminal_stiffness_mpa"]
        kappa_maximum = parent_stiffness["max_stiffness"]
        assert kappa_initial == parent_stiffness["final_stiffness"]
        assert _sha256(cfg.output_dir / "active-strain-fields.npz") == _sha256(
            cfg.resume_dir / "active-strain-fields.npz"
        )
        for name in ("repair", "repair_receipt"):
            assert (
                reference_receipt[name]["sha256"]
                == parent_protocol["reference_configuration"][name]["sha256"]
            )
        import pyvista as pv

        for name in ("constitutive_volume", "constitutive_skin"):
            previous_mesh = pv.read(
                parent_protocol["reference_configuration"][name]["path"]
            )
            current_mesh = pv.read(reference_receipt[name]["path"])
            np.testing.assert_array_equal(current_mesh.points, previous_mesh.points)
            connectivity = "cells" if name == "constitutive_volume" else "faces"
            np.testing.assert_array_equal(
                getattr(current_mesh, connectivity),
                getattr(previous_mesh, connectivity),
            )
            for association in ("point_data", "cell_data", "field_data"):
                previous_arrays = getattr(previous_mesh, association)
                current_arrays = getattr(current_mesh, association)
                assert set(current_arrays) == set(previous_arrays)
                for array_name in previous_arrays:
                    np.testing.assert_array_equal(
                        current_arrays[array_name], previous_arrays[array_name]
                    )
    pose = torch.zeros(
        6,
        device=model.dof_map.fixed_values.device,
        dtype=model.dof_map.fixed_values.dtype,
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    collision = model.collision
    assert collision is not None
    collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=1e-10, max_iterations=100000
    )
    collision.min_distance = 1e-8
    _set_stiffness(collision, kappa_initial if parent is not None else kappa_anchor)
    assert seed_np.shape == physics.points.shape and np.isfinite(seed_np).all()
    seed = torch.as_tensor(seed_np, device=pose.device, dtype=pose.dtype)
    full_seed = physics.full_skull.extend_seed(seed, pose)
    projected = model.dof_map.to_full(model.dof_map.to_free(full_seed))
    seed = projected[: len(physics.points)].detach().clone()
    seed_collision = audit_collision_state(physics, seed, pose)
    assert seed_collision["state_feasible"], seed_collision
    state = model.State(u=projected.detach().clone())
    state.collision = collision.state_at(state.u)
    initial_material_gradient = torch.zeros_like(state.u)
    model.warp_model.grad(state.u, initial_material_gradient)
    initial_contact_gradient = torch.zeros_like(state.u)
    collision.grad(state.collision, state.u, initial_contact_gradient)
    initial_force_components_n = {
        name: 1e6
        * float(torch.linalg.vector_norm(model.dof_map.to_free_grad(gradient)))
        for name, gradient in {
            "material": initial_material_gradient,
            "contact": initial_contact_gradient,
            "total": initial_material_gradient + initial_contact_gradient,
        }.items()
    }
    if reference_receipt is not None and parent is None and cfg.seed_endpoint is None:
        assert seed_collision["active_contact_count"] == 0, seed_collision
        assert initial_force_components_n["contact"] == 0.0, initial_force_components_n
    verify_equivalence(model, state, strain_context, strain_receipt)
    benchmark.write_json(cfg.output_dir / "active-strain-mapping.json", strain_receipt)
    del strain_context
    delegate = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
    problem = CachedProblem(delegate, exact_curvature=False)
    direction = torch.full_like(model.dof_map.to_free(state.u), 1e-6)
    prewarm_started = time.perf_counter()
    with torch.no_grad():
        delegate.fun(state)
        anchor_gradient = delegate.grad(state)
        delegate.hess_diag(state)
        delegate.hess_prod(state, direction)
        delegate.hess_quad(state, direction)
        delegate.max_step_size(state, direction)
    torch.cuda.synchronize()
    prewarm_seconds = time.perf_counter() - prewarm_started
    anchor_force = float(torch.linalg.vector_norm(anchor_gradient))
    target_force = max(1e-8, 1e-3 * anchor_force)
    if cfg.target_force_override is not None:
        target_force = cfg.target_force_override
    if parent is not None:
        restarted_energy = float(delegate.fun(state))
        np.testing.assert_allclose(
            anchor_force, parent["terminal_force"], rtol=1e-9, atol=1e-16
        )
        np.testing.assert_allclose(
            restarted_energy, parent["terminal_energy"], rtol=1e-12, atol=1e-18
        )
        continuation["restart_force"] = anchor_force
        continuation["restart_energy"] = restarted_energy
        anchor_force = parent_protocol["contact_stiffness_policy"][
            "tolerance_anchor_force"
        ]
        target_force = parent_protocol["contact_stiffness_policy"][
            "effective_force_tolerance"
        ]
        assert target_force == max(1e-8, 1e-3 * anchor_force)
    _set_stiffness(collision, kappa_initial)
    problem.invalidate()
    state.collision.hess = None
    controller = AdaptiveIPCStiffness(
        collision,
        initial_stiffness=kappa_initial,
        enabled=cfg.mode == "adaptive",
        epsilon_scale=cfg.epsilon_scale,
        max_stiffness=kappa_maximum,
    )
    coverage = audit_required_collision(physics)
    edge_mean = mean_rest_edge_length(model, physics.points)
    max_step = 0.5 * edge_mean
    if parent_protocol is not None:
        assert max_step == parent_protocol["solver"]["max_coordinate_displacement_m"]
    boundary_roundoff = float((projected - full_seed).abs().max())
    boundary_roundoff_limit = float(
        8 * np.finfo(np.float64).eps * np.abs(physics.points).max()
    )
    assert boundary_roundoff <= boundary_roundoff_limit, (
        boundary_roundoff,
        boundary_roundoff_limit,
    )
    setup_seconds = time.perf_counter() - setup_started - prewarm_seconds

    protocol = {
        "schema": "natural-reference-active-strain-hybrid-v1",
        "fixed_boundary": boundary_receipt,
        "config": cfg.model_dump(mode="json"),
        "fixture": {
            "name": "repaired constitutive-reference natural face",
            "neutral_manifest": _record(cfg.neutral_dir / "manifest.json"),
            "eyes_manifest": _record(cfg.eyes_dir / "manifest.json"),
            "seed": _record(seed_path),
            "seed_receipt": _record(seed_receipt_path),
            "prior_equilibrium_used": False,
            "prior_endpoint_used": cfg.resume_dir is not None
            or cfg.seed_endpoint is not None,
            "fem_reference_rebased": reference_receipt is not None,
            "activation": "bulk B=I; skin tangential B=sqrt(I+T/(h*mu)); no additive stress fields",
            "jaw_rotation_rad": 0.0,
        },
        "coverage": coverage,
        "seed_collision": seed_collision,
        "initial_force_components_n": initial_force_components_n,
        "boundary_projection": {
            "maximum_roundoff_m": boundary_roundoff,
            "roundoff_limit_m": boundary_roundoff_limit,
        },
        "materials": {
            "active_strain": strain_receipt,
            "young_moduli_mpa": young,
            "stiffest_material": stiffest_material,
        },
        "contact_stiffness_policy": {
            "anchor_rule": "0.1 * maximum deformable-material Young modulus",
            "anchor_stiffness_mpa": kappa_anchor,
            "initial_stiffness_mpa": kappa_initial,
            "fixed_stiffness_multiplier": cfg.fixed_stiffness_multiplier,
            "maximum_stiffness_mpa": kappa_maximum,
            "epsilon_scale": cfg.epsilon_scale,
            "mode": cfg.mode,
            "tolerance_anchor_force": anchor_force,
            "effective_force_tolerance": target_force,
            "tolerance_rule": "max(1e-8, 1e-3 * free-force norm at anchor kappa before optional arm multiplier)",
        },
        "solver": {
            "method": (
                "exact sparse Newton-CG"
                if cfg.skip_pncg
                else "per-contribution-clamped PNCG then exact sparse Newton-CG"
            ),
            "newton_search_matrix": (
                "exact physical Hessian"
                if cfg.projection == "none"
                else f"per-element PSD projection ({cfg.projection}) of bulk, membrane and IPC barrier Hessians"
            ),
            "pncg": {
                "damping": 0,
                "backtracking": False,
                "curvature": "ipctk GN diagonal and per-contact GN quadratic contributions clamped before summation",
                "window_steps": 20,
                "required_poor_comparisons": 2,
                "minimum_force_reduction": 0.1,
            },
            "newton": {
                "max_steps": cfg.max_newton_steps,
                "linear_rtol": 1e-3,
                "linear_max_steps": 1000,
                "preconditioner": "abs_jacobi",
                "initial_shift": 0.0,
                "shift_policy": cfg.newton_shift_policy,
                "reuse_shift_force_ratio": cfg.reuse_shift_force_ratio,
                "shift": "mean signed Hessian diagonal then x10; 8 attempts",
                "armijo": 1e-4,
                "backtracking": "x0.5, 8 trials",
            },
            "max_coordinate_displacement_m": max_step,
            "ccd": {
                "tolerance_m": 1e-10,
                "max_iterations": 100000,
                "safety": 0.9,
                "min_distance_m": collision.min_distance,
            },
        },
        "sources": {"neutral_newton": neutral_sources, "active_strain": active_sources},
        "model_setup_seconds": setup_seconds,
        "operator_prewarm_seconds": prewarm_seconds,
        "prewarm_scope": "one untimed physical evaluation at the cold seed; no displacement update and no sparse Newton setup",
        "runtime_binding": _record(cfg.output_dir / "profile-input-binding.json"),
        "versions": {
            "ipctk": ipctk_version,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "gpu": torch.cuda.get_device_name(0),
        "pid": os.getpid(),
        "compute_processes_before": _gpu_processes(),
        "timing": "Hierarchical CUDA-synchronized timing includes trace work, adaptive updates, and first sparse Newton setup. The uninstrumented operator prewarm and model construction are excluded; profiling perturbs overlap.",
    }
    if reference_receipt is not None:
        protocol["reference_configuration"] = reference_receipt
    if continuation is not None:
        protocol["continuation"] = continuation
        protocol["solver"]["method"] = (
            "Continuation of hybrid solver's exact sparse Newton-CG phase"
        )
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": True, "stage": "forward"}
    )

    installed = install_inverse_timing(model, cuda_sync=cfg.cuda_sync)
    timer = installed.timer
    for owner, attribute, label, sync in (
        (type(collision), "hess_quad", "collision/hess_quad", True),
        (
            ipctk.BarrierPotential,
            "gauss_newton_hessian_diagonal",
            "ipc/gauss_newton_hessian_diagonal",
            False,
        ),
        (
            type(collision),
            "raw_hess_quad_terms",
            "ipc/per_contact_gauss_newton_batch",
            False,
        ),
        (hybrid_first_solver, "run_pncg_phase", "pncg", True),
        (hybrid_first_solver, "safeguarded_newton", "newton", True),
        (accelerated_solvers, "pcg", "pcg", True),
        (AdaptiveIPCStiffness, "after_update", "contact_stiffness/update", True),
        (HybridHessian, "prepare", "hessian/cache_prepare", False),
        (HybridHessian, "apply", "hessian/spmv", True),
        (HybridHessian, "diagonal", "hessian/diagonal", True),
        (AssembledFemHvp, "__init__", "hessian/fem_constructor", True),
        (AssembledFemHvp, "setup", "hessian/fem_numeric", True),
        (
            projected_hessian.ProjectedAssembledFemHvp,
            "setup",
            "hessian/projected_fem_numeric",
            True,
        ),
        (GpuFreeSparseHessian, "__init__", "hessian/csr_constructor", True),
        (GpuFreeSparseHessian, "setup", "hessian/csr_refresh", True),
    ):
        timer.patch(owner, attribute, label, sync=sync)
    assert not installed.missing and not timer.missing, (
        installed.missing,
        timer.missing,
    )
    original_step = accelerated_solvers.safeguarded_newton_step
    original_diagonal = hybrid_first_solver.SparseNewtonProblem.hess_diag
    original_update = controller.after_update
    hvp_validation: list[dict[str, float]] = []
    newton_steps = 0
    result: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (cfg.output_dir / "trace.jsonl").open("x") as trace:

        def record(row: dict[str, Any]) -> None:
            trace.write(
                json.dumps(
                    benchmark.jsonable(
                        {
                            **row,
                            "stiffness_mpa": controller.current_stiffness,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                )
                + "\n"
            )
            trace.flush()

        def update_stiffness(
            current_problem: Any, current_state: Any, **kwargs: Any
        ) -> Any:
            event = original_update(current_problem, current_state, **kwargs)
            if event["stiffness_changed"]:
                record({"kind": "stiffness_change", **event})
            if (
                kwargs["phase"] == "newton"
                and kwargs["step"] % cfg.checkpoint_steps == 0
            ):
                checkpoint = (
                    cfg.output_dir / "checkpoints" / f"newton-{kwargs['step']:06d}"
                )
                checkpoint.mkdir(parents=True)
                np.savez_compressed(
                    checkpoint / "endpoint.npz",
                    displacement_m=current_state.u[: len(physics.points)].numpy(
                        force=True
                    ),
                )
                benchmark.write_json(
                    checkpoint / "state.json",
                    {
                        "step": kwargs["step"],
                        "force": float(
                            torch.linalg.vector_norm(
                                current_problem.grad(current_state)
                            )
                        ),
                        "energy": float(current_problem.fun(current_state)),
                        "stiffness_mpa": controller.current_stiffness,
                        "previous_min_distance_squared": controller.previous_min_distance_squared,
                        "endpoint": _record(checkpoint / "endpoint.npz"),
                    },
                )
            return event

        def checked_diagonal(current_problem: Any, current_state: Any) -> torch.Tensor:
            diagonal = original_diagonal(current_problem, current_state)
            if not hvp_validation:
                direction = torch.randn(
                    diagonal.shape,
                    device=diagonal.device,
                    dtype=diagonal.dtype,
                    generator=torch.Generator(device=diagonal.device).manual_seed(
                        20260923
                    ),
                )
                reference = current_problem.delegate.hess_prod(current_state, direction)
                actual = current_problem.hessian.apply(current_state, direction)
                relative = float(
                    torch.linalg.vector_norm(actual - reference)
                    / torch.linalg.vector_norm(reference)
                )
                if cfg.projection == "none":
                    assert relative < 1e-10, relative
                hvp_validation.append(
                    {
                        "stiffness_mpa": controller.current_stiffness,
                        "relative_error": relative,
                        "projection": cfg.projection,
                        "meaning": (
                            "exact sparse versus matrix-free HVP"
                            if cfg.projection == "none"
                            else "projected search matrix versus exact matrix-free HVP (diagnostic, not a gate)"
                        ),
                        "projection_last_refresh": dict(projection_stats["last"]),
                    }
                )
            return diagonal

        def traced_step(current_problem: Any, current_state: Any, **kwargs: Any) -> Any:
            nonlocal newton_steps
            current_state, receipt = original_step(
                current_problem, current_state, **kwargs
            )
            newton_steps += 1
            record(
                {
                    "kind": "newton",
                    "step": newton_steps,
                    "force": float(
                        torch.linalg.vector_norm(current_problem.grad(current_state))
                    ),
                    "energy": float(current_problem.fun(current_state)),
                    "accepted_step": receipt,
                }
            )
            return current_state, receipt

        controller.after_update = update_stiffness
        accelerated_solvers.safeguarded_newton_step = traced_step
        hybrid_first_solver.SparseNewtonProblem.hess_diag = checked_diagonal
        try:
            with timer.scope("forward"):
                if parent is None and not cfg.skip_pncg:
                    state, receipt = hybrid_first_solver.hybrid_first(
                        problem,
                        state,
                        atol=target_force,
                        max_step_norm=max_step,
                        linear_rtol=1e-3,
                        max_newton_steps=cfg.max_newton_steps,
                        newton_shift_policy=cfg.newton_shift_policy,
                        reuse_shift_force_ratio=cfg.reuse_shift_force_ratio,
                        callback=record,
                        adaptive_stiffness=controller,
                    )
                elif parent is None:
                    controller.initialize(problem, state)
                    record({"kind": "initial", "step": 0, "force": anchor_force})
                    sparse = hybrid_first_solver.SparseNewtonProblem(problem)
                    problem.newton_sparse_problem = sparse
                    state, newton = hybrid_first_solver.safeguarded_newton(
                        sparse,
                        state,
                        atol=target_force,
                        linear_rtol=1e-3,
                        linear_max_steps=1000,
                        max_steps=cfg.max_newton_steps,
                        max_step_norm=max_step,
                        preconditioner="diag",
                        shift_policy=cfg.newton_shift_policy,
                        reuse_shift_force_ratio=cfg.reuse_shift_force_ratio,
                        shift_scale_policy="signed_mean",
                        post_step=lambda current, step: controller.after_update(
                            sparse,
                            current,
                            phase="newton",
                            step=step,
                            hessian=sparse.hessian,
                        ),
                    )
                    receipt = {
                        "newton_steps": newton["steps"],
                        "trace": newton["trace"],
                        "hessian": dict(sparse.hessian.metadata),
                    }
                else:
                    controller.initialize(problem, state)
                    assert (
                        controller.bbox_diagonal == parent_stiffness["bbox_diagonal_m"]
                    )
                    previous_gap = parent_stiffness["observations"][-1][
                        "minimum_distance_squared"
                    ]
                    np.testing.assert_allclose(
                        controller.previous_min_distance_squared,
                        previous_gap,
                        rtol=1e-12,
                    )
                    controller.previous_min_distance_squared = previous_gap
                    record(
                        {
                            "kind": "initial",
                            "step": 0,
                            "force": continuation["restart_force"],
                            "energy": continuation["restart_energy"],
                        }
                    )
                    sparse = hybrid_first_solver.SparseNewtonProblem(problem)
                    problem.newton_sparse_problem = sparse
                    state, newton = hybrid_first_solver.safeguarded_newton(
                        sparse,
                        state,
                        atol=target_force,
                        linear_rtol=1e-3,
                        linear_max_steps=1000,
                        max_steps=cfg.max_newton_steps,
                        max_step_norm=max_step,
                        preconditioner="diag",
                        shift_policy=cfg.newton_shift_policy,
                        reuse_shift_force_ratio=cfg.reuse_shift_force_ratio,
                        shift_scale_policy="signed_mean",
                        post_step=lambda current, step: controller.after_update(
                            sparse,
                            current,
                            phase="newton",
                            step=step,
                            hessian=sparse.hessian,
                        ),
                    )
                    receipt = {
                        "newton_steps": newton["steps"],
                        "trace": newton["trace"],
                        "hessian": dict(sparse.hessian.metadata),
                    }
            result.update(success=True, forward_receipt=receipt)
        except ForwardConvergenceError as error:
            result.update(
                success=False,
                failure=str(error),
                failure_receipt=benchmark.jsonable(error.receipt),
            )
        finally:
            torch.cuda.synchronize()
            result["forward_wall_seconds"] = time.perf_counter() - started
            result["hvp_validation"] = hvp_validation
            result["projection"] = {
                key: projection_stats[key]
                for key in (
                    "mode",
                    "cell_batch",
                    "refreshes",
                    "projection_seconds",
                    "last",
                )
            }
            result["stiffness"] = controller.receipt()
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            benchmark.write_json(cfg.output_dir / "timing.json", timer.report())
            benchmark.write_json(
                cfg.output_dir / "stiffness.json", controller.receipt()
            )
            installed.uninstall()
            del controller.after_update
            accelerated_solvers.safeguarded_newton_step = original_step
            hybrid_first_solver.SparseNewtonProblem.hess_diag = original_diagonal

    displacement = state.u[: len(physics.points)].detach().clone()
    result["terminal_force"] = float(torch.linalg.vector_norm(problem.grad(state)))
    result["terminal_energy"] = float(problem.fun(state))
    result["terminal_stiffness_mpa"] = controller.current_stiffness
    result["geometry"] = benchmark.jsonable(physics.metrics(displacement))
    result["collision"] = benchmark.jsonable(
        audit_collision_state(physics, displacement, pose)
    )
    result["valid_forward"] = bool(
        result["success"]
        and result["terminal_force"] <= target_force
        and result["geometry"]["inverted_tetrahedra"] == 0
        and result["collision"]["state_feasible"]
    )
    result["compute_processes_after"] = _gpu_processes()
    np.savez_compressed(
        cfg.output_dir / "endpoint.npz",
        displacement_m=displacement.detach().cpu().numpy(),
    )
    benchmark.write_json(
        cfg.output_dir / "summary.json",
        {"protocol": protocol, "result": benchmark.jsonable(result)},
    )
    benchmark.write_json(
        cfg.output_dir / "status.json",
        {
            "running": False,
            "success": result["success"],
            "valid_forward": result["valid_forward"],
        },
    )
    guard_receipt["completed_without_inverse_calls"] = True
    benchmark.write_json(cfg.output_dir / "forward-only-guard.json", guard_receipt)
    cherries.log_metrics(
        {
            "forward/success": float(result["success"]),
            "forward/valid": float(result["valid_forward"]),
            "forward/seconds": result["forward_wall_seconds"],
            "forward/terminal_force": result["terminal_force"],
            "forward/stiffness_mpa": result["terminal_stiffness_mpa"],
        }
    )
    for name in (
        "protocol.json",
        "summary.json",
        "timing.json",
        "stiffness.json",
        "trace.jsonl",
        "endpoint.npz",
        "active-strain-fields.npz",
        "active-strain-check.json",
        "active-strain-mapping.json",
        "active-strain-source-provenance.json",
        "forward-only-guard.json",
    ):
        cherries.log_output(cfg.output_dir / name)


def forbidden_inverse(*_args, **_kwargs):
    message = "Differentiable or adjoint solve invoked in forward-only run"
    raise AssertionError(message)


if __name__ == "__main__":
    with ExitStack() as guard:
        for name in ("forward", "step", "adjoint_solve", "receipt"):
            guard.enter_context(
                patch.object(DifferentiableForward, name, forbidden_inverse)
            )
        cherries.main(main, profile=benchmark.ProfilePerformance)
