# ruff: noqa: C901, E402, PLR0912, PLR0915
"""Fresh forward solves of one frozen Stage 3 Smile tensor in two setups."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS = ROOT / "exp/2026/09/21/stress-activation-loss"
NEUTRAL = ROOT / "exp/2026/09/23/new-neutral"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance"
sys.path[:0] = [
    str(GROUP / "src"),
    str(STRESS / "src"),
    str(NEUTRAL / "src"),
    str(SOLVERS / "src"),
    str(JOINT / "src"),
]

from accelerated_solvers import CachedProblem, safeguarded_newton
from experiment import Profile
from joint_equilibrium import ForwardConvergenceError
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from pncg_first import run_pncg_phase
from stress_physics import FacePhysics, configure, strain_to_activation_inv

LOG = logging.getLogger(__name__)
SOURCE = STRESS / "data/51-visualization-checkpoints-002/l2-normal"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"


class Config(cherries.BaseConfig):
    branch: Literal["inverse-setup", "new-setup"] = "inverse-setup"
    output: Path = Path("10-inverse-setup")
    activation: Path = SOURCE / "l2-normal-rankone_fixed/last.npz"
    mesh: Path = SOURCE / "mesh.npz"
    fixture: Path = FIXTURE
    neutral_dir: Path = NEUTRAL / "data/forward-isfixed-001"
    reference_dir: Path = NEUTRAL / "data/reference-clearance-002"
    force_atol: float = 1e-10
    max_newton_steps: int = 5000
    linear_max_steps: int = 10000
    wall_seconds: float = 1200
    ipc_threads: int = 4
    resume: Path | None = None
    first_phase: Literal["newton", "pncg"] = "newton"
    initial_shift_scale: float = 0.0


def record(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ProjectedContactSearch(HessianProblem):
    """PSD IPC search curvature with the original energy and residual."""

    def prepare_contact(self, state: Any) -> None:
        contact = self.model.collision
        assert contact is not None
        assert state.collision is not None
        if state.collision.hess is None:
            positions = (contact.vertices + state.u[contact.indices]).numpy(force=True)
            state.collision.hess = contact.potential.hessian(
                collisions=state.collision.collisions,
                mesh=contact.collision_mesh,
                X=positions,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
            )

    def hess_prod(self, state: Any, direction: torch.Tensor):
        self.prepare_contact(state)
        return super().hess_prod(state, direction)


@torch.no_grad()
def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    inputs = {
        "activation": record(cfg.activation),
        "mesh": record(cfg.mesh),
        "inverse_protocol": record(SOURCE / "protocol.json"),
    }
    with np.load(cfg.activation) as archive:
        assert str(archive["mode"]) == "rankone_fixed"
        assert str(archive["activation_model"]) == "strain"
        activation = archive["S"].copy()
        historical_valid = bool(archive["solver_valid"])
    with np.load(cfg.mesh) as archive:
        original_points = archive["rest_points"].copy()
        tets = archive["tets"].copy()
        active_ids = archive["active_ids"].copy()
    assert activation.shape == (len(active_ids), 3, 3)
    np.testing.assert_allclose(
        activation, activation.transpose(0, 2, 1), atol=1e-12, rtol=0
    )
    assert np.linalg.eigvalsh(activation).min() > -1e-12
    configure()
    ipctk.set_num_threads(cfg.ipc_threads)
    if cfg.branch == "inverse-setup":
        for name in ("volume.vtu", "skin.vtp"):
            inputs[name] = record(cfg.fixture / name)
        physics = FacePhysics(
            cfg.fixture, activation_model="strain", atol=cfg.force_atol
        )
        model = physics.forward.model
        points = physics.points
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(physics.tets, tets)
        np.testing.assert_array_equal(physics.ids, active_ids)
        materials = physics.materials
        seed = torch.zeros((model.n_points, 3))
        material_spec = physics.material_spec
        initialization = "zero displacement on the exact historical inverse reference; full S applied once"
        expected_skin = False
    else:
        from neutral_active_strain import install_active_strain
        from reference_rebase import build_rebased_physics

        for name in (
            "active-strain-fields.npz",
            "endpoint.npz",
            "protocol.json",
            "stiffness.json",
            "rebased-reference-volume.vtu",
            "rebased-reference-skin.vtp",
        ):
            inputs[f"neutral/{name}"] = record(cfg.neutral_dir / name)
        inputs["reference/receipt"] = record(cfg.reference_dir / "receipt.json")
        inputs["reference/coordinates"] = record(
            cfg.reference_dir / "reference-clearance.npz"
        )
        physics, _ = build_rebased_physics(cfg.reference_dir)
        model = physics.runtime.forward.model
        points = np.asarray(physics.points).copy()
        np.testing.assert_array_equal(np.asarray(physics.tets), tets)
        np.testing.assert_array_equal(
            np.flatnonzero(physics.mesh.cell_data["ActivationMask"]), active_ids
        )
        # CUDA's batched 2x2 eigh requests a quadratic-size temporary workspace.
        # Use the same CUDA operator in small batches, then check the saved B
        # matrices byte for byte below. This changes no material coefficients.
        original_eigh = torch.linalg.eigh

        def bounded_eigh(tensor: torch.Tensor):
            assert tensor.shape[1:] == (2, 2)
            results = [original_eigh(chunk) for chunk in tensor.split(1024)]
            return tuple(
                torch.cat([result[index] for result in results]) for index in (0, 1)
            )

        with patch.object(torch.linalg, "eigh", bounded_eigh):
            materials, strain_receipt, _ = install_active_strain(model)
        strain_receipt["construction_eigh_batch_size"] = 1024
        with np.load(cfg.neutral_dir / "active-strain-fields.npz") as archive:
            np.testing.assert_array_equal(
                materials["skin"]["activation_inv"].numpy(force=True),
                archive["skin_activation_inverse"],
            )
            np.testing.assert_array_equal(
                materials["skin"]["mu"].numpy(force=True), archive["skin_mu_mpa"]
            )
            np.testing.assert_array_equal(
                materials["skin"]["thickness"].numpy(force=True),
                archive["skin_thickness_m"],
            )
        with np.load(cfg.neutral_dir / "endpoint.npz") as archive:
            seed = torch.as_tensor(archive["displacement_m"].copy())
        assert seed.shape == (len(points), 3)
        seed = physics.full_skull.extend_seed(seed, torch.zeros(6))
        assert seed.shape == (model.n_points, 3)
        model.dof_map.fixed_values = physics.boundary(torch.zeros(6)).clone()
        torch.testing.assert_close(
            seed.flatten()[model.dof_map.fixed_indices],
            model.dof_map.fixed_values,
            atol=1e-12,
            rtol=0,
        )
        contact = model.collision
        assert contact is not None
        potential = contact.potential
        stiffness = float(
            json.loads((cfg.neutral_dir / "stiffness.json").read_text())[
                "final_stiffness"
            ]
        )
        contact.potential = ipctk.BarrierPotential(
            type(potential.barrier)(),
            potential.dhat,
            stiffness,
            contact.use_physical_barrier,
        )
        material_spec = {
            "bulk_E_MPa": {"fat": 0.0112, "muscle": 0.012, "aponeurosis": 1.693},
            "bulk_nu": 0.49,
            "skin_energy": "StableNeoHookeanActiveMembrane, exact plane stress and physical-J regularization",
            "skin_prestrain": strain_receipt,
            "contact_enabled": True,
            "collision_policy": "soft FEM against complete cranium, mandible and eyes; soft-soft and rigid-rigid disabled",
            "contact_stiffness_MPa": stiffness,
            "dhat_m": float(potential.dhat),
            "skin_triangles": int(physics.skin.n_cells),
            "skin_mu_MPa_min": float(materials["skin"]["mu"].min()),
            "skin_mu_MPa_max": float(materials["skin"]["mu"].max()),
            "skin_thickness_m": float(materials["skin"]["thickness"][0]),
        }
        initialization = "saved corrected loaded neutral on the repaired constitutive reference; full historical S applied once, jaw held at zero pose"
        expected_skin = True
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    full_fixed = np.r_[is_fixed, np.ones(model.n_points - len(is_fixed), dtype=bool)]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.numpy(force=True),
        np.flatnonzero(np.repeat(full_fixed, 3)),
    )
    registry = model.warp_model.__wrapped__.potentials
    assert ("skin" in registry) == expected_skin
    assert (model.collision is not None) == expected_skin
    packed = strain_to_activation_inv(torch.as_tensor(activation))
    assert len(materials["muscle"]["activation_inv"]) == len(tets)
    materials["muscle"]["activation_inv"] = torch.zeros((len(tets), 6)).index_copy(
        0, torch.as_tensor(active_ids), packed
    )
    model.set_materials(materials)
    if cfg.resume is not None:
        previous = json.loads((cfg.resume / "summary.json").read_text())
        assert previous["config"]["branch"] == cfg.branch
        assert (
            previous["activation_array_sha256"]
            == hashlib.sha256(activation.tobytes()).hexdigest()
        )
        assert previous["inputs"] == inputs
        with np.load(cfg.resume / "final.npz") as archive:
            np.testing.assert_array_equal(archive["S"], activation)
            np.testing.assert_array_equal(archive["rest_points"], points)
            seed = torch.as_tensor(archive["u"].copy())
        initialization += "; resume from preceding numerical endpoint"
    state = model.State(
        u=seed.clone(),
        collision=None if model.collision is None else model.collision.state_at(seed),
    )
    if model.collision is not None:
        initial_contact = model.collision.diagnostics(state.collision, state.u)
        assert initial_contact["contact_numerically_valid"], initial_contact
    else:
        initial_contact = None
    summary = {
        "schema": "smile-fixed-activation-forward-setup-v1",
        "status": "running",
        "config": cfg.model_dump(mode="json"),
        "inputs": inputs,
        "material_spec": material_spec,
        "activation_array_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
        "historical_checkpoint_solver_valid": historical_valid,
        "activation_policy": "exact fixed Stage 3 S; no activation optimization or scaling; B=I+S",
        "initialization": initialization,
        "mesh_points": len(points),
        "cell_count": len(tets),
        "active_cells": len(active_ids),
        "fixed_original_nodes": int(is_fixed.sum()),
        "appended_fixed_nodes": model.n_points - len(is_fixed),
        "reference_max_change_m": float(
            np.max(np.linalg.norm(points - original_points, axis=1))
        ),
        "reference_changed_nodes": int(
            np.count_nonzero(np.any(points != original_points, axis=1))
        ),
        "force_gate_N": cfg.force_atol * 1e6,
        "energy_reference": "Each setup uses its own constitutive energy offset; vertical separation across setups is not an energy difference under one common model",
        "energy_units": "J = MPa m3 * 1e6",
        "inversion_definition": "det(F)<=0 over all original tetrahedra; denominator=1,146,517 cells, including fully fixed cells",
        "initial_contact": initial_contact,
        "solver_policy": "Safeguarded Newton with exact bulk and PSD contact search curvature, Armijo and CCD; fixed activation and original physical energy/force; no inversion rejection so the requested inversion curve remains observable",
    }
    write(out / "summary.json", summary)
    # Snapshot the loaded project source before numerical iteration starts.
    sources = {}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if (
            path.suffix != ".py"
            or not path.is_relative_to(ROOT)
            or ".venv" in path.parts
            or not path.is_file()
        ):
            continue
        relative = path.relative_to(ROOT)
        destination = out / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(relative)] = record(path)
    write(out / "source-manifest.json", sources)
    rest = torch.as_tensor(points)
    cells = torch.as_tensor(tets)
    dm = (rest[cells[:, 1:]] - rest[cells[:, :1]]).transpose(-1, -2)
    dm_inv = torch.linalg.inv(dm)
    assert bool((torch.linalg.det(dm) > 0).all())
    problem = (
        FeasibleExpressionProblem(model=model, collision_step_safety=0.95)
        if model.collision is not None
        else ForwardProblem(model=model)
    )
    start = time.perf_counter()
    deadline = start + cfg.wall_seconds
    rows = []
    fields = [
        "iteration",
        "phase",
        "elapsed_seconds",
        "energy_j",
        "force_n",
        "inverted_cells",
        "cell_count",
        "inverted_percent",
        "minimum_detf",
        "contact_energy_j",
    ]
    trace_path = cherries.output(cfg.output / "trace.csv")
    stream = trace_path.open("w", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()

    def observe(
        current: Any,
        phase: str,
        *,
        energy: float | None = None,
        force: float | None = None,
    ) -> None:
        displacement = current.u[: len(points)]
        deformed = rest + displacement
        ds = (deformed[cells[:, 1:]] - deformed[cells[:, :1]]).transpose(-1, -2)
        jacobians = torch.linalg.det(ds @ dm_inv)
        count = int(torch.count_nonzero(jacobians <= 0))
        energy = float(model.fun(current)) if energy is None else energy
        force = (
            float(torch.linalg.vector_norm(problem.grad(current)))
            if force is None
            else force
        )
        assert np.isfinite(energy)
        assert np.isfinite(force)
        assert bool(torch.isfinite(jacobians).all())
        barrier = (
            0.0
            if model.collision is None
            else float(model.collision.fun(current.collision, current.u))
        )
        row = {
            "iteration": len(rows),
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - start,
            "energy_j": energy * 1e6,
            "force_n": force * 1e6,
            "inverted_cells": count,
            "cell_count": len(tets),
            "inverted_percent": 100 * count / len(tets),
            "minimum_detf": float(jacobians.min()),
            "contact_energy_j": barrier * 1e6,
        }
        rows.append(row)
        writer.writerow(row)
        stream.flush()
        write(out / "progress.json", row)
        if row["iteration"] % 10 == 0:
            checkpoint = out / "latest.tmp.npz"
            np.savez_compressed(
                checkpoint,
                u=current.u.numpy(force=True),
                S=activation,
                rest_points=points,
                tets=tets,
                active_ids=active_ids,
            )
            checkpoint.replace(out / "latest.npz")
        if row["iteration"] % 25 == 0:
            LOG.info(
                "%s %s step %d energy %.6g J force %.6g N inverted %d (%.6g%%)",
                cfg.branch,
                phase,
                row["iteration"],
                row["energy_j"],
                row["force_n"],
                count,
                row["inverted_percent"],
            )
            cherries.set_step(row["iteration"])
            cherries.log_metrics(
                {
                    key: row[key]
                    for key in (
                        "energy_j",
                        "force_n",
                        "inverted_percent",
                        "minimum_detf",
                    )
                }
            )

    observe(state, "initial")
    cached = CachedProblem(
        problem,
        cache_gradient=True,
        exact_curvature=False,
        wall_seconds=cfg.wall_seconds,
    )
    max_step = 0.5 * mean_rest_edge_length(model, points)
    failure = None
    try:

        def pncg_observe(row: dict):
            if row["kind"] != "initial":
                observe(state, "pncg", energy=row["energy"], force=row["force"])

        if cfg.first_phase == "pncg":
            state, pncg_receipt = run_pncg_phase(
                cached,
                state,
                atol=cfg.force_atol,
                max_step_norm=max_step,
                callback=pncg_observe,
            )
            summary["pncg"] = pncg_receipt
        hessian = (
            ProjectedContactSearch(problem, "gpu_contact")
            if model.collision is not None
            else HessianProblem(problem, "matrix_free")
        )
        cached_newton = CachedProblem(
            hessian,
            cache_gradient=True,
            exact_curvature=True,
            wall_seconds=max(1e-3, deadline - time.perf_counter()),
        )
        state, newton_receipt = safeguarded_newton(
            cached_newton,
            state,
            atol=cfg.force_atol,
            linear_rtol=1e-3,
            linear_max_steps=cfg.linear_max_steps,
            max_steps=cfg.max_newton_steps,
            max_step_norm=max_step,
            initial_shift_scale=cfg.initial_shift_scale,
            shift_policy="reuse",
            reuse_shift_force_ratio=0,
            shift_scale_policy="mean_abs",
            post_step=lambda current, _step: observe(current, "newton"),
        )
        summary["newton"] = newton_receipt
        summary["hessian"] = hessian.report()
    except ForwardConvergenceError as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": error.receipt,
        }
        LOG.exception("Forward solve stopped")
    except KeyboardInterrupt:
        failure = {
            "type": "KeyboardInterrupt",
            "message": "Numerical process interrupted; saved last state",
            "receipt": None,
        }
        LOG.warning("Numerical process interrupted; exporting endpoint")
    finally:
        observe(state, "final")
        stream.close()
    final = rows[-1]
    final_contact = (
        None
        if model.collision is None
        else model.collision.diagnostics(state.collision, state.u)
    )
    torch.testing.assert_close(
        state.u.flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
        atol=1e-12,
        rtol=0,
    )
    if model.collision is not None:
        from smile_collision import audit_collision_state

        summary["independent_contact_geometry"] = audit_collision_state(
            physics, state.u[: len(points)], torch.zeros(6)
        )
    converged = final["force_n"] <= cfg.force_atol * 1e6
    summary.update(
        status="force_converged" if converged else "not_force_converged",
        final=final,
        failure=failure,
        final_contact=final_contact,
        elapsed_seconds=time.perf_counter() - start,
        force_converged=converged,
        orientation_valid=final["inverted_cells"] == 0,
        physical_valid=converged
        and final["inverted_cells"] == 0
        and (
            final_contact is None
            or summary["independent_contact_geometry"]["state_feasible"]
        ),
    )
    np.savez_compressed(
        cherries.output(cfg.output / "final.npz"),
        u=state.u.numpy(force=True),
        S=activation,
        rest_points=points,
        tets=tets,
        active_ids=active_ids,
    )
    write(out / "summary.json", summary)
    cherries.log_output(trace_path)
    LOG.info(
        "Final %s: %s; force %.6g N, inverted %d/%d (%.6g%%)",
        cfg.branch,
        summary["status"],
        final["force_n"],
        final["inverted_cells"],
        len(tets),
        final["inverted_percent"],
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
