"""Matched no-skin equilibrium with explicit tensor-stress or Raw6 actuation.

Adapted from the frozen historical diagnosis physics; passive materials,
fixation, target and forward/adjoint tolerances are retained.
"""

# ruff: noqa: ANN001, ANN204, C408, EM101, EM102, PLR0915, PT018, TRY003

from __future__ import annotations

import contextlib
import io
import logging
import math
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import pyvista as pv
import torch
import warp as wp
from tensor_active import StableNeoHookeanTensorActive

from liblaf.apple.common import FRACTION, LAMBDA, MU, NU, E
from liblaf.apple.forward import Forward, ModelBuilder
from liblaf.apple.inverse import DifferentiableForward
from liblaf.apple.solvers.linalg import FallbackSolver
from liblaf.apple.solvers.linalg.base import BaseProblem, Problem, Result
from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes
from liblaf.apple.solvers.optim import Pncg
from liblaf.apple.warp.fem import (
    Koiter,
    NeoHookean,
    StableNeoHookean,
    StableNeoHookeanActive,
)

ROOT = Path(__file__).resolve().parents[6]


class ForwardConvergenceError(RuntimeError):
    """A trial did not meet the declared inner equilibrium tolerance."""

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None):
        self.receipt = receipt
        super().__init__(message)


class SuccessPreferredFallbackSolver(FallbackSolver):
    """Preserve the June adjoint fallback and successful-solver selection."""

    @override
    def compute(self, problem: BaseProblem, state: Any) -> Result:
        typed_problem = cast("Problem", problem)
        absolute_residuals = []
        relative_residuals = []
        success_index: int | None = None
        for index, solver in enumerate(self.solvers):
            solution = solver.solve(typed_problem, state.init_params)
            state.solutions.append(solution)
            absolute_residual = torch.linalg.vector_norm(
                typed_problem.matvec(solution.state.params) - typed_problem.b
            )
            relative_residual = absolute_residual / torch.linalg.vector_norm(
                typed_problem.b
            )
            absolute_residuals.append(absolute_residual)
            relative_residuals.append(relative_residual)
            if solution.success:
                success_index = index
                break
        state.absolute_residuals = torch.as_tensor(absolute_residuals)
        state.relative_residuals = torch.as_tensor(relative_residuals)
        state.best_index = (
            torch.argmin(state.absolute_residuals)
            if success_index is None
            else torch.as_tensor(success_index, dtype=torch.int32)
        )
        return state.result


def line_search_receipt(
    line_search: Pncg.LineSearch, state: Pncg.LineSearchState
) -> dict[str, Any]:
    receipt = {
        "implementation": type(line_search).__name__,
        "max_steps": int(line_search.max_steps),
        "status": "not_run" if state.alpha is None else "accepted",
        "ok": None if state.alpha is None else bool(state.ok),
        "step": int(state.step),
        "alpha": None,
        "f0": None,
        "f_alpha": float(state.f_alpha.detach().cpu()),
    }
    if state.alpha is not None:
        receipt["alpha"] = float(state.alpha.detach().cpu())
        receipt["f0"] = float(state.f0.detach().cpu())
        if not state.ok:
            receipt["status"] = "exhausted"
    return receipt


class StrictLineSearch(Pncg.LineSearch):
    """Reject an exhausted or nonfinite Armijo trial before PNCG advances."""

    def __call__(
        self,
        state: Pncg.LineSearchState,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__call__(state, *args, **kwargs)
        diagnostics = line_search_receipt(self, state)
        if not all(
            math.isfinite(diagnostics[name]) for name in ("alpha", "f0", "f_alpha")
        ):
            raise ForwardConvergenceError(
                f"nonfinite Armijo diagnostics: {diagnostics}", receipt=diagnostics
            )
        if not state.ok:
            raise ForwardConvergenceError(
                f"Armijo search exhausted: {diagnostics}", receipt=diagnostics
            )


class StrictPncg(Pncg):
    """Attach the last computed gradient to strict line-search failures."""

    @override
    def step(self, problem, model_state, opt_state):
        try:
            return super().step(problem, model_state, opt_state)
        except ForwardConvergenceError as error:
            receipt = dict(error.receipt or {})
            receipt.update(
                optimizer_step=int(opt_state.step),
                grad_norm=float(
                    torch.linalg.vector_norm(opt_state.grad).detach().cpu()
                ),
            )
            raise ForwardConvergenceError(str(error), receipt=receipt) from error


def configure():
    assert torch.cuda.is_available()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    wp.config.mode = "release"
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)


def set_material(mesh, young, nu, fraction, *, model):
    mu = young / (2 * (1 + nu))
    classical_lambda = young * nu / ((1 + nu) * (1 - 2 * nu))
    if model not in {"stable", "neo"}:
        raise ValueError(f"unknown volume material model: {model!r}")
    # Exact June convention: both Stable and logarithmic Neo received the
    # classical Lamé lambda.  Do not apply the later Stable lambda correction.
    lambda_code = classical_lambda
    mesh.cell_data[E.vtk] = np.full(mesh.n_cells, young)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_code)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu)
    mesh.cell_data[FRACTION.vtk] = fraction


def active_graph(points, tets, ids, region, fraction):
    """Unique shared-face conductances; never smooth between muscle labels."""
    active = tets[ids]
    face_pattern = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.sort(active[:, face_pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(ids)), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(pair) == 1), "Nonmanifold active tetrahedral face"
    i, j = owner[pair], owner[pair + 1]
    same = region[i] == region[j]
    i, j, face = i[same], j[same], faces[pair[same]]
    xyz = points[face]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    centers = points[active].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    assert np.all(distance > 0)
    # Harmonic tissue fraction makes the finite-volume prior weaken in mixed cells.
    frac = fraction[ids]
    weight = area / distance * (2 * frac[i] * frac[j] / (frac[i] + frac[j]))
    return i, j, weight


class FacePhysics:
    def __init__(
        self,
        fixture,
        *,
        activation_model="tensor",
        skin_factor=0.0,
        fat_factor=1.0,
        muscle_factor=1.0,
        rtol=5e-4,
        atol=1e-10,
        adjoint_rtol=5e-4,
        soft_nu=0.49,
        fat_nu=0.49,
        fat_model="stable",
        skin_nu=0.46,
        target_name="Smile",
        target_scale=1.0,
        line_search_max_steps=30,
    ):
        if skin_factor != 0.0:
            raise ValueError("matched historical comparison requires zero skin energy")
        if not (
            fat_factor == muscle_factor == 1.0
            and soft_nu == 0.49
            and fat_nu == 0.49
            and fat_model == "stable"
            and rtol == adjoint_rtol == 5e-4
            and atol == 1e-10
        ):
            raise ValueError("matched historical material and solver constants changed")
        assert activation_model in {"tensor", "raw6"}
        self.activation_model = activation_model
        self.mesh = pv.read(Path(fixture) / "volume.vtu")
        self.skin = pv.read(Path(fixture) / "skin.vtp")
        self.points = np.asarray(self.mesh.points).copy()
        self.tets = np.asarray(self.mesh.cells).reshape(-1, 5)[:, 1:].copy()
        self.ids = np.flatnonzero(self.mesh.cell_data["ActivationMask"])
        self.region = np.asarray(self.mesh.cell_data["ActivationControlId"], dtype=int)[
            self.ids
        ]
        assert np.array_equal(np.unique(self.region), np.arange(self.region.max() + 1))
        target = np.asarray(self.mesh.point_data[target_name]) * target_scale
        self.target_name = target_name
        self.top = np.flatnonzero(
            np.asarray(self.mesh.point_data["IsFace"], bool)
            & np.isfinite(target).all(axis=1)
        )
        self.target = np.zeros_like(self.points)
        self.target[self.top] = target[self.top]
        self.id_t = torch.as_tensor(self.ids)
        self.top_t = torch.as_tensor(self.top)
        self.region_t = torch.as_tensor(self.region)
        self.n_regions = int(self.region.max() + 1)
        dm = np.transpose(
            self.points[self.tets[:, 1:]] - self.points[self.tets[:, :1]], (0, 2, 1)
        )
        self.dm_inv = np.linalg.inv(dm)
        self.volumes_all = np.linalg.det(dm) / 6
        assert np.all(self.volumes_all > 0)
        fraction = np.asarray(self.mesh.cell_data["MuscleFraction"])
        self.volumes = self.volumes_all[self.ids] * fraction[self.ids]
        self.graph = active_graph(
            self.points, self.tets, self.ids, self.region, fraction
        )
        self.weights = self.surface_weights()
        self.weights_t = torch.as_tensor(self.weights)
        self.D = float(
            np.sqrt(np.sum(self.weights[:, None] * self.target[self.top] ** 2))
        )
        self.region_mass = np.bincount(
            self.region, weights=self.volumes, minlength=self.n_regions
        )
        self.region_mass /= self.region_mass.sum()
        if fat_model not in {"stable", "neo"}:
            raise ValueError("fat_model must be 'stable' or 'neo'")
        actual_fat_nu = soft_nu if fat_nu is None else fat_nu
        for label, value in (
            ("soft_nu", soft_nu),
            ("fat_nu", actual_fat_nu),
            ("skin_nu", skin_nu),
        ):
            if not 0.0 < value < 0.5:
                raise ValueError(f"{label} must lie strictly between zero and 0.5")
        builder = ModelBuilder()
        builder.add_vertices(self.mesh)
        builder.add_fixed(self.mesh)
        for name, young, nu, cls, model in (
            ("aponeurosis", 0.1, 0.35, StableNeoHookean, "stable"),
            (
                "fat",
                0.003 * fat_factor,
                actual_fat_nu,
                NeoHookean if fat_model == "neo" else StableNeoHookean,
                fat_model,
            ),
            (
                "muscle",
                0.03 * muscle_factor,
                soft_nu,
                StableNeoHookeanTensorActive
                if activation_model == "tensor"
                else StableNeoHookeanActive,
                "stable",
            ),
        ):
            set_material(
                self.mesh,
                young,
                nu,
                np.asarray(self.mesh.cell_data[name.title() + "Fraction"]),
                model=model,
            )
            builder.add_potential(cls.from_pyvista(self.mesh, name=name))
        if skin_factor:
            # Fixture has plane-stress Lamé coefficients and zero activation.
            young = 0.2 * skin_factor
            self.skin.cell_data[LAMBDA.vtk] = np.full(
                self.skin.n_cells, young * skin_nu / (1 - skin_nu**2)
            )
            self.skin.cell_data[MU.vtk] = np.full(
                self.skin.n_cells, young / (2 * (1 + skin_nu))
            )
            builder.add_potential(
                Koiter.from_pyvista(self.skin, name="skin", thickness=0.001)
            )
        self.forward = Forward(builder.finalize())
        self.forward_tolerance = {
            "max_steps": 5000,
            "rtol": float(rtol),
            "atol": float(atol),
            "line_search_max_steps": int(line_search_max_steps),
        }
        if line_search_max_steps <= 0:
            raise ValueError("line_search_max_steps must be positive")
        optimizer = self.forward.default_optimizer(
            max_steps=self.forward_tolerance["max_steps"],
            rtol=self.forward_tolerance["rtol"],
            atol=self.forward_tolerance["atol"],
        )
        # Preserve the June inner PNCG implementation and its default line search.
        self.forward.optimizer = optimizer
        self.forward_tolerance["line_search_max_steps"] = int(
            optimizer.line_search.max_steps
        )
        self.diff = DifferentiableForward(self.forward)
        self.diff.adjoint_solver = SuccessPreferredFallbackSolver(
            [
                CupyCG(maxiter=10_000, rtol=adjoint_rtol, atol=0.0),
                CupyMinRes(maxiter=10_000, tol=adjoint_rtol),
            ]
        )
        self.materials = self.forward.model.get_materials()
        self.solve_count = 0
        fat_young = 0.003 * fat_factor
        fat_mu = fat_young / (2 * (1 + actual_fat_nu))
        fat_lambda_classical = (
            fat_young * actual_fat_nu / ((1 + actual_fat_nu) * (1 - 2 * actual_fat_nu))
        )
        muscle_young = 0.03 * muscle_factor
        muscle_mu = muscle_young / (2 * (1 + soft_nu))
        muscle_lambda_classical = (
            muscle_young * soft_nu / ((1 + soft_nu) * (1 - 2 * soft_nu))
        )
        aponeurosis_young = 0.1
        aponeurosis_nu = 0.35
        aponeurosis_mu = aponeurosis_young / (2 * (1 + aponeurosis_nu))
        aponeurosis_lambda_classical = (
            aponeurosis_young
            * aponeurosis_nu
            / ((1 + aponeurosis_nu) * (1 - 2 * aponeurosis_nu))
        )
        self.material_spec = dict(
            fat_E_MPa=0.003 * fat_factor,
            fat_model=fat_model,
            fat_nu=actual_fat_nu,
            fat_mu_code_MPa=fat_mu,
            fat_lambda_classical_MPa=fat_lambda_classical,
            fat_lambda_code_MPa=fat_lambda_classical,
            muscle_E_MPa=0.03 * muscle_factor,
            muscle_model="stable-tensor-active"
            if activation_model == "tensor"
            else "stable-active-strain",
            muscle_nu=soft_nu,
            muscle_mu_code_MPa=muscle_mu,
            muscle_lambda_classical_MPa=muscle_lambda_classical,
            muscle_lambda_code_MPa=muscle_lambda_classical,
            aponeurosis_E_MPa=aponeurosis_young,
            aponeurosis_model="stable",
            aponeurosis_nu=aponeurosis_nu,
            aponeurosis_mu_code_MPa=aponeurosis_mu,
            aponeurosis_lambda_classical_MPa=aponeurosis_lambda_classical,
            aponeurosis_lambda_code_MPa=aponeurosis_lambda_classical,
            skin_E_MPa=0.2 * skin_factor,
            skin_thickness_m=0.001,
            skin_plane_stress=True,
            skin_prestrain=0.0,
            skin_nu=skin_nu,
            volume_lame_convention=(
                "Exact June convention: Stable and active Stable receive classical "
                "lambda without the later plus-mu correction"
            ),
            contact_enabled=False,
        )

    def surface_weights(self):
        # GlobalPointId is guaranteed to match the volume by fixture validation.
        original = np.asarray(self.skin.point_data["GlobalPointId"], dtype=int)
        tri = original[np.asarray(self.skin.faces).reshape(-1, 4)[:, 1:]]
        xyz = self.points[tri]
        area = (
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            / 2
        )
        w = np.zeros(len(self.points))
        np.add.at(w, tri.ravel(), np.repeat(area / 3, 3))
        w = w[self.top]
        assert np.all(w >= 0) and w.sum() > 0
        return w / w.sum()

    def solve(self, packed_active, seed):
        self.forward.state.u = self.forward.state.u.detach().clone()
        with torch.no_grad():
            self.forward.model.update(self.forward.state, torch.as_tensor(seed))
        if self.activation_model == "tensor":
            assert packed_active.shape == (len(self.ids), 3, 3)
            self.materials["muscle"]["active_stress"] = torch.zeros(
                (self.mesh.n_cells, 3, 3)
            ).index_copy(0, self.id_t, packed_active)
        else:
            assert packed_active.shape == (len(self.ids), 6)
            self.materials["muscle"]["activation_inv"] = torch.zeros(
                (self.mesh.n_cells, 6)
            ).index_copy(0, self.id_t, packed_active)
        solver_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(solver_stdout):
                u = self.diff.forward(self.materials).clone()
        except ForwardConvergenceError as error:
            self.solve_count += 1
            self.last_forward = dict(
                success=False,
                steps=error.receipt.get("optimizer_step") if error.receipt else None,
                grad_norm=error.receipt.get("grad_norm") if error.receipt else None,
                result="line_search_failure",
                tolerance=dict(self.forward_tolerance),
                line_search=error.receipt,
                stdout=solver_stdout.getvalue(),
            )
            raise ForwardConvergenceError(
                f"Forward failed: {self.last_forward}", receipt=self.last_forward
            ) from error
        self.solve_count += 1
        solution = self.diff.last_solution
        assert solution is not None
        state = solution.state.convergence_state
        line_search = solution.state.line_search_state
        self.last_forward = dict(
            success=bool(solution.success),
            steps=int(state.step),
            grad_norm=float(state.grad_norm.detach().cpu()),
            result=str(solution.result),
            tolerance=dict(self.forward_tolerance),
            line_search=line_search_receipt(
                self.forward.optimizer.line_search,
                line_search,
            ),
            stdout=solver_stdout.getvalue(),
        )
        # June recorded isolated nonconverged inner solves and let the outer
        # Adam loop continue. The caller decides when a visible failure cluster
        # requires restoring the best valid state and resetting Adam moments.
        assert torch.isfinite(u).all()
        return u

    def check_adjoint(self):
        sol = self.diff.last_adjoint_solution
        assert sol is not None and sol.success, f"Adjoint failed: {sol}"
        return dict(success=bool(sol.success), result=str(sol.result))

    def detf(self, u):
        x = self.points + u
        ds = np.transpose(x[self.tets[:, 1:]] - x[self.tets[:, :1]], (0, 2, 1))
        return np.linalg.det(ds @ self.dm_inv)

    def save_mesh(self, path, u, ainv=None, active_stress=None):
        mesh = self.mesh.copy()
        # Strip unrelated expression arrays to keep exact checkpoint files small.
        for key in list(mesh.point_data):
            if key not in {
                "IsFace",
                "IsFixed",
                "IsLip",
                "FixedMask",
                "FixedValue",
                "CutBoundary",
                "ArtificialCutIncident",
                "CutBoundaryAddedFixed",
                "GlobalPointId",
            }:
                del mesh.point_data[key]
        for key in list(mesh.cell_data):
            if key not in {
                "MuscleId",
                "MuscleFraction",
                "FatFraction",
                "AponeurosisFraction",
                "ActivationMask",
                "ActivationControlId",
            }:
                del mesh.cell_data[key]
        mesh.points = self.points + u
        mesh.point_data["RestPosition"] = self.points
        mesh.point_data["Displacement"] = u
        mesh.point_data["TargetDisplacement"] = self.target
        det = self.detf(u)
        mesh.cell_data["DetF"] = det
        if ainv is not None:
            full = np.broadcast_to(np.eye(3), (len(self.tets), 3, 3)).copy()
            full[self.ids] = ainv
            mesh.cell_data["ActivationInverseMatrix"] = full.reshape(-1, 9)
            mesh.cell_data["DetAinv"] = np.linalg.det(full)
        if active_stress is not None:
            full_q = np.zeros((len(self.tets), 3, 3))
            full_q[self.ids] = active_stress
            mesh.cell_data["ActiveStressMatrixMPa"] = full_q.reshape(-1, 9)
            mesh.cell_data["ActiveStressTraceMPa"] = np.trace(full_q, axis1=1, axis2=2)
        mesh.save(path)
