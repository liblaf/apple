# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915
"""Fixed-fixture no-skin face equilibrium with signed additive active stress."""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override

import numpy as np
import pyvista as pv
import torch
import warp as wp
from stress_material import StableNeoHookeanStress

from liblaf.apple.common import FRACTION, LAMBDA, MU, NU, E
from liblaf.apple.forward import Forward, ModelBuilder
from liblaf.apple.inverse import DifferentiableForward, ImplicitNumericalError
from liblaf.apple.solvers.linalg import BaseProblem, FallbackSolver, Problem, Result
from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes
from liblaf.apple.solvers.optim import Result as OptimizerResult
from liblaf.apple.solvers.optim import Solution as OptimizerSolution
from liblaf.apple.warp.fem import (
    StableNeoHookean,
    StableNeoHookeanActive,
    WarpPotentialFem,
)

LOG = logging.getLogger(__name__)


class ForwardConvergenceError(RuntimeError):
    """A forward solve failed its declared accepted-state conditions."""

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None) -> None:
        self.receipt = receipt
        super().__init__(message)


def _json_receipt(value: Any) -> Any:
    """Convert a failure receipt to JSON-safe finite diagnostic data."""
    if torch.is_tensor(value):
        return _json_receipt(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_receipt(value.item())
    if isinstance(value, dict):
        return {str(key): _json_receipt(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_receipt(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite"
    return value


class SuccessPreferredFallbackSolver(FallbackSolver):
    """Retain the first successful adjoint; otherwise expose the best residual."""

    @override
    def compute(self, problem: BaseProblem, state: Any) -> Result:
        typed_problem = cast("Problem", problem)
        absolute, relative = [], []
        success_index: int | None = None
        finite = []
        for index, solver in enumerate(self.solvers):
            solution = solver.solve(typed_problem, state.init_params)
            state.solutions.append(solution)
            residual = torch.linalg.vector_norm(
                typed_problem.matvec(solution.state.params) - typed_problem.b
            )
            usable = bool(
                torch.isfinite(residual) and torch.isfinite(solution.params).all()
            )
            absolute.append(residual)
            relative.append(residual / torch.linalg.vector_norm(typed_problem.b))
            finite.append(usable)
            if solution.success and usable:
                success_index = index
                break
        state.absolute_residuals = torch.as_tensor(absolute)
        state.relative_residuals = torch.as_tensor(relative)
        ranked = torch.where(
            torch.as_tensor(finite, device=state.absolute_residuals.device),
            state.absolute_residuals,
            torch.full_like(state.absolute_residuals, torch.inf),
        )
        if not bool(torch.isfinite(ranked).any()):
            message = "all adjoint fallback candidates are nonfinite"
            raise ImplicitNumericalError(message)
        state.best_index = (
            torch.argmin(ranked)
            if success_index is None
            else torch.as_tensor(success_index, dtype=torch.int32)
        )
        return state.result


@functools.cache
def _newton_policy() -> tuple[type, Any, Any, Any]:
    """Load the single exact Newton-CG implementation used by neutral fitting.

    Keeping this import here makes the study's provenance explicit while avoiding
    a second copy of the safeguarding/PCG policy.  ``accelerated_solvers`` needs
    the joint study's support modules at import time even though this no-skin
    fixture never creates their contact problem.
    """
    root = Path(__file__).resolve().parents[6]
    for source in (
        root / "exp/2026/09/22/solver-performance/src",
        root / "exp/2026/09/21/joint-activation-material-mandible/src",
    ):
        text = str(source)
        if text not in sys.path:
            sys.path.insert(0, text)
    import accelerated_solvers
    from mesh_step_scale import mean_rest_edge_length

    return (
        accelerated_solvers.CachedProblem,
        accelerated_solvers.ForwardConvergenceError,
        accelerated_solvers.safeguarded_newton,
        mean_rest_edge_length,
    )


def strain_to_activation_inv(strain: torch.Tensor) -> torch.Tensor:
    """Encode dimensionless symmetric ``S`` as native ``B = I + S`` controls."""
    assert strain.shape[-2:] == (3, 3)
    assert bool(torch.isfinite(strain).all())
    assert bool(torch.allclose(strain, strain.mT, rtol=0.0, atol=1e-12))
    return torch.stack(
        (
            strain[..., 0, 0],
            strain[..., 1, 1],
            strain[..., 2, 2],
            strain[..., 0, 1],
            strain[..., 1, 2],
            strain[..., 0, 2],
        ),
        dim=-1,
    )


def install_exact_bulk_diagonal() -> dict[str, str]:
    """Use the neutral Newton policy's unprojected bulk Jacobi diagonal.

    This no-skin fixture has no collision or membrane contribution.  Rebind only
    its two bulk material classes; the energy, gradient, and physical HVP stay
    unchanged.  The process-local binding is reported with every run.
    """
    for material in (StableNeoHookean, StableNeoHookeanStress, StableNeoHookeanActive):
        material.hess_diag_kernel = WarpPotentialFem.make_hess_diag_kernel(
            material.hess_diag_func, clamp_hess_diag=False
        )
    return {
        "scope": "process-local no-skin bulk material kernel binding",
        "bulk": "analytic unprojected diagonal; negative entries retained, including native active strain",
        "skin": "absent by frozen no-skin fixture",
        "contact": "absent by frozen contact-off fixture",
        "energy_gradient_and_hessian_vector_products_changed": "false",
    }


class NewtonCgForwardOptimizer:
    """Forward adapter for the neutral driver's exact safeguarded Newton-CG.

    The referenced policy adds shifts only to its search systems.  The model's
    unshifted Hessian remains the physical operator used by implicit backward.
    """

    def __init__(
        self,
        *,
        force_atol: float,
        force_rtol: float,
        max_steps: int,
        linear_rtol: float = 1e-3,
        linear_max_steps: int = 1_000,
        max_step_norm: float,
        initial_shift_scale: float = 1.0,
        require_convergence: bool = True,
    ) -> None:
        assert force_atol > 0
        assert 0 <= force_rtol < 1
        assert max_steps > 0
        assert 0 < linear_rtol < 1
        assert linear_max_steps > 0
        assert max_step_norm > 0
        assert initial_shift_scale >= 0
        self.force_atol = force_atol
        self.force_rtol = force_rtol
        self.max_steps = max_steps
        self.linear_rtol = linear_rtol
        self.linear_max_steps = linear_max_steps
        self.max_step_norm = max_step_norm
        self.initial_shift_scale = initial_shift_scale
        self.require_convergence = require_convergence
        self.last_receipt: dict[str, Any] | None = None

    def _solution(
        self,
        problem: Any,
        state: Any,
        final_force: float,
        receipt: dict[str, Any],
        *,
        converged: bool,
    ) -> Any:
        self.last_receipt = receipt
        result = (
            OptimizerResult.SUCCESS if converged else OptimizerResult.MAX_STEPS_REACHED
        )
        opt_state = SimpleNamespace(
            params=problem.model.dof_map.to_free(state.u).detach().clone(),
            convergence_state=SimpleNamespace(grad_norm=torch.as_tensor(final_force)),
        )
        opt_state.receipt = receipt
        return OptimizerSolution(result=result, state=opt_state, stats={})

    def minimize(self, problem: Any, state: Any, _params: torch.Tensor) -> Any:
        cached_type, policy_error, safeguarded_newton, _ = _newton_policy()
        cached = cached_type(problem, cache_gradient=True, exact_curvature=True)
        initial_force = float(torch.linalg.vector_norm(cached.grad(state)))
        if not math.isfinite(initial_force):
            message = "nonfinite initial forward force"
            raise ForwardConvergenceError(message)
        threshold = max(self.force_atol, self.force_rtol * initial_force)
        failure = None
        try:
            state, solve = safeguarded_newton(
                cached,
                state,
                atol=threshold,
                linear_rtol=self.linear_rtol,
                linear_max_steps=self.linear_max_steps,
                max_steps=self.max_steps,
                max_step_norm=self.max_step_norm,
                armijo=1e-4,
                max_shift_attempts=8,
                max_backtracking_trials=8,
                backtracking_factor=0.5,
                preconditioner="diag",
                initial_shift_scale=self.initial_shift_scale,
                shift_policy="reset",
                shift_scale_policy="mean_abs",
            )
        except policy_error as error:
            if self.require_convergence:
                raise ForwardConvergenceError(
                    str(error), receipt=getattr(error, "receipt", None)
                ) from error
            failure = {
                "reason": str(error),
                "receipt": _json_receipt(getattr(error, "receipt", None)),
            }
            solve = None
        # This is deliberately a fresh physical force at the returned state.
        # It is not the last pre-update Newton force or a shifted-system norm.
        final_force = float(torch.linalg.vector_norm(problem.grad(state)))
        if not math.isfinite(final_force):
            message = "accepted state has nonfinite physical force"
            raise ForwardConvergenceError(
                message, receipt={"newton": solve, "failure": failure}
            )
        converged = final_force <= threshold and failure is None
        if not converged and self.require_convergence:
            message = "accepted state failed declared physical force gate"
            raise ForwardConvergenceError(
                message,
                receipt={
                    "initial_force_norm": initial_force,
                    "accepted_force_norm": final_force,
                    "accepted_force_threshold": threshold,
                    "newton": solve,
                },
            )
        receipt: dict[str, Any] = {
            "success": converged,
            "method": "neutral-safeguarded-newton-cg",
            "initial_force_norm": initial_force,
            "accepted_force_norm": final_force,
            "accepted_force_threshold": threshold,
            "force_atol": self.force_atol,
            "force_rtol": self.force_rtol,
            "linear_rtol": self.linear_rtol,
            "linear_max_steps": self.linear_max_steps,
            "max_steps": self.max_steps,
            "max_step_norm_m": self.max_step_norm,
            "initial_shift_scale": self.initial_shift_scale,
            "initial_shift_policy": "mean_abs_diagonal",
            "newton": solve,
            "work": dict(cached.counts),
            "search_hessian": "shifted only for Newton-CG search",
            "implicit_hessian": "unshifted physical Hessian",
        }
        if failure is not None:
            receipt["failure"] = failure
        return self._solution(problem, state, final_force, receipt, converged=converged)


def configure() -> None:
    """Configure the shared CUDA runtime without claiming exclusive resources."""
    assert torch.cuda.is_available()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    wp.config.mode = "release"
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)


def lame(young_mpa: float, nu: float) -> tuple[float, float, float]:
    assert young_mpa > 0.0
    assert 0.0 < nu < 0.5
    mu = young_mpa / (2.0 * (1.0 + nu))
    classical = young_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, classical, classical + mu


def set_material(
    mesh: pv.UnstructuredGrid, young_mpa: float, fraction: np.ndarray
) -> dict:
    """Set the new all-SNH nu=.49 convention for one mixture constituent."""
    nu = 0.49
    mu, classical, lambda_code = lame(young_mpa, nu)
    mesh.cell_data[E.vtk] = np.full(mesh.n_cells, young_mpa)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_code)
    mesh.cell_data[FRACTION.vtk] = np.asarray(fraction, dtype=np.float64)
    return {
        "E_MPa": young_mpa,
        "nu": nu,
        "mu_code_MPa": mu,
        "lambda_classical_MPa": classical,
        "lambda_code_MPa": lambda_code,
    }


def active_graph(
    points: np.ndarray,
    tets: np.ndarray,
    ids: np.ndarray,
    region: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same 5-mm-prior graph contract as the frozen face study."""
    active = tets[ids]
    pattern = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.sort(active[:, pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(ids)), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(pair) == 1)
    i, j, face = owner[pair], owner[pair + 1], faces[pair]
    same = region[i] == region[j]
    i, j, face = i[same], j[same], face[same]
    xyz = points[face]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    centers = points[active].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    assert np.all(distance > 0.0)
    f = fraction[ids]
    return i, j, area / distance * (2.0 * f[i] * f[j] / (f[i] + f[j]))


class FacePhysics:
    """Smile fixture with selectable additive-stress or native active-strain muscle."""

    def __init__(
        self,
        fixture: Path | str,
        *,
        rtol: float = 0.0,
        atol: float = 1e-10,
        adjoint_rtol: float = 1e-7,
        adjoint_atol: float = 0.0,
        max_newton_steps: int = 100,
        newton_linear_rtol: float = 1e-3,
        newton_linear_max_steps: int = 10_000,
        activation_model: str = "stress",
    ) -> None:
        assert 0.0 <= rtol < 1.0
        assert atol > 0.0
        assert 0.0 <= adjoint_rtol < 1.0
        assert adjoint_atol >= 0.0
        assert max_newton_steps > 0
        assert 0.0 < newton_linear_rtol < 1.0
        assert newton_linear_max_steps > 0
        assert activation_model in {"stress", "strain"}
        self.activation_model = activation_model
        fixture = Path(fixture)
        self.mesh = pv.read(fixture / "volume.vtu")
        self.skin = pv.read(fixture / "skin.vtp")
        self.points = np.asarray(self.mesh.points).copy()
        self.tets = np.asarray(self.mesh.cells).reshape(-1, 5)[:, 1:].copy()
        self.ids = np.flatnonzero(self.mesh.cell_data["ActivationMask"])
        self.region = np.asarray(self.mesh.cell_data["ActivationControlId"], dtype=int)[
            self.ids
        ]
        assert np.array_equal(np.unique(self.region), np.arange(self.region.max() + 1))
        target = np.asarray(self.mesh.point_data["Smile"])
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
        self.volumes_all = np.linalg.det(dm) / 6.0
        assert np.all(self.volumes_all > 0.0)
        fraction = np.asarray(self.mesh.cell_data["MuscleFraction"], dtype=np.float64)
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
        builder = ModelBuilder()
        builder.add_vertices(self.mesh)
        builder.add_fixed(self.mesh)
        spec = {}
        muscle_material = (
            StableNeoHookeanStress
            if activation_model == "stress"
            else StableNeoHookeanActive
        )
        for name, young, cls in (
            ("fat", 0.0112, StableNeoHookean),
            ("aponeurosis", 1.693, StableNeoHookean),
            ("muscle", 0.012, muscle_material),
        ):
            spec[name] = set_material(
                self.mesh, young, self.mesh.cell_data[name.title() + "Fraction"]
            )
            builder.add_potential(cls.from_pyvista(self.mesh, name=name))
        self.diagonal_policy = install_exact_bulk_diagonal()
        self.forward = Forward(builder.finalize())
        _, _, _, mean_rest_edge_length = _newton_policy()
        mean_edge = mean_rest_edge_length(self.forward.model, self.points)
        assert math.isfinite(mean_edge)
        assert mean_edge > 0.0
        self.forward_tolerance = {
            "force_atol": atol,
            "force_rtol": rtol,
            "adjoint_atol": adjoint_atol,
            "adjoint_rtol": adjoint_rtol,
            "max_newton_steps": max_newton_steps,
            "newton_linear_rtol": newton_linear_rtol,
            "newton_linear_max_steps": newton_linear_max_steps,
            "newton_max_step_norm_m": 0.5 * mean_edge,
            "newton_initial_shift_scale": 1.0,
        }
        self.forward.optimizer = NewtonCgForwardOptimizer(
            force_atol=atol,
            force_rtol=rtol,
            max_steps=max_newton_steps,
            linear_rtol=newton_linear_rtol,
            linear_max_steps=newton_linear_max_steps,
            max_step_norm=self.forward_tolerance["newton_max_step_norm_m"],
        )
        self.diff = DifferentiableForward(
            self.forward,
            forward_residual_atol=atol,
            forward_residual_rtol=rtol,
            adjoint_residual_atol=adjoint_atol,
            adjoint_residual_rtol=adjoint_rtol,
        )
        self.diff.adjoint_solver = SuccessPreferredFallbackSolver(
            [
                CupyCG(maxiter=10_000, rtol=adjoint_rtol, atol=adjoint_atol),
                CupyMinRes(maxiter=10_000, tol=adjoint_rtol),
            ]
        )
        self.materials = self.forward.model.get_materials()
        self.solve_count = 0
        self.material_spec = {
            **{
                f"{name}_{key}": value
                for name, row in spec.items()
                for key, value in row.items()
            },
            "activation_model": activation_model,
            "muscle_model": (
                "stable-neo-hookean-signed-additive-Q"
                if activation_model == "stress"
                else "stable-neo-hookean-native-active-strain"
            ),
            "activation_contract": (
                "finite symmetric Q_MPa supplied by caller; no projection or cap in physics"
                if activation_model == "stress"
                else "finite symmetric dimensionless S supplied by caller; native B=I+S uses raw off-diagonals"
            ),
            "activation_energy": (
                "additive 1/2 Q:(F^T F-I)"
                if activation_model == "stress"
                else "stable-neo-hookean active strain with activated norm ||F B||^2, B=I+S"
            ),
            "activation_volume_determinant": (
                "not applicable to additive stress"
                if activation_model == "stress"
                else "both determinant terms use the physical J=det(F), not det(F B)"
            ),
            "skin_energy": "disabled: no membrane potential is added",
            "contact_enabled": False,
            "collision_policy": "disabled by frozen no-skin fixture; no contact potential or CCD is installed",
            "jaw_enabled": False,
            "fixed_boundary": "copied from frozen volume.vtu IsFixed contract",
            "volume_lame_convention": "lambda_code = lambda_classical + mu for all Stable Neo-Hookean bulk tissues",
            "newton_diagonal_policy": self.diagonal_policy,
        }

    @contextlib.contextmanager
    def accuracy(self, factor: float):
        """Temporarily tighten declared equilibrium and adjoint tolerances.

        This changes no model, material, boundary, or collision setting.  It is
        intended for replaying a proposed outer step before declaring small
        objective changes or stationarity.
        """
        assert 0.0 < factor < 1.0
        optimizer = self.forward.optimizer
        assert isinstance(optimizer, NewtonCgForwardOptimizer)
        previous = dict(self.forward_tolerance)
        previous_force_atol = optimizer.force_atol
        previous_force_rtol = optimizer.force_rtol
        previous_linear_rtol = optimizer.linear_rtol
        previous_diff = {
            "forward_residual_atol": self.diff.forward_residual_atol,
            "forward_residual_rtol": self.diff.forward_residual_rtol,
            "adjoint_residual_atol": self.diff.adjoint_residual_atol,
            "adjoint_residual_rtol": self.diff.adjoint_residual_rtol,
        }
        previous_adjoint = [
            (
                solver,
                getattr(solver, "rtol", None),
                getattr(solver, "atol", None),
                getattr(solver, "tol", None),
            )
            for solver in self.diff.adjoint_solver.solvers
        ]
        try:
            optimizer.force_atol *= factor
            optimizer.force_rtol *= factor
            optimizer.linear_rtol *= factor
            self.diff.forward_residual_atol *= factor
            self.diff.forward_residual_rtol *= factor
            self.diff.adjoint_residual_atol *= factor
            self.diff.adjoint_residual_rtol *= factor
            self.forward_tolerance["force_atol"] = optimizer.force_atol
            self.forward_tolerance["force_rtol"] = optimizer.force_rtol
            self.forward_tolerance["newton_linear_rtol"] = optimizer.linear_rtol
            self.forward_tolerance["adjoint_rtol"] *= factor
            self.forward_tolerance["adjoint_atol"] *= factor
            for solver, rtol, atol_value, tol in previous_adjoint:
                if rtol is not None:
                    solver.rtol = rtol * factor
                if atol_value is not None:
                    solver.atol = atol_value * factor
                if tol is not None:
                    solver.tol = tol * factor
            yield self
        finally:
            optimizer.force_atol = previous_force_atol
            optimizer.force_rtol = previous_force_rtol
            optimizer.linear_rtol = previous_linear_rtol
            for name, value in previous_diff.items():
                setattr(self.diff, name, value)
            self.forward_tolerance = previous
            for solver, rtol, atol_value, tol in previous_adjoint:
                if rtol is not None:
                    solver.rtol = rtol
                if atol_value is not None:
                    solver.atol = atol_value
                if tol is not None:
                    solver.tol = tol

    @contextlib.contextmanager
    def approximate_solves(self):
        """Temporarily retain finite nonconverged primal and adjoint states.

        This is continuation-only plumbing.  Strict solve acceptance remains the
        default and callers must keep the failed receipt as evidence; nonfinite
        states and gradients still fail visibly.
        """
        optimizer = self.forward.optimizer
        assert isinstance(optimizer, NewtonCgForwardOptimizer)
        previous_optimizer = optimizer.require_convergence
        previous_implicit = self.diff.require_convergence
        try:
            optimizer.require_convergence = False
            self.diff.require_convergence = False
            yield self
        finally:
            optimizer.require_convergence = previous_optimizer
            self.diff.require_convergence = previous_implicit

    def surface_weights(self) -> np.ndarray:
        original = np.asarray(self.skin.point_data["GlobalPointId"], dtype=int)
        tri = original[np.asarray(self.skin.faces).reshape(-1, 4)[:, 1:]]
        xyz = self.points[tri]
        area = (
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            / 2
        )
        weights = np.zeros(len(self.points))
        np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
        weights = weights[self.top]
        assert np.all(weights >= 0.0)
        assert weights.sum() > 0.0
        return weights / weights.sum()

    def solve(self, activation: torch.Tensor, seed: np.ndarray) -> torch.Tensor:
        """Solve from stress ``Q_MPa`` or dimensionless active strain ``S``."""
        assert activation.shape == (len(self.ids), 3, 3)
        assert activation.device.type == "cuda"
        assert activation.dtype == torch.float64
        assert bool(torch.isfinite(activation).all())
        assert bool(
            torch.allclose(
                activation, activation.transpose(-1, -2), rtol=0.0, atol=1e-12
            )
        )
        assert seed.shape == self.points.shape
        assert np.isfinite(seed).all()
        self.diff.last_adjoint_solution = None
        self.forward.state.u = self.forward.state.u.detach().clone()
        with torch.no_grad():
            self.forward.model.update(self.forward.state, torch.as_tensor(seed))
        if self.activation_model == "stress":
            self.materials["muscle"]["active_stress"] = torch.zeros(
                (self.mesh.n_cells, 3, 3),
                device=activation.device,
                dtype=activation.dtype,
            ).index_copy(0, self.id_t, activation)
        else:
            self.materials["muscle"]["activation_inv"] = torch.zeros(
                (self.mesh.n_cells, 6),
                device=activation.device,
                dtype=activation.dtype,
            ).index_copy(0, self.id_t, strain_to_activation_inv(activation))
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                u = self.diff.forward(self.materials).clone()
        except ForwardConvergenceError as error:
            self.solve_count += 1
            self.last_forward = self.failure_receipt(error.receipt, captured.getvalue())
            message = "Forward failed"
            raise ForwardConvergenceError(message, receipt=self.last_forward) from error
        self.solve_count += 1
        solution = self.diff.last_solution
        assert solution is not None
        policy = self.forward.optimizer
        assert isinstance(policy, NewtonCgForwardOptimizer)
        policy_receipt = policy.last_receipt
        assert policy_receipt is not None
        implicit_receipt = self.diff.last_forward_receipt
        assert implicit_receipt is not None
        if self.diff.require_convergence:
            assert implicit_receipt.success
        receipt = {
            "success": bool(implicit_receipt.success),
            "solver_success": bool(solution.success),
            "steps": (
                None
                if policy_receipt["newton"] is None
                else int(policy_receipt["newton"]["steps"])
            ),
            "accepted_force_norm": policy_receipt["accepted_force_norm"],
            "accepted_force_threshold": policy_receipt["accepted_force_threshold"],
            "result": str(solution.result),
            "tolerance": dict(self.forward_tolerance),
            "forward_solver": policy_receipt,
            "physical_residual": {
                "absolute": float(implicit_receipt.absolute_residual.detach().cpu()),
                "reference_norm": float(implicit_receipt.reference_norm.detach().cpu()),
                "threshold": float(implicit_receipt.threshold.detach().cpu()),
            },
            "solver_valid": bool(solution.success and implicit_receipt.success),
            "stdout": captured.getvalue(),
        }
        self.last_forward = receipt
        if not solution.success and self.diff.require_convergence:
            message = "Newton-CG did not converge"
            raise ForwardConvergenceError(message, receipt=receipt)
        assert bool(torch.isfinite(u).all())
        return u

    def failure_receipt(
        self, solver_receipt: dict[str, Any] | None, stdout: str
    ) -> dict:
        return {
            "success": False,
            "solver_valid": False,
            "steps": None,
            "accepted_force_norm": None,
            "accepted_force_threshold": None,
            "result": "newton_cg_failure",
            "tolerance": dict(self.forward_tolerance),
            "forward_solver": solver_receipt,
            "stdout": stdout,
        }

    def check_adjoint(self) -> dict:
        receipt = self.diff.last_adjoint_receipt
        assert receipt is not None, "No adjoint solve was recorded"
        if self.diff.require_convergence:
            assert receipt.success, "Adjoint residual exceeded its declared tolerance"
        if receipt.zero_rhs:
            return {
                "success": True,
                "zero_rhs": True,
                "absolute_residual": float(receipt.absolute_residual.detach().cpu()),
                "reference_norm": float(receipt.reference_norm.detach().cpu()),
                "threshold": float(receipt.threshold.detach().cpu()),
            }
        solution = self.diff.last_adjoint_solution
        assert solution is not None, "No adjoint solve was recorded"
        if self.diff.require_convergence:
            assert solution.success, f"Adjoint failed: {solution}"
        state = solution.state
        index = int(state.best_index)
        absolute = float(state.absolute_residuals[index].detach().cpu())
        relative = float(state.relative_residuals[index].detach().cpu())
        if not math.isfinite(absolute):
            message = "adjoint residual is nonfinite"
            raise ImplicitNumericalError(message)
        solver = self.diff.adjoint_solver.solvers[index]
        tolerance = getattr(solver, "rtol", getattr(solver, "tol", None))
        assert tolerance is not None
        return {
            "success": bool(receipt.success),
            "result": str(solution.result),
            "solver_index": index,
            "absolute_residual": absolute,
            "relative_residual": relative if math.isfinite(relative) else "nonfinite",
            "relative_tolerance": tolerance,
            "accepted_absolute_residual": float(
                receipt.absolute_residual.detach().cpu()
            ),
            "accepted_threshold": float(receipt.threshold.detach().cpu()),
            "zero_rhs": False,
        }

    def detf(self, u: np.ndarray) -> np.ndarray:
        x = self.points + u
        ds = np.transpose(x[self.tets[:, 1:]] - x[self.tets[:, :1]], (0, 2, 1))
        return np.linalg.det(ds @ self.dm_inv)

    def save_mesh(
        self, path: Path | str, u: np.ndarray, active_stress: np.ndarray | None = None
    ) -> None:
        mesh = self.mesh.copy()
        mesh.points = self.points + u
        mesh.point_data["RestPosition"] = self.points
        mesh.point_data["Displacement"] = u
        mesh.point_data["TargetDisplacement"] = self.target
        mesh.cell_data["DetF"] = self.detf(u)
        if active_stress is not None:
            assert active_stress.shape == (len(self.ids), 3, 3)
            full = np.zeros((len(self.tets), 3, 3))
            full[self.ids] = active_stress
            mesh.cell_data["ActiveStressMatrixMPa"] = full.reshape(-1, 9)
            mesh.cell_data["ActiveStressTraceMPa"] = np.trace(full, axis1=1, axis2=2)
        mesh.save(path)
