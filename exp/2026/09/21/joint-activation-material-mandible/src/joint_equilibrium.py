"""Implicit equilibrium with owned snapshots and differentiable Dirichlet data.

The fixed displacement derivative includes direct observations and the reaction
of free tissue. Each backward restores its own material/state snapshot, including
an independently rebuilt contact state when frictionless bone contact is enabled.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
from typing import Any, Literal, override

import optree
import torch
from joint_common import HISTORICAL
from joint_newton import NewtonCgError, refine_newton_cg
from torch.autograd.function import once_differentiable

sys.path.insert(0, str(HISTORICAL))
from face_physics import (
    ForwardConvergenceError,
    StrictLineSearch,
    StrictPncg,
    SuccessPreferredFallbackSolver,
    line_search_receipt,
)

from liblaf.apple.forward import Forward
from liblaf.apple.inverse._diff_forward import _AdjointProblem
from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes

LOG = logging.getLogger(__name__)


class LoggedPncg(StrictPncg):
    """Expose long inner solves without changing their stopping criteria."""

    @override
    def step(self, problem: Any, model_state: Any, opt_state: Any) -> None:
        super().step(problem, model_state, opt_state)
        if opt_state.step % 500 == 0:
            LOG.info(
                "Equilibrium inner step %d, free-force norm %.6g",
                opt_state.step,
                float(opt_state.convergence_state.grad_norm),
            )


def configure_cuda() -> None:
    import warp as wp

    assert torch.cuda.is_available()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    wp.config.mode = "release"
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)


def rigid_displacement(
    points: torch.Tensor, pivot: torch.Tensor, pose: torch.Tensor
) -> torch.Tensor:
    """Rotation vector in radians, followed by translation in model metres."""
    r = pose[:3]
    zero = r[0] * 0
    skew = torch.stack(
        (zero, -r[2], r[1], r[2], zero, -r[0], -r[1], r[0], zero)
    ).reshape(3, 3)
    rotation = torch.matrix_exp(skew)
    return (points - pivot) @ rotation.T + pivot + pose[3:] - points


class Equilibrium:
    def __init__(
        self,
        forward: Forward,
        *,
        rtol: float = 5e-4,
        atol: float = 1e-10,
        adjoint_rtol: float = 5e-4,
        max_steps: int = 5000,
        forward_method: Literal["pncg", "newton_cg"] = "pncg",
        newton_linear_rtol: float = 1e-3,
        newton_max_steps: int = 12,
    ) -> None:
        if forward.model.collision is not None:
            from joint_contact import OwnedContact

            assert isinstance(forward.model.collision, OwnedContact)
        self.forward = forward
        self.forward_method = forward_method
        self.newton_linear_rtol = newton_linear_rtol
        self.newton_max_steps = newton_max_steps
        assert self.forward_method in {"pncg", "newton_cg"}
        assert 0 < self.newton_linear_rtol < 1
        assert self.newton_max_steps > 0
        default = forward.default_optimizer(max_steps=max_steps, rtol=rtol, atol=atol)
        forward.optimizer = LoggedPncg(
            criteria=default.criteria, line_search=StrictLineSearch(max_steps=40)
        )
        self.solver = SuccessPreferredFallbackSolver(
            [
                CupyCG(maxiter=10000, rtol=adjoint_rtol, atol=0.0),
                CupyMinRes(maxiter=10000, tol=adjoint_rtol),
            ]
        )
        self.tolerances = {
            "rtol": rtol,
            "atol": atol,
            "adjoint_rtol": adjoint_rtol,
            "max_steps": max_steps,
        }
        self.forward_solver = {
            "method": self.forward_method,
            "newton_linear_rtol": self.newton_linear_rtol,
            "newton_max_steps": self.newton_max_steps,
            "fallback": None,
        }
        self.last_forward: dict[str, Any] = {}
        self.last_adjoint: dict[str, Any] = {}
        self.warm_adjoints: dict[str, torch.Tensor] = {}
        self.forward_count = 0

    def solve(
        self,
        materials: dict,
        fixed_values: torch.Tensor,
        seed: torch.Tensor,
        *,
        key: str,
    ) -> torch.Tensor:
        leaves, spec = optree.tree_flatten(materials)
        return _Implicit.apply(self, key, spec, fixed_values, seed, *leaves)

    def primal(  # noqa: PLR0915
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        forward = self.forward
        model = forward.model
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        forward.state.u = (
            model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        if model.collision is not None:
            collision = model.collision
            prior_contact = collision.state_at(seed.detach())
            change = forward.state.u - seed.detach()
            fraction = float(
                collision.max_step_size(prior_contact, seed.detach(), change)
            )
            if fraction < 1.0:
                self.last_forward = {
                    "success": False,
                    "method": self.forward_method,
                    "failure": "Dirichlet boundary proposal crosses a contact surface",
                    "contact": {"enabled": True, "ccd_boundary_fraction": fraction},
                    "forward_solver": self.forward_solver,
                }
                message = "Dirichlet boundary proposal fails contact CCD"
                raise ForwardConvergenceError(message, receipt=self.last_forward)
            forward.state.collision = collision.state_at(forward.state.u)
            forward.state.collision.boundary_ccd_fraction = fraction
        started = time.perf_counter()
        forward.last_solution = None
        initial_gradient = forward.problem.grad(forward.state)
        initial_gradient_norm = torch.linalg.vector_norm(initial_gradient)
        initial_gradient_is_finite = bool(torch.isfinite(initial_gradient_norm))
        if (
            initial_gradient_is_finite
            and float(initial_gradient_norm) <= self.tolerances["atol"]
        ):
            torch.cuda.synchronize()
            self.forward_count += 1
            self.last_forward = {
                "success": True,
                "method": self.forward_method,
                "seconds": time.perf_counter() - started,
                "steps": 0,
                "grad_norm": float(initial_gradient_norm),
                "result": "initial_equilibrium",
                "tolerances": self.tolerances,
                "forward_solver": self.forward_solver,
                "line_search": {
                    "implementation": type(forward.optimizer.line_search).__name__,
                    "max_steps": int(forward.optimizer.line_search.max_steps),
                    "status": "not_run",
                    "ok": None,
                    "step": 0,
                    "alpha": None,
                    "f0": None,
                    "f_alpha": None,
                },
            }
            assert torch.isfinite(forward.state.u).all()
            self._record_contact()
            return forward.state.u.detach().clone()

        if self.forward_method == "newton_cg":
            try:
                newton_receipt = refine_newton_cg(
                    forward,
                    rtol=self.tolerances["rtol"],
                    atol=self.tolerances["atol"],
                    linear_rtol=self.newton_linear_rtol,
                    linear_max_iterations=10000,
                    max_newton_steps=self.newton_max_steps,
                    armijo_coefficient=1e-4,
                    max_line_search_steps=40,
                )
            except NewtonCgError as error:
                torch.cuda.synchronize()
                self.last_forward = {
                    "success": False,
                    "method": self.forward_method,
                    "seconds": time.perf_counter() - started,
                    "failure": str(error),
                    "solver": error.receipt,
                    "tolerances": self.tolerances,
                    "forward_solver": self.forward_solver,
                }
                raise ForwardConvergenceError(
                    str(error), receipt=self.last_forward
                ) from error
            torch.cuda.synchronize()
            self.forward_count += 1
            self.last_forward = {
                **newton_receipt,
                "solver_seconds": newton_receipt["seconds"],
                "seconds": time.perf_counter() - started,
                "tolerances": self.tolerances,
                "forward_solver": self.forward_solver,
                "line_search": {
                    "implementation": "owned_state_strict_armijo",
                    "max_steps": 40,
                    "status": "accepted",
                    "ok": True,
                    "step": (
                        newton_receipt["trace"][-1]["line_search_steps"]
                        if newton_receipt["trace"]
                        else 0
                    ),
                    "alpha": (
                        newton_receipt["trace"][-1]["accepted_alpha"]
                        if newton_receipt["trace"]
                        else None
                    ),
                    "f0": (
                        newton_receipt["trace"][-1]["energy"]
                        if newton_receipt["trace"]
                        else None
                    ),
                    "f_alpha": (
                        newton_receipt["trace"][-1]["trials"][-1]["energy"]
                        if newton_receipt["trace"]
                        else None
                    ),
                },
            }
            assert torch.isfinite(forward.state.u).all()
            self._record_contact()
            return forward.state.u.detach().clone()

        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                solution = forward.step()
        except ForwardConvergenceError as error:
            self.last_forward = {
                "success": False,
                "method": self.forward_method,
                "seconds": time.perf_counter() - started,
                "failure": str(error),
                "solver": error.receipt,
                "tolerances": self.tolerances,
                "forward_solver": self.forward_solver,
            }
            raise
        torch.cuda.synchronize()
        conv = solution.state.convergence_state
        self.forward_count += 1
        self.last_forward = {
            "success": bool(solution.success),
            "method": self.forward_method,
            "seconds": time.perf_counter() - started,
            "steps": int(conv.step),
            "grad_norm": float(conv.grad_norm),
            "result": str(solution.result),
            "tolerances": self.tolerances,
            "forward_solver": self.forward_solver,
            "line_search": line_search_receipt(
                forward.optimizer.line_search, solution.state.line_search_state
            ),
        }
        assert solution.success, self.last_forward
        assert torch.isfinite(forward.state.u).all()
        self._record_contact()
        return forward.state.u.detach().clone()

    def _record_contact(self) -> None:
        collision = self.forward.model.collision
        if collision is not None:
            receipt = collision.diagnostics(
                self.forward.state.collision, self.forward.state.u
            )
            self.last_forward["contact"] = receipt
            assert receipt["contact_numerically_valid"], receipt


class _Implicit(torch.autograd.Function):
    @staticmethod
    def forward(
        runtime: Equilibrium,
        _key: str,
        spec: optree.PyTreeSpec,
        fixed: torch.Tensor,
        seed: torch.Tensor,
        *leaves: torch.Tensor,
    ) -> torch.Tensor:
        return runtime.primal(spec.unflatten(leaves), fixed, seed)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        runtime, key, spec, fixed, _seed, *leaves = inputs
        ctx.runtime, ctx.key, ctx.spec = runtime, key, spec
        ctx.save_for_backward(output.detach().clone(), fixed.detach().clone(), *leaves)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        runtime = ctx.runtime
        model = runtime.forward.model
        output, fixed, *saved = ctx.saved_tensors
        original = model.get_materials()
        original_fixed = model.dof_map.fixed_values
        started = time.perf_counter()
        try:
            leaves = [
                value.detach().clone().requires_grad_(needed)
                for value, needed in zip(saved, ctx.needs_input_grad[5:], strict=True)
            ]
            materials = ctx.spec.unflatten(leaves)
            model.set_materials(materials)
            model.dof_map.fixed_values = fixed
            state = model.State(u=output.detach().clone())
            if model.collision is not None:
                state.collision = model.collision.state_at(state.u)
            problem = _AdjointProblem(
                b=-model.dof_map.to_free_grad(grad_output),
                model=model,
                model_state=state,
            )
            initial = runtime.warm_adjoints.get(ctx.key, torch.zeros_like(problem.b))
            if torch.count_nonzero(problem.b) == 0:
                p_free = torch.zeros_like(problem.b)
                residual, relative, result = 0.0, 0.0, "zero right-hand side"
            else:
                solution = runtime.solver.solve(problem, initial)
                assert solution.success, f"Adjoint failed: {solution}"
                p_free = solution.params.detach()
                residual = float(
                    torch.linalg.vector_norm(problem.matvec(p_free) - problem.b)
                )
                relative = residual / float(torch.linalg.vector_norm(problem.b))
                assert relative <= runtime.tolerances["adjoint_rtol"] * 1.05, (
                    relative,
                    solution,
                )
                result = str(solution.result)
            runtime.warm_adjoints[ctx.key] = p_free.detach().clone()
            p = model.dof_map.to_full_grad(p_free)
            model.mixed_derivative_prod(state, p)
            gradients = [leaf.grad for leaf in leaves]
            # p = -H_ff^{-1} L_f, hence L_c + H_cf p includes the implicit term.
            fixed_gradient = (grad_output + model.hess_prod(state, p)).flatten()[
                model.dof_map.fixed_indices
            ]
            assert torch.isfinite(fixed_gradient).all()
            for leaf, gradient in zip(leaves, gradients, strict=True):
                if leaf.requires_grad:
                    assert gradient is not None
                    assert torch.isfinite(gradient).all()
            torch.cuda.synchronize()
            runtime.last_adjoint = {
                "success": True,
                "residual": residual,
                "relative_residual": relative,
                "result": result,
                "seconds": time.perf_counter() - started,
                "key": ctx.key,
            }
        finally:
            model.set_materials(original)
            model.dof_map.fixed_values = original_fixed
        return (None, None, None, fixed_gradient, None, *gradients)
