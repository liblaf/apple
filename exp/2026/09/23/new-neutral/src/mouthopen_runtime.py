# ruff: noqa: EM101, PLR0915, TRY003, TRY301
"""Implicit MouthOpen equilibrium with the production hybrid primal.

The Newton shifts in this module are a search device only.  By default,
``Equilibrium``'s existing ``_Implicit`` backward uses the exact unshifted free
Hessian.  An explicit positive adjoint shift selects the separately labelled,
approximate experiment-local backward below.
"""

from __future__ import annotations

import copy
import time
from typing import Any, override

import ipctk
import optree
import torch
from accelerated_solvers import CachedProblem, safeguarded_newton
from hybrid_first_solver import SparseNewtonProblem
from hybrid_hessian import HybridHessian
from joint_equilibrium import Equilibrium, ForwardConvergenceError
from joint_expression_equilibrium import FeasibleExpressionProblem
from pncg_first import run_pncg_phase
from torch.autograd.function import once_differentiable

from liblaf.apple.solvers.linalg.base._problem import Problem as LinearProblem
from liblaf.apple.solvers.linalg.cupy import CupyCG
from liblaf.apple.solvers.utils import is_implemented


class _SparseAdjointProblem:
    """CSR proxy for an owned native adjoint problem.

    The native problem remains responsible for the final residual check and
    fixed-boundary reaction in ``_Implicit.backward``.  This proxy only makes
    the iterative solve use the assembled physical free Hessian.
    """

    def __init__(
        self,
        native: Any,
        hessian: HybridHessian,
        deadline: float | None = None,
        relative_shift: float = 0.0,
    ) -> None:
        self._native = native
        self._hessian = hessian
        self.deadline = deadline
        self.operator_applications = 0
        self.b = native.b
        diagonal = hessian.diagonal(native.model_state)
        assert bool(torch.isfinite(diagonal).all())
        self.diagonal_scale = float(diagonal.abs().mean())
        assert self.diagonal_scale > 0
        assert relative_shift >= 0
        self.relative_shift = relative_shift
        self.absolute_shift = relative_shift * self.diagonal_scale
        self._shift = torch.as_tensor(
            self.absolute_shift, device=diagonal.device, dtype=diagonal.dtype
        )
        shifted_diagonal = diagonal + self._shift
        assert bool(torch.all(shifted_diagonal != 0))
        self._preconditioner = shifted_diagonal.abs().reciprocal()
        # CupySolver asks these exact protocol hooks before it constructs its
        # LinearOperator preconditioner.  Assert the proxy fulfils both.
        assert is_implemented(self, LinearProblem.precondition)
        assert is_implemented(self, LinearProblem.rprecondition)

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise ForwardConvergenceError("declared adjoint wall budget exhausted")
        self.operator_applications += 1
        return (
            self._hessian.apply(self._native.model_state, vector) + self._shift * vector
        )

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matvec(vector)

    def precondition(self, vector: torch.Tensor) -> torch.Tensor:
        return self._preconditioner * vector

    def rprecondition(self, vector: torch.Tensor) -> torch.Tensor:
        return self.precondition(vector)


class SparseAdjointSolver:
    """Route implicit adjoints through an audited CSR operator, optionally shifted."""

    def __init__(self, solver: Any, runtime: MouthOpenHybridEquilibrium) -> None:
        self._solver = solver
        self._runtime = runtime

    def solve(self, problem: Any, initial: torch.Tensor) -> Any:
        # _Implicit creates this structural interface.  Keep other potential
        # users of the linear solver untouched.
        if not all(hasattr(problem, name) for name in ("model", "model_state", "b")):
            return self._solver.solve(problem, initial)
        self._runtime.last_sparse_adjoint = {}
        attempts = []
        stage = "initialization"
        hessian: HybridHessian | None = None
        proxy: _SparseAdjointProblem | None = None
        relative: float | None = None

        def check_deadline(next_stage: str) -> None:
            nonlocal stage
            stage = next_stage
            if (
                self._runtime.deadline is not None
                and time.perf_counter() >= self._runtime.deadline
            ):
                raise ForwardConvergenceError("declared adjoint wall budget exhausted")

        try:
            check_deadline("sparse Hessian construction")
            hessian = HybridHessian(problem.model)
            generator = torch.Generator(device=problem.b.device).manual_seed(20260923)
            direction = torch.randn(
                problem.b.shape,
                device=problem.b.device,
                dtype=problem.b.dtype,
                generator=generator,
            )
            check_deadline("native operator proof")
            native = problem.matvec(direction)
            check_deadline("sparse operator proof")
            sparse = hessian.apply(problem.model_state, direction)
            check_deadline("operator proof comparison")
            denominator = max(float(torch.linalg.vector_norm(native)), 1e-30)
            relative = float(torch.linalg.vector_norm(sparse - native)) / denominator
            assert relative <= 1e-10, relative
            check_deadline("sparse preconditioner setup")
            proxy = _SparseAdjointProblem(
                problem,
                hessian,
                self._runtime.deadline,
                self._runtime.adjoint_relative_shift,
            )
            check_deadline("CG setup")
            solver = CupyCG(
                maxiter=100000,
                rtol=self._runtime.tolerances["adjoint_rtol"],
                atol=0.0,
            )
            seed = initial
            rhs_norm = max(float(torch.linalg.vector_norm(proxy.b)), 1e-30)
            for _ in range(3):
                check_deadline("CG solve")
                before = proxy.operator_applications
                solution = solver.solve(proxy, seed)
                check_deadline("sparse true residual")
                csr_shifted_residual = proxy.matvec(solution.params) - proxy.b
                actual_residual_norm = float(
                    torch.linalg.vector_norm(csr_shifted_residual)
                )
                actual = actual_residual_norm / rhs_norm
                check_deadline("native true residual")
                native_residual = problem.matvec(solution.params) - problem.b
                native_residual_norm = float(torch.linalg.vector_norm(native_residual))
                native_actual = native_residual_norm / rhs_norm
                native_shifted_residual = (
                    native_residual + proxy.absolute_shift * solution.params
                )
                native_shifted_residual_norm = float(
                    torch.linalg.vector_norm(native_shifted_residual)
                )
                native_shifted_actual = native_shifted_residual_norm / rhs_norm
                check_deadline("true residual comparison")
                attempts.append(
                    {
                        "method": "CupyCG",
                        "maxiter": 100000,
                        "operator_applications": proxy.operator_applications - before,
                        "solver_success": bool(solution.success),
                        "shifted_residual_norm": actual_residual_norm,
                        "shifted_relative_residual": actual,
                        "native_shifted_residual_norm": native_shifted_residual_norm,
                        "native_shifted_relative_residual": native_shifted_actual,
                        "original_unshifted_residual_norm": native_residual_norm,
                        "original_unshifted_relative_residual": native_actual,
                    }
                )
                if (
                    max(actual, native_shifted_actual)
                    <= self._runtime.tolerances["adjoint_rtol"]
                ):
                    break
                seed = solution.params.detach().clone()
            self._runtime.last_sparse_adjoint = {
                "method": "hybrid_fem_ipc_free_csr_with_optional_relative_shift",
                "approximate": proxy.absolute_shift > 0,
                # lambda is fixed for all CG restarts in this backward pass.
                "shift": proxy.absolute_shift,
                "relative_shift": proxy.relative_shift,
                "diagonal_scale": proxy.diagonal_scale,
                # Retained aliases make archived exact-run consumers readable.
                "absolute_shift": proxy.absolute_shift,
                "physical_diagonal_abs_mean": proxy.diagonal_scale,
                "operator_relative_error": relative,
                "hessian": dict(hessian.metadata),
                "solver_success": bool(solution.success),
                "shifted_residual_norm": actual_residual_norm,
                "shifted_relative_residual": actual,
                "native_shifted_residual_norm": native_shifted_residual_norm,
                "native_shifted_relative_residual": native_shifted_actual,
                "original_unshifted_residual_norm": native_residual_norm,
                "original_unshifted_relative_residual": native_actual,
                # Compatibility names for exact-run consumers.
                "residual": max(actual_residual_norm, native_shifted_residual_norm),
                "relative_residual": max(actual, native_shifted_actual),
                "actual_relative_residual": actual,
                "native_relative_residual": native_actual,
                "attempts": attempts,
            }
            assert (
                max(actual, native_shifted_actual)
                <= self._runtime.tolerances["adjoint_rtol"]
            ), self._runtime.last_sparse_adjoint
            return solution  # noqa: TRY300
        except ForwardConvergenceError as error:
            self._runtime.last_sparse_adjoint = {
                "success": False,
                "method": "hybrid_fem_ipc_free_csr_with_optional_relative_shift",
                "relative_shift": self._runtime.adjoint_relative_shift,
                "shift": None if proxy is None else proxy.absolute_shift,
                "diagonal_scale": None if proxy is None else proxy.diagonal_scale,
                "failure": str(error),
                "stage": stage,
                "operator_relative_error": relative,
                "hessian": {} if hessian is None else dict(hessian.metadata),
                "attempts": attempts,
            }
            raise


class _ShiftedImplicit(torch.autograd.Function):
    """Implicit backward with a fixed damped free-space adjoint operator."""

    @staticmethod
    def forward(
        runtime: MouthOpenHybridEquilibrium,
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
        ctx.relative_shift = runtime.adjoint_relative_shift
        ctx.save_for_backward(output.detach().clone(), fixed.detach().clone(), *leaves)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        runtime = ctx.runtime
        assert ctx.relative_shift > 0
        assert runtime.adjoint_relative_shift == ctx.relative_shift
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
            from liblaf.apple.inverse._diff_forward import _AdjointProblem

            problem = _AdjointProblem(
                b=-model.dof_map.to_free_grad(grad_output),
                model=model,
                model_state=state,
            )
            initial = runtime.warm_adjoints.get(ctx.key, torch.zeros_like(problem.b))
            if torch.count_nonzero(problem.b) == 0:
                p_free = torch.zeros_like(problem.b)
                shifted_residual = native_shifted_residual = unshifted_residual = 0.0
                shifted_relative = native_shifted_relative = unshifted_relative = 0.0
                result = "zero right-hand side"
                runtime.last_sparse_adjoint = {
                    "method": "hybrid_fem_ipc_free_csr_with_optional_relative_shift",
                    "approximate": True,
                    "relative_shift": runtime.adjoint_relative_shift,
                    "shift": 0.0,
                    "diagonal_scale": None,
                    "absolute_shift": 0.0,
                    "physical_diagonal_abs_mean": None,
                    "shifted_relative_residual": 0.0,
                    "original_unshifted_relative_residual": 0.0,
                    "attempts": [],
                }
            else:
                solution = runtime.solver.solve(problem, initial)
                assert solution.success, runtime.last_sparse_adjoint
                p_free = solution.params.detach()
                sparse = runtime.last_sparse_adjoint
                shifted_residual = float(sparse["shifted_residual_norm"])
                native_shifted_residual = float(sparse["native_shifted_residual_norm"])
                unshifted_residual = float(sparse["original_unshifted_residual_norm"])
                shifted_relative = float(sparse["shifted_relative_residual"])
                native_shifted_relative = float(
                    sparse["native_shifted_relative_residual"]
                )
                unshifted_relative = float(
                    sparse["original_unshifted_relative_residual"]
                )
                assert shifted_relative <= runtime.tolerances["adjoint_rtol"]
                result = str(solution.result)
            runtime.warm_adjoints[ctx.key] = p_free.detach().clone()
            p = model.dof_map.to_full_grad(p_free)
            model.mixed_derivative_prod(state, p)
            gradients = [leaf.grad for leaf in leaves]
            # The shifted free solve changes p only. Mixed material derivatives
            # and the fixed reaction retain the physical, unshifted model Hessian.
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
                "approximate": True,
                "method": "fixed_relative_shift_free_adjoint",
                "relative_shift": runtime.adjoint_relative_shift,
                "shift": runtime.last_sparse_adjoint["shift"],
                "diagonal_scale": runtime.last_sparse_adjoint["diagonal_scale"],
                "residual": max(shifted_residual, native_shifted_residual),
                "relative_residual": max(shifted_relative, native_shifted_relative),
                "shifted_residual_norm": shifted_residual,
                "shifted_relative_residual": shifted_relative,
                "native_shifted_residual_norm": native_shifted_residual,
                "native_shifted_relative_residual": native_shifted_relative,
                "original_unshifted_residual_norm": unshifted_residual,
                # This is measured damping bias, never the shifted-solve gate.
                "original_unshifted_relative_residual": unshifted_relative,
                "original_unshifted_residual_is_bias": True,
                "result": result,
                "seconds": time.perf_counter() - started,
                "key": ctx.key,
            }
        finally:
            model.set_materials(original)
            model.dof_map.fixed_values = original_fixed
        return (None, None, None, fixed_gradient, None, *gradients)


class MouthOpenHybridEquilibrium(Equilibrium):
    """CCD-safe hybrid primal retaining :class:`Equilibrium`'s adjoint."""

    def __init__(
        self,
        *args: Any,
        max_step_norm_m: float,
        collision_step_safety: float = 0.9,
        fixed_stiffness_mpa: float = 0.3386,
        newton_linear_rtol: float = 1e-3,
        newton_max_steps: int = 100,
        adjoint_relative_shift: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            forward_method="pncg",
            newton_linear_rtol=newton_linear_rtol,
            newton_max_steps=newton_max_steps,
            **kwargs,
        )
        assert max_step_norm_m > 0
        assert 0 < collision_step_safety < 1
        assert fixed_stiffness_mpa > 0
        assert adjoint_relative_shift >= 0
        collision = self.forward.model.collision
        assert collision is not None
        self.max_step_norm_m = max_step_norm_m
        self.collision_step_safety = collision_step_safety
        self.fixed_stiffness_mpa = fixed_stiffness_mpa
        self.adjoint_relative_shift = adjoint_relative_shift
        self.last_problem: CachedProblem | None = None
        self.last_sparse_problem: SparseNewtonProblem | None = None
        self.last_sparse_adjoint: dict[str, Any] = {}
        self.deadline: float | None = None

    @override
    def solve(
        self,
        materials: dict,
        fixed_values: torch.Tensor,
        seed: torch.Tensor,
        *,
        key: str,
    ) -> torch.Tensor:
        """Use the pinned exact implicit path unless an explicit shift is requested."""
        if self.adjoint_relative_shift == 0:
            return super().solve(materials, fixed_values, seed, key=key)
        leaves, spec = optree.tree_flatten(materials)
        return _ShiftedImplicit.apply(self, key, spec, fixed_values, seed, *leaves)

    def drop_warm_adjoint(self, key: str) -> None:
        """Discard one incompatible cached adjoint without perturbing other keys."""
        self.warm_adjoints.pop(key, None)

    def _contact_gate(self, state: Any) -> dict[str, Any]:
        collision = self.forward.model.collision
        assert collision is not None
        assert state.collision is not None
        receipt = collision.diagnostics(state.collision, state.u)
        points = (collision.vertices + state.u[collision.indices]).numpy(force=True)
        intersects = bool(
            ipctk.has_intersections(collision.collision_mesh, points, ipctk.LBVH())
        )
        minimum_gap = receipt["minimum_active_distance_m"]
        return {
            "receipt": receipt,
            "no_intersections": not intersects,
            "minimum_active_gap_at_least_buffer": (
                minimum_gap is None or minimum_gap >= collision.min_distance
            ),
        }

    def _restore_fixed_barrier(self) -> None:
        """IPCTK exposes no stiffness getter; rebuild from the recorded κ."""
        collision = self.forward.model.collision
        assert collision is not None
        potential = collision.potential
        collision.potential = ipctk.BarrierPotential(
            type(potential.barrier)(),
            potential.dhat,
            self.fixed_stiffness_mpa,
            collision.use_physical_barrier,
        )

    @override
    def primal(
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """Solve to the original force tolerance using PNCG then sparse Newton."""
        # A trial failure must never be reported as the previous trial's
        # success.  Release the old CSR owner before assembling a new one.
        self.last_forward = {}
        self.last_sparse_problem = None
        self.last_problem = None
        model = self.forward.model
        collision = model.collision
        assert collision is not None
        self._restore_fixed_barrier()
        assert fixed.shape == model.dof_map.fixed_values.shape
        assert seed.shape == self.forward.state.u.shape
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        self.forward.state.u = (
            model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        prior = collision.state_at(seed.detach())
        boundary_change = self.forward.state.u - seed.detach()
        boundary_fraction = float(
            collision.max_step_size(prior, seed.detach(), boundary_change)
        )
        if boundary_fraction < 1.0:
            self.last_forward = {
                "success": False,
                "failure": "Dirichlet boundary proposal fails CCD",
                "contact": {"ccd_boundary_fraction": boundary_fraction},
            }
            raise ForwardConvergenceError(
                self.last_forward["failure"], receipt=self.last_forward
            )
        state = self.forward.state
        state.collision = collision.state_at(state.u)
        initial_contact = self._contact_gate(state)
        if not initial_contact["receipt"]["contact_numerically_valid"]:
            raise ForwardConvergenceError(
                "MouthOpen seed violates contact feasibility", receipt=initial_contact
            )
        delegate = FeasibleExpressionProblem(
            model=model, collision_step_safety=self.collision_step_safety
        )
        remaining = (
            None if self.deadline is None else self.deadline - time.perf_counter()
        )
        if remaining is not None and remaining <= 0:
            raise ForwardConvergenceError("declared forward wall budget exhausted")
        problem = CachedProblem(delegate, exact_curvature=False, wall_seconds=remaining)
        self.last_problem = problem
        started = time.perf_counter()
        try:
            state, pncg = run_pncg_phase(
                problem,
                state,
                atol=self.tolerances["atol"],
                max_step_norm=self.max_step_norm_m,
            )
            if pncg["reason"] != "converged":
                sparse = SparseNewtonProblem(problem)
                self.last_sparse_problem = sparse
                state, newton = safeguarded_newton(
                    sparse,
                    state,
                    atol=self.tolerances["atol"],
                    linear_rtol=self.newton_linear_rtol,
                    linear_max_steps=1000,
                    max_steps=self.newton_max_steps,
                    max_step_norm=self.max_step_norm_m,
                    preconditioner="diag",
                    shift_policy="reuse",
                    reuse_shift_force_ratio=0.0,
                    shift_scale_policy="signed_mean",
                )
            else:
                newton = {"steps": 0, "trace": []}
            terminal_force = float(torch.linalg.vector_norm(problem.grad(state)))
            terminal_contact = self._contact_gate(state)
            success = (
                terminal_force <= self.tolerances["atol"]
                and terminal_contact["receipt"]["contact_numerically_valid"]
                and terminal_contact["no_intersections"]
                and terminal_contact["minimum_active_gap_at_least_buffer"]
            )
            self.last_forward = {
                "success": success,
                "method": "pncg-stall-sparse-newton",
                "seconds": time.perf_counter() - started,
                "grad_norm": terminal_force,
                "force_threshold": self.tolerances["atol"],
                "fixed_stiffness_mpa": self.fixed_stiffness_mpa,
                "pncg": pncg,
                "newton": newton,
                "contact": terminal_contact["receipt"],
                "terminal_gates": {
                    "force": terminal_force <= self.tolerances["atol"],
                    "contact_numerically_valid": terminal_contact["receipt"][
                        "contact_numerically_valid"
                    ],
                    "no_intersections": terminal_contact["no_intersections"],
                    "minimum_active_gap_at_least_buffer": terminal_contact[
                        "minimum_active_gap_at_least_buffer"
                    ],
                },
            }
            if not success:
                raise ForwardConvergenceError(
                    "MouthOpen hybrid terminal gate failed", receipt=self.last_forward
                )
            self.forward_count += 1
            return state.u.detach().clone()
        except ForwardConvergenceError as error:
            if not self.last_forward:
                self.last_forward = {
                    "success": False,
                    "failure": str(error),
                    "receipt": error.receipt,
                }
            physical_energy = float(model.fun(state))
            self.last_forward["failure_state"] = {
                "physical_energy": physical_energy
                if torch.isfinite(torch.as_tensor(physical_energy))
                else None,
                "contact": self._contact_gate(state),
                "rejected_contact_trials": delegate.rejected_contact_trials,
            }
            self.last_failed_displacement = state.u.detach().clone()
            error.receipt = copy.deepcopy(self.last_forward)
            raise


def install_mouthopen_hybrid_runtime(
    physics: Any,
    *,
    forward_atol: float = 1e-8,
    adjoint_rtol: float = 1e-7,
    max_steps: int = 5000,
    max_step_norm_m: float,
    fixed_stiffness_mpa: float = 0.3386,
    collision_step_safety: float = 0.9,
    newton_linear_rtol: float = 1e-3,
    newton_max_steps: int = 100,
    adjoint_relative_shift: float = 0.0,
) -> MouthOpenHybridEquilibrium:
    """Install the fixed-kappa differentiable hybrid runtime on ``physics``."""
    assert forward_atol > 0
    assert 0 < adjoint_rtol < 1
    assert max_steps > 0
    assert adjoint_relative_shift >= 0
    old = physics.runtime
    config = physics.contact_definition["config"]
    assert float(config["stiffness_mpa"]) == fixed_stiffness_mpa
    runtime = MouthOpenHybridEquilibrium(
        old.forward,
        rtol=0.0,
        atol=forward_atol,
        adjoint_rtol=adjoint_rtol,
        max_steps=max_steps,
        max_step_norm_m=max_step_norm_m,
        collision_step_safety=collision_step_safety,
        fixed_stiffness_mpa=fixed_stiffness_mpa,
        newton_linear_rtol=newton_linear_rtol,
        newton_max_steps=newton_max_steps,
        adjoint_relative_shift=adjoint_relative_shift,
    )
    runtime.solver = SparseAdjointSolver(runtime.solver, runtime)
    physics.runtime = runtime
    return runtime


__all__ = ["MouthOpenHybridEquilibrium", "install_mouthopen_hybrid_runtime"]
