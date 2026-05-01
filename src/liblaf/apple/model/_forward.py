from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import attrs
import jarp
import jax.numpy as jnp
from frozendict import frozendict
from jaxtyping import Array, Float
from liblaf.peach.optim import Optimizer

from liblaf.apple.optim import (
    PNCG,
    ConvergenceCriteria,
    HessianDamping,
    LineSearchSettings,
    NonlinearStageObjective,
    PNCGOverrides,
    PNCGResult,
)

from ._model import Model, ModelState
from ._types import (
    Free,
    Full,
    MaterialReference,
    MaterialValues,
    ModelMaterials,
    Scalar,
)

logger: logging.Logger = logging.getLogger(__name__)


@jarp.define
class Objective(NonlinearStageObjective[ModelState]):
    model: Model

    def update_state(self, parameters: Free) -> ModelState:
        u_full: Full = self.model.dirichlet.to_full(parameters)
        return self.model.init_state(u_full)

    def update(self, state: ModelState, u_free: Free) -> ModelState:
        del state
        u_full: Full = self.model.dirichlet.to_full(u_free)
        return self.model.init_state(u_full)

    def objective_value(self, state: ModelState) -> Scalar:
        return self.model.fun(state)

    def fun(self, state: ModelState) -> Scalar:
        return self.objective_value(state)

    def gradient(self, state: ModelState) -> Free:
        grad_full: Full = self.model.grad(state)
        return self.model.dirichlet.get_free(grad_full)

    def grad(self, state: ModelState) -> Free:
        return self.gradient(state)

    def hessian_diagonal(self, state: ModelState) -> Free:
        hess_diag_full: Full = self.model.hess_diag(state)
        return self.model.dirichlet.get_free(hess_diag_full)

    def hess_diag(self, state: ModelState) -> Free:
        return self.hessian_diagonal(state)

    def hess_prod(self, state: ModelState, p_free: Free) -> Free:
        p_full: Full = self.model.dirichlet.to_full(p_free, dirichlet=0.0)
        hess_prod_full: Full = self.model.hess_prod(state, p_full)
        return self.model.dirichlet.get_free(hess_prod_full)

    def curvature_along(self, state: ModelState, direction: Free) -> Scalar:
        p_full: Full = self.model.dirichlet.to_full(direction, dirichlet=0.0)
        return self.model.hess_quad(state, p_full)

    def hess_quad(self, state: ModelState, p_free: Free) -> Scalar:
        return self.curvature_along(state, p_free)


@attrs.frozen
class StageState:
    dirichlet_values: Full | None = None
    material_values: MaterialValues = attrs.field(factory=frozendict)


class StageStateProgram(Protocol):
    def state_at(self, *, progress: float, forward: Forward) -> StageState: ...


@attrs.frozen
class ForwardStage:
    name: str
    progress: float
    initial_guess: Literal["current", "last_successful", "zero"] = "last_successful"
    solver_overrides: PNCGOverrides | None = None


@attrs.frozen
class ForwardStageResult:
    stage: ForwardStage
    solver_result: PNCGResult


@attrs.frozen
class ForwardResult:
    success: bool
    status: Literal["completed", "stage_failed", "plan_failed"]
    final_stage: ForwardStage | None
    final_solver_result: PNCGResult | None
    stage_results: Sequence[ForwardStageResult]


class ForwardPlan(Protocol):
    state_program: StageStateProgram

    def first_stage(self, forward: Forward) -> ForwardStage | None: ...

    def next_stage(
        self,
        *,
        forward: Forward,
        previous_stage: ForwardStage,
        previous_result: ForwardStageResult,
    ) -> ForwardStage | None: ...


@attrs.frozen
class IdentityStageStateProgram:
    def state_at(self, *, progress: float, forward: Forward) -> StageState:
        del progress, forward
        return StageState()


@attrs.frozen
class SingleStagePlan:
    state_program: StageStateProgram = attrs.field(factory=IdentityStageStateProgram)
    stage: ForwardStage = attrs.field(
        factory=lambda: ForwardStage(name="solve", progress=1.0)
    )

    def first_stage(self, forward: Forward) -> ForwardStage | None:
        del forward
        return self.stage

    def next_stage(
        self,
        *,
        forward: Forward,
        previous_stage: ForwardStage,
        previous_result: ForwardStageResult,
    ) -> ForwardStage | None:
        del forward, previous_stage, previous_result
        return None


@attrs.frozen
class ExplicitStagePlan:
    state_program: StageStateProgram
    stages: tuple[ForwardStage, ...]

    def first_stage(self, forward: Forward) -> ForwardStage | None:
        del forward
        if not self.stages:
            return None
        return self.stages[0]

    def next_stage(
        self,
        *,
        forward: Forward,
        previous_stage: ForwardStage,
        previous_result: ForwardStageResult,
    ) -> ForwardStage | None:
        del forward, previous_result
        try:
            index = self.stages.index(previous_stage)
        except ValueError:
            return None
        if index + 1 >= len(self.stages):
            return None
        return self.stages[index + 1]


@attrs.define
class AdaptiveContinuationPlan:
    state_program: StageStateProgram
    initial_progress_step: float = 0.1
    minimum_progress_step: float = 0.01
    maximum_progress_step: float = 0.25
    progress_growth_factor: float = 1.5
    progress_shrink_factor: float = 0.5
    maximum_stage_attempts: int = 5

    _base_progress: float = attrs.field(init=False, default=0.0)
    _current_step: float = attrs.field(init=False, default=0.1)
    _attempts_at_base: int = attrs.field(init=False, default=0)
    _stage_index: int = attrs.field(init=False, default=0)

    def first_stage(self, forward: Forward) -> ForwardStage | None:
        del forward
        self._base_progress = 0.0
        self._current_step = min(self.initial_progress_step, self.maximum_progress_step)
        self._attempts_at_base = 0
        self._stage_index = 0
        return self._make_stage(self._base_progress + self._current_step)

    def next_stage(
        self,
        *,
        forward: Forward,
        previous_stage: ForwardStage,
        previous_result: ForwardStageResult,
    ) -> ForwardStage | None:
        del forward
        if previous_result.solver_result.success:
            self._base_progress = previous_stage.progress
            if self._base_progress >= 1.0:
                return None
            self._attempts_at_base = 0
            self._current_step = min(
                self.maximum_progress_step,
                self._current_step * self.progress_growth_factor,
            )
            return self._make_stage(self._base_progress + self._current_step)
        self._attempts_at_base += 1
        self._current_step *= self.progress_shrink_factor
        if self._attempts_at_base >= self.maximum_stage_attempts:
            return None
        if self._current_step < self.minimum_progress_step:
            return None
        return self._make_stage(self._base_progress + self._current_step)

    def _make_stage(self, progress: float) -> ForwardStage:
        progress = min(1.0, progress)
        stage = ForwardStage(name=f"stage-{self._stage_index}", progress=progress)
        self._stage_index += 1
        return stage


@jarp.define
class Forward:
    model: Model

    def _default_state(self) -> ModelState:
        return self.model.init_state(self.model.u_full)

    state: ModelState = jarp.field(
        default=attrs.Factory(_default_state, takes_self=True), kw_only=True
    )

    @staticmethod
    def _build_default_solver(model: Model) -> Optimizer:
        max_steps: int = max(1000, jnp.ceil(20 * jnp.sqrt(model.n_free)).item())
        maximum_step_inf_norm: Scalar = (
            0.15 * model.edges_length_mean
            if model.edges_length_mean > 0
            else jnp.asarray(jnp.inf)
        )
        return PNCG(
            convergence=ConvergenceCriteria(
                target_absolute_gradient_norm=jnp.asarray(1e-10),
                target_relative_gradient_norm=jnp.asarray(1e-5),
                acceptable_absolute_gradient_norm=jnp.asarray(1e-10),
                acceptable_relative_gradient_norm=jnp.asarray(1e-3),
                maximum_iterations=jnp.asarray(max_steps, dtype=jnp.int32),
            ),
            line_search=LineSearchSettings(
                armijo_sufficient_decrease=jnp.asarray(0.0),
                backtracking_factor=jnp.asarray(0.5),
                maximum_backtracking_steps=jnp.asarray(20, dtype=jnp.int32),
                maximum_step_inf_norm=maximum_step_inf_norm,
            ),
            damping=HessianDamping(constant_strength=jnp.asarray(0.0)),
            stagnation_max_restarts=jnp.asarray(40, dtype=jnp.int32),
            stagnation_patience=jnp.asarray(40, dtype=jnp.int32),
            use_jit=False,
        )

    def _default_solver(self) -> Optimizer:
        return self._build_default_solver(self.model)

    solver: Optimizer = jarp.field(
        default=attrs.Factory(_default_solver, takes_self=True), kw_only=True
    )

    last_result: ForwardResult | None = jarp.field(default=None, kw_only=True)
    last_stage_result: ForwardStageResult | None = jarp.field(
        default=None, kw_only=True
    )
    last_successful_stage: ForwardStage | None = jarp.field(default=None, kw_only=True)
    last_successful_u_free: Free | None = jarp.field(default=None, kw_only=True)
    last_successful_material_values: MaterialValues | None = jarp.field(
        default=None, kw_only=True
    )
    _last_successful_dirichlet_values: Full | None = jarp.field(
        default=None, kw_only=True, repr=False
    )

    def __init__(
        self,
        model: Model,
        solver: Optimizer | None = None,
        *,
        optimizer: Optimizer | None = None,
        state: ModelState | None = None,
    ) -> None:
        if solver is not None and optimizer is not None:
            raise TypeError("pass only one of `solver` or legacy `optimizer`")
        solver = solver or optimizer or self._build_default_solver(model)
        state = state or model.init_state(model.u_full)
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            model=model,
            state=state,
            solver=solver,
        )

    @property
    def optimizer(self) -> Optimizer:
        return self.solver

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self.solver = optimizer

    @property
    def u_free(self) -> Free:
        return self.model.u_free

    @property
    def u_full(self) -> Float[Array, "points dim"]:
        return self.model.u_full

    @property
    def has_successful_stage(self) -> bool:
        return (
            self.last_successful_u_free is not None
            and self.last_successful_material_values is not None
        )

    def solve(
        self,
        plan: ForwardPlan | None = None,
        callback: PNCG.Callback[ModelState] | None = None,
    ) -> ForwardResult:
        if not isinstance(self.solver, PNCG):
            raise TypeError("Forward.solve requires liblaf.apple.optim.PNCG")
        plan = plan or SingleStagePlan()
        stage_results: list[ForwardStageResult] = []
        stage = plan.first_stage(self)
        if stage is None:
            result = ForwardResult(
                success=True,
                status="completed",
                final_stage=None,
                final_solver_result=None,
                stage_results=tuple(),
            )
            self.last_result = result
            return result

        while stage is not None:
            stage_result = self.solve_stage(
                stage, state_program=plan.state_program, callback=callback
            )
            stage_results.append(stage_result)
            next_stage = plan.next_stage(
                forward=self,
                previous_stage=stage,
                previous_result=stage_result,
            )
            if (
                next_stage is not None
                and not stage_result.solver_result.success
                and self.has_successful_stage
            ):
                self.restore_last_successful_stage()
            if next_stage is None:
                result = ForwardResult(
                    success=stage_result.solver_result.success,
                    status=(
                        "completed"
                        if stage_result.solver_result.success
                        else "stage_failed"
                    ),
                    final_stage=stage,
                    final_solver_result=stage_result.solver_result,
                    stage_results=tuple(stage_results),
                )
                self.last_result = result
                return result
            stage = next_stage

        result = ForwardResult(
            success=False,
            status="plan_failed",
            final_stage=None,
            final_solver_result=None,
            stage_results=tuple(stage_results),
        )
        self.last_result = result
        return result

    def solve_stage(
        self,
        stage: ForwardStage,
        *,
        state_program: StageStateProgram,
        callback: PNCG.Callback[ModelState] | None = None,
    ) -> ForwardStageResult:
        if not isinstance(self.solver, PNCG):
            raise TypeError("Forward.solve_stage requires liblaf.apple.optim.PNCG")
        stage_state = state_program.state_at(progress=stage.progress, forward=self)
        self._apply_stage_state(stage_state)
        self._set_initial_guess(stage.initial_guess)
        objective = Objective(model=self.model)
        solver_result = self.solver.solve(
            objective,
            initial_parameters=self.model.u_free,
            overrides=stage.solver_overrides,
            callback=callback,
        )
        logger.info("%s", solver_result)
        self.model.u_free = solver_result.final_iteration.parameters
        self.state = self.model.init_state(self.model.u_full)
        stage_result = ForwardStageResult(stage=stage, solver_result=solver_result)
        self.last_stage_result = stage_result
        if solver_result.success:
            self.last_successful_stage = stage
            self.last_successful_u_free = self.model.u_free
            self.last_successful_material_values = self.read_material_values()
            self._last_successful_dirichlet_values = self._current_dirichlet_values()
        return stage_result

    def read_material_values(self) -> dict[MaterialReference, Array]:
        return self.model.read_material_values()

    def write_material_values(self, values: MaterialValues) -> None:
        self.model.write_material_values(values)

    def restore_last_successful_stage(self) -> None:
        if not self.has_successful_stage:
            raise RuntimeError("no successful stage is available to restore")
        assert self.last_successful_u_free is not None
        assert self.last_successful_material_values is not None
        if self._last_successful_dirichlet_values is not None:
            self._set_dirichlet_values(self._last_successful_dirichlet_values)
        self.write_material_values(self.last_successful_material_values)
        self.model.u_free = self.last_successful_u_free
        self.state = self.model.init_state(self.model.u_full)

    def update_materials(self, materials: ModelMaterials | MaterialValues) -> None:
        if materials and isinstance(next(iter(materials.keys())), MaterialReference):
            self.write_material_values(materials)
            return
        self.model.update_materials(materials)

    def step(
        self,
        callback: Optimizer.Callback[ModelState, Any, Any] | None = None,
        *,
        logging: bool = True,
    ) -> Optimizer.Solution:
        objective = Objective(model=self.model)
        solution: Optimizer.Solution
        solution, self.state = self.solver.minimize(
            objective, self.state, self.model.u_free, callback=callback
        )
        self.model.u_free = solution.params
        if isinstance(self.solver, PNCG):
            stage = ForwardStage(name="step", progress=1.0, initial_guess="current")
            self.last_stage_result = ForwardStageResult(
                stage=stage, solver_result=self.solver._to_result(solution)
            )
            if solution.success:
                self.last_successful_stage = stage
                self.last_successful_u_free = self.model.u_free
                self.last_successful_material_values = self.read_material_values()
                self._last_successful_dirichlet_values = (
                    self._current_dirichlet_values()
                )
        if logging:
            if solution.success:
                logger.info("Forward success: %r", solution.stats)
            else:
                logger.warning("Forward fail: %r", solution)
        return solution

    def _apply_stage_state(self, stage_state: StageState) -> None:
        if stage_state.dirichlet_values is not None:
            self._set_dirichlet_values(stage_state.dirichlet_values)
        if stage_state.material_values:
            self.write_material_values(stage_state.material_values)
        self.model.u_full = self.model.dirichlet.set_fixed(self.model.u_full)
        self.state = self.model.init_state(self.model.u_full)

    def _set_initial_guess(
        self, initial_guess: Literal["current", "last_successful", "zero"]
    ) -> None:
        if initial_guess == "zero":
            u_free = jnp.zeros_like(self.model.u_free)
        elif (
            initial_guess == "last_successful"
            and self.last_successful_u_free is not None
            and self.last_successful_u_free.shape == self.model.u_free.shape
        ):
            u_free = self.last_successful_u_free
        else:
            u_free = self.model.u_free
        self.model.u_free = u_free
        self.state = self.model.init_state(self.model.u_full)

    def _set_dirichlet_values(self, values: Full) -> None:
        self.model.dirichlet.dirichlet_value = self.model.dirichlet.get_fixed(values)
        self.model.u_full = self.model.dirichlet.set_fixed(self.model.u_full)

    def _current_dirichlet_values(self) -> Full:
        return self.model.dirichlet.set_fixed(jnp.zeros_like(self.model.u_full))
