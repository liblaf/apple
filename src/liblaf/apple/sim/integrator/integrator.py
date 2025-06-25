from typing import override

import jax
from jaxtyping import Float

from liblaf.apple import math, struct, utils
from liblaf.apple.sim.params import GlobalParams

type X = Float[jax.Array, " DOF"]
type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class SceneState(struct.ArrayDict):
    @property
    def displacement(self) -> X:
        return self["displacement"]

    @property
    def velocity(self) -> X:
        return self["velocity"]

    @property
    def force(self) -> X:
        return self["force"]

    @property
    def mass(self) -> X:
        return self["mass"]

    @property
    def x_prev(self) -> X:
        return self["x_prev"]


@struct.pytree
class TimeIntegrator(struct.PyTreeMixin, math.AutoDiffMixin):
    # region Procedure

    def pre_time_step(
        self,
        state: SceneState,
        params: GlobalParams,  # noqa: ARG002
    ) -> SceneState:
        return state

    def pre_optim_iter(
        self,
        x: X,
        /,
        state: SceneState,
        params: GlobalParams,  # noqa: ARG002
    ) -> SceneState:
        return state.update(displacement=x)

    def step(
        self,
        x: X,
        /,
        state: SceneState,
        params: GlobalParams,  # noqa: ARG002
    ) -> SceneState:
        return state.update(displacement=x)

    # endregion Procedure

    # region Optimization

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun(self, x: X, /, state: SceneState, params: GlobalParams) -> FloatScalar:
        return super().fun(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac(self, x: X, /, state: SceneState, params: GlobalParams) -> X:
        return super().jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hessp(self, x: X, p: X, /, state: SceneState, params: GlobalParams) -> X:
        return super().hessp(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_diag(self, x: X, /, state: SceneState, params: GlobalParams) -> X:
        return super().hess_diag(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: SceneState, params: GlobalParams
    ) -> FloatScalar:
        return super().hess_quad(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun_and_jac(
        self, x: X, /, state: SceneState, params: GlobalParams
    ) -> tuple[FloatScalar, X]:
        return super().fun_and_jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac_and_hess_diag(
        self, x: X, /, state: SceneState, params: GlobalParams
    ) -> tuple[X, X]:
        return super().jac_and_hess_diag(x, state, params)

    # endregion Optimization
