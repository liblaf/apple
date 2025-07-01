from typing import override

import jax
from jaxtyping import Float

from liblaf.apple import math, struct, utils
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.state import State

type X = Float[jax.Array, " DOF"]
type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class TimeIntegrator(struct.PyTreeMixin, math.AutoDiffMixin):
    # region Procedure

    def make_x0(self, state: State, params: GlobalParams) -> X:
        return state.displacement + state.velocity * params.time_step

    def pre_time_step(self, state: State, params: GlobalParams) -> State:  # noqa: ARG002
        return state

    def pre_optim_iter(self, x: X, /, state: State, params: GlobalParams) -> State:  # noqa: ARG002
        return state.update(displacement=x)

    def step(self, x: X, /, state: State, params: GlobalParams) -> State:  # noqa: ARG002
        return state.update(displacement=x)

    # endregion Procedure

    # region Optimization

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun(self, x: X, /, state: State, params: GlobalParams) -> FloatScalar:
        return super().fun(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac(self, x: X, /, state: State, params: GlobalParams) -> X:
        return super().jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hessp(self, x: X, p: X, /, state: State, params: GlobalParams) -> X:
        return super().hessp(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_diag(self, x: X, /, state: State, params: GlobalParams) -> X:
        return super().hess_diag(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: State, params: GlobalParams
    ) -> FloatScalar:
        return super().hess_quad(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun_and_jac(
        self, x: X, /, state: State, params: GlobalParams
    ) -> tuple[FloatScalar, X]:
        return super().fun_and_jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac_and_hess_diag(
        self, x: X, /, state: State, params: GlobalParams
    ) -> tuple[X, X]:
        return super().jac_and_hess_diag(x, state, params)

    # endregion Optimization
