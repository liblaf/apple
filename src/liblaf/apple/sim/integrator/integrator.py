import abc
from typing import override

import jax
from jaxtyping import Float

from liblaf.apple import math, struct, utils
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.state import State

type X = Float[jax.Array, " DOF"]
type FloatScalar = Float[jax.Array, ""]


class TimeIntegrator(struct.PyTree, math.AutoDiffMixin):
    @property
    def name(self) -> str:
        return type(self).__qualname__

    # region Procedure

    def make_x0(self, state: State, params: GlobalParams) -> X:  # noqa: ARG002
        return state.displacement
        # return state.displacement + state.velocity * params.time_step

    def pre_time_step(self, state: State, params: GlobalParams) -> State:  # noqa: ARG002
        return state

    def pre_optim_iter(self, x: X, /, state: State, params: GlobalParams) -> State:  # noqa: ARG002
        state.update(displacement=x)
        return state

    @abc.abstractmethod
    def step(self, x: X, /, state: State, params: GlobalParams) -> State:
        raise NotImplementedError

    # endregion Procedure

    # region Optimization

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def fun(self, x: X, /, state: State, params: GlobalParams) -> FloatScalar:
        return super().fun(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def jac(self, x: X, /, state: State, params: GlobalParams) -> X:
        return super().jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def hessp(self, x: X, p: X, /, state: State, params: GlobalParams) -> X:
        return super().hessp(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def hess_diag(self, x: X, /, state: State, params: GlobalParams) -> X:
        return super().hess_diag(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: State, params: GlobalParams
    ) -> FloatScalar:
        return super().hess_quad(x, p, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def fun_and_jac(
        self, x: X, /, state: State, params: GlobalParams
    ) -> tuple[FloatScalar, X]:
        return super().fun_and_jac(x, state, params)

    @override
    @utils.not_implemented
    @utils.jit(inline=True)
    def jac_and_hess_diag(
        self, x: X, /, state: State, params: GlobalParams
    ) -> tuple[X, X]:
        return super().jac_and_hess_diag(x, state, params)

    # endregion Optimization
