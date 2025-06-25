from typing import override

import jax.numpy as jnp

from liblaf.apple import struct, utils
from liblaf.apple.sim.params import GlobalParams

from .integrator import FloatScalar, SceneState, TimeIntegrator, X


@struct.pytree
class ImplicitEuler(TimeIntegrator):
    # region Procedure

    @override
    def pre_time_step(self, state: SceneState, params: GlobalParams) -> SceneState:
        return state.update(x_prev=state.displacement)

    @override
    def pre_optim_iter(
        self, x: X, /, state: SceneState, params: GlobalParams
    ) -> SceneState:
        return super().pre_optim_iter(x, state, params)

    @override
    def step(self, x: X, /, state: SceneState, params: GlobalParams) -> SceneState:
        velocity: X = (x - state.x_prev) / params.time_step
        return state.update(displacement=x, velocity=velocity)

    # endregion Procedure

    # region Optimization

    @override
    @utils.jit_method(inline=True)
    def fun(self, x: X, /, state: SceneState, params: GlobalParams) -> FloatScalar:
        x_tilde: X = self.x_tilde(state=state, params=params)
        return (
            0.5
            * jnp.vdot(x - x_tilde, state.mass * (x - x_tilde))
            / params.time_step**2
        )

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: X, /, state: SceneState, params: GlobalParams) -> X:
        x_tilde: X = self.x_tilde(state=state, params=params)
        return state.mass * (x - x_tilde) / params.time_step**2

    @override
    @utils.jit_method(inline=True)
    def hess_diag(self, x: X, /, state: SceneState, params: GlobalParams) -> X:
        return state.mass / params.time_step**2

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: SceneState, params: GlobalParams
    ) -> FloatScalar:
        return jnp.vdot(p, state.mass * p) / params.time_step**2

    # endregion Optimization

    def x_tilde(self, state: SceneState, params: GlobalParams) -> X:
        return (
            state.x_prev
            + params.time_step * state.velocity
            + params.time_step**2 * state.force / state.mass
        )
