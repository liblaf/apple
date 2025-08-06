from typing import override

import jax
import jax.numpy as jnp

from liblaf.apple import utils
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.state import State

from .integrator import FloatScalar, TimeIntegrator, X


class ImplicitEuler(TimeIntegrator):
    # region Procedure

    @override
    def pre_time_step(self, state: State, params: GlobalParams) -> State:
        state.update(x_prev=state.displacement)
        return state

    @override
    def pre_optim_iter(self, x: X, /, state: State, params: GlobalParams) -> State:
        state.update(displacement=x)
        return state

    @override
    @utils.jit(inline=True)
    def step(self, x: X, /, state: State, params: GlobalParams) -> State:
        velocity: X = (x - state.x_prev) / params.time_step
        state.update(displacement=x, velocity=velocity)
        return state

    # endregion Procedure

    # region Optimization

    @override
    @utils.jit(inline=True)
    def fun(self, x: X, /, state: State, params: GlobalParams) -> FloatScalar:
        x_tilde: X = self.x_tilde(state=state, params=params)
        return (
            0.5
            * jnp.vdot(x - x_tilde, state.mass * (x - x_tilde))
            / params.time_step**2
        )

    @override
    @utils.jit(inline=True)
    def jac(self, x: X, /, state: State, params: GlobalParams) -> X:
        # jax.debug.print(
        #     "ImplicitEuler.jac: x = {}, x_tilde = {}, mass = {}, time_step = {}",
        #     x,
        #     self.x_tilde(state=state, params=params),
        #     state.mass,
        #     params.time_step,
        # )
        x_tilde: X = self.x_tilde(state=state, params=params)
        return state.mass * (x - x_tilde) / params.time_step**2

    @override
    @utils.jit(inline=True)
    def hess_diag(self, x: X, /, state: State, params: GlobalParams) -> X:
        return state.mass / params.time_step**2

    @override
    @utils.jit(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: State, params: GlobalParams
    ) -> FloatScalar:
        return jnp.vdot(p, state.mass * p) / params.time_step**2

    # endregion Optimization

    def x_tilde(self, state: State, params: GlobalParams) -> X:
        return (
            state.x_prev
            + params.time_step * state.velocity
            + params.time_step**2 * state.force_ext / state.mass
        )
