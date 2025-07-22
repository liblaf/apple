from typing import override

import jax.numpy as jnp

from liblaf.apple import utils
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.state import State

from .integrator import FloatScalar, TimeIntegrator, X


class TimeIntegratorStatic(TimeIntegrator):
    @override
    @utils.jit(inline=True)
    def fun(self, x: X, /, state: State, params: GlobalParams) -> FloatScalar:
        return jnp.zeros(())

    @override
    @utils.jit(inline=True)
    def jac(self, x: X, /, state: State, params: GlobalParams) -> X:
        return jnp.zeros_like(x)

    @override
    @utils.jit(inline=True)
    def hess_diag(self, x: X, /, state: State, params: GlobalParams) -> X:
        return jnp.zeros_like(x)

    @override
    @utils.jit(inline=True)
    def hess_quad(
        self, x: X, p: X, /, state: State, params: GlobalParams
    ) -> FloatScalar:
        return jnp.zeros(())
