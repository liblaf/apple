import abc
from typing import Self

import cytoolz as toolz
import jax
from jaxtyping import Float

from liblaf.apple import math, struct, utils
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.params import GlobalParams


@struct.pytree
class Energy(struct.PyTreeNode, abc.ABC):
    actors: struct.NodeContainer[Actor] = struct.container(factory=struct.NodeContainer)

    @property
    def actor(self) -> Actor:
        return toolz.first(self.actors.values())

    # region Procedure

    def prepare(self, params: GlobalParams) -> Self:  # noqa: ARG002
        return self

    def with_actors(self, actors: struct.NodeContainer[Actor]) -> Self:
        return self.evolve(actors=actors)

    # endregion Procedure

    # region Optimization

    @utils.not_implemented
    @utils.jit_method
    def fun(self, x: struct.ArrayDict, /, params: GlobalParams) -> Float[jax.Array, ""]:
        if utils.implemented(self.fun_and_jac):
            fun, _ = self.fun_and_jac(x, params)
            return fun
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit_method
    def jac(self, x: struct.ArrayDict, /, params: GlobalParams) -> struct.ArrayDict:
        if utils.implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x, params)
            return jac
        if utils.implemented(self.jac_and_hess_diag):
            jac, _ = self.jac_and_hess_diag(x, params)
            return jac
        return jax.grad(self.fun)(x, params)

    @utils.not_implemented
    @utils.jit_method
    def hessp(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: GlobalParams
    ) -> struct.ArrayDict:
        return math.jvp(self.jac)(x, p, params)

    @utils.not_implemented
    @utils.jit_method
    def hess_diag(
        self, x: struct.ArrayDict, /, params: GlobalParams
    ) -> struct.ArrayDict:
        if utils.implemented(self.jac_and_hess_diag):
            _, hess_diag = self.jac_and_hess_diag(x, params)
            return hess_diag
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit_method
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: GlobalParams
    ) -> Float[jax.Array, ""]:
        return math.tree.vdot(self.hessp(x, p, params), p)

    @utils.not_implemented
    @utils.jit_method
    def fun_and_jac(
        self, x: struct.ArrayDict, /, params: GlobalParams
    ) -> tuple[Float[jax.Array, ""], struct.ArrayDict]:
        if utils.implemented(self.fun):
            return self.fun(x, params), self.jac(x, params)
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit_method
    def jac_and_hess_diag(
        self, x: struct.ArrayDict, /, params: GlobalParams
    ) -> tuple[struct.ArrayDict, struct.ArrayDict]:
        return self.jac(x, params), self.hess_diag(x, params)

    # endregion Optimization
