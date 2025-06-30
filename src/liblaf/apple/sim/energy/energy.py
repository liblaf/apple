import abc
from typing import Self

import jax
from jaxtyping import Array, Float

from liblaf.apple import math, struct, utils
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.params import GlobalParams


@struct.pytree
class Energy(struct.PyTreeNode, abc.ABC):
    @property
    @abc.abstractmethod
    def actors(self) -> struct.NodeContainer[Actor]:
        raise NotImplementedError

    # region Procedure

    def pre_time_step(self, params: GlobalParams) -> Self:  # noqa: ARG002
        return self

    def pre_optim_iter(self, params: GlobalParams) -> Self:  # noqa: ARG002
        return self

    @abc.abstractmethod
    def with_actors(self, actors: struct.NodeContainer[Actor]) -> Self:
        raise NotImplementedError

    # endregion Procedure

    # region Optimization

    @utils.not_implemented
    @utils.jit_method
    def fun(self, x: struct.ArrayDict, /, params: GlobalParams) -> Float[Array, ""]:
        if utils.is_implemented(self.fun_and_jac):
            fun, _ = self.fun_and_jac(x, params)
            return fun
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit_method
    def jac(self, x: struct.ArrayDict, /, params: GlobalParams) -> struct.ArrayDict:
        if utils.is_implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x, params)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
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
        if utils.is_implemented(self.jac_and_hess_diag):
            _, hess_diag = self.jac_and_hess_diag(x, params)
            return hess_diag
        return math.hess_diag(self.fun)(x, params)

    @utils.not_implemented
    @utils.jit_method
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: GlobalParams
    ) -> Float[Array, ""]:
        return math.tree.tree_vdot(self.hessp(x, p, params), p)

    @utils.not_implemented
    @utils.jit_method
    def fun_and_jac(
        self, x: struct.ArrayDict, /, params: GlobalParams
    ) -> tuple[Float[Array, ""], struct.ArrayDict]:
        return self.fun(x, params), self.jac(x, params)

    @utils.not_implemented
    @utils.jit_method
    def jac_and_hess_diag(
        self, x: struct.ArrayDict, /, params: GlobalParams
    ) -> tuple[struct.ArrayDict, struct.ArrayDict]:
        return self.jac(x, params), self.hess_diag(x, params)

    # endregion Optimization
