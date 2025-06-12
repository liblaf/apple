from typing import cast

import jax
from jaxtyping import Float

from liblaf.apple import struct
from liblaf.apple.sim import field as _f
from liblaf.apple.sim import obj as _o


class Energy(struct.Node):
    @property
    def object(self) -> _o.Object:
        return cast("_o.Object", self.ref)

    @property
    def objects(self) -> struct.NodeCollection[_o.Object]:
        return struct.NodeCollection(self.refs)

    def fun(self) -> Float[jax.Array, ""]:
        fun, _jac = self.fun_and_jac()
        return fun

    def jac(self) -> Float[jax.Array, " DoF"]:
        jac, _hess_diag = self.jac_and_hess_diag()
        return jac

    def hessp(self, p: struct.NodeCollection[_f.Field]) -> Float[jax.Array, " DoF"]:
        raise NotImplementedError

    def hess_diag(self) -> Float[jax.Array, " DoF"]:
        _jac, hess_diag = self.jac_and_hess_diag()
        return hess_diag

    def hess_quad(self, p: struct.NodeCollection[_f.Field]) -> Float[jax.Array, " DoF"]:
        raise NotImplementedError

    def fun_and_jac(self) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        return self.fun(), self.jac()

    def jac_and_hess_diag(
        self,
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        return self.jac(), self.hess_diag()
