import flax.struct
import jax
from jaxtyping import Float

from liblaf.apple import utils
from liblaf.apple.sim import field as _f
from liblaf.apple.sim import obj as _o


class Energy(flax.struct.PyTreeNode):
    @utils.jit
    def fun(self, obj: _o.Object) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        fun, _jac = self.fun_and_jac(obj)
        return fun

    def jac(self, obj: _o.Object) -> Float[jax.Array, " DoF"]:
        jac: Float[jax.Array, " DoF"]
        jac, _hess_diag = self.jac_and_hess_diag(obj)
        return jac

    def hessp(self, obj: _o.Object, p: _f.Field) -> Float[jax.Array, " DoF"]:
        raise NotImplementedError

    def hess_diag(self, obj: _o.Object) -> Float[jax.Array, " DoF"]:
        hess_diag: Float[jax.Array, " DoF"]
        _jac, hess_diag = self.jac_and_hess_diag(obj)
        return hess_diag

    def hess_quad(self, obj: _o.Object, p: _f.Field) -> Float[jax.Array, ""]:
        raise NotImplementedError

    @utils.jit
    def fun_and_jac(
        self, obj: _o.Object
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        return self.fun(obj), self.jac(obj)

    @utils.jit
    def jac_and_hess_diag(
        self, obj: _o.Object
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        jac: Float[jax.Array, " DoF"] = self.jac(obj)
        hess_diag: Float[jax.Array, " DoF"] = self.hess_diag(obj)
        return jac, hess_diag
