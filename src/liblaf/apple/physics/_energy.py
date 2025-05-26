import flax.struct
import jax
from jaxtyping import Float

from ._field import Field


class Energy(flax.struct.PyTreeNode):
    field_id: str = flax.struct.field(pytree_node=False, default="field")
    id: str = flax.struct.field(pytree_node=False, default="energy")

    def fun(self, field: Field) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        fun, _jac = self.fun_jac(field)
        return fun

    def jac(self, field: Field) -> Float[jax.Array, " DoF"]:
        jac: Float[jax.Array, " DoF"]
        _fun, jac = self.fun_jac(field)
        return jac

    def hessp(self, field: Field, p: Field) -> Float[jax.Array, " DoF"]: ...

    def hess_diag(self, field: Field) -> Float[jax.Array, " DoF"]:
        hess_diag: Float[jax.Array, " DoF"]
        _jac, hess_diag = self.jac_hess_diag(field)
        return hess_diag

    def hess_quad(self, field: Field, p: Field) -> Float[jax.Array, ""]: ...

    def fun_jac(
        self, field: Field
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        return self.fun(field), self.jac(field)

    def jac_hess_diag(
        self, field: Field
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        return self.jac(field), self.hess_diag(field)
