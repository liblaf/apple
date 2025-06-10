from typing import Any

import flax.struct
import jax
from jaxtyping import Float


class Scene(flax.struct.PyTreeNode):
    energies: dict[str, Any] = flax.struct.field(
        pytree_node=False, default_factory=dict
    )

    def fun(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def jac(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def hess_diag(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def hess_quad(
        self, x: Float[jax.Array, " N"], p: Float[jax.Array, " N"]
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def jac_and_hess_diag(
        self, x: Float[jax.Array, " N"]
    ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
        return self.jac(x), self.hess_diag(x)
