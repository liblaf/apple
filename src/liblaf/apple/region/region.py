import jax
import pylops
from jaxtyping import Float


class Region:
    def fun(self, u: Float[jax.Array, " F"]) -> Float[jax.Array, ""]: ...

    def jac(self, u: Float[jax.Array, " F"]) -> Float[jax.Array, " F"]:
        return jax.grad(self.fun)(u)

    def hess(self, u: Float[jax.Array, " F"]) -> Float[pylops.LinearOperator, "F F"]:
        raise NotImplementedError

    def hess_diag(self, u: Float[jax.Array, " F"]) -> Float[jax.Array, " F"]:
        raise NotImplementedError
