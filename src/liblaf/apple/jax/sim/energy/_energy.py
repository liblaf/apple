import jax

from liblaf.apple.jax import tree
from liblaf.apple.jax.typing import Scalar, Vector


@tree.pytree
class Energy:
    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    def jac(self, u: Vector) -> Vector:
        return jax.grad(self.fun)(u)

    def value_and_jac(self, u: Vector) -> tuple[Scalar, Vector]:
        return jax.value_and_grad(self.fun)(u)

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        output: Vector
        _, output = jax.jvp(self.jac, (u,), (p,))
        return output
