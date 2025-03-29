import attrs
import jax
from jaxtyping import Float, PyTree

from liblaf import apple

from . import MaterialTetra


@apple.register_dataclass()
@attrs.define(kw_only=True)
class Corotated(MaterialTetra):
    def fun(self, u: Float[jax.Array, "C 4 3"], aux: PyTree) -> Float[jax.Array, ""]:
        return None

    def jac(self, u: Float[jax.Array, "P 3"]) -> Float[jax.Array, "P 3"]:
        return None

    def hess(self, u: Float[jax.Array, "C 4 3"]) -> Float[jax.Array, "C 12 12"]: ...

    def _fun_elem(
        self, u: Float[jax.Array, "4 3"], aux: PyTree
    ) -> Float[jax.Array, ""]: ...
