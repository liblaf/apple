import abc

import attrs
import jax
from jaxtyping import Float

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class MaterialTetra(abc.ABC):
    @abc.abstractmethod
    def potential(self, x: Float[jax.Array, "C 4 3"]) -> Float[jax.Array, " C"]: ...

    @abc.abstractmethod
    def potential_single(self, u: Float[jax.Array, "C 3"]) -> Float[jax.Array, " C"]:
        return self.potential(u)[0]

    @abc.abstractmethod
    def jac(self, u: Float[jax.Array, "C 4 3"]) -> Float[jax.Array, "C 4 3"]: ...
