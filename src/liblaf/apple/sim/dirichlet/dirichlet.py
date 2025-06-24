from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Shaped

from liblaf.apple import struct
from liblaf.apple.sim.dofs import DOFs


@struct.pytree
class Dirichlet(struct.PyTreeMixin):
    dofs: DOFs = struct.data(default=None)
    values: Shaped[jax.Array, " dirichlet"] = struct.array(default=None)

    @classmethod
    def union(cls, *dirichlet: Self) -> Self:
        dofs: DOFs = DOFs.union(*(d.dofs for d in dirichlet if d.dofs is not None))
        values: Shaped[jax.Array, " dirichlet"] = jnp.concat(
            [jnp.asarray(d.values).ravel() for d in dirichlet if d.values is not None]
        )
        return cls(dofs=dofs, values=values)

    @property
    def size(self) -> int:
        if self.dofs is None:
            return 0
        return self.dofs.size

    def apply(self, x: ArrayLike, /) -> jax.Array:
        if self.dofs is None:
            return jnp.asarray(x)
        return self.dofs.set(x, self.values)

    def zero(self, x: ArrayLike, /) -> jax.Array:
        if self.dofs is None:
            return jnp.asarray(x)
        return self.dofs.set(x, 0.0)
