from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Shaped
from typing_extensions import deprecated

from liblaf.apple import struct
from liblaf.apple.sim.dofs import DOFs


@struct.pytree
class Dirichlet(struct.PyTreeMixin):
    dofs: DOFs = struct.data(default=None)
    values: Shaped[Array, " dirichlet"] = struct.array(default=None)

    @classmethod
    def from_mask(cls, mask: ArrayLike, values: ArrayLike) -> Self:
        mask = jnp.asarray(mask)
        if not mask.any():
            return cls()
        values = jnp.asarray(values)
        values = jnp.broadcast_to(values, mask.shape)
        mask = mask.ravel()
        values = values.ravel()
        dofs: DOFs = DOFs.from_mask(mask)
        values = values[mask]
        return cls(dofs=dofs, values=values)

    @classmethod
    @deprecated("Manually create Dirichlet conditions instead.")
    def union(cls, *dirichlet: Self) -> Self:
        """...

        Note:
            Dirichlet conditions can only be merged if they are defined on the same DOFs. To avoid unexpected behavior, we do not implement this method.
        """
        dirichlet: list[Self] = [d for d in dirichlet if d.dofs is not None]
        if not dirichlet:
            return cls()
        dofs: DOFs = DOFs.union(*(d.dofs for d in dirichlet))
        values: Shaped[Array, " dirichlet"] = jnp.concat(
            [jnp.asarray(d.values).ravel() for d in dirichlet]
        )
        return cls(dofs=dofs, values=values)

    @property
    def size(self) -> int:
        if self.dofs is None:
            return 0
        return self.dofs.size

    def apply(self, x: ArrayLike, /) -> Array:
        if self.dofs is None:
            return jnp.asarray(x)
        return self.dofs.set(x, self.values)

    def mask(self, x: ArrayLike, /) -> Bool[Array, " DOF"]:
        if self.dofs is None:
            return jnp.asarray(x)
        return self.dofs.set(x, True)  # noqa: FBT003

    def zero(self, x: ArrayLike, /) -> Array:
        if self.dofs is None:
            return jnp.asarray(x)
        return self.dofs.set(x, 0.0)
