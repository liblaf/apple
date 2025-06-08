from typing import Self

import flax.struct
import jax
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple.sim import field as _f


class Object(flax.struct.PyTreeNode):
    displacement_prev: _f.Field = flax.struct.field(default=None)
    displacement: _f.Field = flax.struct.field(default=None)
    force: _f.Field = flax.struct.field(default=None)
    velocity: _f.Field = flax.struct.field(default=None)

    # dirichlet boundary conditions
    dirichlet_index: Integer[jax.Array, " dirichlet"] = flax.struct.field(default=None)
    dirichlet_values: Float[jax.Array, " dirichlet"] = flax.struct.field(default=None)
    free_index: Integer[jax.Array, " free"] = flax.struct.field(default=None)

    @classmethod
    def from_field(cls) -> Self:
        return cls()

    # region Inherited

    @property
    def n_dof(self) -> int:
        return self.displacement.n_dof

    # endregion Inherited

    @property
    def n_dirichlet(self) -> int:
        if self.dirichlet_index is None:
            return 0
        return self.dirichlet_index.size

    @property
    def n_free(self) -> int:
        return self.n_dof - self.n_dirichlet

    def with_dirichlet(
        self,
        dirichlet_index: Integer[ArrayLike, " dirichlet"],
        values: Float[jax.Array, " dirichlet"],
    ) -> Self:
        raise NotImplementedError

    def with_free(self, free_values: Float[ArrayLike, " free"]) -> Self:
        raise NotImplementedError
