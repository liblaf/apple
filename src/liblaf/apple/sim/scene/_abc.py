from typing import Self

import flax.struct
import jax
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple.sim import energy as _e
from liblaf.apple.sim import field as _f
from liblaf.apple.sim import obj as _o


class Scene(flax.struct.PyTreeNode):
    energies: dict[str, _e.Energy] = flax.struct.field(default_factory=dict)
    fields: dict[str, _f.Field] = flax.struct.field(default_factory=dict)
    objects: dict[str, _o.Object] = flax.struct.field(default_factory=dict)

    # region Optimization

    def fun(self, x: Float[ArrayLike, " free"] | None = None) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def hess_quad(
        self, x: Float[ArrayLike, " free"], p: Float[ArrayLike, " free"]
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    # endregion Optimization

    def make_fields(self, x: Float[ArrayLike, " free"]) -> dict[str, _f.Field]:
        fields: dict[str, _f.Field] = {}
        offset = 0
        for obj in self.objects.values():
            n_free = obj.n_free
            fields[obj.id] = obj.with_free(x[offset : offset + n_free])
            offset += n_free
        return fields

    def resolve_collisions(self) -> Self:
        raise NotImplementedError
