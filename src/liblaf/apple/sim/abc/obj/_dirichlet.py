import attrs
import jax
from jaxtyping import Float

from liblaf.apple import struct
from liblaf.apple.sim.abc.field import Field


class Dirichlet(struct.PyTree):
    index: struct.Index = struct.data(
        default=None, converter=attrs.converters.optional(struct.as_index)
    )
    values: Float[jax.Array, " dirichlet"] = struct.array(default=None)

    def apply(self, x: Field, /) -> Field:
        return x.with_values(self.index.set(x.values, self.values))

    def zero(self, x: Field, /) -> Field:
        return x.with_values(self.index.set(x.values, 0.0))
