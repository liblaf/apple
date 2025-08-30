from collections.abc import Iterator, Mapping, MutableMapping

import pyvista as pv
from jaxtyping import Array, ArrayLike

from liblaf.apple.jax import math, tree


@tree.pytree
class GeometryAttributes(MutableMapping[str, Array]):
    association: pv.FieldAssociation = tree.field()
    data: dict[str, Array] = tree.field(factory=dict)

    def __getitem__(self, key: str, /) -> Array:
        return self.data[key]

    def __setitem__(self, key: str, value: Array, /) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)


def as_array_dict(m: Mapping[str, ArrayLike], /) -> dict[str, Array]:
    return {k: math.asarray(v) for k, v in m.items()}
