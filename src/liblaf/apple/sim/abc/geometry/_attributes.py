from collections.abc import Iterator, MutableMapping

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

from liblaf.apple import struct


class GeometryAttributes(struct.PyTree, MutableMapping[str, jax.Array]):
    attributes: pv.DataSetAttributes = struct.static(default=None)

    def __getitem__(self, key: str) -> jax.Array:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.attributes[key])

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        self.attributes[key] = np.asarray(value)

    def __delitem__(self, key: str) -> None:
        del self.attributes[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.attributes.keys()

    def __len__(self) -> int:
        return len(self.attributes)
