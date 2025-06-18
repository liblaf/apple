from typing import override

import jax
from jaxtyping import Integer

from liblaf.apple.struct.tree_util import array

from ._dof_map import DofMap


class DofMapInteger(DofMap):
    _integer: Integer[jax.Array, "..."] = array(default=None)

    @property
    @override
    def integers(self) -> Integer[jax.Array, "..."]:
        return self._integer
