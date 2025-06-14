from typing import Self

from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct


class Region(struct.Node):
    # region Underlying

    @property
    def geometry(self) -> struct.Node:
        raise NotImplementedError

    # endregion Underlying

    # region Delegation

    @property
    def boundary(self) -> "Region":
        raise NotImplementedError

    # endregion Delegation

    # region FEM

    @property
    def h(self): ...

    @property
    def dhdX(self): ...

    # endregion FEM

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Region":
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, " points J"]) -> Self:
        raise NotImplementedError
