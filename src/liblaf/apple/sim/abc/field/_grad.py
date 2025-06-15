from collections.abc import Sequence
from typing import override

from ._field import Field


class FieldGrad(Field):
    @property
    @override
    def shape(self) -> Sequence[int]:
        return (self.n_cells, self.quadrature.dim, *self.dim)
