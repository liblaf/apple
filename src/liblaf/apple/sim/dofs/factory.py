from collections.abc import Sequence

import numpy as np

from .dofs import DOFs
from .range import DOFsRange


def make_dofs(shape: Sequence[int], *, offset: int = 0) -> DOFs:
    return DOFsRange(_range=range(offset, offset + np.prod(shape)), shape=shape)
