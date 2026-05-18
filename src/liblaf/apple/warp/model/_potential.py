from collections.abc import Sequence

import warp as wp

from liblaf import jarp
from liblaf.apple.common import DEFAULT_POTENTIAL_NAME


@jarp.frozen_static
class WarpPotential:
    @wp.struct
    class Materials:
        pass

    name: str = jarp.static(default=DEFAULT_POTENTIAL_NAME, kw_only=True)
    materials: Materials = jarp.static()
    requires_grad: Sequence[str] = jarp.static(default=(), kw_only=True)

    def fun(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def grad(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError
