from collections.abc import Sequence

import attrs
import warp as wp

from liblaf.apple.common import DEFAULT_POTENTIAL_NAME


@attrs.define
class WarpPotential:
    @wp.struct
    class Materials:
        pass

    materials: Materials = attrs.field(factory=Materials, kw_only=True)
    name: str = attrs.field(default=DEFAULT_POTENTIAL_NAME, kw_only=True)
    requires_grad: Sequence[str] = attrs.field(default=(), kw_only=True)

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
