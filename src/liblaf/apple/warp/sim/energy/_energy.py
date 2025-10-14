from collections.abc import Sequence

import warp as wp

from liblaf.apple.jax import tree
from liblaf.apple.warp.sparse import Coo2d


@tree.pytree
class Energy(tree.IdMixin):
    requires_grad: Sequence[str] = tree.field(default=(), kw_only=True)

    def fun(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def jac(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess(self, u: wp.array, output: Coo2d, *, start: int) -> None:
        raise NotImplementedError

    def hess_size(self) -> int:
        raise NotImplementedError

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def fun_and_jac(self, u: wp.array, fun: wp.array, jac: wp.array) -> None:
        self.fun(u, fun)
        self.jac(u, jac)

    def jac_and_hess_diag(
        self, u: wp.array, jac: wp.array, hess_diag: wp.array
    ) -> None:
        self.jac(u, jac)
        self.hess_diag(u, hess_diag)

    def mixed_derivative_prod(self, u: wp.array, p: wp.array) -> dict[str, wp.array]:  # noqa: ARG002
        if not self.requires_grad:
            return {}
        raise NotImplementedError
