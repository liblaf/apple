import attrs
import warp as wp

from ._potential import WarpPotential


@attrs.define
class WarpModel:
    potentials: dict[str, WarpPotential] = attrs.field(factory=dict)

    def fun(self, u: wp.array, output: wp.array) -> None:
        for potential in self.potentials.values():
            potential.fun(u, output)

    def grad(self, u: wp.array, output: wp.array) -> None:
        for potential in self.potentials.values():
            potential.grad(u, output)

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        for potential in self.potentials.values():
            potential.hess_diag(u, output)

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        for potential in self.potentials.values():
            potential.hess_prod(u, p, output)

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        for potential in self.potentials.values():
            potential.hess_quad(u, p, output)
