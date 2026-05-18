import warp as wp
from frozendict import frozendict

from liblaf import jarp

from ._potential import WarpPotential


@jarp.frozen_static
class WarpModel:
    potentials: frozendict[str, WarpPotential] = jarp.static()

    def fun(self, u: wp.array, output: wp.array) -> None:
        output.zero_()
        for potential in self.potentials.values():
            potential.fun(u, output)

    def grad(self, u: wp.array, output: wp.array) -> None:
        output.zero_()
        for potential in self.potentials.values():
            potential.grad(u, output)

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        output.zero_()
        for potential in self.potentials.values():
            potential.hess_diag(u, output)

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        output.zero_()
        for potential in self.potentials.values():
            potential.hess_prod(u, p, output)

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        output.zero_()
        for potential in self.potentials.values():
            potential.hess_quad(u, p, output)
