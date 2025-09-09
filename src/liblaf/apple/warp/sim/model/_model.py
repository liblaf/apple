from collections.abc import Mapping

import warp as wp

from liblaf.apple.jax import tree
from liblaf.apple.warp.sim.energy import Energy


@tree.pytree
class Model:
    energies: Mapping[str, Energy] = tree.field(factory=dict)

    def fun(self, u: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.fun(u, output)

    def jac(self, u: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.jac(u, output)

    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.hess_diag(u, output)

    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.hess_prod(u, p, output)

    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.hess_quad(u, p, output)

    def fun_and_jac(self, u: wp.array, fun: wp.array, jac: wp.array) -> None:
        for energy in self.energies.values():
            energy.fun_and_jac(u, fun, jac)

    def jac_and_hess_diag(
        self, u: wp.array, jac: wp.array, hess_diag: wp.array
    ) -> None:
        for energy in self.energies.values():
            energy.jac_and_hess_diag(u, jac, hess_diag)
