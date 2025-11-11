from collections.abc import Mapping

import warp as wp
from liblaf.peach import tree

from liblaf.apple.warp.sim.energy import Energy
from liblaf.apple.warp.sparse import Coo2d


@tree.define
class Model:
    energies: Mapping[str, Energy] = tree.field(factory=dict)

    def fun(self, u: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.fun(u, output)

    def jac(self, u: wp.array, output: wp.array) -> None:
        for energy in self.energies.values():
            energy.jac(u, output)

    def hess(self, u: wp.array, output: Coo2d) -> None:
        start: int = 0
        for energy in self.energies.values():
            energy.hess(u, output, start=start)
            start += energy.hess_size()

    def hess_size(self) -> int:
        size: int = 0
        for energy in self.energies.values():
            size += energy.hess_size()
        return size

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

    def mixed_derivative_prod(
        self, u: wp.array, p: wp.array
    ) -> dict[str, dict[str, wp.array]]:
        outputs: dict[str, dict[str, wp.array]] = {
            energy.id: energy.mixed_derivative_prod(u, p)
            for energy in self.energies.values()
        }
        return outputs
