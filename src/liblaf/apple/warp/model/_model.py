from collections.abc import Mapping

import attrs
import warp as wp
from torch import Tensor

from ._potential import WarpPotential


@attrs.define
class WarpModel:
    potentials: dict[str, WarpPotential] = attrs.field(factory=dict)

    def get_materials(self) -> dict[str, dict[str, wp.array]]:
        return {
            name: potential.get_materials()
            for name, potential in self.potentials.items()
        }

    def set_materials(
        self, materials: Mapping[str, Mapping[str, wp.array | Tensor]]
    ) -> None:
        for name, potential in self.potentials.items():
            if name in materials:
                potential.set_materials(materials[name])

    def require_grad(self, materials: Mapping[str, Mapping[str, bool]]) -> None:
        for name, pot_materials in materials.items():
            self.potentials[name].require_grad(pot_materials)

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

    def mixed_derivative_prod(self, u: wp.array, p: wp.array) -> None:
        output: wp.array = wp.zeros_like(u)
        tape: wp.Tape = wp.Tape()
        with tape:
            self.grad(u, output)
        tape.zero()
        tape.backward(grads={output: p})
