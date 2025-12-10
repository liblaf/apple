from liblaf.peach import tree

from ._energy import WarpEnergy
from ._model import WarpModel


@tree.define
class WarpModelBuilder:
    energies: dict[str, WarpEnergy] = tree.field(factory=dict)
    dim: int = tree.field(default=3, kw_only=True)

    def add_energy(self, energy: WarpEnergy) -> None:
        self.energies[energy.id] = energy

    def finalize(self) -> WarpModel:
        return WarpModel(dim=self.dim, energies=self.energies)
