import jarp
from frozendict import frozendict

from ._energy import WarpEnergy
from ._model import WarpModel


@jarp.define
class WarpModelBuilder:
    energies: dict[str, WarpEnergy] = jarp.field(factory=dict)

    def add_energy(self, energy: WarpEnergy) -> None:
        self.energies[energy.name] = energy

    def finalize(self) -> WarpModel:
        return WarpModel(energies=frozendict(self.energies))
