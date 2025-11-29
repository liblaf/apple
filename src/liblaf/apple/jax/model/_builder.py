from liblaf.peach import tree

from ._energy import JaxEnergy
from ._model import JaxModel


@tree.define
class JaxModelBuilder:
    energies: dict[str, JaxEnergy] = tree.field(factory=dict, kw_only=True)

    def add_energy(self, energy: JaxEnergy) -> None:
        self.energies[energy.id] = energy

    def finalize(self) -> JaxModel:
        return JaxModel(energies=self.energies)
