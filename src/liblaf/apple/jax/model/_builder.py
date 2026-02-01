import jarp

from ._energy import JaxEnergy
from ._model import JaxModel


@jarp.define
class JaxModelBuilder:
    energies: dict[str, JaxEnergy] = jarp.field(factory=dict, kw_only=True)

    def add_energy(self, energy: JaxEnergy) -> None:
        self.energies[energy.name] = energy

    def finalize(self) -> JaxModel:
        return JaxModel(energies=self.energies)
