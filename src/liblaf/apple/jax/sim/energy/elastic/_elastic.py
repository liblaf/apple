from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy._energy import Energy
from liblaf.apple.jax.sim.region import Region
from liblaf.apple.jax.typing import Scalar, Vector


@tree.pytree
class Elastic(Energy):
    region: Region

    def fun(self, u: Vector) -> Scalar:
        F: Float[Array, "c q J J"] = self.region.deformation_gradient(u)
        Psi: Float[Array, "c q"] = self.energy_density(F)
        return self.region.integrate(Psi).sum()

    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        raise NotImplementedError
