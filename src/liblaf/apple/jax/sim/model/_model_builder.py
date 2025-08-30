import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy import Energy

from ._model import Model


def _default_points() -> Float[Array, "0 J"]:
    return jnp.empty((0, 3))


@tree.pytree
class ModelBuilder:
    energies: list[Energy] = tree.field(factory=list)
    points: Float[Array, "p J"] = tree.array(factory=_default_points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def add_energy(self, energy: Energy) -> None:
        self.energies.append(energy)

    def assign_dofs[T: pv.DataSet](self, mesh: T) -> T:
        mesh.point_data["point-id"] = np.arange(
            self.n_points, self.n_points + mesh.n_points
        )
        self.points = jnp.concat([self.points, mesh.points])
        return mesh

    def finish(self) -> Model:
        raise NotImplementedError
