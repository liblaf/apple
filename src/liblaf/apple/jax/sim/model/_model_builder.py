import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.dirichlet import DirichletBuilder
from liblaf.apple.jax.sim.energy import Energy

from ._model import Model


def _default_points() -> Float[Array, "0 J"]:
    return jnp.empty((0, 3))


@tree.pytree
class ModelBuilder:
    dirichlet: DirichletBuilder = tree.field(factory=DirichletBuilder)
    energies: dict[str, Energy] = tree.field(factory=dict)
    points: Float[Array, "p J"] = tree.array(factory=_default_points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def add_dirichlet(self, mesh: pv.DataSet) -> None:
        self.dirichlet.add(mesh)

    def add_energy(self, energy: Energy) -> None:
        self.energies[energy.id] = energy

    def assign_dofs[T: pv.DataSet](self, mesh: T) -> T:
        mesh.point_data["point-ids"] = np.arange(
            self.n_points, self.n_points + mesh.n_points
        )
        self.points = jnp.concat([self.points, mesh.points])
        self.dirichlet.resize(self.n_points)
        return mesh

    def finish(self) -> Model:
        return Model(energies=self.energies)
