import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim import DirichletBuilder
from liblaf.apple.jax.sim import Energy as EnergyJax
from liblaf.apple.jax.sim import Model as ModelJax
from liblaf.apple.warp.sim import Energy as EnergyWarp
from liblaf.apple.warp.sim import Model as ModelWarp

from ._model import Model, _default_points


@tree.pytree
class ModelBuilder:
    dirichlet: DirichletBuilder = tree.field(factory=DirichletBuilder)
    energies_jax: dict[str, EnergyJax] = tree.field(factory=dict)
    energies_warp: dict[str, EnergyWarp] = tree.field(factory=dict)
    points: Float[Array, "p J"] = tree.array(factory=_default_points)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def add_dirichlet(self, mesh: pv.DataSet) -> None:
        self.dirichlet.add(mesh)

    def add_energy(self, energy: EnergyJax | EnergyWarp) -> None:
        if isinstance(energy, EnergyJax):
            self.energies_jax[energy.id] = energy
        elif isinstance(energy, EnergyWarp):
            self.energies_warp[energy.id] = energy
        else:
            raise NotImplementedError

    def assign_dofs[T: pv.DataSet](self, mesh: T) -> T:
        mesh.point_data["point-ids"] = np.arange(
            self.n_points, self.n_points + mesh.n_points
        )
        self.points = jnp.concat([self.points, mesh.points])
        self.dirichlet.resize(self.n_points)
        return mesh

    def finish(self) -> Model:
        return Model(
            points=self.points,
            dirichlet=self.dirichlet.finish(),
            model_jax=self.finish_jax(),
            model_warp=self.finish_warp(),
        )

    def finish_jax(self) -> ModelJax:
        return ModelJax(energies=self.energies_jax)

    def finish_warp(self) -> ModelWarp:
        return ModelWarp(energies=self.energies_warp)
