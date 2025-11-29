import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float
from liblaf.peach import tree

from liblaf.apple.constants import POINT_ID
from liblaf.apple.jax.model import (
    Dirichlet,
    DirichletBuilder,
    JaxEnergy,
    JaxModelBuilder,
)
from liblaf.apple.warp.model import WarpModelBuilder
from liblaf.apple.warp.model._energy import WarpEnergy

from ._model import Model

type Full = Float[Array, " full"]


@tree.define
class ModelBuilder:
    dirichlet: DirichletBuilder = tree.field(factory=DirichletBuilder)
    jax: JaxModelBuilder = tree.field(factory=JaxModelBuilder)
    warp: WarpModelBuilder = tree.field(factory=WarpModelBuilder)

    def __init__(self, dim: int = 3) -> None:
        dirichlet: DirichletBuilder = DirichletBuilder(dim=dim)
        self.__attrs_init__(dirichlet=dirichlet)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def n_points(self) -> int:
        return self.dirichlet.n_points

    def add_dirichlet(self, obj: pv.DataSet) -> None:
        self.dirichlet.add_pyvista(obj)

    def add_energy(self, energy: JaxEnergy | WarpEnergy) -> None:
        if isinstance(energy, JaxEnergy):
            self.jax.add_energy(energy)
        elif isinstance(energy, WarpEnergy):
            self.warp.add_energy(energy)
        else:
            raise TypeError(energy)

    def assign_global_ids[T: pv.DataSet](self, obj: T) -> T:
        start: int = self.n_points
        stop: int = start + obj.n_points
        self.dirichlet.resize(stop)
        obj.point_data[POINT_ID] = np.arange(start, stop)
        return obj

    def finalize(self) -> Model:
        dirichlet: Dirichlet = self.dirichlet.finalize()
        u_full: Full = jnp.zeros((self.dirichlet.n_points, self.dirichlet.dim))
        u_full = dirichlet.set_dirichlet(u_full)
        return Model(
            dirichlet=dirichlet,
            u_full=u_full,
            jax=self.jax.finalize(),
            warp=self.warp.finalize(),
        )
