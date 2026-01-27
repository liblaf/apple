import jarp
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float

from liblaf import melon
from liblaf.apple.consts import GLOBAL_POINT_ID
from liblaf.apple.jax import Dirichlet, DirichletBuilder
from liblaf.apple.warp import WarpEnergy, WarpModelAdapter, WarpModelBuilder

from ._model import Model

type Full = Float[Array, "points dim"]


@jarp.define
class ModelBuilder:
    dirichlet: DirichletBuilder = jarp.field(factory=DirichletBuilder)
    edges_length_sum: float = jarp.field(default=0.0)
    n_edges: int = jarp.field(default=0)
    warp: WarpModelBuilder = jarp.field(factory=WarpModelBuilder)

    def __init__(self, dim: int = 3) -> None:
        dirichlet: DirichletBuilder = DirichletBuilder(dim=dim)
        self.__attrs_init__(dirichlet=dirichlet)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def n_points(self) -> int:
        return self.dirichlet.n_points

    def add_dirichlet(self, obj: pv.DataSet) -> None:
        self.dirichlet.add_pyvista(obj)

    def add_energy(self, energy: WarpEnergy) -> None:
        self.warp.add_energy(energy)

    def add_points[T: pv.DataSet](self, obj: T) -> T:
        edges_length: Float[np.ndarray, " edges"] = melon.compute_edges_length(obj)
        self.edges_length_sum += np.sum(edges_length)
        self.n_edges += edges_length.size
        start: int = self.n_points
        stop: int = start + obj.n_points
        self.dirichlet.resize(stop)
        obj.point_data[GLOBAL_POINT_ID] = np.arange(start, stop)
        return obj

    def finalize(self) -> Model:
        dirichlet: Dirichlet = self.dirichlet.finalize()
        u_full: Full = jnp.zeros((self.dirichlet.n_points, self.dirichlet.dim))
        u_full = dirichlet.set_fixed(u_full)
        return Model(
            dirichlet=dirichlet,
            u_full=u_full,
            warp=WarpModelAdapter(self.warp.finalize()),
            edges_length_mean=jnp.asarray(self.edges_length_sum / max(1, self.n_edges)),
        )
