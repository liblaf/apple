import pyvista as pv
from frozendict import frozendict

from liblaf import jarp
from liblaf.apple.collision import Collision, CollisionBuilder
from liblaf.apple.warp.model import WarpModel, WarpModelAdapter, WarpPotential

from ._model import Model
from .dof_map import DofMap, DofMapBuilder


@jarp.define
class ModelBuilder:
    collision: CollisionBuilder | None = jarp.field(default=None)
    dof: DofMapBuilder = jarp.field(factory=DofMapBuilder)
    potentials: list[WarpPotential] = jarp.field(factory=list)

    def add_fixed(self, obj: pv.DataSet) -> None:
        self.dof.add_fixed(obj)

    def add_potential(self, potential: WarpPotential) -> None:
        self.potentials.append(potential)

    def add_vertices(self, obj: pv.DataSet) -> None:
        self.dof.add_vertices(obj)

    def finalize(self) -> Model:
        collision: Collision | None = None
        if self.collision is not None:
            collision: Collision = self.collision.finalize()
        dof_map: DofMap = self.dof.finalize()
        warp_model: WarpModel = WarpModel(
            frozendict({potential.name: potential for potential in self.potentials})
        )
        warp_model_adapter: WarpModelAdapter = WarpModelAdapter(
            warp_model, n_points=dof_map.n_points
        )
        return Model(
            dof_map=dof_map, warp_model=warp_model_adapter, collision=collision
        )
