import attrs
import pyvista as pv

from liblaf.apple.collision import Collision, CollisionBuilder
from liblaf.apple.warp.model import WarpModel, WarpModelAdapter, WarpPotential

from ._model import Model
from .dof_map import DofMap, DofMapBuilder


@attrs.define
class ModelBuilder:
    collision: CollisionBuilder | None = attrs.field(default=None)
    dof: DofMapBuilder = attrs.field(factory=DofMapBuilder)
    potentials: list[WarpPotential] = attrs.field(factory=list)

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
            {potential.name: potential for potential in self.potentials}
        )
        warp_model_adapter: WarpModelAdapter = WarpModelAdapter(warp_model)
        return Model(
            dof_map=dof_map, warp_model=warp_model_adapter, collision=collision
        )
