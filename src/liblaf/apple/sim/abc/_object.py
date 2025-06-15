from typing import TYPE_CHECKING, Self, cast, override

from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import math, struct

from ._field import Field

if TYPE_CHECKING:
    from ._filter import Filter


class Object(struct.Node):
    displacement: Field = struct.data(default=None)
    velocity: Field = struct.data(default=None)
    force: Field = struct.data(default=None)

    dof_index: math.Index = struct.data(default=math.make_index())
    origin: "Filter" = struct.data(default=None)

    # region Shape

    @property
    def n_dof(self) -> int:
        return self.displacement.n_dof

    # endregion Shape

    # region Simulation

    def step(
        self,
        displacement: Float[ArrayLike, " DoF"] | None = None,
        velocity: Float[ArrayLike, " DoF"] | None = None,
        force: Float[ArrayLike, " DoF"] | None = None,
    ) -> Self:
        return self.evolve(
            displacement=self.displacement.with_values(displacement),
            velocity=self.velocity.with_values(velocity),
            force=self.force.with_values(force),
        )

    # endregion Simulation

    # region Computational Graph

    @property
    @override
    def networkx_attrs(self) -> struct.NetworkxNodeAttrs:
        attrs: struct.NetworkxNodeAttrs = super().networkx_attrs
        attrs["shape"] = "o"
        return attrs

    @property
    @override
    def refs(self) -> struct.NodeCollection["Filter"]:
        return struct.NodeCollection(self.origin)

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        refs = struct.NodeCollection(refs)
        origin: Filter = refs[self.origin]
        return cast("Self", origin.result)

    # endregion Computational Graph
