from typing import Self

import jax
from jaxtyping import Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim import field as _f
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import region as _r


class Object(struct.Node):
    displacement: _f.Field = struct.data(default=None)
    velocity: _f.Field = struct.data(default=None)
    force: _f.Field = struct.data(default=None)

    dirichlet_index: Integer[jax.Array, " dirichlet"] = struct.array(default=None)
    dirichlet_values: Float[jax.Array, " dirichlet"] = struct.array(default=None)
    dof_index: Integer[jax.Array, " DoF"] = struct.array(default=None)

    @classmethod
    def from_displacement(
        cls,
        displacement: _f.Field,
        velocity: _f.Field | None = None,
        force: _f.Field | None = None,
    ) -> Self:
        self: Self = cls(displacement=displacement, velocity=velocity, force=force)  # pyright: ignore[reportArgumentType]
        return self

    @property
    def geometry(self) -> _g.Geometry:
        return self.displacement.geometry

    @property
    def mesh(self) -> _g.Geometry:
        return self.displacement.mesh

    @property
    def n_cells(self) -> int:
        return self.displacement.n_cells

    @property
    def n_dirichlet(self) -> int:
        if self.dirichlet_index is None:
            return 0
        return self.dirichlet_index.size

    @property
    def n_dof(self) -> int:
        return self.displacement.n_dof

    @property
    def n_free(self) -> int:
        return self.n_dof - self.n_dirichlet

    @property
    def n_points(self) -> int:
        return self.displacement.n_points

    @property
    def region(self) -> _r.Region:
        return self.displacement.region

    def build_dof_indices(self) -> Self:
        raise NotImplementedError

    def set_dirichlet[F: _f.Field](self, field: F, *, zeros: bool) -> F:
        return field.with_values(
            field.values.ravel()
            .at[self.dirichlet_index]
            .set(0.0 if zeros else self.dirichlet_values)
        )

    def with_displacement(self, displacement: _f.Field) -> Self:
        return self.evolve(displacement=displacement)
