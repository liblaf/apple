from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import energy as _e
from liblaf.apple.sim import obj as _o


class Scene(struct.Node):
    energies: struct.NodeCollection[_e.Energy] = struct.data(
        factory=struct.NodeCollection
    )
    objects: struct.NodeCollection[_o.Object] = struct.data(
        factory=struct.NodeCollection
    )

    _n_dof: int = struct.data(default=None)

    # region Optimization

    def fun(self, x: Float[ArrayLike, " N"] | None = None) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def jac(self, x: Float[ArrayLike, " N"] | None = None) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def hess_diag(
        self, x: Float[ArrayLike, " N"] | None = None
    ) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def hess_quad(
        self, x: Float[jax.Array, " N"], p: Float[jax.Array, " N"]
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def fun_and_jac(
        self, x: Float[ArrayLike, " N"] | None = None
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " N"]]:
        return self.fun(x), self.jac(x)

    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " N"] | None = None
    ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
        return self.jac(x), self.hess_diag(x)

    # endregion Optimization

    def add_energy(self, energy: _e.Energy) -> Self:
        raise NotImplementedError  # TODO

    def add_object(self, obj: _o.Object) -> Self:
        objects: struct.NodeCollection[_o.Object] = self.objects.copy()
        objects.update(obj)
        return self.evolve(objects=objects)

    def build_dof_indices(self) -> Self:
        offset: int = 0
        objects: struct.NodeCollection[_o.Object] = self.objects.copy()
        for obj in objects:
            if obj.is_view:
                continue
            objects.update(
                obj.evolve(dof_index=jnp.index_exp[offset : offset + obj.n_dof])
            )
        for obj in objects:
            if not obj.is_view:
                continue
            objects.update(obj.with_deps(objects).build_dof_indices())
        return self.evolve(objects=objects)

    def with_displacement(self, x: Float[jax.Array, " N"]) -> "Scene":
        for obj in self.objects:
            if obj.is_view:
                continue
        return self
