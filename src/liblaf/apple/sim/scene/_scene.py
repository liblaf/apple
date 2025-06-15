from collections.abc import Mapping
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import struct, utils
from liblaf.apple.sim.abc import Energy, FieldCollection, Object


class Scene(struct.PyTree):
    energies: struct.NodeCollection[Energy] = struct.data(factory=struct.NodeCollection)
    objects: struct.NodeCollection[Object] = struct.data(factory=struct.NodeCollection)

    time_step: Float[jax.Array, ""] = struct.array(default=jnp.asarray(1 / 30))

    # region Shape

    @property
    def n_dof(self) -> int:
        return sum(obj.n_dof for obj in self.objects)

    # endregion Shape

    @property
    def dof_index(self) -> Mapping[str, struct.Index]:
        return {obj.id: obj.dof_index for obj in self.objects}

    # region Optimization

    def prepare(self) -> None:
        for obj in self.objects:
            obj.prepare()
        for energy in self.energies:
            energy.prepare()

    @utils.jit
    def fun(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, ""]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fun: Float[jax.Array, ""] = jnp.asarray(0.0)
        for energy in self.energies:
            fun += energy.fun(fields.select(energy.objects.keys()))
        return fun

    @utils.jit
    def jac(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        jac = FieldCollection()
        for energy in self.energies:
            energy_jac: FieldCollection = energy.jac(
                fields.select(energy.objects.keys())
            )
            jac += energy_jac
        return jac.gather(self.dof_index, n_dof=self.n_dof)

    @utils.jit
    def hessp(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fields_p: FieldCollection = self.make_fields_dirichlet_zero(p)
        hessp = FieldCollection()
        for energy in self.energies:
            energy_hessp: FieldCollection = energy.hessp(
                fields.select(energy.objects.keys()),
                fields_p.select(energy.objects.keys()),
            )
            hessp += energy_hessp
        return hessp.gather(self.dof_index, n_dof=self.n_dof)

    @utils.jit
    def hess_diag(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        hess_diag = FieldCollection()
        for energy in self.energies:
            energy_hess_diag: FieldCollection = energy.hess_diag(
                fields.select(energy.objects.keys())
            )
            hess_diag += energy_hess_diag
        return hess_diag.gather(self.dof_index, n_dof=self.n_dof)

    @utils.jit
    def hess_quad(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, ""]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fields_p: FieldCollection = self.make_fields_dirichlet_zero(p)
        hess_quad: Float[jax.Array, ""] = jnp.asarray(0.0)
        for energy in self.energies:
            energy_hess_quad: Float[jax.Array, ""] = energy.hess_quad(
                fields.select(energy.objects.keys()),
                fields_p.select(energy.objects.keys()),
            )
            hess_quad += energy_hess_quad
        return hess_quad

    @utils.jit
    def fun_and_jac(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        fun: Float[jax.Array, ""] = self.fun(x)
        jac: Float[jax.Array, " DoF"] = self.jac(x)
        return fun, jac

    @utils.jit
    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        jac: Float[jax.Array, " DoF"] = self.jac(x)
        hess_diag: Float[jax.Array, " DoF"] = self.hess_diag(x)
        return jac, hess_diag

    # endregion Optimization

    # region State Update

    def step(self, x: Float[ArrayLike, " DoF"], /) -> Self:
        displacement: FieldCollection = self.make_fields_dirichlet(x)
        raise NotImplementedError

    def with_displacement(self, x: Float[ArrayLike, " DoF"], /) -> Self:
        raise NotImplementedError

    # endregion State Update

    # region Utilities

    def make_fields(self, x: Float[ArrayLike, " DoF"], /) -> FieldCollection:
        fields: FieldCollection = FieldCollection()
        for obj in self.objects:
            fields = fields.add(obj.id, obj.displacement.with_values(obj.dof_index(x)))
        return fields

    def make_fields_dirichlet(self, x: Float[ArrayLike, " DoF"], /) -> FieldCollection:
        fields: FieldCollection = self.make_fields(x)
        for obj in self.objects:
            if obj.dirichlet is None:
                continue
            fields = fields.add(obj.id, obj.dirichlet.apply(fields[obj.id]))
        return fields

    def make_fields_dirichlet_zero(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> FieldCollection:
        fields: FieldCollection = self.make_fields(x)
        for obj in self.objects:
            if obj.dirichlet is None:
                continue
            fields = fields.add(obj.id, obj.dirichlet.zero(fields[obj.id]))
        return fields

    # endregion Utilities
