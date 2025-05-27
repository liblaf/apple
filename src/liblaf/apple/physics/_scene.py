from collections.abc import Mapping

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import utils

from ._energy import Energy
from ._field import Field, FieldSpec


class Scene(flax.struct.PyTreeNode):
    energies: dict[str, Energy] = flax.struct.field(default_factory=dict)
    fields: dict[str, FieldSpec] = flax.struct.field(default_factory=dict)

    @property
    def n_dirichlet(self) -> int:
        return sum(field.n_dirichlet for field in self.fields.values())

    @property
    def n_dof(self) -> int:
        return sum(field.n_dof for field in self.fields.values())

    @property
    def n_free(self) -> int:
        return sum(field.n_free for field in self.fields.values())

    @property
    def n_points(self) -> int:
        return sum(field.n_points for field in self.fields.values())

    # region calculus

    @utils.jit
    def fun(self, u: Float[jax.Array, " free"]) -> Float[jax.Array, ""]:
        fields: dict[str, Field] = self.make_fields(u)
        results: list[Float[jax.Array, ""]] = []
        for energy in self.energies.values():
            results.append(energy.fun(fields[energy.field_id]))  # noqa: PERF401
        return jnp.sum(jnp.asarray(results))

    @utils.jit
    def jac(self, u: Float[jax.Array, " free"]) -> Float[jax.Array, " free"]:
        fields: dict[str, Field] = self.make_fields(u)
        energy_jac: dict[str, jax.Array] = {}
        for energy in self.energies.values():
            energy_jac[energy.id] = energy.jac(fields[energy.field_id])
        field_jac: dict[str, jax.Array] = self.energy_to_fields(energy_jac)
        free_jac: Float[jax.Array, " free"] = self.fields_to_free(field_jac)
        return free_jac

    @utils.jit
    def hess_diag(self, u: Float[jax.Array, " free"]) -> Float[jax.Array, " free"]:
        fields: dict[str, Field] = self.make_fields(u)
        energy_hess_diag: dict[str, jax.Array] = {}
        for energy in self.energies.values():
            energy_hess_diag[energy.id] = energy.hess_diag(fields[energy.field_id])
        field_hess_diag: dict[str, jax.Array] = self.energy_to_fields(energy_hess_diag)
        free_hess_diag: Float[jax.Array, " free"] = self.fields_to_free(field_hess_diag)
        return free_hess_diag

    @utils.jit
    def hess_quad(
        self, u: Float[jax.Array, " free"], p: Float[jax.Array, " free"]
    ) -> Float[jax.Array, ""]:
        fields: dict[str, Field] = self.make_fields(u)
        p_fields: dict[str, Field] = self.make_fields_no_dirichlet(p)
        energy_hess_quad: list[Float[jax.Array, ""]] = []
        for energy in self.energies.values():
            energy_hess_quad.append(  # noqa: PERF401
                energy.hess_quad(fields[energy.field_id], p_fields[energy.field_id])
            )
        return jnp.sum(jnp.asarray(energy_hess_quad))

    @utils.jit
    def jac_and_hess_diag(
        self, u: Float[jax.Array, " free"]
    ) -> tuple[Float[jax.Array, " free"], Float[jax.Array, " free"]]:
        fields: dict[str, Field] = self.make_fields(u)
        energy_jac: dict[str, jax.Array] = {}
        energy_hess_diag: dict[str, jax.Array] = {}
        for energy in self.energies.values():
            energy_jac[energy.id], energy_hess_diag[energy.id] = (
                energy.jac_and_hess_diag(fields[energy.field_id])
            )
        field_jac: dict[str, jax.Array] = self.energy_to_fields(energy_jac)
        field_hess_diag: dict[str, jax.Array] = self.energy_to_fields(energy_hess_diag)
        free_jac: Float[jax.Array, " free"] = self.fields_to_free(field_jac)
        free_hess_diag: Float[jax.Array, " free"] = self.fields_to_free(field_hess_diag)
        return free_jac, free_hess_diag

    # endregion calculus

    def add_energy(self, energy: Energy) -> None:
        self.energies[energy.id] = energy

    def add_field(self, field: FieldSpec) -> None:
        self.fields[field.id] = field

    def energy_to_fields(
        self, energy_values: Mapping[str, jax.Array]
    ) -> dict[str, jax.Array]:
        fields: dict[str, jax.Array] = {}
        for field in self.fields.values():
            fields[field.id] = jnp.zeros((field.n_dof,))
        for energy in self.energies.values():
            fields[energy.field_id] += energy_values[energy.id]
        return fields

    def fields_to_free(
        self, field_values: Mapping[str, jax.Array]
    ) -> Float[jax.Array, " free"]:
        free_values: Float[jax.Array, " free"] = jnp.zeros((self.n_free,))
        offset: int = 0
        for field in self.fields.values():
            n_free: int = field.n_free
            free_values = free_values.at[offset : offset + n_free].set(
                field_values[field.id][field.free_index]
            )
            offset += field.n_free
        return free_values

    @utils.jit
    def make_fields(self, free_values: Float[jax.Array, " free"]) -> dict[str, Field]:
        free_values = jnp.asarray(free_values)
        fields: dict[str, Field] = {}
        offset = 0
        for field in self.fields.values():
            n_free: int = field.n_free
            free_values: jax.Array = free_values[offset : offset + n_free]
            fields[field.id] = field.make_field(free_values)
            offset += n_free
        return fields

    @utils.jit
    def make_fields_no_dirichlet(
        self, free_values: Float[jax.Array, " free"]
    ) -> dict[str, Field]:
        free_values = jnp.asarray(free_values)
        fields: dict[str, Field] = {}
        offset = 0
        for field in self.fields.values():
            n_free: int = field.n_free
            free_values: jax.Array = free_values[offset : offset + n_free]
            fields[field.id] = field.make_field_no_dirichlet(free_values)
            offset += n_free
        return fields
