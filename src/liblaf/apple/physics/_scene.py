from collections.abc import Mapping
from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import optim, utils
from liblaf.apple.physics._geometry import Geometry

from ._energy import Energy
from ._field import Field


class Scene(flax.struct.PyTreeNode):
    energies: dict[str, Energy] = flax.struct.field(default_factory=dict)
    fields: dict[str, Field] = flax.struct.field(default_factory=dict)
    optimizer: optim.Optimizer = flax.struct.field(default_factory=optim.PNCG)

    time_step: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0 / 30.0)
    )

    @property
    def free_values(self) -> Float[jax.Array, " free"]:
        free_values: Float[jax.Array, " free"] = jnp.zeros((self.n_free,))
        offset = 0
        for field in self.fields.values():
            n_free: int = field.n_free
            free_values = free_values.at[offset : offset + n_free].set(
                field.free_values
            )
            offset += n_free
        return free_values

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
    def fun(self, u: Float[jax.Array, " free"] | None = None) -> Float[jax.Array, ""]:
        scene: Self = self.with_free_values(u)
        results: list[Float[jax.Array, ""]] = []
        for energy in scene.energies.values():
            results.append(energy.fun(scene.fields[energy.field_id]))  # noqa: PERF401
        return jnp.sum(jnp.asarray(results))

    @utils.jit
    def jac(self, u: Float[jax.Array, " free"]) -> Float[jax.Array, " free"]:
        scene: Self = self.with_free_values(u)
        energy_jac: dict[str, jax.Array] = {}
        for energy in scene.energies.values():
            energy_jac[energy.id] = energy.jac(scene.fields[energy.field_id])
        field_jac: dict[str, jax.Array] = scene.energy_to_fields(energy_jac)
        free_jac: Float[jax.Array, " free"] = scene.fields_to_free(field_jac)
        return free_jac

    @utils.jit
    def hess_diag(self, u: Float[jax.Array, " free"]) -> Float[jax.Array, " free"]:
        scene: Self = self.with_free_values(u)
        energy_hess_diag: dict[str, jax.Array] = {}
        for energy in scene.energies.values():
            energy_hess_diag[energy.id] = energy.hess_diag(
                scene.fields[energy.field_id]
            )
        field_hess_diag: dict[str, jax.Array] = scene.energy_to_fields(energy_hess_diag)
        free_hess_diag: Float[jax.Array, " free"] = scene.fields_to_free(
            field_hess_diag
        )
        return free_hess_diag

    @utils.jit
    def hess_quad(
        self, u: Float[jax.Array, " free"], p: Float[jax.Array, " free"]
    ) -> Float[jax.Array, ""]:
        scene: Self = self.with_free_values(u)
        p_fields: dict[str, Field] = scene.make_fields(p, dirichlet=False)
        energy_hess_quad: list[Float[jax.Array, ""]] = []
        for energy in scene.energies.values():
            energy_hess_quad.append(  # noqa: PERF401
                energy.hess_quad(
                    scene.fields[energy.field_id], p_fields[energy.field_id]
                )
            )
        return jnp.sum(jnp.asarray(energy_hess_quad))

    @utils.jit
    def jac_and_hess_diag(
        self, u: Float[jax.Array, " free"]
    ) -> tuple[Float[jax.Array, " free"], Float[jax.Array, " free"]]:
        scene: Self = self.with_free_values(u)
        energy_jac: dict[str, jax.Array] = {}
        hess_diag_per_energy: dict[str, jax.Array] = {}
        for energy in scene.energies.values():
            energy_jac[energy.id], hess_diag_per_energy[energy.id] = (
                energy.jac_and_hess_diag(scene.fields[energy.field_id])
            )
        field_jac: dict[str, jax.Array] = scene.energy_to_fields(energy_jac)
        hess_diag_per_field: dict[str, jax.Array] = scene.energy_to_fields(
            hess_diag_per_energy
        )
        free_jac: Float[jax.Array, " free"] = scene.fields_to_free(field_jac)
        free_hess_diag: Float[jax.Array, " free"] = scene.fields_to_free(
            hess_diag_per_field
        )
        return free_jac, free_hess_diag

    # endregion calculus

    def add_energy(self, energy: Energy) -> None:
        self.energies[energy.id] = energy

    def add_field(self, field: Field) -> None:
        self.fields[field.id] = field

    def energy_to_fields(
        self, values_per_energy: Mapping[str, jax.Array]
    ) -> dict[str, jax.Array]:
        fields: dict[str, jax.Array] = {}
        for field in self.fields.values():
            fields[field.id] = jnp.zeros((field.n_dof,))
        for energy in self.energies.values():
            fields[energy.field_id] += values_per_energy[energy.id]
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

    @utils.jit(static_argnames=("dirichlet",))
    def make_fields(
        self,
        free_values: Float[jax.Array, " free"] | None = None,
        *,
        dirichlet: bool = True,
    ) -> dict[str, Field]:
        if free_values is None:
            return self.fields
        free_values = jnp.asarray(free_values)
        fields: dict[str, Field] = {}
        offset = 0
        for field in self.fields.values():
            n_free: int = field.n_free
            free_values: jax.Array = free_values[offset : offset + n_free]
            fields[field.id] = field.with_free_values(free_values, dirichlet=dirichlet)
            offset += n_free
        return fields

    def make_geometries(
        self, free: Float[jax.Array, " free"] | None = None
    ) -> dict[str, Geometry]:
        scene: Self = self.with_free_values(free)
        geometries: dict[str, Geometry] = {}
        for field in scene.fields.values():
            geometry: Geometry = field.geometry.warp(field.values)
            geometries[geometry.id] = geometry
        return geometries

    def solve(self, *, callback: optim.Callback | None = None) -> optim.OptimizeResult:
        solution: optim.OptimizeResult = self.optimizer.minimize(
            fun=self.fun,
            x0=self.free_values,
            jac=self.jac,
            hess_diag=self.hess_diag,
            hess_quad=self.hess_quad,
            jac_and_hess_diag=self.jac_and_hess_diag,
            callback=callback,
        )
        return solution

    def step(self, free_values: Float[jax.Array, " free"] | None = None) -> Self:
        fields_prev: dict[str, Field] = self.make_fields()
        fields_next: dict[str, Field] = self.make_fields(free_values)
        for field_id, field_next in fields_next.items():
            prev: Field = fields_prev[field_id]
            fields_next[field_id] = fields_next[field_id].replace(
                values_prev=prev.values,
                velocities=(field_next.values - prev.values) / self.time_step,
            )
        scene: Self = self.replace(fields=fields_next)
        return scene

    @utils.jit
    def with_free_values(
        self, free_values: Float[jax.Array, " free"] | None = None
    ) -> Self:
        return self.replace(fields=self.make_fields(free_values))
