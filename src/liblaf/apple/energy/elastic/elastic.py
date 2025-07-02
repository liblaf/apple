from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import sim, struct, utils


@struct.pytree
class Elastic(sim.Energy):
    actor: sim.Actor = struct.data()
    hess_diag_filter: bool = struct.static(default=True, kw_only=True)
    hess_quad_filter: bool = struct.static(default=True, kw_only=True)

    @classmethod
    def from_actor(
        cls,
        actor: sim.Actor,
        *,
        hess_diag_filter: bool = True,
        hess_quad_filter: bool = True,
    ) -> Self:
        return cls(
            actor=actor,
            hess_diag_filter=hess_diag_filter,
            hess_quad_filter=hess_quad_filter,
        )

    @property
    @override
    def actors(self) -> struct.NodeContainer[sim.Actor]:
        return struct.NodeContainer([self.actor])

    @override
    def with_actors(self, actors: struct.NodeContainer[sim.Actor]) -> Self:
        return self.evolve(actor=actors[self.actor.id])

    def make_field(self, x: struct.ArrayDict, /) -> sim.Field:
        x: Float[jax.Array, "points dim"] = x[self.actor.id]
        return self.actor.make_field(x)

    @override
    @utils.jit_method(inline=True)
    def fun(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        field: sim.Field = self.make_field(x)
        Psi: Float[jax.Array, "c q"] = self.energy_density(field, params)
        Psi: Float[jax.Array, " c"] = field.region.integrate(Psi)
        return jnp.sum(Psi)

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        field: sim.Field = self.make_field(x)
        jac: Float[jax.Array, "c q a J"] = self.energy_density_jac(field, params)
        jac: Float[jax.Array, "c a J"] = field.region.integrate(jac)
        jac: Float[jax.Array, "p J"] = field.region.gather(jac)
        return struct.ArrayDict({self.actor.id: jac})

    @override
    @utils.jit_method(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        field: sim.Field = self.make_field(x)
        hess_diag: Float[jax.Array, "c q a J"] = self.energy_density_hess_diag(
            field, params
        )
        if self.hess_diag_filter:
            hess_diag = jnp.clip(hess_diag, min=0.0)
        # jax.debug.print("Elastic.hess_diag: {}", hess_diag)
        hess_diag: Float[jax.Array, "c a J"] = field.region.integrate(hess_diag)
        hess_diag: Float[jax.Array, "p J"] = field.region.gather(hess_diag)
        return struct.ArrayDict({self.actor.id: hess_diag})

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        field: sim.Field = self.make_field(x)
        field_p: sim.Field = self.make_field(p)
        hess_quad: Float[jax.Array, "c q"] = self.energy_density_hess_quad(
            field, field_p, params
        )
        if self.hess_quad_filter:
            hess_quad = jnp.clip(hess_quad, min=0.0)
        hess_quad: Float[jax.Array, " c"] = field.region.integrate(hess_quad)
        hess_quad: Float[jax.Array, ""] = jnp.sum(hess_quad)
        return hess_quad

    @override
    @utils.jit_method(inline=True)
    def fun_and_jac(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> tuple[Float[jax.Array, ""], struct.ArrayDict]:
        return self.fun(x, params), self.jac(x, params)

    @override
    @utils.jit_method(inline=True)
    def jac_and_hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> tuple[struct.ArrayDict, struct.ArrayDict]:
        return self.jac(x, params), self.hess_diag(x, params)

    def energy_density(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q J J"]:
        raise NotImplementedError

    @utils.jit_method(inline=True)
    def energy_density_jac(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q a J"]:
        PK1: Float[jax.Array, "c q J J"] = self.first_piola_kirchhoff_stress(
            field, params
        )
        dPsidx: Float[jax.Array, "c q a J"] = field.region.gradient_vjp(PK1)
        return dPsidx

    def energy_density_hess_diag(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q a J"]:
        raise NotImplementedError

    def energy_density_hess_quad(
        self, field: sim.Field, p: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        raise NotImplementedError
