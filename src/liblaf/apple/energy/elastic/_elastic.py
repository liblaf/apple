from typing import Self, override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import sim, struct, utils

type Field = Float[Array, "points dim"]


class Elastic(sim.Energy):
    actor: sim.Actor = struct.field()
    hess_diag_filter: bool = struct.field(default=True, kw_only=True)
    hess_quad_filter: bool = struct.field(default=True, kw_only=True)

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
        return self.replace(actor=actors[self.actor.id])

    @property
    def region(self) -> sim.Region:
        return self.actor.region

    @override
    @utils.jit(inline=True)
    def fun(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> Float[Array, ""]:
        field: Field = x[self.actor.id]
        Psi: Float[Array, "c q"] = self.energy_density(field, params)
        Psi: Float[Array, " c"] = self.region.integrate(Psi)
        return jnp.sum(Psi)

    @override
    @utils.jit(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        field: Field = x[self.actor.id]
        jac: Float[Array, "c q a J"] = self.energy_density_jac(field, params)
        jac: Float[Array, "c a J"] = self.region.integrate(jac)
        jac: Float[Array, "p J"] = self.region.gather(jac)
        return struct.ArrayDict({self.actor.id: jac})

    @override
    @utils.jit(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        field: Field = x[self.actor.id]
        hess_diag: Float[Array, "c q a J"] = self.energy_density_hess_diag(
            field, params
        )
        if self.hess_diag_filter:
            hess_diag = jnp.clip(hess_diag, min=0.0)
        # jax.debug.print("Elastic.hess_diag: {}", hess_diag)
        hess_diag: Float[Array, "c a J"] = self.region.integrate(hess_diag)
        hess_diag: Float[Array, "p J"] = self.region.gather(hess_diag)
        return struct.ArrayDict({self.actor.id: hess_diag})

    @override
    @utils.jit(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[Array, ""]:
        field: Field = x[self.actor.id]
        field_p: Field = p[self.actor.id]
        hess_quad: Float[Array, "c q"] = self.energy_density_hess_quad(
            field, field_p, params
        )
        if self.hess_quad_filter:
            hess_quad = jnp.clip(hess_quad, min=0.0)
        hess_quad: Float[Array, " c"] = self.region.integrate(hess_quad)
        hess_quad: Float[Array, ""] = jnp.sum(hess_quad)
        return hess_quad

    @override
    @utils.jit(inline=True)
    def fun_and_jac(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> tuple[Float[Array, ""], struct.ArrayDict]:
        return self.fun(x, params), self.jac(x, params)

    @override
    @utils.jit(inline=True)
    def jac_and_hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> tuple[struct.ArrayDict, struct.ArrayDict]:
        return self.jac(x, params), self.hess_diag(x, params)

    def energy_density(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[Array, "c q"]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[Array, "c q J J"]:
        raise NotImplementedError

    @utils.jit(inline=True)
    def energy_density_jac(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[Array, "c q a J"]:
        PK1: Float[Array, "c q J J"] = self.first_piola_kirchhoff_stress(field, params)
        dPsidx: Float[Array, "c q a J"] = self.region.gradient_vjp(PK1)
        return dPsidx

    def energy_density_hess_diag(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[Array, "c q a J"]:
        raise NotImplementedError

    def energy_density_hess_quad(
        self, field: Field, p: Field, /, params: sim.GlobalParams
    ) -> Float[Array, "c q"]:
        raise NotImplementedError
