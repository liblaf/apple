import attrs
import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from loguru import logger

from liblaf.apple import struct
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import make_dofs
from liblaf.apple.sim.energy import Energy
from liblaf.apple.sim.integrator import ImplicitEuler, TimeIntegrator
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.state import State

from .scene import Scene


@attrs.define
class SceneBuilder:
    actors: struct.NodeContainer[Actor] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )
    actors_concrete: struct.NodeContainer[Actor] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )
    energies: struct.NodeContainer[Energy] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )
    integrator: TimeIntegrator = attrs.field(factory=ImplicitEuler)
    params: GlobalParams = attrs.field(factory=GlobalParams)

    @property
    def dirichlet(self) -> Dirichlet:
        mask: Bool[Array, " DOF"] = jnp.zeros((self.n_dofs,), dtype=bool)
        values: Float[Array, " DOF"] = jnp.zeros((self.n_dofs,))
        for actor in self.actors.values():
            dirichlet_global: Dirichlet | None = actor.dirichlet_global
            if dirichlet_global is None:
                continue
            mask = dirichlet_global.mask(mask)
            values = dirichlet_global.apply(values)
        return Dirichlet.from_mask(mask, values)

    @property
    def n_dofs(self) -> int:
        return sum(actor.n_dofs for actor in self.actors_concrete.values())

    # region Attributes

    @property
    def displacement(self) -> Float[jax.Array, " DOF"]:
        return self.gather_point_data("displacement")

    @property
    def force_ext(self) -> Float[jax.Array, " DOF"]:
        return self.gather_point_data("force-ext")

    @property
    def mass(self) -> Float[jax.Array, " DOF"]:
        mass: Float[jax.Array, " DOF"] = jnp.zeros((self.n_dofs,))
        for actor in self.actors_concrete.values():
            mass = actor.dofs_global.set(
                mass, einops.repeat(actor.mass, "points -> points dim", dim=actor.dim)
            )
        if not jnp.all(mass >= 0):
            logger.error("not all mass >= 0")
        elif not jnp.all(mass > 0):
            logger.warning("not all mass > 0")
        return mass

    @property
    def velocity(self) -> Float[jax.Array, " DOF"]:
        return self.gather_point_data("velocity")

    # endregion Attributes

    # region Builder

    def add_actor(self, actor: Actor) -> Actor:
        self.actors.add(actor)
        return actor

    def add_energy(self, energy: Energy) -> Energy:
        self.energies.add(energy)
        for actor in energy.actors.values():
            self.add_actor(actor)
        return energy

    def assign_dofs(self, actor: Actor) -> Actor:
        actor = actor.with_dofs(
            make_dofs((actor.n_points, actor.dim), offset=self.n_dofs)
        )
        self.actors_concrete.add(actor)
        return actor

    def finish(self) -> Scene:
        return Scene(
            actors=self.actors,
            dirichlet=self.dirichlet,
            energies=self.energies,
            integrator=self.integrator,
            n_dofs=self.n_dofs,
            params=self.params,
            state=State(
                {
                    "displacement": self.displacement,
                    "velocity": self.velocity,
                    "force-ext": self.force_ext,
                    "mass": self.mass,
                }
            ),
        )

    # endregion Builder

    # region Utilities

    def gather_point_data(self, name: str) -> Float[jax.Array, " DOF"]:
        data: Float[jax.Array, " DOF"] = jnp.zeros((self.n_dofs,))
        for actor in self.actors_concrete.values():
            if name not in actor.point_data:
                continue
            data = actor.dofs_global.set(data, actor.point_data[name])
        return data

    # endregion Utilities
