import collections
from collections.abc import Mapping
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import optim, struct, utils
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.energy import Energy
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.scene.problem import SceneProblem


@struct.pytree
class Scene(struct.PyTreeMixin):
    actors: struct.NodeContainer[Actor] = struct.container(factory=struct.NodeContainer)
    dirichlet: Dirichlet = struct.data(factory=Dirichlet)
    energies: struct.NodeContainer[Energy] = struct.container(
        factory=struct.NodeContainer
    )
    params: GlobalParams = struct.data(factory=GlobalParams)

    n_dofs: int = struct.static(kw_only=True)

    @property
    def x0(self) -> Float[jax.Array, " DOF"]:
        x0: Float[jax.Array, " DOF"] = jnp.zeros((self.n_dofs,))
        x0 = self.dirichlet.apply(x0)
        return x0

    # region Optimization

    @utils.jit_method
    def fun(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, ""]:
        fields: struct.ArrayDict = self.scatter(x)
        fun: Float[jax.Array, ""] = jnp.zeros(())
        for energy in self.energies.values():
            fun += energy.fun(fields, self.params)
        return fun

    @utils.jit_method
    def jac(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, " DOF"]:
        fields: struct.ArrayDict = self.scatter(x)
        jac_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            jac_dict += energy.jac(fields, self.params)
        jac: Float[jax.Array, " DOF"] = self.gather(jac_dict)
        jac = self.dirichlet.zero(jac)  # ! apply dirichlet constraints
        return jac

    @utils.jit_method
    def hessp(
        self, x: Float[jax.Array, " DOF"], p: Float[jax.Array, " DOF"], /
    ) -> Float[jax.Array, " DOF"]:
        fields: struct.ArrayDict = self.scatter(x)
        fields_p: struct.ArrayDict = self.scatter(p)
        hessp_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            hessp_dict += energy.hessp(fields, fields_p, self.params)
        hessp: Float[jax.Array, " DOF"] = self.gather(hessp_dict)
        return hessp

    @utils.jit_method
    def hess_diag(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, " DOF"]:
        fields: struct.ArrayDict = self.scatter(x)
        hess_diag_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            hess_diag_dict += energy.hess_diag(fields, self.params)
        hess_diag: Float[jax.Array, " DOF"] = self.gather(hess_diag_dict)
        return hess_diag

    @utils.jit_method
    def hess_quad(
        self, x: Float[jax.Array, " DOF"], p: Float[jax.Array, " DOF"], /
    ) -> Float[jax.Array, ""]:
        fields: struct.ArrayDict = self.scatter(x)
        fields_p: struct.ArrayDict = self.scatter(p)
        hess_quad: Float[jax.Array, ""] = jnp.zeros(())
        for energy in self.energies.values():
            hess_quad += energy.hess_quad(fields, fields_p, self.params)
        return hess_quad

    @utils.jit_method
    def fun_and_jac(
        self, x: Float[jax.Array, " DOF"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DOF"]]:
        return self.fun(x), self.jac(x)

    @utils.jit_method
    def jac_and_hess_diag(
        self, x: Float[jax.Array, " DOF"], /
    ) -> tuple[Float[jax.Array, " DOF"], Float[jax.Array, " DOF"]]:
        return self.jac(x), self.hess_diag(x)

    # endregion Optimization

    # region Procedure

    def prepare(self, x: Float[jax.Array, " DOF"] | None = None) -> Self:
        actors: struct.NodeContainer[Actor] = self.actors
        fields: struct.ArrayDict | Mapping[str, None] = (
            collections.defaultdict(lambda: None) if x is None else self.scatter(x)
        )
        for actor in actors.values():
            actor_new: Actor = actor.prepare(fields[actor.id])
            actors = actors.add(actor_new)
        energies: struct.NodeContainer[Energy] = self.energies
        for energy in energies.values():
            energy_new: Energy = energy.with_actors(actors.key_filter(energy.actors))
            energy_new = energy_new.prepare(self.params)
            energies = energies.add(energy_new)
        return self.evolve(actors=actors, energies=energies)

    def solve(
        self,
        x0: Float[ArrayLike, " DOF"] | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> optim.OptimizeResult:
        if x0 is None:
            x0 = self.x0
        if optimizer is None:
            optimizer = optim.PNCG()
        x0: Float[jax.Array, " DOF"] = jnp.asarray(x0)
        problem = SceneProblem(scene=self)
        result: optim.OptimizeResult = optimizer.minimize(
            problem.fun,
            x0=x0,
            jac=problem.jac,
            hessp=problem.hessp,
            hess_diag=problem.hess_diag,
            hess_quad=problem.hess_quad,
            fun_and_jac=problem.fun_and_jac,
            jac_and_hess_diag=problem.jac_and_hess_diag,
            callback=problem.callback,
        )
        return result

    def step(self, x: Float[jax.Array, " DOF"], /) -> Self:
        raise NotImplementedError

    # endregion Procedure

    # region Utilities

    def gather(self, arrays: struct.ArrayDict, /) -> Float[jax.Array, " DOF"]:
        result: Float[jax.Array, " DOF"] = jnp.zeros((self.n_dofs,))
        for key, value in arrays.items():
            actor: Actor = self.actors[key]
            result = actor.dofs.add(result, value)
        return result

    def scatter(self, x: Float[jax.Array, " DOF"], /) -> struct.ArrayDict:
        return struct.ArrayDict(
            {actor.id: actor.dofs.get(x) for actor in self.actors.values()}
        )

    # endregion Utilities
