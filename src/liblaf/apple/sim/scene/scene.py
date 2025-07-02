from collections.abc import Callable
from typing import Self, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import optim, struct, utils
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.energy import Energy
from liblaf.apple.sim.integrator import TimeIntegrator
from liblaf.apple.sim.params import GlobalParams
from liblaf.apple.sim.scene.problem import SceneProblem
from liblaf.apple.sim.state import State

type X = Float[jax.Array, " DOF"]
type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class Scene(struct.PyTreeMixin):
    actors: struct.NodeContainer[Actor] = struct.container(factory=struct.NodeContainer)
    dirichlet: Dirichlet = struct.data(factory=Dirichlet)
    energies: struct.NodeContainer[Energy] = struct.container(
        factory=struct.NodeContainer
    )
    integrator: TimeIntegrator = struct.data(kw_only=True)
    n_dofs: int = struct.static(kw_only=True)
    state: State = struct.container(factory=State)
    params: GlobalParams = struct.data(factory=GlobalParams)

    @property
    def x0(self) -> X:
        x0: X
        if "displacement" in self.state:
            return self.state.displacement
        x0 = jnp.zeros((self.n_dofs,))
        x0 = self.dirichlet.apply(x0)
        return x0

    # region Optimization

    @utils.jit_method
    def fun(self, x: X, /) -> FloatScalar:
        fields: struct.ArrayDict = self.scatter(x)
        fun: FloatScalar = jnp.zeros(())
        for energy in self.energies.values():
            fun += energy.fun(fields, self.params)
        fun += self.integrator.fun(x, self.state, self.params)
        return fun

    @utils.jit_method(inline=True)
    def jac(self, x: X, /) -> X:
        fields: struct.ArrayDict = self.scatter(x)
        jac_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            jac_dict += energy.jac(fields, self.params)
        jac: X = self.gather(jac_dict)
        integrator_jac: X = self.integrator.jac(x, self.state, self.params)
        # jax.debug.print("integrator.jac(): {}", integrator_jac)
        jac += integrator_jac
        jac = self.dirichlet.zero(jac)  # apply dirichlet constraints
        return jac

    @utils.jit_method(inline=True)
    def hessp(self, x: X, p: X, /) -> X:
        fields: struct.ArrayDict = self.scatter(x)
        fields_p: struct.ArrayDict = self.scatter(p)
        hessp_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            hessp_dict += energy.hessp(fields, fields_p, self.params)
        hessp: X = self.gather(hessp_dict)
        hessp += self.integrator.hessp(x, p, self.state, self.params)
        return hessp

    @utils.jit_method(inline=True)
    def hess_diag(self, x: X, /) -> X:
        fields: struct.ArrayDict = self.scatter(x)
        hess_diag_dict: struct.ArrayDict = struct.ArrayDict()
        for energy in self.energies.values():
            hess_diag_dict += energy.hess_diag(fields, self.params)
        hess_diag: X = self.gather(hess_diag_dict)
        # jax.debug.print("energy.hess_diag: {}", hess_diag)
        integrator_hess_diag: X = self.integrator.hess_diag(x, self.state, self.params)
        # jax.debug.print("integrator.hess_diag: {}", integrator_hess_diag)
        hess_diag += integrator_hess_diag
        return hess_diag

    @utils.jit_method
    def hess_quad(self, x: X, p: X, /) -> FloatScalar:
        fields: struct.ArrayDict = self.scatter(x)
        fields_p: struct.ArrayDict = self.scatter(p)
        hess_quad: FloatScalar = jnp.zeros(())
        for energy in self.energies.values():
            hess_quad += energy.hess_quad(fields, fields_p, self.params)
        hess_quad += self.integrator.hess_quad(x, p, self.state, self.params)
        return hess_quad

    @utils.jit_method
    def fun_and_jac(self, x: X, /) -> tuple[FloatScalar, X]:
        return self.fun(x), self.jac(x)

    @utils.jit_method
    def jac_and_hess_diag(self, x: X, /) -> tuple[X, X]:
        return self.jac(x), self.hess_diag(x)

    # endregion Optimization

    # region Procedure

    @utils.jit_method(inline=True, validate=False)
    def pre_time_step(self) -> Self:
        state: State = self.integrator.pre_time_step(self.state, self.params)
        return self.evolve(state=state)
        # actors: struct.NodeContainer[Actor] = self.actors
        # for actor in self.actors.values():
        #     actor_new: Actor = actor.pre_time_step()
        #     actors = self.actors.add(actor_new)
        # energies: struct.NodeContainer[Energy] = self.energies
        # for energy in energies.values():
        #     energy_new: Energy = energy.with_actors(actors.key_filter(energy.actors))
        #     energy_new = energy_new.pre_time_step(self.params)
        #     energies = energies.add(energy_new)
        # return self.evolve(actors=actors, energies=energies, state=state)

    # @utils.jit_method(inline=True, validate=False)
    def pre_optim_iter(self, x: X | None = None) -> Self:
        if x is None:
            x = self.x0
        state: State = self.integrator.pre_optim_iter(x, self.state, self.params)
        actors: struct.NodeContainer[Actor] = self.actors
        fields: struct.ArrayDict = self.scatter(x)
        for actor in actors.values():
            actor_new: Actor = actor.pre_optim_iter(fields[actor.id])
            actors = actors.add(actor_new)
        energies: struct.NodeContainer[Energy] = self.energies
        for energy in energies.values():
            energy_new: Energy = energy.with_actors(actors.key_filter(energy.actors))
            energy_new = energy_new.pre_optim_iter(self.params)
            energies = energies.add(energy_new)
        return self.evolve(actors=actors, energies=energies, state=state)

    def solve(
        self,
        x0: X | None = None,
        optimizer: optim.Optimizer | None = None,
        callback: Callable | None = None,
    ) -> tuple[Self, optim.OptimizeResult]:
        if optimizer is None:
            optimizer = optim.PNCG()
        scene: Self = self
        scene = scene.pre_time_step()
        if x0 is None:
            x0 = self.integrator.make_x0(self.state, self.params)
        x0: X = jnp.asarray(x0)
        scene = scene.pre_optim_iter(x0)
        problem = SceneProblem(scene=scene, callback=callback)
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
        return cast("Self", problem.scene), result

    def step(self, x: X, /) -> Self:
        state: State = self.integrator.step(x, self.state, self.params)
        return self.evolve(state=state)

    # endregion Procedure

    # region Utilities

    def export_actor(self, actor: Actor) -> Actor:
        return actor.update(
            displacement=actor.dofs.get(self.state.displacement),
            velocity=actor.dofs.get(self.state.velocity),
        )

    def gather(self, arrays: struct.ArrayDict, /) -> X:
        result: X = jnp.zeros((self.n_dofs,))
        for key, value in arrays.items():
            actor: Actor = self.actors[key]
            result = actor.dofs.add(result, value)
        return result

    def scatter(self, x: X, /) -> struct.ArrayDict:
        return struct.ArrayDict(
            {actor.id: actor.dofs.get(x) for actor in self.actors.values()}
        )

    # endregion Utilities
