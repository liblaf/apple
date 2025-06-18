from collections.abc import Iterable
from typing import Self, cast

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import optim, struct, utils
from liblaf.apple.sim.core import Energy, GlobalParams, Object


class Scene(struct.PyTree):
    nodes: struct.FrozenDict[struct.GraphNode] = struct.mapping(
        factory=struct.FrozenDict
    )
    params: GlobalParams = struct.data(factory=GlobalParams)
    topological: Iterable[str] = struct.static(default=None)
    _base_keys: Iterable[str] = struct.static(default=None)
    _energy_keys: Iterable[str] = struct.static(default=None)

    # region Structure

    @property
    def bases(self) -> struct.FrozenDict[Object]:
        return cast("struct.FrozenDict[Object]", self.nodes.select(self._base_keys))

    @property
    def energies(self) -> struct.FrozenDict[Energy]:
        return cast("struct.FrozenDict[Energy]", self.nodes.select(self._energy_keys))

    @property
    def objects(self) -> struct.FrozenDict[Object]:
        return cast("struct.FrozenDict[Object]", self.nodes.filter_instance(Object))

    # endregion Structure

    # region DOF

    @property
    def dof_map(self) -> struct.FrozenDict[struct.DofMap]:
        return struct.FrozenDict({key: node.dof for key, node in self.bases.items()})

    @property
    def n_dof(self) -> int:
        return sum(node.n_dof for node in self.bases.values())

    # endregion DOF

    # region Computational Graph

    def prepare(self, x: Float[ArrayLike, " dof"] | None = None, /) -> Self:
        return self.update(x, prepare=True)

    def step(self, x: Float[ArrayLike, " dof"], /, *, prepare: bool = True) -> Self:
        x = jnp.asarray(x)
        nodes: struct.FrozenDict = self.nodes
        for node in self.bases.values():
            nodes = nodes.copy(node.step(x, self.params))
        scene: Self = self.evolve(nodes=nodes)
        return scene.update(prepare=prepare)

    def update(
        self, x: Float[ArrayLike, " dof"] | None = None, /, *, prepare: bool = False
    ) -> Self:
        nodes: struct.FrozenDict = self.nodes
        if x is not None:
            x = jnp.asarray(x)
            for node in self.bases.values():
                nodes = nodes.copy(node.update(node.dof.get(x)))
        for key in self.topological:
            node: struct.GraphNode = nodes[key]
            node = node.with_deps(nodes.select(node.deps.keys()))
            if prepare:
                node = node.prepare()
            nodes = nodes.copy(node)
        return self.evolve(nodes=nodes)

    # endregion Computational Graph

    # region Optimization

    def solve(
        self, optimizer: optim.Optimizer, callback: optim.Callback | None = None
    ) -> optim.OptimizeResult:
        from ._problem import OptimizationProblem

        scene: Self = self.prepare(self.x0)
        problem = OptimizationProblem(scene, callback=callback)
        return optimizer.minimize(
            problem.fun,
            self.x0,
            jac=problem.jac,
            hessp=problem.hessp,
            hess_diag=problem.hess_diag,
            hess_quad=problem.hess_quad,
            fun_and_jac=problem.fun_and_jac,
            jac_and_hess_diag=problem.jac_and_hess_diag,
            callback=problem.callback,
        )

    @property
    @utils.jit
    def x0(self) -> Float[jax.Array, " dof"]:
        x0: Float[jax.Array, " dof"] = jnp.zeros((self.n_dof,))
        for node in self.bases.values():
            x0 = node.dof.set(x0, node.displacement.values)
        return x0

    @utils.jit
    def fun(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, ""]:
        x = self.dirichlet_apply(x)
        values: struct.DictArray = self.make_field_values(x)
        fun: Float[jax.Array, ""] = jnp.zeros(())
        for energy in self.energies.values():
            deps: struct.DictArray = values.select(energy.deps.keys())
            fun += energy.fun(deps, self.params)
        return fun

    @utils.jit
    def jac(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]:
        x = self.dirichlet_apply(x)
        values: struct.DictArray = self.make_field_values(x)
        jac: struct.DictArray = struct.DictArray()
        for energy in self.energies.values():
            deps: struct.DictArray = values.select(energy.deps.keys())
            jac += energy.jac(deps, self.params)
        jac_values: Float[jax.Array, " dof"] = self.gather_values(jac)
        jac_values = self.dirichlet_zero(jac_values)
        return jac_values

    @utils.jit
    def hessp(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, " dof"]:
        x = self.dirichlet_apply(x)
        values: struct.DictArray = self.make_field_values(x)
        p_values: struct.DictArray = self.make_field_values(p)
        hessp: struct.DictArray = struct.DictArray()
        for energy in self.energies.values():
            deps: struct.DictArray = values.select(energy.deps.keys())
            hessp += energy.hessp(deps, p_values, self.params)
        hessp_values: Float[jax.Array, " dof"] = self.gather_values(hessp)
        return hessp_values

    @utils.jit
    def hess_diag(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]:
        x = self.dirichlet_apply(x)
        values: struct.DictArray = self.make_field_values(x)
        hess_diag: struct.DictArray = struct.DictArray()
        for energy in self.energies.values():
            deps: struct.DictArray = values.select(energy.deps.keys())
            hess_diag += energy.hess_diag(deps, self.params)
        hess_diag_values: Float[jax.Array, " dof"] = self.gather_values(hess_diag)
        return hess_diag_values

    @utils.jit
    def hess_quad(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, ""]:
        x = self.dirichlet_apply(x)
        values: struct.DictArray = self.make_field_values(x)
        p_values: struct.DictArray = self.make_field_values(p)
        hess_quad: Float[jax.Array, ""] = jnp.zeros(())
        for energy in self.energies.values():
            deps: struct.DictArray = values.select(energy.deps.keys())
            hess_quad += energy.hess_quad(deps, p_values, self.params)
        return hess_quad

    @utils.jit
    def fun_and_jac(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " dof"]]:
        return self.fun(x), self.jac(x)

    @utils.jit
    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, " dof"], Float[jax.Array, " dof"]]:
        return self.jac(x), self.hess_diag(x)

    # endregion Optimization

    # region Make Fields

    @utils.jit
    def dirichlet_apply(
        self, x: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, " dof"]:
        x = jnp.asarray(x)
        for node in self.bases.values():
            if node.dirichlet is None:
                continue
            x = node.dirichlet.apply(x)
        return x

    @utils.jit
    def dirichlet_zero(
        self, x: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, " dof"]:
        x = jnp.asarray(x)
        for node in self.bases.values():
            if node.dirichlet is None:
                continue
            x = node.dirichlet.zero(x)
        return x

    @utils.jit
    def gather_values(self, values: struct.DictArray, /) -> Float[jax.Array, " dof"]:
        x: Float[jax.Array, " dof"] = jnp.zeros((self.n_dof,))
        for key, value in values.items():
            obj: Object = self.objects[key]
            x = obj.dof.add(x, value)
        return x

    @utils.jit
    def make_field_values(self, x: Float[ArrayLike, " dof"]) -> struct.DictArray:
        x: Float[ArrayLike, " dof"] = jnp.asarray(x)
        values: struct.DictArray = struct.DictArray(
            {key: obj.dof.get(x) for key, obj in self.objects.items()}
        )
        return values

    # endregion Make Fields
