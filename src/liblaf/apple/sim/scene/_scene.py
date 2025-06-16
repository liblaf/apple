from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import optim, struct, utils
from liblaf.apple.sim.abc import (
    Dirichlet,
    Energy,
    FieldCollection,
    GlobalParams,
    Object,
)

from ._optim import OptimizationProblem


class Scene(struct.NodeCollectionMixin, struct.PyTree):
    nodes: struct.NodeCollection = struct.data(factory=struct.NodeCollection)
    params: GlobalParams = struct.data(default=GlobalParams())
    topological: Sequence[str] = struct.static(default=())
    _base_keys: Iterable[str] = struct.static(default=())
    _energy_keys: Iterable[str] = struct.static(default=())

    dirichlet: Dirichlet = struct.data(default=None)

    # region NodeCollectionMixin

    @override
    def __getitem__(self, key: struct.KeyLike, /) -> struct.Node:
        return self.nodes[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.nodes

    @override
    def __len__(self) -> int:
        return len(self.nodes)

    # endregion NodeCollectionMixin

    # region Structure

    @property
    def bases(self) -> struct.NodeCollection[Object]:
        return self.select(self._base_keys)

    @property
    def dof_index(self) -> Mapping[str, struct.Index]:
        return {obj.id: obj.dof_index for obj in self.objects.values()}

    @property
    def energies(self) -> struct.NodeCollection[Energy]:
        return self.select(self._energy_keys)

    @property
    def objects(self) -> struct.NodeCollection[Object]:
        return self.filter_instance(Object)

    # endregion Structure

    # region Shape

    @property
    def n_dof(self) -> int:
        return sum(obj.n_dof for obj in self.bases.values())

    # endregion Shape

    # region Array

    @property
    @utils.jit
    def x0(self) -> Float[jax.Array, " DoF"]:
        values: Float[jax.Array, " DoF"] = jnp.zeros((self.n_dof,))
        for obj in self.bases.values():
            values = obj.dof_index.set(values, obj.displacement.values)
        return values

    # endregion Array

    # region Optimization

    def make_problem(self, callback: optim.Callback) -> "OptimizationProblem":
        from ._optim import OptimizationProblem

        return OptimizationProblem(self, callback=callback)

    @utils.jit
    def fun(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, ""]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fun: Float[jax.Array, ""] = jnp.asarray(0.0)
        for energy in self.energies.values():
            deps: FieldCollection = fields.select(energy.deps.keys())
            fun += energy.fun(deps, self.params)
        return fun

    @utils.jit
    def jac(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        jac = FieldCollection()
        for energy in self.energies.values():
            deps: FieldCollection = fields.select(energy.deps.keys())
            energy_jac: FieldCollection = energy.jac(deps, self.params)
            jac += energy_jac
        jac_values: jax.Array = jac.gather(self.dof_index, n_dof=self.n_dof)
        if self.dirichlet is not None:
            jac_values = self.dirichlet.apply(jac_values)
        return jac_values

    @utils.jit
    def hessp(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fields_p: FieldCollection = self.make_fields_dirichlet_zero(p)
        hessp = FieldCollection()
        for energy in self.energies.values():
            deps: FieldCollection = fields.select(energy.deps.keys())
            deps_p: FieldCollection = fields_p.select(energy.deps.keys())
            energy_hessp: FieldCollection = energy.hessp(deps, deps_p, self.params)
            hessp += energy_hessp
        return hessp.gather(self.dof_index, n_dof=self.n_dof)

    @utils.jit
    def hess_diag(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        hess_diag = FieldCollection()
        for energy in self.energies.values():
            deps: FieldCollection = fields.select(energy.deps.keys())
            energy_hess_diag: FieldCollection = energy.hess_diag(deps, self.params)
            hess_diag += energy_hess_diag
        return hess_diag.gather(self.dof_index, n_dof=self.n_dof)

    @utils.jit
    def hess_quad(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, ""]:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        fields_p: FieldCollection = self.make_fields_dirichlet_zero(p)
        hess_quad: Float[jax.Array, ""] = jnp.asarray(0.0)
        for energy in self.energies.values():
            deps: FieldCollection = fields.select(energy.deps.keys())
            deps_p: FieldCollection = fields_p.select(energy.deps.keys())
            energy_hess_quad: Float[jax.Array, ""] = energy.hess_quad(
                deps, deps_p, self.params
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

    def solve(
        self, solver: optim.Optimizer, callback: optim.Callback | None = None
    ) -> optim.OptimizeResult:
        problem: OptimizationProblem = self.make_problem(callback=callback)
        result: optim.OptimizeResult = solver.minimize(
            problem.fun,
            self.x0,
            jac=problem.jac,
            hessp=problem.hessp,
            hess_diag=problem.hess_diag,
            hess_quad=problem.hess_quad,
            jac_and_hess_diag=problem.jac_and_hess_diag,
            callback=problem.callback,
        )
        return result

    # endregion Optimization

    # region State Update

    def graph_update(self, bases: struct.NodeCollection, /) -> Self:
        nodes: struct.NodeCollection = self.nodes.update(bases)
        nodes = struct.graph_update(nodes, self.topological)
        return self.evolve(nodes=nodes)

    def prepare(self, x: Float[ArrayLike, " DoF"], /) -> Self:
        scene: Self = self.update(x)
        nodes: struct.NodeCollection = struct.NodeCollection(
            node.prepare() for node in scene.nodes.values()
        )
        return self.evolve(nodes=nodes)

    def step(self, x: Float[ArrayLike, " DoF"], /) -> Self:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        bases: struct.NodeCollection[Object] = struct.NodeCollection(
            obj.step(fields[obj.id], self.params) for obj in self.bases.values()
        )
        return self.graph_update(bases)

    def update(self, x: Float[ArrayLike, " DoF"], /) -> Self:
        fields: FieldCollection = self.make_fields_dirichlet(x)
        bases: struct.NodeCollection[Object] = struct.NodeCollection(
            obj.update(fields[obj.id]) for obj in self.bases.values()
        )
        return self.graph_update(bases)

    # endregion State Update

    # region Make Fields

    def make_fields(self, x: Float[ArrayLike, " DoF"], /) -> FieldCollection:
        return FieldCollection(
            {
                key: obj.displacement.with_values(obj.dof_index(x))
                for key, obj in self.objects.items()
            }
        )

    def make_fields_dirichlet(self, x: Float[ArrayLike, " DoF"], /) -> FieldCollection:
        if self.dirichlet is not None:
            x = self.dirichlet.apply(x)
        return self.make_fields(x)

    def make_fields_dirichlet_zero(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> FieldCollection:
        if self.dirichlet is not None:
            x = self.dirichlet.zero(x)
        return self.make_fields(x)

    # endregion Make Fields
