import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, PyTree

from liblaf.apple import optim

from ._object import Object


@attrs.define
class Physics:
    dirichlet_mask: Bool[jax.Array, " T"]
    objects: dict[str, Object] = attrs.field(factory=dict)

    @property
    def free_mask(self) -> Bool[jax.Array, " T"]:
        return ~self.dirichlet_mask

    @property
    def n_dirichlet(self) -> int:
        return jnp.count_nonzero(self.dirichlet_mask)  # pyright: ignore[reportReturnType]

    @property
    def n_free(self) -> int:
        return self.n_total - self.n_dirichlet

    @property
    def n_total(self) -> int:
        return self.dirichlet_mask.size

    def add(self, obj: Object) -> None:
        self.objects[obj.name] = obj

    def prepare(self, q: PyTree) -> PyTree:
        aux: PyTree = {}
        for obj in self.objects.values():
            aux[obj.name] = obj.prepare()
        return aux

    def solve(
        self,
        q: PyTree,
        method: optim.Optimizer | None = None,
        callback: optim.Callback | None = None,
    ) -> optim.OptimizeResult:
        aux: PyTree = self.prepare(q)
        return optim.minimize(
            self.fun,
            self.initial(q)[self.free_mask],
            args=(q, aux),
            method=method,
            jac=self.jac,
            hessp=self.hessp,
            hess_diag=self.hess_diag,
            hess_quad=self.hess_quad,
            jac_and_hess_diag=self.jac_and_hess_diag,
            callback=callback,
        )

    def fun(
        self, u: Float[jax.Array, " N"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        u: Float[jax.Array, " N"] = self.fill_dirichlet(u, q)
        return jnp.sum(
            jnp.asarray(
                [
                    obj.fun(u[obj.dof_id], q[obj.name], aux[obj.name])
                    for obj in self.objects.values()
                ]
            )
        )

    def jac(
        self, u: Float[jax.Array, " N"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, " N"]:
        u: Float[jax.Array, " N"] = self.fill_dirichlet(u, q)
        jac: Float[jax.Array, " N"] = jnp.zeros((self.n_total,))
        for obj in self.objects.values():
            jac = jac.at[obj.dof_id].add(
                obj.jac(u[obj.dof_id], q[obj.name], aux[obj.name])
            )
        return jac

    def hess(self, x: Float[jax.Array, " N"], q: PyTree, aux: PyTree) -> None:
        raise NotImplementedError

    def hessp(
        self,
        x: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, " N"]:
        u: Float[jax.Array, " T"] = self.fill_dirichlet(x, q)
        p: Float[jax.Array, " T"] = self.fill_zeros(p)
        hessp: Float[jax.Array, " T"] = jnp.zeros((self.n_total,))
        for obj in self.objects.values():
            hessp = hessp.at[obj.dof_id].add(
                obj.hessp(u[obj.dof_id], p[obj.dof_id], q[obj.name], aux[obj.name])
            )
        return hessp[self.free_mask]

    def hess_diag(self, x: Float[jax.Array, " N"], q: PyTree, aux: PyTree) -> None:
        raise NotImplementedError

    def hess_quad(
        self,
        x: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        raise NotImplementedError

    def jac_and_hess_diag(
        self, x: Float[jax.Array, " N"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
        raise NotImplementedError

    def fill_dirichlet(
        self, x: Float[jax.Array, " N"], q: PyTree
    ) -> Float[jax.Array, " T"]:
        x = jnp.asarray(x)
        initial: Float[jax.Array, " T"] = self.initial(q)
        x = initial.at[self.free_mask].set(x)
        return x

    def fill_zeros(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, " T"]:
        x = jnp.asarray(x)
        zeros: Float[jax.Array, " T"] = jnp.zeros((self.n_total,))
        x = zeros.at[self.free_mask].set(x)
        return x

    def initial(self, q: PyTree) -> Float[jax.Array, " T"]:
        if (initial_or_none := q.get("initial")) is not None:
            return jnp.asarray(initial_or_none)
        return jnp.zeros((self.n_total,))
