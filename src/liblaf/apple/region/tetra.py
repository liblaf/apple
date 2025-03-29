import attrs
import einops
import jax
import jax.numpy as jnp
import numpy as np
import pylops
import pyvista as pv
from jaxtyping import Bool, Float, Integer, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class RegionTetra:
    aux: PyTree = attrs.field(factory=dict)
    params: PyTree = attrs.field(factory=dict)
    dirichlet_values: Float[jax.Array, "V 3"] = attrs.field(converter=jnp.asarray)
    dirichlet_mask: Bool[np.ndarray, "V 3"] = attrs.field(
        metadata={"static": True}, converter=np.asarray
    )
    material: apple.material.MaterialTetra = attrs.field(metadata={"static": True})
    mesh: pv.UnstructuredGrid = attrs.field(metadata={"static": True})

    @property
    def cells(self) -> Integer[jax.Array, "c a"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])

    @property
    def free_mask(self) -> Bool[np.ndarray, "V 3"]:
        return ~self.dirichlet_mask

    @property
    def n_dof(self) -> int:
        return jnp.count_nonzero(self.free_mask)  # pyright: ignore[reportReturnType]

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def points(self) -> Float[jax.Array, "V 3"]:
        return jnp.asarray(self.mesh.points)

    def prepare(self) -> None:
        self.aux = {
            "dh_dX": apple.elem.tetra.dh_dX(self.points[self.cells]),
            "dV": apple.elem.tetra.dV(self.points[self.cells]),
        }

    @apple.jit()
    def fun(self, u: Float[jax.Array, " F"], q: PyTree) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "V 3"] = self.fill(u)
        return self.material.fun(u[self.cells], q, self.aux)

    @apple.jit()
    def jac(self, u: Float[jax.Array, " F"], q: PyTree) -> Float[jax.Array, " F"]:
        u: Float[jax.Array, "V 3"] = self.fill(u)
        jac: Float[jax.Array, "C 3"] = self.material.jac(u[self.cells], q, self.aux)
        jac: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            jac, self.cells, self.n_points
        )
        return jac[self.free_mask]

    @apple.jit()
    def fun_jac(
        self, u: Float[jax.Array, " F"], q: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " F"]]:
        u: Float[jax.Array, "V 3"] = self.fill(u)
        fun: Float[jax.Array, ""]
        jac: Float[jax.Array, "C 4 3"]
        fun, jac = self.material.fun_jac(u[self.cells], q, self.aux)
        jac = apple.elem.tetra.segment_sum(jac, self.cells, self.n_points)
        return fun, jac[self.free_mask]

    def hess(
        self, u: Float[jax.Array, " F"], q: PyTree
    ) -> Float[pylops.LinearOperator, "F F"]:
        u: Float[jax.Array, "V 3"] = self.fill(u)
        hess: Float[jax.Array, "C 4 3 4 3"] = self.material.hess(
            u[self.cells], q, self.aux
        )

        def matvec(v: Float[jax.Array, " F"]) -> Float[jax.Array, " F"]:
            v: Float[jax.Array, " F"] = jnp.asarray(v, dtype=hess.dtype)
            return self._hvp(hess, v)

        return pylops.FunctionOperator(
            matvec, matvec, self.n_dof, self.n_dof, dtype=hess.dtype, name="hessian"
        )

    @apple.jit()
    def hess_diag(self, u: Float[jax.Array, " F"], q: PyTree) -> Float[jax.Array, " F"]:
        u: Float[jax.Array, "V 3"] = self.fill(u)
        hess_diag: Float[jax.Array, "C 4 3"] = self.material.hess_diag(
            u[self.cells], q, self.aux
        )
        hess_diag: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            hess_diag, self.cells, self.n_points
        )
        return hess_diag[self.free_mask]

    def fill(self, u_free: Float[jax.Array, " F"]) -> Float[jax.Array, "V 3"]:
        u: Float[jax.Array, "V 3"] = self.dirichlet_values.copy()
        u = u.at[self.free_mask].set(u_free)
        return u

    @apple.jit()
    def _hvp(
        self, hess: Float[jax.Array, "C 4 3 4 3"], v: Float[jax.Array, " F"]
    ) -> Float[jax.Array, " F"]:
        ic("compiling ...")
        v: Float[jax.Array, "V 3"] = self.fill(v)
        hvp: Float[jax.Array, "C 4 3"] = einops.einsum(
            hess, v[self.cells], "C i j k l, C i j -> C k l"
        )
        hvp: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            hvp, self.cells, self.n_points
        )
        return hvp[self.free_mask]
