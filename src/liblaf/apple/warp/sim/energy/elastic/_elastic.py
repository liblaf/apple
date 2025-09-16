from typing import Any, Self, no_type_check, override

import pyvista as pv
import warp as wp

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.quadrature import Scheme
from liblaf.apple.jax.sim.region import Region
from liblaf.apple.warp.sim.energy._energy import Energy
from liblaf.apple.warp.sim.energy.elastic import func
from liblaf.apple.warp.typing import Struct, float_, mat43, vec3, vec4i


@tree.pytree
class Elastic(Energy):
    cells: wp.array = tree.field(default=None)
    dhdX: wp.array = tree.field(default=None)
    dV: wp.array = tree.field(default=None)
    params: Struct = tree.field(default=None)
    quadrature: Scheme = tree.field(default=None)

    energy_density_func: wp.Function = tree.field(default=None)
    energy_density_hess_diag_func: wp.Function = tree.field(default=None)
    energy_density_hess_prod_func: wp.Function = tree.field(default=None)
    energy_density_hess_quad_func: wp.Function = tree.field(default=None)
    first_piola_kirchhoff_stress_func: wp.Function = tree.field(default=None)
    get_cell_params: wp.Function = tree.field(default=None)

    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        self: Self = cls(
            cells=wp.from_jax(region.cells_global, vec4i),
            dhdX=wp.from_jax(region.dhdX, mat43),
            dV=wp.from_jax(region.dV, float_),
            quadrature=region.quadrature,
            **kwargs,
        )
        if self.params is None:
            self.params = self.make_params(region)
        return self

    @classmethod
    def from_pyvista(
        cls, mesh: pv.UnstructuredGrid, *, quadrature: Scheme | None = None, **kwargs
    ) -> Self:
        region: Region = Region.from_pyvista(mesh, grad=True, quadrature=quadrature)
        return cls.from_region(region, **kwargs)

    def __attrs_post_init__(self) -> None:
        for name in self.requires_grad:
            arr: wp.array = getattr(self.params, name)
            arr.requires_grad = True

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def fun_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            output: wp.array(dtype=float_),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            output[0] += dV[cid, qid] * self.energy_density_func(F, cell_params)

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def jac_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            output: wp.array(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            PK1 = self.first_piola_kirchhoff_stress_func(F, cell_params)  # mat33
            jac = dV[cid, qid] * func.deformation_gradient_vjp(
                dhdX[cid, qid], PK1
            )  # mat43
            for i in range(4):
                output[vid[i]] += jac[i]

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def hess_diag_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            output: wp.array(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            hess_diag = dV[cid, qid] * self.energy_density_hess_diag_func(
                F, dhdX[cid, qid], cell_params
            )  # mat43
            for i in range(4):
                output[vid[i]] += hess_diag[i]

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def hess_prod_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            p: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            output: wp.array(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            p0 = p[vid[0]]  # vec3
            p1 = p[vid[1]]  # vec3
            p2 = p[vid[2]]  # vec3
            p3 = p[vid[3]]  # vec3
            p_cell = wp.matrix_from_rows(p0, p1, p2, p3)  # mat43
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            hess_prod = dV[cid, qid] * self.energy_density_hess_prod_func(
                F, p_cell, dhdX[cid, qid], cell_params
            )  # mat43
            for i in range(4):
                output[vid[i]] += hess_prod[i]

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def hess_quad_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            p: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            output: wp.array(dtype=float_),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            p0 = p[vid[0]]  # vec3
            p1 = p[vid[1]]  # vec3
            p2 = p[vid[2]]  # vec3
            p3 = p[vid[3]]  # vec3
            p_cell = wp.matrix_from_rows(p0, p1, p2, p3)  # mat43
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            output[0] += dV[cid, qid] * self.energy_density_hess_quad_func(
                F, p_cell, dhdX[cid, qid], cell_params
            )

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def fun_and_jac_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            fun: wp.array(dtype=float_),
            jac: wp.array(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            fun[0] += dV[cid, qid] * self.energy_density_func(F, cell_params)
            PK1 = self.first_piola_kirchhoff_stress_func(F, cell_params)  # mat33
            jac_cell = dV[cid, qid] * func.deformation_gradient_vjp(
                dhdX[cid, qid], PK1
            )  # mat43
            for i in range(4):
                jac[vid[i]] += jac_cell[i]

        return kernel  # pyright: ignore[reportReturnType]

    @property
    def jac_and_hess_diag_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            params: Any,
            jac: wp.array(dtype=vec3),
            hess_diag: wp.array(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            vid = cells[cid]  # vec4i
            u0 = u[vid[0]]  # vec3
            u1 = u[vid[1]]  # vec3
            u2 = u[vid[2]]  # vec3
            u3 = u[vid[3]]  # vec3
            u_cell = wp.matrix_from_rows(u0, u1, u2, u3)  # mat43
            F = func.deformation_gradient(u_cell, dhdX[cid, qid])  # mat33
            cell_params = self.get_cell_params(params, cid)  # ParamsElem
            PK1 = self.first_piola_kirchhoff_stress_func(F, cell_params)  # mat33
            jac_cell = dV[cid, qid] * func.deformation_gradient_vjp(
                dhdX[cid, qid], PK1
            )  # mat43
            hess_diag_cell = dV[cid, qid] * self.energy_density_hess_diag_func(
                F, dhdX[cid, qid], cell_params
            )  # mat43
            for i in range(4):
                jac[vid[i]] += jac_cell[i]
                hess_diag[vid[i]] += hess_diag_cell[i]

        return kernel  # pyright: ignore[reportReturnType]

    def make_params(self, region: Region) -> Struct:
        raise NotImplementedError

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.fun_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def jac(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.jac_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_diag_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, p, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_quad_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, p, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def fun_and_jac(self, u: wp.array, fun: wp.array, jac: wp.array) -> None:
        wp.launch(
            self.fun_and_jac_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[fun, jac],
        )

    @override
    def jac_and_hess_diag(
        self, u: wp.array, jac: wp.array, hess_diag: wp.array
    ) -> None:
        wp.launch(
            self.jac_and_hess_diag_kernel,
            dim=(self.n_cells, self.quadrature.n_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[jac, hess_diag],
        )
