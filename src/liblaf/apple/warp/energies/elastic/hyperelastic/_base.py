import functools
from typing import Any, Self, no_type_check, override

import pyvista as pv
import warp as wp
from jaxtyping import Float, Integer
from liblaf.peach import tree

from liblaf.apple.jax.fem import Region
from liblaf.apple.warp import types as _t
from liblaf.apple.warp import utils
from liblaf.apple.warp.model import WarpEnergy

from . import func

type Vector = Float[wp.array, "points dim"]
type Scalar = Float[wp.array, "1"]
mat33 = Any
mat43 = Any
scalar = Any
vec3 = Any
vec4i = Any


@tree.define
class Hyperelastic(WarpEnergy):
    @wp.struct
    class Params:
        pass

    @wp.struct
    class ParamsElem:
        pass

    cells: Integer[wp.array, "c a"]
    dhdX: Integer[wp.array, "c q a J"]
    dV: Integer[wp.array, "c q"]
    params: Params  # pyright: ignore[reportGeneralTypeIssues]

    clamp_hess_diag: bool = True
    clamp_hess_quad: bool = True
    clamp_lambda: bool = True

    @classmethod
    def from_pyvista(cls, obj: pv.DataObject, **kwargs) -> Self:
        region = Region.from_pyvista(obj, grad=True)
        return cls.from_region(region, **kwargs)

    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        self: Self = cls(
            cells=utils.to_warp(region.cells, _t.vec4i),
            dhdX=utils.to_warp(region.dhdX, _t.mat43),
            dV=utils.to_warp(region.dV, _t.float_),
            params=cls.make_params(region),
            **kwargs,
        )
        return self

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_quadrature_points(self) -> int:
        return self.dV.shape[1]

    @override
    def fun(self, u: Vector, output: Scalar) -> None:
        wp.launch(
            self.fun_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def grad(self, u: Vector, output: Vector) -> None:
        wp.launch(
            self.grad_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_diag(self, u: Vector, output: Vector) -> None:
        wp.launch(
            self.hess_diag_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_prod(self, u: Vector, p: Vector, output: Vector) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, p, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def hess_quad(self, u: Vector, p: Vector, output: Scalar) -> None:
        wp.launch(
            self.hess_quad_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, p, self.cells, self.dhdX, self.dV, self.params],
            outputs=[output],
        )

    @override
    def mixed_derivative_prod(self, u: wp.array, p: wp.array) -> dict[str, wp.array]:
        if not self.requires_grad:
            return {}
        for name in self.requires_grad:
            getattr(self.params, name).grad.zero_()
        output: wp.array = wp.zeros_like(u)
        with wp.Tape() as tape:
            self.grad(u, output)
        tape.backward(grads={output: p})
        outputs: dict[str, wp.array] = {
            name: getattr(self.params, name).grad for name in self.requires_grad
        }
        return outputs

    @override
    def value_and_grad(self, u: Vector, value: Scalar, grad: Vector) -> None:
        wp.launch(
            self.value_and_grad_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[value, grad],
        )

    @override
    def grad_and_hess_diag(self, u: Vector, grad: Vector, hess_diag: Vector) -> None:
        wp.launch(
            self.grad_and_hess_diag_kernel,
            dim=(self.n_cells, self.n_quadrature_points),
            inputs=[u, self.cells, self.dhdX, self.dV, self.params],
            outputs=[grad, hess_diag],
        )

    @functools.cached_property
    def fun_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
            output: wp.array(dtype=scalar),
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

    @functools.cached_property
    def grad_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
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
            grad = dV[cid, qid] * func.deformation_gradient_vjp(
                dhdX[cid, qid], PK1
            )  # mat43
            for i in range(4):
                output[vid[i]] += grad[i]

        return kernel  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def hess_diag_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
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
                F, dhdX[cid, qid], cell_params, clamp=wp.static(self.clamp_lambda)
            )  # mat43
            if wp.static(self.clamp_hess_diag):
                zero_vec3 = wp.vector(length=3, dtype=hess_diag.dtype)
                hess_diag = wp.matrix_from_rows(
                    wp.max(hess_diag[0], zero_vec3),
                    wp.max(hess_diag[1], zero_vec3),
                    wp.max(hess_diag[2], zero_vec3),
                    wp.max(hess_diag[3], zero_vec3),
                )
            for i in range(4):
                output[vid[i]] += hess_diag[i]

        return kernel  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def hess_prod_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            p: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
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
                F,
                p_cell,
                dhdX[cid, qid],
                cell_params,
                clamp=wp.static(self.clamp_lambda),
            )  # mat43
            for i in range(4):
                output[vid[i]] += hess_prod[i]

        return kernel  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def hess_quad_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            p: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
            output: wp.array(dtype=scalar),
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
            hess_quad = dV[cid, qid] * self.energy_density_hess_quad_func(
                F,
                p_cell,
                dhdX[cid, qid],
                cell_params,
                clamp=wp.static(self.clamp_lambda),
            )
            if wp.static(self.clamp_hess_quad):
                zero_scalar = hess_quad.dtype(0.0)
                hess_quad = wp.max(hess_quad, zero_scalar)
            output[0] += hess_quad

        return kernel  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def value_and_grad_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
            value: wp.array(dtype=scalar),
            grad: wp.array(dtype=vec3),
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
            value[0] += dV[cid, qid] * self.energy_density_func(F, cell_params)
            PK1 = self.first_piola_kirchhoff_stress_func(F, cell_params)  # mat33
            jac_cell = dV[cid, qid] * func.deformation_gradient_vjp(
                dhdX[cid, qid], PK1
            )  # mat43
            for i in range(4):
                grad[vid[i]] += jac_cell[i]

        return kernel  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def grad_and_hess_diag_kernel(self) -> wp.Kernel:
        @wp.kernel
        @no_type_check
        def kernel(
            u: wp.array(dtype=vec3),
            cells: wp.array(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=scalar),
            params: self.Params,
            grad: wp.array(dtype=vec3),
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
                F, dhdX[cid, qid], cell_params, clamp=wp.static(self.clamp_lambda)
            )  # mat43
            if wp.static(self.clamp_hess_diag):
                zero_vec3 = wp.vector(length=3, dtype=hess_diag_cell.dtype)
                hess_diag_cell = wp.matrix_from_rows(
                    wp.max(hess_diag_cell[0], zero_vec3),
                    wp.max(hess_diag_cell[1], zero_vec3),
                    wp.max(hess_diag_cell[2], zero_vec3),
                    wp.max(hess_diag_cell[3], zero_vec3),
                )
            for i in range(4):
                grad[vid[i]] += jac_cell[i]
                hess_diag[vid[i]] += hess_diag_cell[i]

        return kernel  # pyright: ignore[reportReturnType]

    @classmethod
    @no_type_check
    def make_params(cls, region: Region) -> Params:  # noqa: ARG003
        return cls.Params()

    @staticmethod
    @wp.func
    @no_type_check
    def get_cell_params(params: Params, cid: int) -> ParamsElem:
        raise NotImplementedError

    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_func(F: mat33, params: ParamsElem) -> scalar:
        raise NotImplementedError

    @staticmethod
    @wp.func
    @no_type_check
    def first_piola_kirchhoff_stress_func(F: mat33, params: ParamsElem) -> mat33:
        raise NotImplementedError

    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_diag_func(
        F: mat33, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        raise NotImplementedError

    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_prod_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> mat33:
        raise NotImplementedError

    @staticmethod
    @wp.func
    @no_type_check
    def energy_density_hess_quad_func(
        F: mat33, p: mat43, dhdX: mat43, params: ParamsElem, *, clamp: bool = True
    ) -> scalar:
        raise NotImplementedError
