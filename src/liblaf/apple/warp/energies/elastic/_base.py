from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Self, cast, no_type_check, overload, override

import jarp
import pyvista as pv
import warp as wp

from liblaf.apple.jax import Region
from liblaf.apple.warp.model import WarpEnergy, WarpEnergyState

from . import func

float_ = Any
vec3 = Any
vec4i = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.kernel
@no_type_check
def _deformation_gradient_kernel(
    u: wp.array1d(dtype=vec3),  # (points,)
    cells: wp.array1d(dtype=vec4i),  # (cells,)
    dhdX: wp.array2d(dtype=mat43),  # (cells, quadrature)
    output: wp.array2d(dtype=mat33),  # (cells, quadrature)
) -> None:
    cid, qid = wp.tid()
    cell = cells[cid]  # vec4i
    u_cell = func.get_cell_displacements(u, cell)
    output[cid, qid] = func.deformation_gradient(u_cell, dhdX[cid, qid])


@jarp.frozen_static
class WarpElastic(WarpEnergy):
    cells: wp.array  # (cells,) vec4i
    dhdX: wp.array  # (cells, quadrature) mat43
    dV: wp.array  # (cells, quadrature) float

    energy_density_func: ClassVar[wp.Function]
    first_piola_kirchhoff_func: ClassVar[wp.Function]
    hess_diag_func: ClassVar[wp.Function]
    hess_prod_func: ClassVar[wp.Function]
    hess_quad_func: ClassVar[wp.Function]

    deformation_gradient_kernel: ClassVar[wp.Kernel] = cast(
        "wp.Kernel", _deformation_gradient_kernel
    )
    energy_density_kernel: ClassVar[wp.Kernel]
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel]

    fun_kernel: ClassVar[wp.Kernel]
    grad_kernel: ClassVar[wp.Kernel]
    hess_prod_kernel: ClassVar[wp.Kernel]
    hess_diag_kernel: ClassVar[wp.Kernel]
    hess_quad_kernel: ClassVar[wp.Kernel]

    @overload
    @classmethod
    def from_pyvista(  # pyright: ignore[reportInconsistentOverload]
        cls,
        obj: pv.DataObject,
        *,
        clamp_hess_diag: bool = True,
        clamp_hess_quad: bool = True,
        clamp_lambda: bool = True,
        requires_grad: Sequence[str] = (),
        **kwargs,
    ) -> Self: ...
    @classmethod
    def from_pyvista(cls, obj: pv.DataObject, **kwargs) -> Self:
        region = Region.from_pyvista(obj, grad=True)
        return cls.from_region(region, **kwargs)

    @overload
    @classmethod
    def from_region(  # pyright: ignore[reportInconsistentOverload]
        cls,
        region: Region,
        *,
        clamp_hess_diag: bool = True,
        clamp_hess_quad: bool = True,
        clamp_lambda: bool = True,
        requires_grad: Sequence[str] = (),
        **kwargs,
    ) -> Self: ...
    @classmethod
    def from_region(
        cls, region: Region, *, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        requires_grad = tuple(requires_grad)
        self: Self = cls(
            cells=jarp.to_warp(region.cells_global, (4, wp.int32)),
            dhdX=jarp.to_warp(region.dhdX, (4, 3, None)),
            dV=jarp.to_warp(region.dV),
            materials=cls.make_materials(region, requires_grad),
            requires_grad=requires_grad,
            **kwargs,
        )
        return self

    @classmethod
    def make_materials(cls, region: Region, requires_grad: Sequence[str]) -> Any:
        raise NotImplementedError

    @classmethod
    def materials_struct(cls) -> Any:
        raise NotImplementedError

    @property
    def launch_dim(self) -> tuple[int, int]:
        return self.dhdX.shape

    @property
    def _kernel_inputs(self) -> Iterable[wp.array]:
        return ()

    def deformation_gradient(
        self,
        state: WarpEnergyState,  # noqa: ARG002
        u: wp.array,
        output: wp.array,
    ) -> None:
        wp.launch(
            self.deformation_gradient_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.dhdX],
            outputs=[output],
        )

    def energy_density(
        self,
        state: WarpEnergyState,  # noqa: ARG002
        u: wp.array,
        output: wp.array,
    ) -> None:
        wp.launch(
            self.energy_density_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.dhdX, self.materials],
            outputs=[output],
        )

    def first_piola_kirchhoff(
        self,
        state: WarpEnergyState,  # noqa: ARG002
        u: wp.array,
        output: wp.array,
    ) -> None:
        wp.launch(
            self.first_piola_kirchhoff_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.dhdX, self.materials],
            outputs=[output],
        )

    @override
    def fun(self, state: WarpEnergyState, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.fun_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.dhdX, self.dV, self.materials],
            outputs=[output],
        )

    @override
    def grad(self, state: WarpEnergyState, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.grad_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.dhdX, self.dV, self.materials],
            outputs=[output],
        )

    @override
    def hess_diag(self, state: WarpEnergyState, u: wp.array, output: wp.array) -> None:
        # return
        ic(self.launch_dim, u, self.cells, self.dhdX, self.dV, self.materials, output)
        wp.launch(
            self.hess_diag_kernel,
            dim=(1, 1),
            inputs=[u, self.cells, self.dhdX, self.dV, self.materials],
            outputs=[output],
        )

    @override
    def hess_prod(
        self, state: WarpEnergyState, u: wp.array, v: wp.array, output: wp.array
    ) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=self.launch_dim,
            inputs=[u, v, self.cells, self.dhdX, self.dV, self.materials],
            outputs=[output],
        )

    @override
    def hess_quad(
        self, state: WarpEnergyState, u: wp.array, v: wp.array, output: wp.array
    ) -> None:
        # return
        wp.launch(
            self.hess_quad_kernel,
            dim=self.launch_dim,
            inputs=[u, v, self.cells, self.dhdX, self.dV, self.materials],
            outputs=[output],
        )

    @classmethod
    def make_energy_density_kernel(
        cls, energy_density_func: wp.Function, module: str | None = None
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def energy_density_kernel(
            u: wp.array1d(dtype=vec3),
            cells: wp.array1d(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            materials: Materials,
            output: wp.array2d(dtype=float_),
        ) -> None:
            cid, qid = wp.tid()
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            output[cid, qid] = energy_density_func(F, materials, cid)

        return cast("wp.Kernel", energy_density_kernel)

    @classmethod
    def make_first_piola_kirchhoff_kernel(
        cls, first_piola_kirchhoff_func: wp.Function, module: str | None = None
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def first_piola_kirchhoff_kernel(
            u: wp.array1d(dtype=vec3),
            cells: wp.array1d(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            materials: Materials,
            output: wp.array2d(dtype=mat33),
        ) -> None:
            cid, qid = wp.tid()
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            P = first_piola_kirchhoff_func(F, materials, cid)  # mat33
            output[cid, qid] = P

        return cast("wp.Kernel", first_piola_kirchhoff_kernel)

    @classmethod
    def make_fun_kernel(
        cls, energy_density_func: wp.Function, module: str | None = None
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def fun_kernel(
            u: wp.array1d(dtype=vec3),
            cells: wp.array1d(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            materials: Materials,
            output: wp.array1d(dtype=float_),
        ) -> None:
            cid, qid = wp.tid()
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            Psi = energy_density_func(F, materials, cid)  # float
            wp.atomic_add(output, 0, Psi * dV[cid, qid])

        return cast("wp.Kernel", fun_kernel)

    @classmethod
    def make_grad_kernel(
        cls, first_piola_kirchhoff_func: wp.Function, module: str | None = None
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def grad_kernel(
            u: wp.array1d(dtype=vec3),
            cells: wp.array1d(dtype=vec4i),
            dhdX: wp.array2d(dtype=mat43),
            dV: wp.array2d(dtype=float_),
            materials: Materials,
            output: wp.array1d(dtype=vec3),
        ) -> None:
            cid, qid = wp.tid()
            # if cid < 3:
            #     wp.printf("grad kernel: %d, %d\n", cid, qid)
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            P = first_piola_kirchhoff_func(F, materials, cid)  # mat33
            grad_cell = (
                func.deformation_gradient_vjp(dhdX_cell, P) * dV[cid, qid]
            )  # mat43
            for i in range(4):
                wp.atomic_add(output, cell[i], grad_cell[i])

        return cast("wp.Kernel", grad_kernel)

    @classmethod
    def make_hess_diag_kernel(
        cls,
        hess_diag_func: wp.Function,
        module: str | None = None,
        *,
        clamp_hess_diag: bool = True,
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def hess_diag_kernel(
            u: wp.array1d(dtype=vec3),  # (points,)
            cells: wp.array1d(dtype=vec4i),  # (cells,)
            dhdX: wp.array2d(dtype=mat43),  # (cells, quadrature points)
            dV: wp.array2d(dtype=float_),  # (cells, quadrature points)
            materials: Materials,
            output: wp.array1d(dtype=vec3),  # (points,)
        ) -> None:
            cid, qid = wp.tid()
            # if cid < 3:
            #     wp.printf("hess_diag kernel: %d, %d\n", cid, qid)
            wp.printf("fun kernel launched: %d, %d\n", cid, qid)
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            h_diag = (
                hess_diag_func(F, dhdX_cell, materials, cid) * dV[cid, qid]
            )  # mat43
            # if wp.static(clamp_hess_diag):
            #     h_diag = math.cw_max_4x(
            #         h_diag, wp.matrix(shape=(4, 3), dtype=h_diag.dtype)
            #     )
            for i in range(4):
                wp.atomic_add(output, cell[i], h_diag[i])

        return cast("wp.Kernel", hess_diag_kernel)

    @classmethod
    def make_hess_prod_kernel(
        cls, hess_prod_func: wp.Function, module: str | None = None
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def hess_prod_kernel(
            u: wp.array1d(dtype=vec3),  # (points,)
            v: wp.array1d(dtype=vec3),  # (points,)
            cells: wp.array1d(dtype=vec4i),  # (cells,)
            dhdX: wp.array2d(dtype=mat43),  # (cells, quadrature points)
            dV: wp.array2d(dtype=float_),  # (cells, quadrature points)
            materials: Materials,
            output: wp.array1d(dtype=vec3),  # (points,)
        ) -> None:
            cid, qid = wp.tid()
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            v_cell = func.get_cell_displacements(v, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            h_prod = (
                hess_prod_func(F, v_cell, dhdX_cell, materials, cid) * dV[cid, qid]
            )  # mat43
            for i in range(4):
                wp.atomic_add(output, cell[i], h_prod[i])

        return cast("wp.Kernel", hess_prod_kernel)

    @classmethod
    def make_hess_quad_kernel(
        cls,
        hess_quad_func: wp.Function,
        module: str | None = None,
        *,
        clamp_hess_quad: bool = True,
    ) -> wp.Kernel:
        # @wp.kernel
        @wp.kernel(module=module)
        @no_type_check
        def hess_quad_kernel(
            u: wp.array1d(dtype=vec3),  # (points,)
            v: wp.array1d(dtype=vec3),  # (points,)
            cells: wp.array1d(dtype=vec4i),  # (cells,)
            dhdX: wp.array2d(dtype=mat43),  # (cells, quadrature)
            dV: wp.array2d(dtype=float_),  # (cells, quadrature)
            materials: Materials,
            output: wp.array1d(dtype=float_),  # (1,)
        ) -> None:
            cid, qid = wp.tid()
            cell = cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            v_cell = func.get_cell_displacements(v, cell)  # mat43
            dhdX_cell = dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            h_quad = (
                hess_quad_func(F, v_cell, dhdX_cell, materials, cid) * dV[cid, qid]
            )  # float
            if wp.static(clamp_hess_quad):
                h_quad = wp.max(h_quad, h_quad.dtype(0.0))
            wp.atomic_add(output, 0, h_quad)

        return cast("wp.Kernel", hess_quad_kernel)
