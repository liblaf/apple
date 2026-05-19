from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self, cast, no_type_check, overload, override

import attrs
import pyvista as pv
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math
from liblaf.apple.warp.model import WarpPotential
from liblaf.apple.warp.utils import warp_default_dtype, warp_struct

from . import func, utils

floating = Any
mat33 = Any
mat43 = Any
vec3 = Any
vec4i = Any
Materials = Any
WarpRegion = Any


@wp.kernel
@no_type_check
def _deformation_gradient_kernel(
    u: wp.array1d[vec3],  # (points,)
    cells: wp.array1d[vec4i],  # (cells,)
    dhdX: wp.array2d[mat43],  # (cells, quadrature)
    output: wp.array2d[mat33],  # (cells, quadrature)
) -> None:
    cid, qid = wp.tid()
    cell = cells[cid]  # vec4i
    u_cell = func.get_cell_displacements(u, cell)
    output[cid, qid] = func.deformation_gradient(u_cell, dhdX[cid, qid])


@attrs.define
class WarpPotentialFem(WarpPotential):
    @warp_struct
    class _Region:
        cells: wp.array
        """(cells,)"""
        dhdX: wp.array
        """(cells, quadrature)"""
        dV: wp.array
        """(cells, quadrature)"""

        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            return {
                "cells": wp.array1d(dtype=wp.vec4i),
                "dhdX": wp.array2d(dtype=wp.types.matrix((4, 3), dtype)),
                "dV": wp.array2d(dtype=dtype),
            }

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

    region: _Region

    @overload
    @classmethod
    def from_pyvista(  # ty:ignore[invalid-overload]
        cls, obj: pv.DataObject, *, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self: ...
    @classmethod
    def from_pyvista(cls, obj: pv.DataObject, **kwargs) -> Self:
        region: Region = Region.from_pyvista(obj)
        return cls.from_region(region, **kwargs)

    @overload
    @classmethod
    def from_region(  # ty:ignore[invalid-overload]
        cls, region: Region, *, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self: ...
    @classmethod
    def from_region(
        cls, region: Region, *, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        dtype: Any = warp_default_dtype()
        requires_grad = tuple(requires_grad)
        fraction: Float[Tensor, " cells"] = utils.get_fraction(region)
        region_wp: WarpPotentialFem._Region = cls._Region()
        region_wp.cells = wp.from_torch(region.cells_global, wp.vec4i)
        region_wp.dhdX = wp.from_torch(region.dhdX, wp.types.matrix((4, 3), dtype))
        region_wp.dV = wp.from_torch(fraction[:, None] * region.dV, dtype)
        self: Self = cls(
            region=region_wp,
            materials=cls.materials_from_region(region, requires_grad),
            requires_grad=requires_grad,
            **kwargs,
        )
        return self

    @classmethod
    def materials_from_region(cls, region: Region, requires_grad: Sequence[str]) -> Any:
        raise NotImplementedError

    @property
    def launch_dim(self) -> tuple[int, int]:
        return self.region.dhdX.shape

    def deformation_gradient(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.deformation_gradient_kernel,
            dim=self.launch_dim,
            inputs=[u, self.region.cells, self.region.dhdX],
            outputs=[output],
        )

    def energy_density(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.energy_density_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                self.region.cells,
                self.region.dhdX,
                self.region.dV,
                self.materials,
            ],
            outputs=[output],
        )

    def first_piola_kirchhoff(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.first_piola_kirchhoff_kernel,
            dim=self.launch_dim,
            inputs=[u, self.region.cells, self.region.dhdX, self.materials],
            outputs=[output],
        )

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.fun_kernel,
            dim=self.launch_dim,
            inputs=[u, self.region, self.materials],
            outputs=[output],
        )

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.grad_kernel,
            dim=self.launch_dim,
            inputs=[u, self.region, self.materials],
            outputs=[output],
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_diag_kernel,
            dim=self.launch_dim,
            inputs=[u, self.region, self.materials],
            outputs=[output],
        )

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.region, self.materials],
            outputs=[output],
        )

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_quad_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.region, self.materials],
            outputs=[output],
        )

    @classmethod
    def make_energy_density_kernel(
        cls, energy_density_func: Callable[..., Any], module: str | None = "unique"
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def energy_density_kernel(
            u: wp.array1d[vec3],
            cells: wp.array1d[vec4i],
            dhdX: wp.array2d[mat43],
            materials: Materials,
            output: wp.array2d[floating],
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
        cls,
        first_piola_kirchhoff_func: Callable[..., Any],
        module: str | None = "unique",
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def first_piola_kirchhoff_kernel(
            u: wp.array1d[vec3],
            cells: wp.array1d[vec4i],
            dhdX: wp.array2d[mat43],
            materials: Materials,
            output: wp.array2d[mat33],
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
        cls, energy_density_func: Callable[..., Any], module: str | None = "unique"
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def fun_kernel(
            u: wp.array1d[vec3],
            region: WarpRegion,
            materials: Materials,
            output: wp.array1d[floating],
        ) -> None:
            cid, qid = wp.tid()
            cell = region.cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = region.dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            Psi = energy_density_func(F, materials, cid)  # float
            wp.atomic_add(output, 0, Psi * region.dV[cid, qid])

        return cast("wp.Kernel", fun_kernel)

    @classmethod
    def make_grad_kernel(
        cls,
        first_piola_kirchhoff_func: Callable[..., Any],
        module: str | None = "unique",
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def grad_kernel(
            u: wp.array1d[vec3],
            region: WarpRegion,
            materials: Materials,
            output: wp.array1d[vec3],
        ) -> None:
            cid, qid = wp.tid()
            cell = region.cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = region.dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            P = first_piola_kirchhoff_func(F, materials, cid)  # mat33
            grad_cell = (
                func.deformation_gradient_vjp(dhdX_cell, P) * region.dV[cid, qid]
            )  # mat43
            for i in range(4):
                wp.atomic_add(output, cell[i], grad_cell[i])

        return cast("wp.Kernel", grad_kernel)

    @classmethod
    def make_hess_diag_kernel(
        cls,
        hess_diag_func: Callable[..., Any],
        module: str | None = "unique",
        *,
        clamp_hess_diag: bool = True,
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def hess_diag_kernel(
            u: wp.array1d[vec3],  # (points,)
            region: WarpRegion,
            materials: Materials,
            output: wp.array1d[vec3],  # (points,)
        ) -> None:
            cid, qid = wp.tid()
            cell = region.cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            dhdX_cell = region.dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            H_diag = (
                hess_diag_func(F, dhdX_cell, materials, cid) * region.dV[cid, qid]
            )  # mat43
            if wp.static(clamp_hess_diag):
                H_diag = math.cw_max_4x(
                    H_diag, wp.matrix(shape=(4, 3), dtype=H_diag.dtype)
                )
            for i in range(4):
                wp.atomic_add(output, cell[i], H_diag[i])

        return cast("wp.Kernel", hess_diag_kernel)

    @classmethod
    def make_hess_prod_kernel(
        cls, hess_prod_func: Callable[..., Any], module: str | None = "unique"
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def hess_prod_kernel(
            u: wp.array1d[vec3],  # (points,)
            v: wp.array1d[vec3],  # (points,)
            region: WarpRegion,
            materials: Materials,
            output: wp.array1d[vec3],  # (points,)
        ) -> None:
            cid, qid = wp.tid()
            cell = region.cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            v_cell = func.get_cell_displacements(v, cell)  # mat43
            dhdX_cell = region.dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            h_prod = region.dV[cid, qid] * hess_prod_func(
                F, v_cell, dhdX_cell, materials, cid
            )  # mat43
            for i in range(4):
                wp.atomic_add(output, cell[i], h_prod[i])

        return cast("wp.Kernel", hess_prod_kernel)

    @classmethod
    def make_hess_quad_kernel(
        cls,
        hess_quad_func: Callable[..., Any],
        module: str | None = "unique",
        *,
        clamp_hess_quad: bool = True,
    ) -> wp.Kernel:
        @wp.kernel(module=module)
        @no_type_check
        def hess_quad_kernel(
            u: wp.array1d[vec3],  # (points,)
            v: wp.array1d[vec3],  # (points,)
            region: WarpRegion,
            materials: Materials,
            output: wp.array1d[floating],  # (1,)
        ) -> None:
            cid, qid = wp.tid()
            cell = region.cells[cid]  # vec4i
            u_cell = func.get_cell_displacements(u, cell)  # mat43
            v_cell = func.get_cell_displacements(v, cell)  # mat43
            dhdX_cell = region.dhdX[cid, qid]  # mat43
            F = func.deformation_gradient(u_cell, dhdX_cell)  # mat33
            h_quad = region.dV[cid, qid] * hess_quad_func(
                F, v_cell, dhdX_cell, materials, cid
            )  # float
            if wp.static(clamp_hess_quad):
                h_quad = wp.max(h_quad, h_quad.dtype(0.0))
            wp.atomic_add(output, 0, h_quad)

        return cast("wp.Kernel", hess_quad_kernel)
