from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, no_type_check, override

import attrs
import warp as wp

from liblaf.apple.common import FORCE, GLOBAL_POINT_ID
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.model import MaterialField, WarpPotential

floating = Any
vec3 = Any


@wp.kernel
@no_type_check
def fun_kernel(
    force: wp.array1d[vec3],
    indices: wp.array1d[wp.int32],
    u: wp.array1d[vec3],
    output: wp.array1d[floating],
) -> None:
    tid = wp.tid()  # int
    vid = indices[tid]  # int
    W = -wp.dot(force[tid], u[vid])  # float
    wp.atomic_add(output, 0, W)


@wp.kernel
@no_type_check
def grad_kernel(
    force: wp.array1d[vec3], indices: wp.array1d[wp.int32], output: wp.array1d[vec3]
) -> None:
    tid = wp.tid()  # int
    vid = indices[tid]  # int
    f = -force[tid]  # vec3
    wp.atomic_add(output, vid, f)


@attrs.define
class ExternalForce(WarpPotential):
    class Materials(WarpPotential.Materials):
        force: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotential.MATERIAL_FIELDS,
        FORCE.value: MaterialField.POINT.vec3(FORCE.value),
    }

    indices: wp.array[wp.int32]
    materials: Materials = attrs.field(default=None, kw_only=True)

    @classmethod
    @override
    def from_region(
        cls, region: Region, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        indices: wp.array = wp.from_numpy(
            region.mesh.point_data[GLOBAL_POINT_ID.vtk], wp.int32
        )
        self: Self = cls(indices=indices, **kwargs)
        self.materials = self.material_from_region(region, requires_grad=requires_grad)
        return self

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            fun_kernel,
            dim=self.indices.shape,
            inputs=[self.materials.force, self.indices, u, output],
        )

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        del u
        wp.launch(
            grad_kernel,
            dim=self.indices.shape,
            inputs=[self.materials.force, self.indices, output],
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        pass

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        pass

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        pass
