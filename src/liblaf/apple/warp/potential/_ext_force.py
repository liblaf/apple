from typing import Any, Self, no_type_check, override

import attrs
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Float, Integer

from liblaf.apple.common import FORCE, GLOBAL_POINT_ID
from liblaf.apple.warp.model import WarpPotential
from liblaf.apple.warp.utils import warp_default_dtype, warp_struct

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
    @warp_struct
    class Materials:
        force: wp.array

        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            return {"force": wp.array1d(dtype=wp.types.vector(3, dtype=dtype))}

    indices: wp.array[wp.int32]
    materials: Materials

    @classmethod
    def from_pyvista(cls, obj: pv.PolyData) -> Self:
        dtype: Any = warp_default_dtype()
        force: Float[np.ndarray, " V 3"] = obj.point_data[FORCE.vtk]
        indices: Integer[np.ndarray, " V"] = obj.point_data[GLOBAL_POINT_ID.vtk]
        materials: ExternalForce.Materials = cls.Materials()
        materials.force = wp.from_numpy(force, wp.types.vector(3, dtype))
        return cls(indices=wp.from_numpy(indices, wp.int32), materials=materials)

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
