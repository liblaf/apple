from typing import Any, Self, no_type_check, override

import liblaf.jarp.warp.types as wpt
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Float, Integer

from liblaf import jarp
from liblaf.apple.common import FORCE, GLOBAL_POINT_ID
from liblaf.apple.warp.model import WarpPotential

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


@jarp.frozen_static
class ExternalForce(WarpPotential):
    @jarp.struct
    class Materials:
        force: wp.array

        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            return {"force": wp.array1d(dtype=wp.types.vector(3, dtype=dtype))}

    indices: wp.array[wp.int32] = jarp.static()
    materials: Materials = jarp.static()

    @classmethod
    def from_pyvista(cls, obj: pv.PolyData) -> Self:
        force: Float[np.ndarray, " V 3"] = obj.point_data[FORCE.vtk]
        indices: Integer[np.ndarray, " V"] = obj.point_data[GLOBAL_POINT_ID.vtk]
        materials: ExternalForce.Materials = cls.Materials()
        materials.force = wp.from_numpy(force, wpt.vec3)
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
