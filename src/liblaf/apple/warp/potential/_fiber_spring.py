"""Frame-indifferent axial links for fibers and discrete attachments."""

import functools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, no_type_check, override

import attrs
import numpy as np
import torch
import warp as wp
from torch import Tensor

from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.model import (
    ArrayAnnotation,
    MaterialField,
    Struct,
    WarpPotential,
    make_struct,
)
from liblaf.apple.warp.utils import warp_default_dtype

floating = Any
boolean = Any
vec2i = Any
vec3 = Any
Materials = Any


@wp.func
@no_type_check
def _link_vector(
    u: wp.array1d[vec3],
    endpoints: vec2i,
    materials: Materials,
    lid: int,
) -> vec3:
    return materials.rest_vector[lid] + u[endpoints[1]] - u[endpoints[0]]


@wp.func
def _is_slack(extension: floating, tension_only: boolean) -> boolean:
    return tension_only and extension <= extension.dtype(0.0)


@wp.func
@no_type_check
def _gradient_r(
    r: vec3,
    length: floating,
    rest_length: floating,
    stiffness: floating,
) -> vec3:
    return stiffness * (length - rest_length) / length * r


@wp.func
@no_type_check
def _hessian_product_r(
    r: vec3,
    length: floating,
    rest_length: floating,
    stiffness: floating,
    p: vec3,
) -> vec3:
    n = r / length
    transverse = stiffness * (length - rest_length) / length
    return transverse * p + (stiffness - transverse) * wp.dot(n, p) * n


@wp.func
@no_type_check
def _hessian_diagonal_r(
    r: vec3,
    length: floating,
    rest_length: floating,
    stiffness: floating,
) -> vec3:
    n = r / length
    transverse = stiffness * (length - rest_length) / length
    radial_delta = stiffness - transverse
    return wp.vector(
        transverse + radial_delta * n[0] * n[0],
        transverse + radial_delta * n[1] * n[1],
        transverse + radial_delta * n[2] * n[2],
    )


@wp.kernel(module="unique")
@no_type_check
def _fun_kernel(
    u: wp.array1d[vec3],
    endpoints: wp.array1d[vec2i],
    materials: Materials,
    tension_only: boolean,
    output: wp.array1d[floating],
) -> None:
    lid = wp.tid()
    r = _link_vector(u, endpoints[lid], materials, lid)
    length = wp.length(r)
    extension = length - materials.rest_length[lid]
    if _is_slack(extension, tension_only):
        return
    energy = extension.dtype(0.5) * materials.stiffness[lid] * extension * extension
    wp.atomic_add(output, 0, energy)


@wp.kernel(module="unique")
@no_type_check
def _grad_kernel(
    u: wp.array1d[vec3],
    endpoints: wp.array1d[vec2i],
    materials: Materials,
    tension_only: boolean,
    output: wp.array1d[vec3],
) -> None:
    lid = wp.tid()
    nodes = endpoints[lid]
    r = _link_vector(u, nodes, materials, lid)
    length = wp.length(r)
    extension = length - materials.rest_length[lid]
    if _is_slack(extension, tension_only):
        return
    g = _gradient_r(
        r,
        length,
        materials.rest_length[lid],
        materials.stiffness[lid],
    )
    wp.atomic_add(output, nodes[0], -g)
    wp.atomic_add(output, nodes[1], g)


@wp.kernel(module="unique")
@no_type_check
def _hess_diag_kernel(
    u: wp.array1d[vec3],
    endpoints: wp.array1d[vec2i],
    materials: Materials,
    tension_only: boolean,
    output: wp.array1d[vec3],
) -> None:
    lid = wp.tid()
    nodes = endpoints[lid]
    r = _link_vector(u, nodes, materials, lid)
    length = wp.length(r)
    extension = length - materials.rest_length[lid]
    if _is_slack(extension, tension_only):
        return
    diag = _hessian_diagonal_r(
        r,
        length,
        materials.rest_length[lid],
        materials.stiffness[lid],
    )
    wp.atomic_add(output, nodes[0], diag)
    wp.atomic_add(output, nodes[1], diag)


@wp.kernel(module="unique")
@no_type_check
def _hess_prod_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    endpoints: wp.array1d[vec2i],
    materials: Materials,
    tension_only: boolean,
    output: wp.array1d[vec3],
) -> None:
    lid = wp.tid()
    nodes = endpoints[lid]
    r = _link_vector(u, nodes, materials, lid)
    length = wp.length(r)
    extension = length - materials.rest_length[lid]
    if _is_slack(extension, tension_only):
        return
    dp = p[nodes[1]] - p[nodes[0]]
    hess_dp = _hessian_product_r(
        r,
        length,
        materials.rest_length[lid],
        materials.stiffness[lid],
        dp,
    )
    wp.atomic_add(output, nodes[0], -hess_dp)
    wp.atomic_add(output, nodes[1], hess_dp)


@wp.kernel(module="unique")
@no_type_check
def _hess_quad_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    endpoints: wp.array1d[vec2i],
    materials: Materials,
    tension_only: boolean,
    output: wp.array1d[floating],
) -> None:
    lid = wp.tid()
    nodes = endpoints[lid]
    r = _link_vector(u, nodes, materials, lid)
    length = wp.length(r)
    extension = length - materials.rest_length[lid]
    if _is_slack(extension, tension_only):
        return
    dp = p[nodes[1]] - p[nodes[0]]
    hess_dp = _hessian_product_r(
        r,
        length,
        materials.rest_length[lid],
        materials.stiffness[lid],
        dp,
    )
    wp.atomic_add(output, 0, wp.dot(dp, hess_dp))


def _from_arrays_only(region: Region, annotation: ArrayAnnotation) -> wp.array:
    del region, annotation
    message = "FiberSpring requires FiberSpring.from_arrays()"
    raise TypeError(message)


def _as_numpy(value: np.ndarray | Tensor, *, name: str) -> np.ndarray:
    del name
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_endpoints(endpoints: np.ndarray) -> int:
    if endpoints.ndim != 2 or endpoints.shape[1:] != (2,):
        message = f"endpoints must have shape (links, 2), got {endpoints.shape}"
        raise ValueError(message)
    if not np.issubdtype(endpoints.dtype, np.integer):
        message = "endpoints must contain integer global node IDs"
        raise TypeError(message)
    if np.any(endpoints < 0) or np.any(endpoints > np.iinfo(np.int32).max):
        message = "endpoints must be nonnegative 32-bit global node IDs"
        raise ValueError(message)
    if np.any(endpoints[:, 0] == endpoints[:, 1]):
        message = "each FiberSpring link requires two distinct endpoints"
        raise ValueError(message)
    return endpoints.shape[0]


def _validate_rest_vectors(rest_vectors: np.ndarray, n_links: int) -> np.ndarray:
    if rest_vectors.shape != (n_links, 3):
        message = (
            f"rest_vectors must have shape ({n_links}, 3), got {rest_vectors.shape}"
        )
        raise ValueError(message)
    if not np.all(np.isfinite(rest_vectors)):
        message = "rest_vectors must be finite"
        raise ValueError(message)
    reference_lengths = np.linalg.norm(rest_vectors, axis=1)
    if np.any(reference_lengths <= 0.0):
        message = "rest_vectors must have positive length"
        raise ValueError(message)
    return reference_lengths


def _validate_positive(values: np.ndarray, n_links: int, name: str) -> None:
    if values.shape != (n_links,):
        message = f"{name} must have shape ({n_links},), got {values.shape}"
        raise ValueError(message)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        message = f"{name} must be finite and positive"
        raise ValueError(message)


@attrs.define
class FiberSpring(WarpPotential):
    """Axial links between distinct global nodes.

    For link ``(i, j)``, the current vector is
    ``rest_vector + u[j] - u[i]``. Thus ``rest_vector`` represents
    ``X[j] - X[i]`` and the energy depends on current positions ``X + u`` rather
    than on displacement difference alone. Rigid translations and rotations of
    both endpoints preserve its length and energy.

    Bilateral links use ``0.5 * k * (length - rest_length)**2``. Tension-only
    links replace the extension by its positive part. The resulting energy is
    exactly C1 at recruitment; its Hessian is discontinuous there, and the
    implementation assigns equality to the slack-side zero Hessian.

    ``rest_length`` may differ from ``norm(rest_vector)`` to represent slack or
    prestretch. Active-link derivatives require a strictly positive current
    length. Current-length collapse lies outside the bilateral potential's
    derivative domain and is not regularized by a hidden epsilon; a
    tension-only link at collapse remains in its locally zero slack branch.

    These links are a discrete axial modeling choice. They are not evidence for
    a measured dermal attachment law.
    """

    class Materials(WarpPotential.Materials):
        rest_vector: wp.array
        rest_length: wp.array
        stiffness: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        "rest_vector": MaterialField(
            name="rest_vector",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=_from_arrays_only,
        ),
        "rest_length": MaterialField(
            name="rest_length",
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_from_arrays_only,
        ),
        "stiffness": MaterialField(
            name="stiffness",
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_from_arrays_only,
        ),
    }

    endpoints: wp.array
    tension_only: bool = attrs.field(default=False, kw_only=True)
    materials: Materials = attrs.field(default=None, kw_only=True)

    @functools.cached_property
    def material_struct(self) -> Struct:
        return _fiber_material_struct(warp_default_dtype())

    @classmethod
    @override
    def from_region(
        cls, region: Region, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        del cls, region, requires_grad, kwargs
        message = "FiberSpring requires FiberSpring.from_arrays()"
        raise TypeError(message)

    @classmethod
    def from_arrays(
        cls,
        endpoints: np.ndarray | Tensor,
        rest_vectors: np.ndarray | Tensor,
        stiffness: np.ndarray | Tensor,
        *,
        rest_lengths: np.ndarray | Tensor | None = None,
        tension_only: bool = False,
        requires_grad: Sequence[str] = (),
        device: Any = None,
        **kwargs,
    ) -> Self:
        """Construct links from global node pairs and per-link properties."""
        endpoints_np = _as_numpy(endpoints, name="endpoints")
        rest_vectors_np = _as_numpy(rest_vectors, name="rest_vectors")
        stiffness_np = _as_numpy(stiffness, name="stiffness")
        n_links = _validate_endpoints(endpoints_np)
        reference_lengths = _validate_rest_vectors(rest_vectors_np, n_links)
        _validate_positive(stiffness_np, n_links, "stiffness")
        if rest_lengths is None:
            rest_lengths_np = reference_lengths
        else:
            rest_lengths_np = _as_numpy(rest_lengths, name="rest_lengths")
        _validate_positive(rest_lengths_np, n_links, "rest_lengths")
        unknown = set(requires_grad) - cls.MATERIAL_FIELDS.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            message = f"unknown FiberSpring material fields: {names}"
            raise ValueError(message)

        self = cls(
            endpoints=wp.from_numpy(
                np.ascontiguousarray(endpoints_np, dtype=np.int32),
                dtype=wp.vec2i,
                device=device,
            ),
            tension_only=tension_only,
            **kwargs,
        )
        materials: Any = self.material_struct()
        material_values = {
            "rest_vector": np.ascontiguousarray(rest_vectors_np),
            "rest_length": np.ascontiguousarray(rest_lengths_np),
            "stiffness": np.ascontiguousarray(stiffness_np),
        }
        for name, value in material_values.items():
            annotation = self.material_vars[name].annotation
            array = wp.from_numpy(value, dtype=annotation.dtype, device=device)
            array.requires_grad = name in requires_grad
            setattr(materials, name, array)
        self.materials = materials
        return self

    @property
    def launch_dim(self) -> int:
        return self.endpoints.shape[0]

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            _fun_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                self.endpoints,
                self.materials,
                self.tension_only,
                output,
            ],
            device=self.endpoints.device,
        )

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            _grad_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                self.endpoints,
                self.materials,
                self.tension_only,
                output,
            ],
            device=self.endpoints.device,
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            _hess_diag_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                self.endpoints,
                self.materials,
                self.tension_only,
                output,
            ],
            device=self.endpoints.device,
        )

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            _hess_prod_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                p,
                self.endpoints,
                self.materials,
                self.tension_only,
                output,
            ],
            device=self.endpoints.device,
        )

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            _hess_quad_kernel,
            dim=self.launch_dim,
            inputs=[
                u,
                p,
                self.endpoints,
                self.materials,
                self.tension_only,
                output,
            ],
            device=self.endpoints.device,
        )


@functools.cache
def _fiber_material_struct(dtype: Any) -> Struct:
    material_vars = tuple(
        field.make(dtype) for field in FiberSpring.MATERIAL_FIELDS.values()
    )
    return make_struct(
        material_vars,
        module=FiberSpring.__module__,
        qualname=FiberSpring.__qualname__,
    )
