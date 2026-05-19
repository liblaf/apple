from collections.abc import Mapping
from typing import Any

import attrs
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from ._model import WarpModel

type Scalar = Float[Tensor, ""]
type Full = Float[Tensor, "points dim"]


@attrs.define
class WarpModelAdapter:
    __wrapped__: WarpModel = attrs.field()

    def get_materials(self) -> dict[str, dict[str, Tensor]]:
        materials: dict[str, dict[str, wp.array]] = self.__wrapped__.get_materials()
        return {
            pot_name: {mat_name: wp.to_torch(arr) for mat_name, arr in pot_mat.items()}
            for pot_name, pot_mat in materials.items()
        }

    def set_materials(
        self, materials: Mapping[str, Mapping[str, wp.array | Tensor]]
    ) -> None:
        self.__wrapped__.set_materials(materials)

    def require_grad(self, materials: Mapping[str, Mapping[str, bool]]) -> None:
        self.__wrapped__.require_grad(materials)

    def fun(self, u: Full) -> Scalar:
        output: Scalar = torch.zeros((1,), dtype=u.dtype, device=u.device)
        u_wp: wp.array = _from_torch_vec3(u)
        output_wp: wp.array = _from_torch_float(output)
        with _stream(u):
            self.__wrapped__.fun(u_wp, output_wp)
        return output[0]

    def grad(self, u: Full, output: Full) -> None:
        u_wp: wp.array = _from_torch_vec3(u)
        output_wp: wp.array = _from_torch_vec3(output)
        with _stream(u):
            self.__wrapped__.grad(u_wp, output_wp)

    def hess_diag(self, u: Full, output: Full) -> None:
        u_wp: wp.array = _from_torch_vec3(u)
        output_wp: wp.array = _from_torch_vec3(output)
        with _stream(u):
            self.__wrapped__.hess_diag(u_wp, output_wp)

    def hess_prod(self, u: Full, p: Full, output: Full) -> None:
        u_wp: wp.array = _from_torch_vec3(u)
        p_wp: wp.array = _from_torch_vec3(p)
        output_wp: wp.array = _from_torch_vec3(output)
        with _stream(u):
            self.__wrapped__.hess_prod(u_wp, p_wp, output_wp)

    def hess_quad(self, u: Full, p: Full) -> Scalar:
        output: Scalar = torch.zeros((1,), dtype=u.dtype, device=u.device)
        u_wp: wp.array = _from_torch_vec3(u)
        p_wp: wp.array = _from_torch_vec3(p)
        output_wp: wp.array = _from_torch_float(output)
        with _stream(u):
            self.__wrapped__.hess_quad(u_wp, p_wp, output_wp)
        return output[0]

    def mixed_derivative_prod(self, u: Full, p: Full) -> None:
        u_wp: wp.array = _from_torch_vec3(u)
        p_wp: wp.array = _from_torch_vec3(p)
        with _stream(u):
            self.__wrapped__.mixed_derivative_prod(u_wp, p_wp)


def _from_torch_float(x: Tensor) -> wp.array:
    floating: Any = wp.dtype_from_torch(x.dtype)
    return wp.from_torch(x, dtype=floating)


def _from_torch_vec3(x: Tensor) -> wp.array:
    floating: Any = wp.dtype_from_torch(x.dtype)
    vec3: Any = wp.types.vector(3, floating)
    return wp.from_torch(x, dtype=vec3)


def _stream(x: Tensor) -> wp.ScopedStream:
    stream_torch: torch.cuda.Stream = torch.cuda.current_stream(x.device)
    stream_wp: wp.Stream = wp.stream_from_torch(stream_torch)
    return wp.ScopedStream(stream_wp)
