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
