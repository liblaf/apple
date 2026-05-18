import functools
from collections.abc import Callable
from typing import Any, cast

import warp as wp
from jaxtyping import Array, Float

from liblaf import jarp

from ._model import WarpModel

type Scalar = Float[Array, ""]
type Full = Float[Array, "points dim"]


@jarp.define
class WarpModelAdapter:
    __wrapped__: WarpModel = jarp.static()
    n_points: int = jarp.static()

    def fun(self, u: Full) -> Scalar:
        (output,) = self._fun(u)
        return output[0]

    def grad(self, u: Full) -> Full:
        (output,) = self._grad(u)
        return output

    def hess_diag(self, u: Full) -> Full:
        (output,) = self._hess_diag(u)
        return output

    def hess_prod(self, u: Full, p: Full) -> Full:
        (output,) = self._hess_prod(u, p)
        return output

    def hess_quad(self, u: Full, p: Full) -> Full:
        (output,) = self._hess_quad(u, p)
        return output[0]

    @functools.cached_property
    def _fun(self) -> jarp.warp.FfiCallableProtocol:
        @jarp.warp.jax_callable(
            generic=True, num_outputs=1, output_dims={"output": (1,)}
        )
        def fun_callable_factory(dtype: Any) -> Callable[..., None]:
            vec3 = wp.types.vector(3, dtype)

            def fun_callable(u: wp.array1d[vec3], output: wp.array1d[dtype]) -> None:
                u: wp.array = cast("wp.array", u)
                output: wp.array = cast("wp.array", output)
                self.__wrapped__.fun(u, output)

            return fun_callable

        return fun_callable_factory

    @functools.cached_property
    def _grad(self) -> jarp.warp.FfiCallableProtocol:
        @jarp.warp.jax_callable(
            generic=True, num_outputs=1, output_dims={"output": (self.n_points,)}
        )
        def grad_callable_factory(dtype: Any) -> Callable[..., None]:
            vec3 = wp.types.vector(3, dtype)

            def grad_callable(u: wp.array1d[vec3], output: wp.array1d[vec3]) -> None:
                u: wp.array = cast("wp.array", u)
                output: wp.array = cast("wp.array", output)
                self.__wrapped__.grad(u, output)

            return grad_callable

        return grad_callable_factory

    @functools.cached_property
    def _hess_diag(self) -> jarp.warp.FfiCallableProtocol:
        @jarp.warp.jax_callable(
            generic=True, num_outputs=1, output_dims={"output": (self.n_points,)}
        )
        def hess_diag_callable_factory(dtype: Any) -> Callable[..., None]:
            vec3 = wp.types.vector(3, dtype)

            def hess_diag_callable(
                u: wp.array1d[vec3], output: wp.array1d[vec3]
            ) -> None:
                u: wp.array = cast("wp.array", u)
                output: wp.array = cast("wp.array", output)
                self.__wrapped__.hess_diag(u, output)

            return hess_diag_callable

        return hess_diag_callable_factory

    @functools.cached_property
    def _hess_prod(self) -> jarp.warp.FfiCallableProtocol:
        @jarp.warp.jax_callable(
            generic=True,
            num_outputs=1,
            output_dims={"output": (self.n_points,)},
        )
        def hess_prod_callable_factory(
            u_dtype: Any, p_dtype: Any
        ) -> Callable[..., None]:
            vec3 = wp.types.vector(3, u_dtype)
            p_vec3 = wp.types.vector(3, p_dtype)

            def hess_prod_callable(
                u: wp.array1d[vec3], p: wp.array1d[p_vec3], output: wp.array1d[vec3]
            ) -> None:
                u: wp.array = cast("wp.array", u)
                p: wp.array = cast("wp.array", p)
                output: wp.array = cast("wp.array", output)
                self.__wrapped__.hess_prod(u, p, output)

            return hess_prod_callable

        return hess_prod_callable_factory

    @functools.cached_property
    def _hess_quad(self) -> jarp.warp.FfiCallableProtocol:
        @jarp.warp.jax_callable(
            generic=True, num_outputs=1, output_dims={"output": (1,)}
        )
        def hess_quad_callable_factory(
            u_dtype: Any, p_dtype: Any
        ) -> Callable[..., None]:
            vec3 = wp.types.vector(3, u_dtype)
            p_vec3 = wp.types.vector(3, p_dtype)

            def hess_quad_callable(
                u: wp.array1d[vec3], p: wp.array1d[p_vec3], output: wp.array1d[u_dtype]
            ) -> None:
                u: wp.array = cast("wp.array", u)
                p: wp.array = cast("wp.array", p)
                output: wp.array = cast("wp.array", output)
                self.__wrapped__.hess_quad(u, p, output)

            return hess_quad_callable

        return hess_quad_callable_factory
