import functools
from collections.abc import Callable, Mapping
from typing import Annotated, Any

import jarp
import jarp.warp.types as wpt
import jax.numpy as jnp
import warp as wp
from jarp.warp import FfiCallableProtocol
from jaxtyping import Array, Bool, Float
from warp.jax_experimental import GraphMode

from ._energy import WarpEnergy
from ._model import WarpModel
from ._state import WarpModelState

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Annotated[Array, ""]
type Vector = Annotated[Array, "points dim"]


@jarp.define
class WarpModelAdapterState:
    __wrapped__: WarpModelState = jarp.static(alias="wrapped")
    u: Vector
    marker: Bool[Array, " 1"] = jarp.array(factory=lambda: jnp.zeros((1,), bool))


@jarp.frozen_static
class WarpModelAdapter:
    __wrapped__: WarpModel = jarp.field(alias="wrapped")

    @property
    def energies(self) -> Mapping[str, WarpEnergy]:
        return self.__wrapped__.energies

    def init_state(self, u: Vector) -> WarpModelAdapterState:
        u_wp: wp.array = jarp.to_warp(u, (3, Any))
        wrapped: WarpModelState = self.__wrapped__.init_state(u_wp)
        return WarpModelAdapterState(wrapped, u=u)

    @jarp.jit(inline=True)
    def update(
        self, state: WarpModelAdapterState, u: Vector
    ) -> tuple[WarpModelAdapterState, Vector]:
        update_callable = _make_update_callable(self.__wrapped__, state.__wrapped__)
        state.u = u
        # prevent this from being DCE'd away
        (state.marker,) = update_callable(u)
        return state, u

    def update_materials(self, materials: ModelMaterials) -> None:
        self.__wrapped__.update_materials(materials)

    @jarp.jit(inline=True)
    def fun(self, state: WarpModelAdapterState) -> Scalar:
        fun_callable = _make_fun_callable(self.__wrapped__, state.__wrapped__)
        output: Float[Array, " 1"]
        (output,) = fun_callable(state.u, output_dims={"output": (1,)})
        return output[0]

    @jarp.jit(inline=True)
    def grad(self, state: WarpModelAdapterState) -> Vector:
        grad_callable = _make_grad_callable(self.__wrapped__, state.__wrapped__)
        output: Float[Array, "points 3"]
        (output,) = grad_callable(state.u, output_dims={"output": state.u.shape})
        return output

    @jarp.jit(inline=True)
    def hess_diag(self, state: WarpModelAdapterState) -> Vector:
        hess_diag_callable = _make_hess_diag_callable(
            self.__wrapped__, state.__wrapped__
        )
        output: Float[Array, "points 3"]
        (output,) = hess_diag_callable(state.u, output_dims={"output": state.u.shape})
        return output

    @jarp.jit(inline=True)
    def hess_prod(self, state: WarpModelAdapterState, v: Vector) -> Vector:
        hess_prod_callable = _make_hess_prod_callable(
            self.__wrapped__, state.__wrapped__
        )
        output: Float[Array, "points 3"]
        (output,) = hess_prod_callable(
            state.u, v, output_dims={"output": state.u.shape}
        )
        return output

    @jarp.jit(inline=True)
    def hess_quad(self, state: WarpModelAdapterState, v: Vector) -> Scalar:
        hess_quad_callable = _make_hess_quad_callable(
            self.__wrapped__, state.__wrapped__
        )
        output: Float[Array, " 1"]
        (output,) = hess_quad_callable(state.u, v, output_dims={"output": (1,)})
        return output[0]


# TODO: use weakref to avoid memory leaks
@functools.lru_cache
def _make_update_callable(
    model: WarpModel, state: WarpModelState
) -> jarp.warp.FfiCallableProtocol:
    # This is some dirty trickery to make sure that the update function is
    # JAX-traceable even though it is not pure (it mutates the state).
    @jarp.jax_callable(
        generic=True, graph_mode=GraphMode.WARP, output_dims={"marker": (1,)}
    )
    def update_callable(u_dtype: type) -> Callable[..., None]:
        def update_callable_inner(
            u: wp.array1d(dtype=wp.types.vector(3, u_dtype)),
            marker: wp.array1d(dtype=wp.bool),
        ) -> None:
            nonlocal state
            state = model.update(state, u)
            marker.zero_()

        return update_callable_inner

    return update_callable


@functools.lru_cache
def _make_fun_callable(model: WarpModel, state: WarpModelState) -> FfiCallableProtocol:
    @jarp.jax_callable(generic=False, graph_mode=GraphMode.WARP)
    def fun_callable(
        u: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        output: wp.array1d(dtype=wpt.float_),
    ) -> None:
        model.fun(state, u, output)

    return fun_callable


@functools.lru_cache
def _make_grad_callable(model: WarpModel, state: WarpModelState) -> FfiCallableProtocol:
    @jarp.jax_callable(generic=False, graph_mode=GraphMode.WARP)
    def grad_callable(
        u: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        output: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
    ) -> None:
        model.grad(state, u, output)

    return grad_callable


@functools.lru_cache
def _make_hess_diag_callable(
    model: WarpModel, state: WarpModelState
) -> FfiCallableProtocol:
    @jarp.jax_callable(generic=False, graph_mode=GraphMode.WARP)
    def hess_diag_callable(
        u: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        output: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
    ) -> None:
        model.hess_diag(state, u, output)

    return hess_diag_callable


@functools.lru_cache
def _make_hess_prod_callable(
    model: WarpModel, state: WarpModelState
) -> FfiCallableProtocol:
    @jarp.jax_callable(generic=False, graph_mode=GraphMode.WARP)
    def hess_prod_callable(
        u: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        v: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        output: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
    ) -> None:
        model.hess_prod(state, u, v, output)

    return hess_prod_callable


@functools.lru_cache
def _make_hess_quad_callable(
    model: WarpModel, state: WarpModelState
) -> FfiCallableProtocol:
    @jarp.jax_callable(generic=False, graph_mode=GraphMode.WARP)
    def hess_quad_callable(
        u: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        v: wp.array1d(dtype=wp.types.vector(3, wpt.float_)),
        output: wp.array1d(dtype=wpt.float_),
    ) -> None:
        model.hess_quad(state, u, v, output)

    return hess_quad_callable
