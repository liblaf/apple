import equinox as eqx
from jaxtyping import Array

from liblaf.apple.jax.typing import Scalar, Vector

from ._model import Model


@eqx.filter_jit
def fun(x: Vector, model: Model, *args, **kwargs) -> Scalar:
    return model.fun(x, *args, **kwargs)


@eqx.filter_jit
def jac(x: Vector, model: Model, *args, **kwargs) -> Vector:
    return model.jac(x, *args, **kwargs)


@eqx.filter_jit
def hess_diag(x: Vector, model: Model, *args, **kwargs) -> Vector:
    return model.hess_diag(x, *args, **kwargs)


@eqx.filter_jit
def hess_prod(x: Vector, p: Vector, model: Model, *args, **kwargs) -> Vector:
    return model.hess_prod(x, p, *args, **kwargs)


@eqx.filter_jit
def hess_quad(x: Vector, p: Vector, model: Model, *args, **kwargs) -> Scalar:
    return model.hess_quad(x, p, *args, **kwargs)


@eqx.filter_jit
def fun_and_jac(x: Vector, model: Model, *args, **kwargs) -> tuple[Scalar, Vector]:
    return model.fun_and_jac(x, *args, **kwargs)


@eqx.filter_jit
def jac_and_hess_diag(
    x: Vector, model: Model, *args, **kwargs
) -> tuple[Vector, Vector]:
    return model.jac_and_hess_diag(x, *args, **kwargs)


def mixed_derivative_prod(
    x: Vector, p: Vector, model: Model, *args, **kwargs
) -> dict[str, dict[str, Array]]:
    return model.mixed_derivative_prod(x, p, *args, **kwargs)
