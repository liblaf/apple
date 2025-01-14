import dataclasses
import enum
import functools
from collections.abc import Callable, Mapping
from typing import Any

import comet_ml as comet
import jax
import jax.numpy as jnp
import numpy.typing as npt
import scipy
import scipy.optimize
from jaxtyping import Float

from liblaf import apple


class Problem:
    _raw_fun: Callable
    args: tuple = ()

    def __init__(self, fun: Callable, args: tuple = ()) -> None:
        self._raw_fun = fun
        self.args = args

    def fun(self, x: Float[npt.ArrayLike, " N"]) -> Float[jax.Array, ""]:
        x = jnp.asarray(x)
        return self._fun_jit(x)

    def jac(self, x: Float[npt.ArrayLike, " N"]) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        return self._jac_jit(x)

    def hess(self, x: Float[npt.ArrayLike, " N"]) -> Float[jax.Array, " N N"]:
        x = jnp.asarray(x)
        return self._hess_jit(x)

    def hessp(
        self, x: Float[npt.ArrayLike, " N"], p: Float[npt.ArrayLike, " N"]
    ) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        p = jnp.asarray(p, x.dtype)
        return self._hessp_jit(x, p)

    def _fun(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        return self._raw_fun(x, *self.args)

    @functools.cached_property
    def _fun_jit(self) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]]:
        return jax.jit(self._fun)

    def _jac(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        return jax.grad(self._fun)(x)

    @functools.cached_property
    def _jac_jit(self) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]]:
        return jax.jit(self._jac)

    def _hess(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, " N N"]:
        return jax.hessian(self._fun)(x)

    @functools.cached_property
    def _hess_jit(self) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N N"]]:
        return jax.jit(self._hess)

    def _hessp(
        self, x: Float[jax.Array, " N"], p: Float[jax.Array, " N"]
    ) -> Float[jax.Array, " N"]:
        return apple.hvp(self._fun, x, p)

    @functools.cached_property
    def _hessp_jit(
        self,
    ) -> Callable[
        [Float[jax.Array, " N"], Float[jax.Array, " N"]], Float[jax.Array, " N"]
    ]:
        return jax.jit(self._hessp)


class MinimizeMethod(enum.StrEnum):
    NEWTON_CG = "Newton-CG"
    TRUST_CONSTR = "trust-constr"


@dataclasses.dataclass(kw_only=True)
class MinimizeMethodInfo:
    name: MinimizeMethod
    jac: bool = False
    hess: bool = False
    hessp: bool = False
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)


# TODO: switch to LowerDict
METHODS: dict[str, MinimizeMethodInfo] = {
    "newton-cg": MinimizeMethodInfo(
        name=MinimizeMethod.NEWTON_CG,
        jac=True,
        hess=True,
        hessp=True,
        options={"disp": True},
    ),
    "trust-constr": MinimizeMethodInfo(
        name=MinimizeMethod.TRUST_CONSTR,
        jac=True,
        hess=True,
        hessp=True,
        options={"disp": True, "verbose": 3},
    ),
}


class Callback:
    step: int = 0

    def __call__(self, intermediate_result: scipy.optimize.OptimizeResult) -> None:
        if self.experiment is not None:
            for key, value in intermediate_result.items():
                if isinstance(value, int | float):
                    self.experiment.log_metric(key, value, step=self.step)
        self.step += 1

    @property
    def experiment(self) -> comet.CometExperiment | None:
        return comet.get_running_experiment()


def minimize(
    fun: Callable,
    x0: Float[npt.ArrayLike, " N"],
    args: tuple = (),
    method: MinimizeMethod | str = MinimizeMethod.NEWTON_CG,
    jac: Callable | None = None,
    hess: Callable | None = None,
    hessp: Callable | None = None,
    options: Mapping[str, Any] = {},
    callback: Callable | None = None,
) -> scipy.optimize.OptimizeResult:
    problem = Problem(fun=fun, args=args)
    info: MinimizeMethodInfo = METHODS[method.lower()]
    if (jac is None) and info.jac:
        jac = problem.jac
    if (hess is None) and info.hess and (not info.hessp):
        hess = problem.hess
    if (hessp is None) and info.hessp:
        hessp = problem.hessp
    if callback is None:
        callback = Callback()
        if callback.experiment is not None:
            # TODO: generate a unique name as context
            # TODO: switch to liblaf.cherries for experiment tracking
            with callback.experiment.context_manager("minimize"):
                return scipy.optimize.minimize(
                    fun,
                    x0,
                    method=method,
                    jac=jac,
                    hess=hess,
                    hessp=hessp,
                    options=options,
                    callback=callback,
                )
    return scipy.optimize.minimize(
        fun,
        x0,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        options=options,
        callback=callback,
    )
