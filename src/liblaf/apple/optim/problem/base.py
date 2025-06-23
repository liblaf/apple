from collections.abc import Callable
from typing import Protocol


class BaseProblem(Protocol):
    fun: Callable | None
    jac: Callable | None
    hess: Callable | None
    hessp: Callable | None
    hess_diag: Callable | None
    hess_quad: Callable | None
    fun_and_jac: Callable | None
    jac_and_hess_diag: Callable | None
    callback: Callable | None
