from collections.abc import Callable

from liblaf.apple import struct


class BaseProblem(struct.PyTree):
    fun: Callable | None = struct.field(default=None)
    jac: Callable | None = struct.field(default=None)
    hess: Callable | None = struct.field(default=None)
    hessp: Callable | None = struct.field(default=None)
    hess_diag: Callable | None = struct.field(default=None)
    hess_quad: Callable | None = struct.field(default=None)
    fun_and_jac: Callable | None = struct.field(default=None)
    jac_and_hess_diag: Callable | None = struct.field(default=None)
    callback: Callable | None = struct.field(default=None, metadata={"jit": False})
