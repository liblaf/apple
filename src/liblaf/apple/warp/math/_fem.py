# ruff: noqa: PYI042

from typing import Any, no_type_check

import warp as wp

type mat43 = Any
type mat33 = Any


@wp.func
@no_type_check
def gradient(u: mat43, dhdX: mat43) -> mat33:
    r"""$\pdv{u}{x}$."""
    return wp.transpose(u) @ dhdX


@wp.func
@no_type_check
def deformation_gradient(u: mat43, dhdX: mat43) -> mat33:
    r"""$F = \pdv{u}{x} + I$."""
    return gradient(u, dhdX) + wp.identity(3, dtype=u.dtype)


@wp.func
@no_type_check
def deformation_gradient_jvp(dhdX: mat43, p: mat43) -> mat33:
    r"""$\pdv{F}{x} p$."""
    return wp.transpose(p) @ dhdX


@wp.func
@no_type_check
def deformation_gradient_vjp(dhdX: mat43, p: mat33) -> mat43:
    r"""$\pdv{F}{x}^T p$."""
    return dhdX @ wp.transpose(p)
