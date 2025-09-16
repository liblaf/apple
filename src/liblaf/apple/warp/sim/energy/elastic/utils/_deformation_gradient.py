from typing import Any, no_type_check

import warp as wp


@wp.func
@no_type_check
def gradient(u: Any, dhdX: Any):
    r"""$\frac{\partial u}{\partial x}$.

    Args:
        u (mat43): ...
        dhdX (mat43): ...

    Returns:
        (mat33): ...
    """
    return wp.transpose(u) @ dhdX


@wp.func
@no_type_check
def deformation_gradient(u: Any, dhdX: Any):
    r"""$F = \frac{\partial u}{\partial x} + I$.

    Args:
        u (mat43): ...
        dhdX (mat43): ...

    Returns:
        F (mat33): ...
    """
    return gradient(u, dhdX) + wp.identity(3, dtype=type(u[0, 0]))


@wp.func
@no_type_check
def deformation_gradient_jvp(dhdX: Any, p: Any):
    r"""$\frac{\partial F}{\partial x} p$.

    Args:
        dhdX (mat43): ...
        p (mat43): ...

    Returns:
        (mat33): ...
    """
    return wp.transpose(p) @ dhdX


@wp.func
@no_type_check
def deformation_gradient_vjp(dhdX: Any, p: Any):
    r"""$\frac{\partial F}{\partial x}^T p$.

    Args:
        dhdX (mat43): ...
        p (mat33): ...

    Returns:
        (mat43): ...
    """
    return dhdX @ wp.transpose(p)
