import abc
import functools

import jax
import numpy as np
from jaxtyping import Float, Integer


class Region:
    @property
    @abc.abstractmethod
    def n_points(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_cells(self) -> int: ...

    @property
    def cells(self) -> Integer[np.ndarray, "c a"]: ...

    @functools.cached_property
    @abc.abstractmethod
    def h(self) -> Float[jax.Array, "a q"]:
        """Element shape function array `h_aq` of shape function `a` evaluated at quadrature point `q`."""
        ...

    @functools.cached_property
    @abc.abstractmethod
    def dV(self) -> Float[jax.Array, "q c"]:
        """Numeric *differential volume element* as product of determinant of geometric gradient `dV_qc = det(dXdr)_qc w_q` and quadrature weight `w_q`, evaluated at quadrature point `q` for every cell `c`."""
        ...

    @functools.cached_property
    @abc.abstractmethod
    def dhdX(self) -> Float[jax.Array, "a J q c"]:
        """Partial derivative of element shape functions `dhdX_aJqc` of shape function `a` w.r.t. undeformed coordinate `J` evaluated at quadrature point `q` for every cell `c`."""
        ...

    @functools.cached_property
    @abc.abstractmethod
    def d2hdXdX(self) -> Float[jax.Array, "a I J q c"]: ...
