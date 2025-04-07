from collections.abc import Sequence
from typing import override

import attrs
import jax
from jaxtyping import Float, PyTree

from liblaf import apple

from ._abc import MaterialPoint, MaterialPointElement


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class MaterialPointInertiaElement(MaterialPointElement):
    @property
    @override
    def required_params(self) -> Sequence[str]:
        return ("displacement", "mass", "time-step", "velocity")

    @override
    def fun(
        self, u: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        dt: Float[jax.Array, ""] = q["time-step"]
        # f_ext: Float[jax.Array, " 3"] = q["external-force"]
        mass: Float[jax.Array, ""] = q["mass"]
        u_prev: Float[jax.Array, " 3"] = q["displacement"]
        v_prev: Float[jax.Array, " 3"] = q["velocity"]
        u_tilde: Float[jax.Array, " 3"] = u_prev + dt * v_prev
        # TODO: add external force
        return 0.5 * mass * apple.math.norm_sqr(u - u_tilde) / dt**2


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class MaterialPointInertia(MaterialPoint):
    elem: MaterialPointElement = attrs.field(factory=MaterialPointInertiaElement)
