from jaxtyping import Array, Float

from liblaf import jarp

type Full = Float[Array, "points dim"]


@jarp.define
class ModelState:
    u: Full
