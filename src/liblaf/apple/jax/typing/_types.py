import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

type Scalar = Float[Array, ""]
type UpdatesData = Float[Array, "Any ..."]
type UpdatesIndex = Integer[Array, " Any"]
type Updates = tuple[UpdatesData, UpdatesIndex]
type Vector = Float[Array, "*DoF"]


int_ = jnp.int32
float_ = jnp.float64
