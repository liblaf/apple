import jax
from jaxtyping import Float

type F3 = Float[jax.Array, "3"]
type F33 = Float[jax.Array, "3 3"]
type F4 = Float[jax.Array, "4"]
type F43 = Float[jax.Array, "4 3"]
type FScalar = Float[jax.Array, ""]
