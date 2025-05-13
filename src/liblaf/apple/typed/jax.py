import jax
from jaxtyping import Float

type Mat33 = Float[jax.Array, "3 3"]
type Vec3 = Float[jax.Array, "3"]
