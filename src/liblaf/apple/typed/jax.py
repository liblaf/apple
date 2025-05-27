import jax
from jaxtyping import Float

type Mat12x12 = Float[jax.Array, "12 12"]
type Mat33 = Float[jax.Array, "3 3"]
type Mat43 = Float[jax.Array, "4 3"]
type Mat44 = Float[jax.Array, "4 4"]
type Mat99 = Float[jax.Array, "9 9"]
type Mat9x12 = Float[jax.Array, "9 12"]
type Vec12 = Float[jax.Array, "12"]
type Vec3 = Float[jax.Array, "3"]
type Vec4 = Float[jax.Array, "4"]
type Vec9 = Float[jax.Array, "9"]
