import jax
import warp as wp

float_ = wp.float64 if jax.config.read("jax_enable_x64") else wp.float32
int_ = wp.int32
mat33 = wp.types.matrix((3, 3), float_)
mat43 = wp.types.matrix((4, 3), float_)
vec4i = wp.types.vector(4, int_)
vec6 = wp.types.vector(6, float_)
