from warp import types
from warp.types import mat33, vec2, vec3, vec4


class mat34f(types.matrix(shape=(3, 4), dtype=types.float32)): ...  # noqa: N801


class mat43f(types.matrix(shape=(4, 3), dtype=types.float32)): ...  # noqa: N801


class vec9f(types.vector(length=9, dtype=types.float32)): ...  # noqa: N801


class vec12f(types.vector(length=12, dtype=types.float32)): ...  # noqa: N801


mat34 = mat34f
mat43 = mat43f


vec9 = vec9f
vec12 = vec12f

__all__ = ["mat33", "mat34", "mat43", "vec2", "vec3", "vec4", "vec9", "vec12"]
