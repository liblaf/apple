# ruff: noqa: N801

import warp.types


class vec2(warp.types.vector(2, float)): ...


class vec3(warp.types.vector(3, float)): ...


class vec4(warp.types.vector(4, float)): ...


class vec9(warp.types.vector(9, float)): ...


class vec12(warp.types.vector(12, float)): ...


class mat33(warp.types.matrix((3, 3), float)): ...


class mat34(warp.types.matrix((3, 4), float)): ...


class mat43(warp.types.matrix((4, 3), float)): ...


class mat44(warp.types.matrix((4, 4), float)): ...


__all__ = ["mat33", "mat34", "mat43", "mat44", "vec2", "vec3", "vec4", "vec9", "vec12"]
