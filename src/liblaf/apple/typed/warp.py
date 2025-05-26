from warp import types


class mat43f(types.matrix(shape=(4, 3), dtype=types.float32)): ...  # noqa: N801


mat43 = mat43f
