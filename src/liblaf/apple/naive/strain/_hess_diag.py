from liblaf.apple.typed.jax import Mat9x12, Vec9, Vec12


def h3_diag(dFdx: Mat9x12, g3: Vec9) -> Vec12:
    return (dFdx.T @ g3) ** 2
