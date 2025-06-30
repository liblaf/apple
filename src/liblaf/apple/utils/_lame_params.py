def lame_params(*, E: float, nu: float) -> tuple[float, float]:
    r"""...

    $$
    \lambda = \frac{E \nu}{(1 + \nu)(1 - 2 \nu)}, \quad
    \mu = \frac{E}{2 (1 + \nu)}
    $$

    Args:
        E: Young's modulus
        nu: Poisson's ratio
    """
    lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
    mu: float = E / (2 * (1 + nu))  # Shear modulus
    return lambda_, mu
