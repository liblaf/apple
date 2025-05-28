def lame_params(E: float, nu: float) -> tuple[float, float]:
    lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
    mu: float = E / (2 * (1 + nu))  # Shear modulus
    return lambda_, mu
