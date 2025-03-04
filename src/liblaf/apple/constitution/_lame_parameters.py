import jax
import numpy as np


def E_nu_to_lame[T: (float, np.ndarray, jax.Array)](E: T, nu: T) -> tuple[T, T]:
    lmbda: T = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu: T = E / (2 * (1 + nu))
    return lmbda, mu


def lame_to_E_nu[T: (float, np.ndarray, jax.Array)](lmbda: T, mu: T) -> tuple[T, T]:
    E: T = mu * (3 * lmbda + 2 * mu) / (lmbda + mu)
    nu: T = lmbda / (2 * (lmbda + mu))
    return E, nu
