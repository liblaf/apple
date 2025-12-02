import numpy as np
import numpy.typing as npt


def assert_fraction_close(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    *,
    atol: float = 0.0,
    fraction: float = 0.01,
    rtol: float = 1e-7,
) -> None:
    __tracebackhide__ = True
    actual: np.ndarray = np.asarray(actual)
    expected: np.ndarray = np.asarray(expected)
    diff: np.ndarray = np.abs(actual - expected)
    n_fail: int = np.count_nonzero(diff > atol + rtol * np.abs(expected))  # pyright: ignore[reportAssignmentType]
    if n_fail < fraction * actual.size:
        return
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
