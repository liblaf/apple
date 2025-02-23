import jax
import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    jax.config.update("jax_platforms", "cpu")
