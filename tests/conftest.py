import jax
import pytest


@pytest.fixture(autouse=True)
def setup_jax() -> None:
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    jax.config.update("jax_platforms", "cpu")
