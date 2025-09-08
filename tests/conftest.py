import datetime

import hypothesis
import jax
import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    jax.config.update("jax_debug_nans", True)  # noqa: FBT003
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    hypothesis.settings.register_profile(
        "default", deadline=datetime.timedelta(seconds=5)
    )
