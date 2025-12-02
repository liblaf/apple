import hypothesis
import jax
import pytest
import warp as wp


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    hypothesis.settings.register_profile("default", max_examples=10, deadline=None)
    jax.config.update("jax_check_tracer_leaks", True)  # noqa: FBT003
    jax.config.update("jax_debug_nans", True)  # noqa: FBT003
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verbose_warnings = True
    wp.init()
