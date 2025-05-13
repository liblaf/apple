import pytest
import warp as wp


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    wp.set_mempool_release_threshold(wp.get_preferred_device(), 0.25)
