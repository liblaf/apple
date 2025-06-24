import hypothesis
import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    hypothesis.settings.register_profile("default", max_examples=10)
