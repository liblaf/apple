import jax
import pytest
from jaxtyping import PRNGKeyArray


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    jax.config.update("jax_platforms", "cpu")


@pytest.fixture(scope="session")
def rng(request: pytest.FixtureRequest) -> PRNGKeyArray:
    node: pytest.Item = request.node
    return jax.random.key(hash(node.name))
