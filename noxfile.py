from typing import Any

import liblaf.nox_recipes as recipes
import nox
from liblaf.nox_recipes import Resolution

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
nox.options.tags = ["test"]
PYPROJECT: dict[str, Any] = nox.project.load_toml("pyproject.toml")
PYTHON_VERSIONS: list[str] = nox.project.python_versions(PYPROJECT)


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True, tags=["test"])
@nox.parametrize(
    "resolution",
    [
        nox.param(Resolution.HIGHEST, id="highest", tags=["highest"]),
        nox.param(Resolution.LOWEST, id="lowest", tags=["lowest"]),
        nox.param(Resolution.LOWEST_DIRECT, id="lowest-direct", tags=["lowest-direct"]),
    ],
)
def test(s: nox.Session, resolution: Resolution | None) -> None:
    extras: list[str] = []
    if (cuda_version := recipes.cuda_driver_version()) is not None:
        if cuda_version >= 13000:
            extras.append("cuda13")
        elif cuda_version >= 12000:
            extras.append("cuda12")
    recipes.setup_uv(s, groups=["test"], resolution=resolution)
    recipes.pytest(s, suppress_no_test_exit_code=True)
