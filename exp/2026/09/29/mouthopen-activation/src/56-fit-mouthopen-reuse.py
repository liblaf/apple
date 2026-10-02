"""Repeat the full MouthOpen fit with Newton search shifts reused."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mouthopen_fit_55", GROUP / "src/55-fit-mouthopen.py"
)
assert spec is not None
assert spec.loader is not None
FIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIT
spec.loader.exec_module(FIT)

import stress_physics  # noqa: E402


class Config(FIT.Config):
    output: Path = Path("56-mouthopen-fit-reuse")
    search_shift_policy: str = "reuse"


def install_reuse_policy() -> None:
    """Change only the search-shift policy passed by the 55 optimizer adapter."""
    original_policy = stress_physics._newton_policy  # noqa: SLF001

    def policy() -> tuple[type, Any, Any, Any]:
        cached, error_type, safeguarded_newton, mean_edge = original_policy()

        def safeguarded_reuse(*args: Any, **kwargs: Any) -> Any:
            assert kwargs["shift_policy"] == "reset"
            kwargs["shift_policy"] = "reuse"
            return safeguarded_newton(*args, **kwargs)

        return cached, error_type, safeguarded_reuse, mean_edge

    stress_physics._newton_policy = policy  # noqa: SLF001


def main(cfg: Config) -> None:
    assert cfg.search_shift_policy == "reuse"
    assert cfg.steps == 200
    assert cfg.output == Path("56-mouthopen-fit-reuse")
    install_reuse_policy()
    FIT.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=FIT.Profile)
