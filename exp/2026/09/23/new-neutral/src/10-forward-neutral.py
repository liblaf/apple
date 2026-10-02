"""Forward solve the repaired neutral with the optimized adaptive-IPC hybrid."""

from __future__ import annotations

import functools
import importlib.util
import json
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from liblaf import cherries
from liblaf.apple.inverse import DifferentiableForward

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path.insert(0, str(SOLVERS))
spec = importlib.util.spec_from_file_location(
    "new_neutral_hybrid_driver", SOLVERS / "63-test-neutral-adaptive-ipc.py"
)
assert spec is not None
assert spec.loader is not None
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


class Config(driver.Config):
    output_dir: Path = GROUP / "data/forward-002"


def forbidden_inverse(*_args, **_kwargs):
    message = "Differentiable or adjoint solve invoked in forward-only run"
    raise AssertionError(message)


def main(cfg: Config) -> None:
    driver.bind_frozen_neutral_load = functools.partial(
        driver.bind_frozen_neutral_load,
        allow_isfixed_boundary=True,
        unused_inverse_sha256="0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
    )
    with ExitStack() as stack:
        for name in ("forward", "step", "adjoint_solve", "receipt"):
            stack.enter_context(
                patch.object(DifferentiableForward, name, forbidden_inverse)
            )
        driver.main(cfg)
    shutil.copy2(__file__, cfg.output_dir / "10-forward-neutral.py")
    (cfg.output_dir / "forward-only-guard.json").write_text(
        json.dumps(
            {
                "differentiable_forward_entrypoints_guarded": [
                    "forward",
                    "step",
                    "adjoint_solve",
                    "receipt",
                ],
                "inverse_or_adjoint_called": False,
                "reviewed_inverse_sha256": "0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=driver.benchmark.ProfilePerformance)
