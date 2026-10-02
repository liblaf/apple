"""Run full-boundary self-contact with the reproducible IPC collision set."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import ipctk

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "projected_contact_runner",
    Path(__file__).with_name("21-run-contact-projected-search.py"),
)
assert spec is not None
assert spec.loader is not None
PROJECTED = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PROJECTED
spec.loader.exec_module(PROJECTED)
RUNNER = PROJECTED.RUNNER
ORIGINAL_BUILD = RUNNER.build_self_contact


def build_standard_contact(*args: Any, **kwargs: Any) -> tuple[Any, dict]:
    contact, receipt = ORIGINAL_BUILD(*args, **kwargs)
    contact.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    receipt.update(
        {
            "collision_set_type": "IPC",
            "area_weighting": True,
            "physical_barrier": True,
            "formulation": "area-weighted sum of IPC primitive barriers; not the improved-max convergent approximation",
            "selection_evidence": str(
                Path(__file__).parents[1]
                / "data/17-contact-set-reproducibility/summary.json"
            ),
        }
    )
    return contact, receipt


class Config(PROJECTED.Config):
    output: Path = Path("22-standard-ipc-transition")


def main(cfg: Config) -> None:
    RUNNER.build_self_contact = build_standard_contact
    PROJECTED.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=RUNNER.Profile)
