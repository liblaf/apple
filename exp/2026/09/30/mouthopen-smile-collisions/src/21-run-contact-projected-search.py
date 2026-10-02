"""Use IPC's PSD contact Hessian only for forward Newton search directions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import ipctk

from liblaf import cherries
from liblaf.apple.forward.hessian._problem import HessianProblem

spec = importlib.util.spec_from_file_location(
    "full_boundary_contact_runner",
    Path(__file__).with_name("20-run-contact-transition.py"),
)
assert spec is not None
assert spec.loader is not None
RUNNER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUNNER
spec.loader.exec_module(RUNNER)


class ProjectedContactSearch(HessianProblem):
    """Clamp contact stencil curvature; retain exact bulk, energy, and forces."""

    def hess_prod(self, state: Any, direction: Any) -> Any:
        contact = self.model.collision
        assert contact is not None
        assert state.collision is not None
        if state.collision.hess is None:
            positions = (contact.vertices + state.u[contact.indices]).numpy(force=True)
            state.collision.hess = contact.potential.hessian(
                collisions=state.collision.collisions,
                mesh=contact.collision_mesh,
                X=positions,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
            )
        return super().hess_prod(state, direction)

    def report(self) -> dict:
        return {
            **super().report(),
            "contact_search_hessian": "IPC stencilwise PSD CLAMP; search direction only",
            "bulk_hessian": "exact unprojected",
            "energy_gradient_and_force_gate_changed": False,
            "implicit_adjoint": "not evaluated; this run only solves forward states",
        }


class Config(RUNNER.Config):
    output: Path = Path("21-contact-projected-search")


def main(cfg: Config) -> None:
    RUNNER.HessianProblem = ProjectedContactSearch
    RUNNER.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=RUNNER.Profile)
