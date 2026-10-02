"""Check contact energy derivatives along the free contact descent direction."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from transition_contact import build_self_contact  # noqa: E402

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("16-contact-directional-check")
    fixture: Path = (
        ROOT / "exp/2026/09/29/mouthopen-activation/data/30-pruned-fixture/volume.vtu"
    )
    failed: Path = GROUP / "data/20-contact-transition/failed-solver-state.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    volume = pv.read(cfg.fixture)
    fixed = torch.as_tensor(np.asarray(volume.point_data["IsFixed"], bool))
    contact, receipt = build_self_contact(volume)
    with np.load(cfg.failed) as z:
        failed_u = z["u"].copy()
    states = {
        "rest": torch.zeros((volume.n_points, 3), dtype=torch.float64),
        "failed_solver_state": torch.as_tensor(failed_u, dtype=torch.float64),
    }
    result = {
        "schema": "contact-free-direction-finite-difference-v1",
        "sources_sha256": {
            str(path.resolve()): sha256(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("transition_contact.py"),
                cfg.fixture,
                cfg.failed,
            ]
        },
        "contact_receipt": receipt,
        "direction": "negative complete contact gradient restricted to free FEM nodes, normalized to unit Euclidean norm",
        "states": {},
    }
    for name, u in states.items():
        LOG.info("Finite differences for %s", name)
        state = contact.state_at(u)
        energy = float(contact.fun(state, u))
        gradient = torch.zeros_like(u)
        contact.grad(state, u, gradient)
        gradient[fixed] = 0
        direction = -gradient / torch.linalg.vector_norm(gradient)
        analytic = float(torch.sum(gradient * direction))
        checks = []
        for epsilon in (1e-7, 1e-8, 1e-9):
            plus_state = contact.state_at(u + epsilon * direction)
            minus_state = contact.state_at(u - epsilon * direction)
            plus = float(contact.fun(plus_state, u + epsilon * direction))
            minus = float(contact.fun(minus_state, u - epsilon * direction))
            fd = (plus - minus) / (2 * epsilon)
            checks.append(
                {
                    "epsilon_m": epsilon,
                    "energy_plus": plus,
                    "energy_minus": minus,
                    "central_difference": fd,
                    "relative_error": abs(fd - analytic) / abs(analytic),
                    "active_stencils_plus": len(plus_state.collisions),
                    "active_stencils_minus": len(minus_state.collisions),
                }
            )
        result["states"][name] = {
            "energy": energy,
            "free_contact_gradient_l2": float(torch.linalg.vector_norm(gradient)),
            "analytic_directional_derivative": analytic,
            "active_stencils_base": len(state.collisions),
            "checks": checks,
        }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
