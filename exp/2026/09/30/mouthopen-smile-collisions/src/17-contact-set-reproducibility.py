"""Isolate fresh broad-phase, active-set, and fixed-stencil contact behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from transition_contact import OwnedContact, build_self_contact  # noqa: E402

LOG = logging.getLogger(__name__)
EPSILONS = (1e-7, 1e-8, 1e-9)


class Config(cherries.BaseConfig):
    output: Path = Path("17-contact-set-reproducibility")
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


def state_value(
    contact: OwnedContact, u: torch.Tensor
) -> tuple[float, int, np.ndarray, object]:
    state = contact.state_at(u)
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    return (
        float(contact.fun(state, u)),
        len(state.collisions),
        gradient.numpy(force=True),
        state,
    )


def fixed_candidate_value(
    contact: OwnedContact, base_state: object, u: torch.Tensor
) -> tuple[float, int]:
    positions = np.asfortranarray(
        (contact.vertices + u[contact.indices]).numpy(force=True)
    )
    collisions = ipctk.NormalCollisions()
    collisions.use_area_weighting = True
    collisions.collision_set_type = contact.collision_set_type
    collisions.build(
        candidates=base_state.candidates,
        mesh=contact.collision_mesh,
        vertices=positions,
        dhat=contact.potential.dhat,
        dmin=contact.dmin,
    )
    energy = float(
        contact.potential(collisions, mesh=contact.collision_mesh, X=positions)
    )
    return energy, len(collisions)


def central(plus: float, minus: float, epsilon: float, analytic: float) -> dict:
    fd = (plus - minus) / (2 * epsilon)
    return {
        "central_difference": fd,
        "relative_error": abs(fd - analytic) / abs(analytic),
    }


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    volume = pv.read(cfg.fixture)
    fixed = torch.as_tensor(np.asarray(volume.point_data["IsFixed"], bool))
    contact, receipt = build_self_contact(volume)
    with np.load(cfg.failed) as arrays:
        failed_u = arrays["u"].copy()
    states = {
        "rest": torch.zeros((volume.n_points, 3), dtype=torch.float64),
        "failed_solver_state": torch.as_tensor(failed_u, dtype=torch.float64),
    }
    result = {
        "schema": "contact-set-reproducibility-v1",
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
        "set_types": list(ipctk.NormalCollisions.CollisionSetType.__members__),
        "notes": {
            "identical_repeats": "Five independently built contact states at bit-identical FEM coordinates.",
            "fixed_candidates": "Build each perturbed active set from the single base state's candidate list.",
            "fixed_stencils": "Evaluate the original base active set at perturbed coordinates without rebuilding it.",
            "derivative": "Unit negative contact gradient restricted to free FEM nodes; full area-weighted physical barrier retained.",
        },
        "states": {},
    }
    for name, u in states.items():
        LOG.info("Auditing %s", name)
        variants = {}
        for (
            type_name,
            set_type,
        ) in ipctk.NormalCollisions.CollisionSetType.__members__.items():
            contact.collision_set_type = set_type
            repeated = [state_value(contact, u) for _ in range(5)]
            base_energy, base_count, base_gradient, base_state = repeated[0]
            gradient = torch.as_tensor(base_gradient.copy())
            gradient[fixed] = 0
            norm = float(torch.linalg.vector_norm(gradient))
            detail = {
                "identical_repeats": [
                    {
                        "energy": energy,
                        "active_stencils": count,
                        "free_gradient_l2": float(
                            np.linalg.norm(grad[~fixed.numpy(force=True)])
                        ),
                        "free_gradient_delta_from_first_l2": float(
                            np.linalg.norm(
                                (grad - base_gradient)[~fixed.numpy(force=True)]
                            )
                        ),
                    }
                    for energy, count, grad, _ in repeated
                ],
                "base_energy": base_energy,
                "base_active_stencils": base_count,
                "free_gradient_l2": norm,
                "checks": [],
            }
            if norm > 0:
                direction = -gradient / norm
                analytic = float(torch.sum(gradient * direction))
                detail["analytic_directional_derivative"] = analytic
                for epsilon in EPSILONS:
                    up, down = u + epsilon * direction, u - epsilon * direction
                    fresh_plus, fresh_plus_n, _, _ = state_value(contact, up)
                    fresh_minus, fresh_minus_n, _, _ = state_value(contact, down)
                    candidate_plus, candidate_plus_n = fixed_candidate_value(
                        contact, base_state, up
                    )
                    candidate_minus, candidate_minus_n = fixed_candidate_value(
                        contact, base_state, down
                    )
                    stencil_plus = float(contact.fun(base_state, up))
                    stencil_minus = float(contact.fun(base_state, down))
                    detail["checks"].append(
                        {
                            "epsilon_m": epsilon,
                            "fresh": {
                                **central(fresh_plus, fresh_minus, epsilon, analytic),
                                "plus_energy": fresh_plus,
                                "minus_energy": fresh_minus,
                                "plus_stencils": fresh_plus_n,
                                "minus_stencils": fresh_minus_n,
                            },
                            "fixed_candidates": {
                                **central(
                                    candidate_plus, candidate_minus, epsilon, analytic
                                ),
                                "plus_energy": candidate_plus,
                                "minus_energy": candidate_minus,
                                "plus_stencils": candidate_plus_n,
                                "minus_stencils": candidate_minus_n,
                            },
                            "fixed_stencils": {
                                **central(
                                    stencil_plus, stencil_minus, epsilon, analytic
                                ),
                                "plus_energy": stencil_plus,
                                "minus_energy": stencil_minus,
                                "stencils": base_count,
                            },
                        }
                    )
            variants[type_name] = detail
        result["states"][name] = variants
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
