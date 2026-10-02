# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915
"""CPU admission check for the repaired fixed-reference rigid IPC surface."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries
from liblaf.apple.forward.dof_map import DofMap

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from fixed_reference_contact import (  # noqa: E402
    audit_fixed_reference_contact,
    build_fixed_reference_contact,
    extend_dof_map,
)


class Config(cherries.BaseConfig):
    output: Path = Path("51-fixed-reference-contact-check")
    fixture: Path = GROUP / "data/50-fixed-reference/fixture/volume.vtu"
    prepared_pose: Path = (
        ROOT / "exp/2026/09/29/mouthopen-activation/data/10-mandible/prepared.npz"
    )
    stiffness_mpa: float = 1.3544
    dhat_m: float = 1e-4
    minimum_distance_m: float = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    fixture = cherries.input(cfg.fixture)
    prepared_pose = cherries.input(cfg.prepared_pose)
    volume = pv.read(fixture)
    reference = build_fixed_reference_contact(
        volume,
        stiffness_mpa=cfg.stiffness_mpa,
        dhat_m=cfg.dhat_m,
        minimum_distance_m=cfg.minimum_distance_m,
    )
    assert reference.receipt["collision_set_type"] == "IPC"
    assert reference.receipt["soft_soft_contact"] is False
    assert reference.receipt["rigid_rigid_contact"] is False
    assert reference.receipt["source_triangles_excluded"] == 0
    assert reference.receipt["barrier"]["dhat_m"] == cfg.dhat_m
    assert reference.receipt["barrier"]["stiffness_mpa"] == cfg.stiffness_mpa
    assert reference.receipt["barrier"]["ccd_min_distance_m"] == cfg.minimum_distance_m
    points = np.asarray(volume.points, dtype=np.float64)
    full_points = reference.full_reference_points(points)
    indices = reference.contact.indices.numpy(force=True)
    np.testing.assert_allclose(
        reference.contact.vertices.numpy(force=True),
        full_points[indices],
        rtol=0,
        atol=0,
    )
    assert np.array_equal(np.unique(indices), indices)
    assert np.array_equal(
        indices[: len(reference.soft_physical_ids)], reference.soft_physical_ids
    )
    assert np.array_equal(indices[-len(reference.eye_ids) :], reference.eye_ids)
    assert reference.full_mandible_mask.sum() == len(
        reference.physical_mandible_ids
    ) + len(reference.mandible_ids)
    assert np.all(reference.full_mandible_mask[reference.physical_mandible_ids])
    assert np.all(reference.full_mandible_mask[reference.mandible_ids])
    u0 = torch.zeros((reference.full_point_count, 3), dtype=torch.float64)
    state = reference.contact.state_at(u0)
    diagnostics = audit_fixed_reference_contact(reference, u0)
    contact_energy = float(reference.contact.fun(state, u0))
    contact_force = torch.zeros_like(u0)
    reference.contact.grad(state, u0, contact_force)
    contact_force_norm = float(torch.linalg.vector_norm(contact_force))
    has_intersections = diagnostics["scoped_has_intersections"]
    assert diagnostics["contact_numerically_valid"]
    assert diagnostics["clearance_at_least_dhat"]
    assert not has_intersections
    assert contact_energy == 0.0
    assert contact_force_norm == 0.0
    with np.load(prepared_pose, allow_pickle=False) as archive:
        pose = torch.as_tensor(archive["pose"], dtype=torch.float64)
        pivot = torch.as_tensor(archive["pivot"], dtype=torch.float64)
    physical = torch.as_tensor(points)
    boundary = reference.boundary_values(physical, pose, pivot)
    assert boundary.shape == (reference.full_point_count, 3)
    assert torch.equal(
        boundary[reference.cranium_ids],
        torch.zeros_like(boundary[reference.cranium_ids]),
    )
    assert torch.equal(
        boundary[reference.eye_ids], torch.zeros_like(boundary[reference.eye_ids])
    )
    seed = reference.extend_seed(
        boundary[: reference.physical_point_count], pose, pivot
    )
    assert torch.equal(seed, boundary)
    ccd_fraction = float(reference.contact.max_step_size(state, u0, boundary))
    assert 0 < ccd_fraction <= 1
    fixed = np.flatnonzero(np.asarray(volume.point_data["IsFixed"], dtype=bool))
    original = DofMap(
        n_points=volume.n_points,
        fixed_indices=torch.as_tensor(
            np.repeat(fixed * 3, 3) + np.tile(np.arange(3), len(fixed))
        ),
        fixed_values=torch.zeros(len(fixed) * 3),
        free_indices=torch.as_tensor(
            np.setdiff1d(
                np.arange(volume.n_points * 3),
                np.repeat(fixed * 3, 3) + np.tile(np.arange(3), len(fixed)),
            )
        ),
    )
    extended = extend_dof_map(original, reference, boundary)
    assert extended.n_points == reference.full_point_count
    assert torch.equal(extended.free_indices, original.free_indices)
    assert torch.equal(extended.to_full(extended.to_free(boundary)), boundary)
    result = {
        "schema": "fixed-reference-contact-cpu-admission-v1",
        "fixture": {"path": str(fixture.resolve()), "sha256": sha256(fixture)},
        "prepared_pose": {
            "path": str(prepared_pose.resolve()),
            "sha256": sha256(prepared_pose),
        },
        "source": {
            str(Path(__file__).resolve()): sha256(Path(__file__)),
            str(
                Path(__file__).with_name("fixed_reference_contact.py").resolve()
            ): sha256(Path(__file__).with_name("fixed_reference_contact.py")),
        },
        "contact": reference.receipt,
        "rest": {
            **diagnostics,
            "soft_rigid_has_intersections": has_intersections,
            "energy_mpa_m3": contact_energy,
            "force_l2_mpa_m2": contact_force_norm,
        },
        "full_pose_boundary": {
            "ccd_fraction": ccd_fraction,
            "full_step_collision_free": ccd_fraction == 1.0,
            "cranium_zero": True,
            "eyes_zero": True,
            "mandible_rigid": True,
        },
        "dof_map": {
            "physical_free_dofs_preserved": True,
            "appended_rigid_dofs_fixed": int(
                reference.full_point_count - volume.n_points
            )
            * 3,
        },
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
