"""Check saved expression states with fixed-fixed pairs exempt from IPC contact."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from transition_contact import OwnedContact, build_self_contact  # noqa: E402

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("19-deformable-contact")
    fixture: Path = PARENT / "data/30-pruned-fixture/volume.vtu"
    mouthopen: Path = PARENT / "data/70-mouthopen-four-stage/rankone_learned/last.npz"
    frame000: Path = (
        PARENT / "data/91-smile-mouthopen-transition-003/frames/frame-000.npz"
    )
    frame120: Path = (
        PARENT / "data/91-smile-mouthopen-transition-003/frames/frame-120.npz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_filter() -> dict:
    points = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, -0.5, -1],
            [0, -0.5, 1],
            [0, 0.5, 0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = ipctk.CollisionMesh(points, ipctk.edges(faces), faces)
    mesh.init_adjacencies()
    broad = ipctk.LBVH()
    assert ipctk.has_intersections(mesh, points, broad)
    fixed_filter = ipctk.make_vertex_patches_filter(np.zeros(6, dtype=np.int32))
    assert not fixed_filter(0, 3)
    mesh.can_collide = fixed_filter
    assert not ipctk.has_intersections(mesh, points, broad)
    mixed_filter = ipctk.make_vertex_patches_filter(
        np.array([0, 0, 0, 1, 2, 3], dtype=np.int32)
    )
    assert mixed_filter(0, 3)
    assert mixed_filter(3, 4)
    mesh.can_collide = mixed_filter
    assert ipctk.has_intersections(mesh, points, broad)
    return {
        "unfiltered_crossing_detected": True,
        "all_fixed_crossing_exempted": True,
        "free_fixed_crossing_detected": True,
        "free_free_pairs_allowed": True,
    }


def active_gap(contact: OwnedContact, u: torch.Tensor) -> dict:
    state = contact.state_at(u)
    count = len(state.collisions)
    positions = np.asfortranarray(
        (contact.vertices + u[contact.indices]).numpy(force=True)
    )
    squared = (
        float(
            state.collisions.compute_minimum_distance(contact.collision_mesh, positions)
        )
        if count
        else None
    )
    gap = math.sqrt(max(squared, 0.0)) if squared is not None else None
    intersects = bool(
        ipctk.has_intersections(contact.collision_mesh, positions, ipctk.LBVH())
    )
    return {
        "has_intersections": intersects,
        "active_contact_count": count,
        "minimum_active_gap_m": gap,
        "minimum_gap_scope": "active IPC stencils only; null if none",
    }


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    filter_receipt = verify_filter()
    volume = pv.read(cfg.fixture)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    full, full_receipt = build_self_contact(volume)
    scoped, scoped_receipt = build_self_contact(volume)
    full.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    scoped.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    local_fixed = fixed[scoped.indices.numpy(force=True)]
    patches = np.zeros(len(local_fixed), dtype=np.int32)
    patches[~local_fixed] = np.arange(1, int((~local_fixed).sum()) + 1, dtype=np.int32)
    collision_filter = ipctk.make_vertex_patches_filter(patches)
    fixed_ids = np.flatnonzero(local_fixed)
    free_ids = np.flatnonzero(~local_fixed)
    assert not collision_filter(int(fixed_ids[0]), int(fixed_ids[-1]))
    assert collision_filter(int(fixed_ids[0]), int(free_ids[0]))
    assert collision_filter(int(free_ids[0]), int(free_ids[-1]))
    scoped.collision_mesh.can_collide = collision_filter
    states = {"rest": np.zeros((volume.n_points, 3), dtype=np.float64)}
    for name, path in {
        "mouthopen_rankone_learned": cfg.mouthopen,
        "old_transition_frame000": cfg.frame000,
        "old_transition_frame120": cfg.frame120,
    }.items():
        with np.load(path) as arrays:
            states[name] = arrays["u"].copy()
    result = {
        "schema": "saved-expression-deformable-ipc-contact-preflight-v1",
        "sources_sha256": {
            str(path.resolve()): sha256(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("transition_contact.py"),
                cfg.fixture,
                cfg.mouthopen,
                cfg.frame000,
                cfg.frame120,
            ]
        },
        "policy": "All boundary faces retained. Standard IPC normal collisions. Every vertex pair involving at least one free FEM vertex is eligible; only fixed-fixed vertex pairs are exempt.",
        "filter_unit_check": filter_receipt,
        "filter_counts": {
            "local_fixed_vertices": int(local_fixed.sum()),
            "local_free_vertices": int((~local_fixed).sum()),
            "distinct_free_patches": int(np.unique(patches[~local_fixed]).size),
        },
        "contact_receipt": full_receipt,
        "scoped_contact_receipt": scoped_receipt,
        "states": {},
    }
    for name, displacement in states.items():
        assert displacement.shape == (volume.n_points, 3)
        u = torch.as_tensor(displacement)
        LOG.info("Auditing %s", name)
        result["states"][name] = {
            "full": active_gap(full, u),
            "deformable_scope": active_gap(scoped, u),
        }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
