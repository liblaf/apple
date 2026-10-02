"""CPU audit of active IPC stencils and contact force on the pruned boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

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
KINDS = ("vv_collisions", "ev_collisions", "ee_collisions", "fv_collisions")


class Config(cherries.BaseConfig):
    output: Path = Path("15-contact-preflight")
    fixture: Path = (
        ROOT / "exp/2026/09/29/mouthopen-activation/data/30-pruned-fixture/volume.vtu"
    )
    pilot: Path = GROUP / "data/20-contact-transition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_state(
    contact: OwnedContact,
    displacement: np.ndarray,
    fixed: np.ndarray,
    labels: dict[str, np.ndarray | list[str]],
    lip: np.ndarray,
    nonmanifold: np.ndarray,
) -> dict:
    u = torch.as_tensor(displacement, dtype=torch.float64)
    state = contact.state_at(u)
    positions = np.asfortranarray(
        (contact.vertices + u[contact.indices]).numpy(force=True)
    )
    mesh = contact.collision_mesh
    local_to_global = contact.indices.numpy(force=True)
    nearest = []
    counts = Counter()
    zone_counts = Counter()
    for kind in KINDS:
        for stencil in getattr(state.collisions, kind):
            local = np.asarray(
                stencil.vertex_ids(mesh.edges, mesh.faces), dtype=np.int64
            )
            global_ids = local_to_global[local[local >= 0]]
            gap_sq = float(stencil.compute_distance(positions, mesh.edges, mesh.faces))
            gap = float(np.sqrt(max(gap_sq, 0.0)))
            is_fixed = bool(fixed[global_ids].all())
            is_lip = bool(lip[global_ids].any())
            near_nonmanifold = bool(nonmanifold[global_ids].any())
            counts[kind] += 1
            zone_counts["all_fixed" if is_fixed else "contains_free"] += 1
            zone_counts["touches_lip" if is_lip else "away_from_lips"] += 1
            zone_counts[
                "touches_nonmanifold" if near_nonmanifold else "away_from_nonmanifold"
            ] += 1
            nearest.append(
                (gap, kind, global_ids.tolist(), is_fixed, is_lip, near_nonmanifold)
            )
    nearest.sort(key=lambda item: item[0])
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    norm = torch.linalg.vector_norm(gradient, dim=1).numpy(force=True)
    zone_force = {}
    for name, mask in {
        "all_fixed_nodes": fixed,
        "free_nodes": ~fixed,
        "lip_nodes": lip,
        "nonmanifold_vertices": nonmanifold,
    }.items():
        zone_force[name] = {
            "gradient_l2": float(np.linalg.norm(gradient.numpy(force=True)[mask])),
            "gradient_l1_vertex_norm": float(norm[mask].sum()),
            "nonzero_vertices": int(np.count_nonzero(norm[mask])),
        }
    group_force = {}
    names = np.asarray(labels["names"])
    for group_id, name in enumerate(names):
        mask = labels["ids"] == group_id
        group_force[str(name)] = float(norm[mask].sum())
    return {
        "active_stencils_by_type": dict(counts),
        "active_stencils_by_zone": dict(zone_counts),
        "minimum_active_gap_m": nearest[0][0] if nearest else None,
        "nearest_active_stencils": [
            {
                "gap_m": gap,
                "kind": kind,
                "volume_ids": ids,
                "all_fixed": all_fixed,
                "touches_lip": touches_lip,
                "touches_nonmanifold_vertex": touches_nonmanifold,
            }
            for gap, kind, ids, all_fixed, touches_lip, touches_nonmanifold in nearest[
                :20
            ]
        ],
        "barrier_energy": float(contact.fun(state, u)),
        "contact_gradient_l2_full": float(torch.linalg.vector_norm(gradient)),
        "contact_gradient_l2_free": float(
            torch.linalg.vector_norm(gradient[~torch.as_tensor(fixed)])
        ),
        "force_by_zone": zone_force,
        "force_l1_vertex_norm_by_group": group_force,
        "distance_note": "Active stencil Euclidean gaps; these exclude inactive primitive pairs.",
        "force_note": "Contact gradient only, before bulk FEM force; fixed reactions are projected out of the free solve.",
    }


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists()
    volume = pv.read(cfg.fixture)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    group_ids = np.asarray(volume.point_data["GroupId"], dtype=np.int64)
    group_names = [
        str(name) for name in np.asarray(volume.field_data["GroupName"]).ravel()
    ]
    lip = np.array(["Lip" in group_names[i] for i in group_ids])
    surface = volume.extract_surface(algorithm=None, pass_pointid=True)
    ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = ids[np.asarray(surface.faces).reshape(-1, 4)[:, 1:]]
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold = np.zeros(volume.n_points, dtype=bool)
    nonmanifold[unique_edges[edge_counts > 2]] = True
    contact, receipt = build_self_contact(volume)
    sources = {
        str(path.resolve()): sha256(path)
        for path in [
            Path(__file__),
            Path(__file__).with_name("transition_contact.py"),
            cfg.fixture,
            cfg.pilot / "source-manifest.json",
        ]
    }
    states = {"rest": np.zeros((volume.n_points, 3), dtype=np.float64)}
    failed_path = cfg.pilot / "failed-solver-state.npz"
    if failed_path.exists():
        with np.load(failed_path) as z:
            states["failed_solver_state"] = z["u"].copy()
        sources[str(failed_path.resolve())] = sha256(failed_path)
    result = {
        "schema": "full-boundary-contact-cpu-preflight-v1",
        "sources_sha256": sources,
        "contact_receipt": receipt,
        "mesh": {
            "volume_points": volume.n_points,
            "boundary_faces": len(faces),
            "all_fixed_boundary_faces": int(np.count_nonzero(fixed[faces].all(axis=1))),
            "nonmanifold_boundary_edges": int(np.count_nonzero(edge_counts > 2)),
            "nonmanifold_boundary_vertices": int(nonmanifold.sum()),
            "lip_boundary_vertices": int(lip[ids].sum()),
        },
        "states": {},
        "failed_state_present": failed_path.exists(),
    }
    for name, displacement in states.items():
        LOG.info("Auditing %s", name)
        assert displacement.shape == (volume.n_points, 3)
        result["states"][name] = audit_state(
            contact,
            displacement,
            fixed,
            {"ids": group_ids, "names": group_names},
            lip,
            nonmanifold,
        )
    out.joinpath("summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "contact/rest_active_stencils": sum(
                result["states"]["rest"]["active_stencils_by_type"].values()
            ),
            "contact/rest_free_gradient_l2": result["states"]["rest"][
                "contact_gradient_l2_free"
            ],
        }
    )
    LOG.info("Wrote %s", out / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
