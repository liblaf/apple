"""Audit completeness and ownership of the frozen FEM contact surface."""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_contact import build_owned_contact
from joint_data import PreparedInputs, _collision_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_npz: Path = GROUP / "data/prepared/inputs.npz"
    prepared_manifest: Path = GROUP / "data/prepared/manifest.json"
    contact_config: Path = GROUP / "data/contact/config.json"
    checkpoint: Path = GROUP / "data/neutral-convergence-010-contact/terminal.pt"
    output_dir: Path = cherries.output("fem-contact-surface-audit", mkdir=True)


def load_stable_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    )
    return torch.load(path, map_location="cpu", weights_only=False), hashlib.sha256(
        payload
    ).hexdigest()


def boundary_data(volume: pv.UnstructuredGrid) -> dict[str, Any]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.point_data["GlobalPointId"] = original
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    local_faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    faces = original[local_faces]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    cranium_id, mandible_id = names.index("Cranium"), names.index("Mandible")
    face_labels = labels[faces]
    cranium = np.all(face_labels == cranium_id, axis=1)
    mandible = np.all(face_labels == mandible_id, axis=1)
    soft = np.all(~np.isin(face_labels, (cranium_id, mandible_id)), axis=1)
    mixed = ~(cranium | mandible | soft)
    kind = np.full(boundary.n_cells, 3, dtype=np.int8)
    kind[cranium], kind[mandible], kind[soft] = 0, 1, 2
    points = np.asarray(volume.points, dtype=np.float64)
    triangles = points[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return {
        "boundary": boundary,
        "original": original,
        "faces": faces,
        "names": names,
        "labels": labels,
        "cranium_id": cranium_id,
        "mandible_id": mandible_id,
        "masks": {
            "cranium": cranium,
            "mandible": mandible,
            "soft": soft,
            "mixed": mixed,
        },
        "kind": kind,
        "area": area,
    }


def mixed_topology(data: dict[str, Any], points: np.ndarray) -> dict[str, Any]:
    faces = data["faces"]
    labels = data["labels"]
    cranium_id = data["cranium_id"]
    mandible_id = data["mandible_id"]
    mixed = data["masks"]["mixed"]
    kind = data["kind"]
    area = data["area"]
    edge_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for face_id, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_faces[tuple(sorted((int(first), int(second))))].append(face_id)
    adjacency = [set() for _ in range(len(faces))]
    for incident in edge_faces.values():
        for face_id in incident:
            adjacency[face_id].update(set(incident) - {face_id})

    unvisited = set(map(int, np.flatnonzero(mixed)))
    components: list[dict[str, Any]] = []
    aggregate_faces: collections.Counter[tuple[str, str]] = collections.Counter()
    aggregate_components: collections.Counter[tuple[str, str]] = collections.Counter()
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        component = [seed]
        while stack:
            face_id = stack.pop()
            for other in adjacency[face_id]:
                if mixed[other] and other in unvisited:
                    unvisited.remove(other)
                    stack.append(other)
                    component.append(other)
        component_ids = np.asarray(component, dtype=np.int64)
        values = labels[faces[component_ids]].ravel()
        present = "".join(
            value
            for value, predicate in (
                ("C", np.any(values == cranium_id)),
                ("M", np.any(values == mandible_id)),
                ("S", np.any(~np.isin(values, (cranium_id, mandible_id)))),
            )
            if predicate
        )
        touching = sorted(
            {
                int(kind[other])
                for face_id in component
                for other in adjacency[face_id]
                if not mixed[other]
            }
        )
        touches = "".join({0: "C", 1: "M", 2: "S"}[value] for value in touching)
        key = (present, touches or "none")
        aggregate_faces[key] += len(component)
        aggregate_components[key] += 1
        ids = np.unique(faces[component_ids])
        components.append(
            {
                "component_id": len(components),
                "faces": len(component),
                "area_m2": float(area[component_ids].sum()),
                "labels_present": present,
                "touches_pure_classes_by_edge": touches or "none",
                "bounds_m": [
                    points[ids].min(axis=0).tolist(),
                    points[ids].max(axis=0).tolist(),
                ],
                "boundary_cell_ids": sorted(component),
            }
        )

    face_patterns: collections.Counter[tuple[int, int, int]] = collections.Counter()
    pattern_area: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for face_id in np.flatnonzero(mixed):
        values = labels[faces[face_id]]
        key = (
            int(np.sum(values == cranium_id)),
            int(np.sum(values == mandible_id)),
            int(np.sum(~np.isin(values, (cranium_id, mandible_id)))),
        )
        face_patterns[key] += 1
        pattern_area[key] += float(area[face_id])

    selected_vertices = np.unique(faces[~mixed])
    mixed_vertices = np.unique(faces[mixed])
    mixed_only = np.setdiff1d(mixed_vertices, selected_vertices)
    nonmanifold = {
        edge: incident for edge, incident in edge_faces.items() if len(incident) != 2
    }
    nonmanifold_signatures = collections.Counter(
        tuple(
            sorted(
                {0: "C", 1: "M", 2: "S", 3: "X"}[int(kind[index])] for index in incident
            )
        )
        for incident in nonmanifold.values()
    )
    return {
        "mixed_faces": int(mixed.sum()),
        "mixed_area_m2": float(area[mixed].sum()),
        "mixed_boundary_area_fraction": float(area[mixed].sum() / area.sum()),
        "mixed_vertices": len(mixed_vertices),
        "mixed_vertices_also_on_selected_faces": len(
            np.intersect1d(mixed_vertices, selected_vertices)
        ),
        "mixed_only_vertices_omitted_from_collision_mesh": len(mixed_only),
        "mixed_only_vertex_label_counts": dict(
            sorted(
                collections.Counter(
                    data["names"][value] for value in labels[mixed_only]
                ).items()
            )
        ),
        "face_patterns": {
            f"cranium={key[0]},mandible={key[1]},soft={key[2]}": {
                "faces": value,
                "area_m2": pattern_area[key],
            }
            for key, value in sorted(face_patterns.items())
        },
        "component_count": len(components),
        "component_aggregate": {
            f"labels={key[0]},touches={key[1]}": {
                "components": aggregate_components[key],
                "faces": value,
            }
            for key, value in sorted(aggregate_faces.items())
        },
        "components": components,
        "boundary_edge_incidence": dict(
            sorted(collections.Counter(map(len, edge_faces.values())).items())
        ),
        "four_face_nonmanifold_edges": len(nonmanifold),
        "nonmanifold_edge_face_class_signatures": {
            "+".join(key): value
            for key, value in sorted(nonmanifold_signatures.items())
        },
        "interpretation": "GroupId was transferred by nearest source triangle and provides no independent free-versus-bonded attachment tag. Mixed faces are omitted by declared modeling policy; topology alone does not validate every component as anatomical attachment.",
    }


def subset_surface(
    data: dict[str, Any], mask: np.ndarray, points: np.ndarray
) -> pv.PolyData:
    boundary = data["boundary"].copy()
    boundary.points = points[data["original"]]
    return (
        boundary.extract_cells(np.flatnonzero(mask))
        .extract_surface(algorithm=None)
        .triangulate()
    )


def mixed_intersection_audit(
    data: dict[str, Any], points: np.ndarray
) -> dict[str, Any]:
    mixed = subset_surface(data, data["masks"]["mixed"], points)
    mixed_faces = np.asarray(mixed.faces).reshape(-1, 4)[:, 1:]
    mixed_global = np.asarray(mixed.point_data["GlobalPointId"], dtype=np.int64)[
        mixed_faces
    ]
    result: dict[str, Any] = {}
    for bone in ("cranium", "mandible"):
        surface = subset_surface(data, data["masks"][bone], points)
        faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        global_faces = np.asarray(surface.point_data["GlobalPointId"], dtype=np.int64)[
            faces
        ]
        pairs, _, _, _ = _collision_geometry(mixed, surface)
        adjacent = np.asarray(
            [
                np.intersect1d(mixed_global[first], global_faces[second]).size > 0
                for first, second in pairs
            ],
            dtype=bool,
        )
        result[bone] = {
            "raw_triangle_intersection_pairs": len(pairs),
            "pairs_sharing_a_global_fem_vertex": int(adjacent.sum()),
            "nonadjacent_intersection_pairs": int((~adjacent).sum()),
        }
    return result


def candidate_key(kind: str, candidate: Any) -> tuple[Any, ...]:
    if kind == "ee":
        return kind, *sorted((candidate.edge0_id, candidate.edge1_id))
    if kind == "fv":
        return kind, candidate.face_id, candidate.vertex_id
    if kind == "ev":
        return kind, candidate.edge_id, candidate.vertex_id
    return kind, *sorted((candidate.vertex0_id, candidate.vertex1_id))


def candidate_set(
    candidates: Any,
    mesh: Any,
    patch: np.ndarray,
    *,
    cross_patch_only: bool,
) -> set[tuple[Any, ...]]:
    result: set[tuple[Any, ...]] = set()
    for kind in ("vv", "ev", "ee", "fv"):
        for candidate in getattr(candidates, f"{kind}_candidates"):
            ids = np.asarray(
                candidate.vertex_ids(mesh.edges, mesh.faces), dtype=np.int64
            )
            ids = ids[ids >= 0]
            if cross_patch_only and np.unique(patch[ids]).size != 2:
                continue
            result.add(candidate_key(kind, candidate))
    return result


def primitive_ownership(
    state: Any,
    contact: Any,
    labels: np.ndarray,
    names: list[str],
    cranium_id: int,
    mandible_id: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    invalid: list[dict[str, Any]] = []
    global_ids = np.asarray(contact.indices, dtype=np.int64)
    for role, object_, suffix in (
        ("candidate", state.candidates, "candidates"),
        ("active", state.collisions, "collisions"),
    ):
        for kind in ("vv", "ev", "ee", "fv"):
            for primitive in getattr(object_, f"{kind}_{suffix}"):
                ids = np.asarray(
                    primitive.vertex_ids(
                        contact.collision_mesh.edges, contact.collision_mesh.faces
                    ),
                    dtype=np.int64,
                )
                ids = ids[ids >= 0]
                global_ = global_ids[ids]
                values = labels[global_]
                has_cranium = np.any(values == cranium_id)
                has_mandible = np.any(values == mandible_id)
                has_soft = np.any(~np.isin(values, (cranium_id, mandible_id)))
                if has_cranium and not has_mandible and has_soft:
                    target = "cranium"
                elif has_mandible and not has_cranium and has_soft:
                    target = "mandible"
                else:
                    target = "invalid"
                    invalid.append(
                        {
                            "role": role,
                            "kind": kind,
                            "global_point_ids": global_.tolist(),
                            "labels": [names[value] for value in values],
                        }
                    )
                counts[f"{role}_{kind}_{target}"] += 1
    return dict(sorted(counts.items())), invalid


def candidate_completeness(
    contact: Any,
    displacement: torch.Tensor,
    labels: np.ndarray,
    names: list[str],
    cranium_id: int,
    mandible_id: int,
) -> dict[str, Any]:
    state = contact.state_at(displacement)
    positions = (contact.vertices + displacement[contact.indices]).numpy(force=True)
    unfiltered = ipctk.CollisionMesh(
        np.asarray(contact.collision_mesh.rest_positions),
        np.asarray(contact.collision_mesh.edges),
        np.asarray(contact.collision_mesh.faces),
    )
    unfiltered.init_adjacencies()
    all_candidates = ipctk.Candidates()
    all_candidates.build(
        mesh=unfiltered,
        vertices=positions,
        inflation_radius=contact.inflation_radius,
        broad_phase=ipctk.LBVH(),
    )
    global_ids = np.asarray(contact.indices, dtype=np.int64)
    patch = np.isin(labels[global_ids], (cranium_id, mandible_id))
    filtered_set = candidate_set(
        state.candidates,
        contact.collision_mesh,
        patch,
        cross_patch_only=False,
    )
    independent_cross_set = candidate_set(
        all_candidates, unfiltered, patch, cross_patch_only=True
    )
    counts, invalid = primitive_ownership(
        state, contact, labels, names, cranium_id, mandible_id
    )
    missing = independent_cross_set - filtered_set
    extra = filtered_set - independent_cross_set
    assert not invalid
    assert not missing
    assert not extra
    return {
        "unfiltered_broad_phase_candidates": len(all_candidates),
        "independent_soft_bone_candidates": len(independent_cross_set),
        "runtime_filtered_candidates": len(filtered_set),
        "missing_runtime_soft_bone_candidates": len(missing),
        "extra_runtime_candidates": len(extra),
        "candidate_and_active_ownership_counts": counts,
        "invalid_ownership_primitives": invalid,
        "runtime_active_collision_representatives": len(state.collisions),
        "scope": "selected pure-face collision mesh at this state; candidate counts are representation diagnostics, not contact area",
    }


def support_audit(data: dict[str, Any], prepared: PreparedInputs) -> dict[str, Any]:
    faces = data["faces"]
    labels = data["labels"]
    masks = data["masks"]
    cranium = np.asarray(prepared.arrays["cranium_node_ids"], dtype=np.int64)
    mandible = np.asarray(prepared.arrays["mandible_node_ids"], dtype=np.int64)
    pure_cranium = np.unique(faces[masks["cranium"]])
    pure_mandible = np.unique(faces[masks["mandible"]])
    pure_soft = np.unique(faces[masks["soft"]])
    return {
        "cranium_support_nodes": len(cranium),
        "mandible_support_nodes": len(mandible),
        "support_overlap": len(np.intersect1d(cranium, mandible)),
        "pure_cranium_vertices": len(pure_cranium),
        "pure_cranium_vertices_outside_fixed_support": len(
            np.setdiff1d(pure_cranium, cranium)
        ),
        "pure_mandible_vertices": len(pure_mandible),
        "pure_mandible_vertices_outside_rigid_support": len(
            np.setdiff1d(pure_mandible, mandible)
        ),
        "pure_soft_vertices_in_either_bone_support": len(
            np.intersect1d(pure_soft, np.union1d(cranium, mandible))
        ),
        "cranium_support_noncranium_labels": int(
            np.sum(labels[cranium] != data["cranium_id"])
        ),
        "mandible_support_nonmandible_labels": int(
            np.sum(labels[mandible] != data["mandible_id"])
        ),
    }


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    mixed = summary["mixed_boundary"]
    reference = summary["candidate_completeness"]["reference"]
    current = summary["candidate_completeness"]["current_neutral"]
    report = f"""# Frozen FEM contact-surface audit

The current IPC surface is complete under its declared pure-face discretization:
all {summary["partition"]["pure_cranium_faces"]:,} pure cranium faces,
{summary["partition"]["pure_mandible_faces"]:,} pure mandible faces, and
{summary["partition"]["pure_soft_faces"]:,} pure soft faces are selected. All
pure cranium vertices belong to fixed cranium support, all pure mandible
vertices belong to differentiable rigid-jaw support, and no pure-soft vertex
belongs to either bone support.

An independent unfiltered broad phase contains exactly the same soft-to-bone
candidate subset as the runtime patch filter. Reference has
{reference["independent_soft_bone_candidates"]:,} independent cross-patch and
{reference["runtime_filtered_candidates"]:,} runtime candidates; current
neutral has {current["independent_soft_bone_candidates"]:,} and
{current["runtime_filtered_candidates"]:,}. Missing and extra counts are zero
in both states. Every candidate and active stencil contains soft tissue plus
exactly one of cranium or mandible. The moving-jaw and fixed-cranium surfaces
therefore use the same complete selected collision map; swept CCD rebuilds from
that map and filter.

The omitted {mixed["mixed_faces"]:,} faces occupy
{100 * mixed["mixed_boundary_area_fraction"]:.4f}% of boundary area. They are
not all literally bone-soft faces: the exact patterns include 413
cranium-mandible faces and two cranium-mandible-soft faces. The nearest-source
GroupId transfer has no independent free-versus-bonded tag. Mixed-component
topology includes components that bridge the expected pure classes and small
one-sided label islands, so “bonded mixed faces” is a declared discretized
model policy rather than verified attachment anatomy. The collision mesh also
omits {mixed["mixed_only_vertices_omitted_from_collision_mesh"]:,} vertices
used only by mixed faces.

Mixed faces have no nonadjacent triangle intersections with either selected
pure bone surface in the reference or current neutral state; every raw
mixed/pure-bone intersection shares a global FEM vertex. No concrete omitted
free-face penetration is therefore verified. The full FEM boundary has
{mixed["four_face_nonmanifold_edges"]} four-face edges, another mesh-topology
limitation that does not change the exact candidate-completeness result.

The frozen FEM contact map is justified as the current numerical model choice,
with zero unhandled selected soft-to-bone candidates in the two audited states.
This does not validate mixed attachment anatomy or complete registered source
bone coverage. Active mechanics and the input manifest were not changed.
"""
    (output_dir / "report.md").write_text(report)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(cfg.prepared_npz, cfg.prepared_manifest)
    volume = pv.read(prepared.manifest["fixture"]["volume_path"])
    data = boundary_data(volume)
    checkpoint, checkpoint_hash = load_stable_checkpoint(cfg.checkpoint)
    displacement = checkpoint["primal"]["neutral"].detach().cpu()
    assert displacement.shape == (volume.n_points, 3)
    config = json.loads(cfg.contact_config.read_text())
    contact, receipt = build_owned_contact(
        volume, prepared.arrays["cranium_node_ids"], config
    )
    masks = data["masks"]
    selected = masks["cranium"] | masks["mandible"] | masks["soft"]
    assert np.all(selected ^ masks["mixed"])
    reference_points = np.asarray(volume.points, dtype=np.float64)
    current_points = reference_points + displacement.numpy(force=True)
    mixed = mixed_topology(data, reference_points)
    support = support_audit(data, prepared)
    assert support["support_overlap"] == 0
    assert support["pure_cranium_vertices_outside_fixed_support"] == 0
    assert support["pure_mandible_vertices_outside_rigid_support"] == 0
    assert support["pure_soft_vertices_in_either_bone_support"] == 0
    summary = {
        "schema": "joint-fem-contact-surface-audit-v1",
        "purpose": "read-only completeness and ownership audit of the frozen selected FEM contact surface",
        "prepared_inputs_sha256": sha256(cfg.prepared_npz),
        "prepared_manifest_sha256": sha256(cfg.prepared_manifest),
        "contact_config_sha256": sha256(cfg.contact_config),
        "contact_implementation_sha256": sha256(GROUP / "src/joint_contact.py"),
        "group_transfer_source": {
            "path": str(
                Path(os.environ["APPLE_MELON_HEAD"]).parent / "src/42-gen-masks.py"
            ),
            "sha256": sha256(
                Path(
                    str(
                        Path(os.environ["APPLE_MELON_HEAD"]).parent
                        / "src/42-gen-masks.py"
                    )
                )
            ),
            "semantics": "boundary point GroupId is transferred from the closest source triangle with snap_to_closest_point=True; no free-versus-bonded attachment field is created",
        },
        "checkpoint": {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "update": checkpoint["update"],
            "neutral_converged": bool(checkpoint["neutral_converged"]),
        },
        "partition": {
            "boundary_faces": len(data["faces"]),
            "selected_faces": int(selected.sum()),
            "pure_cranium_faces": int(masks["cranium"].sum()),
            "pure_mandible_faces": int(masks["mandible"].sum()),
            "pure_soft_faces": int(masks["soft"].sum()),
            "mixed_faces_omitted": int(masks["mixed"].sum()),
            "pure_face_omissions": 0,
        },
        "support_ownership": support,
        "runtime_surface_receipt": receipt,
        "mixed_boundary": mixed,
        "mixed_vs_selected_bone_intersections": {
            "reference": mixed_intersection_audit(data, reference_points),
            "current_neutral": mixed_intersection_audit(data, current_points),
        },
        "candidate_completeness": {
            "reference": candidate_completeness(
                contact,
                torch.zeros_like(displacement),
                data["labels"],
                data["names"],
                data["cranium_id"],
                data["mandible_id"],
            ),
            "current_neutral": candidate_completeness(
                contact,
                displacement,
                data["labels"],
                data["names"],
                data["cranium_id"],
                data["mandible_id"],
            ),
        },
        "conclusion": {
            "selected_fem_soft_bone_candidate_coverage": "pass",
            "unhandled_selected_soft_bone_candidates_reference": 0,
            "unhandled_selected_soft_bone_candidates_current_neutral": 0,
            "verified_omitted_free_pure_bone_faces": 0,
            "mixed_attachment_anatomy_validated": False,
            "complete_source_bone_coverage_validated": False,
            "interpretation": "The frozen pure-face contact surface is internally complete as a discretized numerical model. Omitted mixed faces remain an explicit attachment-model uncertainty because source GroupId transfer does not encode free-versus-bonded anatomy.",
            "active_mechanics_changed": False,
        },
    }
    for state in summary["mixed_vs_selected_bone_intersections"].values():
        assert all(
            value["nonadjacent_intersection_pairs"] == 0 for value in state.values()
        )
    write_json(cfg.output_dir / "summary.json", summary)
    write_report(cfg.output_dir, summary)
    cherries.log_metrics(
        {
            "fem_contact/pure_face_omissions": 0,
            "fem_contact/reference_missing_candidates": 0,
            "fem_contact/current_missing_candidates": 0,
            "fem_contact/mixed_faces": mixed["mixed_faces"],
        }
    )
    LOG.info("Wrote FEM contact surface audit to %s", cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
