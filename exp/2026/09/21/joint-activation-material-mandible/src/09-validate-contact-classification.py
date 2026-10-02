# ruff: noqa: EM101, EM102, TRY003
"""Regression for topology-aware neutral FEM contact classification."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from joint_common import ProfileJoint, write_json
from joint_data import PreparedInputs, audit_neutral_oral_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = cherries.input("prepared")
    checkpoints: tuple[Path, ...] = (
        cherries.input("neutral-prestress-001/best-admissible.pt"),
        cherries.input("neutral-prestress-010/best-admissible.pt"),
    )
    output: Path = cherries.output(
        "prepared/neutral-pilot-oral-audit-v2.json", mkdir=True
    )


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_contact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "numerical_geometry_ok",
            "baseline_contact_pairs",
            "current_contact_pairs",
            "new_contact_pairs",
            "worsened_inherited_pairs",
            "baseline_topology",
            "current_topology",
            "new_nonadjacent_or_overrun_pairs",
            "admission_semantics",
        )
    }


def main(cfg: Config) -> None:
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    reference = np.asarray(volume.points, dtype=np.float64)
    runs = []
    for checkpoint_path in cfg.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        displacement = (
            checkpoint["primal"]["neutral"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        if displacement.shape != reference.shape:
            raise ValueError(
                f"checkpoint displacement shape changed: {checkpoint_path}"
            )
        audit = audit_neutral_oral_geometry(prepared, reference + displacement)
        if not audit["neutral_invariants_ok"]:
            raise ValueError(f"corrected neutral invariant failed: {checkpoint_path}")
        if not audit["numerical_geometry_admissible"]:
            raise ValueError(f"corrected FEM geometry failed: {checkpoint_path}")
        if audit["anatomical_validation"]:
            raise ValueError("numerical audit must not claim anatomical validation")
        for name, record in (
            ("lip", audit["fem_lip"]),
            ("upper_oral", audit["fem_mandible_oral"]["upper_oral"]),
            ("lower_oral", audit["fem_mandible_oral"]["lower_oral"]),
            ("soft_cranium", audit["fem_contact_surfaces"]["soft_cranium"]),
            ("soft_mandible", audit["fem_contact_surfaces"]["soft_mandible"]),
            (
                "mandible_cranium",
                audit["fem_contact_surfaces"]["mandible_cranium"],
            ),
        ):
            if record["new_nonadjacent_or_overrun_pairs"] != 0:
                raise ValueError(
                    f"unexpected nonadjacent/overrun pair in {name}: {checkpoint_path}"
                )
        runs.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_update": checkpoint["metrics"]["update"],
                "saved_numerical_metrics": checkpoint["metrics"]["metrics"],
                "neutral_invariants_ok": audit["neutral_invariants_ok"],
                "numerical_geometry_admissible": audit["numerical_geometry_admissible"],
                "anatomical_validation": audit["anatomical_validation"],
                "full_admissible": audit["admissible"],
                "mandible_pose_consistency": audit["mandible_pose_consistency"],
                "fem_lip": compact_contact(audit["fem_lip"]),
                "fem_mandible_upper_oral": compact_contact(
                    audit["fem_mandible_oral"]["upper_oral"]
                ),
                "fem_mandible_lower_oral": compact_contact(
                    audit["fem_mandible_oral"]["lower_oral"]
                ),
                "fem_soft_cranium": compact_contact(
                    audit["fem_contact_surfaces"]["soft_cranium"]
                ),
                "fem_soft_mandible": compact_contact(
                    audit["fem_contact_surfaces"]["soft_mandible"]
                ),
                "fem_mandible_cranium": compact_contact(
                    audit["fem_contact_surfaces"]["mandible_cranium"]
                ),
            }
        )
    classifier = audit["fem_contact_surfaces"]
    result = {
        "schema_version": 2,
        "purpose": "corrected topology-aware post-hoc neutral FEM contact audit; supersedes the numerical interpretation of v1 without modifying it",
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "raw_vtk_pairs_preserved": True,
        "topology_exclusion": "exclude a shared-vertex/shared-edge pair only when its collision segment endpoints remain within the exact shared topology at the dtype-and-scale-derived tolerance",
        "hard_numerical_gate": "finite zero-pose support plus no nonadjacent or adjacency-overrun FEM intersections; detF/inversion is checked by the physics runner",
        "anatomical_validation": False,
        "source_lip_nonadjacent_pairs": prepared.manifest["oral_contact_qa"][
            "inherited_lip_intersections"
        ]["source_contact_pairs_after_shared_seam_removal"],
        "source_lip_role": "anatomical QA defect; insufficient source-to-FEM correspondence prevents using it as a blanket FEM numerical failure",
        "fem_boundary_partition": {
            key: classifier[key]
            for key in (
                "boundary_triangles",
                "pure_cranium_triangles",
                "pure_mandible_triangles",
                "pure_soft_triangles",
                "bonded_mixed_transition_triangles",
            )
        },
        "runs": runs,
        "all_neutral_invariants_ok": all(run["neutral_invariants_ok"] for run in runs),
        "all_numerical_geometry_admissible": all(
            run["numerical_geometry_admissible"] for run in runs
        ),
        "all_anatomically_validated": all(run["anatomical_validation"] for run in runs),
    }
    write_json(cfg.output, result)
    cherries.log_metrics(
        {
            "contact/runs": len(runs),
            "contact/source_lip_nonadjacent_pairs": result[
                "source_lip_nonadjacent_pairs"
            ],
            "contact/all_numerical_geometry_admissible": float(
                result["all_numerical_geometry_admissible"]
            ),
        }
    )
    LOG.info("Wrote corrected topology-aware audit to %s", cfg.output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
