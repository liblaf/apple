"""Classify finite inversions in the completed contact-off MouthOpen forward trial."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("64-forward-inversions")
    forward: Path = GROUP / "data/49-forward-contact-off"
    fixture: Path = GROUP / "data/30-pruned-fixture"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detf(points: np.ndarray, tets: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    current = points + displacement
    rest_dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    current_dm = np.transpose(current[tets[:, 1:]] - current[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(current_dm) / np.linalg.det(rest_dm)


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    summary_path, final_path = cfg.forward / "summary.json", cfg.forward / "final.npz"
    forward = json.loads(summary_path.read_text())
    assert forward["status"] == "completed"
    assert forward["final_checkpoint"]["sha256"] == sha256(final_path)
    with np.load(final_path, allow_pickle=False) as source:
        displacement = source["displacement"].copy()
        assert float(source["fraction"]) == 1
    mesh = pv.read(cfg.fixture / "volume.vtu")
    points = np.asarray(mesh.points)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    jacobian = detf(points, tets, displacement)
    inverted_ids = np.flatnonzero(jacobian <= 0)
    assert len(inverted_ids) == int(forward["final"]["inverted_cells"])
    rest_dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    rest_volume = np.linalg.det(rest_dm) / 6
    assert np.all(rest_volume > 0)
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    group_id = np.asarray(mesh.point_data["GroupId"])
    group_names = [str(name) for name in mesh.field_data["GroupName"]]
    cranium, mandible = group_names.index("Cranium"), group_names.index("Mandible")
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    original = np.asarray(mesh.cell_data["OriginalCellId"], dtype=np.int64)
    fractions = np.column_stack(
        [
            np.asarray(mesh.cell_data[name], dtype=float)
            for name in (
                "FatFraction",
                "MuscleFraction",
                "AponeurosisFraction",
                "SMASFraction",
            )
        ]
    )
    materials = np.asarray(("fat", "muscle", "aponeurosis", "smas"))
    dominant = materials[np.argmax(fractions, axis=1)]
    records = []
    for cell in inverted_ids:
        corners = tets[cell]
        fixed_corners = fixed[corners]
        fixed_groups = group_id[corners[fixed_corners]]
        records.append(
            {
                "new_cell_id": int(cell),
                "original_cell_id": int(original[cell]),
                "J": float(jacobian[cell]),
                "rest_volume_m3": float(rest_volume[cell]),
                "activation_active": bool(active[cell]),
                "fixed_corner_count": int(fixed_corners.sum()),
                "fixed_cranium_corner_count": int(
                    np.count_nonzero(fixed_groups == cranium)
                ),
                "fixed_mandible_corner_count": int(
                    np.count_nonzero(fixed_groups == mandible)
                ),
                "mixed_fixed_cranium_mandible": bool(
                    np.any(fixed_groups == cranium) and np.any(fixed_groups == mandible)
                ),
                "dominant_material": str(dominant[cell]),
                "fractions": {
                    name: float(value)
                    for name, value in zip(
                        ("fat", "muscle", "aponeurosis", "smas"),
                        fractions[cell],
                        strict=True,
                    )
                },
            }
        )
    records.sort(key=lambda row: (row["J"], row["new_cell_id"]))
    (out / "inverted-cells.json").write_text(json.dumps(records, indent=2) + "\n")

    def count(key: str, value: object) -> int:
        return sum(row[key] == value for row in records)

    result = {
        "schema": "mouthopen-forward-inversion-audit-v1",
        "inputs": {
            "forward_summary": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
            "forward_checkpoint": {
                "path": str(final_path),
                "sha256": sha256(final_path),
            },
            "volume": {
                "path": str(cfg.fixture / "volume.vtu"),
                "sha256": sha256(cfg.fixture / "volume.vtu"),
            },
        },
        "count": len(records),
        "rest_volume_fraction": float(
            rest_volume[inverted_ids].sum() / rest_volume.sum()
        ),
        "J": {
            "minimum": float(jacobian[inverted_ids].min()),
            "median": float(np.median(jacobian[inverted_ids])),
            "p90_least_negative": float(np.quantile(jacobian[inverted_ids], 0.9)),
            "severe_J_at_most_minus_1": int(
                np.count_nonzero(jacobian[inverted_ids] <= -1)
            ),
            "severe_J_at_most_minus_5": int(
                np.count_nonzero(jacobian[inverted_ids] <= -5)
            ),
        },
        "activation": {
            "active": int(active[inverted_ids].sum()),
            "inactive": int((~active[inverted_ids]).sum()),
        },
        "fixed_corners": {str(n): count("fixed_corner_count", n) for n in range(5)},
        "mixed_fixed_cranium_mandible": sum(
            row["mixed_fixed_cranium_mandible"] for row in records
        ),
        "dominant_material": {
            name: count("dominant_material", name) for name in materials
        },
        "records": {
            "path": str(out / "inverted-cells.json"),
            "sort": "increasing J, then new_cell_id",
        },
        "interpretation": "This is a cell classification only. It does not establish a mechanical cause of the inversions or validate the contact-off geometry.",
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "inverted_cells": len(records),
            "activation_active_inverted_cells": result["activation"]["active"],
            "minimum_J": result["J"]["minimum"],
        }
    )
    LOG.info("Wrote inversion classification to %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
