"""Prepare a separate all-muscle/original-fixation fixture for Raw6 comparisons."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "face-activation-materials/data/10-fixture"
OUTPUT = ROOT / "data/12-historical-fixture"


def digest(path: Path) -> dict[str, str | int]:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {OUTPUT}")
    source_volume = SOURCE / "volume.vtu"
    source_skin = SOURCE / "skin.vtp"
    for path in (source_volume, source_skin):
        if not path.is_file():
            raise FileNotFoundError(path)

    mesh = pv.read(source_volume)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("expected an unstructured volume fixture")
    active = np.asarray(mesh.cell_data["HistoricalActivationMask"], dtype=bool)
    fixed = np.asarray(mesh.point_data["HistoricalIsFixed"], dtype=bool)
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
    region_ids = np.unique(muscle_id[active])
    names = np.asarray(mesh.field_data["MuscleName"]).astype(str)
    if active.sum() != 288_235 or fixed.sum() != 27_036 or len(region_ids) != 103:
        raise ValueError("historical mask cardinality changed")

    control_id = np.full(mesh.n_cells, -1, dtype=np.int64)
    for index, value in enumerate(region_ids):
        control_id[active & (muscle_id == value)] = index
    if not np.array_equal(np.unique(control_id[active]), np.arange(len(region_ids))):
        raise ValueError("historical control IDs are not contiguous")

    mesh.point_data["IsFixed"] = fixed
    mesh.point_data["FixedMask"] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data["FixedValue"] = np.zeros((mesh.n_points, 3), dtype=np.float64)
    mesh.cell_data["ActivationMask"] = active
    mesh.cell_data["ActivationControlId"] = control_id
    mesh.field_data["ActivationRegionMuscleId"] = region_ids
    mesh.field_data["ActivationRegionName"] = names[region_ids]

    OUTPUT.mkdir(parents=True)
    output_volume = OUTPUT / "volume.vtu"
    output_skin = OUTPUT / "skin.vtp"
    mesh.save(output_volume)
    shutil.copy2(source_skin, output_skin)
    summary = {
        "schema_version": 1,
        "purpose": "separate original-fixation/all-historical-muscle fixture for Raw6 and Raw6-S comparisons",
        "supported_methods": ["Raw6", "Raw6-S", "G5", "G5-S"],
        "unsupported_methods": {
            "FiberSmooth": "fibers were estimated only for the 35-region September screen; excluded historical regions were not inferred here",
            "FiberModes": "the September interpolation basis covers only its 35-region active mask",
            "Region5Modes": "the September interpolation basis covers only its 35-region active mask",
        },
        "active_tetrahedra": int(active.sum()),
        "activation_regions": len(region_ids),
        "fixed_vertices": int(fixed.sum()),
        "fixed_coordinate_dofs": int(3 * fixed.sum()),
        "skin_geometry_role": "retained only for target area weights; run with --skin-factor 0 for zero skin energy",
        "source_hashes": {
            "volume": digest(source_volume),
            "skin": digest(source_skin),
        },
        "output_hashes": {
            "volume": digest(output_volume),
            "skin": digest(output_skin),
            "script": digest(Path(__file__)),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
