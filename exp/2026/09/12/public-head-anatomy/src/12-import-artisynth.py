# Copyright 2026 liblaf
"""Acquire and export the pinned public ArtiSynth Badin face model.

The exported muscle curves retain upstream names and marker identifiers.  The
four declared MAS paths remain audit-only because the marker file consumed by
``BadinFaceDemo`` does not contain their marker numbers.  Right-side curves are
the demo's explicit runtime reflection across ``y=0``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from anatomy_common import ProfileCometNoCommit, sha256, write_json
from artisynth_source import (
    PINNED_COMMIT,
    REPOSITORY_URL,
    SOURCE_FILES,
    ArtiSynthFaceSource,
    MusclePath,
    load_face_source,
)

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = cherries.output("12-public-models/artisynth", mkdir=True)
    offline: bool = False


def volume_grid(source: ArtiSynthFaceSource) -> pv.UnstructuredGrid:
    """Convert mixed ANSYS hex/wedge connectivity without changing cell order."""
    cells: list[np.ndarray] = []
    vtk_types = np.empty(source.mesh.cell_types.size, dtype=np.uint8)
    for index, (kind, point_ids) in enumerate(
        zip(
            source.mesh.cell_types,
            source.mesh.cell_point_indices,
            strict=True,
        )
    ):
        if kind == "hex8":
            connectivity = point_ids
            vtk_types[index] = pv.CellType.HEXAHEDRON
        else:
            assert kind == "wedge6"
            assert point_ids[2] == point_ids[3]
            assert point_ids[6] == point_ids[7]
            connectivity = point_ids[[0, 1, 2, 4, 5, 6]]
            vtk_types[index] = pv.CellType.WEDGE
        cells.append(np.concatenate(([connectivity.size], connectivity)))
    grid = pv.UnstructuredGrid(np.concatenate(cells), vtk_types, source.mesh.points_m)
    grid.point_data["AnsysNodeNumber"] = source.mesh.node_numbers
    grid.cell_data["AnsysElementNumber"] = source.mesh.element_numbers
    for column, name in enumerate(
        (
            "AnsysMaterialNumber",
            "AnsysElementTypeNumber",
            "AnsysRealConstantNumber",
            "AnsysSectionNumber",
            "AnsysCoordinateSystemNumber",
        )
    ):
        grid.cell_data[name] = source.mesh.ansys_attributes[:, column]
    grid.field_data["LengthUnit"] = np.asarray(["m"])
    grid.field_data["SourceRepository"] = np.asarray([REPOSITORY_URL])
    grid.field_data["SourceCommit"] = np.asarray([PINNED_COMMIT])
    return grid


def muscle_polydata(paths: tuple[MusclePath, ...]) -> pv.PolyData:
    """Create one VTK polyline cell per resolved muscle path."""
    assert paths
    assert all(path.points_m is not None for path in paths)
    points: list[np.ndarray] = []
    marker_numbers: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    offset = 0
    for path in paths:
        assert path.points_m is not None
        count = path.points_m.shape[0]
        points.append(path.points_m)
        marker_numbers.append(path.source_marker_numbers)
        lines.append(np.concatenate(([count], np.arange(offset, offset + count))))
        offset += count
    poly = pv.PolyData(np.concatenate(points), lines=np.concatenate(lines))
    poly.point_data["SourceMarkerNumber"] = np.concatenate(marker_numbers)
    poly.cell_data["PathId"] = np.asarray([path.path_id for path in paths])
    poly.cell_data["MuscleIndex"] = np.asarray(
        [path.muscle_index for path in paths], dtype=np.int64
    )
    poly.cell_data["MuscleName"] = np.asarray([path.muscle_name for path in paths])
    poly.cell_data["AnatomicalName"] = np.asarray(
        [path.anatomical_name for path in paths]
    )
    poly.cell_data["FascicleIndex"] = np.asarray(
        [path.fascicle_index for path in paths], dtype=np.int64
    )
    poly.cell_data["Side"] = np.asarray([path.side for path in paths])
    poly.cell_data["Derivation"] = np.asarray([path.derivation for path in paths])
    poly.field_data["LengthUnit"] = np.asarray(["m"])
    poly.field_data["SourceCommit"] = np.asarray([PINNED_COMMIT])
    return poly


def attachment_polydata(source: ArtiSynthFaceSource) -> pv.PolyData:
    """Export attachment sets as vertices with both upstream index conventions."""
    points = np.concatenate([item.points_m for item in source.attachments])
    poly = pv.PolyData(points)
    poly.verts = np.column_stack(
        (np.ones(points.shape[0], dtype=np.int64), np.arange(points.shape[0]))
    ).ravel()
    poly.point_data["AttachmentId"] = np.concatenate(
        [
            np.full(item.source_numbers.size, index, dtype=np.int64)
            for index, item in enumerate(source.attachments)
        ]
    )
    poly.point_data["AttachmentName"] = np.concatenate(
        [np.full(item.source_numbers.size, item.name) for item in source.attachments]
    )
    poly.point_data["SourceNumber"] = np.concatenate(
        [item.source_numbers for item in source.attachments]
    )
    poly.point_data["FemPointIndex"] = np.concatenate(
        [item.point_indices for item in source.attachments]
    )
    poly.point_data["AnsysNodeNumber"] = np.concatenate(
        [item.node_numbers for item in source.attachments]
    )
    poly.field_data["LengthUnit"] = np.asarray(["m"])
    poly.field_data["AttachmentNames"] = np.asarray(
        [item.name for item in source.attachments]
    )
    poly.field_data["SourceReference"] = np.asarray(
        [item.source_reference for item in source.attachments]
    )
    poly.field_data["ConsumerAction"] = np.asarray(
        [item.consumer_action for item in source.attachments]
    )
    poly.field_data["AnatomicalLimit"] = np.asarray(
        [item.anatomical_limit for item in source.attachments]
    )
    poly.field_data["SourceCommit"] = np.asarray([PINNED_COMMIT])
    return poly


def validate_exports(
    source: ArtiSynthFaceSource,
    volume_path: Path,
    source_paths_path: Path,
    runtime_paths_path: Path,
    attachments_path: Path,
) -> dict[str, object]:
    """Read the VTK files back and verify counts and identifier round trips."""
    volume = pv.read(volume_path)
    source_paths = pv.read(source_paths_path)
    runtime_paths = pv.read(runtime_paths_path)
    attachments = pv.read(attachments_path)
    resolved = tuple(path for path in source.source_muscle_paths if path.resolved)
    expected_attachment_count = sum(
        item.source_numbers.size for item in source.attachments
    )
    assert volume.n_points == source.mesh.points_m.shape[0]
    assert volume.n_cells == source.mesh.cell_types.size
    assert np.array_equal(volume["AnsysNodeNumber"], source.mesh.node_numbers)
    assert np.array_equal(volume["AnsysElementNumber"], source.mesh.element_numbers)
    for index, (kind, point_ids) in enumerate(
        zip(source.mesh.cell_types, source.mesh.cell_point_indices, strict=True)
    ):
        expected = point_ids if kind == "hex8" else point_ids[[0, 1, 2, 4, 5, 6]]
        assert np.array_equal(volume.get_cell(index).point_ids, expected), index
    assert source_paths.n_cells == len(resolved)
    assert runtime_paths.n_cells == len(source.runtime_muscle_paths)
    assert np.array_equal(
        source_paths["SourceMarkerNumber"],
        np.concatenate([path.source_marker_numbers for path in resolved]),
    )
    assert np.array_equal(
        runtime_paths["SourceMarkerNumber"],
        np.concatenate(
            [path.source_marker_numbers for path in source.runtime_muscle_paths]
        ),
    )
    assert np.array_equal(
        source_paths.cell_data["PathId"],
        np.asarray([path.path_id for path in resolved]),
    )
    assert np.array_equal(
        runtime_paths.cell_data["PathId"],
        np.asarray([path.path_id for path in source.runtime_muscle_paths]),
    )
    assert attachments.n_points == expected_attachment_count
    assert attachments.n_cells == expected_attachment_count
    assert np.array_equal(
        attachments["SourceNumber"],
        np.concatenate([item.source_numbers for item in source.attachments]),
    )
    assert np.array_equal(
        attachments["FemPointIndex"],
        np.concatenate([item.point_indices for item in source.attachments]),
    )
    assert np.array_equal(
        attachments["AnsysNodeNumber"],
        np.concatenate([item.node_numbers for item in source.attachments]),
    )
    return {
        "volume": {
            "points": volume.n_points,
            "cells": volume.n_cells,
            "hex8_cells": int(
                np.count_nonzero(volume.celltypes == pv.CellType.HEXAHEDRON)
            ),
            "wedge6_cells": int(
                np.count_nonzero(volume.celltypes == pv.CellType.WEDGE)
            ),
            "node_number_round_trip": True,
            "element_number_round_trip": True,
            "cell_order_and_connectivity_round_trip": True,
        },
        "source_paths": {
            "resolved_polyline_cells": source_paths.n_cells,
            "points": source_paths.n_points,
            "source_marker_number_round_trip": True,
            "path_id_round_trip": True,
        },
        "runtime_paths": {
            "polyline_cells": runtime_paths.n_cells,
            "points": runtime_paths.n_points,
            "source_marker_number_round_trip": True,
            "path_id_round_trip": True,
        },
        "attachments": {
            "vertex_cells": attachments.n_cells,
            "points": attachments.n_points,
            "fem_point_index_round_trip": True,
            "ansys_node_number_round_trip": True,
            "source_number_round_trip": True,
        },
    }


def main(cfg: Config) -> None:
    output = cfg.output
    output.mkdir(parents=True, exist_ok=True)
    source = load_face_source(output / "source-cache", offline=cfg.offline)
    volume_path = output / "volume.vtu"
    source_paths_path = output / "muscle-paths-source.vtp"
    runtime_paths_path = output / "muscle-paths-runtime.vtp"
    attachments_path = output / "attachments.vtp"
    volume_grid(source).save(volume_path)
    muscle_polydata(
        tuple(path for path in source.source_muscle_paths if path.resolved)
    ).save(source_paths_path)
    muscle_polydata(source.runtime_muscle_paths).save(runtime_paths_path)
    attachment_polydata(source).save(attachments_path)
    validation = validate_exports(
        source,
        volume_path,
        source_paths_path,
        runtime_paths_path,
        attachments_path,
    )
    output_files = (
        volume_path,
        source_paths_path,
        runtime_paths_path,
        attachments_path,
    )
    source_manifest = [
        {
            "path": item.path,
            "sha256": item.sha256,
            "role": item.role,
            "cached_path": str(source.cached_files[item.path].relative_to(output)),
        }
        for item in SOURCE_FILES
    ]
    write_json(
        output / "source-manifest.json",
        {
            "repository": REPOSITORY_URL,
            "commit": PINNED_COMMIT,
            "files": source_manifest,
        },
    )
    audit = source.to_jsonable(include_arrays=False)
    audit["exports"] = {
        "files": [
            {
                "path": path.name,
                "sha256": sha256(path),
                "format": path.suffix.removeprefix("."),
            }
            for path in output_files
        ],
        "validation": validation,
        "index_semantics": {
            "volume/AnsysNodeNumber": "upstream one-based ANSYS node identifier",
            "volume/AnsysElementNumber": "upstream one-based ANSYS element identifier",
            "muscle_paths/SourceMarkerNumber": (
                "identifier in the separate marker .node file; not a face FEM node"
            ),
            "attachments/SourceNumber": (
                "ANSYS node number for the first three sets; zero-based FEM point-list "
                "index for jaw_rigid_attachment"
            ),
            "attachments/FemPointIndex": "zero-based row in volume points",
            "attachments/AnsysNodeNumber": "resolved upstream ANSYS node identifier",
        },
        "known_absent": (
            "No FRO/frontalis path is declared by this pinned Badin source package; "
            "no substitute path or attachment was fabricated."
        ),
    }
    write_json(output / "audit.json", audit)
    cherries.log_metrics(
        {
            "artisynth/volume_points": validation["volume"]["points"],
            "artisynth/volume_cells": validation["volume"]["cells"],
            "artisynth/resolved_source_paths": validation["source_paths"][
                "resolved_polyline_cells"
            ],
            "artisynth/runtime_paths": validation["runtime_paths"]["polyline_cells"],
            "artisynth/attachment_points": validation["attachments"]["points"],
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
