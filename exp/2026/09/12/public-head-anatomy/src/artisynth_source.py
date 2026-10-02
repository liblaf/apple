# Copyright 2026 liblaf
"""Pinned reader for the public ArtiSynth Badin face source assets.

The upstream model mixes several kinds of data that should not be conflated:

* the ANSYS ``.node``/``.elem`` files are the full facial soft-tissue FEM;
* the muscle ``.node`` file contains embedded marker coordinates, not FEM nodes;
* the ANSYS macro joins those markers into modeled muscle paths;
* named attachment files are operational node sets, not segmented ligaments.

This module preserves those distinctions and records which quantities ArtiSynth
derives from the source paths.  It does not assign anatomical certainty that is
not present upstream.
"""

# Data-validation failures include the offending source path or declaration.
# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

REPOSITORY_URL = "https://github.com/artisynth/artisynth_models"
PINNED_COMMIT = "e37be3f1b4831f347c055cbf7124b6bd57414f98"
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/artisynth/artisynth_models/{PINNED_COMMIT}/"
)
UPSTREAM_CITATION_URL = "https://www.artisynth.org/Models/Citations"
UPSTREAM_MODEL_CITATION = (
    "Stavness I, Nazari M, Flynn C, Perrier P, Payan Y, Lloyd JE, Fels S. "
    "Coupled Biomechanical Modeling of the Face, Jaw, Skull, Tongue, and Hyoid "
    "Bone. In: 3D Multiscale Physiological Human. Springer; 2014:253-274."
)

# BadinFaceDemo uses this plane and tolerance to retain the supplied left side,
# then reflects its muscle markers to create the runtime right side.
MID_SAGITTAL_AXIS = 1
MID_SAGITTAL_TOLERANCE_M = 1.0e-4


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One integrity-pinned upstream file."""

    path: str
    sha256: str
    role: str


SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile(
        "README",
        "a0a970a3b860f4f84ae79477b175ae9712c0505a296d99f25e534ed36a7060c3",
        "upstream package description",
    ),
    SourceFile(
        "LICENSE",
        "cf8c3cf168bb2661cf51ca26159e2203e2e7d0158ec20d17409abb565a3f8294",
        "upstream source-and-data redistribution terms",
    ),
    SourceFile(
        "src/artisynth/models/face/BadinFaceDemo.java",
        "3712ade4a593b8d2f206570a63e80110475d77a352a832ef51a44d2b332b6f01",
        "default geometry, muscle, fixation, mirroring, and jaw-attachment consumer",
    ),
    SourceFile(
        "src/artisynth/models/face/BadinFemMuscleFaceDemo.java",
        "52ac0865eae84234946a4a7ab0e84b031754f77f88933471a0bb70beadc4a123",
        "path-to-element-domain and element-direction consumer",
    ),
    SourceFile(
        "src/artisynth/models/face/AnsysFaceMuscleFiberReader.java",
        "92399bb6accbd233f96ab67b493f847f633d606c41198628bce53f6a0c404567",
        "ANSYS muscle-path consumer",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/badinface_oop_csa_midsagittal.node",
        "dbb0b877e9b4d3644de0d8cf6d02a3564c82837b91a78722fef50047746a06cb",
        "default full-face FEM nodes",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/badinface_oop_csa_midsagittal.elem",
        "5fa5676f248a46d613ea52c3330ea341ff00e880a2f639da097f717c992b2af5",
        "default full-face FEM elements",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/face_muscles_CH_with_LLS.node",
        "5ce873b855529e8cd7a38b00ad1eef6ec49bf4b3c65b812510903165444919a9",
        "left-side embedded muscle-marker coordinates",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/face_muscles_topology_flynn.mac",
        "15d6a34fd797be605dc6d5be443f24d06eb14a1c32f49f8737ee5ec9f1c71429",
        "modeled muscle-path topology and source names",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/edge_nodes.nodenum",
        "6884abbde53558fce836a69a5fbd57cb5f08cf514cf6e247e78e8370b75569ee",
        "world-fixed truncation-edge FEM node numbers",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/nose_nodes_to_fix.txt",
        "0657808b77ed886cbc8d58beada664703ea508e312f1ea6e77e7fd07ec5f8f18",
        "world-fixed inner-nose FEM node numbers",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/zygomatic_ligament_attachments.txt",
        "627c78ba5ff0a7846f9f5df976606ecaf35922b092111cae2a8a059dce971ab4",
        "six world-fixed FEM node numbers named for the zygomatic ligament",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/face_jaw_attachments.txt",
        "22aff0641675d6776e9d8e4653c94ef86ebe0d2165d65bcadef2c1fc1292979d",
        "zero-based FEM point-list indices rigidly attached to the jaw",
    ),
    SourceFile(
        "src/artisynth/models/face/geometry/badinjaw.obj",
        "a903416d0efc2b4458c53ea1683921a3abf55b1e915c9cb900adebd9c6e5457b",
        "rigid jaw receiver surface for the jaw attachment set",
    ),
)

_FACE_NODE_PATH = SOURCE_FILES[5].path
_FACE_ELEMENT_PATH = SOURCE_FILES[6].path
_MUSCLE_NODE_PATH = SOURCE_FILES[7].path
_MUSCLE_MACRO_PATH = SOURCE_FILES[8].path

_FORTRAN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")


@dataclass(frozen=True, slots=True)
class AnsysFaceMesh:
    """Full facial FEM in the source coordinate system.

    ``cell_node_numbers`` retains the eight ANSYS slots.  Wedges use the
    upstream degenerate-hex convention in which slots 3/4 and 7/8 repeat.
    ``cell_point_indices`` is the same connectivity resolved into row indices
    of ``points_m``.
    """

    node_numbers: NDArray[np.int64]
    points_m: NDArray[np.float64]
    element_numbers: NDArray[np.int64]
    cell_node_numbers: NDArray[np.int64]
    cell_point_indices: NDArray[np.int64]
    cell_types: NDArray[np.str_]
    ansys_attributes: NDArray[np.int64]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "node_numbers": self.node_numbers.tolist(),
            "points_m": self.points_m.tolist(),
            "element_numbers": self.element_numbers.tolist(),
            "cell_node_numbers": self.cell_node_numbers.tolist(),
            "cell_point_indices": self.cell_point_indices.tolist(),
            "cell_types": self.cell_types.tolist(),
            "ansys_attributes": self.ansys_attributes.tolist(),
        }


@dataclass(frozen=True, slots=True)
class MusclePath:
    """One macro-declared muscle polyline.

    Marker numbers belong to the separate muscle-marker file.  They are not
    node numbers of the facial FEM.  A path with missing markers is preserved
    as an unresolved declaration with ``points_m=None``.
    """

    path_id: str
    muscle_index: int
    muscle_name: str
    anatomical_name: str
    fascicle_index: int
    source_marker_numbers: NDArray[np.int64]
    points_m: NDArray[np.float64] | None
    missing_marker_numbers: NDArray[np.int64]
    side: Literal["left", "right", "midline", "cross_midline", "unresolved"]
    derivation: Literal["source_polyline", "runtime_mirror_y"]
    runtime_enabled: bool

    @property
    def resolved(self) -> bool:
        return self.points_m is not None

    def to_jsonable(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "muscle_index": self.muscle_index,
            "muscle_name": self.muscle_name,
            "anatomical_name": self.anatomical_name,
            "fascicle_index": self.fascicle_index,
            "source_marker_numbers": self.source_marker_numbers.tolist(),
            "points_m": None if self.points_m is None else self.points_m.tolist(),
            "missing_marker_numbers": self.missing_marker_numbers.tolist(),
            "side": self.side,
            "derivation": self.derivation,
            "runtime_enabled": self.runtime_enabled,
        }


@dataclass(frozen=True, slots=True)
class AttachmentSet:
    """A node set with the reference convention and consumer action retained."""

    name: str
    source_path: str
    source_reference: Literal["ansys_node_number", "zero_based_point_index"]
    source_numbers: NDArray[np.int64]
    point_indices: NDArray[np.int64]
    node_numbers: NDArray[np.int64]
    points_m: NDArray[np.float64]
    consumer_action: str
    anatomical_limit: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "source_reference": self.source_reference,
            "source_numbers": self.source_numbers.tolist(),
            "point_indices": self.point_indices.tolist(),
            "node_numbers": self.node_numbers.tolist(),
            "points_m": self.points_m.tolist(),
            "consumer_action": self.consumer_action,
            "anatomical_limit": self.anatomical_limit,
        }


@dataclass(frozen=True, slots=True)
class ArtiSynthFaceSource:
    """Parsed public source data and a machine-readable interpretation audit."""

    mesh: AnsysFaceMesh
    source_muscle_paths: tuple[MusclePath, ...]
    runtime_muscle_paths: tuple[MusclePath, ...]
    attachments: tuple[AttachmentSet, ...]
    cached_files: dict[str, Path]
    audit: dict[str, object]

    def to_jsonable(self, *, include_arrays: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "audit": self.audit,
            "cached_files": {
                key: str(value) for key, value in self.cached_files.items()
            },
            "source_muscle_paths": [
                path.to_jsonable() for path in self.source_muscle_paths
            ],
            "runtime_muscle_paths": [
                path.to_jsonable() for path in self.runtime_muscle_paths
            ],
            "attachments": [item.to_jsonable() for item in self.attachments],
        }
        if include_arrays:
            data["mesh"] = self.mesh.to_jsonable()
        else:
            unique, counts = np.unique(self.mesh.cell_types, return_counts=True)
            data["mesh"] = {
                "n_points": int(self.mesh.points_m.shape[0]),
                "n_cells": int(self.mesh.cell_node_numbers.shape[0]),
                "cell_type_counts": {
                    str(name): int(count)
                    for name, count in zip(unique, counts, strict=True)
                },
                "units": "m",
            }
        return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sources(cache_dir: Path | str, *, offline: bool = False) -> dict[str, Path]:
    """Fetch the integrity-pinned source subset or validate an existing cache.

    Files are stored below ``cache_dir`` with their upstream relative paths.
    Existing files must match the manifest; a mismatched cache fails visibly.
    """
    cache_dir = Path(cache_dir)
    result: dict[str, Path] = {}
    for source in SOURCE_FILES:
        destination = cache_dir / source.path
        if destination.exists():
            actual = _sha256(destination)
            if actual != source.sha256:
                raise ValueError(
                    f"cached source checksum mismatch for {source.path}: "
                    f"expected {source.sha256}, got {actual}"
                )
        else:
            if offline:
                raise FileNotFoundError(f"missing offline source: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.download")
            request = urllib.request.Request(
                RAW_BASE_URL + source.path,
                headers={"User-Agent": "liblaf-apple-public-anatomy"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            actual = _sha256(temporary)
            if actual != source.sha256:
                temporary.unlink()
                raise ValueError(
                    f"downloaded source checksum mismatch for {source.path}: "
                    f"expected {source.sha256}, got {actual}"
                )
            temporary.replace(destination)
        result[source.path] = destination
    return result


def parse_ansys_nodes(
    path: Path | str,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Parse a four-column ANSYS node file, including adjacent signed fields."""
    numbers: list[int] = []
    points: list[tuple[float, float, float]] = []
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("!", maxsplit=1)[0].strip()
        if not line:
            continue
        tokens = _FORTRAN_NUMBER.findall(line)
        if len(tokens) != 4:
            raise ValueError(f"{path}:{line_number}: expected node number and xyz")
        numbers.append(int(tokens[0]))
        xyz = tuple(
            float(value.replace("D", "E").replace("d", "e")) for value in tokens[1:]
        )
        points.append(xyz)
    node_numbers = np.asarray(numbers, dtype=np.int64)
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.shape != (node_numbers.size, 3):
        raise ValueError(f"{path}: invalid point array shape {points_array.shape}")
    if np.unique(node_numbers).size != node_numbers.size:
        raise ValueError(f"{path}: duplicate node numbers")
    return node_numbers, points_array


def parse_ansys_elements(
    path: Path | str, node_numbers: NDArray[np.int64]
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.str_],
    NDArray[np.int64],
]:
    """Parse the pinned 14-column ANSYS mixed hex/wedge element format."""
    element_numbers: list[int] = []
    cells: list[list[int]] = []
    attributes: list[list[int]] = []
    cell_types: list[str] = []
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("!", maxsplit=1)[0].strip()
        if not line:
            continue
        values = [int(value) for value in _INTEGER.findall(line)]
        if len(values) != 14:
            raise ValueError(f"{path}:{line_number}: expected 14 integer columns")
        cell = values[:8]
        unique_count = len(set(cell))
        if unique_count == 8:
            cell_type = "hex8"
        elif unique_count == 6 and cell[3] == cell[2] and cell[7] == cell[6]:
            cell_type = "wedge6"
        else:
            raise ValueError(
                f"{path}:{line_number}: unsupported degenerate element connectivity"
            )
        cells.append(cell)
        attributes.append(values[8:13])
        element_numbers.append(values[13])
        cell_types.append(cell_type)

    element_array = np.asarray(element_numbers, dtype=np.int64)
    cell_node_numbers = np.asarray(cells, dtype=np.int64)
    attribute_array = np.asarray(attributes, dtype=np.int64)
    if np.unique(element_array).size != element_array.size:
        raise ValueError(f"{path}: duplicate element numbers")

    lookup = {int(number): index for index, number in enumerate(node_numbers)}
    try:
        point_indices = np.asarray(
            [[lookup[int(number)] for number in cell] for cell in cell_node_numbers],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"{path}: element references missing node {error.args[0]}"
        ) from error
    return (
        element_array,
        cell_node_numbers,
        point_indices,
        np.asarray(cell_types, dtype=np.str_),
        attribute_array,
    )


def load_ansys_face_mesh(
    node_path: Path | str, element_path: Path | str
) -> AnsysFaceMesh:
    """Load the full face FEM without converting or tetrahedralizing it."""
    node_numbers, points_m = parse_ansys_nodes(node_path)
    elements = parse_ansys_elements(element_path, node_numbers)
    return AnsysFaceMesh(node_numbers, points_m, *elements)


def _macro_integer(text: str, name: str) -> int:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(\d+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"missing macro declaration {name}")
    return int(match.group(1))


def _path_side(
    points_m: NDArray[np.float64],
) -> Literal["left", "right", "midline", "cross_midline"]:
    transverse = points_m[:, MID_SAGITTAL_AXIS]
    if np.all(np.abs(transverse) <= MID_SAGITTAL_TOLERANCE_M):
        return "midline"
    if np.all(transverse <= MID_SAGITTAL_TOLERANCE_M):
        return "left"
    if np.all(transverse >= -MID_SAGITTAL_TOLERANCE_M):
        return "right"
    return "cross_midline"


def parse_muscle_paths(
    marker_path: Path | str, macro_path: Path | str
) -> tuple[MusclePath, ...]:
    """Parse every declared source muscle path, including unresolved masseter paths."""
    marker_numbers, marker_points = parse_ansys_nodes(marker_path)
    marker_lookup = {
        int(number): marker_points[index] for index, number in enumerate(marker_numbers)
    }
    text = Path(macro_path).read_text()
    active = {
        int(match.group(1)): (match.group(2), match.group(3).strip())
        for match in re.finditer(
            r"^\s*ACTIV\((\d+)\)\s*=\s*([A-Za-z0-9_]+)\s*!\s*(.+)$",
            text,
            flags=re.MULTILINE,
        )
    }
    declared_fiber_counts = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(
            r"^\s*NB_FIBERS\((\d+)\)\s*=\s*(\d+)", text, flags=re.MULTILINE
        )
    }
    declared_node_counts = {
        (int(match.group(1)), int(match.group(2))): int(match.group(3))
        for match in re.finditer(
            r"^\s*NB_NODES_FIBER\((\d+),(\d+)\)\s*=\s*(\d+)",
            text,
            flags=re.MULTILINE,
        )
    }
    assignments = {
        (int(match.group(1)), int(match.group(2)), int(match.group(3))): int(
            match.group(4)
        )
        for match in re.finditer(
            r"^\s*FIBER\((\d+),(\d+),(\d+)\)\s*=\s*(\d+)",
            text,
            flags=re.MULTILINE,
        )
    }

    n_muscles = _macro_integer(text, "nb_muscles")
    if set(active) != set(range(1, n_muscles + 1)):
        raise ValueError("macro muscle names do not cover nb_muscles")
    if set(declared_fiber_counts) != set(active):
        raise ValueError("macro fiber counts do not cover named muscles")

    result: list[MusclePath] = []
    for muscle_index in range(1, n_muscles + 1):
        abbreviation, anatomical_name = active[muscle_index]
        for fascicle_index in range(1, declared_fiber_counts[muscle_index] + 1):
            key = (muscle_index, fascicle_index)
            if key not in declared_node_counts:
                raise ValueError(f"missing node count for muscle path {key}")
            count = declared_node_counts[key]
            try:
                source_numbers = np.asarray(
                    [
                        assignments[muscle_index, fascicle_index, node_index]
                        for node_index in range(1, count + 1)
                    ],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    f"incomplete macro path assignment {error.args[0]}"
                ) from error
            missing = np.asarray(
                [
                    number
                    for number in source_numbers
                    if int(number) not in marker_lookup
                ],
                dtype=np.int64,
            )
            points: NDArray[np.float64] | None
            side: Literal["left", "right", "midline", "cross_midline", "unresolved"]
            if missing.size:
                points = None
                side = "unresolved"
            else:
                points = np.asarray(
                    [marker_lookup[int(number)] for number in source_numbers],
                    dtype=np.float64,
                )
                side = _path_side(points)
            result.append(
                MusclePath(
                    path_id=f"{abbreviation}:{fascicle_index}:source",
                    muscle_index=muscle_index,
                    muscle_name=abbreviation,
                    anatomical_name=anatomical_name,
                    fascicle_index=fascicle_index,
                    source_marker_numbers=source_numbers,
                    points_m=points,
                    missing_marker_numbers=missing,
                    side=side,
                    derivation="source_polyline",
                    runtime_enabled=abbreviation != "MAS" and points is not None,
                )
            )

    unresolved_names = {path.muscle_name for path in result if not path.resolved}
    if unresolved_names != {"MAS"}:
        raise ValueError(
            f"unexpected unresolved muscle paths: {sorted(unresolved_names)}"
        )
    return tuple(result)


def make_runtime_muscle_paths(
    source_paths: tuple[MusclePath, ...],
) -> tuple[MusclePath, ...]:
    """Return usable source paths plus ArtiSynth's explicitly derived mirrors."""
    result: list[MusclePath] = []
    for path in source_paths:
        if not path.runtime_enabled:
            continue
        if path.points_m is None or path.side != "left":
            raise ValueError(
                f"runtime source path is not a resolved left path: {path.path_id}"
            )
        result.append(path)
        mirrored = path.points_m.copy()
        mirrored[:, MID_SAGITTAL_AXIS] *= -1.0
        result.append(
            MusclePath(
                path_id=f"{path.muscle_name}:{path.fascicle_index}:runtime-mirror-right",
                muscle_index=path.muscle_index,
                muscle_name=path.muscle_name,
                anatomical_name=path.anatomical_name,
                fascicle_index=path.fascicle_index,
                source_marker_numbers=path.source_marker_numbers.copy(),
                points_m=mirrored,
                missing_marker_numbers=np.empty(0, dtype=np.int64),
                side="right",
                derivation="runtime_mirror_y",
                runtime_enabled=True,
            )
        )
    return tuple(result)


def _parse_integer_list(path: Path | str) -> NDArray[np.int64]:
    values = np.asarray(
        [int(value) for value in _INTEGER.findall(Path(path).read_text())],
        dtype=np.int64,
    )
    if values.size == 0 or np.unique(values).size != values.size:
        raise ValueError(f"{path}: expected a nonempty unique integer list")
    return values


def _attachment(
    *,
    name: str,
    path: Path,
    source_path: str,
    source_reference: Literal["ansys_node_number", "zero_based_point_index"],
    mesh: AnsysFaceMesh,
    consumer_action: str,
    anatomical_limit: str,
) -> AttachmentSet:
    source_numbers = _parse_integer_list(path)
    if source_reference == "ansys_node_number":
        lookup = {int(number): index for index, number in enumerate(mesh.node_numbers)}
        try:
            point_indices = np.asarray(
                [lookup[int(number)] for number in source_numbers], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                f"{source_path}: missing FEM node number {error.args[0]}"
            ) from error
    else:
        point_indices = source_numbers.copy()
        if np.any(point_indices < 0) or np.any(point_indices >= mesh.node_numbers.size):
            raise ValueError(f"{source_path}: point-list index out of range")
    return AttachmentSet(
        name=name,
        source_path=source_path,
        source_reference=source_reference,
        source_numbers=source_numbers,
        point_indices=point_indices,
        node_numbers=mesh.node_numbers[point_indices],
        points_m=mesh.points_m[point_indices],
        consumer_action=consumer_action,
        anatomical_limit=anatomical_limit,
    )


def load_attachments(
    files: dict[str, Path], mesh: AnsysFaceMesh
) -> tuple[AttachmentSet, ...]:
    """Load only node sets consumed by the default full-face demo."""
    geometry = "src/artisynth/models/face/geometry/"
    specs = (
        (
            "truncation_edge_fixed",
            "edge_nodes.nodenum",
            "ansys_node_number",
            "set non-dynamic in world coordinates",
            "operational cut-boundary fixation, not an anatomical attachment",
        ),
        (
            "inner_nose_fixed",
            "nose_nodes_to_fix.txt",
            "ansys_node_number",
            "set non-dynamic in world coordinates",
            "operational support set; no receiving anatomy or compliance is encoded",
        ),
        (
            "zygomatic_ligament_fixed",
            "zygomatic_ligament_attachments.txt",
            "ansys_node_number",
            "set six nodes non-dynamic in world coordinates",
            "the anatomical name labels a fixation set; no ligament geometry, second endpoint, direction, rest length, or constitutive law is supplied",
        ),
        (
            "jaw_rigid_attachment",
            "face_jaw_attachments.txt",
            "zero_based_point_index",
            "rigidly attach FEM points to the badinjaw rigid body",
            "a modeled rigid tie to jaw, not a measured muscle-to-bone or muscle-to-dermis insertion map",
        ),
    )
    result: list[AttachmentSet] = []
    for name, filename, reference, action, limit in specs:
        source_path = geometry + filename
        result.append(
            _attachment(
                name=name,
                path=files[source_path],
                source_path=source_path,
                source_reference=reference,  # type: ignore[arg-type]
                mesh=mesh,
                consumer_action=action,
                anatomical_limit=limit,
            )
        )
    return tuple(result)


def load_face_source(
    cache_dir: Path | str, *, offline: bool = False
) -> ArtiSynthFaceSource:
    """Fetch, validate, and parse the pinned default full-face source subset."""
    files = fetch_sources(cache_dir, offline=offline)
    mesh = load_ansys_face_mesh(files[_FACE_NODE_PATH], files[_FACE_ELEMENT_PATH])
    source_paths = parse_muscle_paths(
        files[_MUSCLE_NODE_PATH], files[_MUSCLE_MACRO_PATH]
    )
    runtime_paths = make_runtime_muscle_paths(source_paths)
    attachments = load_attachments(files, mesh)
    cell_types, cell_counts = np.unique(mesh.cell_types, return_counts=True)
    source_enabled = tuple(path for path in source_paths if path.runtime_enabled)
    source_missing = tuple(path for path in source_paths if not path.resolved)
    audit: dict[str, object] = {
        "source": {
            "repository": REPOSITORY_URL,
            "commit": PINNED_COMMIT,
            "demo_class": "artisynth.models.face.BadinFaceDemo",
            "fem_demo_class": "artisynth.models.face.BadinFemMuscleFaceDemo",
            "license": (
                "ArtiSynth Models package custom permissive terms: retain notices "
                "for source/data redistribution, reproduce notices for binaries, "
                "and cite original work in academic publications"
            ),
            "citation_url": UPSTREAM_CITATION_URL,
            "model_citation": UPSTREAM_MODEL_CITATION,
        },
        "coordinate_system": {
            "length_unit": "m",
            "unit_status": (
                "inferred from the consuming Java model, which reads at scale 1, "
                "uses a 0.006 m muscle-domain radius and SI density/stress; raw ANSYS "
                "files contain no unit header"
            ),
            "mid_sagittal_plane": "y=0",
            "source_muscle_side": "left (negative y), with right side mirrored at runtime",
        },
        "mesh": {
            "points": int(mesh.points_m.shape[0]),
            "cells": int(mesh.cell_node_numbers.shape[0]),
            "cell_type_counts": {
                str(name): int(count)
                for name, count in zip(cell_types, cell_counts, strict=True)
            },
            "tissue_labels": None,
            "skin_layer": None,
        },
        "muscles": {
            "macro_author_and_date": "Julie Groleau; created 2007-06-04, modified 2007-07-17",
            "declared_names": sorted({path.muscle_name for path in source_paths}),
            "declared_source_paths": len(source_paths),
            "resolved_left_source_paths": len(source_enabled),
            "runtime_paths_including_mirrors": len(runtime_paths),
            "unresolved_paths": [path.path_id for path in source_missing],
            "unresolved_reason": (
                "the selected marker file omits all MAS marker numbers; the Java "
                "demo also explicitly removes the MAS bundle"
            ),
            "path_semantics": (
                "modeled embedded marker polylines; provenance does not establish "
                "specimen-measured fascicles"
            ),
            "derived_element_semantics": (
                "BadinFemMuscleFaceDemo disables axial paths, selects elements within "
                "0.006 m of each path, and computes element directions; OOP then uses "
                "a special ring-element selection and centroid-derived directions"
            ),
        },
        "attachments": {
            item.name: {
                "count": int(item.source_numbers.size),
                "source_reference": item.source_reference,
                "consumer_action": item.consumer_action,
                "anatomical_limit": item.anatomical_limit,
            }
            for item in attachments
        },
        "known_absent_or_unproven": [
            "measured fascicle trajectories or uncertainty",
            "volumetric per-cell fiber measurements",
            "muscle volume labels in the base FEM",
            "explicit muscle origins and insertions tied to receiving surfaces",
            "muscle-to-dermis attachment map",
            "segmented zygomatic or mandibular ligament geometry",
            "retinacula cutis, regional SMAS, or aponeurosis geometry",
            "attachment directions, reference lengths, slack, or constitutive laws",
            "same-donor provenance for face, muscle paths, skull, and jaw",
        ],
    }
    return ArtiSynthFaceSource(
        mesh=mesh,
        source_muscle_paths=source_paths,
        runtime_muscle_paths=runtime_paths,
        attachments=attachments,
        cached_files=files,
        audit=audit,
    )
