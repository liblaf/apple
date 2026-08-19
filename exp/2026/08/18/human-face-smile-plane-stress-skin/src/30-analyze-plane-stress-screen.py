from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _reference import (
    GROUP_DIR,
    KOITER_IMPLEMENTATION,
    KOITER_IMPLEMENTATION_SHA256,
    LEGACY_INVERSE,
    LEGACY_INVERSE_SHA256,
    MATERIAL_REFERENCE_GROUP,
    MATERIAL_REFERENCE_SRC,
    PREPARED_MESH,
    REPO_ROOT,
    RUNTIME_REFERENCE_SRC,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    VOLUME_FORWARD_IMPLEMENTATION,
    VOLUME_FORWARD_IMPLEMENTATION_SHA256,
    VOLUME_LAME_IMPLEMENTATION,
    VOLUME_LAME_IMPLEMENTATION_SHA256,
    enable_reference_modules,
)
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

from liblaf import cherries
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

enable_reference_modules()

from _human_face_output import (  # noqa: E402
    bumpiness_metrics,
    surface_edges_for_mask,
)
from _material_heuristics import (  # noqa: E402
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 3
AGGREGATE_SCHEMA_VERSION = 4
EXPECTED_EVALUATIONS = 41
TERMINAL_STEP = EXPECTED_EVALUATIONS - 1
EXPECTED_LR = 0.3

CORRECTED_LABEL = "isface-e0200-p000"
NO_SKIN_LABEL = "no-skin"
HISTORICAL_LABEL = "old-e100-p000"
PRIMARY_CASE_ORDER = (CORRECTED_LABEL, NO_SKIN_LABEL)
SECONDARY_CASE_ORDER = (HISTORICAL_LABEL,)
RENDER_CASE_ORDER = (CORRECTED_LABEL, HISTORICAL_LABEL, NO_SKIN_LABEL)
EXPECTED_CASES = set(RENDER_CASE_ORDER)
EXPECTED_PARAMETERS: dict[str, tuple[float | None, float | None]] = {
    CORRECTED_LABEL: (1.0, 0.0),
    HISTORICAL_LABEL: (1.0, 0.0),
    NO_SKIN_LABEL: (None, None),
}
DISPLAY_NAMES = {
    CORRECTED_LABEL: "corrected IsFace + plane stress + cut fixed",
    HISTORICAL_LABEL: "old-boundary full skin + 3D lambda control",
    NO_SKIN_LABEL: "old-boundary no-skin control",
}

DESIGN = "isface-plane-stress-hard-fixed-corrected-baseline-inverse"
MANIFEST_DESIGN = "isface-plane-stress-corrected-baseline"
MANIFEST_NAME = "10-corrected-baseline-manifest.json"
AGGREGATE_NAME = "20-corrected-baseline-screen-summary.json"
CORRECTED_STEM = (
    "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen"
)
OLD_DATA_DIR = MATERIAL_REFERENCE_GROUP / "data"
CORRECTED_DATA_DIR = GROUP_DIR / "data"
CORRECTED_CASE_SUMMARY_ARCHIVE = (
    CORRECTED_DATA_DIR / f"{CORRECTED_STEM}-summary-final.json"
)
CORRECTED_AGGREGATE_ARCHIVE = (
    CORRECTED_DATA_DIR / "20-corrected-baseline-screen-summary-final.json"
)
ARCHIVE_METADATA_SNAPSHOT_POLICY = (
    "unique post-rewrite copies avoid the Local plugin same-name overwrite bug"
)

ANALYZED_INVERSE = Path(__file__).with_name("20-inverse-plane-stress-screen.py")
ANALYZED_INVERSE_SIZE_BYTES = 67_639
ANALYZED_INVERSE_SHA256 = (
    "8c5d75ea06d66e60800d1c83c800d365bef01372340f88a650ef44732ea18f4d"
)
PREPARE_IMPLEMENTATION = Path(__file__).with_name("10-prepare-plane-stress-skin.py")
PREPARE_IMPLEMENTATION_SHA256 = (
    "b0a547389dbb192e46732e84bd649d27ee4e89246bf6823d7dcc587322d4bed9"
)

PREPARED_MESH_SIZE_BYTES = 76_792_914
PREPARED_MESH_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_SKIN_COMPONENTS = 1
EXPECTED_SKIN_BOUNDARY_EDGES = 707
EXPECTED_FULL_BOUNDARY_TRIANGLES = 128_172
EXPECTED_FULL_UNASSIGNED_GROUP_POINTS = 6_000
EXPECTED_ARTIFICIAL_CUT_TRIANGLES = 13_165
EXPECTED_CUT_INCIDENT_VERTICES = 6_980
EXPECTED_CUT_PREEXISTING_FIXED_VERTICES = 380
EXPECTED_CUT_NEWLY_FIXED_VERTICES = 6_600
EXPECTED_HISTORICAL_FIXED_VERTICES = 27_036
EXPECTED_HISTORICAL_FIXED_DOFS = 81_108
EXPECTED_MODEL_FIXED_VERTICES = 33_636
EXPECTED_MODEL_FIXED_DOFS = 100_908
EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256 = (
    "8207cda8f9e11dbb4406f683e5ad818a6950e3515ac373719514094fb5b7fe5d"
)
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    "ca39cdc839855be34e75222964a1e5c129dd210e8800c684d7e6d1ce6424f138"
)
HARD_FIXED_CUT_BOUNDARY_POLICY = "all-artificial-cut-incident-vertices-hard-fixed"
CUT_BOUNDARY_MARKER = "source skin triangle touches mapped GroupId=-1 vertex"
KOITER_ENERGY_MEASURE = "fixed original reference area"
INVERSE_RUNTIME_DEPENDENCIES = (
    (
        "runtime/_human_face_case.py",
        RUNTIME_REFERENCE_SRC / "_human_face_case.py",
        "5e9e67be5246ecc9cf38c3a5c53fe4e2041c3b6af272dceab1ed8c94a9bf5d68",
    ),
    (
        "runtime/_human_face_config.py",
        RUNTIME_REFERENCE_SRC / "_human_face_config.py",
        "eca100cc6bdd4e2a1ac689c6e2e7e02cf80a9bea8fa9ac12e9590eca5f23ffb6",
    ),
    (
        "runtime/_human_face_forward.py",
        RUNTIME_REFERENCE_SRC / "_human_face_forward.py",
        "2d0ff39b13555300c000e6dd43e16c274752263b703746ad8174072033819e03",
    ),
    (
        "runtime/_human_face_loop.py",
        RUNTIME_REFERENCE_SRC / "_human_face_loop.py",
        "97a47be4f6140a0213a84b460c8585f92e38657b01db4ae21f67177048f915c5",
    ),
    (
        "runtime/_human_face_mesh.py",
        RUNTIME_REFERENCE_SRC / "_human_face_mesh.py",
        "f1e1cdc806273c4ce5a37e52e3032d357b44bfd201de3fc58c35d793d11454bc",
    ),
    (
        "runtime/_human_face_metrics.py",
        RUNTIME_REFERENCE_SRC / "_human_face_metrics.py",
        "1407d2988444b31332f2688c6535eca5db58b5be31d63fae6abd6bf8bf78e0c1",
    ),
    (
        "runtime/_human_face_output.py",
        RUNTIME_REFERENCE_SRC / "_human_face_output.py",
        "29bae977a4b31e82276aca15fdaae3bdda37e6a3e71493876b6fd973db1a1c61",
    ),
    (
        "runtime/_human_face_runtime.py",
        RUNTIME_REFERENCE_SRC / "_human_face_runtime.py",
        "b2aefe4b5cd702c837d08442f7b588fafb80f1e6c8a745eed874ce18fdce1f45",
    ),
    (
        "runtime/_human_face_skin.py",
        RUNTIME_REFERENCE_SRC / "_human_face_skin.py",
        "a3bded895ff949dab274707e068d323d1277284a46680fe513b069e207a119a9",
    ),
    (
        "runtime/_human_face_targets.py",
        RUNTIME_REFERENCE_SRC / "_human_face_targets.py",
        "34a1583fcb8f90f357647dd4574e2e7ef27f8049f2b3ba1e2fa7dc838fcbb696",
    ),
    (
        "material/_material_heuristics.py",
        MATERIAL_REFERENCE_SRC / "_material_heuristics.py",
        "d21091bb931ed2d218d65f72305792e1a48ced5e703d6b344388d2d1d803c84f",
    ),
    (
        "core/src/liblaf/apple/inverse/_diff_forward.py",
        REPO_ROOT / "src/liblaf/apple/inverse/_diff_forward.py",
        "72de3eeb2a1cfe9addc29aea812f13c077a4f2e098ab65bbe564837d04a5fe30",
    ),
)
EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256 = (
    "3086071201576008047a0b86394e4282c8dc2d37bc0c21a8c8bd4edc73932426"
)
AREA_ATOL_M2 = 5.0e-13
JSON_RTOL = 1.0e-10
JSON_ATOL = 1.0e-12
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14

FACE_GROUPS = (
    "Chin",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
)
LAME_CONVERSION = (
    "thin-membrane plane-stress reduction: "
    "lambda = E * nu / (1 - nu**2); "
    "mu = E / (2 * (1 + nu))"
)
VOLUME_LAME_CONVERSION = (
    "unchanged 3D isotropic volume convention: "
    "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
    "mu = E / (2 * (1 + nu))"
)


@dataclass(frozen=True)
class FileIdentity:
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {"size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class HistoricalSpec:
    label: str
    candidate: str
    stem: str
    summary: FileIdentity
    trace: FileIdentity
    history: FileIdentity
    result: FileIdentity
    target: FileIdentity


NO_SKIN_SPEC = HistoricalSpec(
    label=NO_SKIN_LABEL,
    candidate=NO_SKIN_LABEL,
    stem="20-human-face-smile-no-skin-lr3-material-no-skin-screen",
    summary=FileIdentity(
        114_796,
        "4f3fdb590df48377453a7df4b990cd99df3d8e03ee274da1ae376c2bd04fd1da",
    ),
    trace=FileIdentity(
        86_560,
        "ab8167401cc3de9c4c58f284d665824b80ed56e673b768f4deea43a8d0f43a95",
    ),
    history=FileIdentity(
        2_077_120_296,
        "45e3aef89f62e0ac8f88ea0f08d4c1deaef57ae336ecd856cd23a99f26305642",
    ),
    result=FileIdentity(
        148_115_004,
        "31ea5e9631dc9112e832d2a47cc6fd84e9fffc7a403b20828089a46d36395a49",
    ),
    target=FileIdentity(
        84_419_492,
        "58a2f997dec6e9b3d39e02ab122b9dfc5f0689815e4bbd613a786d21a41a4075",
    ),
)

HISTORICAL_SPEC = HistoricalSpec(
    label=HISTORICAL_LABEL,
    candidate="e100-p000",
    stem="20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen",
    summary=FileIdentity(
        123_434,
        "cba0574628ddef2f41fa79af14e9f84577e3d1fea9a1dec2ec6796822e621d65",
    ),
    trace=FileIdentity(
        87_435,
        "9afef41e0a7553666fe87fb8c464624af51c1ed2e421e33a09af22689007fae5",
    ),
    history=FileIdentity(
        2_073_226_098,
        "05550fa7559c2f78aad6f34460edf58a6fe3a18b3dd4c7527231d366dfabb80d",
    ),
    result=FileIdentity(
        148_064_384,
        "0596f3dcf378f745d80533ac6bd7c0c3f289846e6320e761ef5e10d899e556d5",
    ),
    target=FileIdentity(
        84_419_492,
        "58a2f997dec6e9b3d39e02ab122b9dfc5f0689815e4bbd613a786d21a41a4075",
    ),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_metric_skin: Path = cherries.input(SOURCE_SKIN)
    input_manifest: Path = cherries.input(MANIFEST_NAME)
    input_corrected_summary: Path = cherries.input(AGGREGATE_NAME)
    output_json: Path = cherries.output(
        "30-corrected-baseline-analysis.json", mkdir=True
    )
    output_csv: Path = cherries.output(
        "30-corrected-baseline-trajectories.csv", mkdir=True
    )
    output_table: Path = cherries.output(
        "30-corrected-baseline-checkpoints.md", mkdir=True
    )
    output_plot: Path = cherries.output(
        "30-corrected-baseline-trajectories.png", mkdir=True
    )
    output_terminal_views: Path = cherries.output(
        "30-corrected-baseline-terminal-views.png", mkdir=True
    )
    output_matched_views: Path = cherries.output(
        "30-corrected-baseline-matched-views.png", mkdir=True
    )


@dataclass
class TemporalHistory:
    label: str
    path: Path
    reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, label: str, path: Path) -> TemporalHistory:
        reader = pv.get_reader(path)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        information = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not information.Has(key):
            msg = f"{label} history has no TIME_STEPS: {path}"
            raise ValueError(msg)
        times = np.asarray(
            [information.Get(key, index) for index in range(information.Length(key))],
            dtype=np.float64,
        )
        expected = np.arange(EXPECTED_EVALUATIONS, dtype=np.float64)
        if not np.array_equal(times, expected):
            msg = f"{label} history TIME_STEPS are not exactly 0..40"
            raise ValueError(msg)
        return cls(label=label, path=path, reader=reader, times=times)

    def frame(self, step: int, *, deep_copy: bool = False) -> pv.UnstructuredGrid:
        if not 0 <= step < self.times.size:
            msg = f"{self.label} has no history step {step}"
            raise IndexError(msg)
        vtk_reader = self.reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        result = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if not isinstance(result, pv.UnstructuredGrid):
            result = result.cast_to_unstructured_grid()
        return result.copy(deep=True) if deep_copy else result


@dataclass(frozen=True)
class CaseInput:
    label: str
    candidate: str
    cohort: str
    material_model: str
    summary_path: Path
    trace_path: Path
    history_path: Path
    result_path: Path
    target_path: Path
    summary: dict[str, Any]
    trace: list[dict[str, Any]]
    identities: dict[str, dict[str, int | str | bool]]
    history: TemporalHistory


@dataclass(frozen=True)
class SurfaceBasis:
    base_points: np.ndarray
    base_cells: np.ndarray
    base_celltypes: np.ndarray
    base_global_ids: np.ndarray
    historical_is_fixed: np.ndarray
    historical_fixed_mask: np.ndarray
    historical_fixed_value: np.ndarray
    corrected_is_fixed: np.ndarray
    cut_mesh_ids: np.ndarray
    cut_global_ids: np.ndarray
    cut_boundary_provenance: dict[str, Any]
    tets: np.ndarray
    rest_six_volume: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    target_rms: float
    activation_mask: np.ndarray
    legacy_edges: np.ndarray
    skin: pv.PolyData
    skin_points: np.ndarray
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray
    target_vertex_normals: np.ndarray
    face_triangle_mask: np.ndarray
    face_vertex_ids: np.ndarray
    face_edges: np.ndarray
    boundary_edges: np.ndarray
    boundary_vertex_ids: np.ndarray
    boundary_band_ids: np.ndarray
    deep_interior_ids: np.ndarray
    contraction_tri_0: np.ndarray
    contraction_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    face_focus: np.ndarray
    face_scale: float
    mouth_focus: np.ndarray
    mouth_scale: float
    eye_cheek_focus: np.ndarray
    eye_cheek_scale: float


def reject_json_constant(value: str) -> None:
    msg = f"non-standard JSON constant {value!r}"
    raise ValueError(msg)


def validate_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{context} contains non-finite number {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, context=f"{context}[{index}]")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
    )
    if not isinstance(value, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    validate_finite_json(value, context=str(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                msg = f"blank JSONL row at {path}:{line_number}"
                raise ValueError(msg)
            row = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(row, dict):
                msg = f"expected a JSON object at {path}:{line_number}"
                raise TypeError(msg)
            validate_finite_json(row, context=f"{path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing input artifact: {path}"
        raise FileNotFoundError(msg)
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_byte_identical_archive(
    live: Path,
    archive: Path,
    *,
    context: str,
) -> dict[str, Any]:
    live_identity = file_identity(live)
    archive_identity = file_identity(archive)
    require_equal(archive_identity, live_identity, f"{context} file identity")
    if archive.read_bytes() != live.read_bytes():
        msg = f"{context} is not byte-identical to {live}"
        raise ValueError(msg)
    return {
        "path": str(archive),
        **archive_identity,
        "live_path": str(live),
        "byte_identical_to_live": True,
    }


def require_recorded_artifact(
    summary: dict[str, Any],
    *,
    name: str,
    path: Path,
) -> dict[str, int | str | bool]:
    """Require one non-self-referential artifact identity from the sidecar."""
    recorded_path = Path(str(summary.get(f"artifact/{name}_path", "")))
    require_equal(
        recorded_path.resolve(), path.resolve(), f"corrected {name} artifact path"
    )
    identity = file_identity(path)
    require_equal(
        summary.get(f"artifact/{name}_size_bytes"),
        identity["size_bytes"],
        f"corrected {name} artifact size",
    )
    require_equal(
        summary.get(f"artifact/{name}_sha256"),
        identity["sha256"],
        f"corrected {name} artifact SHA-256",
    )
    return {**identity, "recorded_in_sidecar": True, "sha256_verified": True}


def validate_inverse_runtime_bundle(bundle: Any) -> None:
    if not isinstance(bundle, dict):
        msg = "corrected aggregate inverse runtime bundle is not an object"
        raise TypeError(msg)
    require_equal(
        set(bundle),
        {"algorithm", "bundle_sha256", "files"},
        "corrected aggregate inverse runtime bundle schema",
    )
    require_equal(
        bundle["algorithm"],
        "sha256(label + NUL + file_sha256 + newline), ordered",
        "corrected aggregate inverse runtime bundle algorithm",
    )
    require_equal(
        bundle["bundle_sha256"],
        EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256,
        "corrected aggregate inverse runtime bundle SHA-256",
    )
    files = bundle["files"]
    if not isinstance(files, list):
        msg = "corrected aggregate inverse runtime files is not a list"
        raise TypeError(msg)
    require_equal(
        len(files),
        len(INVERSE_RUNTIME_DEPENDENCIES),
        "corrected aggregate inverse runtime file count",
    )
    payload = bytearray()
    for index, (recorded, expected) in enumerate(
        zip(files, INVERSE_RUNTIME_DEPENDENCIES, strict=True)
    ):
        if not isinstance(recorded, dict):
            msg = f"corrected aggregate inverse runtime file {index} is not an object"
            raise TypeError(msg)
        require_equal(
            set(recorded),
            {"label", "path", "size_bytes", "sha256"},
            f"corrected aggregate inverse runtime file {index} schema",
        )
        label, path, expected_sha256 = expected
        require_equal(recorded["label"], label, f"inverse runtime file {index} label")
        require_equal(
            Path(str(recorded["path"])).resolve(),
            path.resolve(),
            f"inverse runtime file {label} path",
        )
        identity = file_identity(path)
        require_equal(
            identity,
            {
                "size_bytes": recorded["size_bytes"],
                "sha256": expected_sha256,
            },
            f"inverse runtime file {label} live identity",
        )
        require_equal(
            recorded["sha256"],
            expected_sha256,
            f"inverse runtime file {label} recorded SHA-256",
        )
        payload.extend(f"{label}\0{expected_sha256}\n".encode())
    require_equal(
        hashlib.sha256(payload).hexdigest(),
        EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256,
        "recomputed inverse runtime bundle SHA-256",
    )


def require_identity(
    path: Path,
    expected: FileIdentity,
    *,
    context: str,
) -> dict[str, int | str | bool]:
    actual = file_identity(path)
    if actual != expected.as_dict():
        msg = (
            f"{context} identity mismatch: expected {expected.as_dict()}, got {actual}"
        )
        raise ValueError(msg)
    return {**actual, "sha256_verified": True}


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def require_close(actual: Any, expected: Any, context: str) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=JSON_RTOL, abs_tol=JSON_ATOL
    ):
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def safe_sibling(directory: Path, basename: Any, *, context: str) -> Path:
    name = Path(str(basename))
    if name.name != str(name) or name.is_absolute():
        msg = f"{context} is not a safe basename: {basename!r}"
        raise ValueError(msg)
    result = (directory / name).resolve()
    if result.parent != directory.resolve():
        msg = f"{context} escapes {directory}: {result}"
        raise ValueError(msg)
    return result


def compare_trace_rows(
    label: str,
    embedded: list[dict[str, Any]],
    recorded: list[dict[str, Any]],
) -> None:
    require_equal(len(recorded), len(embedded), f"{label} live trace length")
    for step, (source, live) in enumerate(zip(embedded, recorded, strict=True)):
        require_equal(
            set(source) - set(live),
            {"time/live_plot_s"},
            f"{label} trace step {step} embedded-only keys",
        )
        live_plot_s = source["time/live_plot_s"]
        if not math.isfinite(float(live_plot_s)) or float(live_plot_s) < 0.0:
            msg = f"{label} trace step {step} has invalid time/live_plot_s"
            raise ValueError(msg)
        expected = {
            key: value for key, value in source.items() if key != "time/live_plot_s"
        }
        require_equal(set(live), set(expected), f"{label} trace step {step} keys")
        for key, expected_value in expected.items():
            if isinstance(expected_value, float):
                require_close(live[key], expected_value, f"{label} step {step} {key}")
            else:
                require_equal(live[key], expected_value, f"{label} step {step} {key}")


def triangle_faces(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64)
    if surface.n_cells == 0 or encoded.size != 4 * surface.n_cells:
        msg = "surface is not a non-empty packed triangle mesh"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "surface contains a non-triangle cell"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def canonical_global_face_hash(
    surface: pv.PolyData, triangle_mask: np.ndarray | None = None
) -> str:
    triangles = triangle_faces(surface)
    if triangle_mask is not None:
        triangles = triangles[triangle_mask]
    ids = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    keys = np.sort(ids[triangles], axis=1).astype("<i8", copy=False)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = np.ascontiguousarray(keys[order])
    digest = hashlib.sha256()
    digest.update(str(keys.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(keys.tobytes())
    return digest.hexdigest()


def validate_metric_roi_intersection(
    metric_skin: pv.PolyData, corrected_skin: pv.PolyData
) -> dict[str, Any]:
    face_mask = np.asarray(metric_skin.cell_data["IsFaceTriangle"], dtype=bool)
    contraction = np.asarray(
        metric_skin.cell_data["ContractionPrestrainMask"], dtype=bool
    )
    if face_mask.shape != (metric_skin.n_cells,) or contraction.shape != (
        metric_skin.n_cells,
    ):
        msg = "pinned metric-surface ROI arrays are malformed"
        raise ValueError(msg)
    if np.any(contraction & ~face_mask):
        msg = "pinned contraction ROI escapes the all-vertex IsFace triangle set"
        raise ValueError(msg)
    metric_hash = canonical_global_face_hash(metric_skin, face_mask)
    corrected_hash = canonical_global_face_hash(corrected_skin)
    require_equal(
        metric_hash,
        corrected_hash,
        "metric IsFace/corrected-skin global triangle intersection",
    )
    return {
        "metric_isface_global_face_sha256": metric_hash,
        "corrected_skin_global_face_sha256": corrected_hash,
        "intersection_exact": True,
        "contraction_triangles": int(contraction.sum()),
        "metadata_policy": (
            "contraction ROI comes only from the hash-pinned historical surface "
            "metadata after exact IsFace topology intersection; none of its "
            "historical material arrays enter corrected mechanics"
        ),
    }


def triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
        msg = "surface contains a non-finite or degenerate triangle"
        raise ValueError(msg)
    return vectors, 0.5 * norms, vectors / norms[:, None]


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def edge_incidence(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.vstack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    owners = np.tile(np.arange(triangles.shape[0], dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1))]
    stops = np.r_[starts[1:], edges.shape[0]]
    counts = stops - starts
    unique = edges[starts]
    lengths = np.linalg.norm(points[unique[:, 1]] - points[unique[:, 0]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        msg = "surface contains an invalid edge length"
        raise ValueError(msg)
    interior = counts == 2
    tri_0 = owners[starts[interior]]
    tri_1 = owners[starts[interior] + 1]
    return unique, counts, tri_0, tri_1, lengths[interior]


def triangle_component_count(
    n_triangles: int, tri_0: np.ndarray, tri_1: np.ndarray
) -> int:
    parent = np.arange(n_triangles, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in zip(tri_0, tri_1, strict=True):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    return len({find(index) for index in range(n_triangles)})


def encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    if encoded.size != 5 * mesh.n_cells:
        msg = "prepared mesh connectivity is not pure tetrahedral"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        msg = "prepared mesh contains a non-tetrahedral cell"
        raise ValueError(msg)
    return encoded[:, 1:].copy()


def six_volume(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,ij->i",
        points[tets[:, 1]] - points[tets[:, 0]],
        np.cross(
            points[tets[:, 2]] - points[tets[:, 0]],
            points[tets[:, 3]] - points[tets[:, 0]],
        ),
    )


def map_global_ids(mesh_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    if np.unique(mesh_ids).size != mesh_ids.size:
        msg = "prepared mesh GlobalPointId values are not unique"
        raise ValueError(msg)
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        msg = "surface GlobalPointId values do not map to the prepared mesh"
        raise ValueError(msg)
    return order[positions]


def canonical_global_triangle_sha256(
    surface: pv.PolyData, triangle_mask: np.ndarray
) -> str:
    """Hash sorted GlobalPointId triples with the formal probe convention."""
    triangles = triangle_faces(surface)
    if triangle_mask.shape != (surface.n_cells,):
        msg = "triangle-selection mask has the wrong shape"
        raise ValueError(msg)
    ids = np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    keys = np.sort(ids[triangles[triangle_mask]], axis=1).astype("<i8", copy=False)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return hashlib.sha256(np.ascontiguousarray(keys[order]).tobytes()).hexdigest()


def validate_cut_boundary_topology(  # noqa: PLR0915
    base: pv.UnstructuredGrid,
    source_skin: pv.PolyData,
    base_global_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive and bind the conservative artificial-cut Dirichlet boundary."""
    triangles = triangle_faces(source_skin)
    source_group_ids = np.asarray(source_skin.point_data["GroupId"], dtype=np.int64)
    if source_group_ids.shape != (source_skin.n_points,):
        msg = "pinned source skin GroupId field is malformed"
        raise ValueError(msg)
    unassigned = source_group_ids == -1
    require_equal(
        int(unassigned.sum()),
        EXPECTED_FULL_UNASSIGNED_GROUP_POINTS,
        "artificial-cut marker point count",
    )
    cut_triangles = np.any(unassigned[triangles], axis=1)
    require_equal(
        int(cut_triangles.sum()),
        EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
        "artificial-cut triangle count",
    )
    cut_topology_sha256 = canonical_global_triangle_sha256(source_skin, cut_triangles)
    require_equal(
        cut_topology_sha256,
        EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256,
        "artificial-cut global triangle topology SHA-256",
    )

    source_global_ids = np.asarray(
        source_skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    source_to_mesh = map_global_ids(base_global_ids, source_global_ids)
    cut_source_ids = np.unique(triangles[cut_triangles])
    cut_mesh_ids = np.sort(source_to_mesh[cut_source_ids]).astype(np.int64, copy=False)
    require_equal(
        int(cut_mesh_ids.size),
        EXPECTED_CUT_INCIDENT_VERTICES,
        "artificial-cut incident vertex count",
    )
    cut_global_ids = np.sort(base_global_ids[cut_mesh_ids]).astype("<i8", copy=False)
    cut_global_ids_sha256 = hashlib.sha256(
        np.ascontiguousarray(cut_global_ids).tobytes()
    ).hexdigest()
    require_equal(
        cut_global_ids_sha256,
        EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256,
        "artificial-cut incident GlobalPointId SHA-256",
    )

    historical_is_fixed = np.asarray(base.point_data["IsFixed"], dtype=bool)
    historical_fixed_mask = np.asarray(base.point_data[FIXED_MASK.vtk], dtype=bool)
    historical_fixed_value = np.asarray(
        base.point_data[FIXED_VALUE.vtk], dtype=np.float64
    )
    if historical_is_fixed.shape != (base.n_points,):
        msg = "prepared IsFixed field is malformed"
        raise ValueError(msg)
    if historical_fixed_mask.shape != (base.n_points, 3):
        msg = "prepared FixedMask field is malformed"
        raise ValueError(msg)
    if historical_fixed_value.shape != (base.n_points, 3):
        msg = "prepared FixedValue field is malformed"
        raise ValueError(msg)
    if not np.array_equal(
        historical_fixed_mask,
        np.repeat(historical_is_fixed[:, None], 3, axis=1),
    ):
        msg = "prepared FixedMask differs from repeated IsFixed"
        raise ValueError(msg)
    if not np.array_equal(
        historical_fixed_value, np.zeros_like(historical_fixed_value)
    ):
        msg = "prepared FixedValue is not exact zero"
        raise ValueError(msg)
    require_equal(
        int(historical_is_fixed.sum()),
        EXPECTED_HISTORICAL_FIXED_VERTICES,
        "historical fixed vertex count",
    )
    require_equal(
        int(historical_fixed_mask.sum()),
        EXPECTED_HISTORICAL_FIXED_DOFS,
        "historical fixed DoF count",
    )
    preexisting = historical_is_fixed[cut_mesh_ids]
    require_equal(
        int(preexisting.sum()),
        EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
        "artificial-cut preexisting fixed vertices",
    )
    require_equal(
        int((~preexisting).sum()),
        EXPECTED_CUT_NEWLY_FIXED_VERTICES,
        "artificial-cut newly fixed vertices",
    )
    is_face = np.asarray(base.point_data["IsFace"], dtype=bool)
    if np.any(is_face[cut_mesh_ids]):
        msg = "artificial-cut incident vertices overlap IsFace"
        raise ValueError(msg)
    corrected_is_fixed = historical_is_fixed.copy()
    corrected_is_fixed[cut_mesh_ids] = True
    require_equal(
        int(corrected_is_fixed.sum()),
        EXPECTED_MODEL_FIXED_VERTICES,
        "hard-fixed model vertex count",
    )
    require_equal(
        3 * int(corrected_is_fixed.sum()),
        EXPECTED_MODEL_FIXED_DOFS,
        "hard-fixed model DoF count",
    )
    provenance = {
        "policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
        "marker": CUT_BOUNDARY_MARKER,
        "triangles": EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
        "triangle_topology_sha256": cut_topology_sha256,
        "incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "incident_global_ids_sha256": cut_global_ids_sha256,
        "preexisting_fixed_vertices": EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
        "newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
        "total_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
        "model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
        "fixed_value_m": [0.0, 0.0, 0.0],
        "hard_fixed_is_ground_truth": False,
        "interpretation": (
            "user-selected conservative approximation; historical and no-skin "
            "comparisons retain their old boundary and are controls only"
        ),
    }
    return (
        cut_mesh_ids,
        cut_global_ids,
        historical_is_fixed.copy(),
        historical_fixed_mask.copy(),
        historical_fixed_value.copy(),
        provenance,
    )


def bounds_camera(
    points: np.ndarray, *, aspect: float = 1.35, padding: float = 1.12
) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    focus = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    scale = 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus, padding * scale


def vertex_normals(
    points: np.ndarray,
    triangles: np.ndarray,
    area_vectors: np.ndarray,
    triangle_mask: np.ndarray,
) -> np.ndarray:
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, triangles[triangle_mask, local], area_vectors[triangle_mask])
    used = np.unique(triangles[triangle_mask])
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms[used] <= np.finfo(np.float64).eps):
        msg = "IsFace target contains a vertex with undefined normal"
        raise ValueError(msg)
    normals[used] /= norms[used, None]
    return normals


def build_surface_basis(  # noqa: C901, PLR0912, PLR0915
    base: pv.UnstructuredGrid, skin: pv.PolyData
) -> SurfaceBasis:
    base_points = np.asarray(base.points, dtype=np.float64).copy()
    base_ids = (
        np.asarray(base.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in base.point_data
        else np.arange(base.n_points, dtype=np.int64)
    )
    (
        cut_mesh_ids,
        cut_global_ids,
        historical_is_fixed,
        historical_fixed_mask,
        historical_fixed_value,
        cut_boundary_provenance,
    ) = validate_cut_boundary_topology(base, skin, base_ids)
    corrected_is_fixed = historical_is_fixed.copy()
    corrected_is_fixed[cut_mesh_ids] = True
    skin_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    skin_mesh_ids = map_global_ids(base_ids, skin_ids)
    skin_points = np.asarray(skin.points, dtype=np.float64).copy()
    if not np.array_equal(skin_points, base_points[skin_mesh_ids]):
        msg = "metric skin points differ from prepared GlobalPointId coordinates"
        raise ValueError(msg)
    triangles = triangle_faces(skin)
    rest_vectors, rest_area, _ = triangle_geometry(skin_points, triangles)
    target = np.nan_to_num(
        np.asarray(base.point_data["Smile"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(base.point_data["SmileLossMask"], dtype=bool)
    target_rms = float(
        np.linalg.norm(target[loss_mask]) / math.sqrt(int(loss_mask.sum()))
    )
    if not math.isfinite(target_rms) or target_rms <= 0.0:
        msg = "prepared Smile target RMS is invalid"
        raise ValueError(msg)
    target_points = skin_points + target[skin_mesh_ids]
    target_vectors, target_area, target_normals = triangle_geometry(
        target_points, triangles
    )
    face_mask = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    if int(face_mask.sum()) != EXPECTED_SKIN_TRIANGLES:
        msg = "metric skin IsFace triangle count changed"
        raise ValueError(msg)
    face_vertex_ids = np.unique(triangles[face_mask])
    face_triangles = triangles[face_mask]
    face_edges = unique_edges(face_triangles)
    (
        face_unique_edges,
        face_edge_counts,
        face_tri_0,
        face_tri_1,
        face_interior_lengths,
    ) = edge_incidence(skin_points, face_triangles)
    if np.any(face_edge_counts > 2):
        msg = "IsFace metric ROI has a nonmanifold edge"
        raise ValueError(msg)
    face_components = triangle_component_count(
        face_triangles.shape[0], face_tri_0, face_tri_1
    )
    if face_components != EXPECTED_SKIN_COMPONENTS:
        msg = (
            "IsFace metric ROI component count changed: "
            f"{face_components} != {EXPECTED_SKIN_COMPONENTS}"
        )
        raise ValueError(msg)
    boundary_edges = face_unique_edges[face_edge_counts == 1]
    if boundary_edges.shape[0] != EXPECTED_SKIN_BOUNDARY_EDGES:
        msg = (
            "IsFace open-membrane boundary changed: "
            f"{boundary_edges.shape[0]} != {EXPECTED_SKIN_BOUNDARY_EDGES}"
        )
        raise ValueError(msg)
    boundary_vertex_ids = np.unique(boundary_edges)
    band_mask = np.isin(face_edges[:, 0], boundary_vertex_ids) | np.isin(
        face_edges[:, 1], boundary_vertex_ids
    )
    boundary_band_ids = np.unique(face_edges[band_mask])
    deep_interior_ids = np.setdiff1d(
        face_vertex_ids, boundary_band_ids, assume_unique=True
    )
    if deep_interior_ids.size == 0:
        msg = "IsFace seam band consumes the complete facial ROI"
        raise ValueError(msg)
    target_vertex_normals = vertex_normals(
        target_points, triangles, target_vectors, face_mask
    )

    contraction = np.asarray(skin.cell_data["ContractionPrestrainMask"], dtype=bool)
    if contraction.shape != (skin.n_cells,):
        msg = "metric skin contraction mask is malformed"
        raise ValueError(msg)
    if np.any(contraction & ~face_mask):
        msg = "contraction ROI escapes the IsFace metric domain"
        raise ValueError(msg)
    face_cell_ids = np.flatnonzero(face_mask)
    tri_0 = face_cell_ids[face_tri_0]
    tri_1 = face_cell_ids[face_tri_1]
    selected = contraction[tri_0] & contraction[tri_1]
    contraction_tri_0 = tri_0[selected]
    contraction_tri_1 = tri_1[selected]
    contraction_weight = face_interior_lengths[selected]
    if contraction_weight.size == 0:
        msg = "contraction ROI contains no interior edge"
        raise ValueError(msg)
    target_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                target_normals[contraction_tri_0],
                target_normals[contraction_tri_1],
            ),
            -1.0,
            1.0,
        )
    )

    face_focus, face_scale = bounds_camera(skin_points[face_vertex_ids])
    lip = np.asarray(base.point_data["IsLip"], dtype=bool)[skin_mesh_ids]
    lip &= np.isin(np.arange(skin.n_points), face_vertex_ids)
    if not np.any(lip):
        msg = "metric IsFace surface has no lip points"
        raise ValueError(msg)
    mouth_focus, mouth_scale = bounds_camera(skin_points[lip], padding=1.25)
    names = tuple(str(value) for value in skin.field_data["GroupName"])
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {
        "EyelidTop",
        "EyelidBottom",
        "EyelidOuterTop",
        "EyelidOuterBottom",
    }
    eyelid_group_ids = [
        index for index, name in enumerate(names) if name in eyelid_names
    ]
    eyelid = np.isin(group_ids, eyelid_group_ids)
    eyelid &= np.isin(np.arange(skin.n_points), face_vertex_ids)
    if not np.any(eyelid):
        msg = "metric IsFace surface has no eyelid-group points"
        raise ValueError(msg)
    one_eye = eyelid & (skin_points[:, 0] >= np.median(skin_points[eyelid, 0]))
    eye_focus, _ = bounds_camera(skin_points[one_eye])
    face_height = float(np.ptp(skin_points[face_vertex_ids, 1]))
    eye_focus = eye_focus.copy()
    eye_focus[1] -= 0.08 * face_height
    eye_scale = 0.24 * face_height

    tets = encoded_tetrahedra(base)
    rest_six_volume = six_volume(base_points, tets)
    if np.any(np.abs(rest_six_volume) <= np.finfo(np.float64).eps):
        msg = "prepared mesh contains a zero-volume tetrahedron"
        raise ValueError(msg)
    activation_mask = np.asarray(base.cell_data["ActivationMask"], dtype=bool)
    if activation_mask.shape != (base.n_cells,) or not np.any(activation_mask):
        msg = "prepared ActivationMask is malformed or empty"
        raise ValueError(msg)
    return SurfaceBasis(
        base_points=base_points,
        base_cells=np.asarray(base.cells).copy(),
        base_celltypes=np.asarray(base.celltypes).copy(),
        base_global_ids=base_ids,
        historical_is_fixed=historical_is_fixed,
        historical_fixed_mask=historical_fixed_mask,
        historical_fixed_value=historical_fixed_value,
        corrected_is_fixed=corrected_is_fixed,
        cut_mesh_ids=cut_mesh_ids,
        cut_global_ids=cut_global_ids,
        cut_boundary_provenance=cut_boundary_provenance,
        tets=tets,
        rest_six_volume=rest_six_volume,
        target=target,
        loss_mask=loss_mask,
        target_rms=target_rms,
        activation_mask=activation_mask,
        legacy_edges=surface_edges_for_mask(base, loss_mask),
        skin=skin,
        skin_points=skin_points,
        skin_mesh_ids=skin_mesh_ids,
        triangles=triangles,
        rest_area=rest_area,
        target_area=target_area,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
        target_vertex_normals=target_vertex_normals,
        face_triangle_mask=face_mask,
        face_vertex_ids=face_vertex_ids,
        face_edges=face_edges,
        boundary_edges=boundary_edges,
        boundary_vertex_ids=boundary_vertex_ids,
        boundary_band_ids=boundary_band_ids,
        deep_interior_ids=deep_interior_ids,
        contraction_tri_0=contraction_tri_0,
        contraction_tri_1=contraction_tri_1,
        contraction_target_dihedral=target_dihedral,
        contraction_edge_weight=contraction_weight,
        face_focus=face_focus,
        face_scale=face_scale,
        mouth_focus=mouth_focus,
        mouth_scale=mouth_scale,
        eye_cheek_focus=eye_focus,
        eye_cheek_scale=eye_scale,
    )


def validate_case_summary(
    *,
    label: str,
    candidate: str,
    summary: dict[str, Any],
    recorded_trace: list[dict[str, Any]],
    summary_path: Path,
    history_path: Path,
    trace_path: Path,
    corrected: bool,
) -> list[dict[str, Any]]:
    require_equal(summary.get("candidate"), candidate, f"{label} candidate")
    require_equal(summary.get("stage"), "screen", f"{label} stage")
    require_equal(summary.get("status"), "ok", f"{label} status")
    require_equal(summary.get("validation/errors"), [], f"{label} validation")
    require_equal(
        summary.get("baseline/completed"), expected=True, context=f"{label} budget"
    )
    for key in (
        "inverse/evaluations",
        "baseline/evaluations",
        "baseline/evaluations_expected",
        "history/frames",
        "history_frames",
    ):
        require_equal(summary.get(key), EXPECTED_EVALUATIONS, f"{label} {key}")
    require_equal(
        summary.get("baseline/mandatory_optimizer_steps"),
        TERMINAL_STEP,
        f"{label} optimizer steps",
    )
    require_close(summary.get("baseline/fixed_lr"), EXPECTED_LR, f"{label} LR")
    require_equal(
        summary.get("baseline/lr_deviation_count"), 0, f"{label} LR deviations"
    )
    require_equal(
        summary.get("activation/mode"),
        "per-muscle-tet-6dof",
        f"{label} activation mode",
    )
    require_equal(summary.get("activation_inv/initial_rms"), 0.0, f"{label} init")
    require_equal(
        summary.get("activation_inv/initial_max_abs"), 0.0, f"{label} init max"
    )
    require_equal(
        summary.get("initial_displacement/enabled"),
        expected=False,
        context=f"{label} initial displacement",
    )
    require_equal(
        Path(str(summary.get("artifact/summary_path"))).resolve(),
        summary_path.resolve(),
        f"{label} summary path",
    )
    require_equal(
        Path(str(summary.get("history/path"))).name,
        history_path.name,
        f"{label} history basename",
    )
    require_equal(
        Path(str(summary.get("trace/path"))).name,
        trace_path.name,
        f"{label} trace basename",
    )
    require_equal(
        (
            summary.get("candidate/young_min_scale"),
            summary.get("candidate/prestrain_gain"),
        ),
        EXPECTED_PARAMETERS[label],
        f"{label} material parameters",
    )
    require_equal(
        summary.get("skin/enabled"), label != NO_SKIN_LABEL, f"{label} skin enabled"
    )
    if corrected:
        exact = {
            "design": DESIGN,
            "material/skin_domain": "all-vertex IsFace filtered PolyData",
            "material/skin_lame_conversion": LAME_CONVERSION,
            "material/skin_koiter_energy_measure": KOITER_ENERGY_MEASURE,
            "material/volume_lame_conversion": VOLUME_LAME_CONVERSION,
            "material/skin_E_MPa": 0.2,
            "material/skin_nu": 0.49,
            "material/skin_prestrain": "p000",
            "protocol/fresh_zero_activation": True,
            "protocol/fresh_zero_displacement": True,
            "protocol/forward_initial_displacement_exact_zero": True,
            "protocol/forward_initial_displacement_max_abs_m": 0.0,
            "protocol/optimizer_steps": 40,
            "protocol/evaluations": 41,
            "skin/domain": "all-vertex IsFace filtered PolyData",
            "skin/koiter_input_triangles": EXPECTED_SKIN_TRIANGLES,
            "skin/koiter_input_points": EXPECTED_SKIN_POINTS,
            "skin/E_MPa": 0.2,
            "skin/nu": 0.49,
            "skin/prestrain": "p000",
            "skin/lame_conversion": LAME_CONVERSION,
            "skin/koiter_energy_measure": KOITER_ENERGY_MEASURE,
            "volume/lame_conversion": VOLUME_LAME_CONVERSION,
            "cut_boundary/policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
            "cut_boundary/marker": CUT_BOUNDARY_MARKER,
            "cut_boundary/reference_path": str(SOURCE_SKIN),
            "cut_boundary/reference_size_bytes": SOURCE_SKIN_SIZE_BYTES,
            "cut_boundary/reference_sha256": SOURCE_SKIN_SHA256,
            "cut_boundary/triangles": EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
            "cut_boundary/triangle_topology_sha256": (
                EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256
            ),
            "cut_boundary/incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "cut_boundary/incident_global_ids_sha256": (
                EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
            ),
            "cut_boundary/preexisting_fixed_vertices": (
                EXPECTED_CUT_PREEXISTING_FIXED_VERTICES
            ),
            "cut_boundary/newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
            "cut_boundary/total_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "cut_boundary/hard_fixed_is_ground_truth": False,
            "cut_boundary/model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
            "cut_boundary/model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
            "cut_boundary/fixed_values_max_abs_m": 0.0,
            "cut_boundary/configured_exact_zero": True,
            "cut_boundary/readback_incident_vertices": (EXPECTED_CUT_INCIDENT_VERTICES),
            "cut_boundary/readback_incident_global_ids_sha256": (
                EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
            ),
            "cut_boundary/readback_total_fixed_vertices": (
                EXPECTED_CUT_INCIDENT_VERTICES
            ),
            "cut_boundary/readback_model_total_fixed_vertices": (
                EXPECTED_MODEL_FIXED_VERTICES
            ),
            "cut_boundary/readback_model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
            "cut_boundary/readback_fixed_values_max_abs_m": 0.0,
            "cut_boundary/readback_displacement_rms_m": 0.0,
            "cut_boundary/readback_displacement_max_abs_m": 0.0,
            "cut_boundary/readback_exact_zero": True,
        }
        for key, expected in exact.items():
            require_equal(summary.get(key), expected, f"{label} {key}")
        for key, expected in (
            (
                "archive/canonical_case_summary_path",
                CORRECTED_CASE_SUMMARY_ARCHIVE,
            ),
            ("archive/canonical_aggregate_path", CORRECTED_AGGREGATE_ARCHIVE),
        ):
            require_equal(
                Path(str(summary.get(key, ""))).resolve(),
                expected.resolve(),
                f"{label} {key}",
            )
        require_equal(
            summary.get("archive/metadata_snapshot_policy"),
            ARCHIVE_METADATA_SNAPSHOT_POLICY,
            f"{label} archive metadata snapshot policy",
        )
        require_equal(
            summary.get("comparison/numerically_eligible_pending_visual_review"),
            expected=True,
            context=f"{label} numerical eligibility",
        )
    trace = summary.get("trace")
    if not isinstance(trace, list) or not all(isinstance(row, dict) for row in trace):
        msg = f"{label} embedded trace must be a list of objects"
        raise TypeError(msg)
    require_equal(len(trace), EXPECTED_EVALUATIONS, f"{label} trace length")
    finite_keys = (
        "loss/total",
        "loss/mm2",
        "loss/m2",
        "target/error_rms",
        "target/error_rms_mm",
        "activation_inv/rms",
        "activation_inv/max_abs",
        "grad/norm",
        "forward/relative_grad_norm",
        "adjoint/relative_residual",
    )
    for step, row in enumerate(trace):
        require_equal(int(row.get("step", -1)), step, f"{label} trace step")
        require_equal(
            row.get("forward/success"),
            expected=True,
            context=f"{label} forward {step}",
        )
        require_equal(
            row.get("adjoint/success"),
            expected=True,
            context=f"{label} adjoint {step}",
        )
        require_close(row.get("inverse/lr"), EXPECTED_LR, f"{label} LR {step}")
        for key in finite_keys:
            if not math.isfinite(float(row.get(key, math.nan))):
                msg = f"{label} step {step} has non-finite {key}"
                raise ValueError(msg)
        expected_loss_mm2 = float(row["target/error_rms_mm"]) ** 2 / 3.0
        expected_loss_m2 = float(row["target/error_rms"]) ** 2 / 3.0
        require_close(
            row["loss/mm2"], expected_loss_mm2, f"{label} loss/mm2 identity {step}"
        )
        require_close(
            row["loss/m2"], expected_loss_m2, f"{label} loss/m2 identity {step}"
        )
    require_equal(trace[0]["activation_inv/rms"], 0.0, f"{label} step-0 RMS")
    require_equal(trace[0]["activation_inv/max_abs"], 0.0, f"{label} step-0 max")
    compare_trace_rows(label, list(trace), recorded_trace)
    return list(trace)


def corrected_paths() -> dict[str, Path]:
    return {
        "summary": CORRECTED_DATA_DIR / f"{CORRECTED_STEM}-summary.json",
        "trace": CORRECTED_DATA_DIR / f"{CORRECTED_STEM}-trace.jsonl",
        "history": CORRECTED_DATA_DIR / f"{CORRECTED_STEM}-steps.vtkhdf",
        "result": CORRECTED_DATA_DIR / f"{CORRECTED_STEM}.vtu",
        "target": CORRECTED_DATA_DIR / f"{CORRECTED_STEM}-target.vtu",
    }


def historical_paths(spec: HistoricalSpec) -> dict[str, Path]:
    return {
        "summary": OLD_DATA_DIR / f"{spec.stem}-summary.json",
        "trace": OLD_DATA_DIR / f"{spec.stem}-trace.jsonl",
        "history": OLD_DATA_DIR / f"{spec.stem}-steps.vtkhdf",
        "result": OLD_DATA_DIR / f"{spec.stem}.vtu",
        "target": OLD_DATA_DIR / f"{spec.stem}-target.vtu",
    }


def validate_manifest(  # noqa: C901, PLR0912, PLR0915
    cfg: Config, base: pv.UnstructuredGrid
) -> tuple[dict[str, Any], dict[str, Any], pv.PolyData, dict[str, Any]]:
    manifest = read_json(cfg.input_manifest)
    expected_keys = {
        "schema_version",
        "complete",
        "design",
        "experiment",
        "purpose",
        "input_mesh",
        "input_mesh_identity",
        "inputs",
        "fixed_design",
        "domain_contract",
        "constitutive_contract",
        "n_candidates",
        "candidate_validation_errors",
        "validation_errors",
        "candidates",
    }
    require_equal(set(manifest), expected_keys, "manifest top-level schema")
    for key, expected in (
        ("schema_version", MANIFEST_SCHEMA_VERSION),
        ("complete", True),
        ("design", MANIFEST_DESIGN),
        ("n_candidates", 1),
        ("candidate_validation_errors", {}),
        ("validation_errors", []),
    ):
        require_equal(manifest[key], expected, f"manifest {key}")
    require_equal(
        Path(str(manifest["input_mesh"])).resolve(),
        cfg.input_mesh.resolve(),
        "manifest input mesh path",
    )
    mesh_identity = file_identity(cfg.input_mesh)
    require_equal(
        manifest["input_mesh_identity"], mesh_identity, "manifest mesh identity"
    )
    fixed = manifest["fixed_design"]
    exact_fixed = {
        "candidate_labels": [CORRECTED_LABEL],
        "skin_domain": "all-vertex IsFace physically filtered PolyData",
        "skin_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin_components": EXPECTED_SKIN_COMPONENTS,
        "skin_E_MPa": 0.2,
        "skin_nu": 0.49,
        "skin_prestrain": "p000",
        "skin_lame_conversion": LAME_CONVERSION,
        "volume_lame_conversion": VOLUME_LAME_CONVERSION,
        "inverse_activation_initialization": "fresh exact zero",
        "inverse_optimizer": "Adam",
        "inverse_lr": EXPECTED_LR,
        "inverse_optimizer_steps": TERMINAL_STEP,
        "inverse_evaluations": EXPECTED_EVALUATIONS,
    }
    for key, expected in exact_fixed.items():
        require_equal(fixed.get(key), expected, f"manifest fixed_design {key}")
    require_close(
        fixed.get("skin_area_m2"), EXPECTED_SKIN_AREA_M2, "manifest skin area"
    )
    domain = manifest["domain_contract"]
    exact_domain = {
        "full_boundary_triangles": EXPECTED_FULL_BOUNDARY_TRIANGLES,
        "source_outer_triangles": 115_007,
        "artificial_cut_triangles": 13_165,
        "skin_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin_artificial_cut_overlap_triangles": 0,
        "skin_fixed_overlap_triangles": 0,
        "skin_disallowed_group_overlap_triangles": 0,
        "skin_nonfinite_target_triangles": 0,
        "validation_errors": [],
        "validation_ok": True,
        "selection": "all three triangle vertices have IsFace=true",
        "face_group_allowlist": list(FACE_GROUPS),
    }
    for key, expected in exact_domain.items():
        require_equal(domain.get(key), expected, f"manifest domain {key}")
    require_equal(
        set(domain.get("observed_skin_group_names", [])),
        set(FACE_GROUPS),
        "manifest observed face groups",
    )
    require_equal(
        domain.get("isface_global_face_key_sha256"),
        domain.get("topology_reference_global_face_key_sha256"),
        "manifest IsFace/topology-reference identity",
    )
    constitutive = manifest["constitutive_contract"]
    for key, expected in (
        ("skin", LAME_CONVERSION),
        ("volume", VOLUME_LAME_CONVERSION),
        ("skin/E_MPa", 0.2),
        ("skin/nu", 0.49),
        ("skin/thickness_m", 0.001),
        ("skin/prestrain", "none; ActivationInv is exactly zero"),
        ("heterogeneous_material_fields", False),
    ):
        require_equal(constitutive.get(key), expected, f"constitutive {key}")
    candidates = manifest["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 1:
        msg = "manifest must contain exactly one corrected candidate"
        raise ValueError(msg)
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        msg = "manifest candidate must be an object"
        raise TypeError(msg)
    for key, expected in (
        ("schema_version", MANIFEST_SCHEMA_VERSION),
        ("label", CORRECTED_LABEL),
        ("young_min_scale", 1.0),
        ("prestrain_gain", 0.0),
        ("skin/nu", 0.49),
        ("skin/thickness_m", 0.001),
        ("skin/lame_conversion", LAME_CONVERSION),
        ("skin/domain", "all-vertex IsFace filtered PolyData"),
        ("content/n_points", EXPECTED_SKIN_POINTS),
        ("content/n_triangles", EXPECTED_SKIN_TRIANGLES),
        ("topology/components", EXPECTED_SKIN_COMPONENTS),
        ("validation/errors", []),
        ("validation/ok", True),
        ("readback/errors", []),
        ("readback/ok", True),
    ):
        require_equal(candidate.get(key), expected, f"manifest candidate {key}")
    require_close(
        candidate.get("content/area_m2"),
        EXPECTED_SKIN_AREA_M2,
        "manifest candidate area",
    )
    skin_path = (cfg.input_manifest.parent / str(candidate["skin/path"])).resolve()
    expected_skin = (
        CORRECTED_DATA_DIR / "10-corrected-baseline/skin-isface-e0200-p000.vtp"
    )
    require_equal(skin_path, expected_skin.resolve(), "corrected skin path")
    skin_identity = file_identity(skin_path)
    require_equal(
        candidate["skin/file_identity"], skin_identity, "corrected skin identity"
    )
    skin = pv.read(skin_path)
    if not isinstance(skin, pv.PolyData):
        msg = "corrected skin is not PolyData"
        raise TypeError(msg)
    triangles = triangle_faces(skin)
    require_equal(skin.n_points, EXPECTED_SKIN_POINTS, "corrected skin points")
    require_equal(skin.n_cells, EXPECTED_SKIN_TRIANGLES, "corrected skin triangles")
    ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    base_ids = (
        np.asarray(base.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in base.point_data
        else np.arange(base.n_points, dtype=np.int64)
    )
    mapped = map_global_ids(base_ids, ids)
    if not np.array_equal(
        np.asarray(skin.points, dtype=np.float64),
        np.asarray(base.points, dtype=np.float64)[mapped],
    ):
        msg = "corrected skin points differ from prepared GlobalPointId"
        raise ValueError(msg)
    E = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    expected_lambda = E * nu / (1.0 - np.square(nu))
    expected_mu = E / (2.0 * (1.0 + nu))
    formulas = {
        "SkinYoungModulusMPa": (E, np.full(skin.n_cells, 0.2)),
        "SkinPoissonRatio": (nu, np.full(skin.n_cells, 0.49)),
        LAMBDA.vtk: (np.asarray(skin.cell_data[LAMBDA.vtk]), expected_lambda),
        MU.vtk: (np.asarray(skin.cell_data[MU.vtk]), expected_mu),
        FRACTION.vtk: (np.asarray(skin.cell_data[FRACTION.vtk]), np.ones(skin.n_cells)),
    }
    for name, (actual, expected) in formulas.items():
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
        ):
            msg = f"corrected skin {name} violates the homogeneous plane-stress rule"
            raise ValueError(msg)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    if activation.shape != (skin.n_cells, 3) or not np.array_equal(
        activation, np.zeros_like(activation)
    ):
        msg = "corrected skin prestrain is not exact p000"
        raise ValueError(msg)
    for name in ("IsFaceTriangle", "SourceOuterTriangle"):
        if not np.all(np.asarray(skin.cell_data[name], dtype=np.int8) == 1):
            msg = f"corrected skin {name} is not one everywhere"
            raise ValueError(msg)
    for name in ("ArtificialCutTriangle", "FixedTriangle", "DisallowedGroupTriangle"):
        if np.any(np.asarray(skin.cell_data[name], dtype=np.int8) != 0):
            msg = f"corrected skin overlaps {name}"
            raise ValueError(msg)
    area = triangle_geometry(np.asarray(skin.points), triangles)[1]
    require_close(float(math.fsum(area)), EXPECTED_SKIN_AREA_M2, "live skin area")
    content = {
        "topology_sha256": skin_topology_content_hash(skin),
        "material_sha256": skin_material_content_hash(skin),
        "solver_sha256": skin_solver_content_hash(skin),
    }
    for name, digest in content.items():
        require_equal(candidate[f"content/{name}"], digest, f"corrected skin {name}")
        require_equal(
            candidate[f"readback/content/{name}"],
            digest,
            f"corrected skin readback {name}",
        )
    return (
        manifest,
        candidate,
        skin,
        {
            "manifest": file_identity(cfg.input_manifest),
            "skin": skin_identity,
            "mesh": mesh_identity,
            "skin_content": content,
        },
    )


def validate_aggregate(
    cfg: Config, manifest_identity: dict[str, int | str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = read_json(cfg.input_corrected_summary)
    expected_keys = {
        "schema_version",
        "complete",
        "design",
        "stage",
        "candidate_set",
        "input_mesh",
        "input_candidates",
        "input_candidates_identity",
        "fresh_zero_activation",
        "fresh_zero_displacement",
        "activation_mode",
        "activation_shared",
        "activation_transferred_between_candidates",
        "forward_builder_shared_between_candidates",
        "inverse_lr",
        "inverse_max_steps",
        "plot_backend",
        "acceptance_policy",
        "hard_failures",
        "cases",
        "experiment",
        "n_candidates",
        "activation_transferred",
        "inverse_optimizer_steps",
        "inverse_evaluations",
        "input_cut_reference",
        "input_cut_reference_identity",
        "archive_policy",
        "constitutive_policy",
        "domain_policy",
        "boundary_policy",
        "implementation",
        "visual_review",
        "execution_scope",
    }
    require_equal(set(aggregate), expected_keys, "corrected aggregate schema")
    exact = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "complete": True,
        "design": DESIGN,
        "stage": "screen",
        "candidate_set": CORRECTED_LABEL,
        "input_candidates_identity": manifest_identity,
        "fresh_zero_activation": True,
        "fresh_zero_displacement": True,
        "activation_mode": "per-muscle-tet-6dof-unconstrained",
        "activation_shared": False,
        "activation_transferred_between_candidates": False,
        "forward_builder_shared_between_candidates": False,
        "inverse_lr": EXPECTED_LR,
        "inverse_max_steps": TERMINAL_STEP,
        "hard_failures": [],
        "n_candidates": 1,
        "activation_transferred": False,
        "inverse_optimizer_steps": TERMINAL_STEP,
        "inverse_evaluations": EXPECTED_EVALUATIONS,
        "execution_scope": "single 40-update scientific corrected baseline",
        "input_cut_reference": str(SOURCE_SKIN),
        "input_cut_reference_identity": {
            "size_bytes": SOURCE_SKIN_SIZE_BYTES,
            "sha256": SOURCE_SKIN_SHA256,
        },
    }
    for key, expected in exact.items():
        require_equal(aggregate[key], expected, f"corrected aggregate {key}")
    require_equal(
        Path(str(aggregate["input_mesh"])).resolve(),
        cfg.input_mesh.resolve(),
        "corrected aggregate mesh path",
    )
    require_equal(
        Path(str(aggregate["input_candidates"])).resolve(),
        cfg.input_manifest.resolve(),
        "corrected aggregate manifest path",
    )
    archive_policy = aggregate["archive_policy"]
    require_equal(
        set(archive_policy),
        {
            "canonical_case_summary_path",
            "canonical_aggregate_path",
            "canonical_copies_are_byte_identical_to_live",
            "reason",
        },
        "corrected aggregate archive policy schema",
    )
    for key, expected in (
        ("canonical_case_summary_path", CORRECTED_CASE_SUMMARY_ARCHIVE),
        ("canonical_aggregate_path", CORRECTED_AGGREGATE_ARCHIVE),
    ):
        require_equal(
            Path(str(archive_policy[key])).resolve(),
            expected.resolve(),
            f"corrected aggregate archive policy {key}",
        )
    require_equal(
        archive_policy["canonical_copies_are_byte_identical_to_live"],
        expected=True,
        context="corrected aggregate archive byte-identity claim",
    )
    require_equal(
        archive_policy["reason"],
        ARCHIVE_METADATA_SNAPSHOT_POLICY,
        "corrected aggregate archive reason",
    )
    require_byte_identical_archive(
        cfg.input_corrected_summary,
        CORRECTED_AGGREGATE_ARCHIVE,
        context="corrected aggregate canonical archive",
    )
    domain = aggregate["domain_policy"]
    exact_domain = {
        "skin": "all-vertex IsFace filtered PolyData",
        "koiter_input_triangles": EXPECTED_SKIN_TRIANGLES,
        "koiter_input_area_m2": EXPECTED_SKIN_AREA_M2,
        "components": EXPECTED_SKIN_COMPONENTS,
        "artificial_cut_overlap_triangles": 0,
        "fixed_overlap_triangles": 0,
        "disallowed_group_overlap_triangles": 0,
        "face_group_allowlist": list(FACE_GROUPS),
        "teeth_and_gingiva_proximity": "diagnostic only",
    }
    require_equal(domain, exact_domain, "corrected aggregate domain policy")
    constitutive = aggregate["constitutive_policy"]
    require_equal(
        constitutive,
        {
            "skin": LAME_CONVERSION,
            "volume": VOLUME_LAME_CONVERSION,
            "skin_E_MPa": 0.2,
            "skin_nu": 0.49,
            "skin_prestrain": "p000",
            "skin_koiter_energy_measure": KOITER_ENERGY_MEASURE,
        },
        "corrected aggregate constitutive policy",
    )
    require_equal(
        aggregate["boundary_policy"],
        {
            "policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
            "marker": CUT_BOUNDARY_MARKER,
            "reference_path": str(SOURCE_SKIN),
            "reference_size_bytes": SOURCE_SKIN_SIZE_BYTES,
            "reference_sha256": SOURCE_SKIN_SHA256,
            "triangles": EXPECTED_ARTIFICIAL_CUT_TRIANGLES,
            "triangle_topology_sha256": EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256,
            "incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "incident_global_ids_sha256": (EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256),
            "preexisting_fixed_vertices": EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
            "newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
            "total_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
            "model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
            "fixed_values_max_abs_m": 0.0,
            "configured_exact_zero": True,
            "readback_displacement_max_abs_m": 0.0,
            "readback_exact_zero": True,
            "hard_fixed_is_ground_truth": False,
            "interpretation": "user-approved conservative approximation",
        },
        "corrected aggregate boundary policy",
    )
    require_equal(
        aggregate["visual_review"].get("status"),
        "pending",
        "corrected aggregate visual status",
    )
    implementation = aggregate["implementation"]
    expected_implementation_keys = {
        "producer/path",
        "producer/size_bytes",
        "producer/sha256",
        "prepare/path",
        "prepare/sha256",
        "prepare/identity_verified_stable",
        "reference_inverse/path",
        "reference_inverse/sha256",
        "reference_inverse/identity_verified_stable",
        "intentional_changes",
        "koiter/path",
        "koiter/sha256",
        "volume_lame/path",
        "volume_lame/sha256",
        "volume_forward/path",
        "volume_forward/sha256",
        "inverse_runtime_bundle",
    }
    require_equal(
        set(implementation),
        expected_implementation_keys,
        "corrected aggregate implementation schema",
    )
    required_implementation = {
        "producer/path": ANALYZED_INVERSE,
        "producer/size_bytes": ANALYZED_INVERSE_SIZE_BYTES,
        "producer/sha256": ANALYZED_INVERSE_SHA256,
        "prepare/path": PREPARE_IMPLEMENTATION,
        "prepare/sha256": PREPARE_IMPLEMENTATION_SHA256,
        "reference_inverse/path": LEGACY_INVERSE,
        "reference_inverse/sha256": LEGACY_INVERSE_SHA256,
        "koiter/path": KOITER_IMPLEMENTATION,
        "koiter/sha256": KOITER_IMPLEMENTATION_SHA256,
        "volume_lame/path": VOLUME_LAME_IMPLEMENTATION,
        "volume_lame/sha256": VOLUME_LAME_IMPLEMENTATION_SHA256,
        "volume_forward/path": VOLUME_FORWARD_IMPLEMENTATION,
        "volume_forward/sha256": VOLUME_FORWARD_IMPLEMENTATION_SHA256,
    }
    for key, expected_value in required_implementation.items():
        actual = implementation.get(key)
        expected_normalized = expected_value
        if key.endswith("/path"):
            actual = Path(str(actual)).resolve()
            expected_normalized = Path(expected_value).resolve()
        require_equal(
            actual, expected_normalized, f"corrected aggregate implementation {key}"
        )
    require_equal(
        implementation.get("prepare/identity_verified_stable"),
        expected=True,
        context="corrected aggregate prepare stability",
    )
    require_equal(
        implementation.get("reference_inverse/identity_verified_stable"),
        expected=True,
        context="corrected aggregate reference-inverse stability",
    )
    require_equal(
        implementation.get("intentional_changes"),
        [
            "one corrected homogeneous p000 candidate only",
            "physically filtered 29,899-triangle IsFace Koiter input",
            "plane-stress skin lambda with unchanged skin mu",
            "fixed original-reference-area Koiter energy weight",
            "unchanged 3D volume material conversion",
            "all 6,980 artificial-cut incident vertices fixed to exact zero",
        ],
        "corrected aggregate intentional changes",
    )
    validate_inverse_runtime_bundle(implementation.get("inverse_runtime_bundle"))
    for path, expected_sha256, name in (
        (PREPARE_IMPLEMENTATION, PREPARE_IMPLEMENTATION_SHA256, "prepare"),
        (LEGACY_INVERSE, LEGACY_INVERSE_SHA256, "reference inverse"),
        (KOITER_IMPLEMENTATION, KOITER_IMPLEMENTATION_SHA256, "Koiter"),
        (VOLUME_LAME_IMPLEMENTATION, VOLUME_LAME_IMPLEMENTATION_SHA256, "volume Lame"),
        (
            VOLUME_FORWARD_IMPLEMENTATION,
            VOLUME_FORWARD_IMPLEMENTATION_SHA256,
            "volume forward",
        ),
    ):
        require_equal(sha256_file(path), expected_sha256, f"live {name} SHA-256")
    cases = aggregate["cases"]
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        msg = "corrected aggregate must contain exactly one case object"
        raise ValueError(msg)
    return aggregate, cases[0]


def load_corrected_case(
    aggregate_row: dict[str, Any],
    candidate: dict[str, Any],
) -> CaseInput:
    paths = corrected_paths()
    require_equal(
        aggregate_row.get("case"), CORRECTED_STEM, "corrected aggregate case stem"
    )
    require_equal(
        Path(str(aggregate_row.get("artifact/summary_path"))).resolve(),
        paths["summary"].resolve(),
        "corrected aggregate summary path",
    )
    summary = read_json(paths["summary"])
    require_equal(summary, aggregate_row, "corrected aggregate/sidecar equality")
    identities: dict[str, dict[str, int | str | bool]] = {
        "summary": {
            **file_identity(paths["summary"]),
            "aggregate_row_exact_match": True,
        }
    }
    identities["summary_archive"] = require_byte_identical_archive(
        paths["summary"],
        CORRECTED_CASE_SUMMARY_ARCHIVE,
        context="corrected case-summary canonical archive",
    )
    for name in ("trace", "history", "result", "target"):
        identities[name] = require_recorded_artifact(
            summary, name=name, path=paths[name]
        )
    trace_file = safe_sibling(
        CORRECTED_DATA_DIR,
        summary.get("trace/path"),
        context="corrected trace path",
    )
    history_file = safe_sibling(
        CORRECTED_DATA_DIR,
        summary.get("history/path"),
        context="corrected history path",
    )
    require_equal(trace_file, paths["trace"].resolve(), "corrected trace filename")
    require_equal(
        history_file, paths["history"].resolve(), "corrected history filename"
    )
    recorded_trace = read_jsonl(paths["trace"])
    trace = validate_case_summary(
        label=CORRECTED_LABEL,
        candidate=CORRECTED_LABEL,
        summary=summary,
        recorded_trace=recorded_trace,
        summary_path=paths["summary"],
        history_path=paths["history"],
        trace_path=paths["trace"],
        corrected=True,
    )
    require_equal(
        summary.get("provenance/skin_file_sha256"),
        candidate["skin/file_identity"]["sha256"],
        "corrected summary skin identity",
    )
    # The case sidecar records the mesh path, while the manifest and live input
    # gates above bind its exact size and SHA-256. Do not require invented
    # duplicate provenance keys that the formal producer does not emit.
    require_equal(
        Path(str(summary.get("input_mesh", ""))).resolve(),
        PREPARED_MESH.resolve(),
        "corrected summary input mesh path",
    )
    require_equal(
        summary.get("skin/file_identity"),
        candidate["skin/file_identity"],
        "corrected summary skin file identity",
    )
    for key, expected in (
        (
            "provenance/skin_size_bytes",
            candidate["skin/file_identity"]["size_bytes"],
        ),
        (
            "provenance/skin_topology_sha256",
            candidate["content/topology_sha256"],
        ),
        (
            "provenance/skin_material_sha256",
            candidate["content/material_sha256"],
        ),
        (
            "provenance/skin_solver_sha256",
            candidate["content/solver_sha256"],
        ),
    ):
        require_equal(summary.get(key), expected, f"corrected summary {key}")
    return CaseInput(
        label=CORRECTED_LABEL,
        candidate=CORRECTED_LABEL,
        cohort="primary-corrected",
        material_model=(
            "IsFace-only homogeneous E=.2 p000 plane-stress Koiter; fixed "
            "reference area; artificial cut hard-fixed"
        ),
        summary_path=paths["summary"],
        trace_path=paths["trace"],
        history_path=paths["history"],
        result_path=paths["result"],
        target_path=paths["target"],
        summary=summary,
        trace=trace,
        identities=identities,
        history=TemporalHistory.open(CORRECTED_LABEL, paths["history"]),
    )


def load_historical_case(spec: HistoricalSpec) -> CaseInput:
    paths = historical_paths(spec)
    identities = {
        "summary": require_identity(paths["summary"], spec.summary, context=spec.label),
        "trace": require_identity(paths["trace"], spec.trace, context=spec.label),
        "history": require_identity(paths["history"], spec.history, context=spec.label),
        "result": require_identity(paths["result"], spec.result, context=spec.label),
        "target": require_identity(paths["target"], spec.target, context=spec.label),
    }
    summary = read_json(paths["summary"])
    recorded_trace = read_jsonl(paths["trace"])
    trace = validate_case_summary(
        label=spec.label,
        candidate=spec.candidate,
        summary=summary,
        recorded_trace=recorded_trace,
        summary_path=paths["summary"],
        history_path=paths["history"],
        trace_path=paths["trace"],
        corrected=False,
    )
    return CaseInput(
        label=spec.label,
        candidate=spec.candidate,
        cohort=(
            "old-boundary-external-no-skin-control"
            if spec.label == NO_SKIN_LABEL
            else "old-boundary-secondary-historical-diagnostic"
        ),
        material_model=(
            "no Koiter skin; historical IsFixed boundary"
            if spec.label == NO_SKIN_LABEL
            else (
                "full extracted boundary, historical 3D lambda in Koiter; "
                "historical IsFixed boundary"
            )
        ),
        summary_path=paths["summary"],
        trace_path=paths["trace"],
        history_path=paths["history"],
        result_path=paths["result"],
        target_path=paths["target"],
        summary=summary,
        trace=trace,
        identities=identities,
        history=TemporalHistory.open(spec.label, paths["history"]),
    )


def field_scalar(frame: pv.UnstructuredGrid, name: str) -> float:
    if name not in frame.field_data:
        msg = f"history frame is missing field_data[{name!r}]"
        raise KeyError(msg)
    values = np.asarray(frame.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = f"history field {name!r} is not one finite scalar"
        raise ValueError(msg)
    return float(values[0])


def validate_corrected_cut_marker_fields(
    mesh: pv.UnstructuredGrid,
    basis: SurfaceBasis,
    *,
    context: str,
) -> None:
    required = {
        "HistoricalIsFixed",
        "ArtificialCutIncident",
        "CutBoundaryPreexistingFixed",
        "CutBoundaryAddedFixed",
    }
    missing = sorted(required - set(mesh.point_data))
    if missing:
        msg = f"{context} is missing hard-fixed cut fields: {missing}"
        raise KeyError(msg)
    incident = np.zeros(mesh.n_points, dtype=bool)
    incident[basis.cut_mesh_ids] = True
    preexisting = incident & basis.historical_is_fixed
    added = incident & ~basis.historical_is_fixed
    expected = {
        "HistoricalIsFixed": basis.historical_is_fixed,
        "ArtificialCutIncident": incident,
        "CutBoundaryPreexistingFixed": preexisting,
        "CutBoundaryAddedFixed": added,
    }
    for name, values in expected.items():
        actual = np.asarray(mesh.point_data[name], dtype=bool)
        if not np.array_equal(actual, values):
            msg = f"{context} {name} contract changed"
            raise ValueError(msg)
    require_equal(
        int(preexisting.sum()),
        EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
        f"{context} preexisting cut marker count",
    )
    require_equal(
        int(added.sum()),
        EXPECTED_CUT_NEWLY_FIXED_VERTICES,
        f"{context} added cut marker count",
    )


def validate_frame(  # noqa: C901, PLR0912, PLR0915
    case: CaseInput,
    basis: SurfaceBasis,
    frame: pv.UnstructuredGrid,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        frame.n_points != basis.base_points.shape[0]
        or frame.n_cells != basis.tets.shape[0]
    ):
        msg = f"{case.label} step {step} dimensions changed"
        raise ValueError(msg)
    for name, actual, expected in (
        ("rest points", frame.points, basis.base_points),
        ("connectivity", frame.cells, basis.base_cells),
        ("cell types", frame.celltypes, basis.base_celltypes),
    ):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            msg = f"{case.label} step {step} {name} changed"
            raise ValueError(msg)
    ids = np.asarray(frame.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if not np.array_equal(ids, basis.base_global_ids):
        msg = f"{case.label} step {step} GlobalPointId changed"
        raise ValueError(msg)
    frame_is_fixed = np.asarray(frame.point_data["IsFixed"], dtype=bool)
    frame_fixed_mask = np.asarray(frame.point_data[FIXED_MASK.vtk], dtype=bool)
    frame_fixed_value = np.asarray(frame.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    expected_is_fixed = (
        basis.corrected_is_fixed
        if case.label == CORRECTED_LABEL
        else basis.historical_is_fixed
    )
    expected_fixed_mask = np.repeat(expected_is_fixed[:, None], 3, axis=1)
    if not np.array_equal(frame_is_fixed, expected_is_fixed):
        msg = f"{case.label} step {step} IsFixed contract changed"
        raise ValueError(msg)
    if not np.array_equal(frame_fixed_mask, expected_fixed_mask):
        msg = f"{case.label} step {step} FixedMask contract changed"
        raise ValueError(msg)
    if not np.array_equal(frame_fixed_value, np.zeros_like(frame_fixed_value)):
        msg = f"{case.label} step {step} FixedValue is not exact zero"
        raise ValueError(msg)
    if case.label == CORRECTED_LABEL:
        validate_corrected_cut_marker_fields(
            frame, basis, context=f"{case.label} history step {step}"
        )
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    target = np.asarray(frame.point_data["TargetDisplacement"], dtype=np.float64)
    mask = np.asarray(frame.point_data["LossMask"], dtype=bool)
    activation = np.asarray(frame.cell_data["RecoveredActivationInv"], dtype=np.float64)
    live_activation = np.asarray(frame.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    frame_activation_mask = np.asarray(frame.cell_data["ActivationMask"], dtype=bool)
    if displacement.shape != basis.target.shape or not np.isfinite(displacement).all():
        msg = f"{case.label} step {step} displacement is invalid"
        raise ValueError(msg)
    if case.label == CORRECTED_LABEL and not np.array_equal(
        displacement[basis.cut_mesh_ids],
        np.zeros_like(displacement[basis.cut_mesh_ids]),
    ):
        msg = (
            f"{case.label} step {step} violates exact-zero displacement on all "
            f"{EXPECTED_CUT_INCIDENT_VERTICES} artificial-cut vertices"
        )
        raise ValueError(msg)
    if (
        activation.shape != (basis.tets.shape[0], 6)
        or not np.isfinite(activation).all()
    ):
        msg = f"{case.label} step {step} activation is invalid"
        raise ValueError(msg)
    if not np.array_equal(activation, live_activation):
        msg = f"{case.label} step {step} activation fields differ"
        raise ValueError(msg)
    if not np.array_equal(frame_activation_mask, basis.activation_mask):
        msg = f"{case.label} step {step} ActivationMask changed"
        raise ValueError(msg)
    if not np.array_equal(
        activation[~basis.activation_mask],
        np.zeros_like(activation[~basis.activation_mask]),
    ):
        msg = f"{case.label} step {step} activation is nonzero outside ActivationMask"
        raise ValueError(msg)
    active_activation = activation[basis.activation_mask]
    if step == 0 and not np.array_equal(
        active_activation, np.zeros_like(active_activation)
    ):
        msg = f"{case.label} step 0 active activation is not exact zero"
        raise ValueError(msg)
    if not np.array_equal(target, basis.target) or not np.array_equal(
        mask, basis.loss_mask
    ):
        msg = f"{case.label} step {step} target or loss mask changed"
        raise ValueError(msg)
    trace = case.trace[step]
    require_close(field_scalar(frame, "inverse_step"), step, "history step")
    error_rms = float(
        np.linalg.norm((displacement - target)[mask]) / math.sqrt(int(mask.sum()))
    )
    require_close(error_rms, trace["target/error_rms"], f"{case.label} error {step}")
    require_close(
        field_scalar(frame, "inverse_error_rms"),
        error_rms,
        f"{case.label} stored error {step}",
    )
    activation_rms = float(
        np.linalg.norm(active_activation) / math.sqrt(active_activation.size)
    )
    require_close(
        activation_rms,
        trace["activation_inv/rms"],
        f"{case.label} activation RMS {step}",
    )
    require_close(
        np.abs(active_activation).max(),
        trace["activation_inv/max_abs"],
        f"{case.label} activation max {step}",
    )
    return displacement, activation


def area_weighted_point_rms(
    point_vectors: np.ndarray,
    triangles: np.ndarray,
    triangle_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    squared = np.sum(np.square(point_vectors), axis=1)
    triangle_squared = np.mean(squared[triangles], axis=1)
    active = triangle_mask & (weights > 0.0)
    return float(
        np.sqrt(
            np.dot(weights[active], triangle_squared[active]) / weights[active].sum()
        )
    )


def scalar_graph_laplacian(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros(values.shape[0], dtype=np.int64)
    np.add.at(neighbor_sum, edges[:, 0], values[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], values[edges[:, 0]])
    np.add.at(neighbor_count, edges[:, 0], 1)
    np.add.at(neighbor_count, edges[:, 1], 1)
    active = neighbor_count > 0
    result = np.zeros_like(values)
    result[active] = values[active] - neighbor_sum[active] / neighbor_count[active]
    return result


def vector_rms(values: np.ndarray) -> float:
    return float(np.linalg.norm(values) / math.sqrt(values.shape[0]))


def activation_matrix_diagnostics(active_activation: np.ndarray) -> dict[str, Any]:
    matrices = np.zeros((active_activation.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 + active_activation[:, 0]
    matrices[:, 1, 1] = 1.0 + active_activation[:, 1]
    matrices[:, 2, 2] = 1.0 + active_activation[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = active_activation[:, 3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = active_activation[:, 4]
    matrices[:, 0, 2] = matrices[:, 2, 0] = active_activation[:, 5]
    eigenvalues = np.linalg.eigvalsh(matrices)
    determinant = np.prod(eigenvalues, axis=1)
    singular_values = np.abs(eigenvalues)
    condition = singular_values.max(axis=1) / np.maximum(
        singular_values.min(axis=1), np.finfo(np.float64).tiny
    )
    if not all(
        np.isfinite(values).all() for values in (eigenvalues, determinant, condition)
    ):
        msg = "I + ActivationInv diagnostics are non-finite"
        raise ValueError(msg)
    return {
        "activation/I_plus_Ainv_min_eigenvalue": float(eigenvalues.min()),
        "activation/I_plus_Ainv_max_eigenvalue": float(eigenvalues.max()),
        "activation/I_plus_Ainv_min_determinant": float(determinant.min()),
        "activation/I_plus_Ainv_max_condition_number": float(condition.max()),
        "warning/I_plus_Ainv_non_spd_active_tets": int(
            np.sum(eigenvalues.min(axis=1) <= 0.0)
        ),
        "warning/I_plus_Ainv_nonpositive_det_active_tets": int(
            np.sum(determinant <= 0.0)
        ),
        "warning/I_plus_Ainv_policy": (
            "unconstrained activation conditioning is reported for interpretation; "
            "it is not an automatic visual-artifact veto"
        ),
    }


def frame_metrics(
    case: CaseInput,
    basis: SurfaceBasis,
    frame: pv.UnstructuredGrid,
    step: int,
) -> dict[str, Any]:
    displacement, activation = validate_frame(case, basis, frame, step)
    residual = displacement - basis.target
    error_rms = vector_rms(residual[basis.loss_mask])
    skin_displacement = displacement[basis.skin_mesh_ids]
    skin_residual = residual[basis.skin_mesh_ids]
    deformed = basis.skin_points + skin_displacement
    deformed_vectors, _, deformed_normals = triangle_geometry(deformed, basis.triangles)
    deformed_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                deformed_normals[basis.contraction_tri_0],
                deformed_normals[basis.contraction_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    dihedral_delta = deformed_dihedral - basis.contraction_target_dihedral
    dihedral_rms = float(
        np.sqrt(
            np.dot(basis.contraction_edge_weight, np.square(dihedral_delta))
            / basis.contraction_edge_weight.sum()
        )
    )
    residual_normal = np.einsum("ij,ij->i", skin_residual, basis.target_vertex_normals)
    normal_lap = scalar_graph_laplacian(residual_normal, basis.face_edges)
    face_normal_lap_rms = float(
        np.linalg.norm(normal_lap[basis.face_vertex_ids])
        / math.sqrt(basis.face_vertex_ids.size)
    )
    boundary_residual = vector_rms(skin_residual[basis.boundary_vertex_ids])
    band_residual = vector_rms(skin_residual[basis.boundary_band_ids])
    interior_residual = vector_rms(skin_residual[basis.deep_interior_ids])
    band_normal_lap = float(
        np.linalg.norm(normal_lap[basis.boundary_band_ids])
        / math.sqrt(basis.boundary_band_ids.size)
    )
    interior_normal_lap = float(
        np.linalg.norm(normal_lap[basis.deep_interior_ids])
        / math.sqrt(basis.deep_interior_ids.size)
    )
    deformed_volume = six_volume(basis.base_points + displacement, basis.tets)
    det_f = deformed_volume / basis.rest_six_volume
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    if not np.isfinite(det_f).all() or not np.isfinite(signed_normal_ratio).all():
        msg = f"{case.label} step {step} deformation diagnostics are non-finite"
        raise ValueError(msg)
    legacy = bumpiness_metrics(
        mask=basis.loss_mask,
        edges=basis.legacy_edges,
        displacement=displacement,
        target=basis.target,
    )
    rest_error = area_weighted_point_rms(
        skin_residual,
        basis.triangles,
        basis.face_triangle_mask,
        basis.rest_area,
    )
    target_error = area_weighted_point_rms(
        skin_residual,
        basis.triangles,
        basis.face_triangle_mask,
        basis.target_area,
    )
    rest_target = area_weighted_point_rms(
        basis.target[basis.skin_mesh_ids],
        basis.triangles,
        basis.face_triangle_mask,
        basis.rest_area,
    )
    target_target = area_weighted_point_rms(
        basis.target[basis.skin_mesh_ids],
        basis.triangles,
        basis.face_triangle_mask,
        basis.target_area,
    )
    active_activation = activation[basis.activation_mask]
    activation_diagnostics = activation_matrix_diagnostics(active_activation)
    cut_displacement = displacement[basis.cut_mesh_ids]
    cut_displacement_exact_zero = bool(
        np.array_equal(cut_displacement, np.zeros_like(cut_displacement))
    )
    metrics = {
        "case": case.label,
        "display_name": DISPLAY_NAMES[case.label],
        "cohort": case.cohort,
        "material_model": case.material_model,
        "step": step,
        "target/error_rms_m": error_rms,
        "target/error_rms_mm": 1e3 * error_rms,
        "target/error_rms_fraction_of_target": error_rms / basis.target_rms,
        "target/face_rest_area_weighted_error_rms_m": rest_error,
        "target/face_rest_area_weighted_error_fraction": rest_error / rest_target,
        "target/face_target_area_weighted_error_rms_m": target_error,
        "target/face_target_area_weighted_error_fraction": target_error / target_target,
        "bumpiness/contraction_target_relative_dihedral_rms_rad": dihedral_rms,
        "bumpiness/contraction_target_relative_dihedral_rms_deg": math.degrees(
            dihedral_rms
        ),
        "bumpiness/residual_normal_laplacian_rms_m": face_normal_lap_rms,
        "bumpiness/displacement_laplacian_rms_m": legacy[
            "bumpiness/displacement_laplacian_rms"
        ],
        "seam/boundary_edge_count": int(basis.boundary_edges.shape[0]),
        "seam/boundary_vertex_count": int(basis.boundary_vertex_ids.size),
        "seam/boundary_band_vertex_count": int(basis.boundary_band_ids.size),
        "seam/deep_interior_vertex_count": int(basis.deep_interior_ids.size),
        "seam/boundary_residual_rms_m": boundary_residual,
        "seam/boundary_band_residual_rms_m": band_residual,
        "seam/deep_interior_residual_rms_m": interior_residual,
        "seam/boundary_band_to_interior_residual_ratio": band_residual
        / max(interior_residual, np.finfo(np.float64).tiny),
        "seam/boundary_band_residual_normal_laplacian_rms_m": band_normal_lap,
        "seam/deep_interior_residual_normal_laplacian_rms_m": interior_normal_lap,
        "seam/boundary_band_to_interior_normal_laplacian_ratio": band_normal_lap
        / max(interior_normal_lap, np.finfo(np.float64).tiny),
        "activation/rms": float(
            np.linalg.norm(active_activation) / math.sqrt(active_activation.size)
        ),
        "activation/max_abs": float(np.abs(active_activation).max()),
        "activation/active_tet_rms": float(
            np.linalg.norm(active_activation) / math.sqrt(active_activation.size)
        ),
        "cut_boundary/policy": (
            HARD_FIXED_CUT_BOUNDARY_POLICY
            if case.label == CORRECTED_LABEL
            else "historical IsFixed; old-boundary control only"
        ),
        "cut_boundary/incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "cut_boundary/displacement_rms_m": float(
            np.linalg.norm(cut_displacement) / math.sqrt(EXPECTED_CUT_INCIDENT_VERTICES)
        ),
        "cut_boundary/displacement_max_abs_m": float(np.abs(cut_displacement).max()),
        "cut_boundary/displacement_exact_zero": cut_displacement_exact_zero,
        **activation_diagnostics,
        "warning/inverted_tets": int(np.sum(det_f <= 0.0)),
        "warning/inverted_tet_fraction": float(np.mean(det_f <= 0.0)),
        "warning/detF_min": float(det_f.min()),
        "warning/isface_folded_triangles": int(
            np.sum(basis.face_triangle_mask & (signed_normal_ratio <= 0.0))
        ),
        "warning/isface_folded_triangle_fraction": float(
            np.mean(signed_normal_ratio[basis.face_triangle_mask] <= 0.0)
        ),
        "warning/policy": (
            "inversion, fold, and seam concentration are visual-review evidence; "
            "small visually imperceptible counts are not automatic vetoes"
        ),
    }
    validate_finite_json(metrics, context=f"{case.label} step {step}")
    return metrics


def validate_static_artifacts(  # noqa: C901, PLR0912, PLR0915
    case: CaseInput, basis: SurfaceBasis
) -> None:
    target_mesh = pv.read(case.target_path)
    result_mesh = pv.read(case.result_path)
    for name, mesh in (("target", target_mesh), ("result", result_mesh)):
        if not isinstance(mesh, pv.UnstructuredGrid):
            msg = f"{case.label} {name} is not an UnstructuredGrid"
            raise TypeError(msg)
        for field, actual, expected in (
            ("points", mesh.points, basis.base_points),
            ("connectivity", mesh.cells, basis.base_cells),
            ("cell types", mesh.celltypes, basis.base_celltypes),
        ):
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                msg = f"{case.label} {name} {field} changed"
                raise ValueError(msg)
        stored_target = np.asarray(
            mesh.point_data["TargetDisplacement"], dtype=np.float64
        )
        stored_mask = np.asarray(mesh.point_data["LossMask"], dtype=bool)
        if not np.array_equal(stored_target, basis.target) or not np.array_equal(
            stored_mask, basis.loss_mask
        ):
            msg = f"{case.label} {name} target contract changed"
            raise ValueError(msg)
        if name == "target":
            if GLOBAL_POINT_ID.vtk in mesh.point_data:
                msg = (
                    f"{case.label} target unexpectedly contains post-builder "
                    "GlobalPointId metadata"
                )
                raise ValueError(msg)
        else:
            ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
            if not np.array_equal(ids, basis.base_global_ids):
                msg = f"{case.label} {name} GlobalPointId contract changed"
                raise ValueError(msg)
        is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
        fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)
        fixed_value = np.asarray(mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64)
        is_corrected_solver_state = case.label == CORRECTED_LABEL and name == "result"
        expected_is_fixed = (
            basis.corrected_is_fixed
            if is_corrected_solver_state
            else basis.historical_is_fixed
        )
        if not np.array_equal(is_fixed, expected_is_fixed):
            msg = f"{case.label} {name} IsFixed contract changed"
            raise ValueError(msg)
        if not np.array_equal(
            fixed_mask, np.repeat(expected_is_fixed[:, None], 3, axis=1)
        ):
            msg = f"{case.label} {name} FixedMask contract changed"
            raise ValueError(msg)
        if not np.array_equal(fixed_value, np.zeros_like(fixed_value)):
            msg = f"{case.label} {name} FixedValue is not exact zero"
            raise ValueError(msg)
        if is_corrected_solver_state:
            validate_corrected_cut_marker_fields(
                mesh, basis, context=f"{case.label} {name}"
            )
    best_step = int(case.summary["best/step"])
    if not 0 <= best_step <= TERMINAL_STEP:
        msg = f"{case.label} best step is outside 0..40"
        raise ValueError(msg)
    best = case.history.frame(best_step, deep_copy=True)
    result_displacement = np.asarray(
        result_mesh.point_data["Displacement"], dtype=np.float64
    )
    if case.label == CORRECTED_LABEL and not np.array_equal(
        result_displacement[basis.cut_mesh_ids],
        np.zeros_like(result_displacement[basis.cut_mesh_ids]),
    ):
        msg = "corrected result violates exact-zero artificial-cut displacement"
        raise ValueError(msg)
    result_activation = np.asarray(
        result_mesh.cell_data["RecoveredActivationInv"], dtype=np.float64
    )
    result_live_activation = np.asarray(
        result_mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
    )
    if not np.array_equal(result_activation, result_live_activation):
        msg = f"{case.label} result activation fields differ"
        raise ValueError(msg)
    if not np.array_equal(
        result_activation[~basis.activation_mask],
        np.zeros_like(result_activation[~basis.activation_mask]),
    ):
        msg = f"{case.label} result activation is nonzero outside ActivationMask"
        raise ValueError(msg)
    if not np.array_equal(
        result_displacement,
        np.asarray(best.point_data["Displacement"], dtype=np.float64),
    ):
        msg = f"{case.label} result is not its declared best history frame"
        raise ValueError(msg)
    if not np.array_equal(
        result_activation,
        np.asarray(best.cell_data["RecoveredActivationInv"], dtype=np.float64),
    ):
        msg = f"{case.label} result activation is not its best history frame"
        raise ValueError(msg)


def scan_cases(
    cases: list[CaseInput], basis: SurfaceBasis
) -> dict[str, list[dict[str, Any]]]:
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        validate_static_artifacts(case, basis)
        rows = [
            frame_metrics(case, basis, case.history.frame(step), step)
            for step in range(EXPECTED_EVALUATIONS)
        ]
        trajectories[case.label] = rows
    return trajectories


def select_checkpoints(
    trajectories: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    primary_terminal = [
        trajectories[label][TERMINAL_STEP] for label in PRIMARY_CASE_ORDER
    ]
    tau = max(
        float(row["target/error_rms_fraction_of_target"]) for row in primary_terminal
    )
    selected: dict[str, dict[str, Any]] = {}
    for label in RENDER_CASE_ORDER:
        nearest = min(
            trajectories[label],
            key=lambda row: (
                abs(float(row["target/error_rms_fraction_of_target"]) - tau),
                int(row["step"]),
            ),
        )
        selected[label] = {
            **nearest,
            "matching/tau": tau,
            "matching/signed_error": float(
                nearest["target/error_rms_fraction_of_target"]
            )
            - tau,
            "matching/absolute_error": abs(
                float(nearest["target/error_rms_fraction_of_target"]) - tau
            ),
            "matching/contributes_to_tau": label in PRIMARY_CASE_ORDER,
        }
    terminal = [trajectories[label][TERMINAL_STEP] for label in RENDER_CASE_ORDER]
    matched = [selected[label] for label in RENDER_CASE_ORDER]
    primary_values = np.asarray(
        [
            selected[label]["target/error_rms_fraction_of_target"]
            for label in PRIMARY_CASE_ORDER
        ],
        dtype=np.float64,
    )
    matching = {
        "tau": tau,
        "tau_rule": (
            "maximum terminal target-error fraction across corrected and no-skin "
            "only; both selected checkpoints are actual saved frames with no "
            "interpolation; historical full-boundary+3D is excluded"
        ),
        "primary_case_order": list(PRIMARY_CASE_ORDER),
        "secondary_excluded_from_tau": list(SECONDARY_CASE_ORDER),
        "selection": {
            label: {
                "step": int(selected[label]["step"]),
                "fidelity": selected[label]["target/error_rms_fraction_of_target"],
                "absolute_error": selected[label]["matching/absolute_error"],
            }
            for label in RENDER_CASE_ORDER
        },
        "primary_selected_fidelity_spread": float(np.ptp(primary_values)),
        "method": "nearest saved discrete frame; no interpolation",
    }
    return terminal, matched, matching


def corrected_no_skin_effects(
    rows: list[dict[str, Any]], *, checkpoint: str
) -> dict[str, Any]:
    by_label = {row["case"]: row for row in rows}
    require_equal(
        set(PRIMARY_CASE_ORDER) <= set(by_label),
        expected=True,
        context=f"{checkpoint} primary effect inputs",
    )
    corrected = by_label[CORRECTED_LABEL]
    no_skin = by_label[NO_SKIN_LABEL]
    metrics = (
        "target/error_rms_fraction_of_target",
        "target/face_target_area_weighted_error_fraction",
        "bumpiness/contraction_target_relative_dihedral_rms_deg",
        "bumpiness/residual_normal_laplacian_rms_m",
        "seam/boundary_band_to_interior_residual_ratio",
        "seam/boundary_band_to_interior_normal_laplacian_ratio",
    )
    effects: dict[str, Any] = {
        "checkpoint": checkpoint,
        "comparison": (
            "corrected hard-fixed minus old-boundary no-skin; ratio is corrected "
            "/ no-skin. This is a primary control comparison, not a skin-only "
            "causal contrast, because the boundary condition also differs"
        ),
        "corrected_step": int(corrected["step"]),
        "no_skin_step": int(no_skin["step"]),
    }
    for metric in metrics:
        corrected_value = float(corrected[metric])
        no_skin_value = float(no_skin[metric])
        effects[f"{metric}/delta"] = corrected_value - no_skin_value
        effects[f"{metric}/ratio"] = corrected_value / max(
            abs(no_skin_value), np.finfo(np.float64).tiny
        )
    return effects


CSV_FIELDS = (
    "case",
    "display_name",
    "cohort",
    "material_model",
    "step",
    "target/error_rms_fraction_of_target",
    "target/error_rms_mm",
    "target/face_rest_area_weighted_error_fraction",
    "target/face_target_area_weighted_error_fraction",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
    "bumpiness/displacement_laplacian_rms_m",
    "seam/boundary_band_to_interior_residual_ratio",
    "seam/boundary_band_to_interior_normal_laplacian_ratio",
    "activation/rms",
    "activation/max_abs",
    "activation/I_plus_Ainv_min_eigenvalue",
    "activation/I_plus_Ainv_min_determinant",
    "activation/I_plus_Ainv_max_condition_number",
    "cut_boundary/policy",
    "cut_boundary/displacement_rms_m",
    "cut_boundary/displacement_max_abs_m",
    "cut_boundary/displacement_exact_zero",
    "warning/I_plus_Ainv_non_spd_active_tets",
    "warning/inverted_tets",
    "warning/detF_min",
    "warning/isface_folded_triangles",
)


def write_csv(path: Path, trajectories: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for label in RENDER_CASE_ORDER:
            for row in trajectories[label]:
                writer.writerow({key: row[key] for key in CSV_FIELDS})


def write_table(
    path: Path,
    terminal: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    matching: dict[str, Any],
) -> None:
    lines = [
        "# Corrected baseline checkpoints",
        "",
        f"Primary matching tau: `{matching['tau']:.8g}`. The historical old-skin case is excluded from tau.",
        "",
        "| checkpoint | case | role | step | error/target | target-area error/target | dihedral deg | residual-normal Lap mm | seam residual ratio | seam normal-Lap ratio | cut max mm | cut exact zero | inverted | folded |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    lines.extend(
        (
            "| {checkpoint} | {case} | {cohort} | {step} | {error:.6g} | "
            "{area:.6g} | {dihedral:.6g} | {normal:.6g} | {seam:.6g} | "
            "{seam_lap:.6g} | {cut_max:.6g} | {cut_zero} | {inverted} | {folded} |".format(
                checkpoint=checkpoint,
                case=row["case"],
                cohort=row["cohort"],
                step=row["step"],
                error=row["target/error_rms_fraction_of_target"],
                area=row["target/face_target_area_weighted_error_fraction"],
                dihedral=row["bumpiness/contraction_target_relative_dihedral_rms_deg"],
                normal=1e3 * row["bumpiness/residual_normal_laplacian_rms_m"],
                seam=row["seam/boundary_band_to_interior_residual_ratio"],
                seam_lap=row["seam/boundary_band_to_interior_normal_laplacian_ratio"],
                cut_max=1e3 * row["cut_boundary/displacement_max_abs_m"],
                cut_zero=row["cut_boundary/displacement_exact_zero"],
                inverted=row["warning/inverted_tets"],
                folded=row["warning/isface_folded_triangles"],
            )
        )
        for checkpoint, rows in (("terminal", terminal), ("matched", matched))
        for row in rows
    )
    lines.extend(
        [
            "",
            "The IsFace membrane has exactly 707 boundary edges. Seam metrics are interpretation and visual-review evidence, not automatic vetoes.",
            "",
            "The no-skin and old-skin cases retain the historical boundary. They are controls, not boundary-matched causal material ablations.",
            "",
            "Small inversion or fold counts are also warnings only when the artifact is visually imperceptible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trajectories(
    path: Path,
    trajectories: dict[str, list[dict[str, Any]]],
    terminal: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    matching: dict[str, Any],
) -> None:
    colors = {
        CORRECTED_LABEL: "#0072B2",
        HISTORICAL_LABEL: "#D55E00",
        NO_SKIN_LABEL: "#009E73",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for label in RENDER_CASE_ORDER:
        rows = trajectories[label]
        steps = np.asarray([row["step"] for row in rows])
        fidelity = np.asarray(
            [row["target/error_rms_fraction_of_target"] for row in rows]
        )
        linestyle = "--" if label == HISTORICAL_LABEL else "-"
        axes_flat[0].plot(
            steps,
            fidelity,
            label=DISPLAY_NAMES[label],
            color=colors[label],
            linestyle=linestyle,
        )
        axes_flat[1].plot(
            fidelity,
            [
                row["bumpiness/contraction_target_relative_dihedral_rms_deg"]
                for row in rows
            ],
            color=colors[label],
            linestyle=linestyle,
        )
        axes_flat[2].plot(
            fidelity,
            [1e3 * row["bumpiness/residual_normal_laplacian_rms_m"] for row in rows],
            color=colors[label],
            linestyle=linestyle,
        )
        axes_flat[3].plot(
            fidelity,
            [row["seam/boundary_band_to_interior_residual_ratio"] for row in rows],
            color=colors[label],
            linestyle=linestyle,
        )
    by_terminal = {row["case"]: row for row in terminal}
    by_matched = {row["case"]: row for row in matched}
    for label in RENDER_CASE_ORDER:
        for point, marker in ((by_terminal[label], "s"), (by_matched[label], "o")):
            x = point["target/error_rms_fraction_of_target"]
            axes_flat[0].scatter(
                [point["step"]], [x], color=colors[label], marker=marker, s=35
            )
            axes_flat[1].scatter(
                [x],
                [point["bumpiness/contraction_target_relative_dihedral_rms_deg"]],
                color=colors[label],
                marker=marker,
                s=35,
            )
            axes_flat[2].scatter(
                [x],
                [1e3 * point["bumpiness/residual_normal_laplacian_rms_m"]],
                color=colors[label],
                marker=marker,
                s=35,
            )
            axes_flat[3].scatter(
                [x],
                [point["seam/boundary_band_to_interior_residual_ratio"]],
                color=colors[label],
                marker=marker,
                s=35,
            )
    axes_flat[0].axhline(float(matching["tau"]), color="black", linestyle=":")
    axes_flat[0].set(xlabel="inverse evaluation step", ylabel="target error / target")
    axes_flat[1].set(
        xlabel="target error / target",
        ylabel="target-relative contraction dihedral RMS [deg]",
    )
    axes_flat[2].set(
        xlabel="target error / target",
        ylabel="residual-normal Laplacian RMS [mm]",
    )
    axes_flat[3].set(
        xlabel="target error / target",
        ylabel="boundary-band / deep-interior residual RMS",
    )
    axes_flat[0].legend(fontsize="small")
    for axis in axes_flat:
        axis.grid(alpha=0.3)
    fig.suptitle("Corrected hard-fixed baseline vs old-boundary controls")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def deformed_face(basis: SurfaceBasis, displacement: np.ndarray) -> pv.PolyData:
    surface = basis.skin.copy(deep=True)
    surface.points = basis.skin_points + displacement[basis.skin_mesh_ids]
    face = surface.extract_cells(np.flatnonzero(basis.face_triangle_mask))
    return face.extract_surface(algorithm="dataset_surface")


def checkpoint_displacement(case: CaseInput, checkpoint: dict[str, Any]) -> np.ndarray:
    frame = case.history.frame(int(checkpoint["step"]), deep_copy=True)
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    if not np.isfinite(displacement).all():
        msg = f"{case.label} render displacement is non-finite"
        raise ValueError(msg)
    return displacement


def render_contact_sheet(
    path: Path,
    *,
    title: str,
    basis: SurfaceBasis,
    cases: dict[str, CaseInput],
    checkpoints: list[dict[str, Any]],
) -> None:
    by_label = {row["case"]: row for row in checkpoints}
    surfaces: list[tuple[str, pv.PolyData, str]] = [
        ("target", deformed_face(basis, basis.target), "reference target")
    ]
    for label in RENDER_CASE_ORDER:
        row = by_label[label]
        displacement = checkpoint_displacement(cases[label], row)
        annotation = (
            f"{label} | step {row['step']}\n"
            f"err/target={row['target/error_rms_fraction_of_target']:.4f} | "
            f"area={row['target/face_target_area_weighted_error_fraction']:.4f}\n"
            f"dih={row['bumpiness/contraction_target_relative_dihedral_rms_deg']:.3f} deg | "
            f"nLap={1e3 * row['bumpiness/residual_normal_laplacian_rms_m']:.3f} mm\n"
            f"seam={row['seam/boundary_band_to_interior_residual_ratio']:.3f} | "
            f"cut={1e3 * row['cut_boundary/displacement_max_abs_m']:.3g} mm | "
            f"inv={row['warning/inverted_tets']} | "
            f"fold={row['warning/isface_folded_triangles']}"
        )
        surfaces.append((label, deformed_face(basis, displacement), annotation))
    front = np.asarray((0.0, 0.0, 1.0))
    views = (
        ("front", front, basis.face_focus, basis.face_scale),
        (
            "30 degree",
            np.asarray(
                (math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0)))
            ),
            basis.face_focus,
            basis.face_scale,
        ),
        ("mouth", front, basis.mouth_focus, basis.mouth_scale),
        ("eye-cheek (+x)", front, basis.eye_cheek_focus, basis.eye_cheek_scale),
    )
    plotter = pv.Plotter(
        shape=(len(views), len(surfaces)),
        off_screen=True,
        window_size=(2350, 1600),
        lighting="light kit",
        border=False,
    )
    plotter.set_background("white")
    for row_id, (view_name, direction, focus, scale) in enumerate(views):
        for column, (label, surface, annotation) in enumerate(surfaces):
            plotter.subplot(row_id, column)
            plotter.add_mesh(
                surface,
                color="#d8b49c",
                smooth_shading=True,
                specular=0.15,
                show_edges=False,
            )
            plotter.add_text(
                f"{view_name} | {annotation}",
                position="upper_left",
                font_size=8,
                color="black",
            )
            plotter.enable_parallel_projection()
            camera_focus = np.asarray(focus, dtype=np.float64)
            plotter.camera.position = tuple(camera_focus + 0.30 * direction)
            plotter.camera.focal_point = tuple(camera_focus)
            plotter.camera.up = (0.0, 1.0, 0.0)
            plotter.camera.parallel_scale = float(scale)
            if label == "target" and row_id == 0:
                plotter.add_text(
                    title, position="lower_left", font_size=9, color="black"
                )
    plotter.screenshot(path)
    plotter.close()


def temporary_output(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def validate_config(cfg: Config) -> None:
    exact_inputs = {
        "input_mesh": (cfg.input_mesh, PREPARED_MESH),
        "input_metric_skin": (cfg.input_metric_skin, SOURCE_SKIN),
        "input_manifest": (
            cfg.input_manifest,
            CORRECTED_DATA_DIR / MANIFEST_NAME,
        ),
        "input_corrected_summary": (
            cfg.input_corrected_summary,
            CORRECTED_DATA_DIR / AGGREGATE_NAME,
        ),
    }
    for name, (actual, expected) in exact_inputs.items():
        require_equal(actual.resolve(), expected.resolve(), name)
    expected_outputs = {
        "output_json": CORRECTED_DATA_DIR / "30-corrected-baseline-analysis.json",
        "output_csv": CORRECTED_DATA_DIR / "30-corrected-baseline-trajectories.csv",
        "output_table": CORRECTED_DATA_DIR / "30-corrected-baseline-checkpoints.md",
        "output_plot": CORRECTED_DATA_DIR / "30-corrected-baseline-trajectories.png",
        "output_terminal_views": (
            CORRECTED_DATA_DIR / "30-corrected-baseline-terminal-views.png"
        ),
        "output_matched_views": (
            CORRECTED_DATA_DIR / "30-corrected-baseline-matched-views.png"
        ),
    }
    actual_outputs = {name: getattr(cfg, name) for name in expected_outputs}
    for name, expected in expected_outputs.items():
        require_equal(actual_outputs[name].resolve(), expected.resolve(), name)
    if len({path.resolve() for path in actual_outputs.values()}) != len(actual_outputs):
        msg = "analysis output paths must be distinct"
        raise ValueError(msg)
    guarded_paths = [
        path
        for output in actual_outputs.values()
        for path in (output, temporary_output(output))
    ]
    stale = [str(path) for path in guarded_paths if path.exists()]
    if stale:
        msg = f"refusing to overwrite existing analysis outputs: {stale}"
        raise FileExistsError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"analysis requires the Agg backend, got {mpl.get_backend()}"
        raise RuntimeError(msg)


def run(cfg: Config) -> None:
    # This analyzer performs only static reads, metrics, and rendering. The
    # formal producer artifacts were audited before this entry point was enabled.
    validate_config(cfg)
    require_equal(
        file_identity(ANALYZED_INVERSE),
        {
            "size_bytes": ANALYZED_INVERSE_SIZE_BYTES,
            "sha256": ANALYZED_INVERSE_SHA256,
        },
        "analyzed inverse implementation identity",
    )
    require_equal(
        file_identity(PREPARE_IMPLEMENTATION),
        {
            "size_bytes": PREPARE_IMPLEMENTATION.stat().st_size,
            "sha256": PREPARE_IMPLEMENTATION_SHA256,
        },
        "prepare implementation identity",
    )
    require_equal(
        file_identity(cfg.input_mesh),
        {
            "size_bytes": PREPARED_MESH_SIZE_BYTES,
            "sha256": PREPARED_MESH_SHA256,
        },
        "prepared mesh identity",
    )
    require_equal(
        file_identity(cfg.input_metric_skin),
        {"size_bytes": SOURCE_SKIN_SIZE_BYTES, "sha256": SOURCE_SKIN_SHA256},
        "metric skin identity",
    )
    base = pv.read(cfg.input_mesh)
    metric_skin = pv.read(cfg.input_metric_skin)
    if not isinstance(base, pv.UnstructuredGrid):
        msg = "prepared mesh is not an UnstructuredGrid"
        raise TypeError(msg)
    if not isinstance(metric_skin, pv.PolyData):
        msg = "metric skin is not PolyData"
        raise TypeError(msg)
    _manifest, candidate, corrected_skin, manifest_source = validate_manifest(cfg, base)
    metric_roi_provenance = validate_metric_roi_intersection(
        metric_skin, corrected_skin
    )
    require_equal(
        metric_roi_provenance["corrected_skin_global_face_sha256"],
        candidate["content/global_face_key_sha256"],
        "corrected skin/candidate global face identity",
    )
    require_equal(
        metric_roi_provenance["corrected_skin_global_face_sha256"],
        _manifest["domain_contract"]["isface_global_face_key_sha256"],
        "corrected skin/audited-domain global face identity",
    )
    basis = build_surface_basis(base, metric_skin)
    aggregate, aggregate_row = validate_aggregate(cfg, manifest_source["manifest"])
    corrected = load_corrected_case(aggregate_row, candidate)
    no_skin = load_historical_case(NO_SKIN_SPEC)
    historical = load_historical_case(HISTORICAL_SPEC)
    cases = {case.label: case for case in (corrected, historical, no_skin)}
    require_equal(set(cases), EXPECTED_CASES, "analysis case set")
    ordered_cases = [cases[label] for label in RENDER_CASE_ORDER]
    trajectories = scan_cases(ordered_cases, basis)
    terminal, matched, matching = select_checkpoints(trajectories)
    primary_effects = {
        "terminal": corrected_no_skin_effects(terminal, checkpoint="terminal"),
        "matched": corrected_no_skin_effects(matched, checkpoint="matched"),
    }
    source = {
        "prepared_mesh": {
            "path": str(cfg.input_mesh),
            **file_identity(cfg.input_mesh),
        },
        "metric_surface": {
            "path": str(cfg.input_metric_skin),
            "role": (
                "pinned geometry and IsFace/contraction masks only; historical "
                "material arrays are not used as the corrected mechanics"
            ),
            **file_identity(cfg.input_metric_skin),
            "roi_intersection": metric_roi_provenance,
        },
        "corrected_manifest": {
            "path": str(cfg.input_manifest),
            **manifest_source["manifest"],
        },
        "corrected_skin": manifest_source,
        "corrected_aggregate": {
            "path": str(cfg.input_corrected_summary),
            **file_identity(cfg.input_corrected_summary),
            "schema_version": aggregate["schema_version"],
            "canonical_archive": require_byte_identical_archive(
                cfg.input_corrected_summary,
                CORRECTED_AGGREGATE_ARCHIVE,
                context="corrected aggregate canonical archive",
            ),
        },
        "artificial_cut_boundary": {
            **basis.cut_boundary_provenance,
            "history_validation": (
                "all 41 corrected frames independently read and required exact-zero "
                "displacement at the bound 6,980-vertex GlobalPointId set"
            ),
        },
        "implementations": {
            "inverse/path": str(ANALYZED_INVERSE),
            "inverse/size_bytes": ANALYZED_INVERSE_SIZE_BYTES,
            "inverse/sha256": ANALYZED_INVERSE_SHA256,
            "prepare/path": str(PREPARE_IMPLEMENTATION),
            "prepare/sha256": PREPARE_IMPLEMENTATION_SHA256,
            "inverse_runtime_bundle_sha256": (EXPECTED_INVERSE_RUNTIME_BUNDLE_SHA256),
        },
        "cases": {
            case.label: {
                "cohort": case.cohort,
                "summary_path": str(case.summary_path),
                "trace_path": str(case.trace_path),
                "history_path": str(case.history_path),
                "result_path": str(case.result_path),
                "target_path": str(case.target_path),
                "identities": case.identities,
            }
            for case in ordered_cases
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "kind": "human-face-smile-corrected-plane-stress-baseline-analysis",
        "design": DESIGN,
        "case_order": list(RENDER_CASE_ORDER),
        "cohorts": {
            "primary_tau_cases": list(PRIMARY_CASE_ORDER),
            "secondary_historical_diagnostic": list(SECONDARY_CASE_ORDER),
        },
        "protocol": {
            "evaluations_per_case": EXPECTED_EVALUATIONS,
            "terminal_step": TERMINAL_STEP,
            "fixed_learning_rate": EXPECTED_LR,
            "fresh_zero_activation": True,
            "fresh_zero_displacement": True,
            "forward_initial_displacement_exact_zero": True,
            "activation_mode": "per-muscle-tet-6dof-unconstrained",
            "history_time_steps": "exact 0..40",
            "cut_boundary_policy": HARD_FIXED_CUT_BOUNDARY_POLICY,
            "cut_boundary_incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "model_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
            "model_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
            "per_frame_cut_displacement": "required exact zero",
            "corrected_inverse_started_by_analyzer": False,
        },
        "matching": matching,
        "corrected_vs_no_skin_effects": primary_effects,
        "terminal_checkpoints": terminal,
        "matched_checkpoints": matched,
        "trajectories": trajectories,
        "metric_contract": {
            "target_point_fidelity": (
                "RMS(Displacement-TargetDisplacement) / RMS(TargetDisplacement) "
                "on SmileLossMask"
            ),
            "target_area_fidelity": (
                "triangle-area-weighted vertex RMS on all-vertex IsFace triangles, "
                "reported with both rest- and target-area weights"
            ),
            "target_relative_dihedral": (
                "rest-edge-length-weighted RMS deformed-minus-target dihedral on "
                "the pinned contraction ROI"
            ),
            "residual_normal_roughness": (
                "umbrella-Laplacian RMS of target-normal residual displacement on "
                "the IsFace graph"
            ),
            "seam": {
                "boundary_edges": EXPECTED_SKIN_BOUNDARY_EDGES,
                "boundary_band": "boundary vertices plus their one-ring neighbors",
                "comparison": "boundary-band versus deep-interior residual and normal-Laplacian RMS",
                "policy": "interpretation and visual-review evidence; not an automatic veto",
            },
            "activation_spatial_jump": (
                "intentionally not computed: this experiment has no reviewed "
                "anatomically weighted muscle-tet adjacency. Activation RMS, max, "
                "inactive-zero, and I+ActivationInv eigen/determinant/condition "
                "diagnostics are reported instead"
            ),
        },
        "acceptance_policy": {
            "hard_failures": [
                "missing or identity-mismatched artifact",
                "non-finite metric or incomplete 41-frame history",
                "failed forward or adjoint trace row",
                "target, topology, material, or implementation contract mismatch",
                "any nonzero artificial-cut displacement in a corrected history frame",
            ],
            "visual_review_warnings_only": [
                "small inverted tetrahedron count",
                "small folded IsFace triangle count",
                "boundary-seam concentration",
            ],
            "historical_case_role": (
                "secondary old-boundary diagnostic only; its full-boundary/3D "
                "activation and mechanics are not a corrected baseline and do not "
                "define tau"
            ),
            "no_skin_case_role": (
                "hash-pinned old-boundary primary control; because its boundary "
                "differs from the corrected hard-fixed baseline, the comparison is "
                "not a skin-only causal ablation"
            ),
        },
        "render_policy": {
            "columns": ["Target", *RENDER_CASE_ORDER],
            "views": ["front", "30 degree", "mouth", "eye-cheek (+x)"],
            "projection": "parallel",
            "terminal": str(cfg.output_terminal_views),
            "matched": str(cfg.output_matched_views),
        },
        "visual_review": {
            "status": "pending",
            "next_inverse_automatic": False,
            "required_action": (
                "review terminal and matched front, 30 degree, mouth, and "
                "eye-cheek views before discussing any further inverse"
            ),
        },
        "source": source,
        "limitations": [
            "single target and no replicate",
            "41 evaluations are a fixed-budget screen, not convergence",
            "the corrected and historical activations were optimized independently",
            "the hash-pinned no-skin and old-skin controls retain the historical boundary",
            (
                "the corrected target artifact is a pre-builder target contract and "
                "therefore retains prepared historical boundary metadata; the result "
                "and all history frames carry and enforce the hard-fixed solver state"
            ),
            "nearest-fidelity matching uses saved frames without interpolation",
            "the IsFace membrane is an open facial ROI with 707 boundary edges",
        ],
    }
    validate_finite_json(payload, context="analysis payload")
    outputs = (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_plot,
        cfg.output_terminal_views,
        cfg.output_matched_views,
    )
    staged = {path: temporary_output(path) for path in outputs}
    published: list[Path] = []
    try:
        staged[cfg.output_json].write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        write_csv(staged[cfg.output_csv], trajectories)
        write_table(staged[cfg.output_table], terminal, matched, matching)
        plot_trajectories(
            staged[cfg.output_plot], trajectories, terminal, matched, matching
        )
        render_contact_sheet(
            staged[cfg.output_terminal_views],
            title="terminal fixed-budget states",
            basis=basis,
            cases=cases,
            checkpoints=terminal,
        )
        render_contact_sheet(
            staged[cfg.output_matched_views],
            title="nearest discrete primary-tau states",
            basis=basis,
            cases=cases,
            checkpoints=matched,
        )
        for path in outputs:
            staged[path].replace(path)
            published.append(path)
    except Exception:
        for path in staged.values():
            path.unlink(missing_ok=True)
        for path in published:
            path.unlink(missing_ok=True)
        raise
    for path in outputs:
        cherries.log_output(path)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    cherries.main(run)
