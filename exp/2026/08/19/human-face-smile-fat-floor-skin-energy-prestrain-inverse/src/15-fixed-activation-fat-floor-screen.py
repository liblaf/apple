from __future__ import annotations

# This is a deliberately blocked formal producer with dense, contextual gates.
# ruff: noqa: C901, EM101, EM102, PERF401, PLR0912, PLR0915, SLF001, TRY003
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from vtkmodules.vtkCommonExecutionModel import (
    vtkStreamingDemandDrivenPipeline as StreamingPipeline,
)

from liblaf import cherries, melon
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "fixed-activation-positive-fat-floor-dual-seed-forward-screen"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PRODUCER = Path(__file__).resolve()

# Static review must flip only this forward-specific blocker.  The other three
# approvals are negative invariants: this producer must never run an inverse,
# adjoint, or backward pass.
FORWARD_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
INVERSE_EXECUTION_APPROVED = False
ADJOINT_EXECUTION_APPROVED = False
BACKWARD_EXECUTION_APPROVED = False
APPROVAL_BLOCKER = (
    "NO-GO: the four fixed-activation fat-floor forwards await static review; "
    "inverse, adjoint, and backward execution are permanently out of scope"
)

# The first material producer is explicitly rejected: its output inherited a
# stale ``SkinActivationInvDiag`` cell field from the source VTP.  These v2
# names and identity sentinels are deliberately non-existent until the clean
# producer is statically reviewed and its artifacts are frozen below.
REJECTED_V1_MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-manifest.json"
REJECTED_V1_MANIFEST_SHA256 = (
    "843179e074d00ead3469cd9e0e5f69f2f0b521398e86709d1ecb466bda2f26a9"
)
CLEAN_V2_CONTRACT_READY = True
CLEAN_V2_DESIGN = "corrected-isface-two-case-selective-efat-c020-inverse-materials-v2"
PREPARED_MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-v2-manifest.json"
PREPARE_IMPLEMENTATION = GROUP_DIR / "src/10-prepare-material-cases.py"
CLEAN_V2_SKIN_PATHS = {
    "HFP0": GROUP_DIR
    / "data/10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
    "HFP1": GROUP_DIR
    / "data/10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
}
CLEAN_V2_BLOCKER = (
    "NO-GO: clean v2 material producer/manifest/VTP identities have not landed; "
    "the rejected v1 manifest must never be substituted"
)
MATERIAL_APPROVAL_DISABLED_ASSIGNMENT = (
    b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
)
MATERIAL_APPROVAL_ENABLED_ASSIGNMENT = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
PREPARED_MESH = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
)
DRIVER_SKIN = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/"
    "10-material-candidates/skin-e100-p000.vtp"
)
OLD_GROUP = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse"
)
OLD_DATA = OLD_GROUP / "data"
OLD_AGGREGATE = OLD_DATA / "20-selective-skin-prestrain-inverse-summary-final.json"
REVIEWED_PROBE = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/src/"
    "15-forward-domain-conversion-probe.py"
)
REVIEWED_PROBE_SRC = REVIEWED_PROBE.parent
REVIEWED_REFERENCE = REVIEWED_PROBE_SRC / "_reference.py"
RUNTIME_METRICS = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_metrics.py"
)
RUNTIME_CONFIG = (
    REPO_ROOT / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/src/"
    "_human_face_config.py"
)
RUNTIME_FORWARD = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_forward.py"
)
RUNTIME_OUTPUT = (
    REPO_ROOT / "exp/2026/06/17/human-face-smile-prestrain-v2/src/_human_face_output.py"
)
CORE_MODULI = REPO_ROOT / "src/liblaf/apple/common/_moduli.py"
KOITER_IMPLEMENTATION = REPO_ROOT / "src/liblaf/apple/warp/fem/_koiter.py"

OUTPUT_SUMMARY = GROUP_DIR / "data/15-fixed-activation-fat-floor-screen-summary.json"
OUTPUT_TABLE = GROUP_DIR / "data/15-fixed-activation-fat-floor-screen-table.md"
OUTPUT_ROOT = GROUP_DIR / "data/15-fixed-activation-fat-floor-screen"
OUTPUT_ROOT_NAME = OUTPUT_ROOT.name

EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_ACTIVE_TETS = 288_235
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_EXPANDING_TRIANGLES = 16_723
EXPECTED_MODEL_FIXED_VERTICES = 33_636
EXPECTED_MODEL_FIXED_DOFS = 100_908
EXPECTED_CUT_VERTICES = 6_980
EXPECTED_TARGET_RMS_M = 0.005310139062299789
EXPECTED_FRAMES = 41
EXPECTED_TIMES = tuple(range(EXPECTED_FRAMES))
EXPECTED_TOPOLOGY_HASHES = {
    "points": "ec9544035eeb2eee2b733f16584a17a1873a0622855905c8d2a98113aab44a74",
    "cells": "61678752f43b9bbd641602c71fb79ee802d4c6753d1adebc6647b2ff0a9bbab3",
    "celltypes": "9a7caed190d749ea866232198a7902bb4eacb72690aed702a1e4683d208aa342",
    "global_ids": "e95db0c1c49794f4c72edce768aa741e23205e7d047c242c190d20a1e674de34",
    "raw_smile": "49766a469e4fa9ef276990aacb5a92860a86b4d930b384d3fd1c7b87ee0875c6",
    "target": "823d503d67916988bad9aba52efc7303ee943bc7c9206112f2b3ee8b5e2ff375",
    "target_finite": "16ff833d4cf7c5e7e04162fa1ca23d870e667b1a8cc412df47a84a07ebc8116c",
    "loss_mask": "7f3d956377de1fccb5be08e7c8809ad62ae5f770b61a09a238cdde724a9a4d68",
    "is_fixed": "ca57d727e1be3f81f41f1727ad02879cf0e119e5c0f7c7af093a003aada6af2e",
    "fixed_mask": "c36c07ca22fc145215d38b7ac03b1b1b498cc1d851824f9298e874b594bb5c7b",
    "fixed_value": "cd15d6cc5c4f7df89e19c02300d1a34711a78ec6535547b029110528d36e0545",
    "cut": "0b540cc5229a8985ea2c5c75c5919cc00fb150bf15c72d9ac118d18561897d1a",
    "activation_mask": "5cc4bd20c083cf5adf59a1d023e285157bec035ffe6f0ee686a3e0211b360d9e",
}
EXPECTED_EXPANSION_MASK_SHA256 = (
    "1da30d0805e41ebb56de39fb26ccd54c2b7a8bd7f4d1257459cbc7b9aa0bc05e"
)
EXPECTED_CONTRACTION_MASK_SHA256 = (
    "276296bf0dab911ded6d6609f5288c8f4560cb4d92211188aba11d30222ddeab"
)
EXPECTED_CONTRACTING_TRIANGLES = 13_159
EXPECTED_CONTRACTION_INTERIOR_EDGES = 18_038
EXPECTED_CONTRACTION_TRIANGLE_PAIR_SHA256 = (
    "2cdabb614e0d909fbcc3d797da865d7b346b5c86a74959cedde7dcafcf7af5a0"
)
EXPECTED_CONTRACTION_EDGE_WEIGHT_SHA256 = (
    "86f34abba95b70eb740dff065af96f14fd2852a7b89cc34d34f0484de93bdd63"
)
EXPECTED_FULL_INTERIOR_EDGES = 44_495
EXPECTED_EXPANSION_INTERIOR_EDGES = 23_316
EXPECTED_EXPANSION_VERTICES = 9_953
EXPECTED_EXPANSION_GRAPH_EDGES = 26_853
EXPECTED_EXPANSION_TRIANGLE_PAIR_SHA256 = (
    "438a4e3123d22b9cf45981daf4b7167cb6edfb774c8a378d5fe4e8fe1818a6bb"
)
EXPECTED_EXPANSION_EDGE_WEIGHT_SHA256 = (
    "d5c92c1c10cde33eebbc7582b6d299f7d7dbd50c516b1b166559af8d3a49932e"
)
EXPECTED_EXPANSION_GRAPH_EDGE_SHA256 = (
    "87e20c0dac1dcc950c1530647aafaab2a60cfde63681f27c86a455c01848f280"
)
EXPECTED_EXPANSION_TARGET_RMS_M = 0.00551044304488279

EXPECTED_OLD_METRICS = {
    "H1P0": {
        "target/error_rms_m": 0.002711933813887892,
        "metric/D_full_deg": 9.557820596896375,
        "metric/D_contraction_deg": 9.890388520550617,
        "metric/L_full_mm": 0.1640810209037049,
        "metric/D_exp_deg": 9.022411067068647,
        "metric/L_exp_mm": 0.14806483822802338,
        "metric/Q95_exp_mm": 0.31302647178428045,
        "warning/isface_folded_triangles": 2,
        "warning/inverted_tets": 1,
    },
    "H1P1": {
        "target/error_rms_m": 0.002708134919531868,
        "metric/D_full_deg": 7.048900530196572,
        "metric/D_contraction_deg": 4.93827293954525,
        "metric/L_full_mm": 0.14407833941972015,
        "metric/D_exp_deg": 8.196673134273233,
        "metric/L_exp_mm": 0.13867216256753595,
        "metric/Q95_exp_mm": 0.29135965318055296,
        "warning/isface_folded_triangles": 2,
        "warning/inverted_tets": 2,
    },
}

Seed = Literal["zero", "old-equilibrium"]


@dataclass(frozen=True)
class IdentitySpec:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FrameSpec:
    material_case: str
    source_case: str
    step: int
    history: IdentitySpec
    trace: IdentitySpec
    activation_sha256: str
    displacement_sha256: str
    expected_target_error_rms_m: float


@dataclass
class TemporalHistory:
    case_id: str
    path: Path
    reader: Any
    times: np.ndarray

    @classmethod
    def open(cls, case_id: str, path: Path) -> TemporalHistory:
        reader = pv.get_reader(path)
        vtk_reader = reader.reader
        vtk_reader.UpdateInformation()
        info = vtk_reader.GetOutputInformation(0)
        key = StreamingPipeline.TIME_STEPS()
        if not info.Has(key):
            raise ValueError(f"{case_id} history has no TIME_STEPS: {path}")
        times = np.asarray(
            [info.Get(key, index) for index in range(info.Length(key))],
            dtype=np.float64,
        )
        if not np.array_equal(times, np.arange(EXPECTED_FRAMES, dtype=np.float64)):
            raise ValueError(f"{case_id} history TIME_STEPS are not exactly 0..40")
        return cls(case_id=case_id, path=path, reader=reader, times=times)

    def frame(self, step: int) -> pv.UnstructuredGrid:
        if not 0 <= step < EXPECTED_FRAMES:
            raise IndexError(f"invalid {self.case_id} history step {step}")
        vtk_reader = self.reader.reader
        vtk_reader.UpdateTimeStep(float(self.times[step]))
        mesh = pv.wrap(vtk_reader.GetOutputDataObject(0))
        if not isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
        return mesh.copy(deep=True)


@dataclass(frozen=True)
class MetricBasis:
    base_points: np.ndarray
    cells: np.ndarray
    celltypes: np.ndarray
    global_ids: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    target_rms: float
    is_fixed: np.ndarray
    fixed_mask: np.ndarray
    fixed_value: np.ndarray
    cut_mask: np.ndarray
    activation_mask: np.ndarray
    skin: pv.PolyData
    skin_mesh_ids: np.ndarray
    triangles: np.ndarray
    full_edges: np.ndarray
    expansion_edges: np.ndarray
    expansion_vertices: np.ndarray
    rest_area: np.ndarray
    target_vertex_normals: np.ndarray
    full_tri_0: np.ndarray
    full_tri_1: np.ndarray
    full_target_dihedral: np.ndarray
    full_edge_weight: np.ndarray
    contraction_tri_0: np.ndarray
    contraction_tri_1: np.ndarray
    contraction_target_dihedral: np.ndarray
    contraction_edge_weight: np.ndarray
    expansion_tri_0: np.ndarray
    expansion_tri_1: np.ndarray
    expansion_target_dihedral: np.ndarray
    expansion_edge_weight: np.ndarray
    expansion_mask: np.ndarray
    tets: np.ndarray
    rest_six_volume: np.ndarray
    rest_area_vectors: np.ndarray
    rest_area_vector_norm: np.ndarray


INPUTS = {
    "prepared_manifest": IdentitySpec(
        PREPARED_MANIFEST,
        53_161,
        "1dd835e2638b4fc3789e79c8b9c620383889a0e0963a0f82a5ea0b43b2375693",
    ),
    "prepare_implementation": IdentitySpec(
        PREPARE_IMPLEMENTATION,
        54_601,
        "28d0c3667d3d803c58e8dc60e52c896cd62758ac586918fcf8f86b9da94985f5",
    ),
    "prepared_mesh": IdentitySpec(
        PREPARED_MESH,
        76_792_914,
        "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
    ),
    "driver_skin": IdentitySpec(
        DRIVER_SKIN,
        38_742_137,
        "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
    ),
    "old_aggregate": IdentitySpec(
        OLD_AGGREGATE,
        387_036,
        "cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6",
    ),
    "reviewed_probe": IdentitySpec(
        REVIEWED_PROBE,
        87_717,
        "741d3f3db966f8b1e25b389a8734176fb6991a6872e6f8a1a8b875bd3ec5e2f5",
    ),
    "reviewed_reference": IdentitySpec(
        REVIEWED_REFERENCE,
        4_108,
        "470db910d6bec9ec81e06b5b46512781a188c252683b44b57b539ddb63295615",
    ),
    "runtime_metrics": IdentitySpec(
        RUNTIME_METRICS,
        3_775,
        "1407d2988444b31332f2688c6535eca5db58b5be31d63fae6abd6bf8bf78e0c1",
    ),
    "runtime_config": IdentitySpec(
        RUNTIME_CONFIG,
        2_992,
        "fcd7757486c3f0664816a6595e17af27a87ffec1c9c9e24b18908506b444ffeb",
    ),
    "runtime_forward": IdentitySpec(
        RUNTIME_FORWARD,
        8_205,
        "2d0ff39b13555300c000e6dd43e16c274752263b703746ad8174072033819e03",
    ),
    "runtime_output": IdentitySpec(
        RUNTIME_OUTPUT,
        8_395,
        "29bae977a4b31e82276aca15fdaae3bdda37e6a3e71493876b6fd973db1a1c61",
    ),
    "core_moduli": IdentitySpec(
        CORE_MODULI,
        1_210,
        "9d5c14f27b9a08a8a4f9cd3ce4e3076f2375ed1108e84e94d307c9439e1a303d",
    ),
    "koiter": IdentitySpec(
        KOITER_IMPLEMENTATION,
        17_329,
        "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c",
    ),
}

FRAME_SPECS = (
    FrameSpec(
        material_case="HFP0",
        source_case="H1P0",
        step=14,
        history=IdentitySpec(
            OLD_DATA / "20-h1p0-steps.vtkhdf",
            2_066_227_811,
            "abd81c7c51480ccc10c35488f22a7734e82c171093fc4ef5443762d18507c93c",
        ),
        trace=IdentitySpec(
            OLD_DATA / "20-h1p0-trace.jsonl",
            79_563,
            "a8686640dabe2da6b6499bdf4c05c2f18af713d9a92ed0728c715a397b20e830",
        ),
        activation_sha256="fe8210127eb43d8c152fe7e2cddfbdbc0d4c7807d9050058eb231e5e763c48c6",
        displacement_sha256="50f9a83a785725fbb679007a8d2f0fe1c85b99988ada350e5b181b1b08ea2a74",
        expected_target_error_rms_m=0.002711933813887892,
    ),
    FrameSpec(
        material_case="HFP1",
        source_case="H1P1",
        step=12,
        history=IdentitySpec(
            OLD_DATA / "20-h1p1-steps.vtkhdf",
            2_072_876_568,
            "c32ede23f73ae27a301d845e6f8598494f03e1927dce043118e1a52b49092b8b",
        ),
        trace=IdentitySpec(
            OLD_DATA / "20-h1p1-trace.jsonl",
            79_592,
            "3771439cfc5cf8fad7073b62bb4df5f9716acad9cb30d28f93e57b3530ce5ba7",
        ),
        activation_sha256="8c971f4464ff1ef265e3a38b6d09faa390d06deeb8fe9f1cfe0d4ce3429cc427",
        displacement_sha256="0313c5cfec4fe88cf652939bbee26297ff935f2f4e003e76b4de179db3151f6c",
        expected_target_error_rms_m=0.002708134919531868,
    ),
)

EXPECTED_CASE_ORDER = (
    "HFP0-from-H1P0-frame014-zero",
    "HFP0-from-H1P0-frame014-old-equilibrium",
    "HFP1-from-H1P1-frame012-zero",
    "HFP1-from-H1P1-frame012-old-equilibrium",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input(PREPARED_MANIFEST)
    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_driver_skin: Path = cherries.input(DRIVER_SKIN)
    input_old_aggregate: Path = cherries.input(OLD_AGGREGATE)
    # Multi-gigabyte histories are identity-checked directly and deliberately
    # never registered/logged as Cherries inputs.
    input_h1p0_history: Path = FRAME_SPECS[0].history.path
    input_h1p0_trace: Path = cherries.input(FRAME_SPECS[0].trace.path)
    input_h1p1_history: Path = FRAME_SPECS[1].history.path
    input_h1p1_trace: Path = cherries.input(FRAME_SPECS[1].trace.path)
    output_summary: Path = cherries.output(
        "15-fixed-activation-fat-floor-screen-summary.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "15-fixed-activation-fat-floor-screen-table.md", mkdir=True
    )
    output_dir_name: str = OUTPUT_ROOT_NAME
    require_solver_success: bool = True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _identity_dict(spec: IdentitySpec) -> dict[str, int | str]:
    return {"size_bytes": spec.size_bytes, "sha256": spec.sha256}


def _bytes_identity(content: bytes) -> dict[str, int | str]:
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _replace_exact_source_line(
    source: bytes, *, expected: bytes, replacement: bytes
) -> tuple[bytes, int]:
    lines = source.splitlines(keepends=True)
    count = 0
    for index, line in enumerate(lines):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        if body == expected:
            lines[index] = replacement + ending
            count += 1
    return b"".join(lines), count


def _require_identity(name: str, spec: IdentitySpec) -> dict[str, Any]:
    actual = _file_identity(spec.path)
    expected = _identity_dict(spec)
    if actual != expected:
        raise ValueError(f"{name} identity changed: {actual} != {expected}")
    return {"path": str(spec.path), **actual}


def _raw_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_array_hash(
    name: str, values: np.ndarray, expected: str, *, dtype: str
) -> str:
    actual = _raw_sha256(values, dtype=dtype)
    if actual != expected:
        raise ValueError(f"{name} hash changed: {actual} != {expected}")
    return actual


def _normalized_smile(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    rows = [
        json.loads(line, parse_constant=reject_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != EXPECTED_FRAMES or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not an exact 41-row trace")
    if [int(row["step"]) for row in rows] != list(EXPECTED_TIMES):
        raise ValueError(f"{path} trace steps are not exactly 0..40")
    return rows


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _case_id(spec: FrameSpec, seed: Seed) -> str:
    return f"{spec.material_case}-from-{spec.source_case}-frame{spec.step:03d}-{seed}"


def _case_path(spec: FrameSpec, seed: Seed) -> Path:
    return OUTPUT_ROOT / _case_id(spec, seed) / "result.vtu"


def _validate_config(cfg: Config) -> None:
    if not CLEAN_V2_CONTRACT_READY:
        raise RuntimeError(CLEAN_V2_BLOCKER)
    if PREPARED_MANIFEST.resolve() == REJECTED_V1_MANIFEST.resolve():
        raise RuntimeError("clean v2 manifest path aliases the rejected v1 manifest")
    if INPUTS["prepared_manifest"].sha256 == REJECTED_V1_MANIFEST_SHA256:
        raise RuntimeError("clean v2 manifest pin aliases the rejected v1 identity")
    exact = {
        "input_manifest": PREPARED_MANIFEST,
        "input_mesh": PREPARED_MESH,
        "input_driver_skin": DRIVER_SKIN,
        "input_old_aggregate": OLD_AGGREGATE,
        "input_h1p0_history": FRAME_SPECS[0].history.path,
        "input_h1p0_trace": FRAME_SPECS[0].trace.path,
        "input_h1p1_history": FRAME_SPECS[1].history.path,
        "input_h1p1_trace": FRAME_SPECS[1].trace.path,
        "output_summary": OUTPUT_SUMMARY,
        "output_table": OUTPUT_TABLE,
    }
    changed = [
        name
        for name, expected in exact.items()
        if Path(getattr(cfg, name)).resolve() != expected.resolve()
    ]
    if changed or cfg.output_dir_name != OUTPUT_ROOT_NAME:
        raise ValueError(
            f"fixed forward screen paths cannot be overridden: {changed}, "
            f"output_dir_name={cfg.output_dir_name!r}"
        )
    if cfg.require_solver_success is not True:
        raise ValueError("formal forward screen must gate every solver success")
    if (
        INVERSE_EXECUTION_APPROVED
        or ADJOINT_EXECUTION_APPROVED
        or BACKWARD_EXECUTION_APPROVED
    ):
        raise RuntimeError("inverse/adjoint/backward approvals must remain false")
    if not FORWARD_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
        raise RuntimeError(APPROVAL_BLOCKER)
    stale = [
        path
        for path in (
            OUTPUT_SUMMARY,
            OUTPUT_TABLE,
            OUTPUT_ROOT,
            _temporary_path(OUTPUT_SUMMARY),
            _temporary_path(OUTPUT_TABLE),
            *(
                candidate
                for spec in FRAME_SPECS
                for seed in ("zero", "old-equilibrium")
                for candidate in (
                    _case_path(spec, seed),
                    _temporary_path(_case_path(spec, seed)),
                )
            ),
        )
        if path.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing stale forward-screen outputs: {stale}")


def _unregistered_cherries_path(value: str | Path, *_: Any, **__: Any) -> Path:
    return Path(value)


def _load_pinned_module(
    path: Path, expected_sha256: str, *, module_name: str
) -> ModuleType:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"pinned module changed before import: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_reviewed_probe() -> ModuleType:
    source_dir = str(REVIEWED_PROBE_SRC)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    _load_pinned_module(
        REVIEWED_REFERENCE,
        INPUTS["reviewed_reference"].sha256,
        module_name="_reference",
    )
    original_input = cherries.input
    original_output = cherries.output
    try:
        cherries.input = _unregistered_cherries_path
        cherries.output = _unregistered_cherries_path
        probe = _load_pinned_module(
            REVIEWED_PROBE,
            INPUTS["reviewed_probe"].sha256,
            module_name="_fat_floor_fixed_activation_reviewed_probe",
        )
    finally:
        cherries.input = original_input
        cherries.output = original_output
    expected_modules = {
        "_human_face_metrics": RUNTIME_METRICS,
        "_human_face_config": RUNTIME_CONFIG,
        "_human_face_forward": RUNTIME_FORWARD,
        "_human_face_output": RUNTIME_OUTPUT,
    }
    for name, expected in expected_modules.items():
        module = sys.modules.get(name)
        actual = None if module is None else getattr(module, "__file__", None)
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise ImportError(
                f"reviewed probe imported {name} from {actual}, not {expected}"
            )
    return probe


def _identity_from_record(name: str, path: Path, record: Any) -> IdentitySpec:
    if not isinstance(record, dict):
        raise TypeError(f"{name} identity is not an object")
    if set(record) != {"size_bytes", "sha256"}:
        raise ValueError(f"{name} identity fields changed: {sorted(record)}")
    size = record.get("size_bytes")
    sha256 = record.get("sha256")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"{name} size is invalid: {size!r}")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{name} SHA-256 is invalid: {sha256!r}")
    return IdentitySpec(path=path, size_bytes=size, sha256=sha256)


def _validate_prepare_provenance(
    producer: dict[str, Any],
) -> tuple[IdentitySpec, dict[str, Any]]:
    if Path(str(producer.get("path"))).resolve() != PREPARE_IMPLEMENTATION.resolve():
        raise ValueError("clean v2 producer path changed")
    reviewed = producer.get("statically_reviewed_preapproval_source")
    executable = producer.get("executable_source")
    reconstruction = producer.get("approval_only_reconstruction")
    if not all(isinstance(row, dict) for row in (reviewed, executable, reconstruction)):
        raise TypeError("clean v2 producer dual-source provenance is incomplete")
    expected_reviewed = _identity_dict(INPUTS["prepare_implementation"])
    if reviewed.get("file_identity") != expected_reviewed:
        raise ValueError("clean v2 preapproval producer identity differs from the pin")
    if reviewed.get("approval_assignment") != (
        "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
    ):
        raise ValueError("clean v2 preapproval assignment provenance changed")
    live = PREPARE_IMPLEMENTATION.read_bytes()
    live_identity = _bytes_identity(live)
    if live_identity == expected_reviewed:
        preapproval = live
        enabled, count = _replace_exact_source_line(
            live,
            expected=MATERIAL_APPROVAL_DISABLED_ASSIGNMENT,
            replacement=MATERIAL_APPROVAL_ENABLED_ASSIGNMENT,
        )
        live_mode = "statically-reviewed-preapproval"
    elif live_identity == executable.get("file_identity"):
        enabled = live
        preapproval, count = _replace_exact_source_line(
            live,
            expected=MATERIAL_APPROVAL_ENABLED_ASSIGNMENT,
            replacement=MATERIAL_APPROVAL_DISABLED_ASSIGNMENT,
        )
        live_mode = "approved-executable"
    else:
        raise ValueError("live clean v2 producer is neither reviewed nor executable")
    if count != 1 or _bytes_identity(preapproval) != expected_reviewed:
        raise ValueError("clean v2 producer is not an approval-only source transition")
    executable_identity = _bytes_identity(enabled)
    if executable.get("file_identity") != executable_identity:
        raise ValueError("clean v2 executable producer identity changed")
    if executable.get("approval_assignment") != (
        "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
    ):
        raise ValueError("clean v2 executable assignment provenance changed")
    expected_reconstruction = {
        "verified": True,
        "replacement_count": 1,
        "permitted_edit": ("EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False -> True"),
        "reconstructed_executable_identity": executable_identity,
    }
    if reconstruction != expected_reconstruction:
        raise ValueError("clean v2 approval-only reconstruction record changed")
    live_spec = IdentitySpec(
        PREPARE_IMPLEMENTATION,
        int(live_identity["size_bytes"]),
        str(live_identity["sha256"]),
    )
    return live_spec, {
        "path": str(PREPARE_IMPLEMENTATION),
        "live_mode": live_mode,
        "live_identity": live_identity,
        "statically_reviewed_preapproval_source": reviewed,
        "executable_source": executable,
        "approval_only_reconstruction": reconstruction,
    }


def _validate_clean_v2_manifest(
    manifest: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, IdentitySpec],
    IdentitySpec,
    dict[str, Any],
]:
    if manifest.get("schema_version") != 2:
        raise ValueError("clean v2 material manifest schema_version changed")
    if manifest.get("design") != CLEAN_V2_DESIGN:
        raise ValueError(
            f"clean v2 material manifest design changed: {manifest.get('design')!r}"
        )
    if manifest.get("complete") is not True:
        raise ValueError("clean v2 material manifest is incomplete")
    approval = manifest.get("approval")
    if not isinstance(approval, dict):
        raise TypeError("clean v2 approval record is absent")
    if approval.get("material_preparation_static_review") is not True:
        raise ValueError("clean v2 material preparation lacks static approval")
    if approval.get("forward_or_adjoint_smoke_approved") is not False:
        raise ValueError("clean v2 manifest unexpectedly approves a forward/adjoint")
    if approval.get("inverse_execution_approved") is not False:
        raise ValueError("clean v2 manifest unexpectedly approves an inverse")
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise TypeError("clean v2 producer record is absent")
    live_producer, producer_metrics = _validate_prepare_provenance(producer)
    if manifest.get("case_order") != ["HFP1", "HFP0"]:
        raise ValueError("clean v2 material case order must be exactly HFP1,HFP0")
    candidates = manifest.get("cases")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("clean v2 manifest must contain exactly two cases")
    by_case: dict[str, dict[str, Any]] = {}
    identities: dict[str, IdentitySpec] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("clean v2 case is not an object")
        case_id = candidate.get("case_id")
        if case_id not in CLEAN_V2_SKIN_PATHS or case_id in by_case:
            raise ValueError(f"unexpected or duplicate clean v2 case: {case_id!r}")
        if candidate.get("generated") is not True:
            raise ValueError(f"{case_id} is not certified as generated")
        skin = candidate.get("skin")
        if not isinstance(skin, dict):
            raise TypeError(f"{case_id} skin record is absent")
        expected_path = CLEAN_V2_SKIN_PATHS[case_id]
        if Path(str(skin.get("path"))).resolve() != expected_path.resolve():
            raise ValueError(f"{case_id} clean v2 skin path changed")
        if skin.get("points") != EXPECTED_SKIN_POINTS:
            raise ValueError(f"{case_id} clean v2 skin point count changed")
        if skin.get("triangles") != EXPECTED_SKIN_TRIANGLES:
            raise ValueError(f"{case_id} clean v2 skin triangle count changed")
        arrays = skin.get("arrays")
        if not isinstance(arrays, dict):
            raise TypeError(f"{case_id} clean v2 array manifest is absent")
        required = {
            ACTIVATION_INV.vtk,
            "C020PrestrainEnabled",
            "ClippedTargetRestAreaRatio",
            "ExpandingTriangle",
            FRACTION.vtk,
            "RestArea",
            "SelectiveFatFloor",
            "SkinPoissonRatio",
            "SkinYoungModulusMPa",
            "SkinActivationInvDiag",
            "StressFreeAreaRatio",
            "TargetRestAreaRatio",
            "ArtificialCutTriangle",
            "DisallowedGroupTriangle",
            "FixedTriangle",
            "GingivaProximityTriangle",
            "IsFaceTriangle",
            "SourceOuterTriangle",
            "TeethProximityTriangle",
            LAMBDA.vtk,
            MU.vtk,
        }
        if set(arrays) != required:
            raise ValueError(
                f"{case_id} clean v2 material array contract changed: {sorted(arrays)}"
            )
        identity = _identity_from_record(
            f"{case_id} clean v2 skin", expected_path, skin.get("file_identity")
        )
        if identity.sha256 in {
            "2199b33ba7896bfde82a9e1fcf12e7782e9e89daa742b787eb267a824f1ae855",
            "f3c2ebaf95f7b82c15a15743ef2a1be3eea378f1a89f9044df379179732c6bf7",
        }:
            raise ValueError(f"{case_id} clean v2 skin aliases a rejected v1 VTP")
        by_case[case_id] = candidate
        identities[case_id] = identity
    if set(by_case) != set(CLEAN_V2_SKIN_PATHS):
        raise ValueError("clean v2 manifest case set changed")
    output_contract = manifest.get("output_contract")
    if not isinstance(output_contract, dict):
        raise TypeError("clean v2 output contract is absent")
    if Path(str(output_contract.get("manifest_path"))).resolve() != (
        PREPARED_MANIFEST.resolve()
    ):
        raise ValueError("clean v2 output contract points at another manifest")
    return by_case, identities, live_producer, producer_metrics


def _validate_old_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    exact = {
        "schema_version": 1,
        "design": "corrected-isface-selective-e000-c020-three-case-inverse",
        "complete": True,
        "case_order": ["H0P1", "H1P1", "H1P0"],
        "n_cases": 3,
    }
    changed = {
        key: (aggregate.get(key), expected)
        for key, expected in exact.items()
        if aggregate.get(key) != expected
    }
    if changed:
        raise ValueError(f"old inverse aggregate contract changed: {changed}")
    if aggregate.get("hard_failures") != []:
        raise ValueError("old inverse aggregate contains a hard failure")
    cases = aggregate.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("old inverse aggregate case records changed")
    by_case = {case.get("case_id"): case for case in cases if isinstance(case, dict)}
    if set(by_case) != {"H0P1", "H1P1", "H1P0"}:
        raise ValueError("old inverse aggregate case IDs changed")
    for spec in FRAME_SPECS:
        case = by_case[spec.source_case]
        expected = {
            "artifact/history_path": str(spec.history.path),
            "artifact/history_size_bytes": spec.history.size_bytes,
            "artifact/history_sha256": spec.history.sha256,
            "artifact/trace_path": str(spec.trace.path),
            "artifact/trace_size_bytes": spec.trace.size_bytes,
            "artifact/trace_sha256": spec.trace.sha256,
            "history/frames": EXPECTED_FRAMES,
            "best/step": 40,
            "final/step": 40.0,
            "activation/mode": "per-muscle-tet-6dof",
        }
        changed = {
            key: (case.get(key), value)
            for key, value in expected.items()
            if case.get(key) != value
        }
        if changed:
            raise ValueError(
                f"{spec.source_case} old artifact contract changed: {changed}"
            )
    return {"verified": True, "source_case_order": aggregate["case_order"]}


def _require_finite_array(name: str, values: np.ndarray) -> None:
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} is non-numeric or non-finite")


def _validate_manifest_array(
    case_id: str, name: str, values: np.ndarray, record: Any
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"{case_id}/{name} array record is not an object")
    if record.get("association") != "cell":
        raise ValueError(f"{case_id}/{name} association is not cell")
    dtype = record.get("dtype")
    shape = record.get("shape")
    expected_hash = record.get("sha256_le_c")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        raise TypeError(f"{case_id}/{name} dtype or shape record is malformed")
    if list(values.shape) != shape:
        raise ValueError(f"{case_id}/{name} shape changed: {values.shape} != {shape}")
    _require_finite_array(f"{case_id}/{name}", values)
    if record.get("finite") is not True:
        raise ValueError(f"{case_id}/{name} manifest does not certify finiteness")
    actual_hash = _raw_sha256(values, dtype=dtype)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{case_id}/{name} exact array hash changed: "
            f"{actual_hash} != {expected_hash}"
        )
    cast = np.asarray(values, dtype=np.dtype(dtype))
    if not math.isclose(
        float(cast.min()), float(record.get("min")), rel_tol=0.0, abs_tol=0.0
    ) or not math.isclose(
        float(cast.max()), float(record.get("max")), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"{case_id}/{name} exact range changed")
    return {"dtype": dtype, "shape": shape, "sha256_le_c": actual_hash}


def _triangle_faces(skin: pv.PolyData) -> np.ndarray:
    faces = np.asarray(skin.faces, dtype=np.int64)
    if faces.size != 4 * skin.n_cells:
        raise ValueError("clean v2 skin is not pure packed triangles")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("clean v2 skin contains a non-triangle")
    return faces[:, 1:].copy()


def _map_global_ids(mesh: pv.UnstructuredGrid, skin: pv.PolyData) -> np.ndarray:
    mesh_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    skin_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if np.unique(mesh_ids).size != mesh_ids.size:
        raise ValueError("mesh GlobalPointId is not unique")
    if np.unique(skin_ids).size != skin_ids.size:
        raise ValueError("skin GlobalPointId is not unique")
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], skin_ids)
    if np.any(positions >= mesh_ids.size) or not np.array_equal(
        mesh_ids[order[positions]], skin_ids
    ):
        raise ValueError("clean v2 skin does not map exactly to the prepared mesh")
    mapped = order[positions]
    if not np.array_equal(
        np.asarray(skin.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[mapped],
    ):
        raise ValueError("clean v2 skin coordinates differ from the prepared mesh")
    return mapped


def _validate_skin(
    *,
    mesh: pv.UnstructuredGrid,
    case_id: str,
    candidate: dict[str, Any],
    identity: IdentitySpec,
    probe: ModuleType,
) -> tuple[pv.PolyData, dict[str, Any]]:
    _require_identity(f"clean v2 {case_id} skin", identity)
    loaded = pv.read(identity.path)
    if not isinstance(loaded, pv.PolyData):
        raise TypeError(f"clean v2 {case_id} skin is not PolyData")
    skin = loaded.copy(deep=True)
    if skin.n_points != EXPECTED_SKIN_POINTS or skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        raise ValueError(f"clean v2 {case_id} skin dimensions changed")
    arrays = candidate["skin"]["arrays"]
    array_metrics = {
        name: _validate_manifest_array(
            case_id, name, np.asarray(skin.cell_data[name]), record
        )
        for name, record in arrays.items()
    }
    triangles = _triangle_faces(skin)
    mapped = _map_global_ids(mesh, skin)
    domain = probe._validate_domain(mesh, skin, "isface")
    raw_ratio = np.asarray(skin.cell_data["TargetRestAreaRatio"], dtype=np.float64)
    expansion = np.asarray(skin.cell_data["ExpandingTriangle"], dtype=bool)
    selective = np.asarray(skin.cell_data["SelectiveFatFloor"], dtype=bool)
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    poisson = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    lambda_ = np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(skin.cell_data[MU.vtk], dtype=np.float64)
    fraction = np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    activation_diag = np.asarray(
        skin.cell_data["SkinActivationInvDiag"], dtype=np.float64
    )
    enabled = np.asarray(skin.cell_data["C020PrestrainEnabled"], dtype=bool)
    clipped = np.asarray(skin.cell_data["ClippedTargetRestAreaRatio"], dtype=np.float64)
    stress_free = np.asarray(skin.cell_data["StressFreeAreaRatio"], dtype=np.float64)
    if not np.array_equal(expansion, raw_ratio > 1.0):
        raise ValueError(f"{case_id} expansion mask is not strict raw R > 1")
    if not np.array_equal(selective, expansion):
        raise ValueError(f"{case_id} selective fat-floor mask differs from expansion")
    if int(expansion.sum()) != EXPECTED_EXPANDING_TRIANGLES:
        raise ValueError(f"{case_id} expansion triangle count changed")
    _require_array_hash(
        f"{case_id} expansion mask",
        expansion,
        EXPECTED_EXPANSION_MASK_SHA256,
        dtype="u1",
    )
    expected_young = np.where(expansion, 0.003, 0.2)
    if not np.array_equal(young, expected_young):
        raise ValueError(f"{case_id} selective E formula changed")
    if not np.array_equal(poisson, np.full_like(poisson, 0.49)):
        raise ValueError(f"{case_id} skin Poisson ratio changed")
    expected_lambda = young * poisson / (1.0 - np.square(poisson))
    expected_mu = young / (2.0 * (1.0 + poisson))
    if not np.allclose(lambda_, expected_lambda, rtol=1.0e-13, atol=1.0e-14):
        raise ValueError(f"{case_id} plane-stress lambda formula changed")
    if not np.allclose(mu, expected_mu, rtol=1.0e-13, atol=1.0e-14):
        raise ValueError(f"{case_id} plane-stress mu formula changed")
    if not np.array_equal(fraction, np.ones_like(fraction)):
        raise ValueError(f"{case_id} Koiter fraction is not exact one")
    if not np.array_equal(activation_diag, activation[:, 0]):
        raise ValueError(f"{case_id} clean v2 SkinActivationInvDiag is not recertified")
    expected_clipped = np.clip(raw_ratio, 0.5, 1.0)
    if not np.array_equal(clipped, expected_clipped):
        raise ValueError(f"{case_id} clipped target/rest ratio changed")
    if case_id == "HFP0":
        if np.any(enabled) or not np.array_equal(activation, np.zeros_like(activation)):
            raise ValueError("HFP0 is not exact p000")
        if not np.array_equal(stress_free, np.ones_like(stress_free)):
            raise ValueError("HFP0 stress-free area ratio is not exact one")
    else:
        rho = math.pow(0.98, 2) * expected_clipped
        expected_diag = np.reciprocal(np.sqrt(rho)) - 1.0
        expected_activation = np.stack(
            (expected_diag, expected_diag, np.zeros_like(expected_diag)), axis=1
        )
        if not np.all(enabled) or not np.array_equal(activation, expected_activation):
            raise ValueError("HFP1 c020 ActivationInv formula changed")
        if not np.array_equal(stress_free, rho):
            raise ValueError("HFP1 stress-free area ratio formula changed")
    return skin, {
        **domain,
        "case_id": case_id,
        "identity": {"path": str(identity.path), **_identity_dict(identity)},
        "mapped_mesh_point_sha256_le_i8": _raw_sha256(mapped, dtype="<i8"),
        "triangle_sha256_le_i8": _raw_sha256(triangles, dtype="<i8"),
        "expanding_triangles": int(expansion.sum()),
        "SkinActivationInvDiag_rederived_from_ActivationInv": True,
        "manifest_arrays": array_metrics,
        "formula_readback_verified": True,
    }


def _load_base_mesh() -> pv.UnstructuredGrid:
    loaded = pv.read(PREPARED_MESH)
    if not isinstance(loaded, pv.UnstructuredGrid):
        raise TypeError("prepared anatomy is not an UnstructuredGrid")
    mesh = loaded.copy(deep=True)
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError("prepared anatomy dimensions changed")
    _require_array_hash(
        "prepared points",
        np.asarray(mesh.points),
        EXPECTED_TOPOLOGY_HASHES["points"],
        dtype="<f8",
    )
    _require_array_hash(
        "prepared cells",
        np.asarray(mesh.cells),
        EXPECTED_TOPOLOGY_HASHES["cells"],
        dtype="<i8",
    )
    _require_array_hash(
        "prepared celltypes",
        np.asarray(mesh.celltypes),
        EXPECTED_TOPOLOGY_HASHES["celltypes"],
        dtype="u1",
    )
    global_ids = np.arange(mesh.n_points, dtype=np.int64)
    if GLOBAL_POINT_ID.vtk in mesh.point_data and not np.array_equal(
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64), global_ids
    ):
        raise ValueError("prepared anatomy contains non-canonical GlobalPointId")
    mesh.point_data[GLOBAL_POINT_ID.vtk] = global_ids
    _require_array_hash(
        "canonical GlobalPointId",
        global_ids,
        EXPECTED_TOPOLOGY_HASHES["global_ids"],
        dtype="<i8",
    )
    raw_smile = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
    _require_array_hash(
        "prepared raw Smile",
        raw_smile,
        EXPECTED_TOPOLOGY_HASHES["raw_smile"],
        dtype="<f8",
    )
    target = _normalized_smile(raw_smile)
    loss_mask = np.asarray(mesh.point_data["SmileLossMask"], dtype=bool)
    target_finite = np.asarray(mesh.point_data["TargetFinite"], dtype=bool)
    _require_array_hash(
        "prepared TargetFinite",
        target_finite,
        EXPECTED_TOPOLOGY_HASHES["target_finite"],
        dtype="u1",
    )
    if not np.array_equal(target_finite, np.isfinite(raw_smile).all(axis=1)):
        raise ValueError("prepared TargetFinite differs from finite raw Smile rows")
    if np.any(loss_mask & ~target_finite):
        raise ValueError("SmileLossMask contains a non-finite raw Smile row")
    _require_array_hash(
        "prepared smile target",
        target,
        EXPECTED_TOPOLOGY_HASHES["target"],
        dtype="<f8",
    )
    _require_array_hash(
        "prepared smile loss mask",
        loss_mask,
        EXPECTED_TOPOLOGY_HASHES["loss_mask"],
        dtype="u1",
    )
    target_rms = float(np.linalg.norm(target[loss_mask]) / math.sqrt(loss_mask.sum()))
    if not math.isclose(
        target_rms, EXPECTED_TARGET_RMS_M, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(f"prepared smile target RMS changed: {target_rms}")
    activation_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    if int(activation_mask.sum()) != EXPECTED_ACTIVE_TETS:
        raise ValueError("prepared anatomy active-tet count changed")
    _require_array_hash(
        "prepared activation mask",
        activation_mask,
        EXPECTED_TOPOLOGY_HASHES["activation_mask"],
        dtype="u1",
    )
    return mesh


def _validate_hard_fixed_mesh(
    mesh: pv.UnstructuredGrid, cut_ids: np.ndarray
) -> dict[str, Any]:
    is_fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)
    fixed_value = np.asarray(mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64)
    cut = np.asarray(mesh.point_data["ArtificialCutIncident"], dtype=bool)
    if is_fixed.shape != (EXPECTED_POINTS,):
        raise ValueError("hard-fixed IsFixed shape changed")
    if fixed_mask.shape != (EXPECTED_POINTS, 3):
        raise ValueError("hard-fixed FixedMask shape changed")
    if fixed_value.shape != (EXPECTED_POINTS, 3):
        raise ValueError("hard-fixed FixedValue shape changed")
    if int(is_fixed.sum()) != EXPECTED_MODEL_FIXED_VERTICES:
        raise ValueError("hard-fixed vertex count changed")
    if int(fixed_mask.sum()) != EXPECTED_MODEL_FIXED_DOFS:
        raise ValueError("hard-fixed DoF count changed")
    if int(cut.sum()) != EXPECTED_CUT_VERTICES:
        raise ValueError("artificial-cut vertex count changed")
    if not np.array_equal(np.flatnonzero(cut), np.sort(cut_ids)):
        raise ValueError("configured cut IDs differ from persisted cut marker")
    if not np.array_equal(fixed_mask, np.repeat(is_fixed[:, None], 3, axis=1)):
        raise ValueError("hard-fixed mask differs from IsFixed")
    if not np.array_equal(fixed_value, np.zeros_like(fixed_value)):
        raise ValueError("hard-fixed values are not exact zero")
    for name, values, dtype in (
        ("is_fixed", is_fixed, "u1"),
        ("fixed_mask", fixed_mask, "u1"),
        ("fixed_value", fixed_value, "<f8"),
        ("cut", cut, "u1"),
    ):
        _require_array_hash(
            f"hard-fixed {name}",
            values,
            EXPECTED_TOPOLOGY_HASHES[name],
            dtype=dtype,
        )
    return {
        "fixed_vertices": int(is_fixed.sum()),
        "fixed_dofs": int(fixed_mask.sum()),
        "cut_vertices": int(cut.sum()),
        "is_fixed_sha256_u1": EXPECTED_TOPOLOGY_HASHES["is_fixed"],
        "fixed_mask_sha256_u1": EXPECTED_TOPOLOGY_HASHES["fixed_mask"],
        "fixed_value_sha256_le_f8": EXPECTED_TOPOLOGY_HASHES["fixed_value"],
        "cut_sha256_u1": EXPECTED_TOPOLOGY_HASHES["cut"],
    }


def _validate_frame(
    *,
    spec: FrameSpec,
    base_mesh: pv.UnstructuredGrid,
    hard_fixed_mesh: pv.UnstructuredGrid,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require_identity(f"{spec.source_case} history", spec.history)
    _require_identity(f"{spec.source_case} trace", spec.trace)
    trace = _read_jsonl(spec.trace.path)
    trace_row = trace[spec.step]
    if trace_row.get("forward/success") is not True:
        raise ValueError(f"{spec.source_case}@{spec.step} old forward was unsuccessful")
    if trace_row.get("adjoint/success") is not True:
        raise ValueError(
            f"{spec.source_case}@{spec.step} source adjoint was unsuccessful"
        )
    if not math.isclose(
        float(trace_row.get("target/error_rms")),
        spec.expected_target_error_rms_m,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} trace target RMS changed")
    history = TemporalHistory.open(spec.source_case, spec.history.path)
    frame = history.frame(spec.step)
    if frame.n_points != EXPECTED_POINTS or frame.n_cells != EXPECTED_TETS:
        raise ValueError(f"{spec.source_case}@{spec.step} dimensions changed")
    if not np.array_equal(frame.points, base_mesh.points):
        raise ValueError(f"{spec.source_case}@{spec.step} rest points changed")
    if not np.array_equal(frame.cells, base_mesh.cells) or not np.array_equal(
        frame.celltypes, base_mesh.celltypes
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} topology changed")
    target = np.asarray(frame.point_data["TargetDisplacement"], dtype=np.float64)
    loss_mask = np.asarray(frame.point_data["LossMask"], dtype=bool)
    if not np.array_equal(
        target,
        _normalized_smile(np.asarray(base_mesh.point_data["Smile"], dtype=np.float64)),
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} target changed")
    if not np.array_equal(
        loss_mask, np.asarray(base_mesh.point_data["SmileLossMask"], dtype=bool)
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} loss mask changed")
    displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
    activation = np.asarray(frame.cell_data["ActivationInv"], dtype=np.float64)
    recovered = np.asarray(frame.cell_data["RecoveredActivationInv"], dtype=np.float64)
    if displacement.shape != (EXPECTED_POINTS, 3):
        raise ValueError(f"{spec.source_case}@{spec.step} displacement shape changed")
    if activation.shape != (EXPECTED_TETS, 6):
        raise ValueError(f"{spec.source_case}@{spec.step} activation shape changed")
    _require_finite_array(f"{spec.source_case} displacement", displacement)
    _require_finite_array(f"{spec.source_case} activation", activation)
    if not np.array_equal(activation, recovered):
        raise ValueError(f"{spec.source_case}@{spec.step} activation fields differ")
    _require_array_hash(
        f"{spec.source_case}@{spec.step} displacement",
        displacement,
        spec.displacement_sha256,
        dtype="<f8",
    )
    _require_array_hash(
        f"{spec.source_case}@{spec.step} activation",
        activation,
        spec.activation_sha256,
        dtype="<f8",
    )
    for field, expected, dtype in (
        (GLOBAL_POINT_ID.vtk, EXPECTED_TOPOLOGY_HASHES["global_ids"], "<i8"),
        ("TargetDisplacement", EXPECTED_TOPOLOGY_HASHES["target"], "<f8"),
        ("LossMask", EXPECTED_TOPOLOGY_HASHES["loss_mask"], "u1"),
        ("IsFixed", EXPECTED_TOPOLOGY_HASHES["is_fixed"], "u1"),
        (FIXED_MASK.vtk, EXPECTED_TOPOLOGY_HASHES["fixed_mask"], "u1"),
        (FIXED_VALUE.vtk, EXPECTED_TOPOLOGY_HASHES["fixed_value"], "<f8"),
        ("ArtificialCutIncident", EXPECTED_TOPOLOGY_HASHES["cut"], "u1"),
    ):
        _require_array_hash(
            f"{spec.source_case}@{spec.step} {field}",
            np.asarray(frame.point_data[field]),
            expected,
            dtype=dtype,
        )
        if field in {"TargetDisplacement", "LossMask"}:
            # These history-only aliases were already checked above against
            # normalized Smile and SmileLossMask on the prepared mesh.
            continue
        if not np.array_equal(
            np.asarray(frame.point_data[field]),
            np.asarray(hard_fixed_mesh.point_data[field]),
        ):
            raise ValueError(f"{spec.source_case}@{spec.step} {field} boundary changed")
    frame_activation_mask = np.asarray(frame.cell_data["ActivationMask"], dtype=bool)
    _require_array_hash(
        f"{spec.source_case}@{spec.step} ActivationMask",
        frame_activation_mask,
        EXPECTED_TOPOLOGY_HASHES["activation_mask"],
        dtype="u1",
    )
    if not np.array_equal(
        frame_activation_mask,
        np.asarray(base_mesh.cell_data["ActivationMask"], dtype=bool),
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} ActivationMask changed")
    if not np.array_equal(
        activation[~frame_activation_mask],
        np.zeros_like(activation[~frame_activation_mask]),
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} activation escaped its mask")
    fixed = np.asarray(frame.point_data["IsFixed"], dtype=bool)
    cut = np.asarray(frame.point_data["ArtificialCutIncident"], dtype=bool)
    if not np.array_equal(displacement[fixed], np.zeros_like(displacement[fixed])):
        raise ValueError(f"{spec.source_case}@{spec.step} violates exact fixed zero")
    if not np.array_equal(displacement[cut], np.zeros_like(displacement[cut])):
        raise ValueError(f"{spec.source_case}@{spec.step} violates exact cut zero")
    error_rms = float(
        np.linalg.norm((displacement - target)[loss_mask])
        / math.sqrt(int(loss_mask.sum()))
    )
    if not math.isclose(
        error_rms,
        spec.expected_target_error_rms_m,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{spec.source_case}@{spec.step} frame target RMS changed")
    return (
        activation.copy(),
        displacement.copy(),
        {
            "source_case": spec.source_case,
            "step": spec.step,
            "history": {"path": str(spec.history.path), **_identity_dict(spec.history)},
            "trace": {"path": str(spec.trace.path), **_identity_dict(spec.trace)},
            "activation_sha256_le_f8": spec.activation_sha256,
            "displacement_sha256_le_f8": spec.displacement_sha256,
            "target_error_rms_m": error_rms,
            "source_forward_success": True,
            "source_adjoint_success": True,
        },
    )


def _triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area_vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    norm = np.linalg.norm(area_vectors, axis=1)
    if np.any(norm <= np.finfo(np.float64).eps) or not np.isfinite(norm).all():
        raise ValueError("surface contains a degenerate or non-finite triangle")
    return area_vectors, 0.5 * norm, area_vectors / norm[:, None]


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _interior_edge_adjacency(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_triangles = triangles.shape[0]
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    owners = np.tile(np.arange(n_triangles, dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    ordered_owners = owners[order]
    starts = np.r_[
        0,
        1 + np.flatnonzero(np.any(ordered_edges[1:] != ordered_edges[:-1], axis=1)),
    ]
    counts = np.diff(np.r_[starts, ordered_edges.shape[0]])
    if np.any(counts > 2):
        raise ValueError("skin surface contains a non-manifold edge")
    interior_starts = starts[counts == 2]
    interior_edges = ordered_edges[interior_starts]
    tri_0 = ordered_owners[interior_starts]
    tri_1 = ordered_owners[interior_starts + 1]
    edge_weight = np.linalg.norm(
        points[interior_edges[:, 1]] - points[interior_edges[:, 0]], axis=1
    )
    if not np.isfinite(edge_weight).all() or np.any(edge_weight <= 0.0):
        raise ValueError("skin surface contains an invalid interior edge length")
    return interior_edges, tri_0, tri_1, edge_weight


def _vertex_normals(
    points: np.ndarray, triangles: np.ndarray, area_vectors: np.ndarray
) -> np.ndarray:
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, triangles[:, local], area_vectors)
    norm = np.linalg.norm(normals, axis=1)
    used = np.unique(triangles)
    if np.any(norm[used] <= np.finfo(np.float64).eps):
        raise ValueError("target surface contains a vertex with undefined normal")
    normals[used] /= norm[used, None]
    return normals


def _encoded_tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    if encoded.size != 5 * mesh.n_cells:
        raise ValueError("prepared anatomy is not pure tetrahedral")
    encoded = encoded.reshape(-1, 5)
    if not np.all(encoded[:, 0] == 4):
        raise ValueError("prepared anatomy contains a non-tetrahedron")
    return encoded[:, 1:].copy()


def _six_volume(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,ij->i",
        points[tets[:, 1]] - points[tets[:, 0]],
        np.cross(
            points[tets[:, 2]] - points[tets[:, 0]],
            points[tets[:, 3]] - points[tets[:, 0]],
        ),
    )


def _build_metric_basis(
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    target: np.ndarray,
    loss_mask: np.ndarray,
) -> tuple[MetricBasis, dict[str, Any]]:
    skin_mesh_ids = _map_global_ids(mesh, skin)
    triangles = _triangle_faces(skin)
    rest_points = np.asarray(skin.points, dtype=np.float64)
    rest_vectors, rest_area, _ = _triangle_geometry(rest_points, triangles)
    target_points = rest_points + target[skin_mesh_ids]
    target_vectors, _, target_normals = _triangle_geometry(target_points, triangles)
    target_vertex_normals = _vertex_normals(target_points, triangles, target_vectors)
    full_edges = _unique_edges(triangles)
    _, full_tri_0, full_tri_1, full_edge_weight = _interior_edge_adjacency(
        rest_points, triangles
    )
    if full_tri_0.size != EXPECTED_FULL_INTERIOR_EDGES:
        raise ValueError(
            f"full interior edge count changed: {full_tri_0.size} != "
            f"{EXPECTED_FULL_INTERIOR_EDGES}"
        )
    full_target_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i", target_normals[full_tri_0], target_normals[full_tri_1]
            ),
            -1.0,
            1.0,
        )
    )
    raw_ratio = np.asarray(skin.cell_data["TargetRestAreaRatio"], dtype=np.float64)
    contraction_mask = raw_ratio < 1.0
    if int(contraction_mask.sum()) != EXPECTED_CONTRACTING_TRIANGLES:
        raise ValueError("strict raw-R contraction triangle count changed")
    _require_array_hash(
        "strict raw-R contraction mask",
        contraction_mask,
        EXPECTED_CONTRACTION_MASK_SHA256,
        dtype="u1",
    )
    contraction_selector = contraction_mask[full_tri_0] & contraction_mask[full_tri_1]
    contraction_tri_0 = full_tri_0[contraction_selector]
    contraction_tri_1 = full_tri_1[contraction_selector]
    contraction_edge_weight = full_edge_weight[contraction_selector]
    if contraction_tri_0.size != EXPECTED_CONTRACTION_INTERIOR_EDGES:
        raise ValueError("contraction-only interior edge count changed")
    _require_array_hash(
        "contraction triangle-pair",
        np.column_stack((contraction_tri_0, contraction_tri_1)),
        EXPECTED_CONTRACTION_TRIANGLE_PAIR_SHA256,
        dtype="<i8",
    )
    _require_array_hash(
        "contraction interior edge weights",
        contraction_edge_weight,
        EXPECTED_CONTRACTION_EDGE_WEIGHT_SHA256,
        dtype="<f8",
    )
    contraction_target_dihedral = np.arccos(
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
    expansion_mask = np.asarray(skin.cell_data["ExpandingTriangle"], dtype=bool)
    expansion_selector = expansion_mask[full_tri_0] & expansion_mask[full_tri_1]
    expansion_tri_0 = full_tri_0[expansion_selector]
    expansion_tri_1 = full_tri_1[expansion_selector]
    expansion_edge_weight = full_edge_weight[expansion_selector]
    if expansion_tri_0.size != EXPECTED_EXPANSION_INTERIOR_EDGES:
        raise ValueError("expansion-only interior edge count changed")
    _require_array_hash(
        "expansion triangle-pair",
        np.column_stack((expansion_tri_0, expansion_tri_1)),
        EXPECTED_EXPANSION_TRIANGLE_PAIR_SHA256,
        dtype="<i8",
    )
    _require_array_hash(
        "expansion interior edge weights",
        expansion_edge_weight,
        EXPECTED_EXPANSION_EDGE_WEIGHT_SHA256,
        dtype="<f8",
    )
    expansion_target_dihedral = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i",
                target_normals[expansion_tri_0],
                target_normals[expansion_tri_1],
            ),
            -1.0,
            1.0,
        )
    )
    expansion_edges = _unique_edges(triangles[expansion_mask])
    expansion_vertices = np.unique(triangles[expansion_mask])
    if expansion_edges.shape[0] != EXPECTED_EXPANSION_GRAPH_EDGES:
        raise ValueError("expansion graph edge count changed")
    if expansion_vertices.size != EXPECTED_EXPANSION_VERTICES:
        raise ValueError("expansion graph vertex count changed")
    _require_array_hash(
        "expansion graph edges",
        expansion_edges,
        EXPECTED_EXPANSION_GRAPH_EDGE_SHA256,
        dtype="<i8",
    )
    target_rms = float(np.linalg.norm(target[loss_mask]) / math.sqrt(loss_mask.sum()))
    expansion_target_rms = float(
        np.linalg.norm(target[skin_mesh_ids[expansion_vertices]])
        / math.sqrt(expansion_vertices.size)
    )
    if not math.isclose(
        target_rms, EXPECTED_TARGET_RMS_M, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("metric target RMS changed")
    if not math.isclose(
        expansion_target_rms,
        EXPECTED_EXPANSION_TARGET_RMS_M,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("expansion target RMS changed")
    tets = _encoded_tetrahedra(mesh)
    rest_six_volume = _six_volume(np.asarray(mesh.points), tets)
    if np.any(np.abs(rest_six_volume) <= np.finfo(np.float64).eps):
        raise ValueError("prepared anatomy contains a zero-volume tetrahedron")
    basis = MetricBasis(
        base_points=np.asarray(mesh.points, dtype=np.float64).copy(),
        cells=np.asarray(mesh.cells).copy(),
        celltypes=np.asarray(mesh.celltypes).copy(),
        global_ids=np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk]).copy(),
        target=target.copy(),
        loss_mask=loss_mask.copy(),
        target_rms=target_rms,
        is_fixed=np.asarray(mesh.point_data["IsFixed"], dtype=bool).copy(),
        fixed_mask=np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool).copy(),
        fixed_value=np.asarray(mesh.point_data[FIXED_VALUE.vtk]).copy(),
        cut_mask=np.asarray(
            mesh.point_data["ArtificialCutIncident"], dtype=bool
        ).copy(),
        activation_mask=np.asarray(mesh.cell_data["ActivationMask"], dtype=bool).copy(),
        skin=skin.copy(deep=True),
        skin_mesh_ids=skin_mesh_ids,
        triangles=triangles,
        full_edges=full_edges,
        expansion_edges=expansion_edges,
        expansion_vertices=expansion_vertices,
        rest_area=rest_area,
        target_vertex_normals=target_vertex_normals,
        full_tri_0=full_tri_0,
        full_tri_1=full_tri_1,
        full_target_dihedral=full_target_dihedral,
        full_edge_weight=full_edge_weight,
        contraction_tri_0=contraction_tri_0,
        contraction_tri_1=contraction_tri_1,
        contraction_target_dihedral=contraction_target_dihedral,
        contraction_edge_weight=contraction_edge_weight,
        expansion_tri_0=expansion_tri_0,
        expansion_tri_1=expansion_tri_1,
        expansion_target_dihedral=expansion_target_dihedral,
        expansion_edge_weight=expansion_edge_weight,
        expansion_mask=expansion_mask,
        tets=tets,
        rest_six_volume=rest_six_volume,
        rest_area_vectors=rest_vectors,
        rest_area_vector_norm=np.linalg.norm(rest_vectors, axis=1),
    )
    return basis, {
        "full_interior_edges": int(full_tri_0.size),
        "full_graph_edges": int(full_edges.shape[0]),
        "contracting_triangles": int(contraction_mask.sum()),
        "contraction_interior_edges": int(contraction_tri_0.size),
        "expansion_triangles": int(expansion_mask.sum()),
        "expansion_interior_edges": int(expansion_tri_0.size),
        "expansion_graph_edges": int(expansion_edges.shape[0]),
        "expansion_vertices": int(expansion_vertices.size),
        "target_rms_m": target_rms,
        "expansion_target_rms_m": expansion_target_rms,
        "D_full_definition": (
            "rest-edge-length-weighted RMS of deformed minus target unsigned "
            "dihedral over all 44,495 IsFace interior edges"
        ),
        "D_exp_definition": (
            "same D restricted to interior edges whose two incident triangles "
            "satisfy raw TargetRestAreaRatio > 1"
        ),
        "D_contraction_definition": (
            "primary historical-comparison D restricted to interior edges whose "
            "two incident triangles satisfy raw TargetRestAreaRatio < 1"
        ),
        "L_full_definition": (
            "RMS over all IsFace vertices of scalar graph Laplacian of "
            "target-normal displacement residual"
        ),
        "L_exp_definition": (
            "RMS over expansion-incident vertices after computing one scalar "
            "Laplacian on the full IsFace graph"
        ),
        "Q95_exp_definition": (
            "95th percentile of abs(the same full-IsFace scalar Laplacian) "
            "restricted to expansion-incident vertices"
        ),
    }


def _scalar_graph_laplacian(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
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


def _weighted_dihedral_rms(
    normals: np.ndarray,
    tri_0: np.ndarray,
    tri_1: np.ndarray,
    target_dihedral: np.ndarray,
    weights: np.ndarray,
) -> float:
    deformed = np.arccos(
        np.clip(np.einsum("ij,ij->i", normals[tri_0], normals[tri_1]), -1.0, 1.0)
    )
    delta = deformed - target_dihedral
    return float(np.sqrt(np.dot(weights, np.square(delta)) / weights.sum()))


def _screen_metrics(
    basis: MetricBasis, displacement: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if displacement.shape != basis.target.shape:
        raise ValueError("screen displacement shape changed")
    _require_finite_array("screen displacement", displacement)
    residual = displacement - basis.target
    error_rms = float(
        np.linalg.norm(residual[basis.loss_mask])
        / math.sqrt(int(basis.loss_mask.sum()))
    )
    skin_displacement = displacement[basis.skin_mesh_ids]
    skin_residual = residual[basis.skin_mesh_ids]
    deformed = np.asarray(basis.skin.points, dtype=np.float64) + skin_displacement
    deformed_vectors, _, deformed_normals = _triangle_geometry(
        deformed, basis.triangles
    )
    d_full = _weighted_dihedral_rms(
        deformed_normals,
        basis.full_tri_0,
        basis.full_tri_1,
        basis.full_target_dihedral,
        basis.full_edge_weight,
    )
    d_contraction = _weighted_dihedral_rms(
        deformed_normals,
        basis.contraction_tri_0,
        basis.contraction_tri_1,
        basis.contraction_target_dihedral,
        basis.contraction_edge_weight,
    )
    d_exp = _weighted_dihedral_rms(
        deformed_normals,
        basis.expansion_tri_0,
        basis.expansion_tri_1,
        basis.expansion_target_dihedral,
        basis.expansion_edge_weight,
    )
    residual_normal = np.einsum("ij,ij->i", skin_residual, basis.target_vertex_normals)
    full_laplacian = _scalar_graph_laplacian(residual_normal, basis.full_edges)
    l_full = float(np.linalg.norm(full_laplacian) / math.sqrt(full_laplacian.size))
    exp_values = full_laplacian[basis.expansion_vertices]
    l_exp = float(np.linalg.norm(exp_values) / math.sqrt(exp_values.size))
    q95_exp = float(np.quantile(np.abs(exp_values), 0.95))
    deformed_six_volume = _six_volume(basis.base_points + displacement, basis.tets)
    det_f = deformed_six_volume / basis.rest_six_volume
    signed_normal_ratio = np.einsum(
        "ij,ij->i", deformed_vectors, basis.rest_area_vectors
    ) / np.square(basis.rest_area_vector_norm)
    _require_finite_array("det(F)", det_f)
    _require_finite_array("signed normal ratio", signed_normal_ratio)
    edge_jump = np.linalg.norm(
        skin_residual[basis.full_edges[:, 0]] - skin_residual[basis.full_edges[:, 1]],
        axis=1,
    )
    expansion_edge_jump = np.linalg.norm(
        skin_residual[basis.expansion_edges[:, 0]]
        - skin_residual[basis.expansion_edges[:, 1]],
        axis=1,
    )
    folded = signed_normal_ratio <= 0.0
    inverted = det_f <= 0.0
    metrics = {
        "target/error_rms_m": error_rms,
        "target/error_rms_mm": 1.0e3 * error_rms,
        "target/error_rms_fraction_of_target": error_rms / basis.target_rms,
        "metric/D_full_rad": d_full,
        "metric/D_full_deg": math.degrees(d_full),
        "metric/D_contraction_rad": d_contraction,
        "metric/D_contraction_deg": math.degrees(d_contraction),
        "metric/L_full_m": l_full,
        "metric/L_full_mm": 1.0e3 * l_full,
        "metric/D_exp_rad": d_exp,
        "metric/D_exp_deg": math.degrees(d_exp),
        "metric/L_exp_m": l_exp,
        "metric/L_exp_mm": 1.0e3 * l_exp,
        "metric/Q95_exp_m": q95_exp,
        "metric/Q95_exp_mm": 1.0e3 * q95_exp,
        "warning/edge_jump_max_m": float(edge_jump.max()),
        "warning/edge_jump_q95_m": float(np.quantile(edge_jump, 0.95)),
        "warning/expansion_edge_jump_max_m": float(expansion_edge_jump.max()),
        "warning/expansion_edge_jump_q95_m": float(
            np.quantile(expansion_edge_jump, 0.95)
        ),
        "warning/edge_jump_policy": "record-and-visual-review; never a hard reject",
        "warning/inverted_tets": int(inverted.sum()),
        "warning/inverted_tet_fraction": float(inverted.mean()),
        "warning/detF_min": float(det_f.min()),
        "warning/detF_q001": float(np.quantile(det_f, 0.001)),
        "warning/isface_folded_triangles": int(folded.sum()),
        "warning/isface_folded_triangle_fraction": float(folded.mean()),
        "warning/isface_folded_rest_area_fraction": float(
            basis.rest_area[folded].sum() / basis.rest_area.sum()
        ),
        "warning/expansion_folded_triangles": int(
            np.sum(folded & basis.expansion_mask)
        ),
        "warning/expansion_folded_triangle_fraction": float(
            np.mean(folded[basis.expansion_mask])
        ),
        "warning/fold_inversion_policy": (
            "diagnostic and visual-review only; small visually imperceptible "
            "folds/inversions are not a veto"
        ),
    }
    if not all(
        math.isfinite(value)
        for value in metrics.values()
        if isinstance(value, int | float)
    ):
        raise ValueError("screen metrics contain a non-finite value")
    return metrics, {
        "ResidualNormal": residual_normal,
        "ResidualNormalLaplacianFull": full_laplacian,
    }


def _validate_old_metric_pin(source_case: str, metrics: dict[str, Any]) -> None:
    expected = EXPECTED_OLD_METRICS[source_case]
    for key, value in expected.items():
        actual = metrics.get(key)
        if isinstance(value, int):
            if actual != value:
                raise ValueError(f"{source_case} pinned old {key} changed")
        elif not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{source_case} pinned old {key} changed: {actual} != {value}"
            )


def _solver_contract(probe: ModuleType) -> dict[str, Any]:
    expected = {
        "FORWARD_MAX_STEPS": 5000,
        "FORWARD_ATOL": 1.0e-10,
        "FORWARD_RTOL": 5.0e-4,
        "APONEUROSIS_E": 0.1,
        "APONEUROSIS_NU": 0.35,
        "FAT_E": 0.003,
        "FAT_NU": 0.49,
        "MUSCLE_E": 0.03,
        "MUSCLE_NU": 0.49,
        "SKIN_THICKNESS": 0.001,
    }
    changed = {
        name: (getattr(probe, name, None), value)
        for name, value in expected.items()
        if getattr(probe, name, None) != value
    }
    if changed:
        raise ValueError(f"pinned solver/material constants changed: {changed}")
    if torch.get_default_dtype() != torch.float64:
        raise ValueError("formal forward screen requires torch.float64")
    return {
        "builder": "reviewed 15-forward-domain-conversion-probe._build_forward",
        "optimizer": "Forward.default_optimizer",
        "constants": expected,
        "reviewed_builder_identity": _identity_dict(INPUTS["reviewed_probe"]),
        "runtime_config_identity": _identity_dict(INPUTS["runtime_config"]),
        "runtime_forward_identity": _identity_dict(INPUTS["runtime_forward"]),
        "koiter_identity": _identity_dict(INPUTS["koiter"]),
        "torch_default_dtype": str(torch.get_default_dtype()),
    }


def _volume_lambda_mu(young: float, poisson: float) -> tuple[float, float]:
    return (
        young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        young / (2.0 * (1.0 + poisson)),
    )


def _validate_live_materials(
    *,
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    materials: dict[str, dict[str, torch.Tensor]],
    probe: ModuleType,
    expected_muscle_activation: np.ndarray | None,
) -> dict[str, Any]:
    if set(materials) != {"aponeurosis", "fat", "muscle", "skin"}:
        raise ValueError(f"live potential set changed: {sorted(materials)}")
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    metrics: dict[str, Any] = {}
    for name, young, poisson, fraction_name in (
        (
            "aponeurosis",
            probe.APONEUROSIS_E,
            probe.APONEUROSIS_NU,
            probe.APONEUROSIS_FRACTION,
        ),
        ("fat", probe.FAT_E, probe.FAT_NU, probe.FAT_FRACTION),
        ("muscle", probe.MUSCLE_E, probe.MUSCLE_NU, probe.MUSCLE_FRACTION),
    ):
        expected_lambda, expected_mu = _volume_lambda_mu(young, poisson)
        live_lambda = np.asarray(probe.to_numpy(materials[name][LAMBDA.value]))
        live_mu = np.asarray(probe.to_numpy(materials[name][MU.value]))
        live_dv = np.asarray(probe.to_numpy(materials[name]["dV"]))
        fraction = np.asarray(mesh.cell_data[fraction_name], dtype=np.float64)
        integrated_dv = live_dv.reshape(mesh.n_cells, -1).sum(axis=1)
        if not np.allclose(live_lambda, expected_lambda, rtol=1.0e-13, atol=1.0e-14):
            raise ValueError(f"live {name} lambda changed")
        if not np.allclose(live_mu, expected_mu, rtol=1.0e-13, atol=1.0e-14):
            raise ValueError(f"live {name} mu changed")
        if not np.allclose(
            integrated_dv, volume * fraction, rtol=1.0e-10, atol=1.0e-18
        ):
            raise ValueError(f"live {name} integration weights changed")
        metrics.update(
            {
                f"volume/{name}/lambda_MPa": float(expected_lambda),
                f"volume/{name}/mu_MPa": float(expected_mu),
                f"volume/{name}/fraction_sha256_le_f8": _raw_sha256(
                    fraction, dtype="<f8"
                ),
                f"volume/{name}/integrated_dV_sha256_le_f8": _raw_sha256(
                    integrated_dv, dtype="<f8"
                ),
            }
        )
    for field in (LAMBDA, MU, FRACTION, ACTIVATION_INV):
        expected = np.asarray(skin.cell_data[field.vtk], dtype=np.float64)
        live = np.asarray(probe.to_numpy(materials["skin"][field.value]))
        if live.shape != expected.shape or not np.array_equal(live, expected):
            raise ValueError(f"live Koiter {field.vtk} differs from clean v2 VTP")
        metrics[f"koiter/{field.vtk}_sha256_le_f8"] = _raw_sha256(live, dtype="<f8")
    live_activation = np.asarray(
        probe.to_numpy(materials["muscle"][ACTIVATION_INV.value]), dtype=np.float64
    )
    expected_activation = (
        np.zeros_like(live_activation)
        if expected_muscle_activation is None
        else expected_muscle_activation
    )
    if live_activation.shape != (mesh.n_cells, 6) or not np.array_equal(
        live_activation, expected_activation
    ):
        raise ValueError("live muscle activation differs from the fixed contract")
    metrics["muscle/ActivationInv_sha256_le_f8"] = _raw_sha256(
        live_activation, dtype="<f8"
    )
    metrics["exact_live_material_readback"] = True
    return metrics


def _result_readback_gate(
    *,
    path: Path,
    basis: MetricBasis,
    displacement: np.ndarray,
    activation: np.ndarray,
) -> dict[str, Any]:
    loaded = pv.read(path)
    if not isinstance(loaded, pv.UnstructuredGrid):
        raise TypeError(f"forward result readback is not UnstructuredGrid: {path}")
    if loaded.n_points != EXPECTED_POINTS or loaded.n_cells != EXPECTED_TETS:
        raise ValueError(f"forward result dimensions changed during write: {path}")
    if not np.array_equal(loaded.points, basis.base_points):
        raise ValueError(f"forward result rest points changed during write: {path}")
    if not np.array_equal(loaded.cells, basis.cells) or not np.array_equal(
        loaded.celltypes, basis.celltypes
    ):
        raise ValueError(f"forward result topology changed during write: {path}")
    expected_point_arrays = {
        GLOBAL_POINT_ID.vtk: (basis.global_ids, "<i8"),
        "TargetDisplacement": (basis.target, "<f8"),
        "LossMask": (basis.loss_mask, "u1"),
        "Displacement": (displacement, "<f8"),
        "IsFixed": (basis.is_fixed, "u1"),
        FIXED_MASK.vtk: (basis.fixed_mask, "u1"),
        FIXED_VALUE.vtk: (basis.fixed_value, "<f8"),
        "ArtificialCutIncident": (basis.cut_mask, "u1"),
    }
    hashes: dict[str, str] = {}
    for name, (expected, dtype) in expected_point_arrays.items():
        actual = np.asarray(loaded.point_data[name])
        if not np.array_equal(actual, expected):
            raise ValueError(f"forward result {name} readback changed: {path}")
        hashes[f"point/{name}"] = _raw_sha256(actual, dtype=dtype)
    for name in (ACTIVATION_INV.vtk, "RecoveredActivationInv"):
        actual = np.asarray(loaded.cell_data[name], dtype=np.float64)
        if not np.array_equal(actual, activation):
            raise ValueError(f"forward result {name} readback changed: {path}")
        hashes[f"cell/{name}"] = _raw_sha256(actual, dtype="<f8")
    return {
        "strict_readback": True,
        "array_hashes": hashes,
        "file_identity": _file_identity(path),
    }


def _write_result_atomic(
    *,
    path: Path,
    result: pv.UnstructuredGrid,
    basis: MetricBasis,
    displacement: np.ndarray,
    activation: np.ndarray,
) -> dict[str, Any]:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale result or temporary: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    melon.save(result, temporary)
    temporary_gate = _result_readback_gate(
        path=temporary,
        basis=basis,
        displacement=displacement,
        activation=activation,
    )
    temporary.replace(path)
    final_gate = _result_readback_gate(
        path=path,
        basis=basis,
        displacement=displacement,
        activation=activation,
    )
    if temporary_gate["file_identity"] != final_gate["file_identity"]:
        raise RuntimeError("forward result identity changed during atomic rename")
    cherries.log_output(path)
    return {"path": str(path), **final_gate}


def _solve_case(
    *,
    cfg: Config,
    probe: ModuleType,
    base_mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    basis: MetricBasis,
    spec: FrameSpec,
    seed: Seed,
    activation: np.ndarray,
    old_displacement: np.ndarray,
    old_metrics: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    case_id = _case_id(spec, seed)
    case_mesh = base_mesh.copy(deep=True)
    cut_ids, cut_metrics = probe._configure_cut_boundary(
        case_mesh, pv.read(DRIVER_SKIN), "hard-fixed"
    )
    boundary_metrics = _validate_hard_fixed_mesh(case_mesh, cut_ids)
    probe._validate_domain(case_mesh, skin, "isface")
    forward, materials = probe._build_forward(case_mesh, skin.copy(deep=True))
    if forward.model.n_fixed != EXPECTED_MODEL_FIXED_DOFS:
        raise ValueError(f"{case_id} solver fixed-DoF count changed")
    before_materials = _validate_live_materials(
        mesh=case_mesh,
        skin=skin,
        materials=materials,
        probe=probe,
        expected_muscle_activation=None,
    )
    materials["muscle"][ACTIVATION_INV.value] = torch.as_tensor(
        activation,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    forward.model.set_materials(materials)
    injected_materials = _validate_live_materials(
        mesh=case_mesh,
        skin=skin,
        materials=forward.model.get_materials(),
        probe=probe,
        expected_muscle_activation=activation,
    )
    seed_displacement = (
        np.zeros_like(old_displacement) if seed == "zero" else old_displacement.copy()
    )
    if not np.array_equal(
        seed_displacement[basis.is_fixed],
        np.zeros_like(seed_displacement[basis.is_fixed]),
    ):
        raise ValueError(f"{case_id} seed violates exact hard-fixed zero")
    if not np.array_equal(
        seed_displacement[basis.cut_mask],
        np.zeros_like(seed_displacement[basis.cut_mask]),
    ):
        raise ValueError(f"{case_id} seed violates exact cut zero")
    forward.model.update(
        forward.state,
        torch.as_tensor(
            seed_displacement,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        ),
    )
    live_seed_displacement = np.asarray(
        probe.to_numpy(forward.state.u), dtype=np.float64
    ).copy()
    if not np.array_equal(live_seed_displacement, seed_displacement):
        raise ValueError(f"{case_id} solver state does not exactly match its seed")
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed_s = time.perf_counter() - started
    solver_metrics = probe.forward_solution_metrics(solution)
    if cfg.require_solver_success and solver_metrics.get("forward/success") is not True:
        raise RuntimeError(f"{case_id} forward solve failed: {solver_metrics}")
    displacement = np.asarray(probe.to_numpy(forward.state.u), dtype=np.float64).copy()
    if not np.array_equal(
        displacement[basis.is_fixed], np.zeros_like(displacement[basis.is_fixed])
    ):
        raise ValueError(f"{case_id} final displacement violates fixed zero")
    if not np.array_equal(
        displacement[basis.cut_mask], np.zeros_like(displacement[basis.cut_mask])
    ):
        raise ValueError(f"{case_id} final displacement violates cut zero")
    after_materials = _validate_live_materials(
        mesh=case_mesh,
        skin=skin,
        materials=forward.model.get_materials(),
        probe=probe,
        expected_muscle_activation=activation,
    )
    if injected_materials != after_materials:
        raise ValueError(f"{case_id} material content changed during forward solve")
    metrics, diagnostic_arrays = _screen_metrics(basis, displacement)
    comparison_keys = (
        "target/error_rms_m",
        "metric/D_full_deg",
        "metric/D_contraction_deg",
        "metric/L_full_mm",
        "metric/D_exp_deg",
        "metric/L_exp_mm",
        "metric/Q95_exp_mm",
    )
    comparison = {f"comparison/old/{key}": old_metrics[key] for key in comparison_keys}
    comparison.update(
        {
            f"comparison/new_minus_old/{key}": metrics[key] - old_metrics[key]
            for key in comparison_keys
        }
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "material_case": spec.material_case,
        "source_case": spec.source_case,
        "source_step": spec.step,
        "seed": seed,
        "status": "ok",
        "fixed_activation": True,
        "new_inverse": False,
        "adjoint_or_backward_executed": False,
        "activation_sha256_le_f8": spec.activation_sha256,
        "initial_displacement_sha256_le_f8": _raw_sha256(
            live_seed_displacement, dtype="<f8"
        ),
        "initial_displacement_exact_readback": True,
        "forward/elapsed_s": elapsed_s,
        **cut_metrics,
        **{f"boundary/{key}": value for key, value in boundary_metrics.items()},
        **{f"prebuild/{key}": value for key, value in before_materials.items()},
        **{f"postinject/{key}": value for key, value in injected_materials.items()},
        **solver_metrics,
        **metrics,
        **comparison,
    }
    result = probe.make_result_mesh(
        case_mesh,
        basis.target,
        basis.loss_mask,
        displacement,
        activation,
        {
            key: value
            for key, value in row.items()
            if isinstance(value, int | float | bool)
        },
    )
    expansion_vertex = np.zeros(result.n_points, dtype=np.int8)
    expansion_vertex[basis.skin_mesh_ids[basis.expansion_vertices]] = 1
    result.point_data["ExpansionVertex"] = expansion_vertex
    for name, values in diagnostic_arrays.items():
        full = np.zeros(result.n_points, dtype=np.float64)
        full[basis.skin_mesh_ids] = values
        result.point_data[name] = full
    artifact = _write_result_atomic(
        path=_case_path(spec, seed),
        result=result,
        basis=basis,
        displacement=displacement,
        activation=activation,
    )
    row["artifact"] = artifact
    return row, displacement


def _input_specs(
    skins: dict[str, IdentitySpec], live_producer: IdentitySpec
) -> dict[str, IdentitySpec]:
    specs = {
        name: spec for name, spec in INPUTS.items() if name != "prepare_implementation"
    }
    specs["prepare_implementation_live"] = live_producer
    for spec in FRAME_SPECS:
        specs[f"{spec.source_case}_history"] = spec.history
        specs[f"{spec.source_case}_trace"] = spec.trace
    for case_id, identity in skins.items():
        specs[f"clean_v2_{case_id}_skin"] = identity
    return specs


def _snapshot_identities(
    specs: dict[str, IdentitySpec], *, phase: str
) -> dict[str, dict[str, Any]]:
    return {
        name: {"phase": phase, **_require_identity(name, spec)}
        for name, spec in specs.items()
    }


def _post_identity_gate(
    before: dict[str, dict[str, Any]], specs: dict[str, IdentitySpec]
) -> dict[str, dict[str, Any]]:
    after = _snapshot_identities(specs, phase="post")
    for name, before_record in before.items():
        before_identity = {
            "size_bytes": before_record["size_bytes"],
            "sha256": before_record["sha256"],
        }
        after_identity = {
            "size_bytes": after[name]["size_bytes"],
            "sha256": after[name]["sha256"],
        }
        if before_identity != after_identity:
            raise RuntimeError(f"input {name} changed during the forward screen")
    return after


def _branch_summaries(
    *,
    rows: list[dict[str, Any]],
    displacements: dict[str, np.ndarray],
    basis: MetricBasis,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    expansion_mesh_ids = basis.skin_mesh_ids[basis.expansion_vertices]
    for spec in FRAME_SPECS:
        zero_id = _case_id(spec, "zero")
        old_id = _case_id(spec, "old-equilibrium")
        delta = displacements[zero_id] - displacements[old_id]
        loss_delta = float(
            np.linalg.norm(delta[basis.loss_mask])
            / math.sqrt(int(basis.loss_mask.sum()))
        )
        expansion_delta = float(
            np.linalg.norm(delta[expansion_mesh_ids])
            / math.sqrt(expansion_mesh_ids.size)
        )
        by_id = {str(row["case_id"]): row for row in rows}
        metric_delta = {
            f"zero_minus_old/{key}": by_id[zero_id][key] - by_id[old_id][key]
            for key in (
                "target/error_rms_m",
                "metric/D_full_deg",
                "metric/D_contraction_deg",
                "metric/L_full_mm",
                "metric/D_exp_deg",
                "metric/L_exp_mm",
                "metric/Q95_exp_mm",
            )
        }
        summaries.append(
            {
                "material_case": spec.material_case,
                "source_case": spec.source_case,
                "zero_case_id": zero_id,
                "old_equilibrium_case_id": old_id,
                "full_displacement_delta_rms_m": float(
                    np.linalg.norm(delta) / math.sqrt(delta.shape[0])
                ),
                "loss_mask_displacement_delta_rms_m": loss_delta,
                "loss_mask_delta_fraction_of_target_rms": (
                    loss_delta / basis.target_rms
                ),
                "expansion_displacement_delta_rms_m": expansion_delta,
                "expansion_delta_fraction_of_target_rms": (
                    expansion_delta / EXPECTED_EXPANSION_TARGET_RMS_M
                ),
                "policy": (
                    "record branch sensitivity for interpretation; no arbitrary "
                    "dual-seed hard threshold"
                ),
                **metric_delta,
            }
        )
    return summaries


def _table_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| case | source | seed | target RMS (mm) | contraction D (deg) | full D (deg) | full L (mm) | exp D (deg) | exp L (mm) | exp Q95 (mm) | folds | inversions | solver |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["material_case"]),
                    f"{row['source_case']}@{int(row['source_step'])}",
                    str(row["seed"]),
                    f"{float(row['target/error_rms_mm']):.9g}",
                    f"{float(row['metric/D_contraction_deg']):.9g}",
                    f"{float(row['metric/D_full_deg']):.9g}",
                    f"{float(row['metric/L_full_mm']):.9g}",
                    f"{float(row['metric/D_exp_deg']):.9g}",
                    f"{float(row['metric/L_exp_mm']):.9g}",
                    f"{float(row['metric/Q95_exp_mm']):.9g}",
                    str(row["warning/isface_folded_triangles"]),
                    str(row["warning/inverted_tets"]),
                    str(row["forward/success"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale JSON output: {path}")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    if _read_json(temporary) != payload:
        raise RuntimeError(f"strict JSON temporary readback failed: {path}")
    temporary.replace(path)
    if _read_json(path) != payload:
        raise RuntimeError(f"strict JSON final readback failed: {path}")
    cherries.log_output(path)
    return {"path": str(path), **_file_identity(path), "strict_readback": True}


def _write_table_atomic(path: Path, text: str) -> dict[str, Any]:
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale table output: {path}")
    temporary.write_text(text, encoding="utf-8")
    if temporary.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"table temporary readback failed: {path}")
    temporary.replace(path)
    if path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"table final readback failed: {path}")
    cherries.log_output(path)
    return {"path": str(path), **_file_identity(path), "strict_readback": True}


def main(cfg: Config) -> None:
    _validate_config(cfg)
    producer_before = _file_identity(PRODUCER)
    probe = _load_reviewed_probe()
    probe.configure_runtime()
    solver_contract = _solver_contract(probe)

    # The v1 manifest is never read.  Only the independently named and frozen
    # clean-v2 artifact may establish the two skin inputs.
    manifest = _read_json(PREPARED_MANIFEST)
    candidates, skin_identities, live_producer, prepare_provenance = (
        _validate_clean_v2_manifest(manifest)
    )
    specs = _input_specs(skin_identities, live_producer)
    identities_before = _snapshot_identities(specs, phase="pre")
    old_aggregate = _read_json(OLD_AGGREGATE)
    old_aggregate_metrics = _validate_old_aggregate(old_aggregate)

    base_mesh = _load_base_mesh()
    hard_fixed_mesh = base_mesh.copy(deep=True)
    cut_ids, cut_metrics = probe._configure_cut_boundary(
        hard_fixed_mesh, pv.read(DRIVER_SKIN), "hard-fixed"
    )
    boundary_metrics = _validate_hard_fixed_mesh(hard_fixed_mesh, cut_ids)
    skins: dict[str, pv.PolyData] = {}
    skin_metrics: dict[str, Any] = {}
    for case_id in ("HFP1", "HFP0"):
        skins[case_id], skin_metrics[case_id] = _validate_skin(
            mesh=hard_fixed_mesh,
            case_id=case_id,
            candidate=candidates[case_id],
            identity=skin_identities[case_id],
            probe=probe,
        )
    if not np.array_equal(skins["HFP0"].points, skins["HFP1"].points):
        raise ValueError("clean v2 HFP0/HFP1 points differ")
    if not np.array_equal(skins["HFP0"].faces, skins["HFP1"].faces):
        raise ValueError("clean v2 HFP0/HFP1 topology differs")

    frame_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frame_metrics: dict[str, Any] = {}
    for spec in FRAME_SPECS:
        activation, displacement, metrics = _validate_frame(
            spec=spec, base_mesh=base_mesh, hard_fixed_mesh=hard_fixed_mesh
        )
        frame_data[spec.source_case] = (activation, displacement)
        frame_metrics[spec.source_case] = metrics
    target = _normalized_smile(
        np.asarray(hard_fixed_mesh.point_data["Smile"], dtype=np.float64)
    )
    loss_mask = np.asarray(hard_fixed_mesh.point_data["SmileLossMask"], dtype=bool)
    basis, metric_contract = _build_metric_basis(
        hard_fixed_mesh, skins["HFP0"], target, loss_mask
    )
    old_metrics: dict[str, dict[str, Any]] = {}
    for spec in FRAME_SPECS:
        metrics, _ = _screen_metrics(basis, frame_data[spec.source_case][1])
        _validate_old_metric_pin(spec.source_case, metrics)
        old_metrics[spec.source_case] = metrics

    rows: list[dict[str, Any]] = []
    displacements: dict[str, np.ndarray] = {}
    for spec in FRAME_SPECS:
        activation, old_displacement = frame_data[spec.source_case]
        for seed in ("zero", "old-equilibrium"):
            row, displacement = _solve_case(
                cfg=cfg,
                probe=probe,
                base_mesh=base_mesh,
                skin=skins[spec.material_case],
                basis=basis,
                spec=spec,
                seed=seed,
                activation=activation,
                old_displacement=old_displacement,
                old_metrics=old_metrics[spec.source_case],
            )
            rows.append(row)
            displacements[str(row["case_id"])] = displacement
    if tuple(str(row["case_id"]) for row in rows) != EXPECTED_CASE_ORDER:
        raise RuntimeError("four-case output order changed")
    branches = _branch_summaries(rows=rows, displacements=displacements, basis=basis)
    identities_after = _post_identity_gate(identities_before, specs)
    producer_after = _file_identity(PRODUCER)
    if producer_before != producer_after:
        raise RuntimeError("forward-screen producer changed during execution")

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "case_order": list(EXPECTED_CASE_ORDER),
        "n_forward_solves": 4,
        "execution": {
            "fixed_activation_forward_only": True,
            "inverse_executed": False,
            "adjoint_executed": False,
            "backward_executed": False,
            "fresh_forward_per_case": True,
            "seeds": ["exact-zero", "exact-old-corresponding-equilibrium"],
        },
        "approval": {
            "forward_execution_approved_after_static_review": (
                FORWARD_EXECUTION_APPROVED_AFTER_STATIC_REVIEW
            ),
            "inverse_execution_approved": INVERSE_EXECUTION_APPROVED,
            "adjoint_execution_approved": ADJOINT_EXECUTION_APPROVED,
            "backward_execution_approved": BACKWARD_EXECUTION_APPROVED,
        },
        "material_manifest": {
            "path": str(PREPARED_MANIFEST),
            **_file_identity(PREPARED_MANIFEST),
            "design": CLEAN_V2_DESIGN,
            "clean_v2_contract_ready": CLEAN_V2_CONTRACT_READY,
            "rejected_v1_path": str(REJECTED_V1_MANIFEST),
            "rejected_v1_sha256": REJECTED_V1_MANIFEST_SHA256,
            "rejected_v1_read": False,
            "rejection_reason": "stale inherited SkinActivationInvDiag",
        },
        "producer": {"path": str(PRODUCER), **producer_after},
        "material_prepare_provenance": prepare_provenance,
        "solver_contract": solver_contract,
        "old_aggregate": old_aggregate_metrics,
        "input_identities_pre": identities_before,
        "input_identities_post": identities_after,
        "cut_contract": {**cut_metrics, **boundary_metrics},
        "metric_contract": metric_contract,
        "skin_contracts": skin_metrics,
        "source_frames": frame_metrics,
        "old_frame_metrics": old_metrics,
        "cases": rows,
        "dual_seed_deltas": branches,
        "acceptance": {
            "hard_gates": [
                "exact frozen input and producer identities before/after",
                "clean-v2 manifest and every material-array hash",
                "clean-v2 SkinActivationInvDiag exact recertification from ActivationInv",
                "exact topology, target, activation, fixed and cut fields",
                "exact live material readback before and after forward",
                "solver success and finite metrics",
                "strict atomic VTU/JSON/table readback",
            ],
            "warnings_not_vetoes": [
                "small visually imperceptible triangle folds",
                "small tetrahedron inversions",
                "edge-jump diagnostics",
                "dual-seed branch sensitivity",
            ],
        },
    }
    _write_json_atomic(OUTPUT_SUMMARY, aggregate)
    _write_table_atomic(OUTPUT_TABLE, _table_text(rows))
    logger.info("Wrote fixed-activation fat-floor forward screen to %s", OUTPUT_SUMMARY)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
