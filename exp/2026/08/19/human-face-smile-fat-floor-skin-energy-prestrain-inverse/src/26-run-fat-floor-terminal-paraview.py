from __future__ import annotations

# This wrapper prepares four audited IsFace surfaces and delegates every rendered
# pixel and every saved view state to the pinned ParaView 6.1.1 pvbatch executable.
# ruff: noqa: C901, EM101, EM102, TRY003
import hashlib
import json
import logging
import math
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "fat-floor-terminal-native-paraview-comparison-v1"
EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_EXPANDING_TRIANGLES = 16_723
CASE_ORDER = ("H1P0", "HFP0", "H1P1", "HFP1")
VIEW_ORDER = ("front", "30-degree", "mouth", "eye-cheek+x")
MODE_ORDER = ("geometry", "normal-residual")
IMAGE_RESOLUTION = (4000, 3000)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = GROUP_DIR / "data"
OLD_GROUP = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse"
)
OLD_DATA = OLD_GROUP / "data"
OLD_AGGREGATE = OLD_DATA / "20-selective-skin-prestrain-inverse-summary-final.json"
FORMAL_AGGREGATE = DATA_DIR / "20-fat-floor-skin-prestrain-inverse-summary-final.json"
RENDERER = Path(__file__).with_name("25-render-fat-floor-terminal-paraview.py")
PVBATCH = Path("/usr/bin/pvbatch")
BUNDLE_ROOT = DATA_DIR / "26-paraview-fat-floor-terminal"
INPUT_ROOT = BUNDLE_ROOT / "inputs"
PLATE_ROOT = BUNDLE_ROOT / "plates"
CONTRACT = BUNDLE_ROOT / "contract.json"
RECEIPT = DATA_DIR / "26-paraview-fat-floor-terminal-receipt.json"

# The completed formal aggregate and both result VTUs are frozen below.  A
# separate approval edit may later change only this boolean and the renderer
# boolean to True after this pinned-result source passes static review.
PARAVIEW_EXECUTION_APPROVED_AFTER_RESULT_REVIEW = True
EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES: int | None = 270_811
EXPECTED_FORMAL_AGGREGATE_SHA256: str | None = (
    "82d48d6629b7760c0bf6df8fded8fdaae21c5edf7ad525f5965ce51ae2d2f0b2"
)
EXPECTED_HFP0_RESULT_SIZE_BYTES: int | None = 147_659_455
EXPECTED_HFP0_RESULT_SHA256: str | None = (
    "3175b90a4134d69bf159095bb7f1a74f9b67e4f3f23df81a789bae4e638d5fd3"
)
EXPECTED_HFP1_RESULT_SIZE_BYTES: int | None = 147_652_097
EXPECTED_HFP1_RESULT_SHA256: str | None = (
    "f93bf583819048b5d81a674c4f409450e3cd1200e0d3811b3dc98811480d53dd"
)

EXPECTED_RENDERER_PREAPPROVAL_SIZE_BYTES = 11_103
EXPECTED_RENDERER_PREAPPROVAL_SHA256 = (
    "108aec8c00a07d67122d49e701e51bb163f65f6a5d289494330a5efa67e4cb4b"
)
EXPECTED_RENDERER_EXECUTABLE_SIZE_BYTES = 11_102
EXPECTED_RENDERER_EXECUTABLE_SHA256 = (
    "aa4568b78f811d829cded5dc896839c89ea20b9369326dab72dc5fb2375a908b"
)
EXPECTED_PVBATCH_SIZE_BYTES = 18_608
EXPECTED_PVBATCH_SHA256 = (
    "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871"
)


@dataclass(frozen=True)
class IdentitySpec:
    path: Path
    size_bytes: int
    sha256: str


STATIC_INPUTS = {
    "old_aggregate": IdentitySpec(
        OLD_AGGREGATE,
        387_036,
        "cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6",
    ),
    "old_h1p0": IdentitySpec(
        OLD_DATA / "20-h1p0.vtu",
        147_660_525,
        "f12b746850a68e45c40ad6f5ebf4704d9a46f3a173b449aac625c17284a14331",
    ),
    "old_h1p1": IdentitySpec(
        OLD_DATA / "20-h1p1.vtu",
        147_651_733,
        "4554ce82a674402a6d876cc3905cf0dba271cc199cf3482699775d285591c0af",
    ),
    "skin_hfp0": IdentitySpec(
        DATA_DIR / "10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
        1_611_312,
        "8aff20c6ecad328bb436213c7751c25546df3b12a01e3f6e748c24e4b9941f23",
    ),
    "skin_hfp1": IdentitySpec(
        DATA_DIR / "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
        2_072_304,
        "89e0b349b1ba8002bc654325ba2f025c492b6e096c242c4c194ddede72cd117d",
    ),
}


class Config(cherries.BaseConfig):
    input_old_aggregate: Path = cherries.input(OLD_AGGREGATE)
    # This remains a plain Path while the formal producer owns it.  Once complete,
    # its exact identity is recorded in the receipt without copying large results.
    input_formal_aggregate: Path = FORMAL_AGGREGATE
    output_receipt: Path = cherries.output(RECEIPT, mkdir=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_identity(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_identity(label: str, spec: IdentitySpec) -> dict[str, Any]:
    actual = _identity(spec.path)
    expected = {
        "path": str(spec.path.resolve()),
        "size_bytes": spec.size_bytes,
        "sha256": spec.sha256,
    }
    if actual != expected:
        raise ValueError(f"{label} identity changed: {actual} != {expected}")
    return actual


def _raw_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _resolved_inputs() -> dict[str, IdentitySpec]:
    placeholders = {
        "formal aggregate size": EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES,
        "formal aggregate sha256": EXPECTED_FORMAL_AGGREGATE_SHA256,
        "HFP0 result size": EXPECTED_HFP0_RESULT_SIZE_BYTES,
        "HFP0 result sha256": EXPECTED_HFP0_RESULT_SHA256,
        "HFP1 result size": EXPECTED_HFP1_RESULT_SIZE_BYTES,
        "HFP1 result sha256": EXPECTED_HFP1_RESULT_SHA256,
    }
    unresolved = [name for name, value in placeholders.items() if value is None]
    if unresolved:
        raise RuntimeError(
            "NO-GO: formal outputs are not complete and identity-pinned: "
            + ", ".join(unresolved)
        )
    return {
        **STATIC_INPUTS,
        "formal_aggregate": IdentitySpec(
            FORMAL_AGGREGATE,
            int(EXPECTED_FORMAL_AGGREGATE_SIZE_BYTES),
            str(EXPECTED_FORMAL_AGGREGATE_SHA256),
        ),
        "hfp0": IdentitySpec(
            DATA_DIR / "20-hfp0.vtu",
            int(EXPECTED_HFP0_RESULT_SIZE_BYTES),
            str(EXPECTED_HFP0_RESULT_SHA256),
        ),
        "hfp1": IdentitySpec(
            DATA_DIR / "20-hfp1.vtu",
            int(EXPECTED_HFP1_RESULT_SIZE_BYTES),
            str(EXPECTED_HFP1_RESULT_SHA256),
        ),
    }


def _validate_config(cfg: Config) -> dict[str, IdentitySpec]:
    if Path(cfg.input_old_aggregate).resolve() != OLD_AGGREGATE.resolve():
        raise ValueError("old aggregate input cannot be overridden")
    if Path(cfg.input_formal_aggregate).resolve() != FORMAL_AGGREGATE.resolve():
        raise ValueError("formal aggregate input cannot be overridden")
    if Path(cfg.output_receipt).resolve() != RECEIPT.resolve():
        raise ValueError("receipt output cannot be overridden")
    stale = [
        path
        for path in (BUNDLE_ROOT, RECEIPT, _temporary_path(RECEIPT))
        if path.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing stale terminal ParaView outputs: {stale}")
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError(
            "NO-GO: this visualization must run with DEBUG=1/profile=debug"
        )
    specs = _resolved_inputs()
    if not PARAVIEW_EXECUTION_APPROVED_AFTER_RESULT_REVIEW:
        raise RuntimeError(
            "NO-GO: terminal ParaView comparison awaits completed-result review and isolated approval"
        )
    return specs


def _renderer_provenance() -> dict[str, Any]:
    data = RENDERER.read_bytes()
    false_marker = b"PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_RESULT_REVIEW = False"
    true_marker = b"PARAVIEW_RENDER_EXECUTION_APPROVED_AFTER_RESULT_REVIEW = True"
    if data.count(true_marker) != 1 or data.count(false_marker) != 0:
        raise RuntimeError("reviewed ParaView renderer is not uniquely source-approved")
    live = _bytes_identity(data)
    expected_live = {
        "size_bytes": EXPECTED_RENDERER_EXECUTABLE_SIZE_BYTES,
        "sha256": EXPECTED_RENDERER_EXECUTABLE_SHA256,
    }
    if live != expected_live:
        raise ValueError("approved ParaView renderer identity changed")
    reconstructed = _bytes_identity(data.replace(true_marker, false_marker))
    expected_reconstructed = {
        "size_bytes": EXPECTED_RENDERER_PREAPPROVAL_SIZE_BYTES,
        "sha256": EXPECTED_RENDERER_PREAPPROVAL_SHA256,
    }
    if reconstructed != expected_reconstructed:
        raise ValueError("renderer approval-only reconstruction failed")
    return {
        "path": str(RENDERER.resolve()),
        "live": live,
        "statically_reviewed_preapproval": expected_reconstructed,
        "approval_only_reconstruction": True,
    }


def _paraview_version() -> tuple[str, dict[str, Any]]:
    identity = _identity(PVBATCH)
    expected = {
        "path": str(PVBATCH.resolve()),
        "size_bytes": EXPECTED_PVBATCH_SIZE_BYTES,
        "sha256": EXPECTED_PVBATCH_SHA256,
    }
    if identity != expected:
        raise ValueError("pvbatch executable identity changed")
    completed = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
    )
    combined = f"{completed.stdout}\n{completed.stderr}".strip()
    if not combined.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}; got {combined!r}"
        )
    return EXPECTED_PARAVIEW_VERSION, identity


def _snapshot_inputs(
    specs: dict[str, IdentitySpec], phase: str
) -> dict[str, dict[str, Any]]:
    return {
        label: {"phase": phase, **_require_identity(label, spec)}
        for label, spec in specs.items()
    }


def _artifact_matches(row: dict[str, Any], spec: IdentitySpec) -> bool:
    return (
        Path(str(row.get("artifact/result_path", ""))).resolve() == spec.path.resolve()
        and int(row.get("artifact/result_size_bytes", -1)) == spec.size_bytes
        and str(row.get("artifact/result_sha256", "")) == spec.sha256
    )


def _validate_aggregate(
    *,
    aggregate: dict[str, Any],
    source: str,
    specs: dict[str, IdentitySpec],
) -> dict[str, dict[str, Any]]:
    if aggregate.get("schema_version") != 1 or aggregate.get("complete") is not True:
        raise ValueError(f"{source} aggregate is incomplete")
    if aggregate.get("stage") != "formal" or aggregate.get("inverse_evaluations") != 41:
        raise ValueError(f"{source} aggregate is not the 41-evaluation formal run")
    if aggregate.get("hard_failures") != []:
        raise ValueError(f"{source} aggregate has hard failures")
    cases = aggregate.get("cases")
    if not isinstance(cases, list):
        raise TypeError(f"{source} aggregate cases changed")
    by_case = {str(row["case_id"]): row for row in cases}
    expected_cases = ("H1P0", "H1P1") if source == "old" else ("HFP0", "HFP1")
    if not set(expected_cases) <= set(by_case):
        raise ValueError(f"{source} aggregate misses terminal comparison cases")
    result_specs = {
        "H1P0": specs["old_h1p0"],
        "H1P1": specs["old_h1p1"],
        "HFP0": specs["hfp0"],
        "HFP1": specs["hfp1"],
    }
    expected_material = {
        "H1P0": ("selective-e000-where-raw-r-gt-1", "p000", False),
        "H1P1": (
            "selective-e000-where-raw-r-gt-1",
            "c020-raw-area-ratio-floor-050",
            True,
        ),
        "HFP0": ("selective-efat003-where-raw-r-gt-1", "p000", False),
        "HFP1": (
            "selective-efat003-where-raw-r-gt-1",
            "c020-raw-area-ratio-floor-050",
            True,
        ),
    }
    for case_id in expected_cases:
        row = by_case[case_id]
        if row.get("status") != "ok" or row.get("inverse/evaluations") != 41:
            raise ValueError(f"{case_id} is not a successful 41-evaluation inverse")
        if int(row.get("best/step", -1)) != 40:
            raise ValueError(
                f"{case_id} result is not terminal step 40; lean result-path comparison is invalid"
            )
        if not _artifact_matches(row, result_specs[case_id]):
            raise ValueError(f"{case_id} result identity changed")
        actual_material = (
            row.get("material/skin_young_modulus_mode"),
            row.get("material/skin_prestrain_mode"),
            row.get("skin/prestrain_enabled"),
        )
        if actual_material != expected_material[case_id]:
            raise ValueError(f"{case_id} material label contract changed")
        for key, value in row.items():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise ValueError(f"{case_id} has non-finite metric {key}")
    return {case_id: by_case[case_id] for case_id in expected_cases}


def _validate_material_contracts(
    *,
    old: dict[str, Any],
    formal: dict[str, Any],
    skin_hfp0: pv.PolyData,
    skin_hfp1: pv.PolyData,
) -> tuple[np.ndarray, np.ndarray]:
    old_policy = old.get("constitutive_policy", {})
    if (
        old_policy.get("skin_domain") != "all-vertex IsFace filtered PolyData"
        or old_policy.get("selective_rule")
        != "E=0 iff raw target/rest area ratio > 1; otherwise E=0.2 MPa"
    ):
        raise ValueError("old E=0 material policy changed")
    formal_policy = formal.get("constitutive_policy", {})
    if formal_policy.get(
        "skin_domain"
    ) != "all-vertex IsFace filtered PolyData" or formal_policy.get(
        "selective_rule"
    ) != (
        "E=fat modulus (0.003 MPa) iff raw target/rest area ratio > 1; "
        "otherwise E=0.2 MPa"
    ):
        raise ValueError("formal E=.003 material policy changed")
    if (
        old_policy.get("c020_rule") != formal_policy.get("c020_rule")
        or formal_policy.get("c020_rule")
        != "rho=0.98^2*clip(raw target/rest area ratio,0.5,1)"
    ):
        raise ValueError("c020 material policy changed")
    if not isinstance(skin_hfp0, pv.PolyData) or not isinstance(skin_hfp1, pv.PolyData):
        raise TypeError("clean-v2 IsFace skins are not PolyData")
    if (
        skin_hfp0.n_points != EXPECTED_SKIN_POINTS
        or skin_hfp0.n_cells != EXPECTED_SKIN_TRIANGLES
        or not np.array_equal(
            np.asarray(skin_hfp0.points), np.asarray(skin_hfp1.points)
        )
        or not np.array_equal(np.asarray(skin_hfp0.faces), np.asarray(skin_hfp1.faces))
    ):
        raise ValueError("clean-v2 HFP0/HFP1 IsFace topology changed")
    expansion = np.asarray(skin_hfp0.cell_data["ExpandingTriangle"], dtype=bool)
    raw_ratio = np.asarray(skin_hfp0.cell_data["TargetRestAreaRatio"], dtype=np.float64)
    clipped = np.clip(raw_ratio, 0.5, 1.0)
    expected_young = np.where(expansion, 0.003, 0.2)
    for label, skin in (("HFP0", skin_hfp0), ("HFP1", skin_hfp1)):
        if not np.array_equal(
            np.asarray(skin.cell_data["ExpandingTriangle"], dtype=bool), expansion
        ) or not np.array_equal(
            np.asarray(skin.cell_data["TargetRestAreaRatio"], dtype=np.float64),
            raw_ratio,
        ):
            raise ValueError(f"{label} expansion field changed")
        if not np.array_equal(
            np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64),
            expected_young,
        ):
            raise ValueError(f"{label} no longer uses the exact .003 MPa floor")
    if int(expansion.sum()) != EXPECTED_EXPANDING_TRIANGLES:
        raise ValueError("expanding triangle count changed")
    zeros = np.zeros(EXPECTED_SKIN_TRIANGLES, dtype=np.float64)
    if not np.array_equal(
        np.asarray(skin_hfp0.cell_data["SkinActivationInvDiag"], dtype=np.float64),
        zeros,
    ):
        raise ValueError("HFP0 is no longer exact p000")
    stress_free = np.square(0.98) * clipped
    diagonal = np.reciprocal(np.sqrt(stress_free)) - 1.0
    if not np.array_equal(
        np.asarray(skin_hfp1.cell_data["StressFreeAreaRatio"], dtype=np.float64),
        stress_free,
    ) or not np.array_equal(
        np.asarray(skin_hfp1.cell_data["SkinActivationInvDiag"], dtype=np.float64),
        diagonal,
    ):
        raise ValueError("HFP1 is no longer exact c020")
    return expansion, raw_ratio


def _load_result(
    *,
    case_id: str,
    row: dict[str, Any],
    spec: IdentitySpec,
    canonical: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    mesh = pv.read(spec.path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"{case_id} result is not an UnstructuredGrid")
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError(f"{case_id} result dimensions changed")
    required = {
        "GlobalPointId",
        "Displacement",
        "TargetDisplacement",
        "LossMask",
        "IsFixed",
        "ArtificialCutIncident",
        "IsLip",
    }
    if not required <= set(mesh.point_data):
        raise KeyError(f"{case_id} result misses required point arrays")
    fields = {
        "points": np.asarray(mesh.points, dtype=np.float64).copy(),
        "global_ids": np.asarray(
            mesh.point_data["GlobalPointId"], dtype=np.int64
        ).copy(),
        "displacement": np.asarray(
            mesh.point_data["Displacement"], dtype=np.float64
        ).copy(),
        "target": np.asarray(
            mesh.point_data["TargetDisplacement"], dtype=np.float64
        ).copy(),
        "loss_mask": np.asarray(mesh.point_data["LossMask"], dtype=bool).copy(),
        "is_fixed": np.asarray(mesh.point_data["IsFixed"], dtype=bool).copy(),
        "cut": np.asarray(mesh.point_data["ArtificialCutIncident"], dtype=bool).copy(),
        "is_lip": np.asarray(mesh.point_data["IsLip"], dtype=bool).copy(),
    }
    for name, values in fields.items():
        if values.shape[0] != EXPECTED_POINTS or not np.isfinite(values).all():
            raise ValueError(f"{case_id} {name} shape or finiteness changed")
    if canonical is not None:
        for name in (
            "points",
            "global_ids",
            "target",
            "loss_mask",
            "is_fixed",
            "cut",
            "is_lip",
        ):
            if not np.array_equal(fields[name], canonical[name]):
                raise ValueError(f"{case_id} canonical {name} changed")
    fixed_displacement = fields["displacement"][fields["is_fixed"]]
    cut_displacement = fields["displacement"][fields["cut"]]
    if not np.array_equal(
        fixed_displacement, np.zeros_like(fixed_displacement)
    ) or not np.array_equal(cut_displacement, np.zeros_like(cut_displacement)):
        raise ValueError(f"{case_id} violates exact fixed/cut zero")
    error_rms = float(
        np.linalg.norm((fields["displacement"] - fields["target"])[fields["loss_mask"]])
        / math.sqrt(int(fields["loss_mask"].sum()))
    )
    if not math.isclose(
        error_rms, float(row["best/error_rms"]), rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(f"{case_id} result/aggregate target RMS changed")
    return fields


def _skin_mapping(skin: pv.PolyData, canonical_ids: np.ndarray) -> np.ndarray:
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if np.unique(skin_ids).size != EXPECTED_SKIN_POINTS:
        raise ValueError("IsFace GlobalPointId values are not unique")
    order = np.argsort(canonical_ids)
    sorted_ids = canonical_ids[order]
    positions = np.searchsorted(sorted_ids, skin_ids)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(
        sorted_ids[positions], skin_ids
    ):
        raise ValueError("IsFace points do not map exactly into the anatomy mesh")
    return order[positions]


def _triangles(skin: pv.PolyData) -> np.ndarray:
    faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if faces.shape != (EXPECTED_SKIN_TRIANGLES, 4) or not np.all(faces[:, 0] == 3):
        raise ValueError("IsFace skin is not exactly canonical triangles")
    return faces[:, 1:].copy()


def _target_vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    if not np.isfinite(vectors).all() or np.any(
        np.linalg.norm(vectors, axis=1) <= np.finfo(np.float64).eps
    ):
        raise ValueError("target IsFace geometry has a degenerate triangle")
    normals = np.zeros_like(points)
    for local in range(3):
        np.add.at(normals, triangles[:, local], vectors)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("target IsFace geometry has an undefined vertex normal")
    return normals / lengths[:, None]


def _bounds_camera(
    points: np.ndarray, *, padding: float = 1.12, aspect: float = 1.35
) -> tuple[list[float], float]:
    low, high = points.min(axis=0), points.max(axis=0)
    focus = 0.5 * (low + high)
    extent = high - low
    scale = padding * 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    if not np.isfinite(focus).all() or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid fixed ParaView camera")
    return focus.tolist(), scale


def _views(
    *, skin: pv.PolyData, skin_points: np.ndarray, is_lip: np.ndarray
) -> dict[str, dict[str, Any]]:
    face_focus, face_scale = _bounds_camera(skin_points)
    if not np.any(is_lip):
        raise ValueError("mapped IsLip camera mask is empty")
    mouth_focus, mouth_scale = _bounds_camera(skin_points[is_lip], padding=1.25)
    names = tuple(str(value) for value in skin.field_data["GroupName"])
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {"EyelidTop", "EyelidBottom", "EyelidOuterTop", "EyelidOuterBottom"}
    eyelid_ids = [index for index, name in enumerate(names) if name in eyelid_names]
    eyelid = np.isin(group_ids, eyelid_ids)
    if not np.any(eyelid):
        raise ValueError("eyelid camera mask is empty")
    one_eye = eyelid & (skin_points[:, 0] >= np.median(skin_points[eyelid, 0]))
    eye_focus, _ = _bounds_camera(skin_points[one_eye])
    eye_focus[1] -= 0.08 * float(np.ptp(skin_points[:, 1]))
    return {
        "front": {
            "direction": [0.0, 0.0, 1.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "30-degree": {
            "direction": [0.5, 0.0, math.sqrt(3.0) / 2.0],
            "focus": face_focus,
            "parallel_scale": face_scale,
        },
        "mouth": {
            "direction": [0.0, 0.0, 1.0],
            "focus": mouth_focus,
            "parallel_scale": mouth_scale,
        },
        "eye-cheek+x": {
            "direction": [0.0, 0.0, 1.0],
            "focus": eye_focus,
            "parallel_scale": 0.24 * float(np.ptp(skin_points[:, 1])),
        },
    }


def _write_surface_atomic(
    *,
    path: Path,
    rest_skin: pv.PolyData,
    triangles: np.ndarray,
    skin_ids: np.ndarray,
    displacement: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    expansion: np.ndarray,
    raw_ratio: np.ndarray,
) -> dict[str, Any]:
    skin_displacement = displacement[skin_ids]
    skin_target = target[skin_ids]
    residual = skin_displacement - skin_target
    normal_residual_mm = 1.0e3 * np.einsum("ij,ij->i", residual, target_normals)
    points = np.asarray(rest_skin.points, dtype=np.float64) + skin_displacement
    arrays = {
        "GlobalPointId": np.asarray(
            rest_skin.point_data["GlobalPointId"], dtype=np.int64
        ),
        "TargetNormalResidualMM": normal_residual_mm,
        "DisplacementMM": 1.0e3 * skin_displacement,
        "TargetDisplacementMM": 1.0e3 * skin_target,
        "ResidualDisplacementMM": 1.0e3 * residual,
    }
    if not np.isfinite(points).all() or any(
        not np.isfinite(values).all() for values in arrays.values()
    ):
        raise ValueError(f"non-finite ParaView surface values for {path}")
    surface = pv.PolyData(
        points, np.column_stack((np.full(triangles.shape[0], 3), triangles))
    )
    for name, values in arrays.items():
        surface.point_data[name] = values
    surface.cell_data["ExpansionRegion"] = np.asarray(expansion, dtype=np.int8)
    surface.cell_data["TargetRestAreaRatio"] = np.asarray(raw_ratio, dtype=np.float64)
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale ParaView surface: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(temporary)
    loaded = pv.read(temporary)
    if (
        not isinstance(loaded, pv.PolyData)
        or loaded.n_points != EXPECTED_SKIN_POINTS
        or loaded.n_cells != EXPECTED_SKIN_TRIANGLES
        or not np.array_equal(np.asarray(loaded.points), points)
    ):
        raise ValueError(f"ParaView surface temporary readback changed: {path}")
    for name, values in arrays.items():
        if not np.array_equal(np.asarray(loaded.point_data[name]), values):
            raise ValueError(f"ParaView surface {name} readback changed: {path}")
    temporary.replace(path)
    final = pv.read(path)
    if not np.array_equal(np.asarray(final.points), points):
        raise ValueError(f"ParaView surface final readback changed: {path}")
    identity = _identity(path)
    return {
        **identity,
        "point_array_hashes": {
            name: _raw_sha256(values, dtype="<i8" if name == "GlobalPointId" else "<f8")
            for name, values in arrays.items()
        },
        "target_normal_residual_mm": normal_residual_mm,
    }


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
    return _identity(path)


def _prepare_contract(
    *,
    old_aggregate: dict[str, Any],
    formal_aggregate: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    specs: dict[str, IdentitySpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    skins = {
        "HFP0": pv.read(specs["skin_hfp0"].path),
        "HFP1": pv.read(specs["skin_hfp1"].path),
    }
    expansion, raw_ratio = _validate_material_contracts(
        old=old_aggregate,
        formal=formal_aggregate,
        skin_hfp0=skins["HFP0"],
        skin_hfp1=skins["HFP1"],
    )
    result_specs = {
        "H1P0": specs["old_h1p0"],
        "HFP0": specs["hfp0"],
        "H1P1": specs["old_h1p1"],
        "HFP1": specs["hfp1"],
    }
    fields: dict[str, dict[str, np.ndarray]] = {}
    canonical: dict[str, np.ndarray] | None = None
    for case_id in CASE_ORDER:
        values = _load_result(
            case_id=case_id,
            row=rows[case_id],
            spec=result_specs[case_id],
            canonical=canonical,
        )
        if canonical is None:
            canonical = {
                name: array for name, array in values.items() if name != "displacement"
            }
        fields[case_id] = values
    if canonical is None:
        raise RuntimeError("no terminal result loaded")
    skin = skins["HFP0"]
    skin_ids = _skin_mapping(skin, canonical["global_ids"])
    if not np.array_equal(np.asarray(skin.points), canonical["points"][skin_ids]):
        raise ValueError("IsFace rest points differ from the anatomy mesh")
    triangles = _triangles(skin)
    target_points = (
        np.asarray(skin.points, dtype=np.float64) + canonical["target"][skin_ids]
    )
    target_normals = _target_vertex_normals(target_points, triangles)
    views = _views(
        skin=skin,
        skin_points=np.asarray(skin.points, dtype=np.float64),
        is_lip=canonical["is_lip"][skin_ids],
    )
    display = {
        "H1P0": ("H1P0 | E=0 expansion", "p000 (no skin prestrain)"),
        "HFP0": ("HFP0 | E=.003 expansion", "p000 (no skin prestrain)"),
        "H1P1": ("H1P1 | E=0 expansion", "c020 skin prestrain"),
        "HFP1": ("HFP1 | E=.003 expansion", "c020 skin prestrain"),
    }
    inputs: dict[str, Any] = {}
    residual_values: list[np.ndarray] = []
    surface_receipts: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        output = INPUT_ROOT / f"{case_id.lower()}.vtp"
        surface = _write_surface_atomic(
            path=output,
            rest_skin=skin,
            triangles=triangles,
            skin_ids=skin_ids,
            displacement=fields[case_id]["displacement"],
            target=canonical["target"],
            target_normals=target_normals,
            expansion=expansion,
            raw_ratio=raw_ratio,
        )
        residual_values.append(surface.pop("target_normal_residual_mm"))
        row = rows[case_id]
        entry = {
            **surface,
            "case_id": case_id,
            "display_label": display[case_id][0],
            "material_label": display[case_id][1],
            "metric_label": (
                f"step={int(row['best/step'])} | err={float(row['best/error_rms_mm']):.3f} mm\n"
                f"Lres={1.0e3 * float(row['bumpiness/residual_laplacian_rms']):.3f} mm | "
                f"folds={int(row['warning/skin_folded_triangles'])}"
            ),
            "metrics": {
                "best_step": int(row["best/step"]),
                "error_rms_mm": float(row["best/error_rms_mm"]),
                "residual_laplacian_rms_mm": 1.0e3
                * float(row["bumpiness/residual_laplacian_rms"]),
                "folded_triangles": int(row["warning/skin_folded_triangles"]),
            },
        }
        inputs[case_id] = entry
        surface_receipts.append(entry)
    residual_limit = max(
        0.25, float(np.quantile(np.abs(np.concatenate(residual_values)), 0.99))
    )
    if not math.isfinite(residual_limit) or residual_limit >= 100.0:
        raise ValueError("shared residual color limit is invalid")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case_order": list(CASE_ORDER),
        "view_order": list(VIEW_ORDER),
        "mode_order": list(MODE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "views": views,
        "inputs": inputs,
        "normal_residual_shared_limit_mm": residual_limit,
        "normal_residual_shared_limit_definition": (
            "max(0.25 mm, pooled absolute 99th percentile) across H1P0/HFP0/H1P1/HFP1"
        ),
        "surface_domain": {
            "name": "IsFace",
            "points": EXPECTED_SKIN_POINTS,
            "triangles": EXPECTED_SKIN_TRIANGLES,
            "expanding_triangles": EXPECTED_EXPANDING_TRIANGLES,
        },
        "comparison": (
            "terminal step-40 result files in exact order H1P0 E=0, HFP0 E=.003, "
            "H1P1 E=0+c020, HFP1 E=.003+c020"
        ),
        "renderer": (
            "ParaView 6.1.1 native geometry and scalar rendering only; no PyVista rendering"
        ),
    }
    contract_identity = _write_json_atomic(CONTRACT, contract)
    return contract, {
        "contract": contract_identity,
        "surfaces": surface_receipts,
        "skin_mapping_sha256_le_i8": _raw_sha256(skin_ids, dtype="<i8"),
        "triangle_sha256_le_i8": _raw_sha256(triangles, dtype="<i8"),
        "target_normal_sha256_le_f8": _raw_sha256(target_normals, dtype="<f8"),
    }


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"{path} is not a valid PNG header")
    return struct.unpack(">II", header[16:24])


def _expected_plate_paths() -> list[Path]:
    return [
        PLATE_ROOT / f"26-paraview-terminal-{mode}.{suffix}"
        for mode in MODE_ORDER
        for suffix in ("png", "pvsm")
    ]


def _validate_outputs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected_inputs = [
        Path(contract["inputs"][case_id]["path"]) for case_id in CASE_ORDER
    ]
    expected = sorted([CONTRACT, *expected_inputs, *_expected_plate_paths()])
    actual = sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file())
    if actual != expected:
        raise ValueError(
            f"terminal ParaView bundle inventory changed: expected {expected}, got {actual}"
        )
    outputs: list[dict[str, Any]] = []
    for path in _expected_plate_paths():
        if path.suffix == ".png":
            if _png_size(path) != IMAGE_RESOLUTION or path.stat().st_size < 100_000:
                raise ValueError(f"{path} PNG size or payload changed")
        else:
            head = path.read_text(encoding="utf-8", errors="strict")[:1024]
            if "ParaView" not in head and "ServerManagerState" not in head:
                raise ValueError(f"{path} is not a recognizable ParaView state")
        outputs.append(_identity(path))
    for case_id in CASE_ORDER:
        item = contract["inputs"][case_id]
        identity = _identity(Path(item["path"]))
        if (
            identity["size_bytes"] != item["size_bytes"]
            or identity["sha256"] != item["sha256"]
        ):
            raise ValueError(f"{case_id} surface changed during rendering")
    if any(".tmp" in path.name for path in actual):
        raise ValueError("terminal ParaView bundle contains a temporary file")
    return outputs


def main(cfg: Config) -> None:
    specs = _validate_config(cfg)
    wrapper_before = _identity(Path(__file__))
    renderer_provenance = _renderer_provenance()
    paraview_version, pvbatch_identity = _paraview_version()
    inputs_before = _snapshot_inputs(specs, "pre")
    old_aggregate = _read_json(cfg.input_old_aggregate)
    formal_aggregate = _read_json(cfg.input_formal_aggregate)
    old_rows = _validate_aggregate(aggregate=old_aggregate, source="old", specs=specs)
    formal_rows = _validate_aggregate(
        aggregate=formal_aggregate, source="formal", specs=specs
    )
    rows = {
        "H1P0": old_rows["H1P0"],
        "HFP0": formal_rows["HFP0"],
        "H1P1": old_rows["H1P1"],
        "HFP1": formal_rows["HFP1"],
    }

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    INPUT_ROOT.mkdir()
    PLATE_ROOT.mkdir()
    contract, preparation = _prepare_contract(
        old_aggregate=old_aggregate,
        formal_aggregate=formal_aggregate,
        rows=rows,
        specs=specs,
    )
    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--contract",
        str(CONTRACT.resolve()),
        "--input-root",
        str(INPUT_ROOT.resolve()),
        "--output-dir",
        str(PLATE_ROOT.resolve()),
    ]
    logger.info("Running pinned terminal ParaView renderer: %s", command)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=GROUP_DIR,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout)
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")
    if (
        _read_json(CONTRACT) != contract
        or _identity(CONTRACT) != preparation["contract"]
    ):
        raise ValueError("terminal ParaView contract changed during rendering")
    outputs = _validate_outputs(contract)
    inputs_after = _snapshot_inputs(specs, "post")
    if _identity(PVBATCH) != pvbatch_identity:
        raise RuntimeError("pvbatch executable changed during rendering")
    wrapper_after = _identity(Path(__file__))
    if wrapper_before != wrapper_after:
        raise RuntimeError("terminal ParaView wrapper changed during execution")
    if _renderer_provenance() != renderer_provenance:
        raise RuntimeError("terminal ParaView renderer changed during execution")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "execution_profile": "debug",
        "execution": {
            "visualization_only": True,
            "forward_executed": False,
            "inverse_executed": False,
            "adjoint_executed": False,
            "backward_executed": False,
            "pyvista_rendering_executed": False,
            "native_paraview_rendering_executed": True,
        },
        "paraview_version": paraview_version,
        "pvbatch": pvbatch_identity,
        "command": command,
        "wrapper": wrapper_after,
        "renderer_provenance": renderer_provenance,
        "inputs_pre": inputs_before,
        "inputs_post": inputs_after,
        "preparation": preparation,
        "case_order": list(CASE_ORDER),
        "view_order": list(VIEW_ORDER),
        "mode_order": list(MODE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "normal_residual_shared_limit_mm": contract["normal_residual_shared_limit_mm"],
        "outputs": outputs,
        "authority": (
            "all terminal geometry and target-normal-residual plates and PVSM states were generated by ParaView 6.1.1; "
            "PyVista was used only for audited IsFace preparation and strict readback"
        ),
    }
    _write_json_atomic(cfg.output_receipt, receipt)
    for path in sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file()):
        cherries.log_output(path)
    logger.info("Wrote terminal native ParaView receipt to %s", cfg.output_receipt)


if __name__ == "__main__":
    # Fail before Cherries can start its default Comet/Git profile.
    if os.environ.get("DEBUG") != "1":
        raise RuntimeError(
            "NO-GO: set DEBUG=1 before Cherries starts; default profile is forbidden"
        )
    cherries.main(main)
