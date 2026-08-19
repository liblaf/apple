from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _reference import (
    GROUP_DIR,
    PREPARED_MESH,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    file_sha256,
)

from liblaf import cherries
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    GLOBAL_POINT_ID,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
INPUT_SCHEMA_VERSION = 2
INPUT_DESIGN = "fixed-activation-domain-conversion-plus-cut-boundary-bracket"
INPUT_SUMMARY_NAME = "15-forward-domain-conversion-probe-summary.json"
INPUT_RESULT_DIR = "15-forward-domain-conversion-probe"
INPUT_PRODUCER = Path(__file__).with_name("15-forward-domain-conversion-probe.py")
INPUT_PRODUCER_SHA256 = (
    "741d3f3db966f8b1e25b389a8734176fb6991a6872e6f8a1a8b875bd3ec5e2f5"
)

CASE_ORDER = (
    "full-3d-replay",
    "full-plane-stress",
    "isface-3d",
    "isface-plane-stress",
    "isface-plane-stress-cut-fixed",
)
SEED_ORDER = ("zero", "old")
CASE_SETUP = {
    "full-3d-replay": ("full", "3d", "current"),
    "full-plane-stress": ("full", "plane-stress", "current"),
    "isface-3d": ("isface", "3d", "current"),
    "isface-plane-stress": ("isface", "plane-stress", "current"),
    "isface-plane-stress-cut-fixed": (
        "isface",
        "plane-stress",
        "hard-fixed",
    ),
}
CUT_BOUNDARY_POLICY = {
    "current": "historical-isfixed",
    "hard-fixed": "all-artificial-cut-incident-vertices-hard-fixed",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "boundary_sensitivity_checks",
    "branch_checks",
    "comparison/branch_stable_all",
    "comparison/causal_contrasts_eligible",
    "complete",
    "design",
    "domain_material_preflight",
    "expensive_inverse_started",
    "fixed_design",
    "historical_control",
    "historical_replay_check",
    "implementation",
    "input_provenance",
    "inverse/eligibility_status",
    "inverse/eligible_to_start",
    "inverse/required_gates",
    "interpretation",
    "new_forward_cases",
    "output_contract",
    "schema_version",
    "topology_provenance",
    "visual_review",
    "decision_rule_before_inverse",
}

REPORT_METRICS = (
    "target/error_rms_fraction_of_target",
    "target/error_rms_mm",
    "target/face_rest_area_weighted_error_rms_m",
    "target/face_target_area_weighted_error_rms_m",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
    "warning/inverted_tets",
    "warning/detF_min",
    "warning/isface_folded_triangles",
    "forward/success",
    "forward/result",
)

CUT_BOUNDARY_NUMERIC_METRICS = (
    "cut_boundary/triangles",
    "cut_boundary/incident_vertices",
    "cut_boundary/preexisting_fixed_vertices",
    "cut_boundary/newly_fixed_vertices",
    "cut_boundary/total_fixed_vertices",
    "cut_boundary/model_total_fixed_dofs",
    "cut_boundary/seed_projection_vertices",
    "cut_boundary/seed_projection_enforced_zero_vertices",
    "cut_boundary/seed_projection_rms_m",
    "cut_boundary/final_displacement_rms_m",
    "cut_boundary/final_displacement_max_abs_m",
)

NUMERIC_FIELD_METRICS = (
    "target/error_rms_fraction_of_target",
    "target/error_rms_m",
    "target/error_rms_mm",
    "target/face_rest_area_weighted_error_rms_m",
    "target/face_target_area_weighted_error_rms_m",
    "bumpiness/contraction_target_relative_dihedral_rms_rad",
    "bumpiness/contraction_target_relative_dihedral_rms_deg",
    "bumpiness/residual_normal_laplacian_rms_m",
    "warning/inverted_tets",
    "warning/inverted_tet_fraction",
    "warning/detF_min",
    "warning/isface_folded_triangles",
    "warning/isface_folded_triangle_fraction",
    *CUT_BOUNDARY_NUMERIC_METRICS,
)

ROW_REQUIRED_KEYS = {
    "activation/fixed_during_forward",
    "activation/new_inverse_solution",
    "activation/source",
    "activation/transferred",
    "artifact/result_path",
    "artifact/result_sha256",
    "artifact/result_size_bytes",
    "artifact/summary_path",
    "case",
    "conversion",
    "cut_boundary",
    "cut_boundary/hard_fixed_is_ground_truth",
    "cut_boundary/incident_global_ids_sha256",
    "cut_boundary/marker",
    "cut_boundary/policy",
    "cut_boundary/triangle_topology_sha256",
    "domain",
    "forward/success",
    "initial_displacement/source",
    "initial_displacement/projection",
    "interpretation",
    "seed",
    "status",
    *REPORT_METRICS,
    *NUMERIC_FIELD_METRICS,
}

BRANCH_REQUIRED_KEYS = {
    "case",
    "conversion",
    "cut_boundary",
    "domain",
    "gate_domains",
    "interpretation_if_false",
    "stable_within_declared_tolerance",
    "tolerance_fraction_of_target_rms",
    "zero_old/full_displacement_delta_rms_m",
    "zero_old/isface_delta_fraction_of_isface_target_rms",
    "zero_old/isface_displacement_delta_rms_m",
    "zero_old/loss_mask_delta_fraction_of_target_rms",
    "zero_old/loss_mask_displacement_delta_rms_m",
    "zero_old_target_error_fraction_delta",
}

BOUNDARY_SENSITIVITY_REQUIRED_KEYS = {
    "seed",
    "reference_case",
    "bracket_case",
    "full_displacement_delta_rms_m",
    "loss_mask_displacement_delta_rms_m",
    "loss_mask_delta_fraction_of_target_rms",
    "isface_displacement_delta_rms_m",
    "isface_delta_fraction_of_isface_target_rms",
    "target_error_fraction_delta",
    "contraction_target_relative_dihedral_rms_deg_delta",
    "residual_normal_laplacian_rms_m_delta",
    "hard_fixed_is_ground_truth",
    "interpretation",
}

CURRENT_BOUNDARY_CASE = "isface-plane-stress"
HARD_FIXED_BOUNDARY_CASE = "isface-plane-stress-cut-fixed"
HARD_FIXED_INTERPRETATION = (
    "sensitivity bracket only; hard-fixed is not an anatomical ground truth"
)
EXPECTED_CUT_TRIANGLES = 13_165
EXPECTED_CUT_INCIDENT_VERTICES = 6_980
EXPECTED_CUT_PREEXISTING_FIXED_VERTICES = 380
EXPECTED_CUT_NEWLY_FIXED_VERTICES = 6_600
EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256 = (
    "8207cda8f9e11dbb4406f683e5ad818a6950e3515ac373719514094fb5b7fe5d"
)
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    "ca39cdc839855be34e75222964a1e5c129dd210e8800c684d7e6d1ce6424f138"
)
EXPECTED_CURRENT_FIXED_DOFS = 81_108
EXPECTED_HARD_FIXED_DOFS = 100_908

JSON_RTOL = 1.0e-10
JSON_ATOL = 1.0e-12
EXPECTED_MESH_SIZE_BYTES = 76_792_914
EXPECTED_MESH_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
EXPECTED_FULL_TRIANGLES = 128_172
EXPECTED_ISFACE_TRIANGLES = 29_899
EXPECTED_FULL_TOPOLOGY_SHA256 = (
    "5cc5e84531e2eb27fd62d8435b31959be4e1a9e60dcc519bcc4f3df506c430b1"
)
EXPECTED_FULL_UNASSIGNED_GROUP_POINTS = 6_000
EXPECTED_FULL_CUT_TRIANGLES = 13_165
EXPECTED_ISFACE_TOPOLOGY_SHA256 = (
    "1cbfa9a27bc26d4bd937d8fae0ab98bf8b07d977f923bcf25681155523cd82c7"
)
EXPECTED_ISFACE_COMPONENTS = 1
EXPECTED_ISFACE_GROUP_NAMES = (
    "EyelidTop",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "LipBottom",
    "LipTop",
    "LipOuterTop",
    "LipOuterBottom",
    "Chin",
    "Face",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_summary: Path = cherries.input(INPUT_SUMMARY_NAME)
    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_skin: Path = cherries.input(SOURCE_SKIN)
    output_json: Path = cherries.output(
        "16-forward-domain-conversion-analysis.json", mkdir=True
    )
    output_csv: Path = cherries.output(
        "16-forward-domain-conversion-analysis.csv", mkdir=True
    )
    output_table: Path = cherries.output(
        "16-forward-domain-conversion-analysis.md", mkdir=True
    )
    output_zero_views: Path = cherries.output(
        "16-forward-domain-conversion-zero-views.png", mkdir=True
    )
    output_old_views: Path = cherries.output(
        "16-forward-domain-conversion-old-views.png", mkdir=True
    )


@dataclass(frozen=True)
class CaseArtifact:
    label: str
    seed: str
    summary: dict[str, Any]
    path: Path
    identity: dict[str, int | str]
    sidecar_path: Path
    sidecar_identity: dict[str, int | str]
    mesh: pv.UnstructuredGrid
    displacement: np.ndarray
    activation: np.ndarray


@dataclass(frozen=True)
class RenderBasis:
    skin: pv.PolyData
    skin_mesh_ids: np.ndarray
    face_triangle_mask: np.ndarray
    target: np.ndarray
    face_focus: np.ndarray
    face_scale: float
    mouth_focus: np.ndarray
    mouth_scale: float
    eye_cheek_focus: np.ndarray
    eye_cheek_scale: float


def _reject_json_constant(token: str) -> None:
    msg = f"non-finite JSON constant {token!r}"
    raise ValueError(msg)


def _validate_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{context} contains a non-finite number"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, context=f"{context}[{index}]")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if not isinstance(value, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    _validate_finite_json(value, context=str(path))
    return value


def _require_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def _require_close(actual: Any, expected: Any, *, context: str) -> None:
    if isinstance(expected, bool | int):
        _require_equal(actual, expected, context=context)
        return
    if not math.isclose(
        float(actual), float(expected), rel_tol=JSON_RTOL, abs_tol=JSON_ATOL
    ):
        msg = f"{context}: expected {expected!r}, got {actual!r}"
        raise ValueError(msg)


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing input artifact: {path}"
        raise FileNotFoundError(msg)
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _field_scalar(mesh: pv.UnstructuredGrid, name: str) -> int | float | bool:
    if name not in mesh.field_data:
        msg = f"result field_data is missing {name!r}"
        raise KeyError(msg)
    values = np.asarray(mesh.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = f"result field_data[{name!r}] is not one finite scalar"
        raise ValueError(msg)
    return values[0].item()


def _validate_config(cfg: Config) -> None:
    expected_data = GROUP_DIR / "data"
    _require_equal(
        cfg.input_summary.resolve(),
        (expected_data / INPUT_SUMMARY_NAME).resolve(),
        context="probe summary path",
    )
    _require_equal(
        cfg.input_mesh.resolve(), PREPARED_MESH.resolve(), context="prepared mesh path"
    )
    _require_equal(
        cfg.input_skin.resolve(), SOURCE_SKIN.resolve(), context="canonical skin path"
    )
    expected_outputs = (
        expected_data / "16-forward-domain-conversion-analysis.json",
        expected_data / "16-forward-domain-conversion-analysis.csv",
        expected_data / "16-forward-domain-conversion-analysis.md",
        expected_data / "16-forward-domain-conversion-zero-views.png",
        expected_data / "16-forward-domain-conversion-old-views.png",
    )
    actual_outputs = (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_zero_views,
        cfg.output_old_views,
    )
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        _require_equal(actual.resolve(), expected.resolve(), context="analysis output")
    if len({path.resolve() for path in actual_outputs}) != len(actual_outputs):
        msg = "analysis output paths must be distinct"
        raise ValueError(msg)
    stale = [str(path) for path in actual_outputs if path.exists()]
    if stale:
        msg = f"refusing to overwrite existing analysis outputs: {stale}"
        raise FileExistsError(msg)


def _validate_topology_provenance(summary: dict[str, Any]) -> None:
    topology = summary["topology_provenance"]
    if not isinstance(topology, dict) or set(topology) != {
        "prepared_volume",
        "historical_full_skin",
        "artificial_cut_boundary_bracket",
        "isface_roi_derivation",
        "runtime_gate",
    }:
        msg = "topology_provenance schema changed"
        raise ValueError(msg)
    prepared = topology["prepared_volume"]
    full = topology["historical_full_skin"]
    bracket = topology["artificial_cut_boundary_bracket"]
    isface = topology["isface_roi_derivation"]
    if set(prepared) != {"path", "size_bytes", "sha256", "role"}:
        msg = "prepared-volume topology provenance schema changed"
        raise ValueError(msg)
    if set(full) != {
        "path",
        "size_bytes",
        "sha256",
        "role",
        "n_triangles",
        "canonical_global_triangle_sha256",
        "artificial_cut",
    }:
        msg = "historical-full-skin topology provenance schema changed"
        raise ValueError(msg)
    if set(isface) != {
        "source",
        "selection",
        "global_point_id_mapping",
        "n_triangles",
        "canonical_global_triangle_sha256",
        "expected_components",
        "allowed_group_names",
    }:
        msg = "IsFace topology provenance schema changed"
        raise ValueError(msg)
    for actual, expected, context in (
        (Path(str(prepared["path"])).resolve(), PREPARED_MESH.resolve(), "volume path"),
        (prepared["size_bytes"], EXPECTED_MESH_SIZE_BYTES, "volume size"),
        (prepared["sha256"], EXPECTED_MESH_SHA256, "volume SHA-256"),
        (Path(str(full["path"])).resolve(), SOURCE_SKIN.resolve(), "skin path"),
        (full["size_bytes"], SOURCE_SKIN_SIZE_BYTES, "skin size"),
        (full["sha256"], SOURCE_SKIN_SHA256, "skin SHA-256"),
        (
            full["role"],
            "pinned historical full extracted-boundary topology for the "
            "domain-by-conversion causal control",
            "skin role",
        ),
        (full["n_triangles"], EXPECTED_FULL_TRIANGLES, "full triangle count"),
        (
            full["canonical_global_triangle_sha256"],
            EXPECTED_FULL_TOPOLOGY_SHA256,
            "full topology SHA-256",
        ),
        (isface["source"], "pinned historical_full_skin", "IsFace source"),
        (isface["n_triangles"], EXPECTED_ISFACE_TRIANGLES, "IsFace triangle count"),
        (
            isface["canonical_global_triangle_sha256"],
            EXPECTED_ISFACE_TOPOLOGY_SHA256,
            "IsFace topology SHA-256",
        ),
        (
            isface["expected_components"],
            EXPECTED_ISFACE_COMPONENTS,
            "IsFace component count",
        ),
        (
            isface["allowed_group_names"],
            list(EXPECTED_ISFACE_GROUP_NAMES),
            "IsFace group names",
        ),
    ):
        _require_equal(actual, expected, context=f"topology provenance {context}")
    artificial_cut = full["artificial_cut"]
    if set(artificial_cut) != {
        "marker",
        "unassigned_points",
        "triangles_touching_unassigned_points",
        "canonical_global_triangle_sha256",
        "incident_vertices",
        "incident_global_ids_sha256",
        "preexisting_fixed_vertices",
        "newly_fixed_vertices_in_bracket",
        "policy",
    }:
        msg = "artificial-cut topology provenance schema changed"
        raise ValueError(msg)
    _require_equal(
        artificial_cut["marker"],
        "mapped point GroupId == -1",
        context="artificial-cut marker",
    )
    _require_equal(
        artificial_cut["unassigned_points"],
        EXPECTED_FULL_UNASSIGNED_GROUP_POINTS,
        context="artificial-cut unassigned points",
    )
    _require_equal(
        artificial_cut["triangles_touching_unassigned_points"],
        EXPECTED_FULL_CUT_TRIANGLES,
        context="artificial-cut triangle count",
    )
    for key, expected in (
        (
            "canonical_global_triangle_sha256",
            EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256,
        ),
        ("incident_vertices", EXPECTED_CUT_INCIDENT_VERTICES),
        (
            "incident_global_ids_sha256",
            EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256,
        ),
        (
            "preexisting_fixed_vertices",
            EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
        ),
        ("newly_fixed_vertices_in_bracket", EXPECTED_CUT_NEWLY_FIXED_VERTICES),
        (
            "policy",
            "intentional full-boundary diagnostic; never admitted to the IsFace "
            "membrane ROI",
        ),
    ):
        _require_equal(artificial_cut[key], expected, context=f"artificial-cut {key}")
    expected_bracket = {
        "reference_case": CURRENT_BOUNDARY_CASE,
        "bracket_case": HARD_FIXED_BOUNDARY_CASE,
        "reference_policy": CUT_BOUNDARY_POLICY["current"],
        "bracket_policy": CUT_BOUNDARY_POLICY["hard-fixed"],
        "bracket_fixed_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "bracket_newly_fixed_vertices": EXPECTED_CUT_NEWLY_FIXED_VERTICES,
        "fixed_value_m": [0.0, 0.0, 0.0],
        "historical_seed_projection": (
            "measure and zero the 6,600 newly constrained vertices; enforce exact "
            "zero on all 6,980 cut-incident vertices"
        ),
        "hard_fixed_is_ground_truth": False,
        "interpretation": (
            "boundary-condition sensitivity bracket only; it is not an anatomical "
            "boundary claim or an inverse-eligibility gate"
        ),
    }
    _require_equal(
        bracket,
        expected_bracket,
        context="artificial-cut boundary bracket provenance",
    )
    identities = summary["input_provenance"]["identities"]
    _require_equal(
        {key: prepared[key] for key in ("size_bytes", "sha256")},
        identities["mesh"],
        context="volume topology/input identity binding",
    )
    _require_equal(
        {key: full[key] for key in ("size_bytes", "sha256")},
        identities["skin"],
        context="skin topology/input identity binding",
    )
    _require_equal(
        isface["global_point_id_mapping"],
        summary["input_provenance"]["mesh/global_point_id_source"],
        context="IsFace GlobalPointId provenance",
    )
    if not isinstance(topology["runtime_gate"], str) or not topology["runtime_gate"]:
        msg = "topology runtime gate must be a non-empty description"
        raise ValueError(msg)


def _validate_summary(  # noqa: C901, PLR0912, PLR0915
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    _require_equal(
        set(summary), EXPECTED_TOP_LEVEL_KEYS, context="probe summary top-level schema"
    )
    for key, expected in (
        ("schema_version", INPUT_SCHEMA_VERSION),
        ("complete", True),
        ("design", INPUT_DESIGN),
        ("expensive_inverse_started", False),
    ):
        _require_equal(summary[key], expected, context=f"probe summary {key}")

    rows = summary["new_forward_cases"]
    if not isinstance(rows, list):
        msg = "new_forward_cases must be a list"
        raise TypeError(msg)
    expected_pairs = [(case, seed) for case in CASE_ORDER for seed in SEED_ORDER]
    actual_pairs = [(row.get("case"), row.get("seed")) for row in rows]
    _require_equal(actual_pairs, expected_pairs, context="forward case/seed order")
    for row in rows:
        missing = ROW_REQUIRED_KEYS - set(row)
        if missing:
            msg = f"{row.get('case')}/{row.get('seed')} row lacks {sorted(missing)}"
            raise ValueError(msg)
        label = str(row["case"])
        seed = str(row["seed"])
        domain, conversion, cut_boundary = CASE_SETUP[label]
        for key, expected in (
            ("domain", domain),
            ("conversion", conversion),
            ("cut_boundary", cut_boundary),
            ("cut_boundary/policy", CUT_BOUNDARY_POLICY[cut_boundary]),
            (
                "cut_boundary/marker",
                "source skin triangle touches mapped GroupId=-1 vertex",
            ),
            (
                "cut_boundary/triangle_topology_sha256",
                EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256,
            ),
            (
                "cut_boundary/incident_global_ids_sha256",
                EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256,
            ),
            ("cut_boundary/hard_fixed_is_ground_truth", False),
            ("status", "ok"),
            ("activation/transferred", True),
            ("activation/new_inverse_solution", False),
            ("activation/fixed_during_forward", True),
            ("forward/success", True),
        ):
            _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
        for key, expected in (
            ("cut_boundary/triangles", EXPECTED_CUT_TRIANGLES),
            ("cut_boundary/incident_vertices", EXPECTED_CUT_INCIDENT_VERTICES),
            (
                "cut_boundary/preexisting_fixed_vertices",
                EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
            ),
        ):
            _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
        if cut_boundary == "hard-fixed":
            for key, expected in (
                (
                    "cut_boundary/newly_fixed_vertices",
                    EXPECTED_CUT_NEWLY_FIXED_VERTICES,
                ),
                (
                    "cut_boundary/total_fixed_vertices",
                    EXPECTED_CUT_INCIDENT_VERTICES,
                ),
                ("cut_boundary/model_total_fixed_dofs", EXPECTED_HARD_FIXED_DOFS),
            ):
                _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
            expected_projection = (
                "zero-on-newly-hard-fixed-artificial-cut-vertices"
                if seed == "old"
                else "not-required-seed-is-exact-zero"
            )
            _require_equal(
                row["initial_displacement/projection"],
                expected_projection,
                context=f"{label}/{seed} initial_displacement/projection",
            )
            for key in (
                "cut_boundary/final_displacement_rms_m",
                "cut_boundary/final_displacement_max_abs_m",
            ):
                _require_equal(row[key], 0.0, context=f"{label}/{seed} {key}")
        else:
            for key, expected in (
                ("cut_boundary/newly_fixed_vertices", 0),
                (
                    "cut_boundary/total_fixed_vertices",
                    EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
                ),
                ("cut_boundary/model_total_fixed_dofs", EXPECTED_CURRENT_FIXED_DOFS),
                ("cut_boundary/seed_projection_vertices", 0),
                ("cut_boundary/seed_projection_enforced_zero_vertices", 0),
                ("cut_boundary/seed_projection_rms_m", 0.0),
                ("initial_displacement/projection", "none"),
            ):
                _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
        if seed == "zero":
            for key, expected in (
                ("cut_boundary/seed_projection_vertices", 0),
                ("cut_boundary/seed_projection_enforced_zero_vertices", 0),
                ("cut_boundary/seed_projection_rms_m", 0.0),
            ):
                _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
        elif cut_boundary == "hard-fixed":
            for key, expected in (
                (
                    "cut_boundary/seed_projection_vertices",
                    EXPECTED_CUT_NEWLY_FIXED_VERTICES,
                ),
                (
                    "cut_boundary/seed_projection_enforced_zero_vertices",
                    EXPECTED_CUT_INCIDENT_VERTICES,
                ),
            ):
                _require_equal(row[key], expected, context=f"{label}/{seed} {key}")
            value = float(row["cut_boundary/seed_projection_rms_m"])
            if not math.isfinite(value) or value <= 0.0:
                msg = f"{label}/{seed} hard-fixed seed projection RMS must be positive"
                raise ValueError(msg)

    branches = summary["branch_checks"]
    if not isinstance(branches, list) or len(branches) != len(CASE_ORDER):
        msg = "branch_checks must contain exactly five setup rows"
        raise ValueError(msg)
    _require_equal(
        [branch.get("case") for branch in branches],
        list(CASE_ORDER),
        context="branch-check case order",
    )
    for branch in branches:
        missing = BRANCH_REQUIRED_KEYS - set(branch)
        if missing:
            msg = f"branch check {branch.get('case')} lacks {sorted(missing)}"
            raise ValueError(msg)
        if not isinstance(branch["stable_within_declared_tolerance"], bool):
            msg = "branch stability must be Boolean"
            raise TypeError(msg)
        label = str(branch["case"])
        for key, expected in zip(
            ("domain", "conversion", "cut_boundary"),
            CASE_SETUP[label],
            strict=True,
        ):
            _require_equal(branch[key], expected, context=f"branch check {label} {key}")
    branch_stable_all = all(
        bool(branch["stable_within_declared_tolerance"]) for branch in branches
    )
    _require_equal(
        summary["comparison/branch_stable_all"],
        branch_stable_all,
        context="aggregate branch-stability gate",
    )

    replay = summary["historical_replay_check"]
    replay_keys = {
        "case",
        "purpose",
        "loss_mask_delta_rms_m",
        "loss_mask_delta_fraction_of_target_rms",
        "isface_delta_rms_m",
        "isface_delta_fraction_of_isface_target_rms",
        "reproduces_historical_control_within_tolerance",
        "tolerance_fraction_of_corresponding_target_rms",
        "eligibility_if_false",
    }
    if not isinstance(replay, dict) or set(replay) != replay_keys:
        msg = "historical_replay_check schema changed"
        raise ValueError(msg)
    _require_equal(replay["case"], "full-3d-replay/old", context="replay case")
    replay_stable = bool(replay["reproduces_historical_control_within_tolerance"])
    causal_eligible = branch_stable_all and replay_stable
    _require_equal(
        summary["comparison/causal_contrasts_eligible"],
        causal_eligible,
        context="numeric causal-contrast eligibility",
    )
    _require_equal(
        summary["inverse/eligible_to_start"],
        expected=False,
        context="inverse start eligibility before visual review",
    )
    expected_status = (
        "pending-visual-review"
        if causal_eligible
        else "not-eligible-numeric-gates-failed"
    )
    _require_equal(
        summary["inverse/eligibility_status"],
        expected_status,
        context="inverse eligibility status",
    )
    required_gates = summary["inverse/required_gates"]
    if not isinstance(required_gates, dict) or set(required_gates) != {
        "historical_full_3d_replay_stable",
        "all_five_setup_seed_pairs_branch_stable",
        "matched_view_visual_review",
        "policy",
    }:
        msg = "inverse/required_gates schema changed"
        raise ValueError(msg)
    for key, expected in (
        ("historical_full_3d_replay_stable", replay_stable),
        ("all_five_setup_seed_pairs_branch_stable", branch_stable_all),
        ("matched_view_visual_review", "pending"),
    ):
        _require_equal(
            required_gates[key], expected, context=f"inverse required gate {key}"
        )
    if not isinstance(required_gates["policy"], str) or not required_gates["policy"]:
        msg = "inverse gate policy must be a non-empty description"
        raise ValueError(msg)
    _validate_topology_provenance(summary)
    output_contract = summary["output_contract"]
    expected_data = GROUP_DIR / "data"
    expected_output_contract = {
        "root": str(expected_data / INPUT_RESULT_DIR),
        "summary_path": str(expected_data / INPUT_SUMMARY_NAME),
        "table_path": str(
            expected_data / "15-forward-domain-conversion-probe-table.md"
        ),
        "case_order": list(CASE_ORDER),
        "seed_order": list(SEED_ORDER),
        "case_layout": "<root>/<case>/<seed>/{result.vtu,forward-summary.json}",
        "expected_result_vtus": 10,
        "expected_forward_sidecars": 10,
        "overwrite_policy": (
            "refuse before input reads or runtime initialization if summary, "
            "table, or result root already exists"
        ),
    }
    _require_equal(
        output_contract, expected_output_contract, context="probe output contract"
    )
    preflight = summary["domain_material_preflight"]
    if not isinstance(preflight, dict):
        msg = "domain_material_preflight must be an object"
        raise TypeError(msg)
    _require_equal(
        list(preflight), list(CASE_ORDER), context="domain-material preflight order"
    )
    for label, row in preflight.items():
        if not isinstance(row, dict):
            msg = f"domain-material preflight {label} must be an object"
            raise TypeError(msg)
        domain, conversion, cut_boundary = CASE_SETUP[label]
        for key, expected in (
            ("domain/name", domain),
            ("skin/conversion", conversion),
            ("cut_boundary/policy", CUT_BOUNDARY_POLICY[cut_boundary]),
            ("cut_boundary/triangles", EXPECTED_CUT_TRIANGLES),
            ("cut_boundary/incident_vertices", EXPECTED_CUT_INCIDENT_VERTICES),
            (
                "cut_boundary/preexisting_fixed_vertices",
                EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
            ),
        ):
            _require_equal(
                row.get(key),
                expected,
                context=f"domain-material preflight {label} {key}",
            )
    fixed = summary["fixed_design"]
    for key, expected in (
        ("new_forward_solves", 10),
        ("new_inverse_solves", 0),
        ("new_setup_count", 4),
        ("changed_probe_labels", list(CASE_ORDER[1:])),
        ("replayed_reference_label", "full-3d-replay"),
        ("seeds", ["zero", "old historical displacement"]),
        (
            "cut_boundary_policy",
            "compare current IsFixed and hard-zero all 6,980 vertices incident "
            "to the 13,165 artificial-cut triangles, seed matched; report as "
            "sensitivity only, never ground truth",
        ),
    ):
        _require_equal(fixed.get(key), expected, context=f"fixed_design {key}")
    sensitivity = summary["boundary_sensitivity_checks"]
    if not isinstance(sensitivity, list) or len(sensitivity) != len(SEED_ORDER):
        msg = "boundary_sensitivity_checks must contain exactly zero and old rows"
        raise ValueError(msg)
    _require_equal(
        [row.get("seed") for row in sensitivity],
        list(SEED_ORDER),
        context="boundary-sensitivity seed order",
    )
    for row in sensitivity:
        _require_equal(
            set(row),
            BOUNDARY_SENSITIVITY_REQUIRED_KEYS,
            context=f"boundary sensitivity {row.get('seed')} schema",
        )
        for key, expected in (
            ("reference_case", CURRENT_BOUNDARY_CASE),
            ("bracket_case", HARD_FIXED_BOUNDARY_CASE),
            ("hard_fixed_is_ground_truth", False),
            ("interpretation", HARD_FIXED_INTERPRETATION),
        ):
            _require_equal(
                row[key], expected, context=f"boundary sensitivity {row['seed']} {key}"
            )
    return rows


def _load_base_inputs(cfg: Config) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    _require_equal(
        _file_identity(cfg.input_mesh),
        {"size_bytes": EXPECTED_MESH_SIZE_BYTES, "sha256": EXPECTED_MESH_SHA256},
        context="prepared mesh identity",
    )
    _require_equal(
        _file_identity(cfg.input_skin),
        {"size_bytes": SOURCE_SKIN_SIZE_BYTES, "sha256": SOURCE_SKIN_SHA256},
        context="canonical skin identity",
    )
    mesh = pv.read(cfg.input_mesh)
    skin = pv.read(cfg.input_skin)
    if not isinstance(mesh, pv.UnstructuredGrid):
        msg = f"prepared mesh read as {type(mesh).__name__}"
        raise TypeError(msg)
    if not isinstance(skin, pv.PolyData):
        msg = f"canonical skin read as {type(skin).__name__}"
        raise TypeError(msg)
    _require_equal(skin.n_cells, EXPECTED_FULL_TRIANGLES, context="skin triangles")
    face = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    _require_equal(int(face.sum()), EXPECTED_ISFACE_TRIANGLES, context="IsFace tris")
    return mesh, skin


def _load_case_artifacts(  # noqa: C901, PLR0912, PLR0915
    cfg: Config,
    rows: list[dict[str, Any]],
    base: pv.UnstructuredGrid,
) -> list[CaseArtifact]:
    artifacts: list[CaseArtifact] = []
    result_root = cfg.input_summary.parent / INPUT_RESULT_DIR
    expected_files = {
        (result_root / label / seed / name).resolve()
        for label in CASE_ORDER
        for seed in SEED_ORDER
        for name in ("result.vtu", "forward-summary.json")
    }
    if not result_root.is_dir():
        msg = f"missing probe result root: {result_root}"
        raise FileNotFoundError(msg)
    actual_files = {path.resolve() for path in result_root.rglob("*") if path.is_file()}
    _require_equal(
        actual_files,
        expected_files,
        context="exact ten-VTU/ten-sidecar result inventory",
    )
    base_ids = np.arange(base.n_points, dtype=np.int64)
    target = np.nan_to_num(
        np.asarray(base.point_data["Smile"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(base.point_data["SmileLossMask"], dtype=bool)
    for row in rows:
        label = str(row["case"])
        seed = str(row["seed"])
        case_dir = cfg.input_summary.parent / INPUT_RESULT_DIR / label / seed
        expected_result = case_dir / "result.vtu"
        expected_sidecar = case_dir / "forward-summary.json"
        _require_equal(
            Path(str(row["artifact/result_path"])).resolve(),
            expected_result.resolve(),
            context=f"{label}/{seed} result path",
        )
        _require_equal(
            Path(str(row["artifact/summary_path"])).resolve(),
            expected_sidecar.resolve(),
            context=f"{label}/{seed} sidecar path",
        )
        sidecar_identity = _file_identity(expected_sidecar)
        sidecar = _read_json(expected_sidecar)
        _require_equal(
            sidecar,
            row,
            context=f"{label}/{seed} sidecar/aggregate-row equality",
        )
        identity = _file_identity(expected_result)
        _require_equal(
            identity,
            {
                "size_bytes": row["artifact/result_size_bytes"],
                "sha256": row["artifact/result_sha256"],
            },
            context=f"{label}/{seed} result identity",
        )
        result = pv.read(expected_result)
        if not isinstance(result, pv.UnstructuredGrid):
            msg = f"{label}/{seed} result is not an UnstructuredGrid"
            raise TypeError(msg)
        for name, actual, expected in (
            ("point count", result.n_points, base.n_points),
            ("cell count", result.n_cells, base.n_cells),
        ):
            _require_equal(actual, expected, context=f"{label}/{seed} {name}")
        for name, actual, expected in (
            ("rest points", result.points, base.points),
            ("connectivity", result.cells, base.cells),
            ("cell types", result.celltypes, base.celltypes),
        ):
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                msg = f"{label}/{seed} {name} changed"
                raise ValueError(msg)
        result_ids = np.asarray(result.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if not np.array_equal(result_ids, base_ids):
            msg = f"{label}/{seed} GlobalPointId mapping changed"
            raise ValueError(msg)
        displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
        stored_target = np.asarray(
            result.point_data["TargetDisplacement"], dtype=np.float64
        )
        stored_mask = np.asarray(result.point_data["LossMask"], dtype=bool)
        if displacement.shape != target.shape or not np.isfinite(displacement).all():
            msg = f"{label}/{seed} displacement is malformed or non-finite"
            raise ValueError(msg)
        if not np.array_equal(stored_target, target) or not np.array_equal(
            stored_mask, loss_mask
        ):
            msg = f"{label}/{seed} target or loss mask changed"
            raise ValueError(msg)
        for name, expected in (
            ("DisplacementError", displacement - target),
            ("DeformedPoint", np.asarray(base.points) + displacement),
            ("TargetPoint", np.asarray(base.points) + target),
        ):
            if not np.array_equal(np.asarray(result.point_data[name]), expected):
                msg = f"{label}/{seed} point field {name} is inconsistent"
                raise ValueError(msg)
        activation = np.asarray(
            result.cell_data["RecoveredActivationInv"], dtype=np.float64
        )
        live_activation = np.asarray(
            result.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
        )
        if (
            activation.shape != (base.n_cells, 6)
            or not np.isfinite(activation).all()
            or not np.array_equal(activation, live_activation)
        ):
            msg = f"{label}/{seed} transferred activation is invalid"
            raise ValueError(msg)
        for metric in NUMERIC_FIELD_METRICS:
            _require_close(
                _field_scalar(result, metric),
                row[metric],
                context=f"{label}/{seed} stored {metric}",
            )
        artifacts.append(
            CaseArtifact(
                label=label,
                seed=seed,
                summary=row,
                path=expected_result,
                identity=identity,
                sidecar_path=expected_sidecar,
                sidecar_identity=sidecar_identity,
                mesh=result,
                displacement=displacement,
                activation=activation,
            )
        )
    activation_digests = {_array_sha256(case.activation) for case in artifacts}
    if len(activation_digests) != 1:
        msg = "the ten forward results do not contain one identical activation"
        raise ValueError(msg)
    return artifacts


def _cut_boundary_mesh_ids(base: pv.UnstructuredGrid, skin: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "canonical skin is not triangle-only"
        raise ValueError(msg)
    triangles = encoded[:, 1:]
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    cut_triangles = np.any(group_ids[triangles] == -1, axis=1)
    _require_equal(
        int(cut_triangles.sum()),
        EXPECTED_CUT_TRIANGLES,
        context="artificial-cut triangle count",
    )
    skin_global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    canonical_cut = np.sort(skin_global_ids[triangles[cut_triangles]], axis=1).astype(
        "<i8", copy=False
    )
    order = np.lexsort((canonical_cut[:, 2], canonical_cut[:, 1], canonical_cut[:, 0]))
    topology_sha256 = hashlib.sha256(
        np.ascontiguousarray(canonical_cut[order]).tobytes()
    ).hexdigest()
    _require_equal(
        topology_sha256,
        EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256,
        context="artificial-cut topology SHA-256",
    )
    cut_local_ids = np.unique(triangles[cut_triangles])
    _require_equal(
        int(cut_local_ids.size),
        EXPECTED_CUT_INCIDENT_VERTICES,
        context="artificial-cut incident-vertex count",
    )
    mesh_ids = np.sort(skin_global_ids[cut_local_ids]).astype(np.int64, copy=False)
    if (
        np.any(mesh_ids < 0)
        or np.any(mesh_ids >= base.n_points)
        or np.unique(mesh_ids).size != mesh_ids.size
    ):
        msg = "artificial-cut GlobalPointId mapping is invalid"
        raise ValueError(msg)
    incident_sha256 = hashlib.sha256(
        np.ascontiguousarray(mesh_ids.astype("<i8", copy=False)).tobytes()
    ).hexdigest()
    _require_equal(
        incident_sha256,
        EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256,
        context="artificial-cut incident GlobalPointId SHA-256",
    )
    fixed = np.asarray(base.point_data["IsFixed"], dtype=bool)
    _require_equal(
        int(fixed[mesh_ids].sum()),
        EXPECTED_CUT_PREEXISTING_FIXED_VERTICES,
        context="artificial-cut preexisting fixed vertices",
    )
    if np.any(np.asarray(base.point_data["IsFace"], dtype=bool)[mesh_ids]):
        msg = "artificial-cut incident vertices overlap IsFace"
        raise ValueError(msg)
    return mesh_ids


def _validate_cut_boundary_artifacts(  # noqa: C901
    artifacts: list[CaseArtifact],
    base: pv.UnstructuredGrid,
    skin: pv.PolyData,
) -> np.ndarray:
    cut_ids = _cut_boundary_mesh_ids(base, skin)
    base_fixed = np.asarray(base.point_data["IsFixed"], dtype=bool)
    if base_fixed.shape != (base.n_points,):
        msg = "prepared IsFixed mask is malformed"
        raise ValueError(msg)
    for case in artifacts:
        cut_boundary = CASE_SETUP[case.label][2]
        expected_fixed = base_fixed.copy()
        if cut_boundary == "hard-fixed":
            expected_fixed[cut_ids] = True
        expected_incident = np.zeros(base.n_points, dtype=np.int8)
        expected_incident[cut_ids] = 1
        expected_preexisting = np.zeros(base.n_points, dtype=np.int8)
        expected_preexisting[cut_ids[base_fixed[cut_ids]]] = 1
        expected_added = np.zeros(base.n_points, dtype=np.int8)
        if cut_boundary == "hard-fixed":
            expected_added[cut_ids[~base_fixed[cut_ids]]] = 1
        for name, expected in (
            ("HistoricalIsFixed", base_fixed.astype(np.int8)),
            ("ArtificialCutIncident", expected_incident),
            ("CutBoundaryPreexistingFixed", expected_preexisting),
            ("CutBoundaryAddedFixed", expected_added),
            ("IsFixed", expected_fixed),
        ):
            actual = np.asarray(case.mesh.point_data[name])
            if not np.array_equal(actual, expected):
                msg = f"{case.label}/{case.seed} point field {name} changed"
                raise ValueError(msg)
        result_fixed = np.asarray(case.mesh.point_data[FIXED_MASK.vtk], dtype=bool)
        if result_fixed.shape != (base.n_points, 3) or not np.array_equal(
            result_fixed, np.repeat(expected_fixed[:, None], 3, axis=1)
        ):
            msg = f"{case.label}/{case.seed} FixedMask differs from its boundary policy"
            raise ValueError(msg)
        fixed_value = np.asarray(
            case.mesh.point_data[FIXED_VALUE.vtk], dtype=np.float64
        )
        if fixed_value.shape != (base.n_points, 3) or not np.array_equal(
            fixed_value, np.zeros_like(fixed_value)
        ):
            msg = f"{case.label}/{case.seed} FixedValue is not exact zero"
            raise ValueError(msg)
        _require_equal(
            case.summary["cut_boundary/model_total_fixed_dofs"],
            int(result_fixed.sum()),
            context=f"{case.label}/{case.seed} model fixed DoFs",
        )
        newly_fixed = int(
            np.count_nonzero(expected_fixed[cut_ids] & ~base_fixed[cut_ids])
        )
        total_fixed = int(np.count_nonzero(expected_fixed[cut_ids]))
        for key, expected in (
            ("cut_boundary/newly_fixed_vertices", newly_fixed),
            ("cut_boundary/total_fixed_vertices", total_fixed),
        ):
            _require_equal(
                case.summary[key], expected, context=f"{case.label}/{case.seed} {key}"
            )
        cut_displacement = case.displacement[cut_ids]
        rms = float(np.linalg.norm(cut_displacement) / math.sqrt(cut_ids.size))
        max_abs = float(np.abs(cut_displacement).max())
        for key, actual in (
            ("cut_boundary/final_displacement_rms_m", rms),
            ("cut_boundary/final_displacement_max_abs_m", max_abs),
        ):
            _require_close(
                case.summary[key], actual, context=f"{case.label}/{case.seed} {key}"
            )
        if cut_boundary == "hard-fixed" and not np.allclose(
            cut_displacement, 0.0, rtol=0.0, atol=0.0
        ):
            msg = f"{case.label}/{case.seed} violates its exact hard-fixed boundary"
            raise ValueError(msg)
    return cut_ids


def _recompute_boundary_sensitivity(
    artifacts: list[CaseArtifact],
    base: pv.UnstructuredGrid,
    skin: pv.PolyData,
    expected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair = {(case.label, case.seed): case for case in artifacts}
    target = np.nan_to_num(
        np.asarray(base.point_data["Smile"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(base.point_data["SmileLossMask"], dtype=bool)
    loss_target_rms = float(
        np.linalg.norm(target[loss_mask]) / math.sqrt(int(loss_mask.sum()))
    )
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    triangles = encoded[:, 1:]
    face_triangles = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    face_local_ids = np.unique(triangles[face_triangles])
    face_mesh_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)[
        face_local_ids
    ]
    face_target_rms = float(
        np.linalg.norm(target[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
    )
    if (
        not math.isfinite(loss_target_rms)
        or loss_target_rms <= 0.0
        or not math.isfinite(face_target_rms)
        or face_target_rms <= 0.0
    ):
        msg = "invalid target RMS for boundary-sensitivity normalization"
        raise ValueError(msg)

    observed_rows: list[dict[str, Any]] = []
    for seed in SEED_ORDER:
        reference = by_pair[(CURRENT_BOUNDARY_CASE, seed)]
        bracket = by_pair[(HARD_FIXED_BOUNDARY_CASE, seed)]
        delta = bracket.displacement - reference.displacement
        full_rms = float(np.linalg.norm(delta) / math.sqrt(delta.shape[0]))
        loss_rms = float(
            np.linalg.norm(delta[loss_mask]) / math.sqrt(int(loss_mask.sum()))
        )
        face_rms = float(
            np.linalg.norm(delta[face_mesh_ids]) / math.sqrt(face_mesh_ids.size)
        )
        observed_rows.append(
            {
                "seed": seed,
                "reference_case": CURRENT_BOUNDARY_CASE,
                "bracket_case": HARD_FIXED_BOUNDARY_CASE,
                "full_displacement_delta_rms_m": full_rms,
                "loss_mask_displacement_delta_rms_m": loss_rms,
                "loss_mask_delta_fraction_of_target_rms": (loss_rms / loss_target_rms),
                "isface_displacement_delta_rms_m": face_rms,
                "isface_delta_fraction_of_isface_target_rms": (
                    face_rms / face_target_rms
                ),
                "target_error_fraction_delta": (
                    bracket.summary["target/error_rms_fraction_of_target"]
                    - reference.summary["target/error_rms_fraction_of_target"]
                ),
                "contraction_target_relative_dihedral_rms_deg_delta": (
                    bracket.summary[
                        "bumpiness/contraction_target_relative_dihedral_rms_deg"
                    ]
                    - reference.summary[
                        "bumpiness/contraction_target_relative_dihedral_rms_deg"
                    ]
                ),
                "residual_normal_laplacian_rms_m_delta": (
                    bracket.summary["bumpiness/residual_normal_laplacian_rms_m"]
                    - reference.summary["bumpiness/residual_normal_laplacian_rms_m"]
                ),
                "hard_fixed_is_ground_truth": False,
                "interpretation": HARD_FIXED_INTERPRETATION,
            }
        )

    _require_equal(
        len(expected_rows), len(observed_rows), context="boundary sensitivity rows"
    )
    for expected, observed in zip(expected_rows, observed_rows, strict=True):
        _require_equal(
            set(expected),
            set(observed),
            context=f"boundary sensitivity {observed['seed']}",
        )
        for key, value in observed.items():
            if isinstance(value, float):
                _require_close(
                    expected[key],
                    value,
                    context=f"boundary sensitivity {observed['seed']} {key}",
                )
            else:
                _require_equal(
                    expected[key],
                    value,
                    context=f"boundary sensitivity {observed['seed']} {key}",
                )
    return expected_rows


def _bounds_camera(
    points: np.ndarray, *, aspect: float = 1.35, padding: float = 1.12
) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    focus = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    scale = 0.5 * max(float(extent[1]), float(extent[0]) / aspect)
    return focus, padding * scale


def _build_render_basis(mesh: pv.UnstructuredGrid, skin: pv.PolyData) -> RenderBasis:
    skin_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    if np.any(skin_ids < 0) or np.any(skin_ids >= mesh.n_points):
        msg = "canonical skin GlobalPointId values are outside the prepared mesh"
        raise ValueError(msg)
    skin_points = np.asarray(skin.points, dtype=np.float64)
    if not np.array_equal(skin_points, np.asarray(mesh.points)[skin_ids]):
        msg = "canonical skin points do not match the prepared mesh"
        raise ValueError(msg)
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "canonical skin is not triangular"
        raise ValueError(msg)
    triangles = encoded[:, 1:]
    face_triangles = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    face_local_ids = np.unique(triangles[face_triangles])
    face_points = skin_points[face_local_ids]
    face_focus, face_scale = _bounds_camera(face_points)

    lip = np.asarray(mesh.point_data["IsLip"], dtype=bool)[skin_ids]
    lip &= np.isin(np.arange(skin.n_points), face_local_ids)
    if not np.any(lip):
        msg = "canonical IsFace surface has no lip points"
        raise ValueError(msg)
    mouth_focus, mouth_scale = _bounds_camera(skin_points[lip], padding=1.25)

    group_names = tuple(str(value) for value in skin.field_data["GroupName"])
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    eyelid_names = {
        "EyelidTop",
        "EyelidBottom",
        "EyelidOuterTop",
        "EyelidOuterBottom",
    }
    eyelid_ids = [
        index for index, name in enumerate(group_names) if name in eyelid_names
    ]
    eyelid = np.isin(group_ids, eyelid_ids)
    eyelid &= np.isin(np.arange(skin.n_points), face_local_ids)
    if not np.any(eyelid):
        msg = "canonical IsFace surface has no eyelid-group points"
        raise ValueError(msg)
    eyelid_x = skin_points[eyelid, 0]
    one_eye = eyelid & (skin_points[:, 0] >= np.median(eyelid_x))
    eye_focus, _ = _bounds_camera(skin_points[one_eye])
    face_height = float(np.ptp(face_points[:, 1]))
    eye_focus = eye_focus.copy()
    eye_focus[1] -= 0.08 * face_height
    eye_cheek_scale = 0.24 * face_height
    if not math.isfinite(eye_cheek_scale) or eye_cheek_scale <= 0.0:
        msg = "invalid eye-cheek camera scale"
        raise ValueError(msg)
    target = np.nan_to_num(
        np.asarray(mesh.point_data["Smile"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return RenderBasis(
        skin=skin,
        skin_mesh_ids=skin_ids,
        face_triangle_mask=face_triangles,
        target=target,
        face_focus=face_focus,
        face_scale=face_scale,
        mouth_focus=mouth_focus,
        mouth_scale=mouth_scale,
        eye_cheek_focus=eye_focus,
        eye_cheek_scale=eye_cheek_scale,
    )


def _deformed_face(basis: RenderBasis, displacement: np.ndarray) -> pv.PolyData:
    surface = basis.skin.copy(deep=True)
    surface.points = np.asarray(surface.points) + displacement[basis.skin_mesh_ids]
    selected = surface.extract_cells(np.flatnonzero(basis.face_triangle_mask))
    return selected.extract_surface(algorithm="dataset_surface")


def _render_contact_sheet(
    path: Path,
    *,
    seed: str,
    basis: RenderBasis,
    cases: list[CaseArtifact],
) -> None:
    by_label = {case.label: case for case in cases if case.seed == seed}
    _require_equal(set(by_label), set(CASE_ORDER), context=f"{seed} render cases")
    surfaces: list[tuple[str, pv.PolyData, str]] = [
        ("target", _deformed_face(basis, basis.target), "reference target")
    ]
    for label in CASE_ORDER:
        case = by_label[label]
        row = case.summary
        annotation = (
            f"{label} | {seed}\n"
            f"boundary={CASE_SETUP[label][2]}\n"
            f"err/target={row['target/error_rms_fraction_of_target']:.4f} | "
            f"area={1e3 * row['target/face_target_area_weighted_error_rms_m']:.3f} mm\n"
            f"dih={row['bumpiness/contraction_target_relative_dihedral_rms_deg']:.3f} deg | "
            f"nLap={1e3 * row['bumpiness/residual_normal_laplacian_rms_m']:.3f} mm\n"
            f"inv={row['warning/inverted_tets']} | "
            f"fold={row['warning/isface_folded_triangles']}"
        )
        surfaces.append((label, _deformed_face(basis, case.displacement), annotation))

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
        (
            "eye-cheek (+x)",
            front,
            basis.eye_cheek_focus,
            basis.eye_cheek_scale,
        ),
    )
    plotter = pv.Plotter(
        shape=(len(views), len(surfaces)),
        off_screen=True,
        window_size=(560 * len(surfaces), 1600),
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
                    f"fixed-activation probe | seed={seed}",
                    position="lower_left",
                    font_size=9,
                    color="black",
                )
    plotter.screenshot(path)
    plotter.close()


def _report_row(case: CaseArtifact) -> dict[str, Any]:
    row = case.summary
    return {
        "case": case.label,
        "seed": case.seed,
        "domain": row["domain"],
        "conversion": row["conversion"],
        "cut_boundary": CASE_SETUP[case.label][2],
        "cut_boundary/policy": row["cut_boundary/policy"],
        **{key: row[key] for key in REPORT_METRICS},
        **{key: row[key] for key in CUT_BOUNDARY_NUMERIC_METRICS},
        "artifact/result_path": str(case.path),
        "artifact/result_size_bytes": case.identity["size_bytes"],
        "artifact/result_sha256": case.identity["sha256"],
        "artifact/sidecar_path": str(case.sidecar_path),
        "artifact/sidecar_size_bytes": case.sidecar_identity["size_bytes"],
        "artifact/sidecar_sha256": case.sidecar_identity["sha256"],
        "activation/array_sha256": _array_sha256(case.activation),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_table(
    path: Path,
    rows: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    replay: dict[str, Any],
    sensitivity: list[dict[str, Any]],
) -> None:
    lines = [
        "# Fixed-activation domain x conversion probe",
        "",
        "| case | seed | domain | conversion | cut boundary | error/target | rest-area error mm | target-area error mm | dihedral deg | residual-normal Lap mm | inverted tets | folded face tris | forward |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        (
            "| {case} | {seed} | {domain} | {conversion} | {cut_boundary} | {error:.6g} | "
            "{rest_area:.6g} | {target_area:.6g} | {dihedral:.6g} | "
            "{normal_lap:.6g} | {inverted} | {folded} | {forward} |".format(
                case=row["case"],
                seed=row["seed"],
                domain=row["domain"],
                conversion=row["conversion"],
                cut_boundary=row["cut_boundary"],
                error=row["target/error_rms_fraction_of_target"],
                rest_area=1e3 * row["target/face_rest_area_weighted_error_rms_m"],
                target_area=(1e3 * row["target/face_target_area_weighted_error_rms_m"]),
                dihedral=row["bumpiness/contraction_target_relative_dihedral_rms_deg"],
                normal_lap=(1e3 * row["bumpiness/residual_normal_laplacian_rms_m"]),
                inverted=row["warning/inverted_tets"],
                folded=row["warning/isface_folded_triangles"],
                forward=row["forward/result"],
            )
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Zero/old branch checks",
            "",
            "| case | loss delta / target | IsFace delta / target | error-fraction delta | stable |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        (
            "| {case} | {loss:.6g} | {face:.6g} | {error:.6g} | {stable} |".format(
                case=branch["case"],
                loss=branch["zero_old/loss_mask_delta_fraction_of_target_rms"],
                face=branch["zero_old/isface_delta_fraction_of_isface_target_rms"],
                error=branch["zero_old_target_error_fraction_delta"],
                stable=branch["stable_within_declared_tolerance"],
            )
        )
        for branch in branches
    )
    lines.extend(
        [
            "",
            "## Current vs hard-fixed cut-boundary sensitivity",
            "",
            "The hard-fixed variant is a sensitivity bracket, not an anatomical ground truth.",
            "",
            "| seed | loss delta / target | IsFace delta / target | error-fraction delta | dihedral delta deg | residual-normal Lap delta mm |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            "| {seed} | {loss:.6g} | {face:.6g} | {error:.6g} | "
            "{dihedral:.6g} | {normal_lap:.6g} |".format(
                seed=row["seed"],
                loss=row["loss_mask_delta_fraction_of_target_rms"],
                face=row["isface_delta_fraction_of_isface_target_rms"],
                error=row["target_error_fraction_delta"],
                dihedral=row["contraction_target_relative_dihedral_rms_deg_delta"],
                normal_lap=(1e3 * row["residual_normal_laplacian_rms_m_delta"]),
            )
        )
        for row in sensitivity
    )
    lines.extend(
        [
            "",
            "## Historical replay gate",
            "",
            f"- Reproduced within tolerance: {replay['reproduces_historical_control_within_tolerance']}",
            f"- Loss-mask delta / target RMS: {replay['loss_mask_delta_fraction_of_target_rms']:.6g}",
            f"- IsFace delta / target RMS: {replay['isface_delta_fraction_of_isface_target_rms']:.6g}",
            "- Fold and inversion counts are visual-review warnings, not vetoes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config) -> None:
    # Explicitly approved to analyze/render the fixed-activation probe. This
    # entrypoint never runs a forward or inverse solve.
    # The analysis below is intentionally unreachable until the user reviews
    # the completed cheap-forward summary and explicitly approves rendering.
    _require_equal(
        file_sha256(INPUT_PRODUCER),
        INPUT_PRODUCER_SHA256,
        context="approved forward-probe producer SHA-256",
    )
    _validate_config(cfg)
    summary_identity = _file_identity(cfg.input_summary)
    summary = _read_json(cfg.input_summary)
    rows = _validate_summary(summary)
    base, skin = _load_base_inputs(cfg)
    artifacts = _load_case_artifacts(cfg, rows, base)
    _validate_cut_boundary_artifacts(artifacts, base, skin)
    sensitivity = _recompute_boundary_sensitivity(
        artifacts,
        base,
        skin,
        summary["boundary_sensitivity_checks"],
    )
    report_rows = [_report_row(case) for case in artifacts]
    branches = summary["branch_checks"]
    replay = summary["historical_replay_check"]
    all_branches_stable = all(
        bool(branch["stable_within_declared_tolerance"]) for branch in branches
    )
    replay_stable = bool(replay["reproduces_historical_control_within_tolerance"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "kind": "fixed-activation-domain-conversion-static-review",
        "input_summary": {
            "path": str(cfg.input_summary),
            **summary_identity,
            "schema_version": summary["schema_version"],
            "design": summary["design"],
            "producer_path": str(INPUT_PRODUCER),
            "producer_sha256": INPUT_PRODUCER_SHA256,
        },
        "case_order": list(CASE_ORDER),
        "seed_order": list(SEED_ORDER),
        "cases": report_rows,
        "branch_checks": branches,
        "boundary_sensitivity_checks": sensitivity,
        "historical_replay_check": replay,
        "interpretation_gates": {
            "historical_replay_passed": replay_stable,
            "all_zero_old_branches_stable": all_branches_stable,
            "numeric_causal_contrasts_eligible": summary[
                "comparison/causal_contrasts_eligible"
            ],
            "single_seed_ranking_eligible": replay_stable and all_branches_stable,
            "upstream_inverse_eligibility_status": summary[
                "inverse/eligibility_status"
            ],
            "upstream_inverse_required_gates": summary["inverse/required_gates"],
            "inverse_approval_automatic": False,
            "expensive_inverse_started": False,
        },
        "topology_provenance": summary["topology_provenance"],
        "warning_policy": (
            "inverted tetrahedra and folded IsFace triangles are recorded and "
            "rendered, but small visually imperceptible counts are not vetoes"
        ),
        "boundary_sensitivity_policy": {
            "reference_case": CURRENT_BOUNDARY_CASE,
            "bracket_case": HARD_FIXED_BOUNDARY_CASE,
            "hard_fixed_is_ground_truth": False,
            "interpretation": HARD_FIXED_INTERPRETATION,
            "eligibility_effect": (
                "reported as boundary-condition sensitivity; it is not a separate "
                "ground-truth ranking or automatic inverse-approval gate"
            ),
        },
        "visual_review": {
            "status": "pending",
            "next_inverse_automatic": False,
            "required_action": (
                "review both standardized zero- and old-seed contact sheets before "
                "any inverse approval"
            ),
        },
        "renders": {
            "views": ["front", "30 degree", "mouth", "eye-cheek (+x)"],
            "projection": "parallel",
            "surface": "all-vertex IsFace triangles only",
            "zero_seed": str(cfg.output_zero_views),
            "old_seed": str(cfg.output_old_views),
        },
        "limitations": [
            "fixed historical activation is a causal probe, not a recovered activation for changed setups",
            "the two nonlinear initial-displacement seeds are reported separately",
            "the hard-fixed artificial-cut boundary is a sensitivity bracket, not anatomical truth",
            "this analyzer does not run or approve an inverse experiment",
        ],
    }
    _validate_finite_json(payload, context="analysis payload")
    cfg.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(cfg.output_csv, report_rows)
    _write_table(cfg.output_table, report_rows, branches, replay, sensitivity)
    render_basis = _build_render_basis(base, skin)
    _render_contact_sheet(
        cfg.output_zero_views,
        seed="zero",
        basis=render_basis,
        cases=artifacts,
    )
    _render_contact_sheet(
        cfg.output_old_views,
        seed="old",
        basis=render_basis,
        cases=artifacts,
    )
    for path in (
        cfg.output_json,
        cfg.output_csv,
        cfg.output_table,
        cfg.output_zero_views,
        cfg.output_old_views,
    ):
        cherries.log_output(path)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    cherries.main(run)
