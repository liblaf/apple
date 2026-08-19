from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch

from liblaf import cherries
from liblaf.apple.common import (
    ACTIVATION_INV,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)

mpl.use("Agg", force=True)
logger = logging.getLogger(__name__)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
PLANE_STRESS_GROUP = REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin"
PLANE_STRESS_REFERENCE = PLANE_STRESS_GROUP / "src/_reference.py"
PLANE_STRESS_REFERENCE_SHA256 = (
    "470db910d6bec9ec81e06b5b46512781a188c252683b44b57b539ddb63295615"
)
CORRECTED_INVERSE_REFERENCE = (
    PLANE_STRESS_GROUP / "src/20-inverse-plane-stress-screen.py"
)
CORRECTED_INVERSE_REFERENCE_SHA256 = (
    "8c5d75ea06d66e60800d1c83c800d365bef01372340f88a650ef44732ea18f4d"
)
PREPARE_IMPLEMENTATION = Path(__file__).with_name("10-prepare-material-cases.py")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing required file: {path}"
        raise FileNotFoundError(msg)
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _bytes_identity(content: bytes) -> dict[str, int | str]:
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _require_file_identity(
    path: Path, expected: dict[str, Any], *, name: str
) -> dict[str, int | str]:
    actual = _file_identity(path)
    normalized = {
        "size_bytes": int(expected["size_bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if actual != normalized:
        msg = f"{name} identity mismatch: expected {normalized}, got {actual}"
        raise ValueError(msg)
    return actual


def _rehash_identity_tree(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{name} must be an identity leaf or a dictionary of identity nodes"
        raise TypeError(msg)
    leaf_keys = {"path", "size_bytes", "sha256"}
    keys = set(value)
    if keys == leaf_keys:
        raw_path = value["path"]
        size_bytes = value["size_bytes"]
        sha256 = value["sha256"]
        if not isinstance(raw_path, str) or not raw_path:
            msg = f"{name} identity path must be a non-empty string"
            raise TypeError(msg)
        path = Path(raw_path)
        if not path.is_absolute() or path.resolve() != path:
            msg = f"{name} identity path must be exact and absolute: {raw_path!r}"
            raise ValueError(msg)
        if type(size_bytes) is not int or size_bytes <= 0:
            msg = f"{name} identity size is invalid: {size_bytes!r}"
            raise ValueError(msg)
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            msg = f"{name} identity SHA-256 is invalid: {sha256!r}"
            raise ValueError(msg)
        identity = _require_file_identity(path, value, name=name)
        return {"path": raw_path, **identity}
    if keys & leaf_keys:
        msg = f"{name} is a malformed partial identity leaf: {sorted(keys)}"
        raise ValueError(msg)
    if not value:
        msg = f"{name} identity node must not be empty"
        raise ValueError(msg)
    if not all(isinstance(key, str) and key for key in value):
        msg = f"{name} identity node contains an invalid key"
        raise TypeError(msg)
    return {
        key: _rehash_identity_tree(child, name=f"{name}.{key}")
        for key, child in value.items()
    }


def _load_pinned_module(
    path: Path, expected_sha256: str, *, module_name: str
) -> ModuleType:
    _require_file_identity(
        path,
        {"size_bytes": path.stat().st_size, "sha256": expected_sha256},
        name=module_name,
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load pinned module {module_name}: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The corrected inverse source imports a module literally named ``_reference``.
# Install its own pinned reference module first, rather than allowing this new
# experiment directory to change import resolution.
plane_reference = _load_pinned_module(
    PLANE_STRESS_REFERENCE,
    PLANE_STRESS_REFERENCE_SHA256,
    module_name="_reference",
)


def _unregistered_cherries_path(value: str | Path, *_: Any, **__: Any) -> Path:
    """Resolve imported config defaults without registering irrelevant assets."""
    return Path(value)


_cherries_input = cherries.input
_cherries_output = cherries.output
try:
    # Imported reference Config classes otherwise register their historical
    # relative inputs/outputs in this new Cherries run. That caused misleading
    # missing-input warnings and incomplete local snapshots in earlier probes.
    cherries.input = _unregistered_cherries_path
    cherries.output = _unregistered_cherries_path
    corrected = _load_pinned_module(
        CORRECTED_INVERSE_REFERENCE,
        CORRECTED_INVERSE_REFERENCE_SHA256,
        module_name="_selective_skin_corrected_inverse_reference",
    )
finally:
    cherries.input = _cherries_input
    cherries.output = _cherries_output
legacy = corrected.legacy
reference_runtime = legacy.reference_runtime
configure_hard_fixed_cut_boundary = corrected.__dict__[
    "_configure_hard_fixed_cut_boundary"
]
cut_boundary_readback = corrected.__dict__["_cut_boundary_readback"]
require_inverse_runtime_identity = corrected.__dict__[
    "_require_inverse_runtime_identity"
]

DESIGN = "corrected-isface-selective-efat-c020-two-case-inverse"
MANIFEST_DESIGN = "corrected-isface-two-case-selective-efat-c020-inverse-materials-v2"
MANIFEST_SCHEMA_VERSION = 2
AGGREGATE_SCHEMA_VERSION = 1
CASE_ORDER = ("HFP1", "HFP0")
CONTROL_ORDER = ("H0P0", "H0P1")
MANIFEST_CASE_ORDER = CASE_ORDER
CASE_CONTRACT = {
    "HFP1": (
        "selective-efat003-where-raw-r-gt-1",
        "c020-raw-area-ratio-floor-050",
        True,
    ),
    "HFP0": ("selective-efat003-where-raw-r-gt-1", "p000", False),
}
CASE_RELATIVE_SKIN_PATHS = {
    "HFP1": "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
    "HFP0": "10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
}

EXPECTED_SKIN_POINTS = corrected.EXPECTED_SKIN_POINTS
EXPECTED_SKIN_TRIANGLES = corrected.EXPECTED_SKIN_TRIANGLES
EXPECTED_SKIN_AREA_M2 = corrected.EXPECTED_SKIN_AREA_M2
EXPECTED_MODEL_FIXED_VERTICES = corrected.EXPECTED_MODEL_FIXED_VERTICES
EXPECTED_MODEL_FIXED_DOFS = corrected.EXPECTED_MODEL_FIXED_DOFS
EXPECTED_CUT_INCIDENT_VERTICES = corrected.EXPECTED_CUT_INCIDENT_VERTICES
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    corrected.EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
)
EXPECTED_INPUT_MESH_IDENTITY = {
    "size_bytes": 76_792_914,
    "sha256": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
}
KOITER_IMPLEMENTATION = corrected.KOITER_IMPLEMENTATION
EXPECTED_KOITER_IMPLEMENTATION_IDENTITY = {
    "size_bytes": 17_329,
    "sha256": "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c",
}
EXPECTED_BASELINE_INVERSE_ARTIFACTS = {
    "aggregate_summary": {
        "path": str(
            PLANE_STRESS_GROUP / "data/20-corrected-baseline-screen-summary-final.json"
        ),
        "size_bytes": 148_046,
        "sha256": "64a030366053b14eed9ad4da322d910146175fe7bb781e2dca8ee976c03c7045",
    },
    "case_summary": {
        "path": str(
            PLANE_STRESS_GROUP
            / "data/20-human-face-smile-skin-no-prestrain-lr3-corrected-"
            "isface-e0200-p000-screen-summary-final.json"
        ),
        "size_bytes": 126_540,
        "sha256": "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856",
    },
    "result": {
        "path": str(
            PLANE_STRESS_GROUP
            / "data/20-human-face-smile-skin-no-prestrain-lr3-corrected-"
            "isface-e0200-p000-screen.vtu"
        ),
        "size_bytes": 147_657_021,
        "sha256": "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4",
    },
    "history": {
        "path": str(
            PLANE_STRESS_GROUP
            / "data/20-human-face-smile-skin-no-prestrain-lr3-corrected-"
            "isface-e0200-p000-screen-steps.vtkhdf"
        ),
        "size_bytes": 2_066_073_161,
        "sha256": "6e29d7b205e7901681942f0d413b091c5e4bce003ec4d789c2d7f69ded430d24",
    },
    "trace": {
        "path": str(
            PLANE_STRESS_GROUP
            / "data/20-human-face-smile-skin-no-prestrain-lr3-corrected-"
            "isface-e0200-p000-screen-trace.jsonl"
        ),
        "size_bytes": 91_767,
        "sha256": "a0f83957c832a119f6f031fb78a46fe52060d3b190a2ba0a1265f000c5d8cde3",
    },
    "target": {
        "path": str(
            PLANE_STRESS_GROUP
            / "data/20-human-face-smile-skin-no-prestrain-lr3-corrected-"
            "isface-e0200-p000-screen-target.vtu"
        ),
        "size_bytes": 84_419_492,
        "sha256": "89ec02dfd87330f7dc1d303639893f7698ef2e6098480c4e39fa2ad94240206c",
    },
}

PRIOR_SELECTIVE_GROUP = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse"
)
EXPECTED_H0P1_CONTROL_ARTIFACTS = {
    "skin": {
        "path": str(
            PRIOR_SELECTIVE_GROUP / "data/10-prepared-material-cases/skin-h0p1-c020.vtp"
        ),
        "size_bytes": 1_898_983,
        "sha256": "9b69f6cad6cfc7c6deadbd687d2947a44eedf7a295aae3540ac104ea50ebaacf",
    },
    "aggregate_summary": {
        "path": str(
            PRIOR_SELECTIVE_GROUP
            / "data/20-selective-skin-prestrain-inverse-summary-final.json"
        ),
        "size_bytes": 387_036,
        "sha256": "cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6",
    },
    "case_summary": {
        "path": str(PRIOR_SELECTIVE_GROUP / "data/20-h0p1-summary-final.json"),
        "size_bytes": 111_232,
        "sha256": "0ecf17c2a25cc03ebccb42a7ca3bd25bbeaf2fde0b0da3f1dec156efe8d99b2c",
    },
    "result": {
        "path": str(PRIOR_SELECTIVE_GROUP / "data/20-h0p1.vtu"),
        "size_bytes": 147_640_393,
        "sha256": "eabec1d0493f004d066f94f20b5ac6725f8d84245ceccfddc51dc191dd96cde0",
    },
    "history": {
        "path": str(PRIOR_SELECTIVE_GROUP / "data/20-h0p1-steps.vtkhdf"),
        "size_bytes": 2_071_499_339,
        "sha256": "c082c6202218c74a5fa3ef9c01048487a7ba4988621a7628eb1dc6a641debceb",
    },
    "trace": {
        "path": str(PRIOR_SELECTIVE_GROUP / "data/20-h0p1-trace.jsonl"),
        "size_bytes": 79_471,
        "sha256": "657de3a7bc6cbdf361dd805aafa9eedc64bc77a1487b93327e0d82de6257f692",
    },
    "target": {
        "path": str(PRIOR_SELECTIVE_GROUP / "data/20-h0p1-target.vtu"),
        "size_bytes": 84_419_492,
        "sha256": "3d6e3fe1baa48745d8592b109ad513cf7a822f05ece26ccbf2045c57b8f44418",
    },
}

EXPECTED_V1_OUTPUT_IDENTITIES = {
    "manifest": {
        "path": str(GROUP_DIR / "data/10-prepared-material-cases-manifest.json"),
        "size_bytes": 34_033,
        "sha256": "843179e074d00ead3469cd9e0e5f69f2f0b521398e86709d1ecb466bda2f26a9",
    },
    "table": {
        "path": str(GROUP_DIR / "data/10-prepared-material-cases-table.md"),
        "size_bytes": 571,
        "sha256": "e3bd37cc1351a42d665dcb21d6eabc01cba95f5fe2a56de3faa47d480e71c0de",
    },
    "hfp1_skin": {
        "path": str(
            GROUP_DIR
            / "data/10-prepared-material-cases/skin-hfp1-selective-efat-c020.vtp"
        ),
        "size_bytes": 1_933_979,
        "sha256": "2199b33ba7896bfde82a9e1fcf12e7782e9e89daa742b787eb267a824f1ae855",
    },
    "hfp0_skin": {
        "path": str(
            GROUP_DIR
            / "data/10-prepared-material-cases/skin-hfp0-selective-efat-p000.vtp"
        ),
        "size_bytes": 1_611_312,
        "sha256": "f3c2ebaf95f7b82c15a15743ef2a1be3eea378f1a89f9044df379179732c6bf7",
    },
}

EXPECTED_INPUT_MANIFEST_SHA256: str | None = (
    "1dd835e2638b4fc3789e79c8b9c620383889a0e0963a0f82a5ea0b43b2375693"
)
EXPECTED_PREPARE_IMPLEMENTATION_SHA256: str | None = (
    "28d0c3667d3d803c58e8dc60e52c896cd62758ac586918fcf8f86b9da94985f5"
)
EXPECTED_SMOKE_FINAL_AGGREGATE_SHA256: str | None = (
    "ec37bc77fed40691e84517a62148205ffe1bbfcedfea380cf4eb528ce2c86194"
)
EXPECTED_SMOKE_FINAL_AGGREGATE_SIZE_BYTES: int | None = 59_896

SMOKE_EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True
FORMAL_EXECUTION_APPROVED_AFTER_SMOKE_REVIEW = True
SMOKE_APPROVAL_BLOCKER = (
    "NO-GO: the two-case zero-update forward-plus-adjoint smoke awaits static "
    "review, prepared-manifest pinning, and an explicit source approval flip."
)
FORMAL_APPROVAL_BLOCKER = (
    "NO-GO: the two 40-update inverses await successful smoke review, pinned "
    "smoke evidence, and a separate explicit source approval flip."
)

EXPECTED_INPUT_MANIFEST = GROUP_DIR / "data/10-prepared-material-cases-v2-manifest.json"
EXPECTED_FORMAL_SUMMARY = (
    GROUP_DIR / "data/20-fat-floor-skin-prestrain-inverse-summary.json"
)
EXPECTED_FORMAL_TABLE = GROUP_DIR / "data/20-fat-floor-skin-prestrain-inverse-table.md"
EXPECTED_FORMAL_LIVE_PLOT_DIR = GROUP_DIR / "figs/live-fat-floor-skin-prestrain-inverse"
EXPECTED_SMOKE_V1_ROOT = GROUP_DIR / "tmp/20-fat-floor-skin-prestrain-smoke"
EXPECTED_SMOKE_ROOT = GROUP_DIR / "tmp/20-fat-floor-skin-prestrain-smoke-v2"
EXPECTED_SMOKE_SUMMARY = (
    EXPECTED_SMOKE_ROOT / "20-fat-floor-skin-prestrain-inverse-summary.json"
)
EXPECTED_SMOKE_TABLE = (
    EXPECTED_SMOKE_ROOT / "20-fat-floor-skin-prestrain-inverse-table.md"
)
EXPECTED_SMOKE_LIVE_PLOT_DIR = EXPECTED_SMOKE_ROOT / "figs/live"
EXPECTED_SMOKE_FINAL_AGGREGATE = EXPECTED_SMOKE_SUMMARY.with_name(
    f"{EXPECTED_SMOKE_SUMMARY.stem}-final.json"
)

LAME_CONVERSION = corrected.LAME_CONVERSION
VOLUME_LAME_CONVERSION = corrected.VOLUME_LAME_CONVERSION
SKIN_THICKNESS_M = float(legacy.SKIN_THICKNESS)
SKIN_E_MPA = 0.2
FAT_E_MPA = 0.003
SKIN_NU = 0.49
PRESTRAIN_LINEAR_FACTOR = 0.98
PRESTRAIN_AREA_RATIO_FLOOR = 0.5
VOLUME_FRACTION_DV_RTOL = 1.0e-10
VOLUME_FRACTION_DV_ATOL_M3 = 1.0e-18
PRESTRAIN_AREA_FACTOR = PRESTRAIN_LINEAR_FACTOR**2
EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA = 0.09253042789422025
EXPECTED_CANDIDATE_ARRAY_HASHES = {
    "expanding_mask": "1da30d0805e41ebb56de39fb26ccd54c2b7a8bd7f4d1257459cbc7b9aa0bc05e",
    "fat_floor_young": "3d16df172d08edfdb52077ac8961aa80bc3363ed9e9d8b0b1f1a0f3695b82c1e",
    "fat_floor_lambda": "d6ef7eed22196a7d8cb2bbe519d565fbeeafbf7c0ad6285e0de7377b374f2bf1",
    "fat_floor_mu": "09a30f9c6a95003ddbc6e10c3880f5f8888c152656d2b49a499286681f4d05fa",
    "p000_stress_free_area_ratio": "aa74d25a4afece1f232101f98e5fda8177935c3e833ae578be636d9ff42294c2",
    "p000_activation": "051fe4599913dc590cb39aa79f7bb51578efc3323cd9c0a337be804d12d8f224",
    "rho_c020": "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
    "activation_c020": "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55",
    "mask_all_zero": "7bd2b07dd38fb3c32897d8931f62e04381be41f4b91df22c102fb4db1f477ce1",
    "mask_all_one": "d2580c50d0997b172975ce03f35909cfcd6660043ca11b84b86a61b4871b367c",
    "activation_diag_c020": "9886106f0c5a60c8007e6a6eda26eaa9c8bc33b159c9f719307bec5c832664d3",
    "activation_diag_p000": "c1a2cfddfccbbbf0c5c331571cb1ccab9fea8d6afb8c7aa920c74abe1d86f7e2",
    "inherited_int8_all_zero": "7bd2b07dd38fb3c32897d8931f62e04381be41f4b91df22c102fb4db1f477ce1",
    "inherited_int8_all_one": "d2580c50d0997b172975ce03f35909cfcd6660043ca11b84b86a61b4871b367c",
    "teeth_proximity": "c8866dd1aecdb6c7c7fee45c009103f48cc5a4b7a3752926cbb45bc88b88a76e",
}
SMOKE_MIN_GRAD_NORM = 1.0e-12
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14
REQUIRED_FLOAT_ARRAYS = (
    "RestArea",
    "SkinYoungModulusMPa",
    "SkinPoissonRatio",
    LAMBDA.vtk,
    MU.vtk,
    FRACTION.vtk,
    ACTIVATION_INV.vtk,
    "SkinActivationInvDiag",
    "TargetRestAreaRatio",
    "ClippedTargetRestAreaRatio",
    "StressFreeAreaRatio",
)
INHERITED_DOMAIN_ONE_ARRAYS = ("IsFaceTriangle", "SourceOuterTriangle")
INHERITED_DOMAIN_ZERO_ARRAYS = (
    "ArtificialCutTriangle",
    "DisallowedGroupTriangle",
    "FixedTriangle",
)
INHERITED_DIAGNOSTIC_ARRAYS = (
    "GingivaProximityTriangle",
    "TeethProximityTriangle",
)
REQUIRED_INHERITED_INT8_ARRAYS = (
    *INHERITED_DOMAIN_ONE_ARRAYS,
    *INHERITED_DOMAIN_ZERO_ARRAYS,
    *INHERITED_DIAGNOSTIC_ARRAYS,
)
REQUIRED_MASK_ARRAYS = (
    "ExpandingTriangle",
    "SelectiveFatFloor",
    "C020PrestrainEnabled",
)
BASELINE_FLOAT_ARRAYS = (
    "RestArea",
    "SkinYoungModulusMPa",
    "SkinPoissonRatio",
    LAMBDA.vtk,
    MU.vtk,
    FRACTION.vtk,
    ACTIVATION_INV.vtk,
    "StressFreeAreaRatio",
)


@dataclass(frozen=True)
class BatchCase:
    case_id: str
    target: Literal["smile"]
    lr: float
    setup: str
    label: str

    @property
    def skin_enabled(self) -> bool:
        return True

    @property
    def skin_prestrain_enabled(self) -> bool:
        return CASE_CONTRACT[self.case_id][2]

    @property
    def skin_constant_tightening(self) -> float:
        return PRESTRAIN_LINEAR_FACTOR - 1.0 if self.skin_prestrain_enabled else 0.0

    @property
    def setup_label(self) -> str:
        return self.setup

    @property
    def stem(self) -> str:
        return f"20-{self.case_id.lower()}"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(plane_reference.PREPARED_MESH)
    input_cut_reference: Path = cherries.input(plane_reference.SOURCE_SKIN)
    input_manifest: Path = cherries.input("10-prepared-material-cases-v2-manifest.json")
    # Stage-specific paths cannot use cherries.output() here: class defaults are
    # evaluated before CLI overrides, so a smoke run would incorrectly register
    # the absent formal outputs. The selected paths are logged explicitly after
    # their final bytes are written.
    output_summary: Path = Path("data/20-fat-floor-skin-prestrain-inverse-summary.json")
    output_table: Path = Path("data/20-fat-floor-skin-prestrain-inverse-table.md")
    live_plot_dir: Path = Path("figs/live-fat-floor-skin-prestrain-inverse")

    stage: str = "formal"
    case_order: str = ",".join(CASE_ORDER)
    initial_activation_mesh: Path | None = None
    use_initial_displacement: bool = False
    inverse_lr: float = 0.3
    loss_scale: float = legacy.LOSS_SCALE
    adam_eps: float = legacy.ADAM_EPS
    inverse_max_steps: int = 40
    mandatory_baseline_steps: int = 40
    segment_steps: int = 8
    live_snapshot_interval: int = 0
    area_ratio_floor: float = 0.1
    diagnostic_min_delta_rel: float = 1.0e-3
    flat_log_slope_tol: float = 5.0e-3
    aggressive_lr_factor: float = 2.0
    slow_lr_factor: float = 1.5
    lr_shrink_factor: float = 0.5
    max_lr: float = 1.0
    min_lr: float = 0.00375
    loss_deterioration_rel: float = 1.0e-2
    time_budget_hours: float = 6.0
    reserve_minutes: float = 5.0
    step_time_budget_s: float = 180.0
    require_convergence: bool = False
    require_solver_success: bool = True
    max_solver_failure_fraction: float = 0.0


def _reject_json_constant(value: str) -> None:
    msg = f"JSON contains non-standard constant {value!r}"
    raise ValueError(msg)


def _require_finite_json(value: Any, *, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{path} contains non-finite value {value}"
        raise ValueError(msg)
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if not isinstance(payload, dict):
        msg = f"expected a JSON object in {path}"
        raise TypeError(msg)
    _require_finite_json(payload)
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _require_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_exact_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        msg = f"{name} must be {expected}, got {actual}"
        raise ValueError(msg)


def _stage_paths(cfg: Config) -> tuple[Path, Path, Path]:
    if cfg.stage == "formal":
        return (
            EXPECTED_FORMAL_SUMMARY,
            EXPECTED_FORMAL_TABLE,
            EXPECTED_FORMAL_LIVE_PLOT_DIR,
        )
    if cfg.stage == "smoke":
        return (
            EXPECTED_SMOKE_SUMMARY,
            EXPECTED_SMOKE_TABLE,
            EXPECTED_SMOKE_LIVE_PLOT_DIR,
        )
    msg = f"stage must be 'smoke' or 'formal', got {cfg.stage!r}"
    raise ValueError(msg)


def _validate_config(cfg: Config) -> None:
    if EXPECTED_SMOKE_ROOT.resolve() == EXPECTED_SMOKE_V1_ROOT.resolve():
        msg = "smoke retry root must not alias the preserved v1 smoke outputs"
        raise RuntimeError(msg)
    expected_summary, expected_table, expected_plots = _stage_paths(cfg)
    for actual, expected, name in (
        (cfg.input_mesh, plane_reference.PREPARED_MESH, "input_mesh"),
        (cfg.input_cut_reference, plane_reference.SOURCE_SKIN, "input_cut_reference"),
        (cfg.input_manifest, EXPECTED_INPUT_MANIFEST, "input_manifest"),
        (cfg.output_summary, expected_summary, "output_summary"),
        (cfg.output_table, expected_table, "output_table"),
        (cfg.live_plot_dir, expected_plots, "live_plot_dir"),
    ):
        _require_exact_path(actual, expected, name=name)
    if cfg.case_order != ",".join(CASE_ORDER):
        msg = f"case order must remain {CASE_ORDER}, got {cfg.case_order!r}"
        raise ValueError(msg)
    if cfg.initial_activation_mesh is not None or cfg.use_initial_displacement:
        msg = "every inverse must start from exact-zero activation and displacement"
        raise ValueError(msg)
    protocol = {
        "inverse_lr": cfg.inverse_lr,
        "loss_scale": cfg.loss_scale,
        "adam_eps": cfg.adam_eps,
        "segment_steps": cfg.segment_steps,
        "live_snapshot_interval": cfg.live_snapshot_interval,
        "area_ratio_floor": cfg.area_ratio_floor,
        "diagnostic_min_delta_rel": cfg.diagnostic_min_delta_rel,
        "flat_log_slope_tol": cfg.flat_log_slope_tol,
        "aggressive_lr_factor": cfg.aggressive_lr_factor,
        "slow_lr_factor": cfg.slow_lr_factor,
        "lr_shrink_factor": cfg.lr_shrink_factor,
        "max_lr": cfg.max_lr,
        "min_lr": cfg.min_lr,
        "loss_deterioration_rel": cfg.loss_deterioration_rel,
        "time_budget_hours": cfg.time_budget_hours,
        "reserve_minutes": cfg.reserve_minutes,
        "step_time_budget_s": cfg.step_time_budget_s,
        "require_convergence": cfg.require_convergence,
        "require_solver_success": cfg.require_solver_success,
        "max_solver_failure_fraction": cfg.max_solver_failure_fraction,
    }
    if protocol != legacy.EXPECTED_PROTOCOL:
        msg = f"fixed inverse protocol changed: expected {legacy.EXPECTED_PROTOCOL}, got {protocol}"
        raise ValueError(msg)
    expected_steps = 0 if cfg.stage == "smoke" else 40
    if (
        cfg.inverse_max_steps != expected_steps
        or cfg.mandatory_baseline_steps != expected_steps
    ):
        msg = f"{cfg.stage} requires exactly {expected_steps} optimizer updates"
        raise ValueError(msg)
    if str(mpl.get_backend()).lower() != "agg":
        msg = f"non-interactive Agg backend required, got {mpl.get_backend()}"
        raise RuntimeError(msg)


def _require_execution_approval(cfg: Config) -> None:
    if EXPECTED_INPUT_MANIFEST_SHA256 is None:
        msg = "NO-GO: pin EXPECTED_INPUT_MANIFEST_SHA256 after material preparation"
        raise RuntimeError(msg)
    if EXPECTED_PREPARE_IMPLEMENTATION_SHA256 is None:
        msg = (
            "NO-GO: pin EXPECTED_PREPARE_IMPLEMENTATION_SHA256 after the final "
            "static review of material preparation"
        )
        raise RuntimeError(msg)
    if cfg.stage == "smoke":
        if not SMOKE_EXECUTION_APPROVED_AFTER_STATIC_REVIEW:
            raise RuntimeError(SMOKE_APPROVAL_BLOCKER)
        return
    if not FORMAL_EXECUTION_APPROVED_AFTER_SMOKE_REVIEW:
        raise RuntimeError(FORMAL_APPROVAL_BLOCKER)
    if (
        EXPECTED_SMOKE_FINAL_AGGREGATE_SHA256 is None
        or EXPECTED_SMOKE_FINAL_AGGREGATE_SIZE_BYTES is None
    ):
        msg = "NO-GO: formal inverses require pinned completed smoke evidence"
        raise RuntimeError(msg)
    smoke_identity = _require_file_identity(
        EXPECTED_SMOKE_FINAL_AGGREGATE,
        {
            "size_bytes": EXPECTED_SMOKE_FINAL_AGGREGATE_SIZE_BYTES,
            "sha256": EXPECTED_SMOKE_FINAL_AGGREGATE_SHA256,
        },
        name="reviewed smoke final aggregate",
    )
    smoke = _read_json(EXPECTED_SMOKE_FINAL_AGGREGATE)
    if (
        smoke.get("complete") is not True
        or smoke.get("stage") != "smoke"
        or tuple(smoke.get("case_order", [])) != CASE_ORDER
        or int(smoke.get("inverse_evaluations", -1)) != 1
        or any(row.get("status") != "ok" for row in smoke.get("cases", []))
    ):
        msg = f"reviewed smoke evidence is not a completed two-case smoke: {smoke_identity}"
        raise RuntimeError(msg)


def _artifact_paths(data_dir: Path, case: BatchCase) -> Any:
    return legacy.CasePaths.from_case(data_dir, case)


def _canonical_case_summary(path: Path) -> Path:
    return path.with_name(f"{path.stem}-final.json")


def _canonical_aggregate(path: Path) -> Path:
    return path.with_name(f"{path.stem}-final.json")


def _cases() -> list[BatchCase]:
    prestrain_setup = legacy.SETUP_SKIN_ESTIMATED_PRESTRAIN
    no_prestrain_setup = legacy.SETUP_SKIN_NO_PRESTRAIN
    return [
        BatchCase("HFP1", "smile", 0.3, prestrain_setup, "HFP1"),
        BatchCase("HFP0", "smile", 0.3, no_prestrain_setup, "HFP0"),
    ]


def _refuse_stale_outputs(cfg: Config) -> None:
    expected: dict[str, Path] = {
        "aggregate": cfg.output_summary,
        "aggregate temporary": cfg.output_summary.with_name(
            f"{cfg.output_summary.name}.tmp"
        ),
        "aggregate final archive": _canonical_aggregate(cfg.output_summary),
        "table": cfg.output_table,
        "live plot directory": cfg.live_plot_dir,
    }
    for case in _cases():
        paths = _artifact_paths(cfg.output_summary.parent, case)
        expected.update(
            {
                f"{case.case_id} target": paths.target,
                f"{case.case_id} result": paths.result,
                f"{case.case_id} summary": paths.summary,
                f"{case.case_id} summary temporary": paths.summary.with_name(
                    f"{paths.summary.name}.tmp"
                ),
                f"{case.case_id} final summary archive": _canonical_case_summary(
                    paths.summary
                ),
                f"{case.case_id} history": paths.history,
                f"{case.case_id} history temporary": paths.history.with_name(
                    f"{paths.history.name}.tmp"
                ),
                f"{case.case_id} trace": paths.trace,
            }
        )
    stale = [f"{name}: {path}" for name, path in expected.items() if path.exists()]
    if stale:
        msg = (
            f"refusing {cfg.stage} run because expected targets already exist; "
            "archive or remove them explicitly after review: " + "; ".join(stale)
        )
        raise FileExistsError(msg)


def _array_sha256_le_c(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _triangles(skin: pv.PolyData) -> np.ndarray:
    faces = np.asarray(skin.faces, dtype=np.int64)
    if skin.n_cells == 0 or faces.size != 4 * skin.n_cells:
        msg = "skin must contain only non-empty triangles"
        raise ValueError(msg)
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "skin contains non-triangle cells"
        raise ValueError(msg)
    return encoded[:, 1:]


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _topology_digest(skin: pv.PolyData) -> str:
    digest = hashlib.sha256()
    global_ids = np.ascontiguousarray(
        np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk]), dtype="<i8"
    )
    triangles = np.ascontiguousarray(_triangles(skin), dtype="<i8")
    for values in (global_ids, triangles):
        digest.update(str(values.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _verify_declared_array(
    skin: pv.PolyData, name: str, declared: dict[str, Any], *, case_id: str
) -> np.ndarray:
    if declared.get("association") != "cell":
        msg = f"{case_id} {name} must be a declared cell array"
        raise ValueError(msg)
    if name not in skin.cell_data:
        msg = f"{case_id} skin is missing cell array {name!r}"
        raise KeyError(msg)
    dtype = str(declared.get("dtype"))
    expected_dtype = (
        "u1"
        if name in REQUIRED_MASK_ARRAYS
        else ("i1" if name in REQUIRED_INHERITED_INT8_ARRAYS else "<f8")
    )
    if dtype != expected_dtype:
        msg = f"{case_id} {name} dtype declaration changed: {dtype!r}"
        raise ValueError(msg)
    values = np.asarray(skin.cell_data[name])
    expected_shape = (
        (EXPECTED_SKIN_TRIANGLES, 3)
        if name == ACTIVATION_INV.vtk
        else (EXPECTED_SKIN_TRIANGLES,)
    )
    if values.dtype != np.dtype(expected_dtype):
        msg = f"{case_id} {name} live dtype changed: {values.dtype}"
        raise ValueError(msg)
    if (
        tuple(declared.get("shape", [])) != expected_shape
        or values.shape != expected_shape
    ):
        msg = (
            f"{case_id} {name} shape mismatch: declared={declared.get('shape')}, "
            f"live={values.shape}, expected={expected_shape}"
        )
        raise ValueError(msg)
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    if declared.get("finite") is not True or not np.isfinite(canonical).all():
        msg = f"{case_id} {name} contains or declares non-finite values"
        raise ValueError(msg)
    actual_sha256 = _array_sha256_le_c(canonical, dtype)
    if actual_sha256 != str(declared.get("sha256_le_c")):
        msg = f"{case_id} {name} canonical array SHA-256 mismatch"
        raise ValueError(msg)
    actual_min = float(canonical.min())
    actual_max = float(canonical.max())
    if not math.isclose(
        actual_min, float(declared.get("min", math.nan)), rel_tol=0.0, abs_tol=0.0
    ) or not math.isclose(
        actual_max, float(declared.get("max", math.nan)), rel_tol=0.0, abs_tol=0.0
    ):
        msg = f"{case_id} {name} min/max readback differs from manifest"
        raise ValueError(msg)
    return canonical


def _verify_convenience_array_contract(  # noqa: C901
    skin: pv.PolyData,
    live: dict[str, np.ndarray],
    row: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    activation = np.asarray(live[ACTIVATION_INV.vtk], dtype=np.float64)
    diag = np.asarray(live["SkinActivationInvDiag"], dtype=np.float64)
    if not np.array_equal(diag, activation[:, 0]):
        msg = f"{case_id} SkinActivationInvDiag differs from ActivationInv[:, 0]"
        raise ValueError(msg)
    prestrain_enabled = CASE_CONTRACT[case_id][2]
    diag_hash_key = (
        "activation_diag_c020" if prestrain_enabled else "activation_diag_p000"
    )
    diag_hash = _array_sha256_le_c(diag, "<f8")
    if diag_hash != EXPECTED_CANDIDATE_ARRAY_HASHES[diag_hash_key]:
        msg = f"{case_id} authoritative SkinActivationInvDiag hash changed"
        raise ValueError(msg)

    triangles = _triangles(skin)
    for point_name in ("IsFace", "IsFixed", "IsTeeth", "IsGingiva"):
        if point_name not in skin.point_data:
            msg = f"{case_id} is missing convenience point array {point_name}"
            raise KeyError(msg)
    if not np.all(np.asarray(skin.point_data["IsFace"], dtype=bool)):
        msg = f"{case_id} convenience domain contains a non-IsFace point"
        raise ValueError(msg)
    if np.any(np.asarray(skin.point_data["IsFixed"], dtype=bool)):
        msg = f"{case_id} convenience domain overlaps original fixed points"
        raise ValueError(msg)

    expected_values: dict[str, np.ndarray] = {}
    for name in INHERITED_DOMAIN_ONE_ARRAYS:
        expected_values[name] = np.ones(skin.n_cells, dtype=np.int8)
    for name in INHERITED_DOMAIN_ZERO_ARRAYS:
        expected_values[name] = np.zeros(skin.n_cells, dtype=np.int8)
    expected_values["TeethProximityTriangle"] = np.any(
        np.asarray(skin.point_data["IsTeeth"], dtype=bool)[triangles], axis=1
    ).astype(np.int8)
    expected_values["GingivaProximityTriangle"] = np.any(
        np.asarray(skin.point_data["IsGingiva"], dtype=bool)[triangles], axis=1
    ).astype(np.int8)
    expected_hashes = {
        **dict.fromkeys(
            INHERITED_DOMAIN_ONE_ARRAYS,
            EXPECTED_CANDIDATE_ARRAY_HASHES["inherited_int8_all_one"],
        ),
        **dict.fromkeys(
            INHERITED_DOMAIN_ZERO_ARRAYS,
            EXPECTED_CANDIDATE_ARRAY_HASHES["inherited_int8_all_zero"],
        ),
        "TeethProximityTriangle": EXPECTED_CANDIDATE_ARRAY_HASHES["teeth_proximity"],
        "GingivaProximityTriangle": EXPECTED_CANDIDATE_ARRAY_HASHES[
            "inherited_int8_all_zero"
        ],
    }
    inherited_records: dict[str, Any] = {}
    for name in sorted(REQUIRED_INHERITED_INT8_ARRAYS):
        values = np.asarray(live[name], dtype=np.int8)
        if not np.array_equal(values, expected_values[name]):
            msg = f"{case_id} inherited convenience array {name} semantics changed"
            raise ValueError(msg)
        array_hash = _array_sha256_le_c(values, "i1")
        if array_hash != expected_hashes[name]:
            msg = f"{case_id} inherited convenience array {name} hash changed"
            raise ValueError(msg)
        inherited_records[name] = {
            "dtype": "i1",
            "shape": [skin.n_cells],
            "sha256_le_c": array_hash,
            "nonzero": int(np.count_nonzero(values)),
            "exact_semantic_readback": True,
            "unchanged_from_corrected_source": True,
        }
    expected_audit = {
        "complete": True,
        "SkinActivationInvDiag": {
            "dtype": "<f8",
            "shape": [skin.n_cells],
            "sha256_le_c": diag_hash,
            "exactly_equals": "ActivationInv[:, 0]",
        },
        "inherited_cell_arrays": inherited_records,
        "point_domain": {
            "all_IsFace": True,
            "any_IsFixed": False,
            "diagnostic_masks_rederived_from": ["IsTeeth", "IsGingiva"],
        },
    }
    if row["skin"].get("convenience_array_audit") != expected_audit:
        msg = f"{case_id} convenience-array audit declaration changed"
        raise ValueError(msg)
    return {
        "skin/activation_inv_diag_exact": True,
        "skin/activation_inv_diag_sha256": diag_hash,
        "skin/inherited_convenience_arrays_exact": True,
    }


def _verify_skin_formulas(  # noqa: C901, PLR0912, PLR0915
    skin: pv.PolyData, row: dict[str, Any], *, case_id: str
) -> dict[str, Any]:
    arrays = row["skin"].get("arrays")
    if not isinstance(arrays, dict):
        msg = f"{case_id} manifest skin.arrays must be an object"
        raise TypeError(msg)
    required = {
        *REQUIRED_FLOAT_ARRAYS,
        *REQUIRED_MASK_ARRAYS,
        *REQUIRED_INHERITED_INT8_ARRAYS,
    }
    if set(arrays) != required:
        msg = (
            f"{case_id} array set changed: missing={sorted(required - set(arrays))}, "
            f"extra={sorted(set(arrays) - required)}"
        )
        raise ValueError(msg)
    live = {
        name: _verify_declared_array(skin, name, arrays[name], case_id=case_id)
        for name in sorted(required)
    }
    convenience_metrics = _verify_convenience_array_contract(
        skin, live, row, case_id=case_id
    )
    rest_area = np.asarray(live["RestArea"], dtype=np.float64)
    geometric_area = _triangle_area(
        np.asarray(skin.points, dtype=np.float64), _triangles(skin)
    )
    if not np.allclose(rest_area, geometric_area, rtol=FORMULA_RTOL, atol=FORMULA_ATOL):
        msg = f"{case_id} RestArea differs from original skin geometry"
        raise ValueError(msg)
    if not math.isclose(
        float(rest_area.sum()),
        EXPECTED_SKIN_AREA_M2,
        rel_tol=0.0,
        abs_tol=corrected.AREA_ATOL_M2,
    ):
        msg = f"{case_id} IsFace membrane area changed"
        raise ValueError(msg)

    young = np.asarray(live["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(live["SkinPoissonRatio"], dtype=np.float64)
    lambda_ = np.asarray(live[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(live[MU.vtk], dtype=np.float64)
    fraction = np.asarray(live[FRACTION.vtk], dtype=np.float64)
    activation = np.asarray(live[ACTIVATION_INV.vtk], dtype=np.float64)
    stress_free = np.asarray(live["StressFreeAreaRatio"], dtype=np.float64)
    ratio = np.asarray(live["TargetRestAreaRatio"], dtype=np.float64)
    clipped = np.asarray(live["ClippedTargetRestAreaRatio"], dtype=np.float64)
    expanding = np.asarray(live["ExpandingTriangle"], dtype=np.uint8).astype(bool)
    fat_floor = np.asarray(live["SelectiveFatFloor"], dtype=np.uint8).astype(bool)
    c020 = np.asarray(live["C020PrestrainEnabled"], dtype=np.uint8).astype(bool)

    for name, values, expected in (
        (
            "raw target/rest area ratio",
            ratio,
            "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606",
        ),
        (
            "clipped target/rest area ratio",
            clipped,
            "aaf87f8d68485136c0ce09d113ce09de481654613c7d50c80ac2becb40e86e1e",
        ),
    ):
        if _array_sha256_le_c(values, "<f8") != expected:
            msg = f"{case_id} authoritative {name} hash changed"
            raise ValueError(msg)

    if (
        np.any(ratio <= 0.0)
        or int(expanding.sum()) != 16_723
        or not np.array_equal(expanding, ratio > 1.0)
    ):
        msg = f"{case_id} expanding mask is not raw target/rest area ratio > 1"
        raise ValueError(msg)
    expected_clipped = np.clip(ratio, PRESTRAIN_AREA_RATIO_FLOOR, 1.0)
    if not np.allclose(clipped, expected_clipped, rtol=0.0, atol=0.0):
        msg = f"{case_id} clipped area ratio is not clip(raw, 0.5, 1)"
        raise ValueError(msg)
    if not np.array_equal(nu, np.full_like(nu, SKIN_NU)):
        msg = f"{case_id} skin Poisson ratio changed"
        raise ValueError(msg)
    if not np.array_equal(fraction, np.ones_like(fraction)):
        msg = f"{case_id} skin Fraction must remain one"
        raise ValueError(msg)

    _young_mode, prestrain_mode, prestrain_enabled = CASE_CONTRACT[case_id]
    expected_fat_floor = expanding
    expected_young = np.where(expected_fat_floor, FAT_E_MPA, SKIN_E_MPA)
    if not np.array_equal(fat_floor, expected_fat_floor) or not np.array_equal(
        young, expected_young
    ):
        msg = f"{case_id} selective fat-floor Young's-modulus rule changed"
        raise ValueError(msg)
    if (
        not np.all(young > 0.0)
        or float(young.min()) != FAT_E_MPA
        or float(young.max()) != SKIN_E_MPA
    ):
        msg = f"{case_id} Young's modulus must stay in [0.003, 0.2] MPa"
        raise ValueError(msg)
    for name, values, hash_key, dtype in (
        ("fat-floor marker", fat_floor, "expanding_mask", "u1"),
        ("Young's modulus", young, "fat_floor_young", "<f8"),
        ("plane-stress lambda", lambda_, "fat_floor_lambda", "<f8"),
        ("plane-stress mu", mu, "fat_floor_mu", "<f8"),
    ):
        if (
            _array_sha256_le_c(values, dtype)
            != EXPECTED_CANDIDATE_ARRAY_HASHES[hash_key]
        ):
            msg = f"{case_id} authoritative {name} hash changed"
            raise ValueError(msg)
    expected_lambda = young * SKIN_NU / (1.0 - SKIN_NU**2)
    expected_mu = young / (2.0 * (1.0 + SKIN_NU))
    if not np.allclose(
        lambda_, expected_lambda, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
    ) or not np.allclose(mu, expected_mu, rtol=FORMULA_RTOL, atol=FORMULA_ATOL):
        msg = f"{case_id} skin does not use the plane-stress Lamé conversion"
        raise ValueError(msg)

    if prestrain_enabled:
        expected_stress_free = PRESTRAIN_AREA_FACTOR * clipped
        expected_diag = np.power(expected_stress_free, -0.5) - 1.0
        expected_activation = np.column_stack(
            (expected_diag, expected_diag, np.zeros_like(expected_diag))
        )
        expected_c020 = np.ones_like(c020)
        if (
            _array_sha256_le_c(stress_free, "<f8")
            != "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1"
            or _array_sha256_le_c(activation, "<f8")
            != "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55"
        ):
            msg = f"{case_id} authoritative c020 field hash changed"
            raise ValueError(msg)
    else:
        expected_stress_free = np.ones_like(stress_free)
        expected_activation = np.zeros_like(activation)
        expected_c020 = np.zeros_like(c020)
        if (
            _array_sha256_le_c(stress_free, "<f8")
            != EXPECTED_CANDIDATE_ARRAY_HASHES["p000_stress_free_area_ratio"]
            or _array_sha256_le_c(activation, "<f8")
            != EXPECTED_CANDIDATE_ARRAY_HASHES["p000_activation"]
        ):
            msg = f"{case_id} authoritative p000 field hash changed"
            raise ValueError(msg)
    if not np.allclose(
        stress_free, expected_stress_free, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
    ) or not np.allclose(
        activation, expected_activation, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
    ):
        msg = f"{case_id} {prestrain_mode} prestrain formula changed"
        raise ValueError(msg)
    if not np.array_equal(c020, expected_c020):
        msg = f"{case_id} C020PrestrainEnabled marker changed"
        raise ValueError(msg)
    c020_hash_key = "mask_all_one" if prestrain_enabled else "mask_all_zero"
    if _array_sha256_le_c(c020, "u1") != EXPECTED_CANDIDATE_ARRAY_HASHES[c020_hash_key]:
        msg = f"{case_id} authoritative c020 marker hash changed"
        raise ValueError(msg)
    effective_prestrain = c020 & (young > 0.0)
    expected_effective_count = EXPECTED_SKIN_TRIANGLES if prestrain_enabled else 0
    if int(effective_prestrain.sum()) != expected_effective_count:
        msg = (
            f"{case_id} effective c020 support changed: "
            f"{int(effective_prestrain.sum())} != {expected_effective_count}"
        )
        raise ValueError(msg)
    area_weighted_e = float(np.dot(young, rest_area) / rest_area.sum())
    if not math.isclose(
        area_weighted_e,
        EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA,
        rel_tol=1.0e-13,
        abs_tol=1.0e-15,
    ):
        msg = f"{case_id} area-weighted Young's modulus changed"
        raise ValueError(msg)
    return {
        **convenience_metrics,
        "skin/rest_area_m2": float(rest_area.sum()),
        "skin/expanding_triangles": int(expanding.sum()),
        "skin/expanding_area_m2": float(rest_area[expanding].sum()),
        "skin/selective_fat_floor_triangles": int(fat_floor.sum()),
        "skin/selective_fat_floor_area_m2": float(rest_area[fat_floor].sum()),
        "skin/selective_fat_floor_area_fraction": float(
            rest_area[fat_floor].sum() / rest_area.sum()
        ),
        "skin/effective_prestrain_triangles": int(effective_prestrain.sum()),
        "skin/E_min_MPa": float(young.min()),
        "skin/E_max_MPa": float(young.max()),
        "skin/E_area_weighted_mean_MPa": area_weighted_e,
        "skin/lambda_min_MPa": float(lambda_.min()),
        "skin/lambda_max_MPa": float(lambda_.max()),
        "skin/mu_min_MPa": float(mu.min()),
        "skin/mu_max_MPa": float(mu.max()),
        "skin/prestrain_activation_inv_rms": float(
            np.linalg.norm(activation) / math.sqrt(activation.size)
        ),
        "skin/prestrain_activation_inv_max_abs": float(np.abs(activation).max()),
        "skin/target_rest_area_ratio_min": float(ratio.min()),
        "skin/target_rest_area_ratio_max": float(ratio.max()),
    }


def _verify_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("controls")
    if (
        not isinstance(rows, list)
        or tuple(str(row.get("case_id")) for row in rows) != CONTROL_ORDER
    ):
        msg = f"manifest controls must remain ordered {CONTROL_ORDER}"
        raise ValueError(msg)
    by_id = {str(row["case_id"]): row for row in rows}
    expected_by_id = {
        "H0P0": {
            "skin": {
                "path": str(
                    PLANE_STRESS_GROUP
                    / "data/10-corrected-baseline/skin-isface-e0200-p000.vtp"
                ),
                "size_bytes": 1_138_550,
                "sha256": "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
            },
            "inverse_artifacts": EXPECTED_BASELINE_INVERSE_ARTIFACTS,
            "prestrain_mode": "p000",
        },
        "H0P1": {
            "skin": EXPECTED_H0P1_CONTROL_ARTIFACTS["skin"],
            "inverse_artifacts": {
                name: row
                for name, row in EXPECTED_H0P1_CONTROL_ARTIFACTS.items()
                if name != "skin"
            },
            "prestrain_mode": "c020-raw-area-ratio-floor-050",
        },
    }
    verified: dict[str, Any] = {}
    for case_id in CONTROL_ORDER:
        row = by_id[case_id]
        expected = expected_by_id[case_id]
        if (
            row.get("young_modulus_mode") != "homogeneous-e0200"
            or row.get("prestrain_mode") != expected["prestrain_mode"]
            or row.get("reused_not_rerun") is not True
            or row.get("skin") != expected["skin"]
            or row.get("inverse_artifacts") != expected["inverse_artifacts"]
        ):
            msg = f"{case_id} control declaration changed"
            raise ValueError(msg)
        verified_case: dict[str, Any] = {}
        for name, artifact in {
            "skin": expected["skin"],
            **expected["inverse_artifacts"],
        }.items():
            path = Path(str(artifact["path"]))
            identity = _require_file_identity(
                path, artifact, name=f"{case_id} control {name}"
            )
            verified_case[name] = {"path": str(path), **identity}
        aggregate = _read_json(
            Path(str(expected["inverse_artifacts"]["aggregate_summary"]["path"]))
        )
        expected_stage = "screen" if case_id == "H0P0" else "formal"
        if (
            aggregate.get("complete") is not True
            or aggregate.get("stage") != expected_stage
            or int(aggregate.get("inverse_evaluations", -1)) != 41
        ):
            msg = f"pinned {case_id} control is not a completed 40-update inverse"
            raise ValueError(msg)
        verified[case_id] = verified_case
    return verified


def _replace_prepare_approval_assignment(
    source: bytes, *, expected: bytes, replacement: bytes
) -> tuple[bytes, int]:
    lines = source.splitlines(keepends=True)
    replacements = 0
    for index, line in enumerate(lines):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        if body == expected:
            lines[index] = replacement + ending
            replacements += 1
    return b"".join(lines), replacements


def _prepare_review_normalized_sha256(path: Path) -> str:
    source = path.read_bytes()
    disabled = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
    enabled = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
    normalized, enabled_count = _replace_prepare_approval_assignment(
        source, expected=enabled, replacement=disabled
    )
    disabled_count = sum(
        line.rstrip(b"\r\n") == disabled
        for line in normalized.splitlines(keepends=True)
    )
    if enabled_count not in (0, 1) or disabled_count != 1:
        msg = "prepare source must contain exactly one static approval assignment"
        raise ValueError(msg)
    return hashlib.sha256(normalized).hexdigest()


def _prepare_approval_only_reconstruction(
    path: Path,
    *,
    preapproval_identity: dict[str, Any],
    executable_identity: dict[str, Any],
) -> dict[str, Any]:
    source = path.read_bytes()
    disabled = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
    enabled = b"EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
    live_identity = _bytes_identity(source)
    if live_identity == preapproval_identity:
        preapproval = source
        executable, replacement_count = _replace_prepare_approval_assignment(
            source, expected=disabled, replacement=enabled
        )
        live_mode = "statically-reviewed-preapproval"
    elif live_identity == executable_identity:
        executable = source
        preapproval, replacement_count = _replace_prepare_approval_assignment(
            source, expected=enabled, replacement=disabled
        )
        live_mode = "approved-executable"
    else:
        msg = "live prepare source is neither the reviewed nor executable identity"
        raise ValueError(msg)
    if (
        replacement_count != 1
        or _bytes_identity(preapproval) != preapproval_identity
        or _bytes_identity(executable) != executable_identity
    ):
        msg = "prepare source is not the manifest's approval-only transition"
        raise ValueError(msg)
    return {
        "live_mode": live_mode,
        "live_identity": live_identity,
        "preapproval_identity": _bytes_identity(preapproval),
        "reconstructed_executable_identity": _bytes_identity(executable),
        "replacement_count": 1,
        "verified": True,
    }


def _verify_manifest_provenance(  # noqa: C901, PLR0912, PLR0915
    manifest: dict[str, Any],
) -> dict[str, Any]:
    provenance = manifest.get("input_provenance")
    if not isinstance(provenance, dict):
        msg = "manifest input_provenance must be an object"
        raise TypeError(msg)
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "corrected_skin",
        "corrected_manifest",
        "raw_area_ratio_driver",
    }:
        msg = "manifest material-preparation input set changed"
        raise ValueError(msg)
    expected_paths = {
        "corrected_skin": PLANE_STRESS_GROUP
        / "data/10-corrected-baseline/skin-isface-e0200-p000.vtp",
        "corrected_manifest": PLANE_STRESS_GROUP
        / "data/10-corrected-baseline-manifest.json",
        "raw_area_ratio_driver": plane_reference.SOURCE_SKIN,
    }
    verified: dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        declared = inputs[name]
        _require_exact_path(
            Path(str(declared.get("path", ""))),
            expected_path,
            name=f"manifest input_provenance {name}",
        )
        identity = _require_file_identity(
            expected_path, declared, name=f"material-preparation input {name}"
        )
        verified[name] = {"path": str(expected_path), **identity}
        cherries.log_input(expected_path)
    if provenance.get("corrected_manifest_contract_verified") is not True:
        msg = "prepare manifest did not verify the corrected baseline contract"
        raise ValueError(msg)
    control_artifacts = provenance.get("control_artifacts")
    if control_artifacts != {
        "H0P0": EXPECTED_BASELINE_INVERSE_ARTIFACTS,
        "H0P1": EXPECTED_H0P1_CONTROL_ARTIFACTS,
    }:
        msg = "prepare manifest control-artifact provenance changed"
        raise ValueError(msg)
    preserved_v1 = provenance.get("preserved_v1_outputs")
    if preserved_v1 != EXPECTED_V1_OUTPUT_IDENTITIES:
        msg = "prepare manifest preserved-v1 provenance changed"
        raise ValueError(msg)
    verified_v1: dict[str, Any] = {}
    for name, expected in EXPECTED_V1_OUTPUT_IDENTITIES.items():
        path = Path(str(expected["path"]))
        identity = _require_file_identity(path, expected, name=f"preserved v1 {name}")
        verified_v1[name] = {"path": str(path), **identity}
    verified["preserved_v1_outputs"] = verified_v1
    final = provenance.get("final_recheck")
    if not isinstance(final, dict) or final.get("all_unchanged") is not True:
        msg = "prepare manifest lacks a successful final input recheck"
        raise ValueError(msg)

    expected_constants = {
        "skin_E_MPa": SKIN_E_MPA,
        "skin_nu": SKIN_NU,
        "skin_thickness_m": SKIN_THICKNESS_M,
        "skin_lame_conversion": LAME_CONVERSION,
        "skin_energy_measure": "fixed original reference area",
        "linear_tightening": 0.02,
        "length_factor": PRESTRAIN_LINEAR_FACTOR,
        "uniform_natural_area_ratio": PRESTRAIN_AREA_FACTOR,
        "raw_area_ratio_floor": PRESTRAIN_AREA_RATIO_FLOOR,
        "fat_E_MPa": FAT_E_MPA,
        "selective_fat_floor_rule": (
            "E=fat modulus (0.003 MPa) iff raw TargetRestAreaRatio > 1; else E=0.2 MPa"
        ),
        "fat_floor_area_weighted_E_MPa": EXPECTED_FAT_FLOOR_AREA_WEIGHTED_E_MPA,
    }
    if manifest.get("constants") != expected_constants:
        msg = "manifest material constants differ from the approved experiment"
        raise ValueError(msg)
    if manifest.get("candidate_array_hashes_le_c") != EXPECTED_CANDIDATE_ARRAY_HASHES:
        msg = "manifest candidate material-array hashes changed"
        raise ValueError(msg)
    convenience_contract = manifest.get("convenience_array_contract")
    if not isinstance(convenience_contract, dict):
        msg = "manifest convenience_array_contract must be an object"
        raise TypeError(msg)
    expected_requirement = (
        "SkinActivationInvDiag exactly equals ActivationInv[:, 0]; inherited "
        "domain and diagnostic arrays retain their corrected-source values and "
        "rederived semantics"
    )
    source_audit = convenience_contract.get("corrected_source_audit")
    if (
        convenience_contract.get("candidate_requirement") != expected_requirement
        or not isinstance(source_audit, dict)
        or source_audit.get("complete") is not True
        or source_audit.get("SkinActivationInvDiag")
        != {
            "dtype": "<f8",
            "shape": [EXPECTED_SKIN_TRIANGLES],
            "sha256_le_c": EXPECTED_CANDIDATE_ARRAY_HASHES["activation_diag_p000"],
            "exactly_equals": "ActivationInv[:, 0]",
        }
        or source_audit.get("point_domain")
        != {
            "all_IsFace": True,
            "any_IsFixed": False,
            "diagnostic_masks_rederived_from": ["IsTeeth", "IsGingiva"],
        }
    ):
        msg = "manifest corrected-source convenience audit changed"
        raise ValueError(msg)
    source_inherited = source_audit.get("inherited_cell_arrays")
    if not isinstance(source_inherited, dict) or set(source_inherited) != set(
        REQUIRED_INHERITED_INT8_ARRAYS
    ):
        msg = "manifest corrected-source inherited-array set changed"
        raise ValueError(msg)
    expected_nonzero = {
        **dict.fromkeys(INHERITED_DOMAIN_ONE_ARRAYS, EXPECTED_SKIN_TRIANGLES),
        **dict.fromkeys(INHERITED_DOMAIN_ZERO_ARRAYS, 0),
        "GingivaProximityTriangle": 0,
        "TeethProximityTriangle": 52,
    }
    for name, nonzero in expected_nonzero.items():
        expected_hash_key = (
            "inherited_int8_all_one"
            if name in INHERITED_DOMAIN_ONE_ARRAYS
            else (
                "teeth_proximity"
                if name == "TeethProximityTriangle"
                else "inherited_int8_all_zero"
            )
        )
        if source_inherited.get(name) != {
            "dtype": "i1",
            "shape": [EXPECTED_SKIN_TRIANGLES],
            "sha256_le_c": EXPECTED_CANDIDATE_ARRAY_HASHES[expected_hash_key],
            "nonzero": nonzero,
            "exact_semantic_readback": True,
            "unchanged_from_corrected_source": False,
        }:
            msg = f"manifest corrected-source convenience array {name} changed"
            raise ValueError(msg)
    expected_anatomy_contract = {
        "skin": {
            "domain": "all-vertex IsFace filtered PolyData",
            "E_MPa": "case-dependent: 0.003 or 0.2",
            "nu": SKIN_NU,
            "thickness_m": SKIN_THICKNESS_M,
            "lame_conversion": LAME_CONVERSION,
            "energy_measure": "fixed original reference area",
        },
        "volume": {
            "lame_conversion": VOLUME_LAME_CONVERSION,
            "fat": {"E_MPa": 0.003, "nu": 0.49},
            "muscle": {
                "E_MPa": 0.03,
                "nu": 0.49,
                "activation": "fresh exact-zero inverse initialization",
                "parameterization": "independent 6-DoF per active muscle tet",
            },
            "aponeurosis": {"E_MPa": 0.1, "nu": 0.35},
        },
        "boundary": {
            "policy": "all cross-section incident vertices fixed to zero displacement",
            "expected_cut_vertices": 6_980,
            "expected_model_fixed_vertices": 33_636,
            "expected_model_fixed_dofs": 100_908,
        },
    }
    if manifest.get("anatomy_material_contract") != expected_anatomy_contract:
        msg = "manifest anatomy/material/boundary contract changed"
        raise ValueError(msg)
    mapping = manifest.get("mapping")
    required_mapping = {
        "raw_ratio_sha256_le_f8": (
            "da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606"
        ),
        "clipped_ratio_sha256_le_f8": (
            "aaf87f8d68485136c0ce09d113ce09de481654613c7d50c80ac2becb40e86e1e"
        ),
        "rho_c020_sha256_le_f8": (
            "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1"
        ),
        "activation_c020_sha256_le_f8": (
            "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55"
        ),
        "expanding_mask_sha256_u1": (
            "1da30d0805e41ebb56de39fb26ccd54c2b7a8bd7f4d1257459cbc7b9aa0bc05e"
        ),
        "expanding_triangles": 16_723,
        "contracting_triangles": 13_159,
        "unchanged_triangles": 17,
        "floor_clamped_triangles": 31,
    }
    if not isinstance(mapping, dict) or any(
        mapping.get(key) != value for key, value in required_mapping.items()
    ):
        msg = "manifest raw-area-ratio mapping/derivation hashes changed"
        raise ValueError(msg)
    expected_output_contract = {
        "manifest_path": str(EXPECTED_INPUT_MANIFEST),
        "table_path": str(GROUP_DIR / "data/10-prepared-material-cases-v2-table.md"),
        "candidate_root": str(GROUP_DIR / "data/10-prepared-material-cases-v2"),
        "generated_candidate_vtps": 2,
        "controls_reused_by_identity": True,
        "v1_outputs_preserved_by_identity": True,
        "overwrite_policy": "refuse any existing final or temporary output",
    }
    if manifest.get("output_contract") != expected_output_contract:
        msg = "manifest v2 output contract changed"
        raise ValueError(msg)
    approval = manifest.get("approval")
    if (
        not isinstance(approval, dict)
        or approval.get("material_preparation_static_review") is not True
        or approval.get("inverse_execution_approved") is not False
        or approval.get("forward_or_adjoint_smoke_approved") is not False
    ):
        msg = "material manifest approval scope changed"
        raise ValueError(msg)
    return verified


def _load_manifest(  # noqa: C901, PLR0912, PLR0915
    cfg: Config,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_identity = _require_file_identity(
        cfg.input_manifest,
        {
            "size_bytes": cfg.input_manifest.stat().st_size,
            "sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        },
        name="pinned material manifest",
    )
    manifest = _read_json(cfg.input_manifest)
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        msg = f"unexpected manifest schema {manifest.get('schema_version')}"
        raise ValueError(msg)
    if (
        manifest.get("design") != MANIFEST_DESIGN
        or manifest.get("complete") is not True
    ):
        msg = "material manifest design is wrong or incomplete"
        raise ValueError(msg)
    if tuple(manifest.get("case_order", [])) != MANIFEST_CASE_ORDER:
        msg = f"manifest case order must remain {MANIFEST_CASE_ORDER}"
        raise ValueError(msg)
    if tuple(manifest.get("control_order", [])) != CONTROL_ORDER:
        msg = f"manifest control order must remain {CONTROL_ORDER}"
        raise ValueError(msg)
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        msg = "manifest producer must be an object"
        raise TypeError(msg)
    _require_exact_path(
        Path(str(producer.get("path", ""))),
        PREPARE_IMPLEMENTATION,
        name="manifest producer",
    )
    if producer.get("relative_path") != "src/10-prepare-material-cases.py":
        msg = "manifest producer relative path changed"
        raise ValueError(msg)
    preapproval = producer.get("statically_reviewed_preapproval_source")
    executable = producer.get("executable_source")
    reconstruction = producer.get("approval_only_reconstruction")
    if not all(
        isinstance(value, dict) for value in (preapproval, executable, reconstruction)
    ):
        msg = "manifest producer is missing v2 source-provenance objects"
        raise TypeError(msg)
    assert isinstance(preapproval, dict)
    assert isinstance(executable, dict)
    assert isinstance(reconstruction, dict)
    preapproval_identity = preapproval.get("file_identity")
    executable_identity = executable.get("file_identity")
    if not isinstance(preapproval_identity, dict) or not isinstance(
        executable_identity, dict
    ):
        msg = "manifest producer source identities are malformed"
        raise TypeError(msg)
    live_reconstruction = _prepare_approval_only_reconstruction(
        PREPARE_IMPLEMENTATION,
        preapproval_identity=preapproval_identity,
        executable_identity=executable_identity,
    )
    if (
        preapproval.get("approval_assignment")
        != "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False"
        or executable.get("approval_assignment")
        != "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = True"
        or preapproval_identity != live_reconstruction["preapproval_identity"]
        or executable_identity
        != live_reconstruction["reconstructed_executable_identity"]
        or reconstruction
        != {
            "verified": True,
            "replacement_count": 1,
            "permitted_edit": (
                "EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False -> True"
            ),
            "reconstructed_executable_identity": executable_identity,
        }
    ):
        msg = "manifest producer approval-only source reconstruction changed"
        raise ValueError(msg)
    if (
        str(preapproval_identity.get("sha256"))
        != EXPECTED_PREPARE_IMPLEMENTATION_SHA256
        or _prepare_review_normalized_sha256(PREPARE_IMPLEMENTATION)
        != EXPECTED_PREPARE_IMPLEMENTATION_SHA256
    ):
        msg = "prepare producer differs from the statically reviewed source"
        raise ValueError(msg)
    _require_file_identity(
        PREPARE_IMPLEMENTATION,
        live_reconstruction["live_identity"],
        name="live material prepare implementation",
    )
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != len(MANIFEST_CASE_ORDER):
        msg = "manifest must contain exactly HFP1,HFP0 candidate rows"
        raise ValueError(msg)
    by_id = {str(row.get("case_id")): row for row in rows}
    if tuple(by_id) != MANIFEST_CASE_ORDER or len(by_id) != len(rows):
        msg = "manifest case rows are duplicated or reordered"
        raise ValueError(msg)
    manifest["runtime/verified_controls"] = _verify_controls(manifest)
    manifest["runtime/verified_preparation_inputs"] = _verify_manifest_provenance(
        manifest
    )
    for case_id, row in by_id.items():
        young_mode, prestrain_mode, _ = CASE_CONTRACT[case_id]
        if (
            row.get("young_modulus_mode") != young_mode
            or row.get("prestrain_mode") != prestrain_mode
            or row.get("validation", {}).get("ok") is not True
            or row.get("validation", {}).get("errors") != []
        ):
            msg = f"{case_id} manifest row failed its immutable material contract"
            raise ValueError(msg)
        if row.get("generated") is not True:
            msg = f"{case_id} must be a newly generated candidate skin"
            raise ValueError(msg)
        if case_id in CASE_RELATIVE_SKIN_PATHS:
            if (
                row.get("skin", {}).get("relative_path")
                != CASE_RELATIVE_SKIN_PATHS[case_id]
            ):
                msg = f"{case_id} candidate skin relative path changed"
                raise ValueError(msg)
            expected_path = (
                cfg.input_manifest.parent / CASE_RELATIVE_SKIN_PATHS[case_id]
            )
            _require_exact_path(
                Path(str(row["skin"]["path"])), expected_path, name=f"{case_id} skin"
            )
        if row.get("inverse_artifacts") is not None:
            msg = f"{case_id} must not carry reused inverse artifacts"
            raise ValueError(msg)
    manifest["runtime/manifest_identity"] = manifest_identity
    return manifest, by_id


def _verified_skins(
    by_id: dict[str, dict[str, Any]],
    base_mesh: pv.UnstructuredGrid,
) -> dict[str, tuple[Path, pv.PolyData, dict[str, Any]]]:
    verified: dict[str, tuple[Path, pv.PolyData, dict[str, Any]]] = {}
    reference_topology: str | None = None
    for case_id in MANIFEST_CASE_ORDER:
        row = by_id[case_id]
        skin_spec = row["skin"]
        path = Path(str(skin_spec["path"])).resolve()
        identity = _require_file_identity(
            path, skin_spec["file_identity"], name=f"{case_id} skin"
        )
        skin = pv.read(path)
        if not isinstance(skin, pv.PolyData):
            msg = f"{case_id} skin read as {type(skin).__name__}"
            raise TypeError(msg)
        if (
            skin.n_points != EXPECTED_SKIN_POINTS
            or skin.n_cells != EXPECTED_SKIN_TRIANGLES
            or int(skin_spec.get("points", -1)) != EXPECTED_SKIN_POINTS
            or int(skin_spec.get("triangles", -1)) != EXPECTED_SKIN_TRIANGLES
        ):
            msg = f"{case_id} is not the fixed 15,299-point/29,899-triangle IsFace skin"
            raise ValueError(msg)
        global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if (
            global_ids.shape != (skin.n_points,)
            or np.unique(global_ids).size != skin.n_points
            or global_ids.min() < 0
            or global_ids.max() >= base_mesh.n_points
            or not np.array_equal(
                np.asarray(skin.points, dtype=np.float64),
                np.asarray(base_mesh.points, dtype=np.float64)[global_ids],
            )
        ):
            msg = f"{case_id} skin-to-volume GlobalPointId mapping changed"
            raise ValueError(msg)
        topology = _topology_digest(skin)
        if reference_topology is None:
            reference_topology = topology
        elif topology != reference_topology:
            msg = f"{case_id} skin cell order/topology differs across candidates"
            raise ValueError(msg)
        formula_metrics = _verify_skin_formulas(skin, row, case_id=case_id)
        cherries.log_input(path)
        verified[case_id] = (
            path,
            skin,
            {
                **formula_metrics,
                "provenance/skin_file_size_bytes": int(identity["size_bytes"]),
                "provenance/skin_file_sha256": str(identity["sha256"]),
                "provenance/skin_topology_order_sha256": topology,
            },
        )
    return verified


def _tensor_exact_readback(
    actual: torch.Tensor, expected: np.ndarray, *, name: str
) -> None:
    expected_t = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    if actual.shape != expected_t.shape or not torch.equal(actual.detach(), expected_t):
        msg = f"model material readback differs for {name}"
        raise RuntimeError(msg)


def _build_forward(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    _case: BatchCase,
    *,
    area_ratio_floor: float,
    skin_path: Path,
    skin: pv.PolyData,
    row: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Any, pv.PolyData, dict[str, Any]]:
    del area_ratio_floor
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    case_id = str(row["case_id"])
    candidate_skin = skin.copy(deep=True)
    cut_ids, cut_metrics = configure_hard_fixed_cut_boundary(mesh)
    if cut_ids.size != EXPECTED_CUT_INCIDENT_VERTICES:
        msg = f"{case_id} hard-fixed cut vertex count changed"
        raise RuntimeError(msg)

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    volume_contract = (
        (
            "aponeurosis",
            legacy.APONEUROSIS_E,
            legacy.APONEUROSIS_NU,
            legacy.APONEUROSIS_FRACTION,
            StableNeoHookean,
        ),
        ("fat", legacy.FAT_E, legacy.FAT_NU, legacy.FAT_FRACTION, StableNeoHookean),
        (
            "muscle",
            legacy.MUSCLE_E,
            legacy.MUSCLE_NU,
            legacy.MUSCLE_FRACTION,
            StableNeoHookeanActive,
        ),
    )
    expected_volume: dict[str, dict[str, np.ndarray]] = {}
    volume_fraction_inputs: dict[str, tuple[str, np.ndarray]] = {}
    for name, young, nu, fraction_name, potential in volume_contract:
        fraction = np.asarray(mesh.cell_data[fraction_name], dtype=np.float64).copy()
        legacy.set_volume_material(mesh, E=young, nu=nu, fraction=fraction)
        if not np.array_equal(
            np.asarray(mesh.cell_data[FRACTION.vtk], dtype=np.float64), fraction
        ):
            msg = f"{case_id}/{name} generic Fraction input binding changed"
            raise RuntimeError(msg)
        builder.add_potential(potential.from_pyvista(mesh, name=name))
        lambda_ = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = young / (2.0 * (1.0 + nu))
        expected_volume[name] = {
            LAMBDA.value: np.full(mesh.n_cells, lambda_, dtype=np.float64),
            MU.value: np.full(mesh.n_cells, mu, dtype=np.float64),
        }
        volume_fraction_inputs[name] = (fraction_name, fraction)
    builder.add_potential(
        Koiter.from_pyvista(candidate_skin, name="skin", thickness=SKIN_THICKNESS_M)
    )
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=legacy.FORWARD_MAX_STEPS,
        atol=legacy.FORWARD_ATOL,
        rtol=legacy.FORWARD_RTOL,
    )
    if int(forward.model.n_fixed) != EXPECTED_MODEL_FIXED_DOFS:
        msg = f"{case_id} forward fixed-DoF count changed"
        raise RuntimeError(msg)
    if int(torch.count_nonzero(forward.state.u.detach()).item()) != 0:
        msg = f"{case_id} new forward state is not exact-zero displacement"
        raise RuntimeError(msg)

    materials = forward.model.get_materials()
    if set(materials) != {"aponeurosis", "fat", "muscle", "skin"}:
        msg = f"{case_id} forward potential names changed: {sorted(materials)}"
        raise RuntimeError(msg)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    if volume.shape != (mesh.n_cells,) or not np.isfinite(volume).all():
        msg = f"{case_id} input tetrahedron Volume field changed"
        raise RuntimeError(msg)
    volume_readback_metrics: dict[str, float] = {}
    for name, expected in expected_volume.items():
        expected_fields = {"dhdX", "dV", LAMBDA.value, MU.value}
        if name == "muscle":
            expected_fields.add(ACTIVATION_INV.value)
        if set(materials[name]) != expected_fields:
            msg = (
                f"{case_id}/{name} material fields changed: "
                f"{sorted(materials[name])} != {sorted(expected_fields)}"
            )
            raise RuntimeError(msg)
        for field, values in expected.items():
            _tensor_exact_readback(
                materials[name][field], values, name=f"{case_id}/{name}/{field}"
            )
        fraction_name, fraction = volume_fraction_inputs[name]
        if not np.array_equal(
            np.asarray(mesh.cell_data[fraction_name], dtype=np.float64), fraction
        ):
            msg = f"{case_id}/{name} named input Fraction changed"
            raise RuntimeError(msg)
        live_dv = materials[name]["dV"].detach().cpu().numpy()
        if live_dv.shape != (mesh.n_cells, 1) or not np.isfinite(live_dv).all():
            msg = f"{case_id}/{name} live tetrahedron integration weights changed"
            raise RuntimeError(msg)
        integrated_dv = np.asarray(live_dv, dtype=np.float64).sum(axis=1)
        expected_integrated_dv = volume * fraction
        if not np.allclose(
            integrated_dv,
            expected_integrated_dv,
            rtol=VOLUME_FRACTION_DV_RTOL,
            atol=VOLUME_FRACTION_DV_ATOL_M3,
        ):
            msg = f"{case_id}/{name} live dV does not encode the named anatomy Fraction"
            raise RuntimeError(msg)
        volume_readback_metrics.update(
            {
                f"readback/volume_{name}_fraction_sum": float(fraction.sum()),
                f"readback/volume_{name}_weighted_volume_m3": float(
                    integrated_dv.sum()
                ),
            }
        )
    expected_skin_fields = {
        ACTIVATION_INV.value,
        FRACTION.value,
        LAMBDA.value,
        MU.value,
        "rest_edge_01",
        "rest_edge_02",
        "rest_metric_inv",
        "rest_metric_sqrt_det",
    }
    if set(materials["skin"]) != expected_skin_fields:
        msg = (
            f"{case_id}/skin material fields changed: "
            f"{sorted(materials['skin'])} != {sorted(expected_skin_fields)}"
        )
        raise RuntimeError(msg)
    for field, vtk_name in (
        (LAMBDA.value, LAMBDA.vtk),
        (MU.value, MU.vtk),
        (FRACTION.value, FRACTION.vtk),
        (ACTIVATION_INV.value, ACTIVATION_INV.vtk),
    ):
        _tensor_exact_readback(
            materials["skin"][field],
            np.asarray(candidate_skin.cell_data[vtk_name], dtype=np.float64),
            name=f"{case_id}/skin/{field}",
        )

    young_mode, prestrain_mode, prestrain_enabled = CASE_CONTRACT[case_id]
    metrics = {
        "case_id": case_id,
        "material/candidate": case_id,
        "material/skin_path": str(skin_path),
        "material/skin_domain": "all-vertex IsFace filtered PolyData",
        "material/skin_young_modulus_mode": young_mode,
        "material/skin_prestrain_mode": prestrain_mode,
        "material/skin_lame_conversion": LAME_CONVERSION,
        "material/skin_koiter_energy_measure": "fixed original reference area",
        "material/volume_lame_conversion": VOLUME_LAME_CONVERSION,
        "skin/enabled": True,
        "skin/prestrain_enabled": prestrain_enabled,
        "skin/young_spatially_varying": young_mode.startswith("selective-"),
        "skin/domain": "all-vertex IsFace filtered PolyData",
        "skin/koiter_input_points": int(candidate_skin.n_points),
        "skin/koiter_input_triangles": int(candidate_skin.n_cells),
        "skin/E_MPa": (
            SKIN_E_MPA
            if not young_mode.startswith("selective-")
            else "heterogeneous [0.003, 0.2]"
        ),
        "skin/nu": SKIN_NU,
        "skin/thickness": SKIN_THICKNESS_M,
        "skin/lame_conversion": LAME_CONVERSION,
        "skin/koiter_energy_measure": "fixed original reference area",
        "volume/lame_conversion": VOLUME_LAME_CONVERSION,
        "protocol/forward_initial_displacement_exact_zero": True,
        "protocol/forward_initial_displacement_max_abs_m": 0.0,
        "readback/model_materials_exact": True,
        "readback/model_material_field_sets_exact": True,
        "readback/volume_named_fraction_inputs_exact": True,
        "readback/volume_fraction_weighted_dv_formula": True,
        "readback/volume_fraction_weighted_dv_rtol": VOLUME_FRACTION_DV_RTOL,
        "readback/volume_fraction_weighted_dv_atol_m3": (VOLUME_FRACTION_DV_ATOL_M3),
        "readback/volume_materials_unchanged": True,
        "readback/skin_materials_exact": True,
        **volume_readback_metrics,
        **cut_metrics,
        **provenance,
    }
    return forward, candidate_skin, metrics


def _validate_history_contract(path: Path, expected_evaluations: int) -> None:
    from vtkmodules.vtkCommonExecutionModel import (
        vtkStreamingDemandDrivenPipeline as StreamingPipeline,
    )

    reader = pv.get_reader(path)
    vtk_reader = reader.reader
    vtk_reader.UpdateInformation()
    information = vtk_reader.GetOutputInformation(0)
    key = StreamingPipeline.TIME_STEPS()
    times = np.asarray(
        [information.Get(key, index) for index in range(information.Length(key))],
        dtype=np.float64,
    )
    expected = np.arange(expected_evaluations, dtype=np.float64)
    if expected_evaluations == 1 and times.size == 0:
        frame = pv.read(path)
        stored = int(np.asarray(frame.field_data["inverse_step"]).reshape(-1)[0])
        if stored != 0:
            msg = "single-frame smoke history does not store inverse step 0"
            raise ValueError(msg)
        return
    if not np.array_equal(times, expected):
        msg = f"history time steps changed: expected {expected.tolist()}, got {times.tolist()}"
        raise ValueError(msg)


def _validate_case(  # noqa: C901
    summary: dict[str, Any],
    paths: Any,
    skin: pv.PolyData,
    cfg: Config,
    *,
    case_id: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors, warnings, diagnostics = legacy.validate_case(summary, paths, skin, cfg)
    exact = {
        "case_id": case_id,
        "material/skin_young_modulus_mode": CASE_CONTRACT[case_id][0],
        "material/skin_prestrain_mode": CASE_CONTRACT[case_id][1],
        "material/skin_lame_conversion": LAME_CONVERSION,
        "material/skin_koiter_energy_measure": "fixed original reference area",
        "material/volume_lame_conversion": VOLUME_LAME_CONVERSION,
        "skin/domain": "all-vertex IsFace filtered PolyData",
        "skin/koiter_input_triangles": EXPECTED_SKIN_TRIANGLES,
        "skin/koiter_input_points": EXPECTED_SKIN_POINTS,
        "skin/nu": SKIN_NU,
        "skin/thickness": SKIN_THICKNESS_M,
        "readback/model_materials_exact": True,
        "readback/model_material_field_sets_exact": True,
        "readback/volume_named_fraction_inputs_exact": True,
        "readback/volume_fraction_weighted_dv_formula": True,
        "readback/volume_fraction_weighted_dv_rtol": VOLUME_FRACTION_DV_RTOL,
        "readback/volume_fraction_weighted_dv_atol_m3": (VOLUME_FRACTION_DV_ATOL_M3),
        "readback/volume_materials_unchanged": True,
        "readback/skin_materials_exact": True,
        "cut_boundary/policy": corrected.HARD_FIXED_CUT_BOUNDARY_POLICY,
        "cut_boundary/incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
        "cut_boundary/incident_global_ids_sha256": (
            EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256
        ),
        "cut_boundary/model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
        "cut_boundary/model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
        "cut_boundary/configured_exact_zero": True,
        "cut_boundary/hard_fixed_is_ground_truth": False,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            errors.append(f"{key} differs from the locked {case_id} contract")
    if int(summary.get("inverse/evaluations", -1)) != cfg.inverse_max_steps + 1:
        errors.append("evaluation count differs from the stage contract")
    if cfg.stage == "smoke":
        trace = summary.get("trace")
        if (
            not isinstance(trace, list)
            or len(trace) != 1
            or int(trace[0].get("step", -1)) != 0
        ):
            errors.append("smoke must contain exactly the step-0 forward/adjoint trace")
        elif float(trace[0].get("grad/norm", math.nan)) <= SMOKE_MIN_GRAD_NORM:
            errors.append(
                "smoke step-0 activation gradient norm is not greater than "
                f"{SMOKE_MIN_GRAD_NORM:.1e}"
            )
    if paths.result.is_file():
        result = pv.read(paths.result)
        if not isinstance(result, pv.UnstructuredGrid):
            errors.append("result is not an UnstructuredGrid")
        else:
            try:
                diagnostics.update(cut_boundary_readback(result))
            except Exception as error:  # noqa: BLE001
                errors.append(f"hard-fixed result readback failed: {error}")
    if paths.history.is_file():
        try:
            _validate_history_contract(paths.history, cfg.inverse_max_steps + 1)
        except Exception as error:  # noqa: BLE001
            errors.append(f"history contract failed: {error}")
    return sorted(set(errors)), warnings, diagnostics


def _artifact_identity_fields(
    paths: Any, *, skin_path: Path, summary_identity: dict[str, int | str] | None = None
) -> dict[str, int | str]:
    fields: dict[str, int | str] = {}
    for name in ("target", "result", "history", "trace"):
        path = Path(getattr(paths, name)).resolve()
        identity = _file_identity(path)
        fields[f"artifact/{name}_path"] = str(path)
        fields[f"artifact/{name}_size_bytes"] = int(identity["size_bytes"])
        fields[f"artifact/{name}_sha256"] = str(identity["sha256"])
    skin_identity = _file_identity(skin_path)
    fields.update(
        {
            "artifact/skin_path": str(skin_path.resolve()),
            "artifact/skin_size_bytes": int(skin_identity["size_bytes"]),
            "artifact/skin_sha256": str(skin_identity["sha256"]),
            "artifact/summary_path": str(Path(paths.summary).resolve()),
        }
    )
    if summary_identity is not None:
        fields["artifact/summary_size_bytes"] = int(summary_identity["size_bytes"])
        fields["artifact/summary_sha256"] = str(summary_identity["sha256"])
    return fields


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | status | evals | best step | target RMS mm | residual Lap RMS m | activation RMS | inv tets | folds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row['case_id']} | {row.get('status')} | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| {case_id} | ok | {inverse/evaluations} | {best/step} | "
            "{best/error_rms_mm:.9g} | {bumpiness/residual_laplacian_rms:.9g} | "
            "{activation_inv/rms:.9g} | {warning/inverted_tets} | "
            "{warning/skin_folded_triangles} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_header(
    cfg: Config,
    *,
    manifest: dict[str, Any],
    input_identities: dict[str, Any],
    producer_identity: dict[str, int | str],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "complete": False,
        "design": DESIGN,
        "experiment": (
            "HFP0/HFP1 selective fat-modulus floor and c020 inverse with exact-pinned "
            "H0P0/H0P1 controls"
        ),
        "stage": cfg.stage,
        "case_order": list(CASE_ORDER),
        "meeting_case_order": [*CONTROL_ORDER, *CASE_ORDER],
        "n_cases": len(CASE_ORDER),
        "inverse_lr": cfg.inverse_lr,
        "inverse_optimizer_steps": cfg.inverse_max_steps,
        "inverse_evaluations": cfg.inverse_max_steps + 1,
        "execution": "sequential",
        "fresh_zero_activation": True,
        "fresh_zero_displacement": True,
        "activation_mode": "per-muscle-tet-6dof-unconstrained",
        "activation_shared": False,
        "activation_transferred": False,
        "displacement_transferred": False,
        "optimizer_state_shared": False,
        "forward_builder_shared": False,
        "constitutive_policy": {
            "skin_domain": "all-vertex IsFace filtered PolyData",
            "skin_lame_conversion": LAME_CONVERSION,
            "skin_thickness_m": SKIN_THICKNESS_M,
            "skin_nu": SKIN_NU,
            "skin_reference_energy_measure": "fixed original reference area",
            "selective_rule": (
                "E=fat modulus (0.003 MPa) iff raw target/rest area ratio > 1; "
                "otherwise E=0.2 MPa"
            ),
            "c020_rule": "rho=0.98^2*clip(raw target/rest area ratio,0.5,1)",
            "HFP1_interaction": (
                "c020 is stored and mechanically active on all 29,899 IsFace "
                "triangles because the Young's-modulus floor is nonzero"
            ),
            "volume_lame_conversion": VOLUME_LAME_CONVERSION,
            "fat_E_MPa": float(legacy.FAT_E),
            "fat_nu": float(legacy.FAT_NU),
            "muscle_E_MPa": float(legacy.MUSCLE_E),
            "muscle_nu": float(legacy.MUSCLE_NU),
            "aponeurosis_E_MPa": float(legacy.APONEUROSIS_E),
            "aponeurosis_nu": float(legacy.APONEUROSIS_NU),
        },
        "boundary_policy": {
            "policy": corrected.HARD_FIXED_CUT_BOUNDARY_POLICY,
            "incident_vertices": EXPECTED_CUT_INCIDENT_VERTICES,
            "incident_global_ids_sha256": EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256,
            "model_total_fixed_vertices": EXPECTED_MODEL_FIXED_VERTICES,
            "model_total_fixed_dofs": EXPECTED_MODEL_FIXED_DOFS,
            "configured_value_m": 0.0,
            "interpretation": "user-approved conservative approximation",
        },
        "baseline_comparators": manifest["controls"],
        "inputs": input_identities,
        "implementation": {
            "producer_path": str(Path(__file__).resolve()),
            "producer_size_bytes": int(producer_identity["size_bytes"]),
            "producer_sha256": str(producer_identity["sha256"]),
            "producer_unchanged_through_run": False,
            "prepare_path": str(PREPARE_IMPLEMENTATION),
            "prepare_sha256": input_identities["prepare_implementation"]["sha256"],
            "prepare_review_normalized_sha256": (
                EXPECTED_PREPARE_IMPLEMENTATION_SHA256
            ),
            "corrected_inverse_reference_path": str(CORRECTED_INVERSE_REFERENCE),
            "corrected_inverse_reference_sha256": CORRECTED_INVERSE_REFERENCE_SHA256,
            "plane_stress_reference_path": str(PLANE_STRESS_REFERENCE),
            "plane_stress_reference_sha256": PLANE_STRESS_REFERENCE_SHA256,
            "runtime_bundle": runtime_identity,
            "koiter/path": str(KOITER_IMPLEMENTATION),
            "koiter/size_bytes": int(
                input_identities["koiter_implementation"]["size_bytes"]
            ),
            "koiter/sha256": str(input_identities["koiter_implementation"]["sha256"]),
            "koiter_unchanged_through_run": False,
            "runtime_unchanged_through_run": False,
            "input_manifest_unchanged_through_run": False,
            "candidate_skins_unchanged_through_run": False,
            "material_preparation_sources_unchanged_through_run": False,
        },
        "artifact_contract": {
            "case_stems": {case.case_id: case.stem for case in _cases()},
            "history_format": "VTKHDFTemporalUnstructuredGrid",
            "history_time_steps": list(range(cfg.inverse_max_steps + 1)),
            "trace_steps": list(range(cfg.inverse_max_steps + 1)),
            "result_state": "best saved inverse state",
            "history_state": "every evaluated inverse state",
            "case_row_identity_keys": [
                f"artifact/{name}_{suffix}"
                for name in ("summary", "trace", "history", "result", "target", "skin")
                for suffix in ("path", "size_bytes", "sha256")
            ],
        },
        "acceptance_policy": {
            "hard": [
                "all forward and adjoint solves succeed",
                "finite complete fixed-budget trace",
                "exact-zero fresh activation and displacement",
                "exact material/model readback",
                "all artificial-cut incident vertices remain exactly fixed",
                "readable exact-frame artifacts",
            ],
            "warning_only_pending_visual_review": [
                "small inverted tetrahedron count",
                "small folded skin-triangle count",
            ],
        },
        "visual_review": {
            "tool": "ParaView",
            "status": "pending",
            "producer_does_not_render": True,
        },
        "producer_identity": {
            "path": str(Path(__file__).resolve()),
            "size_bytes": int(producer_identity["size_bytes"]),
            "sha256": str(producer_identity["sha256"]),
            "unchanged_through_all_runs": False,
        },
        "execution_contract": {
            "stage": cfg.stage,
            "case_order": list(CASE_ORDER),
            "sequential": True,
            "optimizer": "Adam",
            "optimizer_updates_per_case": cfg.inverse_max_steps,
            "evaluations_per_case": cfg.inverse_max_steps + 1,
            "fresh_zero_activation_per_case": True,
            "fresh_zero_displacement_per_case": True,
            "independent_forward_and_optimizer_per_case": True,
            "smoke_step0_grad_norm_must_exceed": SMOKE_MIN_GRAD_NORM,
        },
        "hard_failures": [],
        "cases": [],
    }


def run(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    _validate_config(cfg)
    _require_execution_approval(cfg)
    _refuse_stale_outputs(cfg)

    producer_identity = _file_identity(Path(__file__).resolve())
    input_mesh_identity = _require_file_identity(
        cfg.input_mesh, EXPECTED_INPUT_MESH_IDENTITY, name="prepared volume mesh"
    )
    cut_reference_identity = _require_file_identity(
        cfg.input_cut_reference,
        {
            "size_bytes": plane_reference.SOURCE_SKIN_SIZE_BYTES,
            "sha256": plane_reference.SOURCE_SKIN_SHA256,
        },
        name="artificial-cut topology reference",
    )
    manifest, by_id = _load_manifest(cfg)
    manifest_identity = _file_identity(cfg.input_manifest)
    prepare_identity = _file_identity(PREPARE_IMPLEMENTATION)
    runtime_identity = require_inverse_runtime_identity(context="pre-solve")
    koiter_identity = _require_file_identity(
        KOITER_IMPLEMENTATION,
        EXPECTED_KOITER_IMPLEMENTATION_IDENTITY,
        name="Koiter implementation",
    )
    for path in (
        cfg.input_mesh,
        cfg.input_cut_reference,
        cfg.input_manifest,
        PREPARE_IMPLEMENTATION,
        CORRECTED_INVERSE_REFERENCE,
        PLANE_STRESS_REFERENCE,
        KOITER_IMPLEMENTATION,
    ):
        cherries.log_input(path)
    for _, path, _ in corrected.INVERSE_RUNTIME_DEPENDENCIES:
        cherries.log_input(path)

    base_mesh = pv.read(cfg.input_mesh)
    if not isinstance(base_mesh, pv.UnstructuredGrid):
        msg = f"prepared mesh read as {type(base_mesh).__name__}"
        raise TypeError(msg)
    verified = _verified_skins(by_id, base_mesh)
    initial_skin_identities = {
        case_id: _file_identity(values[0]) for case_id, values in verified.items()
    }
    preparation_sources = _rehash_identity_tree(
        manifest["runtime/verified_preparation_inputs"],
        name="material_preparation_sources",
    )
    input_identities = {
        "manifest": {"path": str(cfg.input_manifest), **manifest_identity},
        "prepared_mesh": {"path": str(cfg.input_mesh), **input_mesh_identity},
        "cut_reference": {
            "path": str(cfg.input_cut_reference),
            **cut_reference_identity,
        },
        "prepare_implementation": {
            "path": str(PREPARE_IMPLEMENTATION),
            **prepare_identity,
        },
        "koiter_implementation": {
            "path": str(KOITER_IMPLEMENTATION),
            **koiter_identity,
        },
        "candidate_skins": {
            case_id: {"path": str(values[0]), **initial_skin_identities[case_id]}
            for case_id, values in verified.items()
        },
        "material_preparation_sources": preparation_sources,
        "control_artifacts": manifest["runtime/verified_controls"],
    }
    aggregate = _aggregate_header(
        cfg,
        manifest=manifest,
        input_identities=input_identities,
        producer_identity=producer_identity,
        runtime_identity=runtime_identity,
    )
    _atomic_write_json(cfg.output_summary, aggregate)

    legacy.configure_runtime()
    original_builder = reference_runtime.build_forward
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    for case in _cases():
        case_id = case.case_id
        skin_path, skin, provenance = verified[case_id]
        row_spec = by_id[case_id]
        paths = _artifact_paths(cfg.output_summary.parent, case)
        builder_calls = 0

        def independent_builder(
            mesh: pv.UnstructuredGrid,
            inverse_case: BatchCase,
            *,
            area_ratio_floor: float,
            _skin_path: Path = skin_path,
            _skin: pv.PolyData = skin,
            _row: dict[str, Any] = row_spec,
            _provenance: dict[str, Any] = provenance,
            _case_id: str = case_id,
        ) -> tuple[Any, pv.PolyData, dict[str, Any]]:
            nonlocal builder_calls
            builder_calls += 1
            if builder_calls != 1:
                msg = f"{_case_id} requested more than one forward builder"
                raise RuntimeError(msg)
            return _build_forward(
                mesh,
                inverse_case,
                area_ratio_floor=area_ratio_floor,
                skin_path=_skin_path,
                skin=_skin,
                row=_row,
                provenance=_provenance,
            )

        reference_runtime.build_forward = independent_builder
        try:
            summary = legacy.normalize_summary(
                legacy.solve_case(case, base_mesh.copy(deep=True), cfg)
            )
            errors, warnings, diagnostics = _validate_case(
                summary, paths, skin, cfg, case_id=case_id
            )
            if builder_calls != 1:
                errors.append(f"forward builder was called {builder_calls} times")
            enriched = {
                **summary,
                **diagnostics,
                "schema_version": AGGREGATE_SCHEMA_VERSION,
                "design": DESIGN,
                "stage": cfg.stage,
                "case_id": case_id,
                "status": "ok" if not errors else "invalid",
                "builder/fresh_independent": builder_calls == 1,
                "builder/calls": builder_calls,
                "protocol/fresh_zero_activation": True,
                "protocol/fresh_zero_displacement": True,
                "protocol/independent_optimizer_state": True,
                "protocol/optimizer_steps": cfg.inverse_max_steps,
                "protocol/evaluations": cfg.inverse_max_steps + 1,
                "validation/errors": sorted(set(errors)),
                "validation/warnings": sorted(set(warnings)),
                **_artifact_identity_fields(paths, skin_path=skin_path),
            }
            _atomic_write_json(paths.summary, enriched)
            final_summary = _canonical_case_summary(paths.summary)
            final_summary.write_bytes(paths.summary.read_bytes())
            if final_summary.read_bytes() != paths.summary.read_bytes():
                msg = f"{case_id} canonical summary differs from live summary"
                raise RuntimeError(msg)  # noqa: TRY301
            summary_identity = _file_identity(paths.summary)
            final_summary_identity = _file_identity(final_summary)
            enriched.update(
                {
                    **_artifact_identity_fields(
                        paths,
                        skin_path=skin_path,
                        summary_identity=summary_identity,
                    ),
                    "artifact/canonical_summary_path": str(final_summary.resolve()),
                    "artifact/canonical_summary_size_bytes": int(
                        final_summary_identity["size_bytes"]
                    ),
                    "artifact/canonical_summary_sha256": str(
                        final_summary_identity["sha256"]
                    ),
                }
            )
            rows.append(enriched)
            cherries.log_output(final_summary)
            if errors:
                hard_failures.append(f"{case_id}: " + "; ".join(sorted(set(errors))))
        except Exception as error:
            logger.exception("inverse case %s failed", case_id)
            failed = {
                "schema_version": AGGREGATE_SCHEMA_VERSION,
                "design": DESIGN,
                "stage": cfg.stage,
                "case_id": case_id,
                "status": "failed",
                "builder/calls": builder_calls,
                "error": f"{type(error).__name__}: {error}",
                "artifact/summary_path": str(paths.summary.resolve()),
                "artifact/skin_path": str(skin_path.resolve()),
                "artifact/skin_size_bytes": int(
                    initial_skin_identities[case_id]["size_bytes"]
                ),
                "artifact/skin_sha256": str(initial_skin_identities[case_id]["sha256"]),
            }
            paths.summary.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(paths.summary, failed)
            rows.append(failed)
            hard_failures.append(f"{case_id}: {type(error).__name__}: {error}")
        finally:
            reference_runtime.build_forward = original_builder
        aggregate["cases"] = rows
        aggregate["hard_failures"] = hard_failures
        _atomic_write_json(cfg.output_summary, aggregate)
        _write_table(cfg.output_table, rows)

    post_runtime_identity = require_inverse_runtime_identity(context="post-solve")
    post_skin_identities = {
        case_id: _file_identity(values[0]) for case_id, values in verified.items()
    }
    post_preparation_sources = _rehash_identity_tree(
        input_identities["material_preparation_sources"],
        name="material_preparation_sources",
    )
    unchanged = {
        "producer": _file_identity(Path(__file__).resolve()) == producer_identity,
        "manifest": _file_identity(cfg.input_manifest) == manifest_identity,
        "prepare": _file_identity(PREPARE_IMPLEMENTATION) == prepare_identity,
        "prepared_mesh": _file_identity(cfg.input_mesh) == input_mesh_identity,
        "cut_reference": _file_identity(cfg.input_cut_reference)
        == cut_reference_identity,
        "runtime": post_runtime_identity == runtime_identity,
        "koiter": _file_identity(KOITER_IMPLEMENTATION) == koiter_identity,
        "candidate_skins": post_skin_identities == initial_skin_identities,
        "preparation_sources": post_preparation_sources
        == input_identities["material_preparation_sources"],
    }
    if not all(unchanged.values()):
        msg = f"inputs or implementations changed during {cfg.stage}: {unchanged}"
        hard_failures.append(msg)
    aggregate["cases"] = rows
    aggregate["hard_failures"] = hard_failures
    aggregate["post_run_identity_checks"] = unchanged
    aggregate["implementation"]["producer_unchanged_through_run"] = unchanged[
        "producer"
    ]
    aggregate["implementation"]["runtime_unchanged_through_run"] = unchanged["runtime"]
    aggregate["implementation"]["koiter_unchanged_through_run"] = unchanged["koiter"]
    aggregate["implementation"]["input_manifest_unchanged_through_run"] = unchanged[
        "manifest"
    ]
    aggregate["implementation"]["candidate_skins_unchanged_through_run"] = unchanged[
        "candidate_skins"
    ]
    aggregate["implementation"][
        "material_preparation_sources_unchanged_through_run"
    ] = unchanged["preparation_sources"]
    aggregate["producer_identity"]["unchanged_through_all_runs"] = unchanged["producer"]
    aggregate["complete"] = not hard_failures and len(rows) == len(CASE_ORDER)
    _atomic_write_json(cfg.output_summary, aggregate)
    _write_table(cfg.output_table, rows)
    if not aggregate["complete"]:
        for path in (cfg.output_summary, cfg.output_table):
            if path.is_file():
                cherries.log_output(path)
        msg = f"{cfg.stage} batch failed: " + " | ".join(hard_failures)
        raise RuntimeError(msg)

    final_aggregate = _canonical_aggregate(cfg.output_summary)
    final_aggregate.write_bytes(cfg.output_summary.read_bytes())
    if final_aggregate.read_bytes() != cfg.output_summary.read_bytes():
        msg = "canonical aggregate differs from live final aggregate"
        raise RuntimeError(msg)
    for path in (cfg.output_summary, cfg.output_table, final_aggregate):
        cherries.log_output(path)
    logger.info(
        "Wrote completed %s HFP1/HFP0 batch under %s",
        cfg.stage,
        cfg.output_summary.parent,
    )


if __name__ == "__main__":
    cherries.main(run)
