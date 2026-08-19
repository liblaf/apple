from __future__ import annotations

# Experiment-local validation is intentionally fail-closed and explicit.
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import hashlib
import json
import logging
import math
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
DESIGN = "meeting-lame-conversion-only-native-paraview-v1"
EXPECTED_PARAVIEW_VERSION = "6.1.1"
IMAGE_RESOLUTION = (1800, 1800)
STRAIN_LIMIT_PERCENT = 7.322
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_POOLED_ABS_P99_PERCENT = 7.3224542093414104

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = GROUP_DIR / "data"
SOURCE_SKIN = (
    REPO_ROOT
    / "exp/2026/08/17/human-face-smile-material-heuristic-sweep"
    / "data/10-material-candidates/skin-e100-p000.vtp"
)
PROBE_ROOT = (
    REPO_ROOT
    / "exp/2026/08/18/human-face-smile-plane-stress-skin"
    / "data/15-forward-domain-conversion-probe"
)
OLD_RESULT = PROBE_ROOT / "full-3d-replay/zero/result.vtu"
CORRECTED_RESULT = PROBE_ROOT / "full-plane-stress/zero/result.vtu"
ANALYSIS = (
    REPO_ROOT
    / "exp/2026/08/18/human-face-smile-plane-stress-skin"
    / "data/16-forward-domain-conversion-analysis.json"
)
MODULI_SOURCE = REPO_ROOT / "src/liblaf/apple/common/_moduli.py"
KOITER_SOURCE = REPO_ROOT / "src/liblaf/apple/warp/fem/_koiter.py"
RENDERER = Path(__file__).with_name("10-lame-paraview.py")
PVBATCH = Path("/usr/bin/pvbatch")

INPUT_ROOT = DATA_DIR / "10-lame-inputs"
ASSET_ROOT = DATA_DIR / "10-lame-assets"
CONTRACT = DATA_DIR / "10-lame-contract.json"
RECEIPT = DATA_DIR / "10-lame-receipt.json"


@dataclass(frozen=True)
class IdentitySpec:
    path: Path
    size_bytes: int
    sha256: str


INPUT_SPECS = {
    "source_skin": IdentitySpec(
        SOURCE_SKIN,
        38_742_137,
        "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
    ),
    "old_result": IdentitySpec(
        OLD_RESULT,
        148_098_752,
        "d533ff178dcb86bb1fae9c3ad08563f8413261c76a93e809b96f4099df076f97",
    ),
    "corrected_result": IdentitySpec(
        CORRECTED_RESULT,
        148_094_452,
        "c04b3ce317aa14d825032632c4d1d35f6a591d6ba171171ca257b14c38d462b4",
    ),
    "analysis": IdentitySpec(
        ANALYSIS,
        35_544,
        "c7e7e19456ea2cf29d91771ac377297a93af57cf9d91a4ad5fa8efc596eebdf9",
    ),
    "moduli_source": IdentitySpec(
        MODULI_SOURCE,
        1_210,
        "9d5c14f27b9a08a8a4f9cd3ce4e3076f2375ed1108e84e94d307c9439e1a303d",
    ),
    "koiter_source": IdentitySpec(
        KOITER_SOURCE,
        17_329,
        "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c",
    ),
    "pvbatch": IdentitySpec(
        PVBATCH,
        18_608,
        "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871",
    ),
}

CASE_ORDER = ("old-3d", "corrected-plane-stress")
CASE_RESULTS = {
    "old-3d": OLD_RESULT,
    "corrected-plane-stress": CORRECTED_RESULT,
}
ASSET_SPECS = (
    ("10-lame-old-3d-geometry", "old-3d", "geometry"),
    (
        "10-lame-corrected-plane-stress-geometry",
        "corrected-plane-stress",
        "geometry",
    ),
    ("10-lame-old-3d-area-strain", "old-3d", "area-strain"),
    (
        "10-lame-corrected-plane-stress-area-strain",
        "corrected-plane-stress",
        "area-strain",
    ),
)


class Config(cherries.BaseConfig):
    output_receipt: Path = cherries.output(RECEIPT, mkdir=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale Lamé JSON output: {path}")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    if _read_json(temporary) != payload:
        raise RuntimeError(f"temporary JSON readback changed: {path}")
    temporary.replace(path)
    if _read_json(path) != payload:
        raise RuntimeError(f"final JSON readback changed: {path}")
    return _identity(path)


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vectors = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    area = 0.5 * np.linalg.norm(vectors, axis=1)
    if not np.isfinite(area).all() or np.any(area <= np.finfo(np.float64).eps):
        raise ValueError("Lamé IsFace surface contains a degenerate triangle")
    return area


def _field_scalar(mesh: pv.DataSet, name: str) -> float:
    if name not in mesh.field_data:
        raise KeyError(f"missing result field {name}")
    values = np.asarray(mesh.field_data[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values).all():
        raise ValueError(f"malformed result field {name}")
    return float(values[0])


def _canonical_skin() -> dict[str, np.ndarray]:
    skin = pv.read(SOURCE_SKIN)
    if not isinstance(skin, pv.PolyData) or skin.n_cells != 128_172:
        raise ValueError("source skin topology changed")
    encoded = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.shape != (128_172, 4) or not np.all(encoded[:, 0] == 3):
        raise ValueError("source skin is no longer triangular")
    face_mask = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    if int(face_mask.sum()) != EXPECTED_SKIN_TRIANGLES:
        raise ValueError("canonical IsFace triangle count changed")
    source_triangles = encoded[face_mask, 1:]
    used, triangles = np.unique(source_triangles, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    if used.size != EXPECTED_SKIN_POINTS:
        raise ValueError("canonical IsFace point count changed")
    points = np.asarray(skin.points, dtype=np.float64)[used]
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)[used]
    if np.unique(global_ids).size != EXPECTED_SKIN_POINTS:
        raise ValueError("canonical IsFace GlobalPointId is not unique")
    rest_area = _triangle_area(points, triangles)
    if not math.isclose(
        float(rest_area.sum()), EXPECTED_SKIN_AREA_M2, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise ValueError("canonical IsFace area changed")
    return {
        "points": points,
        "global_ids": global_ids,
        "triangles": triangles,
        "rest_area": rest_area,
    }


def _load_result(
    case_id: str,
    path: Path,
    skin_global_ids: np.ndarray,
    canonical: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    mesh = pv.read(path)
    if (
        not isinstance(mesh, pv.UnstructuredGrid)
        or mesh.n_points != EXPECTED_POINTS
        or mesh.n_cells != EXPECTED_TETS
    ):
        raise ValueError(f"{case_id} result dimensions changed")
    global_ids = np.asarray(mesh.point_data["GlobalPointId"], dtype=np.int64)
    order = np.argsort(global_ids)
    sorted_ids = global_ids[order]
    positions = np.searchsorted(sorted_ids, skin_global_ids)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(
        sorted_ids[positions], skin_global_ids
    ):
        raise ValueError(f"{case_id} result does not contain canonical IsFace points")
    skin_indices = order[positions]
    fields = {
        "rest_points": np.asarray(mesh.points, dtype=np.float64),
        "global_ids": global_ids,
        "displacement": np.asarray(mesh.point_data["Displacement"], dtype=np.float64),
        "target": np.asarray(
            mesh.point_data["TargetDisplacement"], dtype=np.float64
        ),
        "activation": np.asarray(mesh.cell_data["Activation"], dtype=np.float64),
        "is_fixed": np.asarray(mesh.point_data["IsFixed"], dtype=bool),
        "skin_indices": skin_indices,
    }
    for name, values in fields.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{case_id} {name} contains non-finite values")
    if canonical is not None:
        for name in ("rest_points", "global_ids", "target", "activation", "is_fixed"):
            if not np.array_equal(fields[name], canonical[name]):
                raise ValueError(f"{case_id} changed fixed probe field {name}")
    expected_lambda = 3.288590604026843 if case_id == "old-3d" else 0.1289643374128175
    for name, expected in (
        ("skin/E_MPa", 0.2),
        ("skin/nu_3d_input", 0.49),
        ("skin/thickness_m", 0.001),
        ("skin/mu_MPa", 0.06711409395973154),
        ("skin/lambda_MPa", expected_lambda),
        ("initial_displacement/rms_m", 0.0),
    ):
        if not math.isclose(
            _field_scalar(mesh, name), expected, rel_tol=1.0e-13, abs_tol=1.0e-15
        ):
            raise ValueError(f"{case_id} field {name} changed")
    return fields


def _write_surface(
    *,
    path: Path,
    case_id: str,
    skin: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    skin_indices = result["skin_indices"]
    displacement = result["displacement"][skin_indices]
    target = result["target"][skin_indices]
    points = skin["points"] + displacement
    deformed_area = _triangle_area(points, skin["triangles"])
    area_ratio = deformed_area / skin["rest_area"]
    area_strain_percent = 100.0 * (area_ratio - 1.0)
    faces = np.column_stack(
        (np.full(EXPECTED_SKIN_TRIANGLES, 3, dtype=np.int64), skin["triangles"])
    )
    surface = pv.PolyData(points, faces)
    arrays = {
        "GlobalPointId": skin["global_ids"],
        "DisplacementMM": 1.0e3 * displacement,
        "TargetDisplacementMM": 1.0e3 * target,
    }
    for name, values in arrays.items():
        surface.point_data[name] = values
    surface.cell_data["AreaRatio"] = area_ratio
    surface.cell_data["AreaStrainPercent"] = area_strain_percent
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing stale Lamé VTP output: {path}")
    surface.save(temporary)
    loaded = pv.read(temporary)
    if (
        not isinstance(loaded, pv.PolyData)
        or loaded.n_points != EXPECTED_SKIN_POINTS
        or loaded.n_cells != EXPECTED_SKIN_TRIANGLES
        or not np.array_equal(np.asarray(loaded.points), points)
    ):
        raise ValueError(f"{case_id} temporary VTP readback changed")
    for name, values in arrays.items():
        if not np.array_equal(np.asarray(loaded.point_data[name]), values):
            raise ValueError(f"{case_id} temporary VTP {name} changed")
    for name, values in (
        ("AreaRatio", area_ratio),
        ("AreaStrainPercent", area_strain_percent),
    ):
        if not np.array_equal(np.asarray(loaded.cell_data[name]), values):
            raise ValueError(f"{case_id} temporary VTP {name} changed")
    temporary.replace(path)
    identity = _identity(path)
    weights = skin["rest_area"]
    metrics = {
        "mean_area_ratio": float(np.dot(weights, area_ratio) / weights.sum()),
        "rest_area_weighted_mean_abs_area_strain_percent": float(
            np.dot(weights, np.abs(area_strain_percent)) / weights.sum()
        ),
        "rest_area_weighted_rms_area_strain_percent": float(
            np.sqrt(np.dot(weights, np.square(area_strain_percent)) / weights.sum())
        ),
        "area_strain_percent_min": float(area_strain_percent.min()),
        "area_strain_percent_max": float(area_strain_percent.max()),
        "display_clipped_triangles": int(
            np.sum(np.abs(area_strain_percent) > STRAIN_LIMIT_PERCENT)
        ),
    }
    receipt = {
        **identity,
        "case_id": case_id,
        "points": EXPECTED_SKIN_POINTS,
        "triangles": EXPECTED_SKIN_TRIANGLES,
        "point_array_hashes": {
            "GlobalPointId": _raw_sha256(arrays["GlobalPointId"], dtype="<i8"),
            "DisplacementMM": _raw_sha256(arrays["DisplacementMM"], dtype="<f8"),
            "TargetDisplacementMM": _raw_sha256(
                arrays["TargetDisplacementMM"], dtype="<f8"
            ),
        },
        "cell_array_hashes": {
            "AreaRatio": _raw_sha256(area_ratio, dtype="<f8"),
            "AreaStrainPercent": _raw_sha256(
                area_strain_percent, dtype="<f8"
            ),
        },
        "metrics": metrics,
    }
    return receipt, points, area_strain_percent


def _validate_analysis() -> dict[str, dict[str, Any]]:
    analysis = _read_json(ANALYSIS)
    if analysis.get("complete") is not True:
        raise ValueError("pinned conversion analysis is incomplete")
    wanted = {("full-3d-replay", "zero"), ("full-plane-stress", "zero")}
    rows = {
        (str(row["case"]), str(row["seed"])): row for row in analysis["cases"]
    }
    if not wanted <= rows.keys():
        raise KeyError("pinned conversion analysis lost the zero-seed full-domain pair")
    output = {
        "old-3d": rows[("full-3d-replay", "zero")],
        "corrected-plane-stress": rows[("full-plane-stress", "zero")],
    }
    for case_id, path in CASE_RESULTS.items():
        row = output[case_id]
        spec = INPUT_SPECS[f"{'old' if case_id == 'old-3d' else 'corrected'}_result"]
        if (
            Path(str(row["artifact/result_path"])).resolve() != path.resolve()
            or int(row["artifact/result_size_bytes"]) != spec.size_bytes
            or str(row["artifact/result_sha256"]) != spec.sha256
        ):
            raise ValueError(f"analysis/result identity changed for {case_id}")
    return output


def _camera(points: np.ndarray) -> dict[str, Any]:
    low, high = points.min(axis=0), points.max(axis=0)
    focus = 0.5 * (low + high)
    extent = high - low
    parallel_scale = 1.10 * 0.5 * max(float(extent[0]), float(extent[1]))
    if not np.isfinite(focus).all() or parallel_scale <= 0.0:
        raise ValueError("invalid Lamé ParaView camera")
    return {
        "direction": [0.0, 0.0, 1.0],
        "focus": focus.tolist(),
        "parallel_scale": parallel_scale,
        "projection": "parallel",
        "basis": "pooled deformed IsFace bounds across both conversion-only cases",
    }


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _paraview_version() -> tuple[str, dict[str, Any]]:
    identity = _require_identity("pvbatch", INPUT_SPECS["pvbatch"])
    completed = subprocess.run(
        [str(PVBATCH), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    combined = f"{completed.stdout}\n{completed.stderr}".strip()
    if not combined.endswith(EXPECTED_PARAVIEW_VERSION):
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, got {combined!r}"
        )
    return EXPECTED_PARAVIEW_VERSION, identity


def _validate_outputs() -> list[dict[str, Any]]:
    expected = sorted(
        path
        for asset_id, _, _ in ASSET_SPECS
        for path in (
            ASSET_ROOT / f"{asset_id}.png",
            ASSET_ROOT / f"{asset_id}.pvsm",
        )
    )
    actual = sorted(path for path in ASSET_ROOT.iterdir() if path.is_file())
    if actual != expected or any(".tmp" in path.name for path in actual):
        raise ValueError("Lamé ParaView output inventory changed")
    outputs: list[dict[str, Any]] = []
    png_hashes: set[str] = set()
    for path in expected:
        identity = _identity(path)
        if path.suffix == ".png":
            if _png_size(path) != IMAGE_RESOLUTION or path.stat().st_size < 100_000:
                raise ValueError(f"Lamé PNG size or payload changed: {path}")
            png_hashes.add(identity["sha256"])
        else:
            head = path.read_text(encoding="utf-8", errors="strict")[:1024]
            if "ParaView" not in head and "ServerManagerState" not in head:
                raise ValueError(f"unrecognized ParaView state: {path}")
        outputs.append(identity)
    if len(png_hashes) != 4:
        raise ValueError("Lamé images are not four distinct renders")
    return outputs


def main(cfg: Config) -> None:
    if cfg.output_receipt.resolve() != RECEIPT.resolve():
        raise ValueError("Lamé receipt path changed")
    wrapper_before = _identity(Path(__file__))
    renderer_before = _identity(RENDERER)
    inputs_before = {
        label: _require_identity(label, spec) for label, spec in INPUT_SPECS.items()
    }
    paraview_version, pvbatch_identity = _paraview_version()
    analysis_rows = _validate_analysis()
    if any(path.exists() for path in (INPUT_ROOT, ASSET_ROOT, CONTRACT, RECEIPT)):
        raise FileExistsError("refusing stale 10-lame meeting outputs")
    INPUT_ROOT.mkdir(parents=True)
    ASSET_ROOT.mkdir()

    skin = _canonical_skin()
    results: dict[str, dict[str, np.ndarray]] = {}
    canonical: dict[str, np.ndarray] | None = None
    for case_id in CASE_ORDER:
        result = _load_result(
            case_id, CASE_RESULTS[case_id], skin["global_ids"], canonical
        )
        if canonical is None:
            canonical = {
                name: values
                for name, values in result.items()
                if name not in {"displacement", "skin_indices"}
            }
        results[case_id] = result

    surface_receipts: dict[str, dict[str, Any]] = {}
    points: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    for case_id in CASE_ORDER:
        receipt, case_points, case_strain = _write_surface(
            path=INPUT_ROOT / f"10-lame-{case_id}.vtp",
            case_id=case_id,
            skin=skin,
            result=results[case_id],
        )
        surface_receipts[case_id] = receipt
        points.append(case_points)
        strains.append(case_strain)

    pooled_abs_p99 = float(np.quantile(np.abs(np.concatenate(strains)), 0.99))
    if not math.isclose(
        pooled_abs_p99,
        EXPECTED_POOLED_ABS_P99_PERCENT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or round(pooled_abs_p99, 3) != STRAIN_LIMIT_PERCENT:
        raise ValueError("Lamé shared area-strain scale changed")

    assets = [
        {
            "asset_id": asset_id,
            "case_id": case_id,
            "mode": mode,
            "png_path": str((ASSET_ROOT / f"{asset_id}.png").resolve()),
            "state_path": str((ASSET_ROOT / f"{asset_id}.pvsm").resolve()),
        }
        for asset_id, case_id, mode in ASSET_SPECS
    ]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case_order": list(CASE_ORDER),
        "layout": "one-view-per-file-no-contact-sheet",
        "image_resolution": list(IMAGE_RESOLUTION),
        "camera": _camera(np.vstack(points)),
        "strain_limit_percent": STRAIN_LIMIT_PERCENT,
        "strain_limit_definition": (
            "symmetric rounded pooled absolute p99 of AreaStrainPercent across "
            "both zero-seed conversion-only IsFace surfaces"
        ),
        "pooled_absolute_p99_percent_unrounded": pooled_abs_p99,
        "inputs": {
            case_id: {
                **surface_receipts[case_id],
                "display_label": (
                    "old 3D Lamé" if case_id == "old-3d" else "plane-stress Lamé"
                ),
            }
            for case_id in CASE_ORDER
        },
        "assets": assets,
        "renderer": (
            "ParaView 6.1.1 native one-view-per-file geometry and cell-scalar "
            "rendering; PyVista used only for audited VTP preparation/readback"
        ),
    }
    contract_identity = _write_json_atomic(CONTRACT, contract)

    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--contract",
        str(CONTRACT.resolve()),
        "--input-root",
        str(INPUT_ROOT.resolve()),
        "--output-root",
        str(ASSET_ROOT.resolve()),
    ]
    logger.info("Running native ParaView Lamé renderer: %s", command)
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
    if _read_json(CONTRACT) != contract or _identity(CONTRACT) != contract_identity:
        raise ValueError("Lamé contract changed during ParaView rendering")
    outputs = _validate_outputs()

    inputs_after = {
        label: _require_identity(label, spec) for label, spec in INPUT_SPECS.items()
    }
    if inputs_before != inputs_after:
        raise RuntimeError("pinned Lamé inputs changed during rendering")
    if _identity(Path(__file__)) != wrapper_before:
        raise RuntimeError("Lamé wrapper changed during rendering")
    if _identity(RENDERER) != renderer_before:
        raise RuntimeError("Lamé ParaView renderer changed during rendering")
    if _identity(PVBATCH) != pvbatch_identity:
        raise RuntimeError("pvbatch changed during rendering")

    material = {
        "E_MPa": 0.2,
        "nu": 0.49,
        "thickness_m": 0.001,
        "prestrain": "none",
        "mu_MPa_both": 0.06711409395973154,
        "lambda_MPa_old_3d": 3.288590604026843,
        "lambda_MPa_corrected_plane_stress": 0.1289643374128175,
        "lambda_old_over_corrected": 25.5,
        "isotropic_area_coefficient_old_over_corrected": 17.114093959731527,
        "volume_material_conversion": "3D Lamé unchanged",
    }
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
            "pyvista_rendering_executed": False,
            "native_paraview_rendering_executed": True,
            "one_view_per_file": True,
            "contact_sheet_created": False,
        },
        "comparison": {
            "kind": "conversion-only fixed-activation forward replay",
            "fixed": (
                "historical full membrane, historical IsFixed, pinned step-40 "
                "e100-p000 activation, exact-zero displacement seed, E, nu, "
                "thickness, and no prestrain"
            ),
            "changed": "skin Lamé conversion only",
            "interpretation": (
                "demonstrates release of artificial in-plane area locking; does "
                "not claim improved target fit or bumpiness"
            ),
        },
        "material": material,
        "paraview_version": paraview_version,
        "pvbatch": pvbatch_identity,
        "command": command,
        "source": {
            "wrapper": wrapper_before,
            "renderer": renderer_before,
        },
        "pinned_inputs_pre": inputs_before,
        "pinned_inputs_post": inputs_after,
        "contract": contract_identity,
        "prepared_surfaces": surface_receipts,
        "analysis_metrics": {
            case_id: {
                key: analysis_rows[case_id][key]
                for key in (
                    "target/error_rms_fraction_of_target",
                    "target/face_target_area_weighted_error_rms_m",
                    "bumpiness/contraction_target_relative_dihedral_rms_deg",
                    "bumpiness/residual_normal_laplacian_rms_m",
                    "warning/inverted_tets",
                    "warning/isface_folded_triangles",
                )
            }
            for case_id in CASE_ORDER
        },
        "shared_area_strain_scale": {
            "display_limit_percent": STRAIN_LIMIT_PERCENT,
            "pooled_absolute_p99_percent_unrounded": pooled_abs_p99,
        },
        "outputs": outputs,
    }
    _write_json_atomic(cfg.output_receipt, receipt)
    for path in [CONTRACT, *INPUT_ROOT.iterdir(), *ASSET_ROOT.iterdir()]:
        cherries.log_output(path)
    logger.info("Wrote four separate native ParaView Lamé images and states")


if __name__ == "__main__":
    cherries.main(main)
