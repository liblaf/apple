from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, TRY003
import hashlib
import importlib.util
import json
import logging
import math
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DESIGN = "meeting-authoritative-homogeneous-vs-fat-floor-c020-2x2-step40"
CASE_ORDER = ("H0P0", "H0P1", "HFP0", "HFP1")
EXPECTED_POINTS = 15_299
EXPECTED_TRIANGLES = 29_899
EXPECTED_EXPANDING_TRIANGLES = 16_723
IMAGE_RESOLUTION = (1_800, 1_600)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_DIR = GROUP_DIR / "data"
ASSET_DIR = DATA_DIR / "20-ablation-assets"
RENDERER = Path(__file__).with_name("20-ablation-render-paraview.py")
PVBATCH = Path("/usr/bin/pvbatch")

BASELINE_DATA = (
    REPO_ROOT / "exp/2026/08/18/human-face-smile-plane-stress-skin/data"
)
SELECTIVE_DATA = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse/data"
)
FAT_FLOOR_DATA = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-fat-floor-skin-energy-prestrain-inverse/data"
)
HEURISTIC_DATA = (
    REPO_ROOT
    / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data"
)
REGISTERED_ANALYZER = SELECTIVE_DATA.parent / "src/30-analyze-selective-skin-prestrain.py"
METRIC_SKIN = HEURISTIC_DATA / "10-material-candidates/skin-e100-p000.vtp"


@dataclass(frozen=True)
class IdentitySpec:
    path: Path
    size_bytes: int
    sha256: str


CASE_SPECS = {
    "H0P0": {
        "label": "homogeneous E, no prestrain",
        "material": IdentitySpec(
            BASELINE_DATA / "10-corrected-baseline/skin-isface-e0200-p000.vtp",
            1_138_550,
            "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
        ),
        "result": IdentitySpec(
            BASELINE_DATA
            / "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen.vtu",
            147_657_021,
            "c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4",
        ),
        "summary": IdentitySpec(
            BASELINE_DATA
            / "20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen-summary-final.json",
            126_540,
            "575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856",
        ),
        "surface": IdentitySpec(
            SELECTIVE_DATA / "30-paraview-inputs/terminal/h0p0.vtp",
            1_772_624,
            "e28fd72de800a93179db44fa070f4555035d8ea1d6626518ed69850dda5f6f5b",
        ),
    },
    "H0P1": {
        "label": "homogeneous E + c020 prestrain",
        "material": IdentitySpec(
            SELECTIVE_DATA / "10-prepared-material-cases/skin-h0p1-c020.vtp",
            1_898_983,
            "9b69f6cad6cfc7c6deadbd687d2947a44eedf7a295aae3540ac104ea50ebaacf",
        ),
        "result": IdentitySpec(
            SELECTIVE_DATA / "20-h0p1.vtu",
            147_640_393,
            "eabec1d0493f004d066f94f20b5ac6725f8d84245ceccfddc51dc191dd96cde0",
        ),
        "summary": IdentitySpec(
            SELECTIVE_DATA / "20-h0p1-summary-final.json",
            111_232,
            "0ecf17c2a25cc03ebccb42a7ca3bd25bbeaf2fde0b0da3f1dec156efe8d99b2c",
        ),
        "surface": IdentitySpec(
            SELECTIVE_DATA / "30-paraview-inputs/terminal/h0p1.vtp",
            1_772_156,
            "c1c736d92195b9ac6d5af32609c4443b1227f4dbfb3817040d2c6717c4524da8",
        ),
    },
    "HFP0": {
        "label": "fat-floor heterogeneous E, no prestrain",
        "material": IdentitySpec(
            FAT_FLOOR_DATA
            / "10-prepared-material-cases-v2/skin-hfp0-selective-efat-p000.vtp",
            1_611_312,
            "8aff20c6ecad328bb436213c7751c25546df3b12a01e3f6e748c24e4b9941f23",
        ),
        "result": IdentitySpec(
            FAT_FLOOR_DATA / "20-hfp0.vtu",
            147_659_455,
            "3175b90a4134d69bf159095bb7f1a74f9b67e4f3f23df81a789bae4e638d5fd3",
        ),
        "summary": IdentitySpec(
            FAT_FLOOR_DATA / "20-hfp0-summary-final.json",
            111_570,
            "b21223ade6fd0e4112ea389e351bd890cd4b60704cd0b50638ebde998c8dea84",
        ),
        "surface": IdentitySpec(
            FAT_FLOOR_DATA / "26-paraview-fat-floor-terminal/inputs/hfp0.vtp",
            2_634_952,
            "a99569a63761a4d76c1f94b01ec4a456700d0eaa6abca0100eb1e1d98ef12861",
        ),
    },
    "HFP1": {
        "label": "fat-floor heterogeneous E + c020 prestrain",
        "material": IdentitySpec(
            FAT_FLOOR_DATA
            / "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp",
            2_072_304,
            "89e0b349b1ba8002bc654325ba2f025c492b6e096c242c4c194ddede72cd117d",
        ),
        "result": IdentitySpec(
            FAT_FLOOR_DATA / "20-hfp1.vtu",
            147_652_097,
            "f93bf583819048b5d81a674c4f409450e3cd1200e0d3811b3dc98811480d53dd",
        ),
        "summary": IdentitySpec(
            FAT_FLOOR_DATA / "20-hfp1-summary-final.json",
            111_700,
            "73238e1a1cdb4f8f398b4d1430874abd22ef768d564225d4e3b07307bdb41540",
        ),
        "surface": IdentitySpec(
            FAT_FLOOR_DATA / "26-paraview-fat-floor-terminal/inputs/hfp1.vtp",
            2_634_336,
            "a595a874ca4ce42c2884a8f6c7705e857cfbe51531a44537ea0c49020972249f",
        ),
    },
}

STATIC_INPUTS = {
    "registered_analyzer": IdentitySpec(
        REGISTERED_ANALYZER,
        66_557,
        "d3225740992d57edfc852026416fe11c1bd4ab94c13c955debb4323c7c280548",
    ),
    "metric_skin": IdentitySpec(
        METRIC_SKIN,
        38_742_137,
        "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f",
    ),
    "pvbatch": IdentitySpec(
        PVBATCH,
        18_608,
        "be482a75b1e52a8b5d9df6c5687c743cc0b5312e30916622d54652a998eb8871",
    ),
}

MATERIAL_FACTORS = (
    {
        "factor_id": "young-h0",
        "source_case": "H0P0",
        "field": "SkinYoungModulusMPa",
        "range": [0.0, 0.2],
        "applies_to": ["H0P0", "H0P1"],
        "title": "Young's modulus factor H0",
        "subtitle": "H0P0 / H0P1 | E = 0.2 MPa everywhere",
        "scalar_title": "skin E (MPa, linear)",
    },
    {
        "factor_id": "young-hf",
        "source_case": "HFP0",
        "field": "SkinYoungModulusMPa",
        "range": [0.0, 0.2],
        "applies_to": ["HFP0", "HFP1"],
        "title": "Young's modulus factor HF",
        "subtitle": "HFP0 / HFP1 | E=.003 MPa where raw R>1; .2 otherwise",
        "scalar_title": "skin E (MPa, linear)",
    },
    {
        "factor_id": "prestrain-p0",
        "source_case": "H0P0",
        "field": "StressFreeAreaRatio",
        "range": [0.4802, 1.0],
        "applies_to": ["H0P0", "HFP0"],
        "title": "Prestrain factor P0",
        "subtitle": "H0P0 / HFP0 | rho = 1; ActivationInv = 0",
        "scalar_title": "stress-free area ratio rho",
    },
    {
        "factor_id": "prestrain-p1",
        "source_case": "H0P1",
        "field": "StressFreeAreaRatio",
        "range": [0.4802, 1.0],
        "applies_to": ["H0P1", "HFP1"],
        "title": "Prestrain factor P1 (c020)",
        "subtitle": "H0P1 / HFP1 | rho=.98^2 clip(raw R,.5,1)",
        "scalar_title": "stress-free area ratio rho",
    },
)


class Config(cherries.BaseConfig):
    output_contract: Path = cherries.output(
        "20-ablation-assets-contract.json", mkdir=True
    )
    output_metrics_json: Path = cherries.output(
        "20-ablation-step40-metrics.json", mkdir=True
    )
    output_metrics_table: Path = cherries.output(
        "20-ablation-step40-metrics.md", mkdir=True
    )
    output_render_inputs: Path = cherries.output(
        "20-ablation-render-inputs", mkdir=True
    )
    output_assets: Path = cherries.output("20-ablation-assets", mkdir=True)
    output_receipt: Path = cherries.output(
        "20-ablation-assets-receipt.json", mkdir=True
    )


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


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if _read_json(temporary) != value:
        raise RuntimeError(f"temporary JSON readback failed: {path}")
    temporary.replace(path)
    return _identity(path)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    if array.dtype.kind == "f":
        array = array.astype("<f8", copy=False)
    elif array.dtype.kind in {"i", "u"}:
        array = array.astype("<i8", copy=False)
    elif array.dtype.kind == "b":
        array = array.astype("u1", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_registered_analyzer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "meeting_registered_metric_implementation", REGISTERED_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load registered analyzer: {REGISTERED_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DISPLAY_NAMES.update(
        {
            "HFP0": CASE_SPECS["HFP0"]["label"],
            "HFP1": CASE_SPECS["HFP1"]["label"],
        }
    )
    return module


def _validate_materials() -> dict[str, Any]:
    expected_hashes = {
        "young_h0": "79258360b0aa0cc628c86628bde75fe2ea114d14b34fe5430cd8f4f28e359c24",
        "young_hf": "3d16df172d08edfdb52077ac8961aa80bc3363ed9e9d8b0b1f1a0f3695b82c1e",
        "rho_p0": "aa74d25a4afece1f232101f98e5fda8177935c3e833ae578be636d9ff42294c2",
        "rho_p1": "d631449a1db997e6ce2eac9d3276ff8a23451461350f917026f19e9d50cc89f1",
        "activation_p0": "051fe4599913dc590cb39aa79f7bb51578efc3323cd9c0a337be804d12d8f224",
        "activation_p1": "1366a17e86a2b182dd9b15512b9dc0664c869e416af7b5e591fbfb347fd53d55",
    }
    records: dict[str, Any] = {}
    meshes: dict[str, pv.PolyData] = {}
    for case_id in CASE_ORDER:
        mesh = pv.read(CASE_SPECS[case_id]["material"].path)
        if not isinstance(mesh, pv.PolyData):
            raise TypeError(f"{case_id} material is not PolyData")
        if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TRIANGLES:
            raise ValueError(f"{case_id} material dimensions changed")
        required = {"SkinYoungModulusMPa", "StressFreeAreaRatio", "ActivationInv"}
        missing = required - set(mesh.cell_data)
        if missing:
            raise KeyError(f"{case_id} material lacks fields: {sorted(missing)}")
        meshes[case_id] = mesh
    reference = meshes["H0P0"]
    for case_id, mesh in meshes.items():
        if not np.array_equal(mesh.points, reference.points) or not np.array_equal(
            mesh.faces, reference.faces
        ):
            raise ValueError(f"{case_id} material topology differs from H0P0")
        young = np.asarray(mesh.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
        rho = np.asarray(mesh.cell_data["StressFreeAreaRatio"], dtype=np.float64)
        activation = np.asarray(mesh.cell_data["ActivationInv"], dtype=np.float64)
        if not all(np.isfinite(array).all() for array in (young, rho, activation)):
            raise ValueError(f"{case_id} material contains non-finite values")
        h_key = "young_h0" if case_id.startswith("H0") else "young_hf"
        p_key = "rho_p0" if case_id.endswith("P0") else "rho_p1"
        a_key = "activation_p0" if case_id.endswith("P0") else "activation_p1"
        if _array_sha256(young) != expected_hashes[h_key]:
            raise ValueError(f"{case_id} Young's-modulus field changed")
        if _array_sha256(rho) != expected_hashes[p_key]:
            raise ValueError(f"{case_id} stress-free-area field changed")
        if _array_sha256(activation) != expected_hashes[a_key]:
            raise ValueError(f"{case_id} prestrain field changed")
        low_count = int(np.count_nonzero(young == 0.003))
        if case_id.startswith("HFP") and low_count != EXPECTED_EXPANDING_TRIANGLES:
            raise ValueError(f"{case_id} fat-floor support changed")
        if case_id.startswith("H0") and np.any(young != 0.2):
            raise ValueError(f"{case_id} is not homogeneous E=.2 MPa")
        records[case_id] = {
            "path": str(CASE_SPECS[case_id]["material"].path.resolve()),
            "points": mesh.n_points,
            "triangles": mesh.n_cells,
            "young_modulus_MPa": {
                "min": float(young.min()),
                "max": float(young.max()),
                "fat_floor_triangles": low_count,
                "array_sha256_le_f8": _array_sha256(young),
            },
            "stress_free_area_ratio": {
                "min": float(rho.min()),
                "max": float(rho.max()),
                "array_sha256_le_f8": _array_sha256(rho),
            },
            "activation_inv": {
                "min": float(activation.min()),
                "max": float(activation.max()),
                "array_sha256_le_f8": _array_sha256(activation),
            },
        }
    return records


def _fake_case(module: Any, case_id: str, summary: dict[str, Any]) -> Any:
    trace = [{} for _ in range(41)]
    trace[40] = {
        "forward/success": True,
        "adjoint/success": True,
        "target/error_rms": float(summary["final/error_rms"]),
        "activation_inv/rms": float(summary["activation_inv/rms"]),
    }
    spec = CASE_SPECS[case_id]
    return module.CaseInput(
        case_id=case_id,
        summary_path=spec["summary"].path,
        canonical_summary_path=spec["summary"].path,
        trace_path=spec["summary"].path,
        history_path=spec["result"].path,
        result_path=spec["result"].path,
        target_path=spec["summary"].path,
        skin_path=spec["material"].path,
        identities={},
        summary=summary,
        trace=trace,
        history=None,
    )


def _write_render_surface(
    output_dir: Path,
    case_id: str,
    source: pv.PolyData,
    point_error_mm: np.ndarray,
) -> dict[str, Any]:
    output = output_dir / f"20-ablation-step40-{case_id.lower()}.vtp"
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite render input: {output}")
    if "TargetPointErrorMM" in source.point_data:
        raise ValueError(f"{case_id} source already has TargetPointErrorMM")
    surface = source.copy(deep=True)
    surface.point_data["TargetPointErrorMM"] = point_error_mm
    surface.save(temporary)
    readback = pv.read(temporary)
    if not isinstance(readback, pv.PolyData):
        raise TypeError(f"{case_id} render-input readback is not PolyData")
    if (
        not np.array_equal(readback.points, source.points)
        or not np.array_equal(readback.faces, source.faces)
        or not np.array_equal(
            np.asarray(readback.point_data["TargetPointErrorMM"]), point_error_mm
        )
        or not np.array_equal(
            np.asarray(readback.point_data["TargetNormalResidualMM"]),
            np.asarray(source.point_data["TargetNormalResidualMM"]),
        )
    ):
        raise RuntimeError(f"{case_id} render-input readback changed")
    temporary.replace(output)
    return {
        **_identity(output),
        "point_error_array_sha256_le_f8": _array_sha256(point_error_mm),
    }


def _registered_metrics(
    render_input_dir: Path,
) -> tuple[dict[str, Any], float, float, dict[str, Any]]:
    module = _load_registered_analyzer()
    baseline = pv.read(CASE_SPECS["H0P0"]["result"].path)
    skin = pv.read(CASE_SPECS["H0P0"]["material"].path)
    metric_skin = pv.read(METRIC_SKIN)
    basis = module._build_basis(baseline, skin, metric_skin)  # noqa: SLF001
    if render_input_dir.exists():
        raise FileExistsError(f"refusing to overwrite render inputs: {render_input_dir}")
    render_input_dir.mkdir(parents=False)
    if not np.all(basis.loss_mask[basis.skin_mesh_ids]):
        raise ValueError("not every IsFace surface point belongs to LossMask")
    off_surface_loss_mask = np.asarray(basis.loss_mask, dtype=bool).copy()
    off_surface_loss_mask[basis.skin_mesh_ids] = False
    if int(np.count_nonzero(basis.loss_mask)) != 15_302:
        raise ValueError("objective LossMask point count changed")
    if int(np.count_nonzero(off_surface_loss_mask)) != 3:
        raise ValueError("expected exactly three off-surface objective points")
    output: dict[str, Any] = {}
    residuals: list[np.ndarray] = []
    point_errors: list[np.ndarray] = []
    render_inputs: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        summary = _read_json(CASE_SPECS[case_id]["summary"].path)
        required = {
            "status": "ok",
            "final/step": 40.0,
            "protocol/optimizer_steps": 40,
            "protocol/evaluations": 41,
            "inverse/forward_fail_count": 0,
            "inverse/adjoint_fail_count": 0,
            "cut_boundary/readback_exact_zero": True,
        }
        changed = {
            key: (summary.get(key), expected)
            for key, expected in required.items()
            if summary.get(key) != expected
        }
        if changed:
            raise ValueError(f"{case_id} step-40 protocol changed: {changed}")
        frame = pv.read(CASE_SPECS[case_id]["result"].path)
        row = module._frame_metrics(  # noqa: SLF001
            _fake_case(module, case_id, summary), basis, frame, 40
        )
        surface = pv.read(CASE_SPECS[case_id]["surface"].path)
        if surface.n_points != EXPECTED_POINTS or surface.n_cells != EXPECTED_TRIANGLES:
            raise ValueError(f"{case_id} render surface dimensions changed")
        residual_mm = 1.0e3 * np.asarray(row["field/residual_normal_m"])
        if not np.array_equal(
            np.asarray(surface.point_data["TargetNormalResidualMM"]), residual_mm
        ):
            raise ValueError(f"{case_id} render residual differs from registered metric")
        displacement = np.asarray(frame.point_data["Displacement"], dtype=np.float64)
        full_residual = displacement - basis.target
        point_error_mm = 1.0e3 * np.linalg.norm(
            full_residual[basis.skin_mesh_ids], axis=1
        )
        off_surface_point_error_mm = 1.0e3 * np.linalg.norm(
            full_residual[off_surface_loss_mask], axis=1
        )
        objective_rms_from_points_mm = math.sqrt(
            (
                float(np.dot(point_error_mm, point_error_mm))
                + float(
                    np.dot(off_surface_point_error_mm, off_surface_point_error_mm)
                )
            )
            / int(np.count_nonzero(basis.loss_mask))
        )
        if not math.isclose(
            objective_rms_from_points_mm,
            float(row["target/error_rms_mm"]),
            rel_tol=1e-13,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{case_id} point errors do not reconstruct target RMS")
        render_inputs[case_id] = _write_render_surface(
            render_input_dir, case_id, surface, point_error_mm
        )
        residuals.append(residual_mm)
        point_errors.append(point_error_mm)
        output[case_id] = {
            "case_id": case_id,
            "label": CASE_SPECS[case_id]["label"],
            "step": 40,
            "target_rms_mm": float(row["target/error_rms_mm"]),
            "contraction_dihedral_rms_deg": float(
                row["bumpiness/contraction_target_relative_dihedral_rms_deg"]
            ),
            "residual_normal_laplacian_rms_mm": 1.0e3
            * float(row["bumpiness/residual_normal_laplacian_rms_m"]),
            "area_ratio_rms_error": float(
                row["area/deformed_to_target_ratio_rms_error"]
            ),
            "activation_rms": float(row["activation/rms"]),
            "folded_skin_triangles": int(row["quality/skin_folded_triangles"]),
            "inverted_tets": int(row["quality/inverted_tets"]),
            "fixed_displacement_exact_zero": bool(
                row["fixed/displacement_exact_zero"]
            ),
            "surface_path": str(CASE_SPECS[case_id]["surface"].path.resolve()),
            "surface_residual_sha256_le_f8": _array_sha256(residual_mm),
            "render_surface_path": render_inputs[case_id]["path"],
            "target_point_error_sha256_le_f8": _array_sha256(point_error_mm),
            "surface_objective_point_count": int(point_error_mm.size),
            "off_surface_objective_point_count": int(
                off_surface_point_error_mm.size
            ),
            "objective_point_count": int(np.count_nonzero(basis.loss_mask)),
            "surface_point_error_rms_mm": float(
                np.linalg.norm(point_error_mm) / math.sqrt(point_error_mm.size)
            ),
            "off_surface_point_error_squared_sum_mm2": float(
                np.dot(off_surface_point_error_mm, off_surface_point_error_mm)
            ),
            "objective_rms_from_point_errors_mm": objective_rms_from_points_mm,
        }
    shared_limit = max(
        0.25, float(np.quantile(np.abs(np.concatenate(residuals)), 0.99))
    )
    if not math.isfinite(shared_limit):
        raise ValueError("shared residual limit is non-finite")
    point_error_limit = float(np.max(np.concatenate(point_errors)))
    if not math.isfinite(point_error_limit) or point_error_limit <= 0.0:
        raise ValueError("shared point-error limit is invalid")
    return output, shared_limit, point_error_limit, render_inputs


def _write_metrics_table(path: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    lines = [
        "# Step-40 corrected 2x2 ablation metrics",
        "",
        "All metrics below were recomputed with the registered `src/30` implementation; no legacy roughness values were mixed in.",
        "",
        "| case | target RMS (mm) | D (deg) | L (mm) | area RMS | activation RMS | folds | inverted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_id in CASE_ORDER:
        row = metrics[case_id]
        lines.append(
            f"| {case_id} | {row['target_rms_mm']:.6f} | "
            f"{row['contraction_dihedral_rms_deg']:.6f} | "
            f"{row['residual_normal_laplacian_rms_mm']:.6f} | "
            f"{row['area_ratio_rms_error']:.6f} | {row['activation_rms']:.7f} | "
            f"{row['folded_skin_triangles']} | {row['inverted_tets']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _identity(path)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _expected_outputs() -> list[Path]:
    names = [f"20-ablation-material-{row['factor_id']}" for row in MATERIAL_FACTORS]
    names.extend(f"20-ablation-step40-{case.lower()}-geometry" for case in CASE_ORDER)
    names.extend(
        f"20-ablation-step40-{case.lower()}-normal-residual" for case in CASE_ORDER
    )
    names.extend(
        f"20-ablation-step40-{case.lower()}-point-error" for case in CASE_ORDER
    )
    return [ASSET_DIR / f"{name}.{suffix}" for name in names for suffix in ("png", "pvsm")]


def _validate_outputs() -> list[dict[str, Any]]:
    expected = sorted(_expected_outputs())
    actual = sorted(path for path in ASSET_DIR.iterdir() if path.is_file())
    if actual != expected:
        raise ValueError(f"rendered asset inventory changed: {actual}")
    records = []
    for path in expected:
        if path.suffix == ".png":
            if _png_dimensions(path) != IMAGE_RESOLUTION or path.stat().st_size < 20_000:
                raise ValueError(f"invalid rendered PNG: {path}")
        else:
            text = path.read_text(encoding="utf-8", errors="strict")[:2_048]
            if "ServerManagerState" not in text:
                raise ValueError(f"invalid ParaView state: {path}")
        records.append(_identity(path))
    return records


def main(cfg: Config) -> None:
    output_paths = (
        cfg.output_contract,
        cfg.output_metrics_json,
        cfg.output_metrics_table,
        cfg.output_receipt,
    )
    stale = [
        str(path)
        for path in (*output_paths, cfg.output_render_inputs, cfg.output_assets)
        if path.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing to overwrite meeting assets: {stale}")

    input_identities: dict[str, Any] = {}
    for label, spec in STATIC_INPUTS.items():
        input_identities[label] = _require_identity(label, spec)
    for case_id in CASE_ORDER:
        input_identities[case_id] = {
            name: _require_identity(f"{case_id} {name}", spec)
            for name, spec in CASE_SPECS[case_id].items()
            if isinstance(spec, IdentitySpec)
        }

    material_records = _validate_materials()
    metrics, residual_limit, point_error_limit, render_inputs = _registered_metrics(
        cfg.output_render_inputs
    )
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case_order": list(CASE_ORDER),
        "step": 40,
        "implementation": input_identities["registered_analyzer"],
        "definitions": {
            "target_rms": "RMS of the full 3D pointwise vector mismatch Displacement-TargetDisplacement on 15,302 LossMask points",
            "target_point_error": "Euclidean magnitude of Displacement-TargetDisplacement at each point; the ParaView IsFace surface contains 15,299 of the 15,302 objective points",
            "D": "rest-edge-length-weighted target-relative dihedral RMS on 18,038 interior edges whose two incident IsFace triangles have strict raw R<1",
            "L": "RMS graph Laplacian of target-normal displacement residual on all IsFace vertices",
            "area": "RestArea-weighted RMS of deformed area / target area - 1",
            "activation": "RMS of six symmetric ActivationInv components on active muscle tetrahedra",
        },
        "metrics": metrics,
    }
    metrics_identity = _write_json(cfg.output_metrics_json, metrics_payload)
    table_identity = _write_metrics_table(cfg.output_metrics_table, metrics)

    camera = {
        "direction": [0.5, 0.0, math.sqrt(3.0) / 2.0],
        "focus": [1.4070500020750487, 2.211375948871603, 0.04575618370825054],
        "parallel_scale": 0.11093353464564434,
        "projection": "parallel",
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case_order": list(CASE_ORDER),
        "image_resolution": list(IMAGE_RESOLUTION),
        "camera": camera,
        "material_factors": list(MATERIAL_FACTORS),
        "cases": {
            case_id: {
                **metrics[case_id],
                "material_path": str(
                    CASE_SPECS[case_id]["material"].path.resolve()
                ),
                "surface_path": str(CASE_SPECS[case_id]["surface"].path.resolve()),
                "material_identity": input_identities[case_id]["material"],
                "surface_identity": input_identities[case_id]["surface"],
                "render_surface_identity": render_inputs[case_id],
            }
            for case_id in CASE_ORDER
        },
        "normal_residual_shared_limit_mm": residual_limit,
        "normal_residual_shared_limit_definition": (
            "max(0.25 mm, pooled absolute 99th percentile) across exact step-40 "
            "H0P0/H0P1/HFP0/HFP1 registered target-normal residuals"
        ),
        "point_error_shared_limit_mm": point_error_limit,
        "point_error_shared_limit_definition": (
            "pooled maximum of ||Displacement-TargetDisplacement|| across the "
            "15,299 IsFace surface points in exact step-40 H0P0/H0P1/HFP0/HFP1; "
            "linear nonnegative scale with no clipping"
        ),
        "point_error_objective_link": {
            "definition": "target RMS = sqrt(sum of squared point-error magnitudes over all LossMask points / 15,302)",
            "surface_objective_points": 15_299,
            "off_surface_objective_points": 3,
            "exact_reconstruction_validated_per_case": True,
        },
        "material_assignment": {
            "H0": "E=0.2 MPa on every IsFace triangle",
            "HF": "E=0.003 MPa iff raw TargetArea/RestArea>1; E=0.2 MPa otherwise",
            "P0": "rho=1; skin ActivationInv exact zero",
            "P1": "rho=0.98^2*clip(raw TargetArea/RestArea,0.5,1); ActivationInv=[rho^-1/2-1,rho^-1/2-1,0]",
            "domain": "29,899 all-vertex IsFace triangles; artificial cross-section excluded",
        },
        "shared_anatomy": {
            "fat": "Stable Neo-Hookean, E=.003 MPa, nu=.49",
            "muscle": "Active Stable Neo-Hookean, E=.03 MPa, nu=.49",
            "aponeurosis": "Stable Neo-Hookean, E=.1 MPa, nu=.35",
            "skin": "Koiter membrane, 1 mm, nu=.49, plane-stress Lame conversion, fixed original RestArea",
            "boundary": "all artificial cross-section incident vertices hard-fixed to exact zero displacement",
            "inverse": "fresh-zero activation/displacement/optimizer; Adam lr=.3; 40 updates and 41 evaluations",
        },
        "excluded_cohorts": [
            "selective E=0 ablation H1P0/H1P1",
            "historical full-boundary 3D-Lame skin results",
        ],
        "renderer": "native ParaView 6.1.1 only; separate files; no contact sheets",
    }
    contract_identity = _write_json(cfg.output_contract, contract)

    cfg.output_assets.mkdir(parents=False, exist_ok=False)
    command = [
        str(PVBATCH),
        str(RENDERER.resolve()),
        "--contract",
        str(cfg.output_contract.resolve()),
        "--output-dir",
        str(cfg.output_assets.resolve()),
    ]
    logger.info("Running native ParaView ablation renderer: %s", command)
    completed = subprocess.run(
        command,
        cwd=GROUP_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout)
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")
    outputs = _validate_outputs()

    for case_id in CASE_ORDER:
        cherries.set_step(40)
        cherries.log_metrics(
            {
                f"{case_id}/target_rms_mm": metrics[case_id]["target_rms_mm"],
                f"{case_id}/D_deg": metrics[case_id]["contraction_dihedral_rms_deg"],
                f"{case_id}/L_mm": metrics[case_id][
                    "residual_normal_laplacian_rms_mm"
                ],
                f"{case_id}/folds": metrics[case_id]["folded_skin_triangles"],
            }
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "status": "ok",
        "paraview_version": "6.1.1",
        "command": command,
        "wrapper": _identity(Path(__file__)),
        "renderer": _identity(RENDERER),
        "contract": contract_identity,
        "metrics": metrics_identity,
        "metrics_table": table_identity,
        "inputs": input_identities,
        "render_inputs": render_inputs,
        "material_validation": material_records,
        "normal_residual_shared_limit_mm": residual_limit,
        "point_error_shared_limit_mm": point_error_limit,
        "point_error_objective_linkage": {
            case_id: {
                key: metrics[case_id][key]
                for key in (
                    "target_rms_mm",
                    "surface_objective_point_count",
                    "off_surface_objective_point_count",
                    "objective_point_count",
                    "surface_point_error_rms_mm",
                    "off_surface_point_error_squared_sum_mm2",
                    "objective_rms_from_point_errors_mm",
                    "target_point_error_sha256_le_f8",
                )
            }
            for case_id in CASE_ORDER
        },
        "outputs": outputs,
        "output_count": len(outputs),
        "separate_png_count": sum(item["path"].endswith(".png") for item in outputs),
        "separate_pvsm_count": sum(
            item["path"].endswith(".pvsm") for item in outputs
        ),
        "visualization_only": True,
        "forward_or_inverse_executed": False,
    }
    _write_json(cfg.output_receipt, receipt)
    logger.info("Wrote %d separate ParaView assets", len(outputs))


if __name__ == "__main__":
    cherries.main(main, profile="debug")
