# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Audit the preserved Raw6 step-8 activation field without solving physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_RESULT = EXPERIMENT / "data/23-raw6-diagnostic-step8"
DEFAULT_FIXTURE = EXPERIMENT / "data/10-fixture"
DEFAULT_GRADIENT_AUDIT = EXPERIMENT / "data/23-raw6-fat049/gradient-audit.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/47-raw6-activation-audit"
DET_INTERVAL = (0.9, 1.1)
CONTRACTION_LIMIT = 0.65
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("object arrays do not have a stable byte receipt")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError(
            "weighted quantiles require finite vectors and positive weights"
        )
    order = np.argsort(values)
    value = values[order]
    weight = weights[order]
    cumulative = np.cumsum(weight)
    cumulative = (cumulative - 0.5 * weight) / cumulative[-1]
    quantiles = np.interp(np.asarray(QUANTILES), cumulative, value)
    return {
        f"q{probability:g}": float(item)
        for probability, item in zip(QUANTILES, quantiles, strict=True)
    }


def distribution(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    total = float(weights.sum())
    return {
        "weighted_mean": float(np.sum(weights * values) / total),
        "weighted_rms": float(np.sqrt(np.sum(weights * values**2) / total)),
        "weighted_quantiles": weighted_quantiles(values, weights),
    }


def raw6_ainv(q: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or q.shape[1] != 6 or q.dtype != np.float64:
        raise ValueError("Raw6 q must be float64 with shape (active_cells, 6)")
    H = np.zeros((len(q), 3, 3), dtype=np.float64)
    H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    H[:, 0, 1] = H[:, 1, 0] = q[:, 3]
    H[:, 1, 2] = H[:, 2, 1] = q[:, 4]
    H[:, 0, 2] = H[:, 2, 0] = q[:, 5]
    return np.eye(3, dtype=np.float64)[None, :, :] + H


def gradient_receipt(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("Raw6 initial gradient audit must contain exactly two records")
    by_epsilon = {float(row["epsilon"]): row for row in payload}
    if set(by_epsilon) != {0.001, 0.0003}:
        raise ValueError("unexpected Raw6 gradient-audit epsilon set")
    for epsilon, row in by_epsilon.items():
        if row.get("method") != "central":
            raise ValueError("Raw6 initial gradient audit must use central differences")
        for solve_name in ("plus_forward", "other_forward"):
            solve = row.get(solve_name)
            if not isinstance(solve, dict) or solve.get("success") is not True:
                raise ValueError(f"gradient audit {epsilon} has a failed forward solve")
        for name in ("analytic", "finite_difference", "relative_error"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"gradient audit {epsilon} has nonfinite {name}")
    coarse = by_epsilon[0.001]
    fine = by_epsilon[0.0003]
    coarse_error = float(coarse["relative_error"])
    fine_error = float(fine["relative_error"])
    if coarse_error >= 0.02:
        raise ValueError("epsilon=0.001 does not meet the declared 2% pass threshold")
    if fine_error <= 0.02:
        raise ValueError("epsilon=0.0003 is no longer the precision-sensitive record")
    return {
        "evaluation_point": "Raw6 zero initialization before optimizer updates",
        "method": "central finite difference",
        "records": [
            {
                "epsilon": 0.001,
                "analytic": float(coarse["analytic"]),
                "finite_difference": float(coarse["finite_difference"]),
                "relative_error": coarse_error,
                "interpretation": "pass; relative error is below the declared 2% threshold",
            },
            {
                "epsilon": 0.0003,
                "analytic": float(fine["analytic"]),
                "finite_difference": float(fine["finite_difference"]),
                "relative_error": fine_error,
                "interpretation": (
                    "precision-sensitive smaller perturbation; report separately and do not "
                    "treat it as the passing receipt"
                ),
            },
        ],
        "overall": (
            "the epsilon=0.001 record passes; the epsilon=0.0003 record is "
            "precision-sensitive"
        ),
    }


def audit(
    result_dir: Path,
    fixture_dir: Path,
    gradient_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    final_path = result_dir / "final.npz"
    summary_path = result_dir / "summary.json"
    manifest_path = result_dir / "artifact-manifest.json"
    volume_path = fixture_dir / "volume.vtu"
    for path in (final_path, summary_path, manifest_path, volume_path, gradient_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_summary = load_json(summary_path)
    if source_summary.get("status") != "diagnostic_stop_not_converged":
        raise ValueError("source must be the preserved nonconverged diagnostic export")
    if source_summary.get("optimization_complete") is not False:
        raise ValueError("source must not claim completed optimization")
    checkpoint = source_summary.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("step") != 8:
        raise ValueError("source checkpoint must be exact Raw6 step 8")

    manifest = load_json(manifest_path)
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict):
        raise TypeError("diagnostic artifact manifest has no files object")
    if declared_files["final.npz"]["sha256"] != sha256(final_path):
        raise ValueError("diagnostic final.npz hash differs from its artifact manifest")
    if declared_files["summary.json"]["sha256"] != sha256(summary_path):
        raise ValueError("diagnostic summary hash differs from its artifact manifest")

    with np.load(final_path, allow_pickle=False) as saved:
        if set(saved.files) != {"q", "u", "Ainv", "step"}:
            raise ValueError("unexpected diagnostic final.npz schema")
        q = saved["q"].copy()
        u = saved["u"].copy()
        ainv = saved["Ainv"].copy()
        step = int(saved["step"])
    if step != 8:
        raise ValueError("loaded checkpoint is not Raw6 step 8")
    if q.shape != (120020, 6) or q.dtype != np.float64:
        raise ValueError("unexpected saved Raw6 q schema")
    if ainv.shape != (120020, 3, 3) or ainv.dtype != np.float64:
        raise ValueError("unexpected saved Raw6 Ainv schema")
    if u.shape != (228660, 3) or u.dtype != np.float64:
        raise ValueError("unexpected saved displacement schema")
    if not np.isfinite(q).all() or not np.isfinite(ainv).all():
        raise ValueError("saved Raw6 activation contains nonfinite values")
    if not np.array_equal(raw6_ainv(q), ainv):
        raise ValueError("saved Raw6 q does not exactly reproduce saved Ainv")

    mesh = pv.read(volume_path)
    active_ids = np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], bool))
    if len(active_ids) != len(q):
        raise ValueError("fixture active-cell ordering differs from saved Raw6 arrays")
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)[active_ids]
    fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)[
        active_ids
    ]
    weight = volume * fraction
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError(
            "active muscle-fraction-volume weights must be finite and positive"
        )
    total_weight = float(weight.sum())

    eigen_ainv = np.linalg.eigvalsh(ainv)
    if not np.isfinite(eigen_ainv).all() or np.any(eigen_ainv <= 0.0):
        raise ValueError("saved Ainv must be symmetric positive definite")
    det_ainv = np.prod(eigen_ainv, axis=1)
    direct_det = np.linalg.det(ainv)
    if not np.allclose(det_ainv, direct_det, rtol=2e-14, atol=2e-15):
        raise ValueError("Ainv determinant and eigenvalue product differ")
    natural_stretch = np.sort(1.0 / eigen_ainv, axis=1)

    recomputed = source_summary.get("independent_recomputation", {}).get("metrics", {})
    for name, observed in (
        ("activation_det_min", det_ainv.min()),
        ("activation_det_max", det_ainv.max()),
        ("activation_eigen_min", eigen_ainv.min()),
        ("activation_eigen_max", eigen_ainv.max()),
    ):
        expected = recomputed.get(name)
        if expected is None or not math.isclose(
            float(observed), float(expected), rel_tol=2e-14, abs_tol=2e-15
        ):
            raise ValueError(
                f"saved activation differs from source summary metric {name}"
            )

    det_outside = (det_ainv < DET_INTERVAL[0]) | (det_ainv > DET_INTERVAL[1])
    contraction = natural_stretch[:, 0] < CONTRACTION_LIMIT
    source_paths = {
        "diagnostic_final_npz": final_path,
        "diagnostic_summary": summary_path,
        "diagnostic_artifact_manifest": manifest_path,
        "fixture_volume": volume_path,
        "raw6_initial_gradient_audit": gradient_path,
        "audit_source": Path(__file__),
    }
    summary = {
        "schema_version": 1,
        "status": "diagnostic_stop_not_converged",
        "optimization_complete": False,
        "physics_resolve_performed": False,
        "fresh_rest_branch_check_performed": False,
        "checkpoint": {
            "method": "Raw6",
            "step": step,
            "q_shape": list(q.shape),
            "Ainv_shape": list(ainv.shape),
            "exact_q_to_Ainv": True,
        },
        "weighting": {
            "definition": "fixture cell_data/Volume * cell_data/MuscleFraction",
            "active_cells": len(active_ids),
            "total_active_tissue_volume_m3": total_weight,
        },
        "activation_detAinv": {
            **distribution(det_ainv, weight),
            "outside_interval": {
                "interval": list(DET_INTERVAL),
                "strict_definition": "detAinv < 0.9 or detAinv > 1.1",
                "cells": int(det_outside.sum()),
                "cell_fraction": float(det_outside.mean()),
                "weighted_fraction": float(weight[det_outside].sum() / total_weight),
            },
        },
        "natural_principal_active_stretch": {
            "definition": (
                "ascending ordered reciprocals of the eigenvalues of saved Ainv"
            ),
            "minimum": distribution(natural_stretch[:, 0], weight),
            "middle": distribution(natural_stretch[:, 1], weight),
            "maximum": distribution(natural_stretch[:, 2], weight),
            "contraction_beyond_35_percent": {
                "strict_definition": "minimum natural principal stretch < 0.65",
                "threshold": CONTRACTION_LIMIT,
                "cells": int(contraction.sum()),
                "cell_fraction": float(contraction.mean()),
                "weighted_fraction": float(weight[contraction].sum() / total_weight),
            },
        },
        "initial_gradient_audit": gradient_receipt(gradient_path),
        "hashes": {
            "files": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in source_paths.items()
            },
            "consumed_arrays": {
                "final.npz/q": array_sha256(q),
                "final.npz/Ainv": array_sha256(ainv),
                "fixture/active_ids": array_sha256(active_ids),
                "fixture/active_volume_times_muscle_fraction": array_sha256(weight),
            },
        },
    }

    output_dir.mkdir(parents=True)
    source_dir = output_dir / "sources"
    source_dir.mkdir()
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    outside = summary["activation_detAinv"]["outside_interval"]["weighted_fraction"]
    contracted = summary["natural_principal_active_stretch"][
        "contraction_beyond_35_percent"
    ]["weighted_fraction"]
    note = f"""# Raw6 step-8 activation audit

This is the preserved accepted step-8 Raw6 checkpoint, not a completed optimization.
No forward solve or fresh-rest branch check was performed in this CPU audit.

- Muscle-fraction-volume-weighted det(Ainv) outside [0.9, 1.1]: {outside:.6%}.
- Muscle-fraction-volume-weighted cells with minimum natural active stretch below 0.65: {contracted:.6%}.
- Natural active stretches are the ordered reciprocals of the saved Ainv eigenvalues.
- The initial Raw6 gradient audit passes at epsilon=0.001 (relative error 0.004192). The epsilon=0.0003 result (0.052993) is precision-sensitive and is reported separately rather than treated as a pass.
"""
    note_path = output_dir / "README.md"
    note_path.write_text(note, encoding="utf-8")
    summary["outputs"] = {
        "note": {"path": str(note_path), "sha256": sha256(note_path)},
        "copied_source": {
            "path": str(source_dir / Path(__file__).name),
            "sha256": sha256(source_dir / Path(__file__).name),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--gradient-audit", type=Path, default=DEFAULT_GRADIENT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = audit(
        args.result_dir.resolve(),
        args.fixture_dir.resolve(),
        args.gradient_audit.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "detAinv_outside_weighted_fraction": summary["activation_detAinv"][
                    "outside_interval"
                ]["weighted_fraction"],
                "contraction_beyond_35_percent_weighted_fraction": summary[
                    "natural_principal_active_stretch"
                ]["contraction_beyond_35_percent"]["weighted_fraction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
