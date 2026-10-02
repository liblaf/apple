# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Compare only the canonical and accepted step-204 continuation endpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from activation_regularization import ActivationRegularization
from continuation_metrics import ContinuationMetrics

GROUP = Path(__file__).resolve().parents[1]
CANONICAL = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/final.npz"
)
BASELINE = GROUP / "data/20-fit300"
REGULARIZED = GROUP / "data/22-reg300"
OUTPUT = GROUP / "data/40-comparison"
LAMBDA = 0.07049801689660026


def record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def vector_rms_mm(values: np.ndarray) -> float:
    return float(1000.0 * np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def read_state(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if not bool(saved["solver_valid"]):
            raise ValueError(f"accepted comparison state is marked invalid: {path}")
        result = {
            name: np.asarray(saved[name]).copy() for name in ("q", "u", "rest_points")
        }
        result["step"] = int(saved["step"])
    if result["q"].shape != (288_235, 6) or result["u"].shape != (228_660, 3):
        raise ValueError(f"unexpected state shapes: {path}")
    if not all(
        np.isfinite(value).all()
        for value in result.values()
        if isinstance(value, np.ndarray)
    ):
        raise FloatingPointError(f"non-finite state array: {path}")
    return result


def relative(delta: float, baseline: float) -> float | None:
    return None if baseline == 0 else delta / baseline


def metric_row(
    name: str, canonical: float, baseline: float, regularized: float
) -> dict[str, float | str | None]:
    base_delta = baseline - canonical
    reg_delta = regularized - canonical
    paired_delta = regularized - baseline
    return {
        "metric": name,
        "canonical200": canonical,
        "baseline204": baseline,
        "regularized204": regularized,
        "baseline204_minus_canonical200": base_delta,
        "regularized204_minus_canonical200": reg_delta,
        "regularized204_minus_baseline204": paired_delta,
        "regularized_vs_baseline_relative": relative(paired_delta, baseline),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    baseline_paths = {
        name: BASELINE / name
        for name in (
            "last.npz",
            "summary.json",
            "origin.json",
            "provenance.json",
            "trace.csv",
            "rejected-geometry.json",
        )
    }
    regularized_paths = {
        name: REGULARIZED / name
        for name in (
            "last.npz",
            "optimizer-latest.pt",
            "summary.json",
            "origin.json",
            "provenance.json",
            "trace.csv",
            "rejected-geometry.json",
        )
    }
    required = [
        CANONICAL,
        *baseline_paths.values(),
        *regularized_paths.values(),
        Path(__file__),
        GROUP / "src/continuation_metrics.py",
        GROUP / "src/activation_regularization.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical = read_state(CANONICAL)
    baseline = read_state(baseline_paths["last.npz"])
    regularized = read_state(regularized_paths["last.npz"])
    if canonical["step"] != 200 or baseline["step"] != regularized["step"] != 204:
        raise ValueError(
            "comparison requires canonical step 200 and accepted step 204 endpoints"
        )
    if not (
        np.array_equal(canonical["rest_points"], baseline["rest_points"])
        and np.array_equal(canonical["rest_points"], regularized["rest_points"])
    ):
        raise ValueError("states do not share exact rest geometry")

    baseline_summary = json.loads(baseline_paths["summary.json"].read_text())
    regularized_summary = json.loads(regularized_paths["summary.json"].read_text())
    baseline_provenance = json.loads(baseline_paths["provenance.json"].read_text())
    regularized_provenance = json.loads(
        regularized_paths["provenance.json"].read_text()
    )
    baseline_rejection = json.loads(
        baseline_paths["rejected-geometry.json"].read_text()
    )
    regularized_rejection = json.loads(
        regularized_paths["rejected-geometry.json"].read_text()
    )
    for summary in (baseline_summary, regularized_summary):
        if summary["last_accepted_step"] != 204 or summary["best_step"] != 204:
            raise ValueError("branch summary does not select accepted step 204")
        if summary["failure"]["step"] != 205:
            raise ValueError("branch summary did not stop on expected rejected step")
    if baseline_rejection["step"] != regularized_rejection["step"] != 205:
        raise ValueError("branches did not reject the same next step")
    if (
        baseline_rejection["new_ids"] != regularized_rejection["new_ids"]
        or baseline_rejection["inverted_ids"] != regularized_rejection["inverted_ids"]
    ):
        raise ValueError("branches did not have the same rejected geometry identity")
    if (
        baseline_summary["materials"] != regularized_summary["materials"]
        or baseline_summary["forward_tolerances"]
        != regularized_summary["forward_tolerances"]
    ):
        raise ValueError(
            "branches do not have identical mechanics or solver tolerances"
        )
    for key in ("resume", "canonical_provenance"):
        if baseline_provenance[key]["sha256"] != regularized_provenance[key]["sha256"]:
            raise ValueError(f"branches do not share pinned parent {key}")

    checkpoint = torch.load(
        regularized_paths["optimizer-latest.pt"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint["step"] != 204
        or checkpoint["objective_law"]
        != "uniform-face-coordinate-mse-mm2-plus-weak-same-muscle-neighbor-ainv-variation"
    ):
        raise ValueError(
            "regularized optimizer checkpoint lacks the step-204 combined objective law"
        )
    if not math.isclose(
        float(checkpoint["regularization_weight"]), LAMBDA, rel_tol=0, abs_tol=1e-16
    ):
        raise ValueError(
            "regularized optimizer checkpoint lambda differs from calibrated lambda"
        )
    if not (
        torch.isfinite(checkpoint["gradient"]).all()
        and np.array_equal(checkpoint["q"].numpy(), regularized["q"])
        and np.array_equal(checkpoint["u"], regularized["u"])
    ):
        raise ValueError(
            "regularized optimizer checkpoint is not the accepted step-204 state"
        )
    if int(checkpoint["optimizer"]["state"][0]["step"]) != 204:
        raise ValueError("regularized optimizer state does not retain step 204")

    diagnostics = ContinuationMetrics()
    regularizer = ActivationRegularization()
    values = {}
    for name, state in (
        ("canonical200", canonical),
        ("baseline204", baseline),
        ("regularized204", regularized),
    ):
        values[name] = {
            **diagnostics.evaluate(state["u"], state["q"]),
            **regularizer.metrics(state["q"]),
        }
    selected = (
        "fit_vector_rms_mm",
        "motion_vector_rms_mm",
        "target_projection",
        "roi_right_nose_to_mouth_fit_vector_rms_mm",
        "roi_right_mouth_corner_fit_vector_rms_mm",
        "roi_right_lateral_cheek_fit_vector_rms_mm",
        "roi_right_lower_cheek_jaw_fit_vector_rms_mm",
        "roughness_right_mouth_corner_normal_displacement_highpass_5mm_rms_mm",
        "roughness_right_mouth_corner_normal_residual_highpass_5mm_rms_mm",
        "roughness_right_lateral_cheek_normal_displacement_highpass_5mm_rms_mm",
        "roughness_right_lateral_cheek_normal_residual_highpass_5mm_rms_mm",
        "roughness_right_lower_cheek_jaw_normal_displacement_highpass_5mm_rms_mm",
        "roughness_right_lower_cheek_jaw_normal_residual_highpass_5mm_rms_mm",
        "regularizer_mean_neighbor_ainv_frobenius_squared",
        "regularizer_neighbor_ainv_frobenius_rms",
    )
    rows = [
        metric_row(
            key,
            *(
                values[name][key]
                for name in ("canonical200", "baseline204", "regularized204")
            ),
        )
        for key in selected
    ]
    geometry = {
        "baseline204_minus_canonical200_all_vector_rms_mm": vector_rms_mm(
            baseline["u"] - canonical["u"]
        ),
        "regularized204_minus_canonical200_all_vector_rms_mm": vector_rms_mm(
            regularized["u"] - canonical["u"]
        ),
        "regularized204_minus_baseline204_all_vector_rms_mm": vector_rms_mm(
            regularized["u"] - baseline["u"]
        ),
        "regularized204_minus_baseline204_face_vector_rms_mm": vector_rms_mm(
            (regularized["u"] - baseline["u"])[diagnostics.top]
        ),
        "regularized204_minus_baseline204_q_rms": float(
            np.sqrt(np.mean((regularized["q"] - baseline["q"]) ** 2))
        ),
        "regularized204_minus_baseline204_ainv_frobenius_rms": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (regularized["q"] - baseline["q"]) ** 2
                        * np.asarray((1, 1, 1, 2, 2, 2)),
                        axis=1,
                    )
                )
            )
        ),
    }
    for key, value in geometry.items():
        if "baseline204_minus_canonical200" in key:
            canonical_value, baseline_value, regularized_value = 0.0, value, 0.0
        else:
            canonical_value, baseline_value, regularized_value = 0.0, 0.0, value
        rows.append(
            metric_row(
                key,
                canonical_value,
                baseline_value,
                regularized_value,
            )
        )
    # The sole paired comparison of the two accepted endpoints is direct, not matched by a label.
    paired_fit = (
        values["regularized204"]["fit_vector_rms_mm"]
        - values["baseline204"]["fit_vector_rms_mm"]
    )
    OUTPUT.mkdir(parents=True)
    fields = list(rows[0])
    with (OUTPUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed_cpu_accepted_state_comparison",
        "scope": "Read-only CPU postprocessing of canonical step 200 and accepted step 204 states; no forward solve, adjoint, optimizer update, render, or rejected-step comparison.",
        "inputs": {
            str(path): record(path) for path in required if path != Path(__file__)
        },
        "source": record(Path(__file__)),
        "accepted_state_contract": {
            "canonical_step": canonical["step"],
            "baseline_last_accepted_step": baseline_summary["last_accepted_step"],
            "regularized_last_accepted_step": regularized_summary["last_accepted_step"],
            "both_rejected_next_step": 205,
            "same_rejected_inverted_ids": baseline_rejection["inverted_ids"],
            "same_rejected_new_ids": baseline_rejection["new_ids"],
            "identical_materials": True,
            "identical_forward_tolerances": True,
            "same_pinned_resume_checkpoint": True,
            "same_pinned_canonical_provenance": True,
            "exact_shared_rest_geometry": True,
        },
        "regularized_checkpoint_contract": {
            "checkpoint_step": int(checkpoint["step"]),
            "q_equals_last_npz": True,
            "u_equals_last_npz": True,
            "gradient_finite": True,
            "optimizer_step": int(checkpoint["optimizer"]["state"][0]["step"]),
            "objective_law": checkpoint["objective_law"],
            "lambda": float(checkpoint["regularization_weight"]),
        },
        "metrics": values,
        "paired_accepted_endpoint": {
            "regularized204_minus_baseline204_fit_rms_mm": paired_fit,
            "relative_to_baseline_fit": relative(
                paired_fit, values["baseline204"]["fit_vector_rms_mm"]
            ),
            **geometry,
        },
        "findings": [
            f"The direct paired accepted-endpoint fit difference is {paired_fit:.12g} mm ({relative(paired_fit, values['baseline204']['fit_vector_rms_mm']):.12g} of baseline step-204 fit).",
            "Both branches accepted only steps 201-204, then rejected step 205 with the same geometry identity; the pre-existing inverted cell remains, so neither endpoint establishes a physically valid solution.",
            "Four updates and a lambda calibrated to one percent of the initial data-gradient RMS do not provide useful evidence of regularization efficacy, reachability, or surface-quality improvement.",
        ],
        "outputs": {"comparison.csv": record(OUTPUT / "comparison.csv")},
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
