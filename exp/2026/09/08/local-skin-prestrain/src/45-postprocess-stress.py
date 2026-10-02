"""Compare actual equilibrated skin stress signs for the four skin runs."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003, TRY301

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

GROUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP / "src"))
from skin_stress import SkinStressDiagnostics  # noqa: E402

OUTPUT = GROUP / "data/45-skin-stress"
FIELD = GROUP / "data/10-prestrain-field/skin-prestrain.npz"
FIELD_VALIDATION = GROUP / "data/10-prestrain-field/validation.json"
SKIN_VALIDATION = GROUP / "data/11-skin-validation/summary.json"
STATE_SPECS = {
    "initial_fixed_q_forward": {
        "skin_zero": (GROUP / "data/20-forward-skin-zero", "final.npz"),
        "skin_local_1pct": (
            GROUP / "data/20-forward-skin-local-1pct",
            "final.npz",
        ),
    },
    "refit_step0_fixed_q_reequilibrium": {
        "skin_zero": (GROUP / "data/30-refit-skin-zero", "step-0000.npz"),
        "skin_local_1pct": (
            GROUP / "data/30-refit-skin-local-1pct",
            "step-0000.npz",
        ),
    },
    "best_over_200_update_budget": {
        "skin_zero": (GROUP / "data/30-refit-skin-zero", "final.npz"),
        "skin_local_1pct": (
            GROUP / "data/30-refit-skin-local-1pct",
            "final.npz",
        ),
    },
}


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def activation_from_q(q: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or q.shape[1] != 6:
        raise AssertionError(f"unexpected packed activation shape: {q.shape}")
    activation = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    activation[:, 0, 0] += q[:, 0]
    activation[:, 1, 1] += q[:, 1]
    activation[:, 2, 2] += q[:, 2]
    activation[:, 0, 1] = activation[:, 1, 0] = q[:, 3]
    activation[:, 1, 2] = activation[:, 2, 1] = q[:, 4]
    activation[:, 0, 2] = activation[:, 2, 0] = q[:, 5]
    return activation


def load_state(
    run_dir: Path,
    state_name: str,
    diagnostics: SkinStressDiagnostics,
    *,
    expected_case: str,
    expected_stage: str,
    expected_field_sha256: str,
    expected_field_validation_sha256: str,
    expected_validation_sha256: str,
    expected_prestrain_triangle_count: int,
    expected_max_contraction: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary_path = run_dir / "summary.json"
    provenance_path = run_dir / "provenance.json"
    trace_path = run_dir / "trace.csv"
    state_path = run_dir / state_name
    summary = json.loads(summary_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if summary["status"] not in {
        "completed_fixed_activation_forward",
        "completed_200_updates_not_stationarity_certified",
    }:
        raise AssertionError(f"incomplete run {run_dir}: {summary['status']}")
    with np.load(state_path, allow_pickle=False) as saved:
        state = {name: np.asarray(saved[name]) for name in saved.files}
    required = {
        "q",
        "Ainv",
        "u",
        "rest_points",
        "active_ids",
        "step",
        "solver_valid",
        "physical_volume_energy",
    }
    if not required.issubset(state):
        raise AssertionError(f"state schema changed in {state_path}: {sorted(state)}")
    if not np.array_equal(state["rest_points"], diagnostics.rest_volume):
        raise AssertionError(f"state rest geometry differs from fixture: {state_path}")
    if state["u"].shape != diagnostics.rest_volume.shape:
        raise AssertionError(f"state displacement shape changed: {state_path}")
    if (
        not bool(state["solver_valid"])
        or not bool(state["physical_volume_energy"])
        or not np.isfinite(state["u"]).all()
        or not np.isfinite(state["q"]).all()
    ):
        raise AssertionError(f"invalid saved equilibrium: {state_path}")
    if not np.array_equal(state["Ainv"], activation_from_q(state["q"])):
        raise AssertionError(
            f"saved Ainv does not exactly reconstruct from q: {state_path}"
        )
    if (
        state["active_ids"].ndim != 1
        or len(state["active_ids"]) != len(state["q"])
        or len(np.unique(state["active_ids"])) != len(state["active_ids"])
    ):
        raise AssertionError(f"invalid active control mapping: {state_path}")
    step = int(state["step"])
    if state_name == "final.npz" and step != int(summary["best_step"]):
        raise AssertionError(
            f"final.npz is not the summary best state: {state_path}, "
            f"{step} != {summary['best_step']}"
        )
    if state_name == "step-0000.npz" and step != 0:
        raise AssertionError(f"expected the refit step-0 state: {state_path}")
    provenance_checks = {
        "case": provenance["case"] == expected_case,
        "stage": provenance["stage"] == expected_stage,
        "field": provenance["field"]["sha256"] == expected_field_sha256,
        "field_validation": provenance["field_validation"]["sha256"]
        == expected_field_validation_sha256,
        "skin_validation": provenance["skin_validation"]["sha256"]
        == expected_validation_sha256,
        "skin_E_MPa": provenance["materials"]["skin_E_MPa"] == 0.2,
        "skin_nu": provenance["materials"]["skin_nu"] == 0.46,
        "skin_thickness_m": provenance["materials"]["skin_thickness_m"] == 0.001,
        "contact_disabled": provenance["materials"]["contact_enabled"] is False,
        "prestrain_triangle_count": provenance["materials"][
            "skin_prestrain_triangle_count"
        ]
        == expected_prestrain_triangle_count,
        "max_contraction": bool(
            np.isclose(
                provenance["materials"]["skin_max_natural_length_contraction"],
                expected_max_contraction,
                atol=2e-17,
                rtol=0,
            )
        ),
    }
    if not all(provenance_checks.values()):
        raise AssertionError(
            f"run provenance does not match stress case: {provenance_checks}"
        )
    with trace_path.open(newline="") as stream:
        trace_rows = list(csv.DictReader(stream))
    matches = [row for row in trace_rows if int(row["step"]) == step]
    if len(matches) != 1:
        raise AssertionError(f"trace does not uniquely contain state step {step}")
    trace = matches[0]
    metric_keys = (
        "objective_mm2",
        "fit_rms_mm",
        "motion_rms_mm",
        "area_weighted_fit_rms_mm",
        "area_weighted_motion_rms_mm",
        "target_projection",
        "roi_right_nose_to_mouth_fit_vector_rms_mm",
        "forward_steps",
        "forward_grad_norm",
        "detF_min",
        "inverted_all_cells",
        "inverted_active_cells",
        "A_eigen_min",
        "non_spd_active_cells",
    )
    state_metrics = {name: float(trace[name]) for name in metric_keys}
    protocol = {
        "canonical_checkpoint_sha256": provenance["canonical_checkpoint"]["sha256"],
        "field_sha256": provenance["field"]["sha256"],
        "skin_validation_sha256": provenance["skin_validation"]["sha256"],
        "source_checkpoint_step": provenance["source_checkpoint_step"],
        "objective": provenance["objective"],
        "optimizer": provenance["optimizer"],
        "forward_tolerances": provenance["forward_tolerances"],
        "local_physics_sha256": provenance["sources"]["local_physics"]["sha256"],
        "active_volume_law_sha256": provenance["sources"]["volume_preserving_active"][
            "sha256"
        ],
        "koiter_sha256": provenance["sources"]["liblaf.apple.warp.fem._koiter"][
            "sha256"
        ],
    }
    return (
        {
            "run_summary": file_record(summary_path),
            "run_provenance": file_record(provenance_path),
            "run_trace": file_record(trace_path),
            "state": file_record(state_path),
            "status": summary["status"],
            "state_step": step,
            "run_best_step": int(summary["best_step"]),
            "last_evaluated_step": int(summary["last_evaluated_step"]),
            "state_semantics": (
                "minimum-objective evaluated solver-valid state"
                if state_name == "final.npz"
                else "refit step-0 re-equilibrium before the first Adam update"
            ),
            "state_metrics": state_metrics,
            "provenance_checks": provenance_checks,
            "paired_protocol": protocol,
        },
        state,
    )


def region_delta(
    local: dict[str, Any],
    zero: dict[str, Any],
) -> dict[str, Any]:
    if local.get("empty") or zero.get("empty"):
        raise AssertionError("comparison region is empty")
    if not np.isclose(
        local["reference_area_support_m2"],
        zero["reference_area_support_m2"],
        atol=0,
        rtol=0,
    ):
        raise AssertionError("paired stress regions have different reference support")
    vector_keys = (
        "principal_elastic_stretch_mean",
        "principal_metric_stress_indicator_mean_mpa",
    )
    scalar_keys = (
        "area_weighted_fraction_biaxial_tension",
        "area_weighted_fraction_biaxial_compression",
        "area_weighted_fraction_mixed_sign",
        "area_weighted_fraction_neutral_or_unclassified",
        "area_weighted_fraction_any_tension",
        "area_weighted_fraction_any_compression",
        "koiter_energy_j",
    )
    return {
        "definition": "skin_local_1pct minus skin_zero",
        "reference_area_support_m2": local["reference_area_support_m2"],
        **{
            f"delta_{key}": (
                np.asarray(local[key], dtype=float) - np.asarray(zero[key], dtype=float)
            ).tolist()
            for key in vector_keys
        },
        **{f"delta_{key}": float(local[key] - zero[key]) for key in scalar_keys},
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        diagnostics = SkinStressDiagnostics()
        skin = pv.read(diagnostics.skin_path)
        fraction = np.asarray(skin.cell_data["Fraction"], dtype=np.float64)
        fraction_receipt = {
            "shape": list(fraction.shape),
            "sha256": array_sha256(fraction),
            "all_finite": bool(np.isfinite(fraction).all()),
            "all_exactly_one": bool(np.array_equal(fraction, np.ones_like(fraction))),
            "reason_checked": (
                "The frozen stress helper evaluates unit Fraction. Exact agreement "
                "with the Koiter input is required before using its energy values."
            ),
        }
        if (
            not fraction_receipt["all_finite"]
            or not fraction_receipt["all_exactly_one"]
        ):
            raise AssertionError(
                f"skin Fraction is not exactly one: {fraction_receipt}"
            )

        with np.load(FIELD, allow_pickle=False) as saved:
            field = {name: np.asarray(saved[name]) for name in saved.files}
        field_validation = json.loads(FIELD_VALIDATION.read_text())
        field_hash_checks = {
            name: array_sha256(field[name]) == expected
            for name, expected in field_validation["hashes"].items()
        }
        if field_validation[
            "status"
        ] != "passed_cpu_prestrain_field_validation" or not all(
            field_hash_checks.values()
        ):
            raise AssertionError(
                f"current field does not match validation: {field_hash_checks}"
            )
        if not np.array_equal(field["point_ids"], diagnostics.point_ids):
            raise AssertionError("field skin point IDs differ from stress fixture")
        if not np.array_equal(field["triangles"], diagnostics.triangles):
            raise AssertionError("field skin triangles differ from stress fixture")
        support = field["w"] > 0
        if not np.any(support):
            raise AssertionError("frozen local prestrain support is empty")
        activation_support = np.linalg.norm(field["activation_inv"], axis=1) > 0
        if not np.array_equal(support, activation_support):
            raise AssertionError("w support differs from ActivationInv support")
        diagnostics.region_area_weights["local_prestrain_support"] = (
            diagnostics.reference_area * support
        )

        cases: dict[str, Any] = {}
        fixed_q_reference: np.ndarray | None = None
        active_ids_reference: np.ndarray | None = None
        field_record = file_record(FIELD)
        field_validation_record = file_record(FIELD_VALIDATION)
        skin_validation_record = file_record(SKIN_VALIDATION)
        for comparison, specs in STATE_SPECS.items():
            cases[comparison] = {}
            stage_q: np.ndarray | None = None
            for case, (run_dir, state_name) in specs.items():
                expected_stage = (
                    "forward" if comparison == "initial_fixed_q_forward" else "refit"
                )
                expected_prestrain_count = (
                    int(support.sum()) if case == "skin_local_1pct" else 0
                )
                expected_max_contraction = (
                    float(field["c"].max()) if case == "skin_local_1pct" else 0.0
                )
                receipt, state = load_state(
                    run_dir,
                    state_name,
                    diagnostics,
                    expected_case=case.replace("_", "-"),
                    expected_stage=expected_stage,
                    expected_field_sha256=field_record["sha256"],
                    expected_field_validation_sha256=field_validation_record["sha256"],
                    expected_validation_sha256=skin_validation_record["sha256"],
                    expected_prestrain_triangle_count=expected_prestrain_count,
                    expected_max_contraction=expected_max_contraction,
                )
                if active_ids_reference is None:
                    active_ids_reference = state["active_ids"].copy()
                elif not np.array_equal(active_ids_reference, state["active_ids"]):
                    raise AssertionError("active control mapping differs across runs")
                actual_activation_inv = (
                    np.zeros_like(field["activation_inv"])
                    if case == "skin_zero"
                    else field["activation_inv"]
                )
                positive_definite = np.linalg.eigvalsh(
                    np.stack(
                        (
                            np.stack(
                                (
                                    1 + actual_activation_inv[:, 0],
                                    actual_activation_inv[:, 2],
                                ),
                                axis=1,
                            ),
                            np.stack(
                                (
                                    actual_activation_inv[:, 2],
                                    1 + actual_activation_inv[:, 1],
                                ),
                                axis=1,
                            ),
                        ),
                        axis=1,
                    )
                )
                if np.any(positive_definite <= 0):
                    raise AssertionError(f"non-SPD skin ActivationInv in {case}")
                receipt["skin_activation_inv_spd_min_eigenvalue"] = float(
                    positive_definite.min()
                )
                stress = diagnostics.evaluate(
                    state["u"],
                    actual_activation_inv,
                    label=f"{comparison}_{case}",
                )
                receipt["arrays"] = {
                    "u_sha256": array_sha256(state["u"]),
                    "q_sha256": array_sha256(state["q"]),
                    "active_ids_sha256": array_sha256(state["active_ids"]),
                    "volume_activation_inv_sha256": array_sha256(state["Ainv"]),
                    "skin_activation_inv_sha256": array_sha256(actual_activation_inv),
                }
                cases[comparison][case] = {
                    "input": receipt,
                    "stress": stress,
                }
                if comparison in {
                    "initial_fixed_q_forward",
                    "refit_step0_fixed_q_reequilibrium",
                }:
                    if stage_q is None:
                        stage_q = state["q"].copy()
                    elif not np.array_equal(stage_q, state["q"]):
                        raise AssertionError(
                            "paired fixed-control skin cases do not share exact q"
                        )

            if comparison in {
                "initial_fixed_q_forward",
                "refit_step0_fixed_q_reequilibrium",
            }:
                if stage_q is None:
                    raise AssertionError("fixed-control comparison has no q")
                if fixed_q_reference is None:
                    fixed_q_reference = stage_q
                elif not np.array_equal(fixed_q_reference, stage_q):
                    raise AssertionError(
                        "initial and re-equilibrated fixed-control states do not share q"
                    )

        comparisons = {}
        fairness = {}
        for comparison, pair in cases.items():
            zero_regions = pair["skin_zero"]["stress"]["regions"]
            local_regions = pair["skin_local_1pct"]["stress"]["regions"]
            if set(zero_regions) != set(local_regions):
                raise AssertionError(f"region schema differs in {comparison}")
            zero_protocol = pair["skin_zero"]["input"]["paired_protocol"]
            local_protocol = pair["skin_local_1pct"]["input"]["paired_protocol"]
            if zero_protocol != local_protocol:
                raise AssertionError(f"paired protocol differs in {comparison}")
            exact_q = (
                pair["skin_zero"]["input"]["arrays"]["q_sha256"]
                == pair["skin_local_1pct"]["input"]["arrays"]["q_sha256"]
            )
            fairness[comparison] = {
                "paired_protocol_exact": True,
                "active_control_mapping_exact": True,
                "q_exact": exact_q,
                "interpretation": (
                    "same activation; isolates the skin natural metric at separately "
                    "accepted equilibria"
                    if exact_q
                    else "independent activations after equal 200-update budgets"
                ),
            }
            comparisons[comparison] = {
                name: region_delta(local_regions[name], zero_regions[name])
                for name in zero_regions
                if name != "prestrain_support"
            }

        summary = {
            "status": "completed_cpu_actual_equilibrated_skin_stress_comparison",
            "scope": (
                "Postprocesses actual saved equilibria of the two skin cases at "
                "the initial fixed-control forward, the refit step-0 warm-start "
                "re-equilibrium with the same activation, and each 200-update "
                "run's best evaluated state. It performs no solve, adjoint, or "
                "optimization."
            ),
            "interpretation_limits": {
                "stress_measure": (
                    "Principal eigenvalues of the Koiter metric constitutive "
                    "indicator lambda*trace(M)+2*mu*M in MPa. Signs are exact for "
                    "this constitutive measure; values are not Cauchy stress."
                ),
                "local_prestrain_support": (
                    "The identical frozen field w>0 triangle mask is used for both "
                    "skin-zero and skin-local-1pct, enabling paired spatial comparison."
                ),
                "prestrain_does_not_predetermine_equilibrated_sign": (
                    "The exact 1% natural-metric contraction is a material input. "
                    "Equilibrium deformation and surrounding tissue determine the "
                    "reported tensile, compressive, or mixed stress fractions."
                ),
                "no_skin_case": (
                    "Omitted because no Koiter membrane is assembled, so it has no "
                    "actual membrane stress."
                ),
                "best_refit": (
                    "A fixed 200-update budget comparison, not a stationarity claim."
                ),
                "fixed_control_reequilibrium": (
                    "Refit step 0 uses exactly the same activation as the initial "
                    "forward. Its different seed can yield a slightly different "
                    "accepted equilibrium within the declared solver tolerance; "
                    "both actual states are retained."
                ),
            },
            "inputs": {
                "field": field_record,
                "field_validation": field_validation_record,
                "skin_validation": skin_validation_record,
                "field_array_hash_checks": field_hash_checks,
                "fixture_skin": file_record(diagnostics.skin_path),
                "skin_fraction_receipt": fraction_receipt,
            },
            "support": {
                "definition": "saved frozen prestrain field w > 0",
                "triangle_count": int(support.sum()),
                "reference_area_m2": float(diagnostics.reference_area[support].sum()),
                "mask_sha256": array_sha256(support),
                "equals_activation_inv_nonzero_mask": True,
            },
            "diagnostics": diagnostics.provenance,
            "cases": cases,
            "fairness": fairness,
            "paired_deltas": comparisons,
            "sources": {
                "postprocessor": file_record(Path(__file__)),
                "frozen_stress_helper": file_record(GROUP / "src/skin_stress.py"),
            },
        }
        source_dir = OUTPUT / "sources"
        source_dir.mkdir()
        for path in (Path(__file__), GROUP / "src/skin_stress.py"):
            shutil.copy2(path, source_dir / path.name)
        tmp = OUTPUT / "summary.tmp"
        tmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        tmp.replace(OUTPUT / "summary.json")
    except BaseException:
        shutil.rmtree(OUTPUT)
        raise


if __name__ == "__main__":
    main()
