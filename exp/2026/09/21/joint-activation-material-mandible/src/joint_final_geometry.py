"""Final-run numerical geometry receipt independent of legacy source screening."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

SCHEMA = "joint-final-run-geometry-v1"
LEGACY_SCHEMA = "joint-legacy-source-box-qa-v1"


def _finite_six(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or np.any(~np.isfinite(result)):
        message = f"{name} must contain six finite values"
        raise ValueError(message)
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        message = f"{name} must be a built-in bool"
        raise TypeError(message)
    return value


def _fem_checks(receipt: dict[str, Any], *, neutral: bool) -> dict[str, bool]:
    checks = {
        "mandible_pose_consistency": _strict_bool(
            receipt["mandible_pose_consistency"]["passed"],
            name="mandible_pose_consistency.passed",
        ),
        "fem_lip": _strict_bool(
            receipt["fem_lip"]["numerical_geometry_ok"],
            name="fem_lip.numerical_geometry_ok",
        ),
        "fem_mandible_upper_oral": _strict_bool(
            receipt["fem_mandible_oral"]["upper_oral"]["numerical_geometry_ok"],
            name="fem_mandible_oral.upper_oral.numerical_geometry_ok",
        ),
        "fem_mandible_lower_oral": _strict_bool(
            receipt["fem_mandible_oral"]["lower_oral"]["numerical_geometry_ok"],
            name="fem_mandible_oral.lower_oral.numerical_geometry_ok",
        ),
        "fem_soft_cranium": _strict_bool(
            receipt["fem_contact_surfaces"]["soft_cranium"]["numerical_geometry_ok"],
            name="fem_contact_surfaces.soft_cranium.numerical_geometry_ok",
        ),
        "fem_soft_mandible": _strict_bool(
            receipt["fem_contact_surfaces"]["soft_mandible"]["numerical_geometry_ok"],
            name="fem_contact_surfaces.soft_mandible.numerical_geometry_ok",
        ),
        "fem_mandible_cranium_endpoint": _strict_bool(
            receipt["fem_contact_surfaces"]["mandible_cranium"][
                "numerical_geometry_ok"
            ],
            name="fem_contact_surfaces.mandible_cranium.numerical_geometry_ok",
        ),
    }
    if neutral:
        checks["neutral_pose_zero"] = _strict_bool(
            receipt["neutral_pose_ok"], name="neutral_pose_ok"
        )
        checks["frozen_neutral_invariants"] = _strict_bool(
            receipt["neutral_invariants_ok"], name="neutral_invariants_ok"
        )
    return checks


def adapt_final_run_geometry_receipt(
    receipt: dict[str, Any],
    *,
    neutral: bool,
    normalized_pose: Any,
    pose_rad_m: Any,
) -> dict[str, Any]:
    """Separate live FEM admission from the obsolete one-degree source box."""
    assert receipt["anatomical_validation"] is False
    assert receipt["mode"] == ("neutral" if neutral else "expression_candidate")
    normalized = _finite_six(normalized_pose, name="normalized_pose")
    physical = _finite_six(pose_rad_m, name="pose_rad_m")
    within_computational_box = bool(
        np.all(normalized >= -1.0) and np.all(normalized <= 1.0)
    )
    if neutral:
        assert np.array_equal(normalized, np.zeros(6))
        assert np.array_equal(physical, np.zeros(6))

    checks = _fem_checks(receipt, neutral=neutral)
    fem_admissible = all(checks.values())
    legacy = {
        "schema": LEGACY_SCHEMA,
        "role": (
            "QA only; the source-screened one-degree candidate box and blocked "
            "anatomy gate do not define final-run numerical admission"
        ),
        "pose_in_candidate_box": _strict_bool(
            receipt["pose_in_candidate_box"], name="pose_in_candidate_box"
        ),
        "numerical_geometry_admissible": _strict_bool(
            receipt["numerical_geometry_admissible"],
            name="numerical_geometry_admissible",
        ),
        "provisional_geometry_ok": _strict_bool(
            receipt["provisional_geometry_ok"], name="provisional_geometry_ok"
        ),
        "admissible": _strict_bool(receipt["admissible"], name="admissible"),
        "gate": receipt["gate"],
        "source_rigid_jaw": deepcopy(receipt["source_rigid_jaw"]),
        "source_geometry_role": receipt["source_geometry_role"],
        "limitations": deepcopy(receipt["limitations"]),
    }
    return {
        "schema": SCHEMA,
        "mode": "neutral" if neutral else "expression_candidate",
        "numerical_geometry_admissible": bool(
            within_computational_box and fem_admissible
        ),
        "fem_numerical_admissible": fem_admissible,
        "computational_pose_admissible": within_computational_box,
        "anatomical_validation": False,
        "pose": {
            "normalized": normalized.tolist(),
            "rad_m": physical.tolist(),
            "normalized_lower": [-1.0] * 6,
            "normalized_upper": [1.0] * 6,
            "bounds_role": "computational proposal bounds, not biological limits",
        },
        "fem_numerical_checks": checks,
        "mandible_pose_consistency": deepcopy(receipt["mandible_pose_consistency"]),
        "fem_lip": deepcopy(receipt["fem_lip"]),
        "fem_mandible_oral": deepcopy(receipt["fem_mandible_oral"]),
        "fem_contact_surfaces": deepcopy(receipt["fem_contact_surfaces"]),
        "legacy_source_box_qa": legacy,
        "rigid_bone_linear_ccd": {
            "included_in_this_receipt": False,
            "scope": (
                "the FEM mandible-cranium field above is an endpoint topology "
                "check; runner evidence records the separate straight-boundary "
                "rigid-bone CCD receipt"
            ),
        },
    }


def validate_final_run_geometry_receipt(
    receipt: dict[str, Any], *, neutral: bool
) -> None:
    """Fail visibly unless every declared final-run numerical check passes."""
    assert receipt["schema"] == SCHEMA, receipt
    assert receipt["mode"] == ("neutral" if neutral else "expression_candidate"), (
        receipt
    )
    assert receipt["anatomical_validation"] is False, receipt
    assert receipt["computational_pose_admissible"] is True, receipt
    assert receipt["fem_numerical_admissible"] is True, receipt
    assert all(receipt["fem_numerical_checks"].values()), receipt
    assert receipt["numerical_geometry_admissible"] is True, receipt
    assert receipt["legacy_source_box_qa"]["schema"] == LEGACY_SCHEMA, receipt
    assert receipt["rigid_bone_linear_ccd"]["included_in_this_receipt"] is False, (
        receipt
    )
