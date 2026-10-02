"""The bounded-dual direction retry is constrained to its persisted receipt."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def policy():
    source = Path(__file__).parents[1] / "src/3180-inverse-smile-retries.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "known_projection_failure"
    )
    scope = {"json": json, "math": math, "Path": Path}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope


def write_direction_receipt(path: Path) -> AssertionError:
    message = "Bounded dual Newton direction unresolved"
    cache = {"path": "/archive/cache/coefficients.npz", "sha256": "cache-hash"}
    ids = [18514, 155249, 607296, 630081]
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "joint_projection_failed",
                "cache": cache,
                "selected_original_ids": ids,
                "failure": {
                    "stage": "bounded_dual",
                    "type": "AssertionError",
                    "message": message,
                },
            }
        )
    )
    (path / "input-receipt.json").write_text(
        json.dumps(
            {
                "cache": cache,
                "row_order": [*ids, "negative_objective_gradient"],
            }
        )
    )
    (path / "certificate.json").write_text(
        json.dumps(
            {
                "status": "unresolved",
                "solver": "row/RHS normalized L-BFGS-B clipped dual plus bounded Newton polishing",
                "infeasibility_claimed": False,
                "solver_success": True,
                "solver_status": 0,
                "solver_message": "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                "polish": [
                    {"iteration": 0, "normalized_projected_gradient_inf": 3e-12}
                ],
                "trust_region_lower_bound": {
                    "box_point_feasible": True,
                    "box_normal_multipliers_nonnegative": True,
                    "box_normal_complementarity_exact": True,
                    "outside_trust_certified": False,
                },
                "failure": {"type": "AssertionError", "message": message},
            }
        )
    )
    return AssertionError(message)


def test_direction_retry_requires_archived_certificate(policy, tmp_path):
    error = write_direction_receipt(tmp_path)
    assert policy["known_projection_failure"](error, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("solver_success", False),
        ("outside_trust_certified", True),
    ],
)
def test_direction_retry_rejects_nonmatching_certificate(
    policy, tmp_path, field, value
):
    error = write_direction_receipt(tmp_path)
    certificate_path = tmp_path / "certificate.json"
    certificate = json.loads(certificate_path.read_text())
    if field in certificate:
        certificate[field] = value
    else:
        certificate["trust_region_lower_bound"][field] = value
    certificate_path.write_text(json.dumps(certificate))
    assert not policy["known_projection_failure"](error, tmp_path)


def test_projection_failure_marks_no_physical_candidate_evaluation():
    source = (
        Path(__file__).parents[1] / "src/3180-inverse-smile-retries.py"
    ).read_text()
    branch = source[
        source.index(
            "except Exception as error:", source.index("joint_trial_dir")
        ) : source.index("if known_projection_failure", source.index("joint_trial_dir"))
    ]
    assert '"physical_candidate_evaluated": False' in branch
