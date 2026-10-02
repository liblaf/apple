"""Known numerical failures are rejected; unrelated failures remain visible."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def policy():
    source = Path(__file__).parents[1] / "src/3130-inverse-smile-retries.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("known_")
    ]
    scope = {"json": json, "math": math, "Path": Path}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope
    )
    return scope


def test_projection_retry_requires_exact_saved_failure(policy, tmp_path):
    message = "Bounded dual Newton line search unresolved"
    error = AssertionError(message)
    assert not policy["known_projection_failure"](error, tmp_path)
    summary = {
        "status": "joint_projection_failed",
        "failure": {
            "stage": "bounded_dual",
            "type": "AssertionError",
            "message": message,
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert policy["known_projection_failure"](error, tmp_path)
    assert not policy["known_projection_failure"](RuntimeError(message), tmp_path)
    assert not policy["known_projection_failure"](
        AssertionError("wrong determinant"), tmp_path
    )
    summary["failure"]["stage"] = "invalid_inputs"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert not policy["known_projection_failure"](error, tmp_path)


def test_adjoint_retry_is_bound_to_the_actual_failed_solver_receipt(policy):
    receipt = {
        "method": "hybrid_fem_ipc_free_csr_with_optional_relative_shift",
        "attempts": [{}],
        "solver_success": False,
        "operator_relative_error": 1e-12,
        "shifted_relative_residual": 2e-7,
        "native_shifted_relative_residual": 2e-7,
    }
    assert policy["known_adjoint_failure"](AssertionError(receipt), receipt, 1e-7)
    assert not policy["known_adjoint_failure"](
        AssertionError(dict(receipt)), receipt, 1e-7
    )
    assert not policy["known_adjoint_failure"](RuntimeError(receipt), receipt, 1e-7)
    assert not policy["known_adjoint_failure"](
        AssertionError("CUDA failure"), receipt, 1e-7
    )
    receipt["solver_success"] = True
    assert policy["known_adjoint_failure"](AssertionError(receipt), receipt, 1e-7)
    receipt["shifted_relative_residual"] = receipt[
        "native_shifted_relative_residual"
    ] = 1e-8
    assert not policy["known_adjoint_failure"](AssertionError(receipt), receipt, 1e-7)


@pytest.mark.parametrize("operator_error", [2e-10, float("nan")])
def test_adjoint_operator_proof_failure_is_terminal(policy, operator_error):
    receipt = {
        "method": "hybrid_fem_ipc_free_csr_with_optional_relative_shift",
        "attempts": [{}],
        "solver_success": False,
        "operator_relative_error": operator_error,
        "shifted_relative_residual": 2e-7,
        "native_shifted_relative_residual": 2e-7,
    }
    assert not policy["known_adjoint_failure"](AssertionError(receipt), receipt, 1e-7)
