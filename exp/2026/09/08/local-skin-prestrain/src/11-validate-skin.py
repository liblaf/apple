"""Validate the exact Koiter convention and local skin assembly on CPU."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
OUTPUT = GROUP / "data/11-skin-validation"
CORE = ROOT / "src/liblaf/apple/warp/fem/_koiter.py"
LOCAL_PHYSICS = GROUP / "src/local_physics.py"
FIELD = GROUP / "data/10-prestrain-field/skin-prestrain.npz"
FIELD_VALIDATION = GROUP / "data/10-prestrain-field/validation.json"
EXPECTED_CORE_SHA256 = (
    "f7b7c9547c82976a130a88faf8df5172312309238c2b0cf8c8e762e1ec463e8c"
)
ARCHIVED_CORE = (
    ROOT
    / "exp/2026/09/08/physical-volume-continuation/data/20-fit300"
    / "sources/runtime/liblaf/apple/warp/fem/_koiter.py",
    ROOT
    / "exp/2026/09/08/physical-volume-continuation/data/22-reg300"
    / "sources/runtime/liblaf/apple/warp/fem/_koiter.py",
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline"
    / "sources/runtime/liblaf/apple/warp/fem/_koiter.py",
)


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def metric(vertices: np.ndarray) -> np.ndarray:
    a = vertices[1] - vertices[0]
    b = vertices[2] - vertices[0]
    return np.array([[a @ a, a @ b], [a @ b, b @ b]])


def triangle_state(
    rest: np.ndarray,
    current: np.ndarray,
    activation_inv: np.ndarray,
    *,
    young_mpa: float = 0.2,
    nu: float = 0.46,
    thickness_m: float = 0.001,
) -> dict[str, Any]:
    """Evaluate the scalar formulas used by the Koiter source without Warp."""
    g0 = metric(rest)
    g = metric(current)
    ainv = np.eye(2) + activation_inv
    effective_inverse_metric = ainv @ np.linalg.inv(g0) @ ainv.T
    elastic_squared = np.linalg.eigvals(effective_inverse_metric @ g)
    if np.max(np.abs(elastic_squared.imag)) > 1e-13:
        raise AssertionError("elastic metric eigenvalues are not real")
    elastic_squared = np.sort(elastic_squared.real)
    m = elastic_squared - 1
    lmbda = young_mpa * nu / (1 - nu**2)
    mu = young_mpa / (2 * (1 + nu))
    stress_indicator = lmbda * m.sum() + 2 * mu * m
    density_mpa = 0.5 * lmbda * m.sum() ** 2 + mu * np.sum(m**2)
    reference_metric_sqrt_det = float(np.sqrt(np.linalg.det(g0)))
    reference_area_m2 = reference_metric_sqrt_det / 2
    energy_j = thickness_m * reference_metric_sqrt_det / 8 * density_mpa * 1e6
    return {
        "principal_elastic_stretch": np.sqrt(elastic_squared).tolist(),
        "principal_metric_strain_eigenvalue": m.tolist(),
        "principal_metric_stress_indicator_mpa": stress_indicator.tolist(),
        "metric_energy_density_mpa": float(density_mpa),
        "reference_metric_sqrt_det_m2": reference_metric_sqrt_det,
        "reference_area_m2": reference_area_m2,
        "koiter_energy_j": float(energy_j),
    }


def source_audit() -> dict[str, Any]:
    source = LOCAL_PHYSICS.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def attribute_call(name: str) -> list[ast.Call]:
        return [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == name
        ]

    add_vertices = attribute_call("add_vertices")
    add_fixed = attribute_call("add_fixed")
    add_potential = attribute_call("add_potential")
    koiter_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Koiter"
        and node.func.attr == "from_pyvista"
    ]
    checks = {
        "one_mesh_vertex_assembly": len(add_vertices) == 1
        and ast.unparse(add_vertices[0].args[0]) == "self.mesh",
        "one_fixed_constraint_assembly": len(add_fixed) == 1
        and ast.unparse(add_fixed[0].args[0]) == "self.mesh",
        "one_volume_potential_call_inside_three_material_loop_plus_one_skin_call": len(
            add_potential
        )
        == 2,
        "one_koiter_skin_construction": len(koiter_calls) == 1,
        "skin_is_conditional": any(
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "skin_factor"
            and koiter_calls[0] in tuple(ast.walk(node))
            for node in ast.walk(tree)
        ),
        "skin_E_is_0p2_MPa": "young = 0.2 * skin_factor" in source,
        "skin_nu_is_0p46": "assert skin_nu == 0.46" in source,
        "skin_thickness_is_1_mm": 'name="skin", thickness=0.001' in source,
        "skin_activation_inv_is_injected": (
            "self.skin.cell_data[ACTIVATION_INV.vtk] = self.skin_activation_inv"
            in source
        ),
        "skin_factor_restricted_to_zero_or_one": (
            "assert skin_factor in (0.0, 1.0)" in source
        ),
        "contact_disabled": "contact_enabled=False" in source
        and "add_collision" not in source
        and "add_contact" not in source,
    }
    if not all(checks.values()):
        raise AssertionError(f"local physics source audit failed: {checks}")
    return {
        "checks": checks,
        "ast_counts": {
            "builder_add_vertices_call_sites": len(add_vertices),
            "builder_add_fixed_call_sites": len(add_fixed),
            "builder_add_potential_call_sites": len(add_potential),
            "Koiter_from_pyvista_call_sites": len(koiter_calls),
        },
        "interpretation": (
            "The first add_potential call site executes for the three declared "
            "volume materials; the second is the conditional skin membrane."
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        core_records = [
            file_record(CORE),
            *(file_record(path) for path in ARCHIVED_CORE),
        ]
        core_source = CORE.read_text()
        core_checks = {
            "live_matches_expected_sha256": core_records[0]["sha256"]
            == EXPECTED_CORE_SHA256,
            "live_and_all_archives_are_byte_identical": len(
                {record["sha256"] for record in core_records}
            )
            == 1,
            "natural_metric_formula_present": (
                "return A_inv @ materials.rest_metric_inv[cid] @ wp.transpose(A_inv)"
                in core_source
            ),
            "original_reference_area_weight_present": (
                "h * fraction * materials.rest_metric_sqrt_det[cid] / fraction.dtype(8.0)"
                in core_source
            ),
        }
        if not all(core_checks.values()):
            raise AssertionError(f"Koiter core audit failed: {core_checks}")

        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        contraction = 0.01
        inverse_stretch = 1 / (1 - contraction)
        packed = np.diag([inverse_stretch - 1, inverse_stretch - 1])
        at_rest = triangle_state(rest, rest, packed)
        at_natural = triangle_state(rest, rest * (1 - contraction), packed)
        zero_at_rest = triangle_state(rest, rest, np.zeros((2, 2)))
        exact_checks = {
            "encoded_inverse_stretch_equals_1_over_0p99": inverse_stretch == 1 / 0.99,
            "rest_state_has_positive_biaxial_stress": all(
                value > 0 for value in at_rest["principal_metric_stress_indicator_mpa"]
            ),
            "rest_state_has_positive_energy": at_rest["koiter_energy_j"] > 0,
            "natural_0p99_scale_has_unit_elastic_stretches": np.allclose(
                at_natural["principal_elastic_stretch"], 1, atol=2e-15, rtol=0
            ),
            "natural_0p99_scale_has_zero_stress": np.allclose(
                at_natural["principal_metric_stress_indicator_mpa"],
                0,
                atol=1e-15,
                rtol=0,
            ),
            "natural_0p99_scale_has_zero_energy": abs(at_natural["koiter_energy_j"])
            <= 1e-24,
            "zero_prestrain_rest_has_zero_stress_and_energy": np.allclose(
                zero_at_rest["principal_metric_stress_indicator_mpa"],
                0,
                atol=0,
                rtol=0,
            )
            and zero_at_rest["koiter_energy_j"] == 0,
            "energy_uses_original_reference_area": at_rest[
                "reference_metric_sqrt_det_m2"
            ]
            == 2 * at_rest["reference_area_m2"],
        }
        if not all(exact_checks.values()):
            raise AssertionError(f"exact 1% triangle check failed: {exact_checks}")

        field_validation = json.loads(FIELD_VALIDATION.read_text())
        with np.load(FIELD, allow_pickle=False) as saved:
            c = np.asarray(saved["c"])
            activation_inv = np.asarray(saved["activation_inv"])
        natural_identity_error = float(
            np.max(np.abs((1 + activation_inv[:, :2]) * (1 - c[:, None]) - 1))
        )
        field_checks = {
            "field_validation_passed": field_validation["status"]
            == "passed_cpu_prestrain_field_validation",
            "all_protected_incident_triangles_are_identity": field_validation["sign"][
                "protected_incident_triangles_are_exact_zero"
            ],
            "protected_incident_triangle_count_is_1315": field_validation["sign"][
                "protected_incident_triangle_count"
            ]
            == 1315,
            "saved_field_max_contraction_is_exactly_1pct": float(c.max()) == 0.01,
            "saved_field_natural_metric_contracts_each_axis_by_1_minus_c": (
                natural_identity_error <= 2e-16
            ),
        }
        if not all(field_checks.values()):
            raise AssertionError(f"saved field mechanics check failed: {field_checks}")

        summary = {
            "status": "passed",
            "scope": (
                "CPU-only validation of the frozen Koiter source, exact 1% "
                "single-triangle convention, saved field encoding, and local "
                "assembly source. No GPU, face solve, adjoint, or update."
            ),
            "koiter_core": {"checks": core_checks, "sources": core_records},
            "exact_1pct_triangle": {
                "checks": exact_checks,
                "contraction": contraction,
                "packed_activation_inv": [
                    inverse_stretch - 1,
                    inverse_stretch - 1,
                    0.0,
                ],
                "at_original_rest_geometry": at_rest,
                "at_natural_0p99_scaled_geometry": at_natural,
                "zero_prestrain_at_rest_geometry": zero_at_rest,
            },
            "saved_field": {
                "checks": field_checks,
                "natural_identity_max_abs_error": natural_identity_error,
                "field": file_record(FIELD),
                "field_validation": file_record(FIELD_VALIDATION),
            },
            "local_assembly": source_audit(),
            "sources": {
                "validator": file_record(Path(__file__)),
                "stress_diagnostics": file_record(GROUP / "src/skin_stress.py"),
                "local_physics": file_record(LOCAL_PHYSICS),
            },
        }
        source_dir = OUTPUT / "sources"
        source_dir.mkdir()
        for path in (Path(__file__), GROUP / "src/skin_stress.py", LOCAL_PHYSICS, CORE):
            shutil.copy2(path, source_dir / path.name)
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    except BaseException:
        shutil.rmtree(OUTPUT)
        raise


if __name__ == "__main__":
    main()
