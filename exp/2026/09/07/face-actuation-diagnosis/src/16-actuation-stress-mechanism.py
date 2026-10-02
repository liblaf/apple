"""CPU-only check of active-strain prestress at explicitly fixed ``F = I``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "data/16-actuation-stress/summary.json"

# The executed FiberModes muscle constituent: 0.03 MPa * muscle_factor 0.8.
YOUNG_MPA = 0.024
NU = 0.46
GAMMA = 0.5
C50_A = math.log(2.0)
FD_STEP = 1.0e-7


def stable_parameters(young_mpa: float, nu: float) -> tuple[float, float]:
    """Return the Stable Neo-Hookean ``(mu, lambda_code)`` in MPa."""
    mu = young_mpa / (2.0 * (1.0 + nu))
    lambda_classical = young_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lambda_classical + mu


MU_MPA, LAMBDA_CODE_MPA = stable_parameters(YOUNG_MPA, NU)


def cofactor(matrix: np.ndarray) -> np.ndarray:
    """Return the cofactor matrix for the nonsingular matrices in this audit."""
    determinant = float(np.linalg.det(matrix))
    return determinant * np.linalg.inv(matrix).T


def energy_density_mpa(F: np.ndarray, ainv: np.ndarray) -> float:
    """Evaluate the production stable-active energy density, in MPa."""
    G = F @ ainv
    determinant = float(np.linalg.det(G))
    return float(
        0.5 * MU_MPA * (np.sum(G * G) - 3.0)
        - MU_MPA * (determinant - 1.0)
        + 0.5 * LAMBDA_CODE_MPA * (determinant - 1.0) ** 2
    )


def first_piola_mpa(F: np.ndarray, ainv: np.ndarray) -> np.ndarray:
    """Evaluate the production ``dPsi/dF`` formula, in MPa."""
    G = F @ ainv
    determinant = float(np.linalg.det(G))
    dpsi_dg = MU_MPA * G + (-MU_MPA + LAMBDA_CODE_MPA * (determinant - 1.0)) * cofactor(
        G
    )
    return dpsi_dg @ ainv.T


def finite_difference_gradient(ainv: np.ndarray) -> np.ndarray:
    """Central-difference ``dPsi/dF`` at the intentionally fixed ``F=I``."""
    result = np.empty((3, 3), dtype=np.float64)
    identity = np.eye(3)
    for row in range(3):
        for column in range(3):
            perturbation = np.zeros((3, 3), dtype=np.float64)
            perturbation[row, column] = FD_STEP
            result[row, column] = (
                energy_density_mpa(identity + perturbation, ainv)
                - energy_density_mpa(identity - perturbation, ainv)
            ) / (2.0 * FD_STEP)
    return result


def case(name: str, ainv: np.ndarray) -> dict[str, Any]:
    """Produce exact determinant labels and an independent CPU gradient check."""
    F = np.eye(3)
    G = F @ ainv
    analytic = first_piola_mpa(F, ainv)
    finite_difference = finite_difference_gradient(ainv)
    absolute_error = np.abs(analytic - finite_difference)
    return {
        "name": name,
        "F_fixed": "identity_3x3",
        "Ainv": ainv.tolist(),
        "Ainv_eigenvalues": np.linalg.eigvalsh(ainv).tolist(),
        "detF_physical": float(np.linalg.det(F)),
        "detAinv_parameter": float(np.linalg.det(ainv)),
        "detG_elastic": float(np.linalg.det(G)),
        "energy_density_MPa": energy_density_mpa(F, ainv),
        "first_piola_MPa": analytic.tolist(),
        "principal_first_piola_MPa": np.linalg.eigvalsh(analytic).tolist(),
        "first_piola_frobenius_MPa": float(np.linalg.norm(analytic)),
        "finite_difference_dPsi_dF_MPa": finite_difference.tolist(),
        "finite_difference_max_abs_error_MPa": float(absolute_error.max()),
    }


def main(output: Path) -> None:
    fiber_ainv = np.diag(
        (math.exp(C50_A), math.exp(-GAMMA * C50_A), math.exp(-GAMMA * C50_A))
    )
    offset_norm = float(np.linalg.norm(fiber_ainv - np.eye(3)))
    raw_offset = offset_norm / math.sqrt(3.0)
    raw_dilation = (1.0 + raw_offset) * np.eye(3)
    raw_compression = (1.0 - raw_offset) * np.eye(3)
    cases = [
        case("Fiber c50, a=ln(2), gamma=0.5", fiber_ainv),
        case("Raw6 isotropic dilation, equal ||Ainv-I||_F", raw_dilation),
        case("Raw6 isotropic compression, equal ||Ainv-I||_F", raw_compression),
    ]
    if max(row["finite_difference_max_abs_error_MPa"] for row in cases) >= 1.0e-8:
        message = "central finite difference did not reproduce first Piola"
        raise AssertionError(message)
    if not np.isclose(cases[0]["detAinv_parameter"], 1.0, atol=1.0e-14):
        message = "Fiber c50 must be volume preserving in Ainv"
        raise AssertionError(message)
    if not all(np.isclose(row["detF_physical"], 1.0) for row in cases):
        message = "this diagnostic must hold physical F at identity"
        raise AssertionError(message)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "CPU constitutive diagnostic; no equilibrium solve",
                "units": {
                    "stress_and_energy_density": "MPa",
                    "meaning": (
                        "pure-muscle constituent stress before the tetrahedron's "
                        "MuscleFraction quadrature weighting"
                    ),
                },
                "fixed_kinematics": {
                    "F": "identity_3x3",
                    "determinant_relation": "detG = detF * detAinv",
                },
                "material": {
                    "young_MPa": YOUNG_MPA,
                    "nu": NU,
                    "mu_MPa": MU_MPA,
                    "lambda_code_MPa": LAMBDA_CODE_MPA,
                },
                "fiber_c50": {
                    "a": C50_A,
                    "gamma": GAMMA,
                    "formula": "Ainv=exp(a)ff^T+exp(-gamma*a)(I-ff^T)",
                },
                "raw6_equal_offset_norm": {
                    "definition": "Raw6 Ainv=I+diag(s,s,s), s=plus_or_minus ||FiberAinv-I||_F/sqrt(3)",
                    "fiber_Ainv_minus_I_frobenius": offset_norm,
                    "absolute_isotropic_raw_offset": raw_offset,
                },
                "source_references": {
                    "active_energy": "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py:20-32",
                    "first_piola": "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py:36-46",
                    "stable_lame_convention": "exp/2026/09/07/face-activation-materials/src/face_physics.py:113-129",
                    "executed_muscle_material": "exp/2026/09/07/face-activation-materials/src/face_physics.py:245-258",
                    "fiber_and_raw6_maps": "exp/2026/09/07/face-activation-materials/src/activation_models.py:36-98",
                    "fraction_quadrature": "src/liblaf/apple/warp/fem/utils/_material.py:16-24",
                },
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
