"""Extract and validate frozen Raw6 activation-induced Cauchy stress fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from active_stress import (
    activation_cauchy_difference,
    active_first_piola,
    deformation_gradient,
    effective_reference_stress,
    raw6_matrix,
)
from experiment import Profile

from liblaf import cherries

BRANCHES = ("smooth-off-normal", "smooth-on-normal")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("132-reference-continuation")
    verification: Path = Path("135-verification/checks.json")
    output: Path = Path("150-active-stress")
    validation_samples: int = 24


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def frozen_sources(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    required = (
        "volume_preserving_active",
        "study_physics",
        "liblaf.apple.warp.fem.func._misc",
        "liblaf.apple.warp.fem.func._deformation",
        "liblaf.apple.warp.fem.func._identity",
        "liblaf.apple.warp.fem.utils._material",
    )
    checked: dict[str, dict[str, str]] = {}
    for name in required:
        item = protocol["sources"][name]
        current, snapshot = Path(item["path"]), Path(item["snapshot"])
        assert digest(current) == item["sha256"], name
        assert digest(snapshot) == item["sha256"], name
        checked[name] = {"path": str(current), "sha256": item["sha256"]}
    return checked


def torch_energy(
    F: torch.Tensor,
    B: torch.Tensor,
    fraction: torch.Tensor,
    mu: float,
    la: float,
) -> torch.Tensor:
    """Frozen W times muscle fraction, matching get_dV's mixture weighting."""
    G = F @ B
    J = torch.linalg.det(F)
    return fraction * (
        0.5 * mu * ((G * G).sum(dim=(-2, -1)) - 3.0)
        - mu * (J - 1.0)
        + 0.5 * la * (J - 1.0).square()
    )


def validation(
    F: np.ndarray,
    B: np.ndarray,
    fraction: np.ndarray,
    mu: float,
    la: float,
    samples: int,
) -> dict[str, float | int | bool]:
    good = np.flatnonzero(np.linalg.det(F) > 0.0)
    assert len(good) >= samples
    # Fixed, evenly distributed sample indices make the receipt deterministic.
    ids = good[np.linspace(0, len(good) - 1, samples, dtype=np.int64)]
    Ft = torch.tensor(F[ids], dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B[ids], dtype=torch.float64)
    ft = torch.tensor(fraction[ids], dtype=torch.float64)
    energy = torch_energy(Ft, Bt, ft, mu, la).sum()
    (grad,) = torch.autograd.grad(energy, Ft)
    expected = fraction[ids, None, None] * active_first_piola(F[ids], B[ids], mu, la)
    gradient_error = np.asarray(grad) - expected
    identity = np.broadcast_to(np.eye(3), B[ids].shape).copy()
    q_identity = effective_reference_stress(identity, fraction[ids], mu)
    q = effective_reference_stress(B[ids], fraction[ids], mu)
    sigma, valid = activation_cauchy_difference(F[ids], q)
    sigma_sign, valid_sign = activation_cauchy_difference(
        F[ids], effective_reference_stress(-B[ids], fraction[ids], mu)
    )
    symmetry = sigma - np.swapaxes(sigma, -1, -2)
    result: dict[str, float | int | bool] = {
        "sample_count": int(samples),
        "all_sample_J_positive": bool(valid.all()),
        "autograd_first_piola_max_abs_error_MPa": float(np.max(np.abs(gradient_error))),
        "autograd_first_piola_max_rel_error": float(
            np.max(np.abs(gradient_error) / np.maximum(1e-12, np.abs(expected)))
        ),
        "identity_Q_max_abs_MPa": float(np.max(np.abs(q_identity))),
        "sign_invariance_max_abs_MPa": float(np.nanmax(np.abs(sigma - sigma_sign))),
        "cauchy_symmetry_max_abs_MPa": float(np.nanmax(np.abs(symmetry))),
        "formula": "autograd of fraction*[mu/2*(||F B||_F^2-3)-mu*(J-1)+lambda/2*(J-1)^2]",
    }
    assert valid_sign.all()
    assert result["autograd_first_piola_max_abs_error_MPa"] < 1e-12
    assert result["autograd_first_piola_max_rel_error"] < 1e-11
    assert result["identity_Q_max_abs_MPa"] == 0.0
    assert result["sign_invariance_max_abs_MPa"] == 0.0
    assert result["cauchy_symmetry_max_abs_MPa"] < 1e-14
    return result


def save_vtu(
    path: Path,
    deformed: np.ndarray,
    tets: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    active_points, inverse = np.unique(tets.reshape(-1), return_inverse=True)
    cells = np.column_stack(
        (np.full(len(tets), 4, dtype=np.int64), inverse.reshape(-1, 4))
    )
    mesh = pv.UnstructuredGrid(
        cells,
        np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8),
        deformed[active_points],
    )
    for name, values in fields.items():
        assert len(values) == len(tets), name
        mesh.cell_data[name] = values
    mesh.save(path, binary=True)


def extract_branch(
    source: Path,
    out: Path,
    branch: str,
    rest: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    fraction: np.ndarray,
    muscle_ids: np.ndarray,
    control_ids: np.ndarray,
    mu: float,
    la: float,
    samples: int,
) -> dict[str, Any]:
    with np.load(source / branch / "last.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 200
        assert bool(state["solver_valid"])
        assert bool(state["physical_volume_energy"])
        q, u = (
            np.asarray(state["q"], dtype=np.float64),
            np.asarray(state["u"], dtype=np.float64),
        )
    cells = tets[active_ids]
    B = raw6_matrix(q)
    F = deformation_gradient(rest, rest + u, cells)
    J = np.linalg.det(F)
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(J))
    Q = effective_reference_stress(B, fraction, mu)
    sigma, valid = activation_cauchy_difference(F, Q)
    eigenvalues = np.full((len(cells), 3), np.nan)
    eigenvectors = np.full((len(cells), 3, 3), np.nan)
    eigenvalues[valid], eigenvectors[valid] = np.linalg.eigh(sigma[valid])
    magnitude_kpa = np.full(len(cells), np.nan)
    magnitude_kpa[valid] = 1000.0 * np.linalg.norm(sigma[valid], axis=(1, 2))
    centers = rest[cells].mean(axis=1)
    deformed_centers = (rest + u)[cells].mean(axis=1)
    assert np.array_equal(active_ids, np.asarray(active_ids, dtype=np.int64))
    np.savez_compressed(
        out / branch / "stress-field.npz",
        active_ids=active_ids,
        rest_centers=centers,
        deformed_centers=deformed_centers,
        F=F,
        B=B,
        J=J,
        Q_MPa=Q,
        sigma_MPa=sigma,
        valid=valid,
        eigenvalues_MPa=eigenvalues,
        eigenvectors=eigenvectors,
        magnitude_kPa=magnitude_kpa,
        muscle_ids=muscle_ids,
        control_ids=control_ids,
        muscle_fraction=fraction,
    )
    save_vtu(
        out / branch / "stress.vtu",
        rest + u,
        cells,
        {
            "GlobalCellId": active_ids,
            "MuscleId": muscle_ids,
            "ActivationControlId": control_ids,
            "MuscleFraction": fraction,
            "J": J,
            "Valid": valid.astype(np.uint8),
            "Q_MPa": Q.reshape(len(Q), 9),
            "Sigma_MPa": sigma.reshape(len(sigma), 9),
            "Eigenvalues_MPa": eigenvalues,
            "MagnitudeKPa": magnitude_kpa,
        },
    )
    report = {
        "state": receipt(source / branch / "last.npz"),
        "active_cells": len(cells),
        "valid_cells": int(valid.sum()),
        "invalid_J_nonpositive_cells": int((~valid).sum()),
        "J_min": float(J.min()),
        "J_max": float(J.max()),
        "magnitude_kPa_99th_valid": float(np.nanpercentile(magnitude_kpa, 99)),
        "validation": validation(F, B, fraction, mu, la, samples),
    }
    return report


def main(cfg: Config) -> None:
    source = cherries.input(cfg.comparison_dir)
    audit_path = cherries.input(cfg.verification)
    protocol_path = source / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert audit["passed"] is True
    assert audit["all_completed"] is True
    assert audit["source_protocol_record"] == receipt(protocol_path)
    assert len(protocol["sources"]) == 100
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = frozen_sources(protocol)
    fixture_path = Path(protocol["fixture"]["volume.vtu"]["path"])
    assert (
        receipt(fixture_path)["sha256"] == protocol["fixture"]["volume.vtu"]["sha256"]
    )
    fixture = pv.read(fixture_path)
    with np.load(source / "mesh.npz", allow_pickle=False) as saved:
        rest = np.asarray(saved["rest_points"], dtype=np.float64)
        tets = np.asarray(saved["tets"], dtype=np.int64)
        active_ids = np.asarray(saved["active_ids"], dtype=np.int64)
    mask = np.asarray(fixture.cell_data["ActivationMask"], dtype=bool)
    assert np.array_equal(active_ids, np.flatnonzero(mask))
    fraction = np.asarray(fixture.cell_data["MuscleFraction"], dtype=np.float64)[
        active_ids
    ]
    muscle_ids = np.asarray(fixture.cell_data["MuscleId"], dtype=np.int32)[active_ids]
    control_ids = np.asarray(fixture.cell_data["ActivationControlId"], dtype=np.int64)[
        active_ids
    ]
    assert np.all(fraction > 0.0)
    assert np.all(muscle_ids >= 0)
    assert np.all(control_ids >= 0)
    mu = float(protocol["materials"]["muscle_mu_code_MPa"])
    la = float(protocol["materials"]["muscle_lambda_code_MPa"])
    report = {
        "passed": True,
        "definition": {
            "B": "I + symmetric Raw6 activation",
            "Q_MPa": "MuscleFraction * mu * (B B^T - I)",
            "sigma_MPa": "F Q F^T / det(F), only where det(F) > 0",
            "interpretation": "activation-dependent Cauchy stress difference at the same physical F; fraction-weighted to match the frozen mixed-cell energy integration",
            "invalid_policy": "J <= 0 stored as valid=false and NaN stress/eigensystem/magnitude; no abs(J) or clipping",
        },
        "inputs": {
            "protocol": receipt(protocol_path),
            "verification": receipt(audit_path),
            "mesh": receipt(source / "mesh.npz"),
            "volume_fixture": receipt(fixture_path),
            "frozen_sources": frozen,
        },
        "materials": {"mu_MPa": mu, "lambda_MPa": la},
        "branches": {},
    }
    for branch in BRANCHES:
        branch_out = output / branch
        branch_out.mkdir()
        report["branches"][branch] = extract_branch(
            source,
            output,
            branch,
            rest,
            tets,
            active_ids,
            fraction,
            muscle_ids,
            control_ids,
            mu,
            la,
            cfg.validation_samples,
        )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
