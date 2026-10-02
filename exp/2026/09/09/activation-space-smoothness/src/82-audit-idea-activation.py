# ruff: noqa: EM101, EM102, PLR0915, PT018, TRY003
"""Audit common effective activation fields for the six selected checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
# E=0.03 MPa and nu=0.49 for muscle, as set_material() records in FacePhysics.
MU_MPA = 0.03 / (2.0 * (1.0 + 0.49))
QREF_MPA = 3.0 * MU_MPA
BACKWARD_ERROR_FACTOR = 64.0


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    selection: Path = GROUP / "data/80-idea-result-selection/selection.json"
    fixture: Path = FIXTURE
    output_dir: Path = GROUP / "data/82-idea-activation-audit-v2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def raw6_b(q: np.ndarray) -> np.ndarray:
    """Historical Raw6 map, whose off-diagonals are unscaled coordinates."""
    assert q.ndim == 2 and q.shape[1] == 6
    b = np.zeros((len(q), 3, 3), dtype=np.float64)
    b[:, 0, 0], b[:, 1, 1], b[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    b[:, 0, 1] = b[:, 1, 0] = q[:, 3]
    b[:, 1, 2] = b[:, 2, 1] = q[:, 4]
    b[:, 0, 2] = b[:, 2, 0] = q[:, 5]
    b += np.eye(3)
    return b


def z_for(
    state: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, str]:
    identifier = state["id"]
    if identifier.startswith("raw6-"):
        b = raw6_b(arrays["q"])
        z = b @ np.swapaxes(b, 1, 2) - np.eye(3)
        if "Ainv" in arrays and not np.allclose(b, arrays["Ainv"], atol=1e-13):
            raise ValueError(f"Raw6 B reconstruction disagrees with Ainv: {identifier}")
        if "Z" in arrays and not np.allclose(z, arrays["Z"], atol=1e-12):
            raise ValueError(
                f"Raw6 Z reconstruction disagrees with stored Z: {identifier}"
            )
        return z, "Z=B B^T-I from q"
    if identifier.startswith("axis-"):
        v = arrays["q"]
        z = (2.0 + np.sum(v * v, axis=1)[:, None, None]) * np.einsum("ni,nj->nij", v, v)
        if not np.allclose(z, arrays["Z"], atol=1e-12):
            raise ValueError(
                f"axis Z reconstruction disagrees with stored Z: {identifier}"
            )
        return z, "Z=(2+||v||^2)vv^T from saved v"
    if identifier == "psd-off-1024":
        q = arrays["q"]
        q_matrix = np.empty((len(q), 3, 3), dtype=np.float64)
        root2 = np.sqrt(2.0)
        q_matrix[:, 0, 0], q_matrix[:, 1, 1], q_matrix[:, 2, 2] = q.T[:3]
        q_matrix[:, 0, 1] = q_matrix[:, 1, 0] = q[:, 3] / root2
        q_matrix[:, 1, 2] = q_matrix[:, 2, 1] = q[:, 4] / root2
        q_matrix[:, 0, 2] = q_matrix[:, 2, 0] = q[:, 5] / root2
        if not np.allclose(arrays["Q"], QREF_MPA * q_matrix, atol=1e-13):
            raise ValueError("PSD Q disagrees with archived QREF * orthonormal q")
        return arrays["Q"] / MU_MPA, "Z=Q/mu from saved Q"
    raise ValueError(identifier)


def deformation_gradients(
    rest: np.ndarray, u: np.ndarray, tets: np.ndarray, active_ids: np.ndarray
) -> np.ndarray:
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm[active_ids])
    deformed = rest + u
    ds = np.transpose(
        deformed[tets[active_ids, 1:]] - deformed[tets[active_ids, :1]],
        (0, 2, 1),
    )
    return ds @ dm_inv


def weighted_mean(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * value) / np.sum(weights))


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    selected = json.loads(cfg.selection.read_text())
    mesh = pv.read(cfg.fixture / "volume.vtu")
    rest = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    fixture_ids = np.flatnonzero(
        np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    )
    fractions = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    weights = np.linalg.det(dm[fixture_ids]) / 6.0 * fractions[fixture_ids]
    if not (np.all(weights > 0) and len(fixture_ids) == 288235):
        raise ValueError("unexpected active fixture geometry")

    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for state in selected["states"]:
        if state["kind"] != "full":
            continue
        path = Path(state["path"])
        if digest(path) != state["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        active_ids = arrays["active_ids"]
        if not np.array_equal(active_ids, fixture_ids):
            raise ValueError(f"active IDs differ from fixture: {state['id']}")
        if "rest_points" in arrays and not np.array_equal(arrays["rest_points"], rest):
            raise ValueError(f"rest geometry differs from fixture: {state['id']}")
        z, construction = z_for(state, arrays)
        z = 0.5 * (z + np.swapaxes(z, 1, 2))
        eigenvalues, eigenvectors = np.linalg.eigh(z)
        spectral_norm = np.max(np.abs(eigenvalues), axis=1)
        negative_tolerance = (
            BACKWARD_ERROR_FACTOR
            * np.finfo(np.float64).eps
            * np.maximum(spectral_norm, 1.0)
        )
        negative_modes = eigenvalues < -negative_tolerance[:, None]
        if state["id"].startswith("axis-") and np.any(negative_modes):
            raise ValueError(
                f"rank-one PSD axis field has material negative mode: {state['id']}"
            )
        positive = np.maximum(eigenvalues, 0.0)
        # Discard only backward-error-scale roundoff from signed-mode metrics.
        negative = np.where(negative_modes, eigenvalues, 0.0)
        total_energy = np.sum(eigenvalues * eigenvalues, axis=1)
        positive_energy = np.sum(positive * positive, axis=1)
        leading_fraction = np.divide(
            positive[:, 2] ** 2,
            positive_energy,
            out=np.zeros_like(total_energy),
            where=positive_energy > 0,
        )
        non_dominant_fraction = np.divide(
            total_energy - positive[:, 2] ** 2,
            total_energy,
            out=np.zeros_like(total_energy),
            where=total_energy > 0,
        )
        # Roundoff can only create an imperceptible negative numerator.
        if np.min(non_dominant_fraction) < -1e-12:
            raise ValueError(f"negative non-dominant energy: {state['id']}")
        non_dominant_fraction = np.clip(non_dominant_fraction, 0.0, 1.0)
        negative_fraction = np.divide(
            np.sum(negative * negative, axis=1),
            total_energy,
            out=np.zeros_like(total_energy),
            where=total_energy > 0,
        )
        amplitude = 1.0 - 1.0 / np.sqrt(1.0 + positive[:, 2])
        f = deformation_gradients(rest, arrays["u"], tets, active_ids)
        transported_norm = np.linalg.norm(
            np.einsum("nij,nj->ni", f, eigenvectors[:, :, 2]), axis=1
        )
        if not np.all(np.isfinite(transported_norm)):
            raise ValueError(f"nonfinite transported directions: {state['id']}")
        row = {
            "id": state["id"],
            "step": int(state["step"]),
            "z_construction": construction,
            "lambda_min_exact": float(np.min(eigenvalues)),
            "lambda_min_p01": float(np.quantile(eigenvalues[:, 0], 0.01)),
            "lambda_max_p50": float(np.quantile(eigenvalues[:, 2], 0.50)),
            "lambda_max_p99": float(np.quantile(eigenvalues[:, 2], 0.99)),
            "negative_backward_error_tolerance_min": float(np.min(negative_tolerance)),
            "negative_backward_error_tolerance_max": float(np.max(negative_tolerance)),
            "negative_eigenvalue_fraction": float(np.mean(negative_modes)),
            "cells_with_negative_mode_fraction": float(np.mean(negative_modes[:, 0])),
            "volume_weighted_negative_spectral_energy_fraction": weighted_mean(
                negative_fraction, weights
            ),
            "volume_weighted_leading_positive_spectral_energy_fraction": weighted_mean(
                leading_fraction, weights
            ),
            "volume_weighted_non_dominant_energy_fraction": weighted_mean(
                non_dominant_fraction, weights
            ),
            "cells_leading_positive_energy_below_0_9_fraction": float(
                np.mean((positive_energy > 0) & (leading_fraction < 0.9))
            ),
            "volume_weighted_display_amplitude": weighted_mean(amplitude, weights),
            "deformed_transport_norm_p01": float(np.quantile(transported_norm, 0.01)),
            "deformed_transport_norm_min": float(np.min(transported_norm)),
        }
        rows.append(row)
        checks[state["id"]] = {
            "checkpoint": str(path),
            "rest_geometry": "checkpoint equals fixture"
            if "rest_points" in arrays
            else "fixture used; checkpoint has no rest_points",
            "z_construction": construction,
            "transport": "F times rest dominant eigenvector, then normalize; no zero/nonfinite vectors",
        }
        cherries.log_metrics(
            {
                "activation": {
                    key: value for key, value in row.items() if isinstance(value, float)
                }
            }
        )

    fields = list(rows[0])
    with (cfg.output_dir / "state-statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "completed",
        "mu_mpa": MU_MPA,
        "mu_source": "E=0.03 MPa, nu=0.49, mu=E/(2*(1+nu)); verified against physical-volume baseline summary and PSD QREF=3*mu source",
        "fixture": str(cfg.fixture),
        "rest_geometry": "volume.vtu points; every checkpoint carrying rest_points matched exactly",
        "negative_mode_rule": "A mode is material only when lambda < -64*eps*max(||Z||_2, 1) for its own symmetric tensor. Exact minima and the tolerance range are retained in each row.",
        "axis_roundoff_interpretation": "C=vv^T and Z=(2+||v||^2)vv^T are algebraically rank-one PSD. Axis eigenvalues inside the per-tensor backward-error tolerance are numerical roundoff, not negative modes.",
        "line_semantics": "dominant positive Z eigenvector in rest coordinates, transported by F and normalized; sign is arbitrary",
        "omitted_by_line": "negative eigenmodes and all non-leading positive modes",
        "recommended_companion": "NonDominantEnergyFraction=1-max(lambda_max(Z),0)^2/||Z||_F^2, defined as zero for exactly zero Z; map it at fixed 0..1. It includes negative and non-leading positive modes. Report negative-eigenvalue fractions separately.",
        "checks": checks,
        "states": rows,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    cherries.log_output(cfg.output_dir / "state-statistics.csv")
    cherries.log_output(cfg.output_dir / "summary.json")


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
