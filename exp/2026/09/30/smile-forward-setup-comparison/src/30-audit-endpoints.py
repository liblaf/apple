"""Independently audit completed fixed-Smile forward endpoints on CPU."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS = ROOT / "exp/2026/09/21/stress-activation-loss"
SOURCE = STRESS / "data/51-visualization-checkpoints-002/l2-normal"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
REFERENCE = ROOT / "exp/2026/09/23/new-neutral/data/reference-clearance-002"
sys.path.insert(0, str(STRESS / "src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    """Locations of the two completed Newton-only forward branches."""

    output: Path = Path("30-audit")
    inverse_run: Path = GROUP / "data/10-inverse-setup-newton"
    new_run: Path = GROUP / "data/10-new-setup-newton-checkpointed"
    activation: Path = SOURCE / "l2-normal-rankone_fixed/last.npz"
    mesh: Path = SOURCE / "mesh.npz"
    fixture: Path = FIXTURE
    reference: Path = REFERENCE / "reference-clearance.npz"
    chunk_cells: int = 50_000
    energy_roundoff_j: float = 1e-9


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_trace(path: Path) -> list[dict[str, Any]]:
    required = {
        "iteration",
        "phase",
        "elapsed_seconds",
        "energy_j",
        "force_n",
        "inverted_cells",
        "cell_count",
        "inverted_percent",
        "minimum_detf",
    }
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None, path
        assert required <= set(reader.fieldnames), (
            path,
            required - set(reader.fieldnames),
        )
        rows = list(reader)
    assert rows, path
    result = []
    for raw in rows:
        row: dict[str, Any] = {"phase": raw["phase"]}
        for key in ("iteration", "inverted_cells", "cell_count"):
            value = float(raw[key])
            assert value.is_integer(), (path, key, raw[key])
            row[key] = int(value)
        for key in (
            "elapsed_seconds",
            "energy_j",
            "force_n",
            "inverted_percent",
            "minimum_detf",
        ):
            row[key] = float(raw[key])
            assert math.isfinite(row[key]), (path, key, row[key])
        assert row["phase"] in {"initial", "newton", "final"}, (path, row["phase"])
        assert row["iteration"] >= 0
        assert row["elapsed_seconds"] >= 0
        assert row["force_n"] >= 0
        assert row["cell_count"] > 0
        assert 0 <= row["inverted_cells"] <= row["cell_count"]
        expected_percent = 100.0 * row["inverted_cells"] / row["cell_count"]
        np.testing.assert_allclose(
            row["inverted_percent"], expected_percent, rtol=0, atol=1e-12
        )
        result.append(row)
    iterations = np.asarray([row["iteration"] for row in result])
    elapsed = np.asarray([row["elapsed_seconds"] for row in result])
    energy = np.asarray([row["energy_j"] for row in result])
    np.testing.assert_array_equal(iterations, np.arange(len(result)))
    assert np.all(np.diff(elapsed) >= 0), path
    assert np.all(np.diff(energy) <= 1e-9), path
    return result


def detf_metrics(
    rest_points: np.ndarray,
    displacement: np.ndarray,
    tets: np.ndarray,
    *,
    chunk_cells: int,
) -> dict[str, Any]:
    """Compute physical det(F) without loading model or solver code."""
    assert chunk_cells > 0
    assert rest_points.shape == displacement.shape
    assert rest_points.ndim == 2
    assert rest_points.shape[1] == 3
    moved = rest_points + displacement
    assert np.isfinite(moved).all()
    inverted, minimum = 0, math.inf
    for start in range(0, len(tets), chunk_cells):
        cells = tets[start : start + chunk_cells]
        dm = np.transpose(
            rest_points[cells[:, 1:]] - rest_points[cells[:, :1]], (0, 2, 1)
        )
        ds = np.transpose(moved[cells[:, 1:]] - moved[cells[:, :1]], (0, 2, 1))
        determinant = np.linalg.det(ds) / np.linalg.det(dm)
        assert np.isfinite(determinant).all()
        inverted += int(np.count_nonzero(determinant <= 0.0))
        minimum = min(minimum, float(determinant.min()))
    return {
        "inverted_cells": inverted,
        "cell_count": len(tets),
        "inverted_percent": 100.0 * inverted / len(tets),
        "minimum_detf": minimum,
    }


def check_final_row(summary: dict[str, Any], row: dict[str, Any]) -> None:
    final = summary["final"]
    assert isinstance(final, dict)
    assert final["phase"] == row["phase"]
    for key in ("iteration", "inverted_cells", "cell_count"):
        assert final[key] == row[key], (key, final[key], row[key])
    for key in (
        "elapsed_seconds",
        "energy_j",
        "force_n",
        "inverted_percent",
        "minimum_detf",
    ):
        np.testing.assert_allclose(final[key], row[key], rtol=0, atol=0, err_msg=key)


def audit_branch(
    *,
    label: str,
    directory: Path,
    source_s: np.ndarray,
    source_tets: np.ndarray,
    source_active_ids: np.ndarray,
    expected_points: np.ndarray,
    fixed: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    summary_path, final_path, trace_path = (
        directory / "summary.json",
        directory / "final.npz",
        directory / "trace.csv",
    )
    for path in (summary_path, final_path, trace_path):
        assert path.is_file(), path
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "smile-fixed-activation-forward-setup-v1"
    assert summary["config"]["branch"] == label
    assert (
        summary["activation_policy"]
        == "exact fixed Stage 3 S; no activation optimization or scaling; B=I+S"
    )
    assert summary["inputs"]["activation"]["sha256"] == sha256(cfg.activation)
    assert summary["inputs"]["mesh"]["sha256"] == sha256(cfg.mesh)
    assert summary["activation_array_sha256"] == array_digest(source_s)
    trace = read_trace(trace_path)
    check_final_row(summary, trace[-1])
    with np.load(final_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"u", "S", "rest_points", "tets", "active_ids"}
        u = archive["u"].copy()
        saved_s = archive["S"].copy()
        points = archive["rest_points"].copy()
        tets = archive["tets"].copy()
        active_ids = archive["active_ids"].copy()
    np.testing.assert_array_equal(saved_s, source_s)
    assert array_digest(saved_s) == array_digest(source_s)
    np.testing.assert_array_equal(points, expected_points)
    np.testing.assert_array_equal(tets, source_tets)
    np.testing.assert_array_equal(active_ids, source_active_ids)
    assert u.ndim == 2
    assert u.shape[1] == 3
    assert u.shape[0] >= len(points)
    assert np.isfinite(u).all()
    np.testing.assert_allclose(u[: len(points)][fixed], 0.0, rtol=0, atol=1e-12)
    metrics = detf_metrics(points, u[: len(points)], tets, chunk_cells=cfg.chunk_cells)
    for key in ("inverted_cells", "cell_count"):
        assert metrics[key] == trace[-1][key], (
            label,
            key,
            metrics[key],
            trace[-1][key],
        )
    for key in ("inverted_percent", "minimum_detf"):
        np.testing.assert_allclose(
            metrics[key], trace[-1][key], rtol=0, atol=1e-12, err_msg=f"{label}/{key}"
        )
    return {
        "numerical_status": summary["status"],
        "recorded_force_converged": bool(summary.get("force_converged", False)),
        "recorded_orientation_valid": bool(summary.get("orientation_valid", False)),
        "recorded_physical_valid": bool(summary.get("physical_valid", False)),
        "inputs": {
            name: record(path)
            for name, path in {
                "summary": summary_path,
                "final": final_path,
                "trace": trace_path,
            }.items()
        },
        "final_trace": trace[-1],
        "cpu_detf": metrics,
        "appended_rigid_nodes": int(len(u) - len(points)),
        "validity": "CPU verification of saved state, source identity, trace, and det(F); forces and contact were not recomputed.",
    }


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not output.exists(), output
    assert cfg.energy_roundoff_j == 1e-9
    with (
        np.load(cfg.activation, allow_pickle=False) as checkpoint,
        np.load(cfg.mesh, allow_pickle=False) as mesh,
    ):
        assert str(checkpoint["mode"]) == "rankone_fixed"
        assert str(checkpoint["activation_model"]) == "strain"
        source_s = checkpoint["S"].copy()
        source_points = mesh["rest_points"].copy()
        source_tets = mesh["tets"].copy()
        source_active_ids = mesh["active_ids"].copy()
    assert source_s.shape == (len(source_active_ids), 3, 3)
    fixture = pv.read(cfg.fixture / "volume.vtu")
    fixture_tets = np.asarray(fixture.cells).reshape(-1, 5)[:, 1:]
    fixed = np.asarray(fixture.point_data["IsFixed"], dtype=bool)
    np.testing.assert_array_equal(fixture.points, source_points)
    np.testing.assert_array_equal(fixture_tets, source_tets)
    np.testing.assert_array_equal(
        np.flatnonzero(fixture.cell_data["ActivationMask"]), source_active_ids
    )
    np.testing.assert_array_equal(
        fixture.point_data["FixedMask"], np.repeat(fixed[:, None], 3, axis=1)
    )
    with np.load(cfg.reference, allow_pickle=False) as archive:
        repaired_points = archive["repaired_points_m"].copy()
    assert repaired_points.shape == source_points.shape
    result = {
        "schema": "smile-forward-setup-cpu-endpoint-audit-v1",
        "status": "verified_completed",
        "scope": "Independent NumPy det(F) audit of completed saved endpoints; GPU force and contact are outside scope.",
        "source": {
            "activation": record(cfg.activation),
            "mesh": record(cfg.mesh),
            "fixture_volume": record(cfg.fixture / "volume.vtu"),
            "repaired_reference": record(cfg.reference),
            "stage": "rankone_fixed",
            "activation_model": "strain",
            "S_sha256": array_digest(source_s),
            "counts": {
                "points": len(source_points),
                "tets": len(source_tets),
                "active_cells": len(source_active_ids),
                "fixed_points": int(fixed.sum()),
            },
        },
        "branches": {
            "inverse_setup": audit_branch(
                label="inverse-setup",
                directory=cfg.inverse_run,
                source_s=source_s,
                source_tets=source_tets,
                source_active_ids=source_active_ids,
                expected_points=source_points,
                fixed=fixed,
                cfg=cfg,
            ),
            "new_setup": audit_branch(
                label="new-setup",
                directory=cfg.new_run,
                source_s=source_s,
                source_tets=source_tets,
                source_active_ids=source_active_ids,
                expected_points=repaired_points,
                fixed=fixed,
                cfg=cfg,
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            f"{name}/inverted_percent": branch["cpu_detf"]["inverted_percent"]
            for name, branch in result["branches"].items()
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
