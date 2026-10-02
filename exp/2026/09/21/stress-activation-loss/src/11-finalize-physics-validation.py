"""Finalize completed full-face derivative receipts without rerunning mechanics."""

from __future__ import annotations

import json
from pathlib import Path

import pydantic_settings as ps
from experiment import Profile
from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes
from run_support import archive, receipt, write_json
from stress_physics import StrictLineSearch, StrictPncg, SuccessPreferredFallbackSolver
from stress_study import FIXTURE, L_REF_MM, SMOOTH_LENGTH_M

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

NUMERICAL_SOLVER_CLASSES = (
    CupyCG,
    CupyMinRes,
    StrictLineSearch,
    StrictPncg,
    SuccessPreferredFallbackSolver,
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("11-validation-finalization")
    validation: Path = Path("10-validation")
    cpu_validation: Path = Path("07-activation-validation/receipt.json")
    tolerance: float = 0.02


def snapshot_receipts(source: Path) -> dict[str, dict[str, str]]:
    """Verify partial pre-failure snapshots before replacing them with a full set."""
    snapshots: dict[str, dict[str, str]] = {}
    root = ROOT.parent
    for snapshot in sorted((source / "sources").rglob("*.py")):
        original = root / snapshot.relative_to(source / "sources")
        assert original.is_file(), original
        snapshots[str(original)] = {
            "source": receipt(original),
            "snapshot": receipt(snapshot),
        }
        assert (
            snapshots[str(original)]["source"]["sha256"]
            == snapshots[str(original)]["snapshot"]["sha256"]
        ), original
    return snapshots


def main(cfg: Config) -> None:
    source = cherries.input(cfg.validation)
    cpu_path = cherries.input(cfg.cpu_validation)
    checks_path = source / "checks.json"
    prior_checks = json.loads(checks_path.read_text())
    checks = prior_checks["checks"]
    assert len(checks) == 8
    assert {
        (row["normal_weight"], row["direction"], row["epsilon"]) for row in checks
    } == {
        (weight, direction, epsilon)
        for weight in (0.0, 1.0)
        for direction in ("uniform-xx", "smooth-deviatoric")
        for epsilon in (0.002, 0.001)
    }
    assert all(row["relative_error"] < cfg.tolerance for row in checks)
    assert json.loads(cpu_path.read_text())["passed"]
    artifacts = {
        name: receipt(source / name)
        for name in ("mesh.npz", "neutral.json", "neutral-gradient.npz", "checks.json")
    }
    partial_snapshots = snapshot_receipts(source)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    recovery = {
        "kind": "metadata_finalization_without_mechanics_rerun",
        "reason": "The original validation completed all eight derivative checks, then source archival failed on a generated relative module path.",
        "original_checks_before_rewrite": artifacts["checks.json"],
        "numerical_artifacts_before_finalization": artifacts,
        "partial_snapshots_verified": partial_snapshots,
        "checks": checks,
        "tolerance": cfg.tolerance,
        "maximum_relative_error": max(row["relative_error"] for row in checks),
        "source_identity_limit": "No complete pre-run core-source archive exists; identities below were recorded during metadata recovery after the completed numerical run.",
    }
    write_json(out / "recovery.json", recovery)
    records = archive(source)
    protocol = {
        "stress_reference_MPa": 0.012 / (2.0 * 1.49),
        "l_ref_mm": L_REF_MM,
        "smooth_length_m": SMOOTH_LENGTH_M,
        "fixture": {
            name: receipt(FIXTURE / name)
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "sources": records,
        "cpu_validation": receipt(cpu_path),
        "geometry": receipt(source / "mesh.npz"),
        "recovery": receipt(out / "recovery.json"),
        "validation_scope": "Eight completed full-face directional checks at two normal weights, two directions, and two central-difference steps; metadata-only recovery did not rerun mechanics.",
    }
    write_json(source / "protocol.json", protocol)
    write_json(
        checks_path,
        {
            "passed": True,
            "status": "completed_metadata_finalization",
            "checks": checks,
            "neutral": json.loads((source / "neutral.json").read_text())["metrics"],
            "maximum_relative_error": recovery["maximum_relative_error"],
            "source_protocol": receipt(source / "protocol.json"),
            "recovery": receipt(out / "recovery.json"),
        },
    )
    cherries.log_metric("maximum_relative_error", recovery["maximum_relative_error"])


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
