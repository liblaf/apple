"""Audit directional-gradient scale at three fixed polished endpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import torch
from block_physics import Physics, configure
from experiment_profile import ProfileCometNoCommit
from liblaf.peach.linalg.cupy import CupyCG

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FROZEN_SOURCE_DIR = EXPERIMENT / "data" / "35-polish" / "sources"
FROZEN_RUNNER = FROZEN_SOURCE_DIR / "20-inverse-constraint-matrix.py"
LOG = logging.getLogger(__name__)

spec = importlib.util.spec_from_file_location("gradient_scale_frozen", FROZEN_RUNNER)
assert spec is not None
assert spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("37-gradient-scale", mkdir=True)
    polished_dir: Path = Path("data/35-polish")
    followup_dir: Path = Path("data/30-followups")
    endpoints: str = "strength-0.1/F-MS,strength-1/F-MS,strength-1/G-MS"
    epsilons: str = "0.01,0.003,0.001,0.0003"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen_sources() -> None:
    for name in (
        "20-inverse-constraint-matrix.py",
        "activation_models.py",
        "block_physics.py",
        "experiment_profile.py",
    ):
        assert sha256(HERE / name) == sha256(FROZEN_SOURCE_DIR / name), (
            f"Live {name} differs from the source frozen by the polish run"
        )


def write_provenance(output_dir: Path) -> None:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        FROZEN_RUNNER,
        FROZEN_SOURCE_DIR / "35-polish-endpoints.py",
        FROZEN_SOURCE_DIR / "activation_models.py",
        FROZEN_SOURCE_DIR / "block_physics.py",
        FROZEN_SOURCE_DIR / "experiment_profile.py",
    ]
    hashes = {}
    for source in sources:
        label = source.relative_to(EXPERIMENT).as_posix()
        hashes[label] = sha256(source)
        shutil.copy2(source, source_dir / source.name)
    base.write_json(
        output_dir / "provenance.json",
        {
            "sources": hashes,
            "frozen_source_dir": FROZEN_SOURCE_DIR.relative_to(EXPERIMENT).as_posix(),
        },
    )


def solver_receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": result["value"],
        "fit": result["fit"],
        "magnitude": result["magnitude"],
        "smoothness": result["smooth"],
        "forward": result["forward"],
        "adjoint": result["adjoint"],
    }


def audit_endpoint(
    physics: Physics,
    group: str,
    method: str,
    epsilons: tuple[float, ...],
    cfg: Config,
) -> dict[str, Any]:
    start = time.perf_counter()
    polished_dir = cfg.polished_dir / group / method
    summary_path = polished_dir / "summary.json"
    final_path = polished_dir / "final.npz"
    fixture_path = cfg.followup_dir / group / "fixture.npz"
    config_path = cfg.followup_dir / group / "run-config.json"
    assert summary_path.is_file()
    assert final_path.is_file()
    assert fixture_path.is_file()
    assert config_path.is_file()

    polished_summary = base.json.loads(summary_path.read_text())
    source_cfg = base.json.loads(config_path.read_text())
    values = source_cfg.copy()
    values.pop("output_dir", None)
    objective_cfg = base.Config(_cli_parse_args=False, **values)
    case = base.Case(**polished_summary["case"])
    assert polished_summary["target"] in ("clean", "noisy")
    assert case.mode in ("F", "G5")

    with np.load(fixture_path) as fixture:
        target = np.asarray(fixture[polished_summary["target"]]).copy()
        scale = float(fixture["D"])
    with np.load(final_path) as final:
        q = torch.as_tensor(np.asarray(final["q"]).copy())
        seed = np.asarray(final["u"]).copy()
        direction = torch.as_tensor(
            np.asarray(final["directional_gradient_direction"]).copy()
        )
        direction_mask = torch.as_tensor(
            np.asarray(final["directional_gradient_mask"]).copy()
        )

    assert q.shape == direction.shape == direction_mask.shape
    assert direction_mask.dtype == torch.bool
    assert torch.isfinite(q).all()
    assert torch.isfinite(direction).all()
    assert torch.all(direction[~direction_mask] == 0)
    assert math.isclose(
        float(torch.linalg.vector_norm(direction)), 1.0, rel_tol=0, abs_tol=1e-12
    )
    assert torch.allclose(am.project(q, case.mode, objective_cfg.amax), q)

    graph = am.face_graph(physics.points, physics.tets, physics.ids)
    angle = math.radians(objective_cfg.fiber_angle)
    fibers = torch.zeros((len(physics.ids), 3))
    fibers[:, 0] = math.cos(angle)
    fibers[:, 2] = math.sin(angle)
    objective = base.Objective(
        physics, target, scale, case, objective_cfg, fibers, graph
    )

    endpoint = objective(q, seed)
    analytic = float(torch.sum(endpoint["grad"] * direction))
    checks = []
    for index, epsilon in enumerate(epsilons):
        plus_q = q + epsilon * direction
        minus_q = q - epsilon * direction
        assert torch.allclose(
            am.project(plus_q, case.mode, objective_cfg.amax),
            plus_q,
            atol=1e-12,
            rtol=0,
        )
        assert torch.allclose(
            am.project(minus_q, case.mode, objective_cfg.amax),
            minus_q,
            atol=1e-12,
            rtol=0,
        )
        plus = objective(plus_q, endpoint["u"])
        minus = objective(minus_q, endpoint["u"])
        finite_difference = (plus["value"] - minus["value"]) / (2 * epsilon)
        relative_error = abs(finite_difference - analytic) / max(
            abs(finite_difference), abs(analytic), 1e-12
        )
        check = {
            "epsilon": epsilon,
            "analytic": analytic,
            "finite_difference": finite_difference,
            "absolute_error": abs(finite_difference - analytic),
            "relative_error": relative_error,
            "plus": solver_receipt(plus),
            "minus": solver_receipt(minus),
        }
        checks.append(check)
        cherries.log_metrics(
            {
                f"{group}/{method}/finite_difference": finite_difference,
                f"{group}/{method}/relative_error": relative_error,
            },
            step=index,
        )
        LOG.info(
            "%s/%s epsilon=%.4g analytic=%.9g FD=%.9g relerr=%.6g",
            group,
            method,
            epsilon,
            analytic,
            finite_difference,
            relative_error,
        )

    result = {
        "endpoint": f"{group}/{method}",
        "case": dataclasses.asdict(case),
        "target": polished_summary["target"],
        "direction_kind": polished_summary["directional_gradient_audit"][
            "direction_kind"
        ],
        "direction_mask_count": int(torch.count_nonzero(direction_mask)),
        "direction_mask_fraction": float(torch.mean(direction_mask.double())),
        "saved_direction_analytic": polished_summary["directional_gradient_audit"][
            "direction_analytic"
        ],
        "recomputed_direction_analytic": analytic,
        "endpoint_solver": solver_receipt(endpoint),
        "checks": checks,
        "num_perturbation_solves": 2 * len(checks),
        "wall_s": time.perf_counter() - start,
        "inputs": {
            "polished_summary": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
            "polished_final": {
                "path": str(final_path),
                "sha256": sha256(final_path),
            },
            "fixture": {"path": str(fixture_path), "sha256": sha256(fixture_path)},
            "run_config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
    }
    destination = cfg.output_dir / group / method
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "audit.npz",
        q=q.cpu().numpy(),
        direction=direction.cpu().numpy(),
        direction_mask=direction_mask.cpu().numpy(),
    )
    base.write_json(destination / "summary.json", result)
    return result


def main(cfg: Config) -> None:
    assert_frozen_sources()
    configure()
    am.validate()
    output_dir = cfg.output_dir.resolve()
    assert not (output_dir / "summary.json").exists(), (
        f"Refusing to overwrite completed audit {output_dir / 'summary.json'}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilons = tuple(float(value) for value in cfg.epsilons.split(","))
    assert epsilons == (0.01, 0.003, 0.001, 0.0003)
    endpoints = tuple(cfg.endpoints.split(","))
    assert endpoints in (
        ("strength-0.1/F-MS", "strength-1/F-MS", "strength-1/G-MS"),
        ("fiber-10deg/F-MS",),
    )
    base.write_json(output_dir / "config.json", cfg.model_dump(mode="json"))
    write_provenance(output_dir)

    physics = Physics(24, 10, rtol=1e-8, atol=1e-13)
    physics.diff.adjoint_solver = CupyCG(maxiter=40000, rtol=1e-9, atol=0.0)
    results = []
    for endpoint in endpoints:
        group, method = endpoint.split("/")
        results.append(audit_endpoint(physics, group, method, epsilons, cfg))
        base.write_json(
            output_dir / "summary.json",
            {
                "method": "fixed saved-direction central finite differences",
                "forward_rtol": physics.rtol,
                "forward_atol": physics.atol,
                "adjoint_rtol": 1e-9,
                "epsilons": epsilons,
                "num_endpoints": len(results),
                "num_perturbation_solves": sum(
                    result["num_perturbation_solves"] for result in results
                ),
                "results": results,
            },
        )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
