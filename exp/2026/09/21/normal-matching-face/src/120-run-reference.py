"""Fresh neutral face fitting with the selected fixed reference-length loss."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile
from reference_study import ReferenceStudy
from study import FIXTURE, ROOT

from liblaf import cherries

SPEC = importlib.util.spec_from_file_location(
    "reference_legacy_runner", Path(__file__).with_name("10-run.py")
)
assert SPEC
assert SPEC.loader
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("130-reference-fit")
    selected_loss: Path = Path("110-shape-loss-config/loss-config.json")
    steps: int = 200
    learning_rate: float = 0.3
    legacy_adam_eps: float = 0.01
    legacy_smooth_coefficient: float = 0.003214147722027223
    smooth_length_m: float = 0.005
    checkpoint_interval: int = 10
    validation_only: bool = False
    gate: Path = Path("121-reference-validation")
    smoke_audit: Path = Path("124-reference-smoke-verification/checks.json")
    branches: str = "smooth-off-normal,smooth-on-normal"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": legacy.digest(path)}


def main(cfg: Config) -> None:  # noqa: PLR0915
    selected_path = cherries.input(cfg.selected_loss)
    selected = json.loads(selected_path.read_text())
    assert selected["length_mode"] == "fixed_physical_length"
    assert selected["normal_weight"] == 1.0
    l_ref_mm = float(selected["l_ref_mm"])
    assert np.isfinite(l_ref_mm)
    assert l_ref_mm > 0
    scale = l_ref_mm**2
    effective = legacy.Config(
        output=cfg.output,
        steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        adam_eps=cfg.legacy_adam_eps / scale,
        beta=selected["normal_weight"],
        smooth_coefficient=cfg.legacy_smooth_coefficient / scale,
        smooth_length_m=cfg.smooth_length_m,
        checkpoint_interval=cfg.checkpoint_interval,
        validation_only=cfg.validation_only,
        gate=cfg.gate,
        branches=cfg.branches,
    )
    assert effective.steps >= 1
    assert effective.checkpoint_interval > 0
    branches = effective.branches.split(",")
    assert set(branches) == {"smooth-off-normal", "smooth-on-normal"}
    assert len(branches) == 2
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    legacy.write_json(out / "config.json", effective.model_dump(mode="json"))
    shutil.copy2(selected_path, out / "loss-config.json")
    study = ReferenceStudy(l_ref_mm, cfg.smooth_length_m)
    p = study.physics
    np.savez_compressed(
        out / "mesh.npz",
        rest_points=p.points,
        skin_ids=study.skin_ids,
        triangles=study.triangles,
        target_displacement_skin=p.target[study.skin_ids],
        skin_vertex_weights=study.weights,
        initial_u=np.zeros_like(p.points),
        active_ids=p.ids,
        tets=p.tets,
        edge_i=p.graph[0],
        edge_j=p.graph[1],
        edge_weight=p.graph[2],
        active_volume_weights=study.active_weights,
        regularizer_factor=study.regularizer_factor,
        fixed_mask=np.asarray(p.mesh.point_data["FixedMask"], bool),
        fixed_values=np.asarray(p.mesh.point_data["FixedValue"]),
    )
    sources = legacy.archive(out)
    fixture = {
        name: record(FIXTURE / name)
        for name in ("volume.vtu", "skin.vtp", "summary.json")
    }
    protocol_path = Path(__file__).parents[1] / "docs/115-reference-fit-protocol.md"
    protocol = {
        "config": effective.model_dump(mode="json"),
        "request_config": cfg.model_dump(mode="json"),
        "selected_loss": selected,
        "selected_loss_record": record(selected_path),
        "normalization": study.normalization,
        "reference_normalization": {
            "l_ref_mm": l_ref_mm,
            "position_coefficient": 1 / scale,
            "normal_weight": selected["normal_weight"],
            "legacy_objective_scale": scale,
            "equivalent_previous_beta": scale
            * study.normalization["N0"]
            / study.normalization["L20"],
            "legacy_adam_eps": cfg.legacy_adam_eps,
            "legacy_smooth_coefficient": cfg.legacy_smooth_coefficient,
            "beta_key_semantics": "direct normal weight in reused runner interface; no initial-error multiplier",
        },
        "start": "q=0, B=I, u=0, fresh zero Adam moments; no resume or fitted inputs",
        "activation": "288235 independent Raw6 symmetric tensors, no projections",
        "objective": "L2/l_ref_mm**2 + normal_weight*normal_chord_squared + smooth_coefficient*R",
        "materials": p.material_spec,
        "forward_tolerance": p.forward_tolerance,
        "surface_points": len(study.skin_ids),
        "surface_triangles": len(study.triangles),
        "volume_points": len(p.points),
        "tetrahedra": len(p.tets),
        "active_cells": len(p.ids),
        "fixture": fixture,
        "sources": sources,
        "protocol_record": record(protocol_path),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": [sys.executable, *sys.argv],
        },
        "inverse_stationarity_claimed": False,
        "mechanical_stability_claimed": False,
    }
    legacy.write_json(out / "protocol.json", protocol)
    shutil.copy2(protocol_path, out / "protocol.md")
    if cfg.validation_only:
        legacy.gradient_check(study, effective, out)
        return
    gate_dir = cherries.input(cfg.gate)
    gate = json.loads((gate_dir / "gradient-validation.json").read_text())
    assert gate["status"] == "passed"
    gate_protocol = json.loads((gate_dir / "protocol.json").read_text())
    assert {k: v["sha256"] for k, v in sources.items()} == {
        k: v["sha256"] for k, v in gate_protocol["sources"].items()
    }
    for key in (
        "fixture",
        "normalization",
        "reference_normalization",
        "selected_loss_record",
        "protocol_record",
    ):
        assert protocol[key] == gate_protocol[key], key
    for key in (
        "learning_rate",
        "adam_eps",
        "beta",
        "smooth_coefficient",
        "smooth_length_m",
    ):
        assert protocol["config"][key] == gate_protocol["config"][key], key
    shutil.copy2(
        gate_dir / "gradient-validation.json", out / "gradient-validation.json"
    )
    normal_gate_path = cherries.input("05-normal-verification/checks.json")
    assert json.loads(normal_gate_path.read_text())["passed"]
    shutil.copy2(normal_gate_path, out / "normal-validation.json")
    preflight = {
        "selected_loss": record(selected_path),
        "protocol": record(protocol_path),
        "gradient_protocol": record(gate_dir / "protocol.json"),
        "gradient_validation": record(gate_dir / "gradient-validation.json"),
        "normal_validation": record(normal_gate_path),
    }
    if cfg.steps > 1:
        smoke_audit_path = cherries.input(cfg.smoke_audit)
        smoke_audit = json.loads(smoke_audit_path.read_text())
        assert smoke_audit["passed"]
        assert smoke_audit["all_completed"]
        smoke_protocol_path = Path(smoke_audit["source_protocol_record"]["path"])
        assert record(smoke_protocol_path) == smoke_audit["source_protocol_record"]
        smoke_protocol = json.loads(smoke_protocol_path.read_text())
        assert smoke_protocol["config"]["steps"] == 1
        assert {k: v["sha256"] for k, v in sources.items()} == {
            k: v["sha256"] for k, v in smoke_protocol["sources"].items()
        }
        for key in (
            "fixture",
            "normalization",
            "reference_normalization",
            "selected_loss_record",
            "protocol_record",
        ):
            assert protocol[key] == smoke_protocol[key], key
        for key in (
            "learning_rate",
            "adam_eps",
            "beta",
            "smooth_coefficient",
            "smooth_length_m",
            "branches",
        ):
            assert protocol["config"][key] == smoke_protocol["config"][key], key
        preflight["smoke_audit"] = record(smoke_audit_path)
        preflight["smoke_protocol"] = record(smoke_protocol_path)
    legacy.write_json(out / "reference-preflight.json", preflight)
    summaries = {
        branch: legacy.run_branch(study, effective, out, branch) for branch in branches
    }
    legacy.write_json(out / "summary.json", summaries)
    assert all(row["failure"] is None for row in summaries.values()), summaries


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
