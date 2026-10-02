"""Run one matched released-axis continuation with the archived numerical code."""
# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]
FROZEN = GROUP / "data/00-frozen-source/apple"
HISTORICAL_SRC = FROZEN / "exp/2026/09/21/stress-activation-loss/src"
sys.path[:0] = [str(FROZEN / "src"), str(HISTORICAL_SRC)]

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Comet
from liblaf.cherries import core, plugins, profiles
from run_support import archive, receipt, run_stage, write_json
from stress_study import StressStudy

from liblaf import cherries

LOGGER = logging.getLogger(__name__)


class BranchLogging(plugins.Logging):
    @property
    def log_file(self) -> Path:
        label = os.environ["SMOOTHNESS_RUN_LABEL"]
        assert label in {"1x", "3x", "10x", "smoke"}
        return self.run.working_dir / "logs" / f"10-run-sweep-{label}.log"


class ProfileSweep(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run, disabled=os.environ.get("DEBUG") == "1"))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(BranchLogging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    multiplier: int = 1
    steps: int = 200
    output: Path = Path("10-sweep/multiplier-1")
    endpoint_diagnostics: bool = True


def historical_runner():
    spec = importlib.util.spec_from_file_location(
        "historical_chain", HISTORICAL_SRC / "46-run-active-strain-chain.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_loaded_sources(protocol: dict) -> dict:
    expected = {
        str((FROZEN.parent / Path(name).relative_to("sources")).resolve()): row[
            "sha256"
        ]
        for name, row in protocol["historical_source_freeze"].items()
    }
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if name.startswith("liblaf.apple"):
            assert path.is_relative_to(FROZEN / "src/liblaf/apple"), (name, path)
        if str(path) in expected:
            assert receipt(path)["sha256"] == expected[str(path)], path
            loaded[name] = receipt(path)
    return loaded


def verify_geometry(study: StressStudy, reference_path: Path) -> None:
    p = study.physics
    actual = {
        "rest_points": p.points,
        "tets": p.tets,
        "active_ids": p.ids,
        "skin_ids": study.skin_ids,
        "triangles": study.triangles,
        "target_displacement_skin": p.target[study.skin_ids],
        "skin_vertex_weights": study.weights,
        "active_volume_weights": study.active_weights,
        "edge_i": p.graph[0],
        "edge_j": p.graph[1],
        "edge_weight": p.graph[2],
        "regularizer_factor": study.regularizer_factor,
    }
    with np.load(reference_path, allow_pickle=False) as reference:
        assert set(actual) == set(reference.files)
        for name, value in actual.items():
            np.testing.assert_allclose(
                value, reference[name], rtol=1e-12, atol=0, err_msg=name
            )


def main(cfg: Config) -> None:
    shared = GROUP / "data/10-sweep"
    protocol = json.loads((shared / "protocol.json").read_text())
    historical = json.loads((shared / "historical-protocol.json").read_text())
    assert cfg.multiplier in protocol["multipliers"]
    assert 0 <= cfg.steps <= protocol["steps"]
    for key in ("parent_checkpoint", "mesh"):
        record = protocol[key]
        assert receipt(Path(record["path"]))["sha256"] == record["sha256"]
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    runner = historical_runner()
    study = StressStudy(activation_model="strain")
    assert study.physics.material_spec == historical["materials"]
    assert study.physics.forward_tolerance == historical["forward_tolerance"]
    verify_geometry(study, Path(protocol["mesh"]["path"]))
    sources = verify_loaded_sources(protocol)
    frozen = archive(out)
    runner.portable_source_freeze(out, frozen)
    runtime = runner.runtime_receipt(out)
    shutil.copy2(__file__, out / "runner.py")
    weight = protocol["base_smooth_weight"] * cfg.multiplier
    branch_protocol = {
        **protocol,
        "multiplier": cfg.multiplier,
        "smooth_weight": weight,
        "steps": cfg.steps,
        "actual_runtime": runtime,
        "geometry_matches_original": True,
        "loaded_numerical_sources": sources,
        "runner": receipt(out / "runner.py"),
    }
    write_json(out / "protocol.json", branch_protocol)
    with np.load(protocol["parent_checkpoint"]["path"], allow_pickle=False) as parent:
        assert str(parent["mode"]) == "rankone_fixed"
        assert int(parent["step"]) == 200
        assert str(parent["activation_model"]) == "strain"
        tensors = torch.as_tensor(parent["S"].copy())
        axes = torch.as_tensor(parent["fixed_axes"].copy())
        seed = parent["u"].copy()
    LOGGER.info(
        "Starting multiplier %d, eta %.9g, %d updates",
        cfg.multiplier,
        weight,
        cfg.steps,
    )
    summary = run_stage(
        study,
        out / "stage",
        "rankone_learned",
        protocol["normal_weight"],
        weight,
        tensors,
        seed,
        steps=cfg.steps,
        learning_rate=protocol["learning_rate"],
        adam_eps=protocol["adam_eps"],
        parent_axes=axes,
        activation_model="strain",
    )
    write_json(out / "summary.json", summary)
    assert summary["status"] == "completed_budget_not_convergence_certified", summary[
        "status"
    ]
    assert summary["last_step"] == cfg.steps, summary["last_step"]
    if cfg.endpoint_diagnostics:
        runner.endpoint_diagnostic(
            study,
            out / "stage",
            "rankone_learned",
            protocol["normal_weight"],
            weight,
            summary,
        )
    write_json(
        out / "completion.json",
        {
            "multiplier": cfg.multiplier,
            "steps": cfg.steps,
            "checkpoint": receipt(out / "stage/last.npz"),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "source_verification_after": verify_loaded_sources(protocol),
            "endpoint_diagnostics_requested": cfg.endpoint_diagnostics,
        },
    )
    cherries.log_metrics(
        {
            "multiplier": cfg.multiplier,
            "smooth_weight": weight,
            **summary["last_metrics"],
        }
    )
    LOGGER.info("Completed multiplier %d: %s", cfg.multiplier, summary["last_metrics"])


if __name__ == "__main__":
    cherries.main(main, profile=ProfileSweep)
