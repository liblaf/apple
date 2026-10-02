"""Run smoothing-only fiber fits on the three held-out noise realizations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import torch
from block_physics import ROOT, Physics, array_hash, configure
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "20-inverse-constraint-matrix.py"
LOG = logging.getLogger(__name__)
SEEDS = (20260917, 20260928, 20261009)

spec = importlib.util.spec_from_file_location(
    "activation_matrix_smoothing_holdout", RUNNER_PATH
)
assert spec is not None
assert spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("32-smoothing-holdout", mkdir=True)
    source_dir: Path = cherries.input("30-followups")
    steps: int = 240


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_sources(output_dir: Path, frozen_source_dir: Path) -> None:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in (
        Path(__file__),
        RUNNER_PATH,
        HERE / "activation_models.py",
        HERE / "block_physics.py",
        HERE / "experiment_profile.py",
    ):
        if source.name != Path(__file__).name:
            frozen_source = frozen_source_dir / source.name
            assert frozen_source.is_file(), f"Missing frozen source: {frozen_source}"
            assert sha256(source) == sha256(frozen_source), (
                f"Live {source.name} differs from the held-out runner snapshot"
            )
        hashes[source.relative_to(ROOT).as_posix()] = sha256(source)
        shutil.copy2(source, source_dir / source.name)
    base.write_json(output_dir / "source-hashes.json", hashes)


def load_fixture(
    source_group: Path, destination_group: Path, physics: Physics
) -> tuple[dict[str, Any], str]:
    source_fixture = source_group / "fixture.npz"
    source_metadata = source_group / "fixture.json"
    assert source_fixture.is_file()
    assert source_metadata.is_file()
    destination_group.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_fixture, destination_group / "fixture.npz")
    shutil.copy2(source_metadata, destination_group / "fixture.json")
    for name in ("rest.vtu", "clean.vtu"):
        source = source_group / name
        assert source.is_file()
        shutil.copy2(source, destination_group / name)

    with np.load(source_fixture) as archive:
        fixture = {name: np.asarray(archive[name]).copy() for name in archive.files}
    assert np.array_equal(fixture["points"], physics.points)
    assert np.array_equal(fixture["tets"], physics.tets)
    assert np.array_equal(fixture["active_ids"], physics.ids)
    metadata = json.loads(source_metadata.read_text())
    assert metadata["points_hash"] == array_hash(physics.points)
    assert metadata["tets_hash"] == array_hash(physics.tets)
    assert metadata["active_ids_hash"] == array_hash(physics.ids)
    assert metadata["target_hashes"]["clean"] == array_hash(fixture["clean"])
    assert metadata["target_hashes"]["noisy"] == array_hash(fixture["noisy"])
    base.write_json(
        destination_group / "input-provenance.json",
        {
            "source_group": source_group.as_posix(),
            "fixture_npz_sha256": sha256(source_fixture),
            "fixture_json_sha256": sha256(source_metadata),
            "source_run_config_sha256": sha256(source_group / "run-config.json"),
        },
    )
    return fixture, sha256(source_fixture)


def source_config(
    source_group: Path, destination_group: Path, cfg: Config, seed: int
) -> Any:
    values = json.loads((source_group / "run-config.json").read_text())
    assert values["nx"] == 24
    assert values["ny"] == 10
    assert values["seed"] == seed
    assert values["noise_rms"] == 0.05
    assert values["weight"] == 0.01
    assert values["fiber_angle"] == 0.0
    assert values["gamma"] == 0.5
    assert values["init"] == 0.0
    assert values["forward_rtol"] == 1e-6
    assert values["forward_atol"] == 1e-11
    values.update(
        {
            "output_dir": destination_group,
            "rows": "F-S",
            "targets": "noisy",
            "steps": cfg.steps,
            "validate_gradients": False,
            "resume": False,
        }
    )
    result = base.Config(_cli_parse_args=False, **values)
    base.write_json(
        destination_group / "run-config.json", result.model_dump(mode="json")
    )
    return result


def run_group(
    physics: Physics,
    graph: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: Config,
    seed: int,
) -> dict[str, Any]:
    group = f"noise005-seed{seed}"
    source_group = cfg.source_dir / group
    destination_group = cfg.output_dir / group
    destination = destination_group / "noisy" / "F-S"
    assert not destination.exists(), (
        f"Refusing to overwrite existing result: {destination}"
    )
    assert (source_group / "noisy" / "F-MS" / "summary.json").is_file(), (
        f"Held-out source group is incomplete: {source_group}"
    )
    fixture, fixture_sha = load_fixture(source_group, destination_group, physics)
    inverse_cfg = source_config(source_group, destination_group, cfg, seed)
    selected = base.cases(inverse_cfg)
    assert len(selected) == 1
    assert selected[0] == base.Case("F-S", "F", 0.0, 0.01)
    case = base.Case(f"F-S__{group}", "F", 0.0, 0.01)

    angle = math.radians(inverse_cfg.fiber_angle)
    fibers = torch.zeros((len(physics.ids), 3))
    fibers[:, 0] = math.cos(angle)
    fibers[:, 2] = math.sin(angle)
    LOG.info("Smoothing-only held-out fit: %s", group)
    try:
        summary = base.run_case(
            physics,
            fixture["noisy"],
            fixture["clean"],
            float(fixture["D"]),
            case,
            "noisy",
            inverse_cfg,
            graph,
            fibers,
            destination,
        )
    except Exception as error:
        destination.mkdir(parents=True, exist_ok=True)
        summary = {
            "case": asdict(case),
            "target": "noisy",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        base.write_json(destination / "failure.json", summary)
        LOG.exception("Smoothing-only held-out fit failed: %s", group)
    summary["group"] = group
    summary["fixture_source"] = (source_group / "fixture.npz").as_posix()
    summary["fixture_sha256"] = fixture_sha
    return summary


def main(cfg: Config) -> None:
    configure()
    am.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(cfg.output_dir / "config.json", cfg.model_dump(mode="json"))
    jobs = [
        {
            "group": f"noise005-seed{seed}",
            "seed": seed,
            "target": "noisy",
            "case": {"name": "F-S", "mode": "F", "lm": 0.0, "ls": 0.01},
        }
        for seed in SEEDS
    ]
    base.write_json(cfg.output_dir / "jobs.json", jobs)
    copy_sources(cfg.output_dir, cfg.source_dir / "sources")

    physics = Physics(24, 10)
    graph = am.face_graph(physics.points, physics.tets, physics.ids)
    results = []
    for seed in SEEDS:
        results.append(run_group(physics, graph, cfg, seed))
        base.write_json(cfg.output_dir / "summary.json", results)
    LOG.info(
        "Smoothing-only holdout finished: %d fits, %d failures",
        len(results),
        sum("error" in result for result in results),
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
