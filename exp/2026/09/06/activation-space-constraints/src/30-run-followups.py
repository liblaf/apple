"""Penalty frontier and held-out checks, using the frozen matrix implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import activation_models as am
import pydantic_settings as ps
from block_physics import ROOT, Physics, configure
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "activation_matrix", Path(__file__).with_name("20-inverse-constraint-matrix.py")
)
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("30-followups", mkdir=True)
    groups: str = "frontier,noise,fiber,cap,gamma"
    steps: int = 240


def specifications(groups):
    jobs = []
    if "frontier" in groups:
        for weight in (0.001, 0.1, 1.0):
            jobs.append(
                (
                    f"strength-{weight:g}",
                    {"weight": weight, "noise_rms": 0.02},
                    ["G-MS", "F-MS"],
                    "noisy",
                )
            )
    if "noise" in groups:
        for seed in (20260917, 20260928, 20261009):
            jobs.append(
                (
                    f"noise005-seed{seed}",
                    {"seed": seed, "noise_rms": 0.05},
                    ["Raw6", "G-MS", "F-MS"],
                    "noisy",
                )
            )
        jobs.append(
            (
                "noise005-init008",
                {"seed": 20260917, "noise_rms": 0.05, "init": 0.08},
                ["Raw6", "F-MS"],
                "noisy",
            )
        )
    if "fiber" in groups:
        for angle in (10.0, 25.0):
            jobs.append(
                (f"fiber-{angle:g}deg", {"fiber_angle": angle}, ["F-MS"], "clean")
            )
    if "cap" in groups:
        for shortening in (0.10, 0.20):
            jobs.append(
                (
                    f"cap-{shortening:g}",
                    {"amax": -math.log(1 - shortening)},
                    ["F-MS"],
                    "clean",
                )
            )
    if "gamma" in groups:
        jobs.append(
            ("transverse-natural-stretch-fixed", {"gamma": 0.0}, ["F-MS"], "clean")
        )
    return jobs


def main(cfg: Config):
    configure()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    jobs = specifications(cfg.groups.split(","))
    base.write_json(out / "jobs.json", jobs)
    source_dir = out / "sources"
    source_dir.mkdir(exist_ok=True)
    hashes = {}
    for src in (
        Path(__file__),
        Path(__file__).with_name("20-inverse-constraint-matrix.py"),
        Path(__file__).with_name("block_physics.py"),
        Path(__file__).with_name("activation_models.py"),
        Path(__file__).with_name("experiment_profile.py"),
    ):
        hashes[str(src.relative_to(ROOT))] = hashlib.sha256(
            src.read_bytes()
        ).hexdigest()
        shutil.copy2(src, source_dir / src.name)
    base.write_json(out / "source-hashes.json", hashes)
    p = Physics(24, 10)
    graph = am.face_graph(p.points, p.tets, p.ids)
    results = []
    for group, overrides, rows, target_name in jobs:
        path = out / group
        path.mkdir(exist_ok=True)
        bc = base.Config(
            _cli_parse_args=False,
            output_dir=path,
            steps=cfg.steps,
            rows=",".join(rows),
            validate_gradients=False,
            **overrides,
        )
        base.write_json(path / "run-config.json", bc.model_dump(mode="json"))
        clean, scale, targets, fibers = base.prepare_targets(p, bc, path)
        angle = math.radians(bc.fiber_angle)
        fibers[:, 0] = math.cos(angle)
        fibers[:, 2] = math.sin(angle)
        for case in base.cases(bc):
            # Distinct metric keys for multiple settings in one Comet run.
            actual = base.Case(f"{case.name}__{group}", case.mode, case.lm, case.ls)
            dest = path / target_name / case.name
            LOG.info("Follow-up %s/%s", group, case.name)
            try:
                summary = base.run_case(
                    p,
                    targets[target_name],
                    clean,
                    scale,
                    actual,
                    target_name,
                    bc,
                    graph,
                    fibers,
                    dest,
                )
            except Exception as error:
                import traceback

                summary = {
                    "case": base.asdict(actual),
                    "target": target_name,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                dest.mkdir(parents=True, exist_ok=True)
                base.write_json(dest / "failure.json", summary)
                LOG.exception("Follow-up failed")
            summary["group"] = group
            results.append(summary)
            base.write_json(out / "summary.json", results)
    LOG.info(
        "Follow-ups finished: %d fits, %d failures",
        len(results),
        sum("error" in x for x in results),
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
