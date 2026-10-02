"""Finish only missing normal fits and reuse verified historical L2 controls."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import calibrated_study as ns
from experiment import Profile

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "calibrated_runner", Path(__file__).with_name("10-run.py")
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Config(cherries.BaseConfig):
    pass


def main(cfg: Config) -> None:
    output = cherries.output("10-comparison")
    baseline = ns.ROOT / "exp/2026/09/21/normal-matching-scratch/data/10-comparison"
    old_protocol = json.loads((baseline / "protocol.json").read_text())
    for source in ns.ns.numerical_sources():
        relative = str(source.relative_to(ns.ROOT))
        assert old_protocol["source_sha256"][relative] == runner.sha(source)
    protocol_path = output / "protocol.json"
    assert not (output / "protocol-as-started.json").exists()
    shutil.copy2(protocol_path, output / "protocol-as-started.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["initialization"] = (
        "All conditions originate from q=u=m=v=0, B=I; reuse historical L2 arrays for four remaining controls."
    )
    protocol["execution_revision"] = (
        "User clarified only eight new normal fits needed; stopped remaining duplicate L2 work. Four duplicate L2 fits had completed."
    )
    provenance = {}
    summaries = json.loads((output / "summary.json").read_text())
    completed = {item["name"] for item in summaries}
    for mode in ns.MODES:
        for variant, _, _, _ in ns.VARIANTS:
            name = f"{mode}/{variant}"
            if name in completed:
                provenance[name] = {
                    "origin": "fresh normal fit"
                    if variant.endswith("normal")
                    else "completed L2 reproduction"
                }
                continue
            if variant.endswith("normal"):
                continue
            destination = output / mode / variant
            if destination.exists():
                interrupted = cherries.output("09-interrupted-duplicate-l2")
                assert not interrupted.exists()
                shutil.move(destination, interrupted)
            source = baseline / mode / variant
            cherries.log_input(source / "summary.json")
            shutil.copytree(source, destination)
            summary = json.loads((destination / "summary.json").read_text())
            summaries.append(summary)
            provenance[name] = {
                "origin": "reused historical L2",
                "source": str(source),
                "file_sha256": {
                    p.name: runner.sha(p) for p in source.iterdir() if p.is_file()
                },
            }
    relative = Path(__file__).relative_to(ns.ROOT)
    snapshot = output / "source" / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    protocol["source_sha256"][str(relative)] = runner.sha(Path(__file__))
    protocol["execution_provenance"] = provenance
    runner.write_json(protocol_path, protocol)
    runner.write_json(output / "summary.json", summaries)
    mesh = ns.ph.build_mesh(100, 10)
    scales = ns.ns.normalization(mesh, ns.HEIGHT)
    config = runner.Config()
    for mode in ns.MODES:
        (output / mode).mkdir(exist_ok=True)
        for variant, smooth, kind, coefficient in ns.VARIANTS:
            name = f"{mode}/{variant}"
            if not variant.endswith("normal") or name in completed:
                continue
            result = runner.run_case(
                config,
                mesh,
                mode,
                ns.HEIGHT,
                variant,
                kind,
                coefficient,
                scales,
                smooth,
                output / mode / variant,
            )
            summaries.append(result)
            provenance[name] = {"origin": "fresh normal fit after execution revision"}
            runner.write_json(output / "summary.json", summaries)
            runner.write_json(protocol_path, protocol)
    assert len(summaries) == 16


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
