"""Matched pose-jump test with a harmonic extension outside the exact carry band."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "pose_jump_test", HERE / "112-test-pose-jump.py"
)
assert spec is not None
assert spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = HERE.parent / "data/pose-jump-002"


def main(cfg: Config) -> None:
    from mouthopen_harmonic_carry import carry_near_mandible

    try:
        with patch("mouthopen_pose_jump.carry_near_mandible", carry_near_mandible):
            runner.main(cfg)
    finally:
        path = cfg.output_dir / "protocol.json"
        if path.exists():
            protocol = json.loads(path.read_text())
            protocol["carry_variant"] = (
                "exact rigid carry in d_hat band; graph-harmonic displacement increment outside the band with exact new fixed values"
            )
            for name in (Path(__file__).name, "mouthopen_harmonic_carry.py"):
                protocol["sources"][name] = runner.fit.record(HERE / name)
            archive = cfg.output_dir / "sources"
            archive.mkdir(exist_ok=True)
            for name, source in protocol["sources"].items():
                shutil.copy2(source["path"], archive / name)
                protocol["sources"][name] = runner.fit.record(archive / name)
            runner.write_json(path, protocol)
            summary_path = cfg.output_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                summary["protocol"] = protocol
                runner.write_json(summary_path, summary)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
