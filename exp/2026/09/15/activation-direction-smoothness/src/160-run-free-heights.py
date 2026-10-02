"""Fill the two missing heights using the original, hash-verified protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
SPEC = importlib.util.spec_from_file_location(
    "original_height_runner", GROUP / "src/10-run.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output: Path = Path("160-free-height-sweep")
    heights: str = "0.10,0.15"
    modes: str = "unconstrained"
    weights: str = "0"


def main(cfg: Config) -> None:
    protocol_path = GROUP / "data/tune-w0/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, (
            relative
        )
    actual = cfg.model_dump(mode="json")
    for key, value in protocol["config"].items():
        if key not in {"output", "heights", "modes"}:
            assert actual[key] == value, key
    cherries.log_input(protocol_path)
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileActivationStudy)
