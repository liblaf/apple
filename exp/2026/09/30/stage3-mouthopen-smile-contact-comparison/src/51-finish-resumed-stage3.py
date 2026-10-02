# Copyright (c) 2026 liblaf
"""Run the finite Stage 3 finalizer with explicitly registered resumed processes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    assert registry["schema"] == "stage3-finalizer-process-registry-v1"
    entries = registry["branches"]
    assert len(entries) == 2
    assert {entry["collision_enabled"] for entry in entries} == {False, True}
    assert len({entry["name"] for entry in entries}) == 2
    finalizer_path = GROUP / "src/50-finish-stage3.py"
    spec = importlib.util.spec_from_file_location(
        "stage3_registered_finalizer", finalizer_path
    )
    assert spec is not None
    assert spec.loader is not None
    finalizer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = finalizer
    spec.loader.exec_module(finalizer)
    finalizer.BRANCHES = tuple(
        finalizer.Branch(
            name=entry["name"],
            pid=entry["pid"],
            start_ticks=entry["proc_start_ticks"],
            collision_enabled=entry["collision_enabled"],
        )
        for entry in entries
    )
    for branch in finalizer.BRANCHES:
        assert branch.name.startswith("20-collision-")
        finalizer.process_identity(branch)
    finalizer.FINISH = Path(registry["finalization_directory"])
    assert finalizer.FINISH.is_absolute()
    assert finalizer.FINISH.parent == GROUP / "data"
    finalizer.main()


if __name__ == "__main__":
    main()
