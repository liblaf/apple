# ruff: noqa: E402
"""Record CPU solver contracts before replaying saved full-face states."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parents[2] / "21/joint-activation-material-mandible/src"))

from joint_common import ProfileJoint, sha256

from liblaf import cherries


class Config(cherries.BaseConfig):
    output: Path = cherries.output("cpu-validation.json", mkdir=True)


def main(cfg: Config) -> None:
    rows = []
    for name in (
        "check_accelerated_solvers.py",
        "check_vertex_blocks.py",
        "check_hybrid.py",
    ):
        source = SOURCE / name
        namespace = runpy.run_path(str(source))
        for key, value in namespace.items():
            if key.startswith("check_") and callable(value):
                started = time.perf_counter()
                value()
                rows.append(
                    {
                        "check": key,
                        "source": name,
                        "source_sha256": sha256(source),
                        "success": True,
                        "seconds": time.perf_counter() - started,
                    }
                )
    cfg.output.write_text(
        json.dumps({"success": True, "checks": rows}, indent=2) + "\n"
    )
    cherries.log_metrics({"validation/checks": len(rows), "validation/success": 1})


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
