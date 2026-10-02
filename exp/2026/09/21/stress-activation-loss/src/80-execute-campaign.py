"""Finite task supervisor: calibration, fitting, figures, and independent audit."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

GROUP = Path(__file__).parents[1]


def publish(**changes):
    path = GROUP / "site/status.json"
    state = json.loads(path.read_text())
    state.update(changes, updated_at=datetime.datetime.now().astimezone().isoformat())
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main():
    assert json.loads((GROUP / "data/12-validation/checks.json").read_text())["passed"]
    logdir = GROUP / "logs"
    logdir.mkdir(exist_ok=True)
    steps = [
        (
            "20-calibrate-smoothness.py",
            "Stress face: shared strong smoothness calibration",
            "calibrating strong smoothness",
        ),
        (
            "40-run-chains.py",
            "Stress face: eight staged activation-loss fits",
            "running staged fits",
        ),
    ]
    receipts = []
    for entry, name, phase in steps:
        publish(status="running", phase=phase)
        env = os.environ.copy()
        env.update(
            CHERRIES_NAME=name,
            CHERRIES_TAGS="face,active-stress,staged,updated-materials,no-skin,strong-smoothness",
        )
        started = datetime.datetime.now().astimezone().isoformat()
        with (logdir / (entry.replace(".py", "") + "-terminal.log")).open(
            "w"
        ) as stream:
            result = subprocess.run(
                [sys.executable, str(GROUP / "src" / entry)],
                cwd=GROUP,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        receipts.append(
            {
                "entrypoint": entry,
                "started_at": started,
                "finished_at": datetime.datetime.now().astimezone().isoformat(),
                "exit_code": result.returncode,
            }
        )
        (GROUP / "data/campaign-receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        if result.returncode != 0:
            publish(
                status="failed",
                phase=f"{phase} stopped; see recorded failure",
                failure={
                    "entrypoint": entry,
                    "exit_code": result.returncode,
                    "log": str(logdir / (entry.replace(".py", "") + "-terminal.log")),
                },
            )
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
