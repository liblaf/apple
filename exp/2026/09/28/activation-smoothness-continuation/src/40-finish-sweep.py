"""Finish this already-running finite sweep and generate its review artifacts."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]


def status_file(data: dict) -> None:
    path = GROUP / "data/10-sweep/finalization-status.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    started = dt.datetime.now(dt.UTC).isoformat()
    state = {"started_at": started, "status": "waiting_for_runs"}
    status_file(state)
    try:
        while True:
            completed = [
                multiplier
                for multiplier in (1, 3, 10)
                if (
                    GROUP / f"data/10-sweep/multiplier-{multiplier}/completion.json"
                ).is_file()
            ]
            state.update(
                completed=completed, checked_at=dt.datetime.now(dt.UTC).isoformat()
            )
            status_file(state)
            if len(completed) == 3:
                break
            jobs = json.loads((GROUP / "data/10-sweep/jobs.json").read_text())
            for multiplier in {1, 3, 10} - set(completed):
                job = jobs[str(multiplier)]
                stat = Path(f"/proc/{job['pid']}/stat").read_text()
                fields = stat.split(") ", 1)[1].split()
                assert fields[19] == job["start_ticks"], "training PID was reused"
                assert fields[0] != "Z", (
                    f"multiplier {multiplier} exited without completion"
                )
            time.sleep(30)
        for script, name, tags in (
            (
                "20-analyze-sweep.py",
                "Released-axis smoothness sweep, final analysis",
                "smile,smoothness,analysis,matched",
            ),
            (
                "30-render-comparison.py",
                "Released-axis smoothness sweep, comparison figure",
                "smile,smoothness,visualization,matched",
            ),
        ):
            state["status"] = f"running_{script}"
            status_file(state)
            env = dict(os.environ)
            env.update(
                CHERRIES_NAME=name,
                CHERRIES_TAGS=tags,
                LIBGL_ALWAYS_SOFTWARE="1",
                CUDA_VISIBLE_DEVICES="",
                PYTHONUNBUFFERED="1",
            )
            with (GROUP / "tmp" / f"{Path(script).stem}.stdout.log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(GROUP / "src" / script)],
                    cwd=GROUP,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        state.update(
            status="completed", completed_at=dt.datetime.now(dt.UTC).isoformat()
        )
        status_file(state)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        status_file(state)
        raise


if __name__ == "__main__":
    main()
