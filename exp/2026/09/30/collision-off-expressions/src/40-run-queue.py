"""Run alternating immutable collision-off fit chunks on one GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

GROUP = Path(__file__).resolve().parent.parent


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-iterations", type=int, default=25)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    assert args.chunk_iterations > 0
    path = GROUP / "data/queue-state.json"
    assert not path.exists(), "Use a new additive queue entrypoint for continuations."
    state = {
        "schema": "collision-off-expression-queue-v1",
        "pid": os.getpid(),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "collision_enabled": False,
        "scheduling": "One numerical process at a time; alternating expression chunks.",
        "chunk_iterations": args.chunk_iterations,
        "expressions": {
            name: {"status": "queued", "chunks": [], "next_chunk": 1}
            for name in ("MouthOpen", "Smile")
        },
    }
    write(path, state)
    while True:
        runnable = [
            (name, item)
            for name, item in state["expressions"].items()
            if item["status"] in ("queued", "continue")
        ]
        if not runnable:
            break
        for name, item in runnable:
            number = item["next_chunk"]
            run = GROUP / "data" / f"{name.lower()}-{number:03d}"
            log = GROUP / "tmp" / f"{name.lower()}-{number:03d}.log"
            command = [
                args.python,
                "-u",
                str(GROUP / "src/10-fit.py"),
                "--expression-name",
                name,
                "--output-dir",
                str(run),
                "--maximum-iterations",
                str(args.chunk_iterations),
            ]
            if item["chunks"]:
                previous = Path(item["chunks"][-1]["run_dir"])
                command += [
                    "--initialization-checkpoint",
                    str(previous / "checkpoint.pt"),
                    "--continue-optimizer-state",
                    "true",
                ]
                prior_summary = json.loads((previous / "summary.json").read_text())
                alpha = min(1.0, 2 * prior_summary["final"].get("alpha", 0.5))
                command += ["--initial-trial-alpha", str(alpha)]
            env = os.environ.copy()
            env.pop("DEBUG", None)
            env.update(
                CHERRIES_NAME=f"{name} collision-off corrected neutral chunk {number:03d}",
                CHERRIES_TAGS=f"{name.lower()},collision-off,isfixed,active-strain,skin-prestrain,joint-jaw,retained-tets,convergence",
                OMP_NUM_THREADS="4",
            )
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            assert not processes, f"GPU already has compute processes: {processes}"
            item["status"] = "running"
            item["active_run_dir"] = str(run)
            with log.open("x") as stream:
                process = subprocess.Popen(
                    command, cwd=GROUP, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                item["active_pid"] = process.pid
                item["command"] = command
                write(path, state)
                gpu_log = GROUP / "tmp" / f"{name.lower()}-{number:03d}-gpu.jsonl"
                with gpu_log.open("x") as gpu_stream:
                    while True:
                        sample = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        ).strip()
                        gpu_stream.write(
                            json.dumps(
                                {
                                    "time": dt.datetime.now(dt.UTC).isoformat(),
                                    "memory_used_mib_and_utilization_percent": sample,
                                }
                            )
                            + "\n"
                        )
                        gpu_stream.flush()
                        try:
                            code = process.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            pass
            result = (
                json.loads((run / "summary.json").read_text())
                if (run / "summary.json").is_file()
                else {"status": "no_summary"}
            )
            item["chunks"].append(
                {
                    "run_dir": str(run),
                    "log": str(log),
                    "exit_code": code,
                    "status": result["status"],
                }
            )
            item["active_pid"] = None
            audit_code = None
            if (run / "endpoint.npz").is_file() and "final" in result:
                audit_log = GROUP / "tmp" / f"{name.lower()}-{number:03d}-audit.log"
                audit_env = dict(env)
                audit_env["CHERRIES_NAME"] = (
                    f"{name} collision-off independent audit chunk {number:03d}"
                )
                with audit_log.open("x") as stream:
                    audit_process = subprocess.Popen(
                        [
                            args.python,
                            "-u",
                            str(GROUP / "src/20-audit.py"),
                            "--run-dir",
                            str(run),
                        ],
                        cwd=GROUP,
                        env=audit_env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    item["status"] = "auditing"
                    item["active_pid"] = audit_process.pid
                    write(path, state)
                    audit_code = audit_process.wait()
                item["chunks"][-1]["audit_exit_code"] = audit_code
                item["active_pid"] = None
            item["next_chunk"] += 1
            if code != 0 or (audit_code is not None and audit_code != 0):
                item["status"] = "needs_diagnosis"
            elif result["status"] == "finite_budget_exhausted":
                item["status"] = "continue"
            elif result["status"] == "stationarity_candidate_requires_audit":
                item["status"] = "stationarity_candidate_requires_audit"
            else:
                item["status"] = "needs_diagnosis"
            write(path, state)
    state["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    write(path, state)


if __name__ == "__main__":
    main()
