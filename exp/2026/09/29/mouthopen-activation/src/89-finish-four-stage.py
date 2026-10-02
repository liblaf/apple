"""Finalize one exact four-stage MouthOpen process with finite local work."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003, TRY300, TRY301

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
DATA = GROUP / "data"
CHAIN = DATA / "70-mouthopen-four-stage"
OUT = DATA / "89-four-stage-finalization"
PYTHON = ROOT / ".venv/bin/python"
STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_status(status: dict[str, Any]) -> None:
    temporary = OUT / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT / "status.json")


def process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text()
        cmdline = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    assert len(fields) > 19
    return {
        "pid": pid,
        "state": fields[0],
        "startticks": int(fields[19]),
        "cmdline": [part.decode() for part in cmdline.rstrip(b"\0").split(b"\0")],
    }


def verify_source_manifest(path: Path) -> dict[str, Any]:
    manifest = load(path)
    if not manifest:
        raise RuntimeError(f"empty numerical source manifest: {path}")
    for name, item in manifest.items():
        snapshot = Path(item["path"])
        source = Path(item["source"])
        if sha256(snapshot) != item["sha256"]:
            raise RuntimeError(f"frozen source hash changed: {name}")
        if sha256(source) != item["sha256"]:
            raise RuntimeError(f"live numerical source changed after freeze: {name}")
    return {"manifest": record(path), "files": len(manifest)}


def verify_chain() -> dict[str, Any]:
    chain_path = CHAIN / "chain-status.json"
    chain = load(chain_path)
    if chain["status"] != "completed_attempt_budgets":
        raise RuntimeError(f"four-stage chain ended with {chain['status']!r}")
    if tuple(chain["stage_sequence"]) != STAGES:
        raise RuntimeError("four-stage sequence changed")
    if tuple(chain["completed"]) != STAGES:
        raise RuntimeError("four-stage completion list is incomplete")
    root_sources = verify_source_manifest(CHAIN / "source-manifest.json")
    stages: dict[str, Any] = {}
    parent = Path(chain["inputs"]["final.npz"]["path"])
    if sha256(parent) != chain["inputs"]["final.npz"]["sha256"]:
        raise RuntimeError("full-jaw parent checkpoint hash changed")
    for mode in STAGES:
        folder = CHAIN / mode
        summary_path = folder / "summary.json"
        summary = load(summary_path)
        if summary["status"] != "completed_attempt_budget":
            raise RuntimeError(f"{mode} ended with {summary['status']!r}")
        if summary["mode"] != mode or summary["activation_model"] != "strain":
            raise RuntimeError(f"{mode} checkpoint identity changed")
        if summary["attempted_updates"] != 200:
            raise RuntimeError(f"{mode} did not spend 200 attempted updates")
        if summary["optimizer_updates"] + summary["skipped_updates"] != 200:
            raise RuntimeError(f"{mode} update accounting does not sum to 200")
        if not summary["solver_converged"]:
            raise RuntimeError(f"{mode} terminal solver receipt is not converged")
        if summary["parent_checkpoint"]["sha256"] != sha256(parent):
            raise RuntimeError(f"{mode} parent checkpoint hash changed")
        checkpoint = folder / "last.npz"
        if summary["final_checkpoint"]["sha256"] != sha256(checkpoint):
            raise RuntimeError(f"{mode} final checkpoint hash changed")
        with np.load(checkpoint, allow_pickle=False) as state:
            if (
                str(state["mode"]) != mode
                or str(state["activation_model"]) != "strain"
                or not bool(state["solver_valid"])
                or not (0 <= int(state["step"]) <= 200)
            ):
                raise RuntimeError(f"{mode} terminal checkpoint metadata is invalid")
        stages[mode] = {
            "summary": record(summary_path),
            "checkpoint": record(checkpoint),
            "attempted_updates": 200,
            "optimizer_updates": summary["optimizer_updates"],
            "skipped_updates": summary["skipped_updates"],
            "source": verify_source_manifest(folder / "source-manifest.json"),
        }
        parent = checkpoint
    return {
        "status": chain["status"],
        "chain": record(chain_path),
        "source": root_sources,
        "stages": stages,
    }


def run_child(
    status: dict[str, Any],
    *,
    name: str,
    script: str,
    args: tuple[str, ...],
    output: Path,
    asset: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source = GROUP / "src" / script
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [str(PYTHON), f"src/{script}", *args]
    log = OUT / f"{name}-terminal.log"
    child = {
        "status": "running",
        "started_at": now(),
        "command": command,
        "cwd": str(GROUP),
        "script": record(source),
        "log": str(log),
        "output": str(output),
        "env": {
            "CHERRIES_NAME": f"MouthOpen four-stage finalization {name}",
            "CHERRIES_TAGS": f"mouthopen,four-stage,finalization,{name}",
        },
    }
    status["children"][name] = child
    status["status"] = f"running_{name}"
    write_status(status)
    env = os.environ.copy()
    env.update(child["env"])
    with log.open("x") as stream:
        code = subprocess.call(
            command, cwd=GROUP, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    child["return_code"] = code
    child["ended_at"] = now()
    child["log"] = record(log)
    child["status"] = "succeeded" if code == 0 else "failed"
    write_status(status)
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}; see {log}")
    if sha256(source) != child["script"]["sha256"]:
        raise RuntimeError(f"{name} script changed while it ran")
    product = output / asset
    if not product.is_file():
        raise FileNotFoundError(product)
    child["asset"] = record(product)
    write_status(status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-seconds", type=float, default=8 * 3600)
    parser.add_argument("--analysis-output", default="80-four-stage-analysis")
    parser.add_argument("--render-output", default="85-mouthopen-four-stage")
    parser.add_argument("--report-output", default="87-four-stage-report")
    parser.add_argument("--render-script", default="85-render-four-stage-mouthopen.py")
    parser.add_argument("--report-script", default="87-report-four-stage-mouthopen.py")
    parser.add_argument("--render-asset", default="manifest.json")
    parser.add_argument("--report-asset", default="report.json")
    options = parser.parse_args()
    if options.max_wait_seconds <= 0 or options.max_wait_seconds > 8 * 3600:
        raise ValueError("max wait must be positive and no longer than 8 hours")
    for name in (
        options.analysis_output,
        options.render_output,
        options.report_output,
    ):
        if Path(name).name != name:
            raise ValueError(f"expected a data directory name, got {name!r}")
    for script in (options.render_script, options.report_script):
        if Path(script).name != script:
            raise ValueError(f"expected an experiment source filename, got {script!r}")
    OUT.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    process_path = CHAIN / "process.json"
    expected = load(process_path)
    status: dict[str, Any] = {
        "schema": "mouthopen-four-stage-one-shot-finalization-v1",
        "status": "initializing",
        "started_at": now(),
        "pipeline_source": record(source),
        "process_receipt": record(process_path),
        "expected_process": expected,
        "max_wait_seconds": options.max_wait_seconds,
        "children": {},
    }
    write_status(status)
    try:
        pid = int(expected["pid"])
        expected_command = [
            str(PYTHON),
            "-u",
            "src/70-run-four-stage-mouthopen.py",
        ]
        if expected["cmdline"] != expected_command:
            raise RuntimeError(
                "process receipt does not identify the 70 numerical runner"
            )
        if Path(expected["script"]["path"]).resolve() != (
            GROUP / "src/70-run-four-stage-mouthopen.py"
        ):
            raise RuntimeError("process receipt points to another numerical source")
        identity = process_identity(pid)
        if identity is None or identity["state"] == "Z":
            raise RuntimeError("exact numerical process was absent at queue launch")
        if identity["startticks"] != expected["startticks"]:
            raise RuntimeError("numerical process PID startticks changed")
        if identity["cmdline"] != expected["cmdline"]:
            raise RuntimeError("numerical process command differs from process receipt")
        if sha256(Path(expected["script"]["path"])) != expected["script"]["sha256"]:
            raise RuntimeError("numerical runner source changed after launch")
        status["observed_process"] = identity
        status["status"] = "waiting_for_numerical_exit"
        write_status(status)
        deadline = time.monotonic() + options.max_wait_seconds
        while True:
            current = process_identity(pid)
            if current is None or current["state"] == "Z":
                break
            if current["startticks"] != expected["startticks"]:
                break
            if current["cmdline"] != expected["cmdline"]:
                raise RuntimeError("numerical process command changed while running")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "numerical process exceeded the eight-hour wait bound"
                )
            time.sleep(10)
        status["process_exit_observed_at"] = now()
        status["status"] = "verifying_chain"
        write_status(status)
        status["chain"] = verify_chain()
        write_status(status)
        run_child(
            status,
            name="80-analysis",
            script="80-analyze-four-stage-mouthopen.py",
            args=("--output", options.analysis_output, "--chain", str(CHAIN)),
            output=DATA / options.analysis_output,
            asset="analysis.json",
        )
        run_child(
            status,
            name="85-render",
            script=options.render_script,
            args=("--output", options.render_output, "--source", CHAIN.name),
            output=DATA / options.render_output,
            asset=options.render_asset,
        )
        run_child(
            status,
            name="87-report",
            script=options.report_script,
            args=("--output", options.report_output, "--chain", str(CHAIN)),
            output=DATA / options.report_output,
            asset=options.report_asset,
        )
        status["status"] = "completed"
        status["ended_at"] = now()
        write_status(status)
        return 0
    except Exception as error:
        status["status"] = (
            "blocked_by_numerical_chain"
            if status["status"] == "verifying_chain"
            else "failed"
        )
        status["ended_at"] = now()
        status["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_status(status)
        raise


if __name__ == "__main__":
    sys.exit(main())
