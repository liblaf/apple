"""One-shot local finalization after an exact MouthOpen fit process exits."""

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

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
DATA = GROUP / "data"
OUT = DATA / "69-finalization"
PYTHON = ROOT / ".venv/bin/python"


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(receipt: dict[str, Any]) -> None:
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / "status.json")


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
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def verify_fit(fit_name: str) -> dict[str, Any]:
    fit = DATA / fit_name
    summary_path = fit / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed_attempt_budget":
        raise RuntimeError(f"fit ended with status {summary['status']!r}")
    if int(summary["attempted_updates"]) != 200:
        raise RuntimeError("fit did not attempt all 200 updates")
    if int(summary["optimizer_updates"]) + int(summary["skipped_updates"]) != 200:
        raise RuntimeError("fit update accounting does not sum to 200")
    checkpoint = fit / "last.npz"
    balance = fit / "gradient-balance.json"
    components = fit / "gradient-components.npz"
    for path in (checkpoint, balance, components):
        if not path.is_file():
            raise FileNotFoundError(path)
    if summary["final_checkpoint"]["sha256"] != sha256(checkpoint):
        raise RuntimeError("fit final checkpoint hash differs from terminal summary")
    gradient = json.loads(balance.read_text())
    if gradient["checkpoint"]["sha256"] != sha256(checkpoint):
        raise RuntimeError("fit gradient balance references another checkpoint")
    if gradient["components"]["sha256"] != sha256(components):
        raise RuntimeError("fit gradient components hash differs from balance")
    return {
        "status": summary["status"],
        "attempted_updates": summary["attempted_updates"],
        "optimizer_updates": summary["optimizer_updates"],
        "skipped_updates": summary["skipped_updates"],
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "gradient_balance": {"path": str(balance), "sha256": sha256(balance)},
        "gradient_components": {
            "path": str(components),
            "sha256": sha256(components),
        },
    }


def run_stage(
    receipt: dict[str, Any],
    *,
    name: str,
    script: str,
    args: tuple[str, ...],
    output: Path,
    asset: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stage output {output}")
    command = [str(PYTHON), f"src/{script}", *args]
    log = OUT / f"{name}-terminal.log"
    stage = {
        "status": "running",
        "started_at": now(),
        "command": command,
        "script_sha256": sha256(GROUP / "src" / script),
        "cwd": str(GROUP),
        "log": str(log),
        "output": str(output),
        "env": {
            "CHERRIES_NAME": f"MouthOpen fitted finalization {name}",
            "CHERRIES_TAGS": f"mouthopen,fit,finalization,{name}",
        },
    }
    receipt["stages"][name] = stage
    receipt["status"] = f"running_{name}"
    write_status(receipt)
    env = os.environ.copy()
    env.update(stage["env"])
    with log.open("x") as stream:
        code = subprocess.call(
            command, cwd=GROUP, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    stage["return_code"] = code
    stage["ended_at"] = now()
    stage["log_sha256"] = sha256(log)
    stage["status"] = "succeeded" if code == 0 else "failed"
    write_status(receipt)
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}; see {log}")
    product = output / asset
    if not product.is_file():
        raise FileNotFoundError(product)
    payload = json.loads(product.read_text())
    if name == "60-analysis" and payload.get("fit") is None:
        raise RuntimeError("60 analysis succeeded without a fitted-state audit")
    if name == "61-render" and (
        payload.get("fit_status") != "completed_attempt_budget"
        or payload.get("fitted_activation_field") is None
    ):
        raise RuntimeError("61 render succeeded without the terminal fitted state")
    stage["asset"] = {"path": str(product), "sha256": sha256(product)}
    write_status(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--startticks", type=int, required=True)
    parser.add_argument("--cmdline-sha256", required=True)
    parser.add_argument("--fit-name", default="56-mouthopen-fit-reuse")
    parser.add_argument("--process-script", default="src/56-fit-mouthopen-reuse.py")
    parser.add_argument("--analysis-output", default="60-mouthopen-fit-analysis")
    parser.add_argument("--render-output", default="61-mouthopen-fit")
    parser.add_argument("--max-wait-seconds", type=float, default=14400)
    options = parser.parse_args()
    for name in (options.fit_name, options.analysis_output, options.render_output):
        if Path(name).name != name:
            raise ValueError(f"expected a data directory name, got {name!r}")
    if not options.process_script.startswith("src/"):
        raise ValueError("process script must be relative to experiment src")
    expected_cmdline = (str(PYTHON), "-u", options.process_script)
    OUT.mkdir(parents=True, exist_ok=False)
    identity = process_identity(options.pid)
    receipt: dict[str, Any] = {
        "schema": "mouthopen-one-shot-finalization-v1",
        "status": "initializing",
        "started_at": now(),
        "pipeline_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "process_identity": identity,
        "expected_process": {
            "pid": options.pid,
            "startticks": options.startticks,
            "cmdline": expected_cmdline,
            "cmdline_sha256": options.cmdline_sha256,
        },
        "fit_name": options.fit_name,
        "analysis_output": options.analysis_output,
        "render_output": options.render_output,
        "max_wait_seconds": options.max_wait_seconds,
        "stages": {},
    }
    write_status(receipt)
    try:
        if identity is None:
            raise RuntimeError("exact fit process was absent at queue launch")
        if identity["startticks"] != options.startticks:
            raise RuntimeError("fit PID startticks changed before queue launch")
        if identity["cmdline_sha256"] != options.cmdline_sha256:
            raise RuntimeError("fit PID cmdline changed before queue launch")
        if tuple(identity["cmdline"]) != expected_cmdline:
            raise RuntimeError("fit PID command is not the expected numerical run")
        receipt["status"] = "waiting_for_fit_exit"
        write_status(receipt)
        deadline = time.monotonic() + options.max_wait_seconds
        while True:
            current = process_identity(options.pid)
            if current is None or current["state"] == "Z":
                break
            if current["startticks"] != options.startticks:
                break  # the exact fit process exited and its PID was reused
            if current["cmdline_sha256"] != options.cmdline_sha256:
                raise RuntimeError("fit process identity changed while running")
            if time.monotonic() >= deadline:
                raise TimeoutError("fit did not exit before the finite wait deadline")
            time.sleep(10)
        receipt["process_exit_observed_at"] = now()
        receipt["status"] = "verifying_fit"
        write_status(receipt)
        receipt["fit"] = verify_fit(options.fit_name)
        write_status(receipt)
        run_stage(
            receipt,
            name="60-analysis",
            script="60-analyze-mouthopen-trial.py",
            args=(
                "--output",
                options.analysis_output,
                "--fit",
                f"data/{options.fit_name}",
            ),
            output=DATA / options.analysis_output,
            asset="analysis.json",
        )
        run_stage(
            receipt,
            name="61-render",
            script="61-render-mouthopen-trial.py",
            args=("--fit", options.fit_name, "--output", options.render_output),
            output=DATA / options.render_output,
            asset="manifest.json",
        )
        receipt["status"] = "completed"
        receipt["ended_at"] = now()
        write_status(receipt)
        return 0
    except Exception as error:
        receipt["status"] = "failed"
        receipt["ended_at"] = now()
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_status(receipt)
        raise


if __name__ == "__main__":
    sys.exit(main())
