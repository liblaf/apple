# Copyright (c) 2026 liblaf
"""Supervise Smile008 fitting, then run one serial independent audit.

It accepts only an explicit, timezone-aware fit cutoff. The supervisor owns
only the fitter and auditor it starts, records complete identities, and only
signals those exact children. It cannot resume or alter an older run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
RUN = GROUP / "data/inverse-smile-coupled-008"
CONTROL = GROUP / "data/remote-smile-control-008"
SOURCE_NAMES = {
    "3130-inverse-smile-retries.py",
    "3140-continue-smile-retries.py",
    "expression_gradient_check.py",
    "3000-inverse-smile-expression-gradient.py",
    "3010-run-smile-expression-gradient.py",
    "180-audit-expression-coupled.py",
}
PARENT_CHECKPOINT_SHA256 = (
    "b826ced1c56942390c6a25a20772fe15b3fa75a563437dd05d4b0fc28279b61e"
)
PARENT_PROGRESS_SHA256 = (
    "2547ddcdfc5364be127cdaf2ba31024c9d0dc320575a61f5ddcca50df816d1af"
)
PARENT_AUDIT_SHA256 = "383b9e6712f051680e2b748013a46b8f7c8fe4572cf4e12a23cbe876dc66be16"
INTERRUPTED: list[int] = []


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, "--fit-stop-utc must include a UTC offset"
    return parsed.astimezone(dt.UTC)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bindings(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    assert set(value) == SOURCE_NAMES, sorted(value)
    for name, digest in value.items():
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest), name
    return value


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def write_receipt(receipt: dict[str, Any]) -> None:
    receipt["updated_at_utc"] = now().isoformat()
    temporary = CONTROL / "job.tmp.json"
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(CONTROL / "job.json")


def gpu_apps() -> list[dict[str, str | int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
        timeout=30,
    )
    rows = []
    for line in output.splitlines():
        if line.strip():
            pid, gpu_uuid = [part.strip() for part in line.split(",", maxsplit=1)]
            rows.append({"pid": int(pid), "gpu_uuid": gpu_uuid})
    return rows


def matching_python_processes(*, exclude_pid: int) -> list[dict[str, Any]]:
    """Return only live Python scripts that could contend with this Smile run."""
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == exclude_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
            stat = (proc / "stat").read_text()
        except FileNotFoundError:
            continue
        argv = [os.fsdecode(item) for item in raw.rstrip(b"\0").split(b"\0")]
        scripts = [item for item in argv if item.endswith(".py")]
        if not scripts:
            continue
        if not any(
            "smile" in Path(script).name.lower()
            or script.startswith((str(GROUP), str(ROOT)))
            for script in scripts
        ):
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        matches.append(
            {
                "pid": int(proc.name),
                "start_ticks": int(fields[19]),
                "boot_id": boot_id,
                "argv": argv,
                "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return sorted(matches, key=lambda item: item["pid"])


def await_gpu_idle(cutoff: dt.datetime) -> float:
    started = time.monotonic()
    while now() < cutoff:
        if not gpu_apps():
            return time.monotonic() - started
        time.sleep(1)
    message = "GPU remains occupied"
    raise TimeoutError(message)


def process_identity(
    process: subprocess.Popen[Any], command: list[str]
) -> dict[str, Any]:
    proc = Path("/proc") / str(process.pid)
    stat = (proc / "stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    raw = (proc / "cmdline").read_bytes()
    argv = [os.fsdecode(part) for part in raw.rstrip(b"\0").split(b"\0")]
    assert argv == command, (argv, command)
    return {
        "pid": process.pid,
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "argv": argv,
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
    }


def signal_owned(
    process: subprocess.Popen[Any], identity: dict[str, Any], number: signal.Signals
) -> bool:
    if process.poll() is not None:
        return False
    assert process_identity(process, identity["argv"]) == identity
    process.send_signal(number)
    return True


def wait_until(process: subprocess.Popen[Any], cutoff: dt.datetime) -> int | None:
    while process.poll() is None and not INTERRUPTED:
        remaining = (cutoff - now()).total_seconds()
        if remaining <= 0:
            break
        try:
            return process.wait(timeout=min(2.0, remaining))
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


def finish_owned(
    process: subprocess.Popen[Any], identity: dict[str, Any], row: dict[str, Any]
) -> int:
    if process.poll() is None:
        row["owned_cleanup"] = []
        for number, grace in (
            (signal.SIGINT, 30),
            (signal.SIGTERM, 30),
            (signal.SIGKILL, 15),
        ):
            if not signal_owned(process, identity, number):
                break
            row["owned_cleanup"].append(
                {"signal": number.name, "at_utc": now().isoformat()}
            )
            try:
                return process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
    return process.wait()


def run_child(
    command: list[str],
    *,
    label: str,
    cutoff: dt.datetime,
    receipt: dict[str, Any],
    sources: dict[str, str],
    hard_stop: dt.datetime,
) -> int:
    assert now() < cutoff
    assert not gpu_apps(), "GPU is occupied before owned child launch"
    matches = matching_python_processes(exclude_pid=os.getpid())
    assert not matches, matches
    source = Path(command[2])
    assert sha256(source) == sources[source.name]
    row: dict[str, Any] = {"command": command, "started_at_utc": now().isoformat()}
    row["matching_python_processes_before"] = matches
    receipt[label] = row
    write_receipt(receipt)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (CONTROL / f"{label}.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=GROUP,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            identity = process_identity(process, command)
            row["identity"] = identity
            row["log"] = str((CONTROL / f"{label}.log").resolve())
            write_receipt(receipt)
            wait_until(process, cutoff)
            row["exit_code"] = finish_owned(process, identity, row)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            raise
    row["ended_at_utc"] = now().isoformat()
    row["gpu_idle_wait_seconds"] = await_gpu_idle(
        cutoff + dt.timedelta(minutes=2) if label == "fit" else hard_stop
    )
    row["gpu_apps_after"] = gpu_apps()
    assert not row["gpu_apps_after"]
    write_receipt(receipt)
    return row["exit_code"]


def on_signal(number: int, _frame: object) -> None:
    INTERRUPTED.append(number)


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-stop-utc", required=True, type=parse_utc)
    parser.add_argument("--source-bindings-json", required=True, type=Path)
    args = parser.parse_args()
    fit_stop: dt.datetime = args.fit_stop_utc
    audit_stop = fit_stop + dt.timedelta(minutes=10)
    hard_stop = audit_stop + dt.timedelta(minutes=2)
    sources = read_bindings(args.source_bindings_json)
    assert now() < fit_stop
    assert not RUN.exists(), RUN
    assert not CONTROL.exists(), CONTROL
    assert not gpu_apps(), "GPU must be empty before Smile008"
    matching_before = matching_python_processes(exclude_pid=os.getpid())
    assert not matching_before, matching_before
    assert os.environ.get("COMET_API_KEY"), "secure Cherries credential missing"
    assert os.environ.get("PYTHONPATH") == str(ROOT / "src")
    assert (
        Path(os.environ["TMPDIR"]).resolve()
        == (GROUP / "tmp/run-smile-008-runtime").resolve()
    )
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
    for name, digest in sources.items():
        assert sha256(GROUP / "src" / name) == digest, name
    CONTROL.mkdir(parents=True)
    (CONTROL / "sources").mkdir()
    snapshots: dict[str, dict[str, str]] = {}
    supervisor = Path(__file__).resolve()
    shutil.copy2(supervisor, CONTROL / "sources" / supervisor.name)
    snapshots[supervisor.name] = record(CONTROL / "sources" / supervisor.name)
    shutil.copy2(args.source_bindings_json, CONTROL / "source-bindings.json")
    assert read_bindings(CONTROL / "source-bindings.json") == sources
    for name, digest in sources.items():
        source = GROUP / "src" / name
        destination = CONTROL / "sources" / name
        shutil.copy2(source, destination)
        assert sha256(destination) == digest
        snapshots[name] = record(destination)
    receipt: dict[str, Any] = {
        "schema": "collision-on-remote-smile-retry-supervision-v1",
        "status": "ready",
        "run_dir": str(RUN.resolve()),
        "control_dir": str(CONTROL.resolve()),
        "source_snapshots": snapshots,
        "source_sha256": {**sources, supervisor.name: sha256(supervisor)},
        "source_bindings": record(CONTROL / "source-bindings.json"),
        "deadlines_utc": {
            "fit": fit_stop.isoformat(),
            "audit": audit_stop.isoformat(),
            "cleanup": hard_stop.isoformat(),
        },
        "initial_gpu_apps": [],
        "matching_python_processes_before": matching_before,
        "inverse_converged_claim": False,
        "interrupted_signals": INTERRUPTED,
    }
    write_receipt(receipt)
    try:
        parent = GROUP / "data/inverse-smile-coupled-006"
        parent_audit = parent / "independent-audit.json"
        assert sha256(parent / "checkpoint.pt") == PARENT_CHECKPOINT_SHA256
        assert sha256(parent / "progress.jsonl") == PARENT_PROGRESS_SHA256
        assert sha256(parent_audit) == PARENT_AUDIT_SHA256
        audit = json.loads(parent_audit.read_text())
        assert audit["valid_forward"] is True
        assert audit["inputs"]["endpoint"]["sha256"] == sha256(parent / "endpoint.npz")
        assert audit["inputs"]["protocol"]["sha256"] == sha256(parent / "protocol.json")
        assert audit["inputs"]["summary"]["sha256"] == sha256(parent / "summary.json")
        receipt["parent_run"] = str(parent)
        receipt["parent_optimizer_steps"] = {"q": 44, "pose": 44}
        receipt["parent_independent_audit"] = record(parent_audit)
        receipt["status"] = "fit_running"
        write_receipt(receipt)
        fit = [
            sys.executable,
            "-u",
            str(GROUP / "src/3140-continue-smile-retries.py"),
            "--output-dir",
            str(RUN),
            "--initialization-checkpoint",
            str(parent / "checkpoint.pt"),
            "--parent-audit-path",
            str(parent_audit),
            "--expected-parent-audit-sha256",
            PARENT_AUDIT_SHA256,
            "--expected-parent-checkpoint-sha256",
            PARENT_CHECKPOINT_SHA256,
            "--expected-parent-progress-sha256",
            PARENT_PROGRESS_SHA256,
            "--deadline-iso-utc",
            fit_stop.isoformat(),
        ]
        fit_exit = run_child(
            fit,
            label="fit",
            cutoff=fit_stop,
            receipt=receipt,
            sources=sources,
            hard_stop=hard_stop,
        )
        receipt["status"] = "fit_exited"
        receipt["fit_exit_code"] = fit_exit
        receipt["saved"] = {
            name: record(RUN / name) if (RUN / name).is_file() else None
            for name in (
                "protocol.json",
                "summary.json",
                "checkpoint.pt",
                "endpoint.npz",
                "progress.jsonl",
            )
        }
        write_receipt(receipt)
        if not (RUN / "endpoint.npz").is_file() or not (RUN / "summary.json").is_file():
            receipt["status"] = "partial_fit_without_endpoint"
            write_receipt(receipt)
            return 1
        if now() >= audit_stop or INTERRUPTED:
            receipt["status"] = "endpoint_saved_audit_window_closed"
            write_receipt(receipt)
            return 1
        receipt["status"] = "audit_running"
        write_receipt(receipt)
        audit_command = [
            sys.executable,
            "-u",
            str(GROUP / "src/180-audit-expression-coupled.py"),
            "--run-dir",
            str(RUN),
        ]
        audit_exit = run_child(
            audit_command,
            label="audit",
            cutoff=audit_stop,
            receipt=receipt,
            sources=sources,
            hard_stop=hard_stop,
        )
        receipt["audit_exit_code"] = audit_exit
        audit_path = RUN / "independent-audit.json"
        receipt["independent_audit"] = (
            record(audit_path) if audit_path.is_file() else None
        )
        valid = (
            audit_exit == 0
            and audit_path.is_file()
            and json.loads(audit_path.read_text())["valid_forward"] is True
        )
        receipt["status"] = (
            "audited_endpoint"
            if valid and fit_exit == 0
            else "audited_partial_endpoint"
            if valid
            else "endpoint_audit_failed"
        )
        write_receipt(receipt)
        result = 0 if valid and fit_exit == 0 else 1
        return result  # noqa: TRY300
    except BaseException as error:
        receipt["status"] = "supervisor_failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_receipt(receipt)
        raise


if __name__ == "__main__":
    for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(watched, on_signal)
    raise SystemExit(main())
