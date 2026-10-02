# Copyright (c) 2026 liblaf
"""Supervise one collision-on Smile fit and its serial endpoint audit.

This process owns only the children it starts. The saved numerical checkpoint is
never rewritten by the supervisor, and a partial endpoint remains reviewable.
"""

from __future__ import annotations

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
RUN = GROUP / "data/inverse-smile-coupled-001"
CONTROL = GROUP / "data/remote-smile-control-001"
FIT_STOP = dt.datetime(2026, 9, 30, 5, 50, tzinfo=dt.UTC)
AUDIT_STOP = dt.datetime(2026, 9, 30, 5, 58, tzinfo=dt.UTC)
HARD_STOP = dt.datetime(2026, 9, 30, 6, 0, tzinfo=dt.UTC)
SOURCES = {
    "178-inverse-smile-rigid.py": "09c52bdd8f289678a31daf1f6d2f74c1312774e280f2b14b536bf1b8db94b448",
    "179-run-smile-collision-on.py": "a94ff8ba2ea29e6be74ec77f165fdcd24e1de1be9d26c15ba011de7703b90354",
    "180-audit-expression-coupled.py": "4196a93c501cdeb21e01c6366ba4a5f2276b701df7275e7f6b84146f1114d8d8",
}
INTERRUPTED: list[int] = []


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
            pid, uuid = [part.strip() for part in line.split(",", maxsplit=1)]
            rows.append({"pid": int(pid), "gpu_uuid": uuid})
    return rows


def await_gpu_idle(cutoff: dt.datetime) -> float:
    started = time.monotonic()
    while now() < cutoff:
        if not gpu_apps():
            return time.monotonic() - started
        time.sleep(1)
    raise TimeoutError("GPU remains occupied")  # noqa: EM101, TRY003


def process_identity(
    process: subprocess.Popen[Any], command: list[str]
) -> dict[str, Any]:
    path = Path("/proc") / str(process.pid)
    stat = (path / "stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    raw = (path / "cmdline").read_bytes()
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
    current = process_identity(process, identity["argv"])
    assert current == identity, "owned child identity changed before signal"
    process.send_signal(number)
    return True


def wait_before(process: subprocess.Popen[Any], cutoff: dt.datetime) -> int | None:
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
        row["deadline_cleanup"] = []
        for number, grace in (
            (signal.SIGINT, 30),
            (signal.SIGTERM, 30),
            (signal.SIGKILL, 15),
        ):
            if not signal_owned(process, identity, number):
                break
            row["deadline_cleanup"].append(
                {"signal": number.name, "at_utc": now().isoformat()}
            )
            try:
                return process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
    return process.wait()


def on_signal(number: int, _frame: object) -> None:
    INTERRUPTED.append(number)


def run_child(
    command: list[str], label: str, cutoff: dt.datetime, receipt: dict[str, Any]
) -> int:
    assert now() < cutoff
    assert not gpu_apps(), "another GPU computation is active"
    source = Path(command[2])
    assert sha256(source) == SOURCES[source.name]
    if label == "fit":
        assert (
            sha256(GROUP / "src/178-inverse-smile-rigid.py")
            == SOURCES["178-inverse-smile-rigid.py"]
        )
    row: dict[str, Any] = {"command": command, "started_at_utc": now().isoformat()}
    receipt[label] = row
    write_receipt(receipt)
    with (CONTROL / f"{label}.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=GROUP,
            env=os.environ.copy(),
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
            wait_before(process, cutoff)
            row["exit_code"] = finish_owned(process, identity, row)
        except BaseException:
            # Popen still owns an unreaped child even if identity capture or a
            # receipt write fails. Do not leave that numerical child orphaned.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            raise
    row["ended_at_utc"] = now().isoformat()
    gpu_cutoff = cutoff + dt.timedelta(minutes=2) if label == "fit" else HARD_STOP
    row["gpu_idle_wait_seconds"] = await_gpu_idle(gpu_cutoff)
    row["gpu_apps_after"] = []
    write_receipt(receipt)
    assert not row["gpu_apps_after"], "GPU computation remains after owned child exit"
    return row["exit_code"]


def main() -> int:  # noqa: PLR0915
    assert now() < FIT_STOP
    assert not Path("/proc/449133").exists(), "old collision-off queue still exists"
    assert not RUN.exists(), RUN
    assert not CONTROL.exists(), CONTROL
    assert not gpu_apps(), "GPU must be empty before Smile launch"
    assert os.environ.get("COMET_API_KEY"), "secure Cherries credential missing"
    assert str(ROOT / "src") in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    assert (
        Path(os.environ["TMPDIR"]).resolve()
        == (GROUP / "tmp/run-smile-001-runtime").resolve()
    )
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
    for name, digest in SOURCES.items():
        assert sha256(GROUP / "src" / name) == digest, name
    CONTROL.mkdir(parents=True)
    (CONTROL / "sources").mkdir()
    snapshots = {}
    supervisor = Path(__file__).resolve()
    shutil.copy2(supervisor, CONTROL / "sources" / supervisor.name)
    assert sha256(CONTROL / "sources" / supervisor.name) == sha256(supervisor)
    snapshots[supervisor.name] = record(CONTROL / "sources" / supervisor.name)
    for name, digest in SOURCES.items():
        source = GROUP / "src" / name
        destination = CONTROL / "sources" / name
        shutil.copy2(source, destination)
        assert sha256(destination) == digest
        snapshots[name] = record(destination)
    receipt: dict[str, Any] = {
        "schema": "collision-on-remote-smile-supervision-v1",
        "status": "ready",
        "run_dir": str(RUN.resolve()),
        "control_dir": str(CONTROL.resolve()),
        "source_snapshots": snapshots,
        "source_sha256": {**SOURCES, supervisor.name: sha256(supervisor)},
        "deadlines_utc": {
            "fit": FIT_STOP.isoformat(),
            "audit": AUDIT_STOP.isoformat(),
            "delivery": HARD_STOP.isoformat(),
        },
        "old_queue_pid_absent": True,
        "initial_gpu_apps": [],
        "inverse_converged_claim": False,
        "interrupted_signals": INTERRUPTED,
    }
    write_receipt(receipt)
    try:
        receipt["status"] = "fit_running"
        write_receipt(receipt)
        fit_command = [
            sys.executable,
            "-u",
            str(GROUP / "src/179-run-smile-collision-on.py"),
            "--output-dir",
            str(RUN),
        ]
        fit_exit = run_child(fit_command, "fit", FIT_STOP, receipt)
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
        if now() >= AUDIT_STOP or INTERRUPTED:
            receipt["status"] = "endpoint_saved_audit_window_closed"
            write_receipt(receipt)
            return 1
        assert (
            sha256(GROUP / "src/180-audit-expression-coupled.py")
            == SOURCES["180-audit-expression-coupled.py"]
        )
        receipt["status"] = "audit_running"
        write_receipt(receipt)
        audit_command = [
            sys.executable,
            "-u",
            str(GROUP / "src/180-audit-expression-coupled.py"),
            "--run-dir",
            str(RUN),
        ]
        audit_exit = run_child(audit_command, "audit", AUDIT_STOP, receipt)
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
            "audited_partial_endpoint"
            if valid and fit_exit != 0
            else "audited_endpoint"
            if valid
            else "endpoint_audit_failed"
        )
        write_receipt(receipt)
        return 0 if valid and fit_exit == 0 else 1  # noqa: TRY300
    except BaseException as error:
        receipt["status"] = "supervisor_failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_receipt(receipt)
        raise


if __name__ == "__main__":
    for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(watched, on_signal)
    raise SystemExit(main())
