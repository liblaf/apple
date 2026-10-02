# Copyright (c) 2026 liblaf
# ruff: noqa: BLE001, C901, EM101, EM102, PLR0912, PLR0915, TRY003, TRY301
"""Reserve the GPU for one audited probe while the v1 queue waits on its child.

The queue parent is stopped by pidfd, its existing fit child is allowed to
finish, and the queue is resumed only after the probe and all GPU work end.
This one-shot controller never edits queue state or signals a numerical child.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GROUP = Path(__file__).resolve().parent.parent
STOP_SIGNALS: list[int] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Process:
    pid: int
    state: str
    ppid: int
    starttime: int


def process(pid: int) -> Process:
    data = (Path("/proc") / str(pid) / "stat").read_text()
    fields = data[data.rfind(")") + 2 :].split()
    return Process(pid, fields[0], int(fields[1]), int(fields[19]))


def command_record(pid: int) -> dict[str, Any]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    assert raw, f"PID {pid} has no live command line"
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "argv": [os.fsdecode(part) for part in raw.rstrip(b"\0").split(b"\0")],
    }


def same_process(pid: int, starttime: int) -> Process | None:
    try:
        found = process(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    return found if found.starttime == starttime else None


def descendants(ancestor: int) -> dict[int, Process]:
    table: dict[int, Process] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            item = process(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        table[item.pid] = item
    selected: dict[int, Process] = {}
    frontier = {ancestor}
    while frontier:
        children = {pid: item for pid, item in table.items() if item.ppid in frontier}
        children = {pid: item for pid, item in children.items() if pid not in selected}
        selected.update(children)
        frontier = set(children)
    return selected


def gpu_apps() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=30,
    )
    result = []
    for row in output.splitlines():
        if not row.strip():
            continue
        pid, uuid = [part.strip() for part in row.split(",", maxsplit=1)]
        result.append({"pid": int(pid), "gpu_uuid": uuid})
    return result


class PidFd:
    """Signal a verified process without a PID reuse window."""

    def __init__(self, pid: int) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        opener = libc.pidfd_open
        opener.argtypes = [ctypes.c_int, ctypes.c_uint]
        opener.restype = ctypes.c_int
        sender = libc.pidfd_send_signal
        sender.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        sender.restype = ctypes.c_int
        descriptor = opener(pid, 0)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), f"pidfd_open({pid})")
        self.descriptor = descriptor
        self.libc = libc
        self.sender = sender

    def send(self, number: signal.Signals) -> None:
        if self.sender(self.descriptor, int(number), None, 0) < 0:
            error = ctypes.get_errno()
            raise OSError(
                error, os.strerror(error), f"pidfd_send_signal({number.name})"
            )

    def close(self) -> None:
        os.close(self.descriptor)


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--queue-starttime", type=int, required=True)
    parser.add_argument("--queue-cmdline-sha256", required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--child-starttime", type=int, required=True)
    parser.add_argument("--child-cmdline-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--child-run-dir", type=Path, required=True)
    parser.add_argument("--audited-run-dir", type=Path, required=True)
    parser.add_argument("--probe-output-dir", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument(
        "--queue-state", type=Path, default=GROUP / "data/queue-state.json"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--child-wait-seconds", type=float, default=7200)
    parser.add_argument("--probe-wait-seconds", type=float, default=5400)
    parser.add_argument("--recovery-wait-seconds", type=float, default=600)
    parser.add_argument("--queue-progress-seconds", type=float, default=1800)
    args = parser.parse_args()
    for key in (
        "queue_pid",
        "queue_starttime",
        "child_pid",
        "child_starttime",
        "child_wait_seconds",
        "probe_wait_seconds",
        "recovery_wait_seconds",
        "queue_progress_seconds",
    ):
        assert getattr(args, key) > 0, key
    assert args.queue_pid != args.child_pid
    for key in ("queue_cmdline_sha256", "child_cmdline_sha256", "probe_sha256"):
        value = getattr(args, key)
        assert len(value) == 64, key
        assert all(char in "0123456789abcdef" for char in value), key
    return args


def load_queue(args: argparse.Namespace) -> tuple[dict, dict]:
    state = json.loads(args.queue_state.read_text())
    assert state["schema"] == "collision-off-expression-queue-v1"
    assert state["pid"] == args.queue_pid
    matching = [
        item
        for item in state["expressions"].values()
        if item.get("active_pid") == args.child_pid
        and item.get("active_run_dir") == str(args.child_run_dir)
    ]
    assert len(matching) == 1, "queue state has no unique expected active child"
    item = matching[0]
    assert item["status"] == "running"
    return state, item


def verify_live(args: argparse.Namespace) -> tuple[dict, Process, Process]:
    state, item = load_queue(args)
    queue = process(args.queue_pid)
    child = process(args.child_pid)
    assert queue.starttime == args.queue_starttime
    assert queue.state not in ("Z", "X", "T", "t")
    assert child.starttime == args.child_starttime
    assert child.state not in ("Z", "X", "T", "t")
    assert child.ppid == queue.pid
    queue_command = command_record(queue.pid)
    child_command = command_record(child.pid)
    assert queue_command["sha256"] == args.queue_cmdline_sha256
    assert child_command["sha256"] == args.child_cmdline_sha256
    assert item["command"] == child_command["argv"]
    output_position = child_command["argv"].index("--output-dir")
    assert child_command["argv"][output_position + 1] == str(args.child_run_dir)
    assert gpu_apps() == [{"pid": child.pid, "gpu_uuid": args.gpu_uuid}]
    return state, queue, child


def verify_child_command(args: argparse.Namespace) -> None:
    command = command_record(args.child_pid)
    assert command["sha256"] == args.child_cmdline_sha256
    output_position = command["argv"].index("--output-dir")
    assert command["argv"][output_position + 1] == str(args.child_run_dir)


def audited_inputs(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    run = args.audited_run_dir
    paths = {
        name: run / name
        for name in (
            "protocol.json",
            "summary.json",
            "endpoint.npz",
            "checkpoint.pt",
            "independent-audit.json",
        )
    }
    records = {name: file_record(path) for name, path in paths.items()}
    protocol = json.loads(paths["protocol.json"].read_text())
    summary = json.loads(paths["summary.json"].read_text())
    audit = json.loads(paths["independent-audit.json"].read_text())
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["collision_enabled"] is False
    assert audit["schema"] == "collision-off-expression-independent-audit-v1"
    assert audit["valid_forward"] is True
    endpoint_hash = records["endpoint.npz"]["sha256"]
    assert summary["endpoint"]["sha256"] == endpoint_hash
    assert audit["inputs"]["endpoint"]["sha256"] == endpoint_hash
    return records


def append_phase(receipt_path: Path, receipt: dict, name: str, **evidence: Any) -> None:
    receipt["phases"].append({"name": name, "time": timestamp(), **evidence})
    receipt["status"] = name
    receipt["signals_received"] = list(STOP_SIGNALS)
    save_json(receipt_path, receipt)


def signaled(number: int, _frame: Any) -> None:
    STOP_SIGNALS.append(number)


def live_tracked(
    tracked: dict[int, int], *, except_pid: int | None = None
) -> list[dict]:
    live = []
    for pid, starttime in tracked.items():
        if pid == except_pid:
            continue
        item = same_process(pid, starttime)
        if item is not None and item.state not in ("Z", "X"):
            live.append(vars(item))
    return live


def tracked_descendants(parent: int, tracked: dict[int, int]) -> list[dict]:
    found = descendants(parent)
    for pid, item in found.items():
        tracked[pid] = item.starttime
    return live_tracked(tracked)


def queue_stopped(args: argparse.Namespace) -> bool:
    item = same_process(args.queue_pid, args.queue_starttime)
    return item is not None and item.state in ("T", "t")


def wait_for_child(
    args: argparse.Namespace,
    receipt_path: Path,
    receipt: dict,
    tracked: dict[int, int],
) -> None:
    deadline = time.monotonic() + args.child_wait_seconds
    while True:
        assert queue_stopped(args), "queue parent is no longer stopped"
        child = same_process(args.child_pid, args.child_starttime)
        assert child is not None, "original child disappeared before queue reaped it"
        assert child.ppid == args.queue_pid
        if child.state not in ("Z", "X"):
            verify_child_command(args)
        live = tracked_descendants(args.queue_pid, tracked)
        apps = gpu_apps()
        if child.state == "Z" and not live and not apps:
            append_phase(
                receipt_path,
                receipt,
                "child_finished_gpu_idle",
                child=vars(child),
                tracked_descendant_starttimes=dict(tracked),
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"child wait exceeded {args.child_wait_seconds} s")
        time.sleep(5)


def safe_to_resume(
    args: argparse.Namespace,
    probe: subprocess.Popen | None,
    tracked: dict[int, int],
) -> tuple[bool, dict]:
    parent = same_process(args.queue_pid, args.queue_starttime)
    child = same_process(args.child_pid, args.child_starttime)
    live = tracked_descendants(args.queue_pid, tracked)
    apps = gpu_apps()
    probe_alive = probe is not None and probe.poll() is None
    safe = (
        parent is not None
        and parent.state in ("T", "t")
        and (child is None or child.state in ("Z", "X"))
        and not live
        and not apps
        and not probe_alive
    )
    return safe, {
        "queue": None if parent is None else vars(parent),
        "original_child": None if child is None else vars(child),
        "live_descendants": live,
        "gpu_compute_apps": apps,
        "probe_alive": probe_alive,
    }


def wait_safe_recovery(
    args: argparse.Namespace,
    probe: subprocess.Popen | None,
    tracked: dict[int, int],
) -> tuple[bool, dict]:
    deadline = time.monotonic() + args.recovery_wait_seconds
    while True:
        safe, evidence = safe_to_resume(args, probe, tracked)
        if safe or time.monotonic() >= deadline:
            return safe, evidence
        time.sleep(5)


def wait_queue_progress(args: argparse.Namespace, prior_chunk_count: int) -> dict:
    deadline = time.monotonic() + args.queue_progress_seconds
    while True:
        state = json.loads(args.queue_state.read_text())
        assert state["schema"] == "collision-off-expression-queue-v1"
        assert state["pid"] == args.queue_pid
        for item in state["expressions"].values():
            chunks = item["chunks"]
            if len(chunks) <= prior_chunk_count:
                continue
            for chunk in chunks:
                if (
                    chunk["run_dir"] == str(args.child_run_dir)
                    and "audit_exit_code" in chunk
                ):
                    return {
                        "queue_state": file_record(args.queue_state),
                        "chunk": chunk,
                    }
                if (
                    chunk["run_dir"] == str(args.child_run_dir)
                    and item["status"] == "needs_diagnosis"
                ):
                    raise RuntimeError("queue reaped original child without an audit")
        if time.monotonic() >= deadline:
            raise TimeoutError("queue did not record and audit the original child")
        time.sleep(5)


def main() -> None:
    args = args_from_cli()
    args.child_run_dir = args.child_run_dir.resolve()
    args.audited_run_dir = args.audited_run_dir.resolve()
    args.probe_output_dir = args.probe_output_dir.resolve()
    args.receipt_dir = args.receipt_dir.resolve()
    args.queue_state = args.queue_state.resolve()
    probe_source = args.probe_source.resolve()
    assert probe_source.parent == GROUP / "src"
    assert probe_source.suffix == ".py"
    assert args.receipt_dir.is_relative_to(GROUP / "data")
    assert args.probe_output_dir.is_relative_to(GROUP / "data")
    assert args.receipt_dir != args.probe_output_dir
    assert not args.receipt_dir.exists()
    assert not args.probe_output_dir.exists()
    assert sha256(probe_source) == args.probe_sha256
    assert args.python.is_file()
    args.receipt_dir.mkdir(parents=True)
    receipt_path = args.receipt_dir / "reservation.json"
    sources = {}
    for name in (
        Path(__file__).name,
        probe_source.name,
        "40-run-queue.py",
        "10-fit.py",
        "20-audit.py",
    ):
        origin = GROUP / "src" / name
        destination = args.receipt_dir / "sources" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(origin, destination)
        sources[name] = {
            "original": file_record(origin),
            "snapshot": file_record(destination),
        }
        assert (
            sources[name]["original"]["sha256"] == sources[name]["snapshot"]["sha256"]
        )
    receipt: dict[str, Any] = {
        "schema": "collision-off-probe-queue-reservation-v1",
        "status": "preflight",
        "phases": [],
        "signals_received": [],
        "controller_pid": os.getpid(),
        "probe_run_dir": str(args.probe_output_dir),
        "probe_pid": None,
        "probe_exit_code": None,
        "expected": {
            "queue_pid": args.queue_pid,
            "queue_starttime": args.queue_starttime,
            "queue_cmdline_sha256": args.queue_cmdline_sha256,
            "child_pid": args.child_pid,
            "child_starttime": args.child_starttime,
            "child_cmdline_sha256": args.child_cmdline_sha256,
            "gpu_uuid": args.gpu_uuid,
            "child_run_dir": str(args.child_run_dir),
            "audited_run_dir": str(args.audited_run_dir),
            "probe_output_dir": str(args.probe_output_dir),
            "probe_source": str(probe_source),
            "probe_sha256": args.probe_sha256,
        },
        "sources": sources,
        "inputs": {},
        "recovery": "If left stopped, verify queue PID/starttime and child/probe/GPU completion before sending SIGCONT to the queue parent only.",
    }
    append_phase(receipt_path, receipt, "receipt_created")
    queue_fd: PidFd | None = None
    stopped = False
    resumed = False
    probe: subprocess.Popen | None = None
    tracked: dict[int, int] = {}
    prior_chunk_count = 0
    try:
        receipt["inputs"] = audited_inputs(args)
        queue_state, queue, child = verify_live(args)
        expression = next(
            item
            for item in queue_state["expressions"].values()
            if item.get("active_pid") == child.pid
        )
        prior_chunk_count = len(expression["chunks"])
        receipt["queue_state_initial"] = {
            "file": file_record(args.queue_state),
            "state": queue_state,
        }
        receipt["queue_initial"] = vars(queue)
        receipt["child_initial"] = vars(child)
        receipt["queue_command"] = command_record(queue.pid)
        receipt["child_command"] = command_record(child.pid)
        append_phase(receipt_path, receipt, "identity_verified")
        queue_fd = PidFd(args.queue_pid)
        # Reverify after pidfd acquisition. No numeric-PID signal fallback exists.
        verify_live(args)
        if STOP_SIGNALS:
            raise InterruptedError(
                "controller received a stop request before reservation"
            )
        queue_fd.send(signal.SIGSTOP)
        stopped = True
        append_phase(receipt_path, receipt, "queue_stop_sent")
        stop_deadline = time.monotonic() + 10
        while not queue_stopped(args):
            if time.monotonic() >= stop_deadline:
                raise TimeoutError("queue did not enter stopped state")
            time.sleep(0.1)
        frozen_state, frozen_item = load_queue(args)
        frozen_child = same_process(args.child_pid, args.child_starttime)
        assert frozen_child is not None
        assert frozen_child.ppid == args.queue_pid
        assert frozen_state["expressions"]
        assert len(frozen_item["chunks"]) == prior_chunk_count
        assert frozen_child.state != "X"
        assert command_record(args.queue_pid)["sha256"] == args.queue_cmdline_sha256
        if frozen_child.state != "Z":
            verify_child_command(args)
        receipt["queue_state_frozen"] = {
            "file": file_record(args.queue_state),
            "state": frozen_state,
        }
        append_phase(receipt_path, receipt, "queue_frozen", child=vars(frozen_child))
        wait_for_child(args, receipt_path, receipt, tracked)
        if STOP_SIGNALS:
            raise InterruptedError(
                "controller received a stop request before probe launch"
            )
        # A stopped parent cannot start its audit or another fit. Revalidate
        # the old queue state and GPU immediately before the one probe launch.
        load_queue(args)
        assert queue_stopped(args)
        assert not gpu_apps()
        assert not args.probe_output_dir.exists()
        assert sha256(probe_source) == args.probe_sha256
        assert audited_inputs(args) == receipt["inputs"]
        append_phase(receipt_path, receipt, "probe_inputs_revalidated")
        if STOP_SIGNALS:
            raise InterruptedError(
                "controller received a stop request before probe launch"
            )
        command = [
            str(args.python),
            "-u",
            str(probe_source),
            "--run-dir",
            str(args.audited_run_dir),
            "--output-dir",
            str(args.probe_output_dir),
        ]
        log = args.receipt_dir / "probe.log"
        with log.open("x") as stream:
            probe = subprocess.Popen(
                command,
                cwd=GROUP,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["probe_pid"] = probe.pid
            append_phase(
                receipt_path,
                receipt,
                "probe_started",
                command=command,
                pid=probe.pid,
                log=str(log),
            )
            deadline = time.monotonic() + args.probe_wait_seconds
            while probe.poll() is None:
                probe_descendants = descendants(probe.pid)
                for pid, item in probe_descendants.items():
                    tracked[pid] = item.starttime
                permitted_gpu_pids = {probe.pid, *probe_descendants}
                apps = gpu_apps()
                monitor = receipt.setdefault(
                    "probe_gpu_monitor",
                    {"samples": 0, "maximum_compute_processes": 0, "last_apps": []},
                )
                monitor["samples"] += 1
                monitor["maximum_compute_processes"] = max(
                    monitor["maximum_compute_processes"], len(apps)
                )
                monitor["last_apps"] = apps
                assert all(
                    app["pid"] in permitted_gpu_pids
                    and app["gpu_uuid"] == args.gpu_uuid
                    for app in apps
                ), f"foreign GPU computation during diagnostic: {apps}"
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"probe wait exceeded {args.probe_wait_seconds} s"
                    )
                time.sleep(5)
        receipt["probe_exit_code"] = probe.returncode
        receipt["post_probe_inputs"] = audited_inputs(args)
        assert receipt["post_probe_inputs"] == receipt["inputs"]
        summary_path = args.probe_output_dir / "summary.json"
        receipt["probe_summary"] = file_record(summary_path)
        receipt["probe_summary_status"] = json.loads(summary_path.read_text())["status"]
        append_phase(receipt_path, receipt, "probe_exited", exit_code=probe.returncode)
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        append_phase(receipt_path, receipt, "controller_exception")
    finally:
        if stopped and not resumed:
            try:
                safe, evidence = wait_safe_recovery(args, probe, tracked)
                receipt["recovery_evidence"] = evidence
                if safe:
                    assert queue_fd is not None
                    assert queue_stopped(args)
                    assert not gpu_apps()
                    assert probe is None or probe.poll() is not None
                    queue_fd.send(signal.SIGCONT)
                    resumed = True
                    append_phase(receipt_path, receipt, "queue_resumed")
                else:
                    append_phase(receipt_path, receipt, "manual_recovery_required")
            except Exception as error:
                receipt["recovery_failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                append_phase(receipt_path, receipt, "manual_recovery_required")
        if queue_fd is not None:
            queue_fd.close()
    if resumed:
        try:
            progress = wait_queue_progress(args, prior_chunk_count)
            receipt["queue_progress"] = progress
            append_phase(receipt_path, receipt, "queue_reaped_and_audited")
        except Exception as error:
            receipt["progress_failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            append_phase(receipt_path, receipt, "queue_progress_unverified")
    if resumed and not any(
        receipt.get(name)
        for name in ("failure", "recovery_failure", "progress_failure")
    ):
        final_phase = (
            "completed"
            if receipt.get("probe_exit_code") == 0
            else "probe_failed_queue_recovered"
        )
        append_phase(receipt_path, receipt, final_phase)
    if (
        receipt.get("failure")
        or receipt.get("recovery_failure")
        or receipt.get("progress_failure")
    ):
        raise SystemExit(1)
    if not resumed or receipt.get("probe_exit_code") != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    for watched in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(watched, signaled)
    main()
