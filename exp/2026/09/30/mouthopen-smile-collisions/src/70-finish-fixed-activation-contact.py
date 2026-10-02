# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM102, TRY003
"""Wait for the recorded fixed-contact solver, then run finite postprocessing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

TERMINAL = {"completed", "blocked", "failed", "interrupted"}


class Config(cherries.BaseConfig):
    run: Path = GROUP / "data/52-fixed-activation-contact"
    audit_output: Path = GROUP / "data/54-fixed-contact-audit"
    completed_render_output: Path = GROUP / "data/60-fixed-activation-contact-render"
    diagnostic_render_output: Path = GROUP / "data/62-fixed-contact-terminal-preview"
    poll_seconds: float = 10.0
    max_wait_seconds: float = 3900.0


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def proc_identity(pid: int) -> tuple[str, int] | None:
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return None
    fields = path.read_text().split()
    assert len(fields) > 21
    return fields[2], int(fields[21])


def stage(status_path: Path, status: dict[str, Any], name: str, **details: Any) -> None:
    status["stages"][name] = details
    status["updated_unix_s"] = time.time()
    write_json(status_path, status)


def run_stage(
    *,
    name: str,
    command: list[str],
    log: Path,
    status_path: Path,
    status: dict[str, Any],
) -> None:
    environment = {
        **os.environ,
        "CHERRIES_NAME": f"Fixed contact postprocess {name}",
        "CHERRIES_TAGS": "mouthopen,contact,fixed-reference,postprocess",
    }
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=GROUP,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    stage(
        status_path,
        status,
        name,
        command=command,
        log=str(log.resolve()),
        returncode=completed.returncode,
        completed=completed.returncode == 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; inspect {log}")


def main(cfg: Config) -> None:
    assert cfg.poll_seconds > 0
    assert cfg.max_wait_seconds > 0
    process_path = cfg.run / "process.json"
    summary_path = cfg.run / "summary.json"
    status_path = cfg.run / "finalization-status.json"
    run_name = cfg.run.name
    process = json.loads(process_path.read_text())
    pid, ticks = int(process["pid"]), int(process["proc_start_ticks"])
    status: dict[str, Any] = {
        "schema": "fixed-contact-finite-postprocessing-v1",
        "run": str(cfg.run.resolve()),
        "process": process,
        "started_unix_s": time.time(),
        "stages": {},
    }
    write_json(status_path, status)
    started = time.monotonic()
    while True:
        identity = proc_identity(pid)
        if identity is None or identity[0] == "Z":
            stage(status_path, status, "wait_for_solver", result="exited")
            break
        if identity[1] != ticks:
            stage(
                status_path,
                status,
                "wait_for_solver",
                result="failed",
                reason="recorded PID was reused before terminal evidence was available",
                observed_start_ticks=identity[1],
            )
            return
        if time.monotonic() - started > cfg.max_wait_seconds:
            stage(
                status_path,
                status,
                "wait_for_solver",
                result="failed",
                reason="watcher wait budget exhausted while the recorded solver remained live",
            )
            return
        stage(
            status_path,
            status,
            "wait_for_solver",
            result="waiting",
            pid=pid,
            proc_start_ticks=ticks,
            elapsed_seconds=time.monotonic() - started,
        )
        time.sleep(cfg.poll_seconds)
    summary = json.loads(summary_path.read_text())
    if summary.get("status") not in TERMINAL:
        stage(
            status_path,
            status,
            "terminal_evidence",
            result="failed",
            reason="summary is not terminal",
            terminal_summary_status=summary.get("status"),
        )
        return
    if summary.get("provenance_verified") is not True:
        stage(
            status_path,
            status,
            "terminal_evidence",
            result="failed",
            reason="terminal summary lacks provenance_verified=true",
        )
        return
    if not (cfg.run / "source-manifest.json").is_file():
        stage(
            status_path,
            status,
            "terminal_evidence",
            result="failed",
            reason="terminal run lacks source-manifest.json",
        )
        return
    stage(
        status_path,
        status,
        "terminal_evidence",
        result="passed",
        terminal_status=summary["status"],
        accepted=len(summary.get("accepted", [])),
        frames=len(summary.get("frames", [])),
    )
    audit_output = cfg.audit_output
    if (audit_output / "summary.json").exists():
        raise FileExistsError(audit_output)
    run_stage(
        name="54_fresh_force_audit",
        command=[
            sys.executable,
            "src/54-audit-fixed-activation-contact.py",
            "--run",
            str(cfg.run),
            "--fresh-force",
            "true",
            "--output",
            str(audit_output),
        ],
        log=GROUP / f"logs/70-finish-{run_name}-54.log",
        status_path=status_path,
        status=status,
    )
    completed = (
        summary["status"] == "completed" and len(summary.get("frames", [])) == 121
    )
    render_output = (
        cfg.completed_render_output if completed else cfg.diagnostic_render_output
    )
    if (render_output / "summary.json").exists():
        raise FileExistsError(render_output)
    command = [
        sys.executable,
        "src/60-render-fixed-activation-contact.py",
        "--source",
        str(cfg.run),
        "--fixture",
        str(Path(summary["config"]["fixture"])),
        "--output",
        str(render_output),
    ]
    stage_name = "60_complete_movie" if completed else "62_terminal_diagnostic_preview"
    if not completed:
        if not summary.get("accepted"):
            stage(
                status_path,
                status,
                stage_name,
                result="skipped",
                reason="no accepted snapshots for a diagnostic preview",
            )
            stage(
                status_path,
                status,
                "final",
                result="terminal_without_render",
                numerical_status=summary["status"],
            )
            return
        command.extend(["--diagnostic-only", "true"])
    run_stage(
        name=stage_name,
        command=command,
        log=GROUP / f"logs/70-finish-{run_name}-{stage_name[:2]}.log",
        status_path=status_path,
        status=status,
    )
    stage(
        status_path,
        status,
        "final",
        result="completed",
        numerical_status=summary["status"],
        render_output=str(render_output.resolve()),
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
