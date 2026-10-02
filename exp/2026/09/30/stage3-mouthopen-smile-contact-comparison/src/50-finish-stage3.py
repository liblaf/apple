# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, TRY003
"""Finite, non-signaling finalizer for the paired Stage-3 transition runs."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PYTHON = ROOT / ".venv/bin/python"
AUDITOR = GROUP / "src/30-audit-stage3-transition.py"
RENDERER = GROUP / "src/40-render-stage3-transition.py"
PREPARED = GROUP / "data/10-stage3"
FINISH = GROUP / "data/50-finish"


@dataclass(frozen=True)
class Branch:
    name: str
    pid: int
    start_ticks: int
    collision_enabled: bool

    @property
    def run(self) -> Path:
        return GROUP / "data" / self.name

    @property
    def audit(self) -> Path:
        return GROUP / "data" / f"30-{self.name.removeprefix('20-')}-audit"

    @property
    def render(self) -> Path:
        return GROUP / "data" / f"40-{self.name.removeprefix('20-')}-render"


BRANCHES = (
    Branch("20-collision-on", 47061, 70330, True),
    Branch("20-collision-off", 47059, 70329, False),
)
TERMINAL = {"completed", "blocked", "failed", "interrupted"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def proc_start_ticks(pid: int) -> int | None:
    """Return Linux /proc stat start ticks, distinguishing a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    closing = stat.rfind(")")
    assert closing >= 0
    fields = stat[closing + 2 :].split()
    # Fields begin at process state (stat field 3); starttime is stat field 22.
    assert len(fields) >= 20
    return int(fields[19])


def process_identity(branch: Branch) -> dict:
    receipt = json.loads((branch.run / "process.json").read_text())
    assert receipt["pid"] == branch.pid
    assert receipt["proc_start_ticks"] == branch.start_ticks
    actual = proc_start_ticks(branch.pid)
    return {
        "pid": branch.pid,
        "recorded_start_ticks": branch.start_ticks,
        "observed_start_ticks": actual,
        "identity_matches": actual == branch.start_ticks,
        "exited": actual is None,
    }


def final_summary(branch: Branch) -> dict | None:
    path = branch.run / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    if summary.get("status") not in TERMINAL:
        return None
    if summary.get("provenance_verified") is not True:
        return None
    assert summary["config"]["collision_enabled"] is branch.collision_enabled
    assert summary["config"]["activation_stage"] == "rankone_fixed"
    return summary


def audit_is_valid(branch: Branch, summary: dict) -> bool:
    path = branch.audit / "summary.json"
    if not path.is_file():
        return False
    audit = json.loads(path.read_text())
    expected_status = (
        "verified_completed"
        if summary["status"] == "completed"
        else "verified_incomplete"
    )
    return (
        audit.get("schema") == "stage3-matched-transition-independent-audit-v1"
        and audit.get("status") == expected_status
        and audit.get("collision_enabled") is branch.collision_enabled
        and audit.get("numerical_status") == summary["status"]
    )


def render_is_valid(branch: Branch, summary: dict) -> bool:
    path = branch.render / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    source = manifest.get("source_summary", {})
    return (
        manifest.get("collision_enabled") is branch.collision_enabled
        and source.get("path") == str((branch.run / "summary.json").resolve())
        and source.get("sha256") == sha256(branch.run / "summary.json")
        and summary["status"] == "completed"
    )


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(name: str, command: list[str]) -> dict:
    FINISH.mkdir(parents=True, exist_ok=True)
    stdout = FINISH / f"{name}.stdout.log"
    stderr = FINISH / f"{name}.stderr.log"
    receipt = FINISH / f"{name}.receipt.json"
    started = time.time()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(
            command, cwd=GROUP, stdout=out, stderr=err, check=False
        )
    value = {
        "command": command,
        "cwd": str(GROUP),
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "returncode": completed.returncode,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    write_json(receipt, value)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return value


def audit(branch: Branch, summary: dict) -> dict:
    if audit_is_valid(branch, summary):
        return {"status": "existing_valid", "path": str(branch.audit / "summary.json")}
    assert not branch.audit.exists(), branch.audit
    return run_command(
        f"audit-{branch.name}",
        [
            str(PYTHON),
            "-u",
            str(AUDITOR),
            "--run",
            str(branch.run.resolve()),
            "--source",
            str(PREPARED.resolve()),
            "--output",
            str(branch.audit.resolve()),
            "--fresh-force",
            "true",
        ],
    )


def render(branch: Branch, summary: dict, other: Branch) -> dict:
    if summary["status"] != "completed":
        return {"status": "skipped_noncompleted"}
    assert len(summary["frames"]) == 121
    if render_is_valid(branch, summary):
        return {
            "status": "existing_valid",
            "path": str(branch.render / "manifest.json"),
        }
    assert not branch.render.exists(), branch.render
    return run_command(
        f"render-{branch.name}",
        [
            str(PYTHON),
            "-u",
            str(RENDERER),
            "--source",
            str(branch.run.resolve()),
            "--comparison-source",
            str(other.run.resolve()),
            "--prepared",
            str(PREPARED.resolve()),
            "--audit",
            str((branch.audit / "summary.json").resolve()),
            "--comparison-audit",
            str((other.audit / "summary.json").resolve()),
            "--output",
            str(branch.render.resolve()),
            "--collision-enabled",
            str(branch.collision_enabled).lower(),
        ],
    )


def main() -> None:
    deadline = time.monotonic() + 14_400
    FINISH.mkdir(parents=True, exist_ok=True)
    audit_receipts: dict[str, dict] = {}
    while True:
        identities = {branch.name: process_identity(branch) for branch in BRANCHES}
        summaries = {branch.name: final_summary(branch) for branch in BRANCHES}
        write_json(
            FINISH / "watch.json",
            {
                "identities": identities,
                "finalized": {
                    name: summary is not None for name, summary in summaries.items()
                },
            },
        )
        for branch in BRANCHES:
            identity = identities[branch.name]
            if (
                identity["observed_start_ticks"] is not None
                and not identity["identity_matches"]
            ):
                raise RuntimeError(f"PID identity changed for {branch.name}")
            summary = summaries[branch.name]
            if summary is not None and branch.name not in audit_receipts:
                audit_receipts[branch.name] = audit(branch, summary)
        if all(summaries.values()):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Stage-3 solver finalization did not arrive within four hours"
            )
        time.sleep(20)

    if not all(summary["status"] == "completed" for summary in summaries.values()):
        render_receipts = {
            branch.name: {"status": "skipped_noncompleted"} for branch in BRANCHES
        }
        write_json(
            FINISH / "finalization.json",
            {"audits": audit_receipts, "renders": render_receipts},
        )
        return
    while True:
        identities = {branch.name: process_identity(branch) for branch in BRANCHES}
        for branch in BRANCHES:
            identity = identities[branch.name]
            if (
                identity["observed_start_ticks"] is not None
                and not identity["identity_matches"]
            ):
                raise RuntimeError(f"PID identity changed for {branch.name}")
        if all(identity["exited"] for identity in identities.values()):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Stage-3 solver processes did not exit within four hours"
            )
        time.sleep(20)
    for branch in BRANCHES:
        assert audit_is_valid(branch, summaries[branch.name])
    render_receipts = {
        branch.name: render(branch, summaries[branch.name], BRANCHES[1 - index])
        for index, branch in enumerate(BRANCHES)
    }
    write_json(
        FINISH / "finalization.json",
        {"audits": audit_receipts, "renders": render_receipts},
    )


if __name__ == "__main__":
    main()
