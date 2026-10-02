# Copyright (c) 2026 liblaf
# ruff: noqa: EM101, EM102, PT018, TRY003, TRY004
"""Correct stale ordinary queue cards after rendering the unchanged live adapters."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

GROUP = Path(__file__).resolve().parent.parent
PILOT_SOURCE = Path(__file__).with_name("51-live-pilot.py")


def load_pilot() -> Any:
    spec = importlib.util.spec_from_file_location(
        "collision_off_live_pilot", PILOT_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT = load_pilot()


def require_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"missing live-reservation receipt: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed live-reservation JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"live-reservation receipt must be an object: {path}")
    return value


def terminal_label(
    summary: dict[str, Any], audit: dict[str, Any] | None
) -> tuple[str, str]:
    """Describe a terminal summary without treating its missing audit as a pass."""
    status = summary.get("status")
    if not isinstance(status, str) or status == "running":
        raise ValueError("stale-card correction requires a terminal summary status")
    inverse = str(bool(summary.get("inverse_converged"))).lower()
    escaped_status = html.escape(status)
    if audit is not None:
        if audit.get("valid_forward") is True:
            return (
                f"AUDITED — independent endpoint audit passed; inverse convergence: {inverse}.",
                "audited",
            )
        return (
            f"AUDIT FAILED — {escaped_status}; inverse convergence: {inverse}.",
            "pending",
        )
    completed = status in {"finite_budget_exhausted", "time_budget_exhausted"}
    prefix = "FINISHED" if completed else "FAILED"
    return (
        f"{prefix} — {escaped_status}; independent endpoint audit pending; inverse convergence: {inverse}.",
        "pending",
    )


def replace_ordinary_card(page: str, name: str, label: str, css: str) -> str:
    prefix = f"<section><h2>{name}</h2>"
    assert page.count(prefix) == 1, (name, page.count(prefix))
    start = page.index(prefix) + len(prefix)
    assert page.startswith('<p class="', start), name
    end = page.index("</p>", start) + len("</p>")
    replacement = f'<p class="{css}">{label}</p>'
    return page[:start] + replacement + page[end:]


def correct_stale_cards(page: str, queue: dict[str, Any]) -> str:
    expressions = queue.get("expressions")
    if not isinstance(expressions, dict):
        raise ValueError("malformed queue expressions")
    for name in ("MouthOpen", "Smile"):
        item = expressions.get(name)
        if not isinstance(item, dict) or item.get("status") != "running":
            continue
        run_text = item.get("active_run_dir")
        if not isinstance(run_text, str):
            raise ValueError(f"malformed stale {name} active_run_dir")
        run = PILOT.local_path(run_text, f"{name}.active_run_dir")
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            continue
        summary = require_json(summary_path)
        if summary.get("status") == "running":
            continue
        audit_path = run / "independent-audit.json"
        audit = require_json(audit_path) if audit_path.is_file() else None
        label, css = terminal_label(summary, audit)
        page = replace_ordinary_card(page, name, label, css)
    return page


def diagnostic_card(operations: dict[str, Any]) -> str:
    """Expose downloaded baseline progress while the reservation owns the queue."""
    reservations = operations.get("diagnostic_reservations")
    if not isinstance(reservations, list) or not reservations:
        return ""
    name = reservations[-1]
    if not isinstance(name, str) or name != Path(name).name:
        raise ValueError("malformed active diagnostic reservation name")
    reservation = require_json(GROUP / "data" / name / "reservation.json")
    if reservation.get("status") != "probe_started":
        return ""
    run = PILOT.local_path(
        reservation.get("probe_run_dir"), "reservation.probe_run_dir"
    )
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        return ""
    summary = require_json(summary_path)
    if summary.get("schema") != "collision-off-smile-saved-baseline-resolution-v1":
        return ""
    if summary.get("expression_name") != "Smile":
        raise ValueError("malformed active baseline expression")
    updates = summary.get("inverse_updates")
    if not isinstance(updates, int) or updates != 0:
        raise ValueError("malformed active baseline inverse-update count")
    status = summary.get("status")
    if not isinstance(status, str):
        raise ValueError("malformed active baseline status")
    labels = {
        "running": ("RUNNING", "running"),
        "baseline_refined_requires_probe": ("FINISHED", "pending"),
        "unresolved_baseline": ("UNRESOLVED", "pending"),
    }
    if status not in labels:
        raise ValueError(f"unknown active baseline status: {status}")
    lifecycle, css = labels[status]
    return f"""<section><h2>Active Smile baseline diagnostic</h2>
<p class="{css}">{lifecycle} — zero inverse updates; downloaded summary status: {html.escape(status)}.</p>
<p>Reservation <code>{html.escape(name)}</code> owns the queue until the diagnostic exits and recovery completes. This downloaded progress is provisional.</p></section>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=GROUP / "data/preview/index.html"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run(
        [sys.executable, str(PILOT_SOURCE), "--output", str(output)], check=True
    )
    queue = require_json(GROUP / "data/queue-state.json")
    operations = require_json(GROUP / "data/operations.json")
    page = correct_stale_cards(output.read_text(), queue)
    ordinary_mouthopen = "<section><h2>MouthOpen</h2>"
    assert page.count(ordinary_mouthopen) == 1
    insertion = page.index(ordinary_mouthopen)
    output.write_text(page[:insertion] + diagnostic_card(operations) + page[insertion:])


if __name__ == "__main__":
    main()
