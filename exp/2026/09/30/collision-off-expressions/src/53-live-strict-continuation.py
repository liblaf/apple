# Copyright (c) 2026 liblaf
# ruff: noqa: EM101, EM102, PT018, TRY003, TRY004
"""Add a local strict-continuation receipt card after the immutable live adapters."""

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
RESERVATION_SOURCE = Path(__file__).with_name("52-live-reservation.py")


def load_reservation() -> Any:
    spec = importlib.util.spec_from_file_location(
        "collision_off_live_reservation", RESERVATION_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_reservation()


def require_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"missing strict-continuation receipt: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"malformed strict-continuation JSON: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"strict-continuation receipt must be an object: {path}")
    return value


def local_path(path_text: Any, label: str) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"malformed strict-continuation {label}")
    return BASE.PILOT.local_path(path_text, label)


def summary_lifecycle(summary: dict[str, Any], driver_status: str) -> tuple[str, str]:
    status = summary.get("status")
    if not isinstance(status, str):
        raise ValueError("malformed strict-continuation summary status")
    inverse = html.escape(str(bool(summary.get("inverse_converged"))).lower())
    if driver_status == "driver_failed":
        return (
            f"FAILED — controller failed; summary status: {html.escape(status)}; inverse convergence: {inverse}.",
            "pending",
        )
    if driver_status == "unresolved":
        return (
            f"UNRESOLVED — controller unresolved; summary status: {html.escape(status)}; inverse convergence: {inverse}.",
            "pending",
        )
    if status == "running":
        return (
            f"RUNNING — provisional accepted-step data; inverse convergence: {inverse}.",
            "running",
        )
    if status in {"finite_budget_exhausted", "time_budget_exhausted"}:
        return (
            f"FINISHED — {html.escape(status)}; inverse convergence: {inverse}.",
            "pending",
        )
    return (
        f"UNRESOLVED — {html.escape(status)}; inverse convergence: {inverse}.",
        "pending",
    )


def audit_truth(audit: dict[str, Any] | None) -> str:
    if audit is None:
        return "Independent endpoint audit pending."
    if audit.get("valid_forward") is True:
        return "Independent endpoint audit passed."
    return "Independent endpoint audit did not pass."


def no_summary_lifecycle(driver_status: str) -> tuple[str, str]:
    labels = {
        "preflight_complete": (
            "READY — controller preflight complete; no numerical result downloaded.",
            "pending",
        ),
        "fit_running": (
            "RUNNING — numerical worker active; no result downloaded.",
            "running",
        ),
        "fit_exited": ("UNRESOLVED — worker exited with no saved result.", "pending"),
        "audit_running": (
            "RUNNING — independent audit active; no result downloaded.",
            "running",
        ),
        "completed_audited_chunk": (
            "UNRESOLVED — completed controller has no downloaded result.",
            "pending",
        ),
        "unresolved": (
            "UNRESOLVED — controller reported unresolved; no result downloaded.",
            "pending",
        ),
        "driver_failed": (
            "FAILED — controller failed; no result downloaded.",
            "pending",
        ),
    }
    if driver_status not in labels:
        raise ValueError(f"unknown strict-continuation driver status: {driver_status}")
    return labels[driver_status]


def final_metrics(summary: dict[str, Any]) -> str:
    final = summary.get("final")
    if final is None:
        return "<p>No completed endpoint metrics downloaded.</p>"
    if not isinstance(final, dict):
        raise ValueError("malformed strict-continuation summary.final")
    steps = final.get("optimizer_steps")
    geometry = final.get("geometry")
    if not isinstance(steps, dict) or not isinstance(geometry, dict):
        raise ValueError("malformed strict-continuation final metrics")
    local, q, pose, inverted = (
        final.get("local_iteration"),
        steps.get("q"),
        steps.get("pose"),
        geometry.get("inverted_tetrahedra"),
    )
    if not all(isinstance(value, int) for value in (local, q, pose, inverted)):
        raise ValueError("malformed strict-continuation accepted-update counts")
    try:
        rms = float(final["fit_rms_mm"])
        force = float(final["force_norm_n"])
        threshold = float(final["force_threshold_n"])
        rest_fraction = float(geometry["inverted_rest_volume_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed strict-continuation final force metrics") from error
    return (
        f"<p>Local accepted updates: {local}; cumulative q/pose updates: {q}/{pose}. "
        f"Fit RMS: {rms:.5g} mm; direct force: {force:.5g} N / threshold {threshold:.5g} N; "
        f"retained inversions: {inverted}; inverted rest volume: {100 * rest_fraction:.6g}%.</p>"
    )


def strict_card(operations: dict[str, Any]) -> str:
    run_text = operations.get("active_strict_continuation_run")
    if run_text is None:
        return ""
    run = local_path(run_text, "active_strict_continuation_run")
    control = run.with_name(run.name + "-control")
    driver_path = control / "driver.json"
    if not driver_path.is_file():
        return f"""<section><h2>Strict MouthOpen continuation</h2>
<p class="pending">RESERVED — no numerical result downloaded.</p>
<p>Run: <code>{html.escape(run.name)}</code>. This is an ordinary continuation from the completed pilot checkpoint; no new refinement event is implied.</p></section>"""
    driver = require_json(driver_path)
    if driver.get("schema") != "collision-off-strict-continuation-driver-v1":
        raise ValueError("malformed strict-continuation driver schema")
    if (
        driver.get("continuation_run") != str(run)
        and Path(str(driver.get("continuation_run"))).name != run.name
    ):
        raise ValueError("strict-continuation driver run binding disagrees")
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        status = driver.get("status")
        if not isinstance(status, str):
            raise ValueError("malformed strict-continuation driver status")
        label, css = no_summary_lifecycle(status)
        return f"""<section><h2>Strict MouthOpen continuation</h2>
<p class="{css}">{label}</p>
<p>Run: <code>{html.escape(run.name)}</code>. This is an ordinary continuation from the completed pilot checkpoint; no new refinement event is implied.</p></section>"""
    summary = require_json(summary_path)
    audit_path = run / "independent-audit.json"
    audit = require_json(audit_path) if audit_path.is_file() else None
    driver_status = driver.get("status")
    if not isinstance(driver_status, str):
        raise ValueError("malformed strict-continuation driver status")
    label, css = summary_lifecycle(summary, driver_status)
    return f"""<section><h2>Strict MouthOpen continuation</h2>
<p class="{css}">{label}</p>
<p>Run: <code>{html.escape(run.name)}</code>; driver status: <code>{html.escape(str(driver.get("status", "unknown")))}</code>. This is an ordinary continuation from the completed pilot checkpoint; no new refinement event is implied.</p>
<p>{audit_truth(audit)}</p>{final_metrics(summary)}</section>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=GROUP / "data/preview/index.html"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run(
        [sys.executable, str(RESERVATION_SOURCE), "--output", str(output)], check=True
    )
    operations = require_json(GROUP / "data/operations.json")
    card = strict_card(operations)
    if not card:
        return
    page = output.read_text()
    ordinary_mouthopen = "<section><h2>MouthOpen</h2>"
    assert page.count(ordinary_mouthopen) == 1
    insertion = page.index(ordinary_mouthopen)
    output.write_text(page[:insertion] + card + page[insertion:])


if __name__ == "__main__":
    main()
