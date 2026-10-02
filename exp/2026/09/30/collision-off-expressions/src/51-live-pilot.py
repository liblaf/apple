# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, PT018, TRY003, TRY004
"""Add the active refined-pilot receipt to the CPU-only live preview."""

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
BASE_SOURCE = Path(__file__).with_name("50-live.py")


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("collision_off_live", BASE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def require_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"missing required live-pilot receipt: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed live-pilot JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"live-pilot receipt must be an object: {path}")
    return value


def require_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing required live-pilot progress: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"malformed progress row {number}: {path}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"progress row {number} is not an object: {path}")
        rows.append(row)
    return rows


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"malformed live-pilot {label}: expected object")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"malformed live-pilot {label}: expected number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"malformed live-pilot {label}: expected number") from error


def local_path(path_text: Any, label: str) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"malformed live-pilot {label}: expected nonempty path")
    return BASE.local_run(path_text)


def pilot_card(operations: dict[str, Any]) -> str:
    control = local_path(
        operations.get("active_pilot_control_dir"), "active_pilot_control_dir"
    )
    driver = require_json(control / "driver.json")
    if driver.get("schema") != "collision-off-refined-pilot-driver-v1":
        raise ValueError("malformed live-pilot driver schema")
    run = local_path(driver.get("pilot_run"), "driver.pilot_run")
    summary = require_json(run / "summary.json")
    final = require_mapping(summary.get("final"), "summary.final")
    refinement = require_json(run / "equilibrium-refinement.json")
    event = require_mapping(driver.get("refinement_event"), "driver.refinement_event")
    if refinement.get("kind") != "equilibrium_refinement_at_unchanged_parameters":
        raise ValueError("malformed live-pilot refinement kind")
    if refinement.get("optimizer_updates") != 0:
        raise ValueError(
            "malformed live-pilot refinement: optimizer_updates must be zero"
        )
    if event.get("kind") != refinement["kind"]:
        raise ValueError("live-pilot driver and refinement receipt disagree")
    parent = local_path(driver.get("parent_run"), "driver.parent_run")
    if Path(refinement.get("parent_run", "")).name != parent.name:
        raise ValueError("live-pilot parent binding disagrees with refinement receipt")
    rows = require_rows(run / "progress.jsonl")
    if rows and rows[-1].get("local_iteration") != final.get("local_iteration"):
        raise ValueError("live-pilot progress and summary local iteration disagree")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("local_iteration"), int):
            raise ValueError(
                f"malformed live-pilot local iteration in progress row {index}"
            )
        number(row.get("fit_rms_mm"), f"progress row {index} fit RMS")
        number(row.get("force_norm_n"), f"progress row {index} force")
    geometry = require_mapping(final.get("geometry"), "summary.final.geometry")
    local_updates = final.get("local_iteration")
    optimizer_steps = require_mapping(
        final.get("optimizer_steps"), "summary.final.optimizer_steps"
    )
    if not isinstance(local_updates, int) or local_updates < 0:
        raise ValueError("malformed live-pilot local accepted-update count")
    cumulative_q, cumulative_pose = (
        optimizer_steps.get("q"),
        optimizer_steps.get("pose"),
    )
    if not isinstance(cumulative_q, int) or not isinstance(cumulative_pose, int):
        raise ValueError("malformed live-pilot cumulative accepted-update counts")
    force = number(final.get("force_norm_n"), "summary.final.force_norm_n")
    threshold = number(
        final.get("force_threshold_n"), "summary.final.force_threshold_n"
    )
    rms = number(final.get("fit_rms_mm"), "summary.final.fit_rms_mm")
    inverted = geometry.get("inverted_tetrahedra")
    cell_fraction = number(
        geometry.get("inverted_fraction"), "summary.final.geometry.inverted_fraction"
    )
    fraction = number(
        geometry.get("inverted_rest_volume_fraction"),
        "summary.final.geometry.inverted_rest_volume_fraction",
    )
    if not isinstance(inverted, int):
        raise ValueError("malformed live-pilot inverted tetrahedron count")
    audit_path = run / "independent-audit.json"
    audit = require_json(audit_path) if audit_path.is_file() else None
    driver_status = str(driver.get("status", summary.get("status", "receipt_pending")))
    if audit is not None:
        if audit.get("valid_forward"):
            label, css = "AUDITED — independent endpoint audit passed", "audited"
        else:
            label, css = (
                "AUDIT FAILED — independent endpoint audit did not pass",
                "pending",
            )
    elif (
        driver_status in {"fit_running", "audit_running"}
        or summary.get("status") == "running"
    ):
        label, css = (
            "RUNNING — provisional accepted-step data pending independent audit",
            "running",
        )
    else:
        label, css = "FINISHED — independent endpoint audit pending", "pending"
    status = html.escape(driver_status)
    refinement_u = number(
        refinement.get("maximum_displacement_change_m"),
        "refinement maximum displacement",
    )
    parent_name, run_name = html.escape(parent.name), html.escape(run.name)
    card = f"""<section><h2>Active corrected pilot · MouthOpen</h2>
<p class="{css}">{label}; driver status: {status}; inverse convergence: {html.escape(str(bool(summary.get("inverse_converged"))).lower())}.</p>
<p>Run: <code>{run_name}</code>. Pilot-local accepted updates: {local_updates}; cumulative accepted q/pose updates: {cumulative_q}/{cumulative_pose}.</p>
<p>Fit RMS: {rms:.5g} mm; direct force: {force:.5g} N / threshold {threshold:.5g} N; retained inversions: {inverted} ({100 * cell_fraction:.6g}% of retained cells; {100 * fraction:.6g}% rest volume).</p>
<p>Parent <code>{parent_name}</code> remains a distinct endpoint. The pilot began with a zero-update equilibrium-refinement event at unchanged parameters (maximum displacement change {refinement_u:.6g} m); its parent history is not merged into these pilot-local curves.</p>
<div class="curves"><figure><figcaption>Pilot-local fit RMS (mm)</figcaption>{BASE.curve(rows, "fit_rms_mm", "#c75f42")}</figure><figure><figcaption>Pilot-local force (N)</figcaption>{BASE.curve(rows, "force_norm_n", "#087d81")}</figure></div></section>"""
    return card


def smile_terminal_note() -> str:
    """Expose a terminal run hidden by the intentionally frozen queue receipt."""
    run = GROUP / "data/smile-004"
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        return ""
    summary = require_json(summary_path)
    if summary.get("status") != "pose_projection_failed":
        return ""
    final = require_mapping(summary.get("final"), "Smile004 summary.final")
    steps = require_mapping(
        final.get("optimizer_steps"), "Smile004 summary.final.optimizer_steps"
    )
    q, pose = steps.get("q"), steps.get("pose")
    if not isinstance(q, int) or not isinstance(pose, int):
        raise ValueError("malformed Smile004 cumulative accepted-update counts")
    audit_path = run / "independent-audit.json"
    audit = require_json(audit_path) if audit_path.is_file() else None
    if audit is not None and audit.get("valid_forward"):
        audit_label, css = "AUDITED — independent endpoint audit passed", "audited"
    elif audit is not None:
        audit_label, css = (
            "AUDIT FAILED — independent endpoint audit did not pass",
            "pending",
        )
    else:
        audit_label, css = "STOPPED — independent audit pending", "pending"
    return f"""<section><h2>Smile004 terminal receipt</h2>
<p class="{css}">{audit_label}; cumulative q/pose updates {q}/{pose}.</p>
<p>The saved failure diagnosis says the predictor force check failed before projection. The ordinary Smile queue card may still read RUNNING because its queue receipt is frozen during the active pilot reservation. This terminal receipt does not establish inverse convergence.</p></section>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=GROUP / "data/preview/index.html"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run(
        [sys.executable, str(BASE_SOURCE), "--output", str(output)], check=True
    )
    operations_path = GROUP / "data/operations.json"
    operations = require_json(operations_path)
    if "active_pilot_control_dir" not in operations:
        return
    page = output.read_text()
    insertion = page.find("<section>")
    if insertion < 0:
        raise ValueError(f"base live preview has no card insertion point: {output}")
    output.write_text(
        page[:insertion]
        + pilot_card(operations)
        + smile_terminal_note()
        + page[insertion:]
    )


if __name__ == "__main__":
    main()
