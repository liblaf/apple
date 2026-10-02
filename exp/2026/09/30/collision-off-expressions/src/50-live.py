"""Build a CPU-only local status page from downloaded collision-off receipts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

GROUP = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        return {"_error": str(error)}


def local_run(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        return path
    if "data" in path.parts:
        return GROUP / "data" / Path(*path.parts[path.parts.index("data") + 1 :])
    return GROUP / "data" / path.name


def curve(rows: list[dict[str, Any]], field: str, color: str) -> str:
    values = [float(row[field]) for row in rows if field in row]
    if not values:
        return "<p>No accepted-update values downloaded yet.</p>"
    low, high = min(values), max(values)
    span = max(high - low, 1e-30)
    count = max(len(values) - 1, 1)
    points = " ".join(
        f"{10 + 300 * index / count:.1f},{90 - 70 * (value - low) / span:.1f}"
        for index, value in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 320 105" role="img" aria-label="{html.escape(field)} curve">'
        f'<path d="M10,90H310M10,20V90" stroke="#aaa" fill="none"/>'
        f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>'
        f'<text x="12" y="15">{high:.5g}</text><text x="12" y="103">{low:.5g}</text></svg>'
    )


def latest_run(name: str, queue: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    if queue is not None:
        item = queue.get("expressions", {}).get(name, {})
        if "active_run_dir" in item:
            candidates.append(local_run(item["active_run_dir"]))
        candidates.extend(
            local_run(chunk["run_dir"]) for chunk in item.get("chunks", [])
        )
    candidates.extend(sorted((GROUP / "data").glob(f"{name.lower()}-[0-9][0-9][0-9]")))
    return max(
        (path for path in candidates if path.is_dir()),
        key=lambda path: path.name,
        default=None,
    )


def expression_card(name: str, queue: dict[str, Any] | None) -> str:
    item = (queue or {}).get("expressions", {}).get(name, {})
    queue_status = item.get("status", "pending_download")
    run = latest_run(name, queue)
    if run is None:
        return f'<section><h2>{name}</h2><p class="pending">PENDING — no downloaded run receipt.</p></section>'
    summary = load_json(run / "summary.json")
    progress_path, audit_path = run / "progress.jsonl", run / "independent-audit.json"
    rows = []
    if progress_path.is_file():
        rows = [
            json.loads(line) for line in progress_path.read_text().splitlines() if line
        ]
    audit = load_json(audit_path)
    if audit and audit.get("valid_forward"):
        label, cls = "AUDITED — independent endpoint audit passed", "audited"
    elif summary and summary.get("status") == "running":
        label, cls = (
            "RUNNING — downloaded checkpoint is not an audited endpoint",
            "running",
        )
    elif queue_status in {"queued", "continue"}:
        label, cls = "PENDING — queue has no audited endpoint", "pending"
    else:
        label, cls = f"{queue_status.upper()} — audit pending or unavailable", "pending"
    final = (summary or {}).get("final", {})
    review = GROUP / "data" / f"review-{run.name}" / "index.html"
    review_link = (
        f'<a href="../review-{run.name}/">Full-surface review</a>'
        if review.is_file()
        else "Full-surface review pending"
    )
    metrics = ""
    if final:
        metrics = (
            f"<p>Accepted updates: {final.get('iteration', '?')}; fit RMS: {final.get('fit_rms_mm', float('nan')):.5g} mm; "
            f"force: {final.get('force_norm_n', float('nan')):.5g} N; inversion: "
            f"{final.get('geometry', {}).get('inverted_tetrahedra', '?')} retained cells.</p>"
        )
    return f"""<section><h2>{name}</h2><p class="{cls}">{label}</p><p>Run: <code>{html.escape(run.name)}</code> · {review_link}</p>{metrics}<div class="curves"><figure><figcaption>Fit RMS (mm)</figcaption>{curve(rows, "fit_rms_mm", "#c75f42")}</figure><figure><figcaption>Force (N)</figcaption>{curve(rows, "force_norm_n", "#087d81")}</figure></div></section>"""


def diagnostic_reservation_card(operations: dict[str, Any] | None) -> str:
    """Show controller reservations without treating stale queue state as stopped."""
    reservations = (operations or {}).get("diagnostic_reservations", [])
    if not reservations:
        return ""
    assert isinstance(reservations, list)
    name = reservations[-1]
    assert isinstance(name, str)
    assert name == Path(name).name
    assert name not in {"", ".", ".."}
    path = GROUP / "data" / name / "reservation.json"
    receipt = load_json(path)
    status = (receipt or {}).get("status", "reservation_download_pending")
    labels = {
        "queue_frozen": "QUEUE TEMPORARILY RESERVED — controller is waiting for the existing fit child to finish.",
        "probe_started": "QUEUE TEMPORARILY RESERVED — diagnostic probe is running.",
        "queue_resumed": "QUEUE RESUMED — controller recovered from the reservation.",
        "queue_reaped_and_audited": "QUEUE REAPED AND AUDITED — controller recorded the child outcome.",
        "completed": "DIAGNOSTIC COMPLETED — see the downloaded reservation receipt.",
        "probe_failed_queue_recovered": "DIAGNOSTIC FAILED; QUEUE RECOVERED — controller resumed the expression queue.",
        "manual_recovery_required": "MANUAL RECOVERY REQUIRED — do not infer queue state from stale receipts.",
    }
    text = labels.get(
        status, "DIAGNOSTIC RESERVATION — awaiting a downloaded controller receipt."
    )
    fields = receipt or {}
    details = [
        f"{name}: <code>{html.escape(str(fields[name]))}</code>"
        for name in ("probe_pid", "probe_run_dir", "probe_exit_code")
        if name in fields
    ]
    source = (
        html.escape(str(path.relative_to(GROUP)))
        if receipt
        else "reservation receipt not downloaded"
    )
    return f'<section><h2>Diagnostic reservation</h2><p class="running">{text}</p><p>Phase: <code>{html.escape(str(status))}</code> · {" · ".join(details) or source}</p></section>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=GROUP / "data/preview/index.html"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    queue = load_json(GROUP / "data/queue-state.json")
    operations = load_json(GROUP / "data/operations.json")
    cards = "".join(expression_card(name, queue) for name in ("MouthOpen", "Smile"))
    reservation = diagnostic_reservation_card(operations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Collision-off expression status</title><style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}section{{border:1px solid #ddd;padding:1rem;margin:1rem 0}}.pending,.running,.audited{{padding:.6rem;font-weight:600}}.pending{{background:#fff1c7}}.running{{background:#dceeff}}.audited{{background:#dff4e5}}.curves{{display:flex;gap:1rem;flex-wrap:wrap}}figure{{margin:0}}svg{{width:320px;height:105px;background:#fafafa}}</style><h1>Collision-off corrected-neutral expressions</h1><p>This page reads downloaded queue, summary, progress, and audit files only. Collision and contact are disabled in the numerical experiment; no collision claim is displayed as an acceptance result.</p>{reservation}{cards}"""
    )


if __name__ == "__main__":
    main()
