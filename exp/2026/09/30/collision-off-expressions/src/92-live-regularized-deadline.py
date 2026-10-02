"""Show the current two-expression deadline run above historic fit results."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

GROUP = Path(__file__).resolve().parent.parent
RUN = GROUP / "data/regularized-deadline-001"
OUTPUT = GROUP / "data/preview/index.html"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def esc(value: object) -> str:
    return html.escape(str(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card() -> str:
    driver = read(RUN / "driver.json") if (RUN / "driver.json").is_file() else None
    reservation = GROUP / "data/regularized-deadline-reservation-001/reservation.json"
    status = (
        read(reservation)["status"]
        if reservation.is_file()
        else "awaiting verified GPU handoff"
    )
    parts = [
        '<section id="regularized-deadline"><h2>Current objective: position + normal + activation smoothness</h2>',
        "<p>Both MouthOpen and Smile run serially on one GPU. Computation cutoff: "
        "30 September, 14:00 Asia/Shanghai. Visualization follows computation.</p>",
        f"<p>Reservation: <strong>{esc(status)}</strong>. ",
        "Earlier L2-only results below are historic baselines. The old L2 queue stays stopped after this run.</p>",
        "<p>Positive smoothness uses face-specific gradient calibration. The full weight sensitivity screen remains incomplete. "
        "A time or iteration limit is not inverse convergence.</p>",
    ]
    if driver is None:
        parts.append(
            "<p>New combined-objective numerical results are pending.</p></section>"
        )
        return "\n".join(parts)
    assert driver["schema"] == "collision-off-regularized-deadline-driver-v1"
    parts.append(f"<p>Driver: {esc(driver['status'])}</p>")
    for expression in ("MouthOpen", "Smile"):
        records = [row for row in driver["runs"] if row["expression"] == expression]
        parts.append(f"<h3>{expression}</h3>")
        if not records:
            parts.append("<p>Allocation reserved; calibration pending.</p>")
            continue
        for row in records:
            parts.append(f"<p>{esc(row['kind'])}: {esc(row['status'])}</p>")
            if "selected_positive_smooth_weight" in row:
                parts.append(
                    f"<p>Calibrated smoothness coefficient: {row['selected_positive_smooth_weight']:.8g}.</p>"
                )
        fit = RUN / f"fit-{expression.lower()}"
        summary_path = fit / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read(summary_path)
        final = summary.get("final")
        if final:
            terms = final["objective_components"]
            parts.append(
                f"<p>Accepted updates: {final['local_iteration']}; position RMS: {final['fit_rms_mm']:.5g} mm; "
                f"normal-angle RMS: {final['normal_angle_rms_deg']:.5g} degrees; "
                f"activation roughness: {final['activation_smoothness']:.6g}; total objective: {terms['total']:.6g}.</p>"
            )
        audit_path = fit / "independent-audit.json"
        if audit_path.is_file():
            audit = read(audit_path)
            bound = (
                audit["inputs"]["endpoint"]["sha256"] == summary["endpoint"]["sha256"]
            )
            if (fit / "endpoint.npz").is_file():
                bound = (
                    bound
                    and digest(fit / "endpoint.npz") == summary["endpoint"]["sha256"]
                )
            assert bound
            parts.append(
                f"<p>Independent forward audit: {'passed' if audit['valid_forward'] else 'failed'}.</p>"
            )
        else:
            parts.append(
                "<p>Independent endpoint audit pending; current metrics are provisional.</p>"
            )
        parts.append(
            f"<p>Inverse convergence: {esc(summary.get('inverse_converged', False))}.</p>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def main() -> None:
    subprocess.run(
        [sys.executable, str(GROUP / "src/53-live-strict-continuation.py")], check=True
    )
    page = OUTPUT.read_text()
    assert "<section>" in page
    index = page.index("<section>")
    OUTPUT.write_text(page[:index] + card() + page[index:])


if __name__ == "__main__":
    main()
