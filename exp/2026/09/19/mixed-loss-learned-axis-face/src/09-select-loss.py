"""Select the balanced mixed data objective from matched loss pilots."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]


def main() -> None:
    pilot = cherries.input(GROUP / "data/08-loss-pilot")
    output = cherries.output("09-loss-selection/selection.json", mkdir=True)
    protocol = json.loads((pilot / "protocol.json").read_text())
    assert protocol["config"]["phase"] == "loss-pilot"
    steps = protocol["config"]["steps"]
    assert steps == 20
    records = []
    for name, spec in protocol["branches"].items():
        summary = json.loads((pilot / name / "summary.json").read_text())
        assert summary["last_step"] == steps
        assert summary["status"] == "completed_budget_not_convergence_certified"
        row = summary["last_metrics"]
        with (pilot / name / "trace.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        assert len(trace) == steps + 1
        assert all(float(trace[-1][key]) == float(value) for key, value in row.items())
        score = max(row["normalized_position_loss"], row["normalized_gradient_loss"])
        records.append(
            {
                "name": name,
                **spec,
                "score": score,
                "all_updates_inversion_free": all(
                    float(r["inverted_all_cells"]) == 0 for r in trace
                ),
                "metrics": row,
            }
        )
    candidates = [
        r for r in records if r["kind"] == "mixed" and r["all_updates_inversion_free"]
    ]
    assert candidates, records
    selected = min(candidates, key=lambda row: (row["score"], row["beta"]))
    result = {
        "status": "selected",
        "selected_beta": selected["beta"],
        "selected_name": selected["name"],
        "selected_score": selected["score"],
        "rule": "Minimum max(L2/L20,Lg/Lg0) at20updates among mixed beta=.25,1,4 with zero inversions at every update; ties favor smaller beta",
        "pilot_steps": steps,
        "candidates": records,
        "loss_normalization": protocol["loss_normalization"],
        "pilot_protocol_sha256": hashlib.sha256(
            (pilot / "protocol.json").read_bytes()
        ).hexdigest(),
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "selection/beta": selected["beta"],
            "selection/balanced_score": selected["score"],
        }
    )


if __name__ == "__main__":
    cherries.main(main)
