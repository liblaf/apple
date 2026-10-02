"""Extract learning-rate and distortion evidence from existing saved traces."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
FIELDS = (
    "step",
    "fit_rms_mm",
    "motion_rms_mm",
    "inverted_all_cells",
    "detF_min",
    "shortening_fraction_p99",
    "smoothness_C",
    "smoothness_Z",
    "gradient_rms",
    "z_update_frobenius_rms",
    "geometry_update_face_vector_rms_mm",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = GROUP / "data/45-learning-rate-evidence"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    return rows


def metrics(row: dict[str, str]) -> dict[str, float | int]:
    return {
        name: int(row[name])
        if name in {"step", "inverted_all_cells"}
        else float(row[name])
        for name in FIELDS
        if name in row
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite evidence: {out}"
    sources = out / "sources"
    sources.mkdir()
    inputs: list[dict[str, str]] = []

    def record(path: Path) -> Path:
        inputs.append({"path": str(path), "sha256": digest(path)})
        return path

    for path in (Path(__file__), GROUP / "src/experiment_profile.py"):
        shutil.copyfile(path, sources / path.name)
    calibration = GROUP / "data/18-learned-axis-calibration-refined"
    settings = json.loads(record(calibration / "summary.json").read_text())
    pilots = json.loads(record(calibration / "pilots.json").read_text())
    endpoints = []
    for pilot in pilots:
        if not pilot["name"].startswith("lr-"):
            continue
        rows = read_trace(record(calibration / pilot["name"] / "trace.csv"))
        first_inversion = next(
            (row for row in rows if int(row["inverted_all_cells"]) > 0), None
        )
        endpoints.append(
            {
                "pilot": pilot["name"],
                "learning_rate": pilot["learning_rate"],
                "screen_passed": pilot["stable"],
                **metrics(rows[-1]),
                "first_inversion_step": None
                if first_inversion is None
                else int(first_inversion["step"]),
                "first_inversion_fit_rms_mm": None
                if first_inversion is None
                else float(first_inversion["fit_rms_mm"]),
                "first_inversion_motion_rms_mm": None
                if first_inversion is None
                else float(first_inversion["motion_rms_mm"]),
            }
        )
    assert len(endpoints) == 4
    with (out / "pilot-endpoints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    primary = {}
    milestone_rows = []
    for case in ("24-learned-axis", "25-learned-axis-smooth"):
        folder = GROUP / "data" / case
        rows = read_trace(record(folder / "trace.csv"))
        summary = json.loads(record(folder / "summary.json").read_text())
        provenance = json.loads(record(folder / "provenance.json").read_text())
        best = min(rows, key=lambda row: float(row["fit_rms_mm"]))
        first_inversion = next(
            row for row in rows if int(row["inverted_all_cells"]) > 0
        )
        peak_inversions = max(rows, key=lambda row: int(row["inverted_all_cells"]))
        primary[case] = {
            "status": summary["status"],
            "optimizer": provenance["optimizer"],
            "failure_policy": provenance["failure_policy"],
            "first_inversion": metrics(first_inversion),
            "best_fit": metrics(best),
            "last_accepted": metrics(rows[-1]),
            "peak_inversions": metrics(peak_inversions),
            "failure": summary["failure"],
        }
        for row in rows:
            if int(row["step"]) in {0, 8, 11, 15, 16, 20, 24, 28, 32, 64, 128}:
                milestone_rows.append({"case": case, **metrics(row)})
    with (out / "primary-milestones.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(milestone_rows[0]))
        writer.writeheader()
        writer.writerows(milestone_rows)
    result = {
        "status": "completed_saved_trace_extraction",
        "scope": "Read-only evidence extraction; no new forward, adjoint, or optimizer replay.",
        "optimizer": {
            key: settings[key] for key in ("adam_eps", "betas", "learning_rates")
        },
        "selection": {
            "pilot_steps": settings["pilot_steps"],
            "selected_lr_pilot": settings["selected_lr_pilot"],
            "largest_tested_rate_selected": settings["largest_tested_rate_selected"],
            "rule": settings["lr_rule"],
            "inversion_free_geometry_required_by_screen": False,
        },
        "pilot_endpoints": endpoints,
        "primary": primary,
        "interpretation_limits": [
            "Smaller-rate pilots achieve substantially less motion at the same update count.",
            "No lower-rate primary trajectory was run to comparable fit, motion, or 128 updates.",
            "Calibration versus primary execution sensitivity is separately documented.",
        ],
        "inputs": inputs,
        "sources": [
            {"path": str(path), "sha256": digest(path)}
            for path in sorted(sources.iterdir())
        ],
        "outputs": [
            {"path": str(path), "sha256": digest(path)}
            for path in sorted(out.glob("*.csv"))
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for path in sorted(out.rglob("*")):
        if path.is_file():
            cherries.log_output(path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(out),
                "pilot_count": len(endpoints),
            }
        )
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
