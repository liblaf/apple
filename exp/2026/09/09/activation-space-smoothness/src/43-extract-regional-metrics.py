"""Extract the frozen matched-state regional diagnostics without re-solving."""

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
REGIONS = (
    "right_mouth_corner",
    "right_lateral_cheek",
    "right_lower_cheek_jaw",
    "right_nose_to_mouth",
)
PAIRS = {"learned-axis": ("axis-off", "axis-on"), "raw6": ("raw6-off", "raw6-on")}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison: Path = GROUP / "data/40-comparison/summary.json"
    output_dir: Path = GROUP / "data/43-regional-matched-metrics"


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite regional extraction: {out}"
    (out / "sources").mkdir()
    sources = {}
    for source in (Path(__file__), GROUP / "src/experiment_profile.py"):
        snapshot = out / "sources" / source.name
        shutil.copy2(source, snapshot)
        snapshot.chmod(0o444)
        sources[source.name] = record(snapshot)
        assert sources[source.name]["sha256"] == record(source)["sha256"]
    comparison = json.loads(cfg.comparison.read_text())
    rows = []
    for pair, (off_case, on_case) in PAIRS.items():
        match = comparison["matches"][pair]
        assert match["available"] and not match["interpolation"]
        off, on = (match["states"][case] for case in (off_case, on_case))
        for region in REGIONS:
            fit_key = f"roi_{region}_fit_vector_rms_mm"
            projection_key = f"low_frequency_{region}_normal_target_projection"
            row = {
                "pair": pair,
                "region": region,
                "off_step": int(off["step"]),
                "on_step": int(on["step"]),
                "off_fit_rms_mm": off[fit_key],
                "on_fit_rms_mm": on[fit_key],
                "fit_change_on_minus_off_mm": on[fit_key] - off[fit_key],
                "off_low_frequency_projection": off[projection_key],
                "on_low_frequency_projection": on[projection_key],
            }
            for name in ("residual", "displacement"):
                key = f"roughness_{region}_normal_{name}_highpass_5mm_rms_mm"
                if region == "right_nose_to_mouth":
                    assert key not in off and key not in on
                    row[f"off_{name}_hp_rms_mm"] = None
                    row[f"on_{name}_hp_rms_mm"] = None
                    row[f"{name}_hp_reduction_percent"] = None
                else:
                    assert off[key] > 0
                    row[f"off_{name}_hp_rms_mm"] = off[key]
                    row[f"on_{name}_hp_rms_mm"] = on[key]
                    row[f"{name}_hp_reduction_percent"] = 100 * (1 - on[key] / off[key])
            rows.append(row)
    csv_path = out / "regional-matched-metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed_extraction_of_frozen_matched_states",
        "comparison": record(cfg.comparison),
        "sources": sources,
        "definitions": {
            "reduction_percent": "100 * (1 - on / off); negative means a larger score with smoothing",
            "regional_fit": "Uniform vertex-vector RMS on the existing frozen regional support",
            "highpass": "Existing area-weighted rest-normal 5 mm high-pass scores; no filter or geometry recomputation",
            "nasolabial_highpass": "Not defined by the frozen protocol; CSV cells are intentionally empty. Nasolabial protection uses fit, low-frequency measurements, and exact sections.",
        },
        "rows": rows,
        "output": record(csv_path),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    cherries.log_output(csv_path)
    cherries.log_output(summary_path)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
