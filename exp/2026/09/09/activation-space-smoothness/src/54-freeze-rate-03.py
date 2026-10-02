"""Freeze the requested rate-0.3 smoothed-arm follow-up."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = GROUP / "data/54-rate-03-settings"


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    parent_path = GROUP / "data/18-learned-axis-calibration-refined/summary.json"
    parent = json.loads(parent_path.read_text())
    provenance_path = GROUP / "data/25-learned-axis-smooth/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checked_sources = []
    for name, record in provenance["sources"].items():
        source = Path(record["path"])
        assert digest(source) == record["sha256"], source
        checked_sources.append({"module": name, **record})
    required = (
        "adam_eps",
        "betas",
        "smooth_length_m",
        "inputs",
        "smoothness_field",
        "initialization_seed",
        "initial_strength",
        "initialization_mode",
        "controls_validation",
        "smoothness_weight",
    )
    settings = {key: parent[key] for key in required}
    original_rate = parent["learning_rates"]["learned-axis"]
    new_rate = 0.3
    plan = GROUP / "docs/54-rate-03-plan.md"
    settings.update(
        {
            "status": "frozen_before_primary_runs",
            "learning_rates": {"learned-axis": new_rate},
            "followup": {
                "purpose": "Learning-rate-only sensitivity relative to the original learned-axis smoothed arm",
                "frozen_at_utc": datetime.now(UTC).isoformat(),
                "original_settings": {
                    "path": str(parent_path),
                    "sha256": digest(parent_path),
                },
                "original_rate": original_rate,
                "rate_fraction": new_rate / original_rate,
                "weight_policy": "Retain the original calibrated C weight without retuning",
                "plan": {"path": str(plan), "sha256": digest(plan)},
                "first_phase_steps": 128,
                "conditional_target_steps": [256, 384, 512],
                "fitting_cutoff_utc": "2026-09-09T04:40:00+00:00",
                "case": "learned-axis-smooth",
            },
        }
    )
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    sources = out / "sources"
    sources.mkdir()
    for path in (Path(__file__), GROUP / "src/experiment_profile.py", plan):
        shutil.copyfile(path, sources / path.name)
    receipt = {
        "status": "frozen",
        "learning_rate": new_rate,
        "original_learning_rate": original_rate,
        "rate_fraction": new_rate / original_rate,
        "settings_sha256": digest(out / "settings.json"),
        "numerical_source_check": "All original run source hashes match current files",
        "original_provenance": {
            "path": str(provenance_path),
            "sha256": digest(provenance_path),
        },
        "checked_sources": checked_sources,
    }
    (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for path in (parent_path, provenance_path, plan):
        cherries.log_input(path)
    for path in (out / "settings.json", out / "summary.json", sources):
        cherries.log_output(path)
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "checked_sources"},
            indent=2,
        )
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
