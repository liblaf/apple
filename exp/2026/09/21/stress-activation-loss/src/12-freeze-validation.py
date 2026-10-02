"""Bind the full-face mechanics gate and final CPU projections to frozen sources."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pydantic_settings as ps
from experiment import Profile
from run_support import archive, receipt, write_json

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("12-validation-inverse-v2")
    mechanics: Path = Path("10-validation-inverse-v2")
    activation: Path = Path("05-activation-validation-inverse-v2-003/receipt.json")


def main(cfg: Config) -> None:
    source = cherries.input(cfg.mechanics)
    cpu_path = cherries.input(cfg.activation)
    checks = json.loads((source / "checks.json").read_text())
    assert checks["passed"]
    baseline = json.loads((source / "protocol.json").read_text())
    cpu = json.loads(cpu_path.read_text())
    assert cpu["passed"]
    assert checks["source_protocol"] == receipt(source / "protocol.json")
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    records = archive(out)
    # Every numerical source must match its fresh validation; no exemptions.
    for path, old in baseline["sources"].items():
        current = receipt(path)
        assert current["sha256"] == old["sha256"], path
        if path not in records:
            dest = out / "sources" / Path(path).relative_to(ROOT.parent)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            records[path] = {**current, "snapshot": str(dest.resolve())}
    activation_path = Path(__file__).with_name("activation_models.py")
    assert cpu["activation_source"]["sha256"] == receipt(activation_path)["sha256"]
    protocol = {
        **baseline,
        "sources": records,
        "cpu_validation": receipt(cpu_path),
        "full_face_validation": receipt(source / "checks.json"),
        "post_gate_source_changes": [],
        "validation_scope": "Full-face signed-Mandel6 mechanics/loss derivatives plus CPU derivatives/constraints for all parameterizations; final optimizer safeguard is exercised by calibration pilots.",
    }
    write_json(out / "sources.json", records)
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "checks.json",
        {
            "passed": True,
            "source_protocol": receipt(out / "protocol.json"),
            "mechanics": receipt(source / "checks.json"),
            "activation": receipt(cpu_path),
            "maximum_full_face_relative_error": checks["maximum_relative_error"],
        },
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
