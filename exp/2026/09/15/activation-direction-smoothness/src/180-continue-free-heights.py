"""Complete the three interrupted height runs with the same continuation policy."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pydantic_settings as ps

from liblaf import cherries

continued = importlib.import_module("120-continue-inexact")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("180-free-height-continuation")


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for height, source in (
        (0.10, "160-free-height-sweep"),
        (0.15, "160-free-height-sweep"),
        (0.20, "tune-w0"),
    ):
        name = f"h{round(height * 1000):03d}-unconstrained-w0"
        destination = cfg.output / f"h{round(height * 1000):03d}"
        continued.main(
            continued.Config(
                output=destination,
                base_case=Path(source) / name,
                height=height,
                reset_forward_on_failure=True,
            )
        )
        summaries.extend(
            json.loads(
                (continued.GROUP / "data" / destination / "summary.json").read_text()
            )
        )
        continued.runner.write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    cherries.main(main, profile=continued.runner.ProfileActivationStudy)
