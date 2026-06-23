from __future__ import annotations

import json
import logging
from pathlib import Path

import pydantic_settings as ps
from _human_face_output import write_table

from liblaf import cherries

logger = logging.getLogger(__name__)


class CombineConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    target: str = "smile"
    case_set: str = "required"
    skin_summary: Path = cherries.input(
        "20-human-face-smile-skin-pre0pct-lr03-summary.json"
    )
    no_skin_summary: Path = cherries.input(
        "20-human-face-smile-no-skin-lr03-summary.json"
    )
    output_summary: Path = cherries.output("20-inverse-summary.json", mkdir=True)
    output_table: Path = cherries.output("20-inverse-table.md", mkdir=True)
    inverse_lr: float = 0.03
    inverse_loss_min_delta: float = 5.0e-8


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(cfg: CombineConfig) -> None:
    rows = [read_summary(cfg.skin_summary), read_summary(cfg.no_skin_summary)]
    summary = {
        "complete": all(row["inverse/converged"] for row in rows),
        "cases": rows,
        "target/requested": cfg.target,
        "case_set/requested": cfg.case_set,
        "inverse/lr": float(cfg.inverse_lr),
        "inverse/loss_min_delta": float(cfg.inverse_loss_min_delta),
        "note": (
            "Combined from converged skin and no-skin case summaries; "
            "per-case inverse/max_steps fields are authoritative."
        ),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_table(cfg.output_table, rows)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
