# ruff: noqa: EM102, TRY003
"""Prepare the CPU-only face fixture used by the comparison runner."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit
from face_fixture import (
    DEFAULT_CUT_REFERENCE,
    DEFAULT_SKIN,
    DEFAULT_VOLUME,
    file_sha256,
    load_face_fixture,
)

from liblaf import cherries

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    """Pinned inputs and Cherries-managed fixture outputs."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    input_mesh: Path = DEFAULT_VOLUME
    input_skin: Path = DEFAULT_SKIN
    input_cut_reference: Path = DEFAULT_CUT_REFERENCE
    output_volume: Path = cherries.output("10-fixture/volume.vtu", mkdir=True)
    output_skin: Path = cherries.output("10-fixture/skin.vtp", mkdir=True)
    output_summary: Path = cherries.output("10-fixture/summary.json", mkdir=True)


def main(cfg: Config) -> None:
    fixture = load_face_fixture(
        mesh_path=cfg.input_mesh,
        skin_path=cfg.input_skin,
        cut_reference_path=cfg.input_cut_reference,
    )
    for output in (cfg.output_volume, cfg.output_skin, cfg.output_summary):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite fixture output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    fixture.mesh.save(cfg.output_volume)
    fixture.skin.save(cfg.output_skin)
    summary = {
        **fixture.summary,
        "outputs": {
            "volume": {
                "path": str(cfg.output_volume.resolve()),
                "sha256": file_sha256(cfg.output_volume),
                "size_bytes": cfg.output_volume.stat().st_size,
            },
            "skin": {
                "path": str(cfg.output_skin.resolve()),
                "sha256": file_sha256(cfg.output_skin),
                "size_bytes": cfg.output_skin.stat().st_size,
            },
        },
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cherries.log_metrics(
        {
            "fixture/active_cells": len(fixture.active_cell_ids),
            "fixture/activation_regions": len(fixture.region_names),
            "fixture/fixed_vertices": int(fixture.mesh.point_data["IsFixed"].sum()),
        }
    )
    logger.info("Wrote %s", cfg.output_volume)
    logger.info("Wrote %s", cfg.output_skin)
    logger.info("Wrote %s", cfg.output_summary)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
