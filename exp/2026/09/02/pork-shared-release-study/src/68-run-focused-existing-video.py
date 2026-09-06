"""Run exactly one saved-state-only 30 FPS focused 2D video render.

Use one invocation per case.  It calls ParaView only and imports no physics
solver; the per-case destination prevents parallel jobs from sharing frames or
video files.
"""

from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)

CASES = {
    "h010-direct-nu49": "direct-nu49",
    "h020-shared": "h020-shared-nu49",
    "h005-l2": "loss-l2",
    "h005-l1": "loss-l1",
    "h005-linf": "loss-linf",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    case: str
    h010_root: Path = Path(
        "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
    )
    h020_root: Path = Path("data/20-canonical-h020")
    output_root: Path = Path("data/70-focused-h010-existing-results")
    renderer: Path = Path(__file__).with_name("63-render-focused-h010-materials.py")


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def main(config: Config) -> None:
    if config.case not in CASES:
        raise ValueError(f"case must be one of {sorted(CASES)}, got {config.case}")
    if not config.renderer.is_file():
        raise FileNotFoundError(config.renderer)
    output = config.output_root / CASES[config.case]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"per-case video output must be empty: {output}")
    command = [
        "/usr/bin/pvpython",
        str(config.renderer.resolve()),
        "--h010-root",
        str(config.h010_root.resolve()),
        "--h020-canonical-root",
        str(config.h020_root.resolve()),
        "--loss-root",
        str(config.h010_root.resolve()),
        "--output-root",
        str(config.output_root.resolve()),
        "--video-case",
        config.case,
    ]
    logger.info("Rendering saved states only: %s", command)
    subprocess.run(command, check=True)
    rendered = json_object(output / "render-receipt.json")
    root_receipt = config.output_root / f"{CASES[config.case]}-video-receipt.json"
    receipt = json_object(root_receipt)
    if (
        receipt.get("video_case") != config.case
        or rendered.get("case") != config.case
        or rendered.get("video", {}).get("fps") != 30
        or rendered.get("one_saved_state_per_png") is not True
    ):
        raise ValueError("per-case rendered-video receipt failed validation")
    cherries.log_asset(output / "evolution.mp4")
    cherries.log_asset(output / "render-receipt.json")
    cherries.log_asset(root_receipt)


if __name__ == "__main__":
    cherries.main(main)
