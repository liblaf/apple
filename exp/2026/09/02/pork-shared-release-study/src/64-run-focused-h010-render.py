"""DEBUG-only Cherries wrapper for the revised saved-state 2D render.

The command launches ParaView only.  It does not import a solver and cannot
perform a forward or inverse physics evaluation.
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


class Config(cherries.BaseConfig):
    """Pin the two exact histories and a fresh material-only destination."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    # Do not register the source directories as Cherries inputs: each contains
    # thousands of saved VTU frames and DEBUG local snapshots would otherwise
    # copy the entire physics record before this render-only program starts.
    h010_root: Path = Path(
        "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
    )
    h020_canonical_root: Path = Path("data/20-canonical-h020")
    loss_root: Path = Path(
        "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
    )
    renderer: Path = Path(__file__).with_name("63-render-focused-h010-materials.py")
    output_dir: Path = cherries.output("70-focused-h010-existing-results", mkdir=True)
    static_only: bool = True


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def preflight(config: Config) -> None:
    required = (
        config.h010_root / "height-high/summary.json",
        config.h010_root / "height-high/history.vtu.series",
        config.h010_root / "height-high/final.vtu",
        config.h020_canonical_root / "h020-shared/summary.json",
        config.h020_canonical_root / "h020-shared/history.vtu.series",
        config.h020_canonical_root / "h020-shared/final.vtu",
        *(
            config.loss_root / f"{name}/summary.json"
            for name in ("baseline", "loss-l1", "loss-linf")
        ),
        *(
            config.loss_root / f"{name}/final.vtu"
            for name in ("baseline", "loss-l1", "loss-linf")
        ),
        config.renderer,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("missing requested saved state or renderer source")
    h010 = json_object(config.h010_root / "height-high/summary.json")
    if h010.get("height") != 0.1 or h010.get("activation_dofs") != 1200:
        raise ValueError("h=.10 selected source is not the independent baseline")
    h020 = json_object(config.h020_canonical_root / "h020-shared/summary.json")
    case = h020.get("case")
    if (
        not isinstance(case, dict)
        or {
            "name": "h020-shared",
            "height": 0.2,
            "poisson": 0.49,
            "protocol": "shared",
        }.items()
        - case.items()
    ):
        raise ValueError("h=.20 selected source is not the exact shared reference")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"render output must be empty: {config.output_dir}")
    cherries.log_input(config.renderer)


def validate_outputs(config: Config) -> None:
    root = config.output_dir.resolve()
    receipt = root / "render-receipt.json"
    required_root = (
        receipt,
        root / "h010-direct-nu49-final-shape.png",
        root / "h020-shared-nu49-final-shape.png",
        root / "h005-l2-final-shape.png",
        root / "h005-l1-final-shape.png",
        root / "h005-linf-final-shape.png",
        root / "h010-shared-activation-square.png",
        root / "h020-shared-activation-square.png",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_root):
        raise FileNotFoundError("missing root render assets")
    metadata = json_object(receipt)
    if metadata.get("available_exact_cases") != ["h010-direct-nu49", "h020-shared"]:
        raise ValueError("rendered case mapping is not the requested exact evidence")
    if metadata.get("missing_exact_cases") != [
        "h010-direct-nu35",
        "h010-shared-nu49",
        "h010-shared-release-nu49",
    ]:
        raise ValueError("missing-history disclosure changed")
    if metadata.get("static_only") != config.static_only:
        raise ValueError("static-only receipt does not match requested run")
    if config.static_only:
        for path in required_root:
            cherries.log_asset(path)
        return
    for output_name in ("direct-nu49", "h020-shared-nu49"):
        case_dir = root / output_name
        files = (
            case_dir / "evolution.mp4",
            case_dir / "final-shape.png",
            case_dir / "render-receipt.json",
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in files):
            raise FileNotFoundError(f"missing rendered history: {case_dir}")
        cherries.log_asset(case_dir / "evolution.mp4")
        cherries.log_asset(case_dir / "final-shape.png")
        cherries.log_asset(case_dir / "render-receipt.json")
    for path in required_root:
        cherries.log_asset(path)


def main(config: Config) -> None:
    preflight(config)
    command = [
        "/usr/bin/pvpython",
        str(config.renderer.resolve()),
        "--h010-root",
        str(config.h010_root.resolve()),
        "--h020-canonical-root",
        str(config.h020_canonical_root.resolve()),
        "--loss-root",
        str(config.loss_root.resolve()),
        "--output-root",
        str(config.output_dir.resolve()),
    ]
    if config.static_only:
        command.append("--static-only")
    logger.info("Running ParaView render only: %s", command)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    logger.info("pvpython stdout:\n%s", completed.stdout or "<empty>")
    logger.info("pvpython stderr:\n%s", completed.stderr or "<empty>")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ParaView renderer failed with exit code {completed.returncode}"
        )
    validate_outputs(config)


if __name__ == "__main__":
    cherries.main(main)
