"""Run the saved-state-only focused h=.20 material renderer with ParaView.

This wrapper deliberately invokes only ``60-render-focused-h020-materials.py``
through ParaView.  It contains no solver import and performs no physics run.
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

CASES = (
    ("h020-direct", "canonical", "direct"),
    ("h020-shared", "canonical", "shared"),
    ("h020-shared-release", "NONSTATIONARY/EXPLORATORY", "shared-release"),
)


class Config(cherries.BaseConfig):
    """Pinned saved histories and the dedicated render-only destination."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    canonical_root: Path = cherries.input("20-canonical-h020")
    exploratory_root: Path = cherries.input("40-exploratory-release-h020")
    renderer: Path = Path(__file__).with_name("60-render-focused-h020-materials.py")
    output_dir: Path = cherries.output("60-focused-h020-materials", mkdir=True)


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def case_root(config: Config, provenance: str) -> Path:
    return (
        config.canonical_root if provenance == "canonical" else config.exploratory_root
    ).resolve()


def validate_source(config: Config, name: str, provenance: str) -> None:
    root = case_root(config, provenance)
    case_dir = root / name
    summary_path = case_dir / "summary.json"
    series_path = case_dir / "history.vtu.series"
    final_path = case_dir / "final.vtu"
    required = (summary_path, series_path, final_path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete saved source for {name}: {case_dir}")

    summary = json_object(summary_path)
    case = summary.get("case")
    if not isinstance(case, dict):
        raise TypeError(f"missing case receipt: {summary_path}")
    expected = {
        "name": name,
        "length": 1.0,
        "height": 0.2,
        "muscle_layout": "band",
        "poisson": 0.49,
    }
    if any(case.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"source is not the required nu=.49 h=.20 case: {summary_path}"
        )

    if provenance == "canonical":
        expected_protocol = "direct" if name == "h020-direct" else "shared"
        if case.get("protocol") != expected_protocol:
            raise ValueError(f"canonical protocol mismatch: {summary_path}")
    else:
        continuation = summary.get("continuation")
        if (
            case.get("protocol") != "shared_then_release"
            or not isinstance(continuation, dict)
            or continuation.get("seed_status") != "NONSTATIONARY/EXPLORATORY"
        ):
            raise ValueError(f"release provenance mismatch: {summary_path}")

    inverse = summary.get("inverse")
    series = json_object(series_path)
    files = series.get("files")
    if (
        not isinstance(inverse, dict)
        or not isinstance(inverse.get("evaluations"), int)
        or not isinstance(files, list)
        or len(files) != inverse["evaluations"]
        or not files
    ):
        raise ValueError(f"invalid saved-state manifest: {series_path}")
    logger.info("Preflight source %s: %s", name, case_dir)
    for path in required:
        cherries.log_asset(path)


def verify_no_nu035_substitution(config: Config) -> None:
    """Reject selected h=.20 sources unless all are explicitly nu=.49.

    The scan is deliberately logged: a nu=.35 h=.20 artifact can coexist in a
    source root, but it must never be one of the three resolved renderer inputs.
    """
    selected = {
        (case_root(config, provenance) / name).resolve()
        for name, provenance, _output_name in CASES
    }
    nu035_candidates: list[Path] = []
    for root in (config.canonical_root.resolve(), config.exploratory_root.resolve()):
        for summary_path in root.rglob("summary.json"):
            summary = json_object(summary_path)
            case = summary.get("case")
            if (
                isinstance(case, dict)
                and case.get("height") == 0.2
                and case.get("poisson") == 0.35
            ):
                nu035_candidates.append(summary_path.parent.resolve())
    if selected.intersection(nu035_candidates):
        raise ValueError("a nu=.35 h=.20 history was selected for rendering")
    logger.info(
        "Verified no nu=.35 h=.20 source is substituted; candidates outside "
        "the selected mapping: %s",
        len(nu035_candidates),
    )


def preflight(config: Config) -> None:
    if not config.renderer.is_file():
        raise FileNotFoundError(config.renderer)
    cherries.log_input(config.renderer)
    for name, provenance, _output_name in CASES:
        validate_source(config, name, provenance)
    verify_no_nu035_substitution(config)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"render output must be empty: {config.output_dir}")


def validate_outputs(config: Config) -> None:
    root = config.output_dir.resolve()
    required_root = (
        root / "render-receipt.json",
        root / "final-comparison.png",
        root / "shared-activation-square.png",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_root):
        raise FileNotFoundError("missing required focused-render root assets")
    receipt = json_object(root / "render-receipt.json")
    if receipt.get("admissible_exact_cases") != [case[0] for case in CASES]:
        raise ValueError("renderer changed the required three-case mapping")

    for name, _provenance, output_name in CASES:
        case_dir = root / output_name
        video = case_dir / "evolution.mp4"
        final = case_dir / "final-shape.png"
        case_receipt = case_dir / "render-receipt.json"
        required = (video, final, case_receipt)
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise FileNotFoundError(f"missing rendered assets for {name}: {case_dir}")
        rendered = json_object(case_receipt)
        if rendered.get("case") != name or rendered.get("status") != "ok":
            raise ValueError(f"invalid render receipt for {name}: {case_receipt}")
        for path in required:
            cherries.log_asset(path)
    for path in required_root:
        cherries.log_asset(path)
    logger.info("Validated focused material assets at %s", root)


def main(config: Config) -> None:
    """Preflight exact history identity, then delegate rendering to ParaView."""
    preflight(config)
    command = [
        "/usr/bin/pvpython",
        str(config.renderer.resolve()),
        "--canonical-root",
        str(config.canonical_root.resolve()),
        "--exploratory-root",
        str(config.exploratory_root.resolve()),
        "--output-root",
        str(config.output_dir.resolve()),
    ]
    logger.info("Running render-only command: %s", command)
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
