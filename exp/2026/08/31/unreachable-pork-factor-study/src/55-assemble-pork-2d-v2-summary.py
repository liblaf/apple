"""Assemble one OFAT receipt from eight completed one-case v2 roots.

This is deliberately receipt-only: it never imports or recomputes mechanics.
"""

from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003, TRY004
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)

EXPECTED_CASES = (
    "baseline",
    "height-low",
    "height-high",
    "loss-l1",
    "loss-linf",
    "mesh-medium",
    "mesh-dense",
    "energy-linear",
)
SHARED_ROOT_KEYS = (
    "design",
    "geometry",
    "materials",
    "activation",
    "derivative_check",
    "paraview",
)
INVERSE_DYNAMIC_KEYS = ("stationarity_pass_cases", "stationarity_fail_cases")


class Config(cherries.BaseConfig):
    """Read eight one-case roots and write the canonical aggregate receipt."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_roots: str = ""
    output_dir: Path = cherries.output("10-pork-2d-v2", mkdir=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, rejecting any other JSON top-level type."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def assert_equal(label: str, values: list[Any]) -> Any:
    """Return the shared value, failing visibly if any receipt differs."""
    if not values:
        raise ValueError(f"no values for {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"inconsistent {label} across one-case roots")
    return first


def shared_inverse_protocol(inverse: Mapping[str, Any]) -> dict[str, Any]:
    """Strip aggregate stationarity lists before comparing shared protocol."""
    return {
        key: value for key, value in inverse.items() if key not in INVERSE_DYNAMIC_KEYS
    }


def read_one_case_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a one-case wrapper and its duplicate nested case receipt."""
    summary_path = root / "summary.json"
    wrapper = read_json(summary_path)
    cases = wrapper.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError(f"expected exactly one case in {summary_path}")
    case = cases[0]
    if not isinstance(case, dict) or not isinstance(case.get("name"), str):
        raise ValueError(f"missing one-case name in {summary_path}")
    nested_path = root / case["name"] / "summary.json"
    nested = read_json(nested_path)
    if nested != case:
        raise ValueError(f"nested/root case receipt mismatch: {root}")
    refinement = case.get("refinement")
    if not isinstance(refinement, dict):
        raise ValueError(f"missing refinement receipt: {nested_path}")
    if refinement.get("failure") is not None:
        raise RuntimeError(
            f"refinement failure in {case['name']}: {refinement['failure']}"
        )
    if refinement.get("trial_forward_failures") != 0:
        raise RuntimeError(f"refinement trial failure in {case['name']}")
    return wrapper, case


def assemble(input_roots: list[Path]) -> dict[str, Any]:
    """Assemble and validate the canonical root receipt without simulation."""
    if len(input_roots) != len(EXPECTED_CASES):
        raise ValueError(f"expected {len(EXPECTED_CASES)} input roots")
    if len(set(input_roots)) != len(input_roots):
        raise ValueError("input roots must be unique")

    wrappers_and_cases = [read_one_case_root(root) for root in input_roots]
    wrappers = [item[0] for item in wrappers_and_cases]
    by_name = {case["name"]: case for _wrapper, case in wrappers_and_cases}
    if len(by_name) != len(wrappers_and_cases) or set(by_name) != set(EXPECTED_CASES):
        raise ValueError(f"expected exactly these cases: {', '.join(EXPECTED_CASES)}")

    shared = {
        key: assert_equal(key, [wrapper.get(key) for wrapper in wrappers])
        for key in SHARED_ROOT_KEYS
    }
    inverse_protocol = assert_equal(
        "inverse protocol",
        [shared_inverse_protocol(wrapper.get("inverse", {})) for wrapper in wrappers],
    )
    cases = [by_name[name] for name in EXPECTED_CASES]
    stationary = [case["name"] for case in cases if case["stationarity"]["passed"]]
    frames = sum(int(case["evaluations"]) for case in cases)
    elapsed_seconds = sum(float(wrapper["elapsed_seconds"]) for wrapper in wrappers)

    return {
        **shared,
        "inverse": {
            **inverse_protocol,
            "stationarity_pass_cases": stationary,
            "stationarity_fail_cases": [
                case["name"] for case in cases if case["name"] not in stationary
            ],
        },
        "cases": cases,
        "elapsed_seconds": elapsed_seconds,
        "receipt": {
            "case_count": len(cases),
            "stationarity_pass_count": len(stationary),
            "stationarity_fail_count": len(cases) - len(stationary),
            "exact_recorded_state_count": frames,
            "evaluation_count": frames,
            "elapsed_seconds_sum": elapsed_seconds,
        },
    }


def main(cfg: Config) -> None:
    """Write the aggregate JSON after validating every source receipt."""
    if not cfg.input_roots:
        raise ValueError("--input-roots must list eight one-case output roots")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty {cfg.output_dir}")
    roots = [Path(part.strip()) for part in cfg.input_roots.split(",") if part.strip()]
    summary = assemble(roots)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.output_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt = summary["receipt"]
    logger.info(
        "Assembled %d cases: stationary=%d/%d frames=%d elapsed=%.3fs",
        receipt["case_count"],
        receipt["stationarity_pass_count"],
        receipt["case_count"],
        receipt["exact_recorded_state_count"],
        receipt["elapsed_seconds_sum"],
    )


if __name__ == "__main__":
    cherries.main(main)
