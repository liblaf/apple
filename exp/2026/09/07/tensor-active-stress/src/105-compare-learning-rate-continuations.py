# ruff: noqa: EM101, TRY003
"""Reuse the immutable saved-state comparison for learning-rate continuations."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "learning_rate_saved_state_comparison", HERE / "80-compare-continuations.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(HERE / "80-compare-continuations.py")
COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARISON
SPEC.loader.exec_module(COMPARISON)
COMPARISON.ROLES = {
    "baseline32probe",
    "largerRate32probe",
    "selectedcontinuationblock",
    "laterregularization",
}
COMPLETED = False


class Config(COMPARISON.Config):
    manifest: Path = HERE.parent / "docs/105-learning-rate-comparison-manifest.json"
    output_dir: Path = cherries.output("105-learning-rate-comparison", mkdir=True)


def run(cfg: Config) -> None:
    """Validate actual saved states; this adapter changes role vocabulary only."""
    global COMPLETED  # noqa: PLW0603
    COMPARISON.main(cfg)
    if not COMPARISON.COMPLETED:
        raise RuntimeError("generic comparison did not complete")
    sources = cfg.output_dir / "sources"
    sources.mkdir()
    records = {}
    for name, source in (
        ("adapter", Path(__file__)),
        ("comparison", HERE / "80-compare-continuations.py"),
        ("measurement", HERE / "40-measure-surface.py"),
    ):
        copied = sources / source.name
        shutil.copy2(source, copied)
        records[name] = {
            "source": COMPARISON.record(source),
            "archived": COMPARISON.record(copied),
        }
    summary_path = cfg.output_dir / "summary.json"
    summary = COMPARISON.read_json(summary_path)
    summary["processing_sources"] = records
    summary["adapter_scope"] = (
        "32-update learning-rate role names only; physical measurements and saved-state checks are unchanged"
    )
    COMPARISON.write_json(summary_path, summary)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
