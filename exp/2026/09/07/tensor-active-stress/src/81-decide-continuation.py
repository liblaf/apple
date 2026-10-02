"""Apply the preregistered probe selection and bounded continuation rules."""

# ruff: noqa: PLR0915

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import pydantic_settings as ps
from continuation_helpers import BASE
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False
MAXIMUM_GLOBAL_STEP = 256
BLOCK_ENDPOINTS = {80: 144, 144: 208, 208: MAXIMUM_GLOBAL_STEP}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("81-continuation-decision", mkdir=True)
    mode: Literal["probe", "block"]
    baseline: Path
    candidate: Path
    previous_decision: Path | None = None


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": BASE.sha256(path)}


def completed(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text())
    assert summary["status"] == "completed_fixed_budget_continuation"
    assert not summary["inverse_convergence_claimed"]
    rows = list(csv.DictReader((path / "trace.csv").open()))
    expected = int(summary["config"]["steps"])
    assert len(rows) == expected + 1
    assert all(row["solver_valid"] == "True" for row in rows)
    start = int(summary["resume"]["parent_global_step"])
    assert [int(row["step"]) for row in rows] == list(
        range(start, start + expected + 1)
    )
    assert [int(row["local_step"]) for row in rows] == list(range(expected + 1))
    assert summary["primary_endpoint"]["step"] == start + expected
    assert summary["config"]["projected_gradient_eta"] == 1.0
    return summary, rows


def endpoint_receipt(row: dict) -> dict:
    return {
        "step": int(row["step"]),
        "area_fit_rms_mm": float(row["area_fit_rms_mm"]),
        "area_motion_rms_mm": float(row["area_motion_rms_mm"]),
        "inverted_tetrahedra": int(row["inverted_tetrahedra"]),
    }


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    base, base_rows = completed(cfg.baseline)
    sources = out / "sources"
    sources.mkdir()
    for path in (Path(__file__), HERE / "docs/72-optimizer-protocol.md"):
        (sources / path.name).write_bytes(path.read_bytes())
    result = {
        "status": "completed_decision",
        "mode": cfg.mode,
        "baseline_summary": record(cfg.baseline / "summary.json"),
        "protocol": record(HERE / "docs/72-optimizer-protocol.md"),
        "geometry_used_for_selection": False,
        "inverse_convergence_claimed": False,
    }
    if cfg.mode == "probe":
        assert cfg.previous_decision is None
        assert base["config"]["phase"] == "baseline-probe"
        assert base["config"]["learning_rate"] == 0.3
        assert base["config"]["adam_eps"] == 0.01
        assert base["config"]["steps"] == 16
        assert base["initial_endpoint"]["step"] == 64
        assert base["primary_endpoint"]["step"] == 80
        initial = float(base_rows[0]["area_fit_rms_mm"])
        final_baseline = float(base_rows[-1]["area_fit_rms_mm"])
        assert final_baseline < initial
        selected = cfg.baseline
        reason = "baseline retained; calibrated probe did not meet the declared fit advantage"
        required_advantage = max(0.01, 0.05 * (initial - final_baseline))
        candidate_summary = cfg.candidate / "summary.json"
        if candidate_summary.is_file():
            candidate, candidate_rows = completed(cfg.candidate)
            assert candidate["config"]["phase"] == "calibrated-probe"
            assert candidate["config"]["steps"] == 16
            assert candidate["initial_endpoint"]["step"] == 64
            assert candidate["primary_endpoint"]["step"] == 80
            assert candidate["resume"]["calibration_receipt"] is not None
            assert candidate["first_update_replay"] is not None
            assert (
                candidate["resume"]["parent_checkpoint"]
                == base["resume"]["parent_checkpoint"]
            )
            assert (
                candidate["resume"]["controls_sha256"]
                == base["resume"]["controls_sha256"]
            )
            assert (
                candidate["resume"]["seed_displacement_sha256"]
                == base["resume"]["seed_displacement_sha256"]
            )
            assert abs(float(candidate_rows[0]["area_fit_rms_mm"]) - initial) < 1e-6
            assert (
                abs(
                    float(candidate_rows[0]["area_motion_rms_mm"])
                    - float(base_rows[0]["area_motion_rms_mm"])
                )
                < 1e-6
            )
            assert int(candidate_rows[0]["inverted_tetrahedra"]) == int(
                base_rows[0]["inverted_tetrahedra"]
            )
            final_candidate = float(candidate_rows[-1]["area_fit_rms_mm"])
            advantage = final_baseline - final_candidate
            if final_candidate < initial and advantage >= required_advantage:
                selected = cfg.candidate
                reason = (
                    "calibrated probe met the preregistered 16-update fit advantage"
                )
            result.update(
                {
                    "candidate_summary": record(candidate_summary),
                    "candidate_final_fit_rms_mm": final_candidate,
                    "calibrated_fit_advantage_mm": advantage,
                    "branch_endpoints": {
                        "baseline": endpoint_receipt(base_rows[-1]),
                        "calibrated": endpoint_receipt(candidate_rows[-1]),
                    },
                }
            )
        else:
            failure = cfg.candidate / "failure.json"
            assert failure.is_file()
            result["candidate_failure"] = {
                "record": record(failure),
                "failure": json.loads(failure.read_text()),
            }
            reason = "baseline retained; calibrated probe failed its numerical validity contract"
        result.update(
            {
                "initial_fit_rms_mm": initial,
                "baseline_final_fit_rms_mm": final_baseline,
                "common_initial_endpoint": endpoint_receipt(base_rows[0]),
                "baseline_final_endpoint": endpoint_receipt(base_rows[-1]),
                "required_calibrated_advantage_mm": required_advantage,
                "selected_run": str(selected.resolve()),
                "selected_checkpoint": record(selected / "optimizer-latest.pt"),
                "reason": reason,
                "next_additional_updates": 64,
                "next_final_global_step": 144,
            }
        )
    else:
        assert cfg.previous_decision is not None
        previous = json.loads(cfg.previous_decision.read_text())
        assert previous["status"] == "completed_decision"
        assert previous["mode"] in {"probe", "block"}
        if previous["mode"] == "block":
            assert previous["stop"] is False
        current_protocol = record(HERE / "docs/72-optimizer-protocol.md")
        assert previous["protocol"] == current_protocol
        assert previous["baseline_summary"] == record(cfg.baseline / "summary.json")
        assert base["config"]["phase"] == "baseline-probe"
        assert base["config"]["learning_rate"] == 0.3
        assert base["config"]["adam_eps"] == 0.01
        assert base["config"]["steps"] == 16
        assert base["initial_endpoint"]["step"] == 64
        assert base["primary_endpoint"]["step"] == 80
        selected_checkpoint = previous["selected_checkpoint"]
        assert record(Path(selected_checkpoint["path"])) == selected_checkpoint
        selected_run = Path(previous["selected_run"])
        selected, _ = completed(selected_run)
        selected_summary = json.dumps(
            record(selected_run / "summary.json"), sort_keys=True
        )
        assert selected_summary in {
            json.dumps(previous["baseline_summary"], sort_keys=True),
            json.dumps(previous.get("candidate_summary"), sort_keys=True),
        }
        assert record(selected_run / "optimizer-latest.pt") == selected_checkpoint
        candidate, rows = completed(cfg.candidate)
        assert candidate["config"]["phase"] == "fit-continuation"
        assert candidate["resume"]["parent_checkpoint"] == selected_checkpoint
        start_step = int(candidate["initial_endpoint"]["step"])
        assert int(selected["primary_endpoint"]["step"]) == start_step
        assert start_step in BLOCK_ENDPOINTS
        expected_endpoint = BLOCK_ENDPOINTS[start_step]
        assert int(candidate["primary_endpoint"]["step"]) == expected_endpoint
        assert int(candidate["config"]["steps"]) == expected_endpoint - start_step
        assert previous["next_additional_updates"] == expected_endpoint - start_step
        assert previous["next_final_global_step"] == expected_endpoint
        for key in ("learning_rate", "adam_eps"):
            setting = selected["config"][key]
            assert candidate["config"][key] == setting
            assert candidate["resume"][f"old_{key}"] == setting
            assert candidate["resume"][f"new_{key}"] == setting
        beginning = float(rows[0]["area_fit_rms_mm"])
        end = float(rows[-1]["area_fit_rms_mm"])
        endpoint = candidate["primary_endpoint"]
        global_step = int(endpoint["step"])
        assert global_step <= MAXIMUM_GLOBAL_STEP
        required_progress = max(0.01, 0.005 * beginning)
        improvement = beginning - end
        rms_ratio = float(endpoint["projected_gradient_mapping_rms"]) / float(
            base_rows[0]["projected_gradient_mapping_rms"]
        )
        max_ratio = float(endpoint["projected_gradient_mapping_max_abs"]) / float(
            base_rows[0]["projected_gradient_mapping_max_abs"]
        )
        if global_step == MAXIMUM_GLOBAL_STEP:
            stop_reason = "declared global update budget reached"
        elif rms_ratio <= 0.05 and max_ratio <= 0.05:
            stop_reason = (
                "relative projected-gradient criterion met at eta=1.0; "
                "no convergence or stationarity claim"
            )
        elif improvement < required_progress:
            stop_reason = "progress-limited at a block boundary; no convergence claim"
        else:
            stop_reason = None
        next_updates = 0 if stop_reason else min(64, MAXIMUM_GLOBAL_STEP - global_step)
        result.update(
            {
                "previous_decision": record(cfg.previous_decision),
                "candidate_summary": record(cfg.candidate / "summary.json"),
                "selected_run": str(cfg.candidate.resolve()),
                "selected_checkpoint": record(cfg.candidate / "optimizer-latest.pt"),
                "block_start_fit_rms_mm": beginning,
                "block_end_fit_rms_mm": end,
                "block_fit_improvement_mm": improvement,
                "required_block_progress_mm": required_progress,
                "block_start_endpoint": endpoint_receipt(rows[0]),
                "block_end_endpoint": endpoint_receipt(rows[-1]),
                "projected_gradient_rms_ratio_to_global64": rms_ratio,
                "projected_gradient_max_ratio_to_global64": max_ratio,
                "stop": stop_reason is not None,
                "reason": stop_reason or "continue with unchanged optimizer settings",
                "next_additional_updates": next_updates,
                "next_final_global_step": global_step + next_updates,
                "maximum_global_step": MAXIMUM_GLOBAL_STEP,
            }
        )
    BASE.write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
