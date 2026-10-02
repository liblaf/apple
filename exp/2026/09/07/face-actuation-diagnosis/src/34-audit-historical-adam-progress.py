"""Capture an exact common-prefix comparison of paired historical Adam runs."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

METHODS = {
    "Raw6": {"color": "#3366cc", "weight": 0.0},
    "Raw6-S": {"color": "#d97706", "weight": 5e-4},
}
CHECKPOINT_COPY_SOURCE_BEFORE = (
    "f4a1509aa8ac31149c14be01584c4256ceb65a3f9b04b658b39cc033846f0337"
)
CHECKPOINT_COPY_SOURCE_AFTER = (
    "691872807eaee0e33ecf222de03ed4880f45344dc7bfeaa5195b17394863e5f6"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> dict[str, str | int]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def stable_read(path: Path) -> bytes:
    first = path.read_bytes()
    time.sleep(0.2)
    second = path.read_bytes()
    if first != second:
        raise RuntimeError(f"live source changed during capture: {path}")
    return first


def parse_csv(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    lines = data.decode().splitlines(keepends=True)
    records = list(csv.DictReader(data.decode().splitlines()))
    return lines, records


def parse_jsonl(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [line for line in data.decode().splitlines(keepends=True) if line.strip()]
    return lines, [json.loads(line) for line in lines]


def bool_csv(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid CSV boolean {value!r}")
    return value == "True"


def assert_consecutive(rows: list[dict[str, Any]], source: str) -> None:
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(len(steps))):
        raise ValueError(f"{source} steps are not consecutive from zero: {steps}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_receipt(path: Path, data: bytes, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes_at_read_time": len(data),
        "sha256_at_read_time": sha256_bytes(data),
        "rows_at_read_time": rows,
        "mutability": (
            "live append-by-rewrite source; hash identifies the copied read-time "
            "snapshot, not the eventual final-run file"
        ),
    }


def save_figure(fig: plt.Figure, output: Path, stem: str) -> dict[str, Any]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": digest(png), "pdf": digest(pdf)}


def add_status_title(
    fig: plt.Figure, title: str, *, status_label: str, watermark: str | None
) -> None:
    fig.suptitle(f"{title}\n{status_label}", fontsize=12, fontweight="bold")
    if watermark is not None:
        fig.text(
            0.5,
            0.5,
            watermark,
            ha="center",
            va="center",
            fontsize=42,
            color="0.75",
            alpha=0.18,
            rotation=25,
            fontweight="bold",
        )


def plot_metric_vs_step(
    paired: list[dict[str, Any]],
    output: Path,
    *,
    key: str,
    ylabel: str,
    stem: str,
    title: str,
    status_label: str,
    watermark: str | None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, style in METHODS.items():
        steps = np.asarray([row["step"] for row in paired])
        values = np.asarray([row[method][key] for row in paired])
        valid = np.asarray([row[method]["solver_valid"] for row in paired], dtype=bool)
        ax.scatter(
            steps[valid],
            values[valid],
            s=24,
            color=style["color"],
            label=method,
            zorder=3,
        )
        if np.any(~valid):
            ax.scatter(
                steps[~valid],
                values[~valid],
                s=55,
                marker="x",
                linewidth=1.8,
                color="#b91c1c",
                label=f"{method} unsuccessful solve",
                zorder=4,
            )
    ax.set_xlabel(
        "continuation local optimizer step (source step 50; identical Adam moment reset)"
    )
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    add_status_title(fig, title, status_label=status_label, watermark=watermark)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, output, stem)


def plot_tradeoff(
    paired: list[dict[str, Any]],
    output: Path,
    *,
    status_label: str,
    watermark: str | None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    for method, style in METHODS.items():
        variation = np.asarray([row[method]["graph_variation_R"] for row in paired])
        fit = np.asarray([row[method]["fit_rms_mm"] for row in paired])
        valid = np.asarray([row[method]["solver_valid"] for row in paired], dtype=bool)
        ax.scatter(
            variation[valid],
            fit[valid],
            s=26,
            color=style["color"],
            label=method,
            zorder=3,
        )
        if np.any(valid):
            indices = np.flatnonzero(valid)
            for index in {int(indices[0]), int(indices[-1])}:
                ax.annotate(
                    str(paired[index]["step"]),
                    (variation[index], fit[index]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                )
        if np.any(~valid):
            ax.scatter(
                variation[~valid],
                fit[~valid],
                s=55,
                marker="x",
                linewidth=1.8,
                color="#b91c1c",
                label=f"{method} unsuccessful solve",
                zorder=4,
            )
    ax.set_xlabel("physical-Frobenius graph variation R (dimensionless)")
    ax.set_ylabel("uniform fit RMS (mm)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    add_status_title(
        fig,
        "Fit-variation tradeoff at saved optimizer evaluations",
        status_label=status_label,
        watermark=watermark,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, output, "fit-vs-graph-variation")


def continuation_pair_contract(
    directories: dict[str, Path], configs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Validate the paired reset-continuation contract without treating paths as equal."""
    differing = sorted(
        key for key in configs["Raw6"] if configs["Raw6"][key] != configs["Raw6-S"][key]
    )
    allowed = ["output_dir", "smoothness_weight", "source_checkpoint", "source_run"]
    if differing != allowed:
        raise ValueError(f"unexpected continuation config differences: {differing}")
    for method, expected in METHODS.items():
        if float(configs[method]["smoothness_weight"]) != expected["weight"]:
            raise ValueError(f"unexpected {method} smoothness weight")
        if int(configs[method]["steps"]) != 150:
            raise ValueError("continuation must declare a 150-local-step budget")
        checkpoint = Path(configs[method]["source_checkpoint"]).resolve()
        source_run = Path(configs[method]["source_run"]).resolve()
        if checkpoint.parent != source_run or checkpoint.name != "step-0050.npz":
            raise ValueError(
                f"{method} source checkpoint is not its immutable step-50 file"
            )
    if (
        Path(configs["Raw6"]["source_run"]).resolve()
        == Path(configs["Raw6-S"]["source_run"]).resolve()
    ):
        raise ValueError(
            "paired continuations must retain distinct interrupted treatments"
        )

    provenance = {
        method: json.loads((directory / "provenance.json").read_text())
        for method, directory in directories.items()
    }

    def required_string_map(
        record: dict[str, Any], key: str, method: str
    ) -> dict[str, str]:
        value = record.get(key)
        if (
            not isinstance(value, dict)
            or not value
            or not all(isinstance(item, str) for item in value.values())
        ):
            raise TypeError(f"{method} provenance lacks string-hash {key}")
        return value

    def required_record_map(
        record: dict[str, Any], key: str, method: str
    ) -> dict[str, dict[str, Any]]:
        value = record.get(key)
        if not isinstance(value, dict) or not value:
            raise TypeError(f"{method} provenance lacks nonempty {key}")
        if not all(isinstance(item, dict) for item in value.values()):
            raise TypeError(f"{method} provenance {key} contains a non-record")
        return value

    def verify_record(record: dict[str, Any], label: str) -> None:
        path = Path(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_bytes(path.read_bytes()) != record.get("sha256")
        ):
            raise ValueError(f"{label} digest is unavailable or changed")

    correction_records = {
        method: provenance[method].get("checkpoint_copy_correction")
        for method in METHODS
    }
    marker_present = {
        method: record is not None for method, record in correction_records.items()
    }
    if len(set(marker_present.values())) != 1:
        raise ValueError("only one paired continuation has a checkpoint-copy marker")
    correction_history: dict[str, Any]
    if not marker_present["Raw6"]:
        for method, directory in directories.items():
            revision = directory / "revisions" / "01-cpu-tensor-snapshot-copy"
            if revision.exists():
                raise ValueError(f"{method} correction revision lacks its marker")
        correction_history = {"kind": "fresh_corrected_runtime_history"}
    else:
        typed_records: dict[str, dict[str, Any]] = {}
        for method, directory in directories.items():
            record = correction_records[method]
            if not isinstance(record, dict):
                raise TypeError(f"{method} checkpoint-copy marker is not a record")
            verify_record(record, f"{method} checkpoint-copy correction")
            correction = json.loads(Path(record["path"]).read_text())
            if (
                correction.get("schema_version") != 1
                or correction.get("source_sha256_before")
                != CHECKPOINT_COPY_SOURCE_BEFORE
                or correction.get("source_sha256_after") != CHECKPOINT_COPY_SOURCE_AFTER
                or correction.get("adam_moment_reset_performed") is not False
                or correction.get("resume_pt_preserved_byte_for_byte") is not True
                or correction.get("affected_failure_path_executed_before_fix")
                is not False
            ):
                raise ValueError(
                    f"{method} checkpoint-copy correction contract differs"
                )
            before_record = correction.get("before_change_record")
            if not isinstance(before_record, dict):
                raise TypeError(f"{method} correction lacks before-change record")
            verify_record(before_record, f"{method} checkpoint-copy before record")
            before = json.loads(Path(before_record["path"]).read_text())
            run_key = "raw6" if method == "Raw6" else "raw6-s"
            frozen_before = (
                before.get("runs", {}).get(run_key, {}).get("frozen_before_change")
            )
            if not isinstance(frozen_before, dict) or not frozen_before:
                raise TypeError(f"{method} correction lacks retained inputs")
            for name, retained in frozen_before.items():
                if not isinstance(retained, dict):
                    raise TypeError(f"{method} retained input is not a record: {name}")
                verify_record(retained, f"{method} retained input {name}")
            old_source = frozen_before.get("35-continue-historical-adam.py")
            if not isinstance(old_source, dict):
                raise TypeError(f"{method} correction lacks initial archived source")
            if (
                before.get("old_source_sha256") != CHECKPOINT_COPY_SOURCE_BEFORE
                or old_source.get("sha256") != CHECKPOINT_COPY_SOURCE_BEFORE
            ):
                raise ValueError(f"{method} initial source revision differs")
            updated = correction.get("updated_archived_sources", {}).get(run_key)
            if not isinstance(updated, dict):
                raise TypeError(f"{method} correction lacks updated archived source")
            verify_record(updated, f"{method} updated archived source")
            current = directory / "sources" / "35-continue-historical-adam.py"
            if (
                Path(updated["path"]) != current
                or updated.get("sha256") != CHECKPOINT_COPY_SOURCE_AFTER
                or sha256_bytes(current.read_bytes()) != CHECKPOINT_COPY_SOURCE_AFTER
            ):
                raise ValueError(f"{method} corrected source revision differs")
            typed_records[method] = record
        if typed_records["Raw6"] != typed_records["Raw6-S"]:
            raise ValueError(
                "paired continuations use different checkpoint-copy records"
            )
        correction_history = {
            "kind": "migrated_checkpoint_copy_correction",
            "record": typed_records["Raw6"],
        }

    runtime_sources = {
        method: required_string_map(provenance[method], "sources", method)
        for method in METHODS
    }
    for method, sources in runtime_sources.items():
        for name, expected_hash in sources.items():
            path = directories[method] / "sources" / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != expected_hash:
                raise ValueError(f"{method} archived runtime source differs: {name}")
    if runtime_sources["Raw6"] != runtime_sources["Raw6-S"]:
        raise ValueError("archived continuation runtime source hashes differ")
    if (
        runtime_sources["Raw6"].get("35-continue-historical-adam.py")
        != CHECKPOINT_COPY_SOURCE_AFTER
    ):
        raise ValueError(
            "archived continuation source lacks corrected runtime revision"
        )
    frozen_inputs = {
        method: required_record_map(provenance[method], "frozen_inputs", method)
        for method in METHODS
    }
    fixture_names = ("fixture_volume", "fixture_skin", "fixture_summary")
    for method in METHODS:
        missing = set(fixture_names).difference(frozen_inputs[method])
        if missing:
            raise KeyError(f"{method} frozen inputs lack {sorted(missing)}")
        for name, record in frozen_inputs[method].items():
            verify_record(record, f"{method} frozen input {name}")
    for name in fixture_names:
        if frozen_inputs["Raw6"][name] != frozen_inputs["Raw6-S"][name]:
            raise ValueError(f"continuation fixture input differs: {name}")
    source_checkpoints: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        checkpoint = provenance[method].get("source_checkpoint")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{method} provenance lacks source_checkpoint")
        verify_record(checkpoint, f"{method} source checkpoint")
        source_checkpoints[method] = checkpoint
    for key in ("git_sha", "python", "torch", "cuda"):
        if key not in provenance["Raw6"] or key not in provenance["Raw6-S"]:
            raise KeyError(f"continuation runtime provenance lacks {key}")
        if provenance["Raw6"][key] != provenance["Raw6-S"][key]:
            raise ValueError(f"continuation runtime provenance differs: {key}")

    source_evidence = {
        method: json.loads((directory / "source.json").read_text())
        for method, directory in directories.items()
    }
    for method, evidence in source_evidence.items():
        if evidence.get("source_global_step") != 50:
            raise ValueError(
                f"{method} continuation did not start at shared source step 50"
            )
        receipt = evidence.get("source_step_solver_receipt", {})
        if not (
            receipt.get("forward", {}).get("success")
            and receipt.get("adjoint", {}).get("success")
        ):
            raise ValueError(f"{method} source step 50 receipt is not solver-valid")
        if evidence.get("checkpoint") != source_checkpoints[method]:
            raise ValueError(f"{method} source checkpoint evidence/provenance differs")
        if evidence.get("inputs") != frozen_inputs[method]:
            raise ValueError(f"{method} frozen input evidence/provenance differs")
    source_configs: dict[str, dict[str, Any]] = {}
    for method, evidence in source_evidence.items():
        source_record = evidence.get("inputs", {}).get("source_config", {})
        path = Path(source_record.get("path", ""))
        if not path.is_file() or sha256_bytes(path.read_bytes()) != source_record.get(
            "sha256"
        ):
            raise ValueError(
                f"{method} source configuration digest is unavailable or changed"
            )
        source_configs[method] = json.loads(path.read_text())
    source_differences = sorted(
        key
        for key in source_configs["Raw6"]
        if source_configs["Raw6"][key] != source_configs["Raw6-S"][key]
    )
    if source_differences != ["output_dir", "smoothness_weight"]:
        raise ValueError(
            f"interrupted treatment contracts differ unexpectedly: {source_differences}"
        )

    if source_checkpoints["Raw6"] == source_checkpoints["Raw6-S"]:
        raise ValueError(
            "paired continuations unexpectedly share a source checkpoint hash"
        )
    return {
        "differing_config_keys": differing,
        "shared_source_global_step": 50,
        "shared_moment_reset_required_on_completed_summary": True,
        "Raw6": configs["Raw6"],
        "Raw6-S": configs["Raw6-S"],
        "shared_runtime_source_hashes": runtime_sources["Raw6"],
        "checkpoint_copy_history": correction_history,
        "shared_fixture_material_inputs": {
            name: frozen_inputs["Raw6"][name] for name in fixture_names
        },
        "distinct_source_checkpoints": {
            method: source_checkpoints[method] for method in METHODS
        },
        "source_treatment_config_differences": source_differences,
    }


def termination_record(
    summary: dict[str, Any] | None,
    config: dict[str, Any],
    trace_rows: list[dict[str, str]],
) -> dict[str, Any]:
    declared_steps = int(config["steps"])
    trace_last_step = int(trace_rows[-1]["step"])
    if summary is None:
        return {
            "summary_present": False,
            "run_state": "ongoing_at_capture",
            "declared_local_steps": declared_steps,
            "last_evaluated_step_at_capture": trace_last_step,
            "fixed_budget_completed": False,
            "selected_best_step": None,
        }
    continuation = summary.get("continuation")
    if not isinstance(continuation, dict):
        raise TypeError("completed continuation summary lacks continuation metadata")
    if (
        continuation.get("source_global_step") != 50
        or continuation.get("optimizer_moments_at_source")
        != "explicitly reset for both methods"
        or continuation.get("uninterrupted_original_trajectory_claimed") is not False
    ):
        raise ValueError("completed continuation reset metadata differs")
    convergence = summary["convergence"]
    last_evaluated_step = int(convergence["last_evaluated_step"])
    best_step = int(convergence["best_valid_step"])
    if last_evaluated_step != trace_last_step:
        raise ValueError(
            "terminal summary last-evaluated step does not match captured trace"
        )
    if int(summary["best"]["step"]) != best_step:
        raise ValueError("terminal summary best-step fields disagree")
    fixed_budget_completed = last_evaluated_step == declared_steps
    return {
        "summary_present": True,
        "run_state": (
            "terminated_after_full_declared_budget"
            if fixed_budget_completed
            else "terminated_before_declared_budget"
        ),
        "termination_status": str(summary["status"]),
        "declared_local_steps": declared_steps,
        "last_evaluated_step": last_evaluated_step,
        "last_solver_valid_step": int(convergence["last_solver_valid_step"]),
        "selected_best_step": best_step,
        "fixed_budget_completed": fixed_budget_completed,
        "stationarity_claimed": bool(convergence["claimed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw6-dir", type=Path, required=True)
    parser.add_argument("--raw6-s-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--through-step", type=int)
    args = parser.parse_args()
    directories = {
        "Raw6": args.raw6_dir.resolve(),
        "Raw6-S": args.raw6_s_dir.resolve(),
    }

    captured: dict[str, dict[str, Any]] = {}
    for method, directory in directories.items():
        trace_path = directory / "trace.csv"
        receipts_path = directory / "solver-receipts.jsonl"
        summary_path = directory / "summary.json"
        trace_data = stable_read(trace_path)
        receipt_data = stable_read(receipts_path)
        summary_data = stable_read(summary_path) if summary_path.is_file() else None
        trace_lines, trace_rows = parse_csv(trace_data)
        receipt_lines, receipt_rows = parse_jsonl(receipt_data)
        assert_consecutive(trace_rows, f"{method} trace")
        assert_consecutive(receipt_rows, f"{method} receipts")
        captured[method] = {
            "directory": directory,
            "trace_path": trace_path,
            "receipts_path": receipts_path,
            "trace_data": trace_data,
            "receipt_data": receipt_data,
            "trace_lines": trace_lines,
            "receipt_lines": receipt_lines,
            "trace_rows": trace_rows,
            "receipt_rows": receipt_rows,
            "summary_path": summary_path,
            "summary_data": summary_data,
            "summary": None if summary_data is None else json.loads(summary_data),
        }

    latest_common = min(
        min(
            len(captured[method]["trace_rows"]),
            len(captured[method]["receipt_rows"]),
        )
        - 1
        for method in METHODS
    )
    through_step = latest_common if args.through_step is None else args.through_step
    if through_step < 0 or through_step > latest_common:
        raise ValueError(
            f"requested through-step {through_step} exceeds latest common {latest_common}"
        )

    configs = {
        method: json.loads((directory / "config.json").read_text())
        for method, directory in directories.items()
    }
    pair_contract = continuation_pair_contract(directories, configs)
    termination = {
        method: termination_record(
            captured[method]["summary"], configs[method], captured[method]["trace_rows"]
        )
        for method in METHODS
    }
    terminal_count = sum(
        int(record["summary_present"]) for record in termination.values()
    )
    full_budget_pair = terminal_count == len(METHODS) and all(
        record["fixed_budget_completed"] for record in termination.values()
    )
    prefix_truncated = through_step < latest_common
    if terminal_count == 0:
        status = "interim_runs_ongoing"
        figure_status = "INTERIM — RUNS ONGOING"
        watermark = "ONGOING"
    elif terminal_count < len(METHODS):
        status = "interim_mixed_terminal_and_ongoing_runs"
        figure_status = "INTERIM — MIXED RUN STATES"
        watermark = "ONGOING"
    elif prefix_truncated:
        status = "terminated_runs_truncated_prefix_snapshot"
        figure_status = f"TERMINATED RUNS — PREFIX THROUGH STEP {through_step}"
        watermark = "PREFIX"
    elif full_budget_pair:
        status = "terminal_full_declared_budget_pair_snapshot"
        figure_status = (
            f"FULL {configs['Raw6']['steps']}-LOCAL-STEP CONTINUATION BUDGETS"
        )
        watermark = None
    else:
        status = "terminal_pair_snapshot_before_full_declared_budget"
        figure_status = "TERMINATED RUNS — EARLY STOP"
        watermark = None

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for method, capture in captured.items():
        stem = method.lower().replace("-", "-")
        trace_snapshot = output / f"{stem}-trace-read-time.csv"
        receipt_snapshot = output / f"{stem}-receipts-read-time.jsonl"
        trace_prefix = output / f"{stem}-trace-prefix-step-{through_step:04d}.csv"
        receipt_prefix = (
            output / f"{stem}-receipts-prefix-step-{through_step:04d}.jsonl"
        )
        trace_snapshot.write_bytes(capture["trace_data"])
        receipt_snapshot.write_bytes(capture["receipt_data"])
        trace_prefix_data = "".join(capture["trace_lines"][: through_step + 2]).encode()
        receipt_prefix_data = "".join(
            capture["receipt_lines"][: through_step + 1]
        ).encode()
        trace_prefix.write_bytes(trace_prefix_data)
        receipt_prefix.write_bytes(receipt_prefix_data)
        summary_snapshot = None
        if capture["summary_data"] is not None:
            summary_path = output / f"{stem}-run-summary.json"
            summary_path.write_bytes(capture["summary_data"])
            summary_snapshot = {
                "source": {
                    "path": str(capture["summary_path"].resolve()),
                    "bytes": len(capture["summary_data"]),
                    "sha256": sha256_bytes(capture["summary_data"]),
                },
                "snapshot": digest(summary_path),
            }
        snapshots[method] = {
            "trace_source": source_receipt(
                capture["trace_path"],
                capture["trace_data"],
                len(capture["trace_rows"]),
            ),
            "receipt_source": source_receipt(
                capture["receipts_path"],
                capture["receipt_data"],
                len(capture["receipt_rows"]),
            ),
            "trace_read_time_snapshot": digest(trace_snapshot),
            "receipt_read_time_snapshot": digest(receipt_snapshot),
            "trace_prefix": digest(trace_prefix),
            "receipt_prefix": digest(receipt_prefix),
            "terminal_summary": summary_snapshot,
        }

    paired = []
    for step in range(through_step + 1):
        item: dict[str, Any] = {"step": step}
        for method in METHODS:
            trace = captured[method]["trace_rows"][step]
            receipt = captured[method]["receipt_rows"][step]
            trace_valid = bool_csv(trace["solver_valid"])
            receipt_valid = bool(
                receipt["forward"]["success"] and receipt["adjoint"]["success"]
            )
            if trace_valid != receipt_valid:
                raise ValueError(
                    f"{method} trace/receipt validity differs at step {step}"
                )
            item[method] = {
                "data_objective_mm2": float(trace["data_objective_mm2"]),
                "total_objective_mm2": float(trace["objective"]),
                "fit_rms_mm": float(trace["fit_rms_mm"]),
                "graph_variation_R": float(trace["smoothness"]),
                "smoothness_penalty_mm2": float(trace["smoothness_penalty_mm2"]),
                "gradient_rms": float(trace["gradient_rms"]),
                "forward_success": bool(receipt["forward"]["success"]),
                "adjoint_success": bool(receipt["adjoint"]["success"]),
                "solver_valid": receipt_valid,
                "forward_steps": int(receipt["forward"]["steps"]),
                "forward_result": receipt["forward"]["result"],
                "adjoint_result": receipt["adjoint"]["result"],
            }
        raw_variation = item["Raw6"]["graph_variation_R"]
        smooth_variation = item["Raw6-S"]["graph_variation_R"]
        item["paired_effect"] = {
            "raw6_s_minus_raw6_data_objective_mm2": (
                item["Raw6-S"]["data_objective_mm2"]
                - item["Raw6"]["data_objective_mm2"]
            ),
            "raw6_s_minus_raw6_fit_rms_mm": (
                item["Raw6-S"]["fit_rms_mm"] - item["Raw6"]["fit_rms_mm"]
            ),
            "raw6_s_minus_raw6_graph_variation_R": smooth_variation - raw_variation,
            "graph_variation_ratio_raw6_s_over_raw6": (
                None if raw_variation == 0.0 else smooth_variation / raw_variation
            ),
        }
        paired.append(item)

    figures = {
        "fit_rms_vs_step": plot_metric_vs_step(
            paired,
            output,
            key="fit_rms_mm",
            ylabel="uniform fit RMS (mm)",
            stem="fit-rms-vs-step",
            title="Matched historical Adam continuation fit",
            status_label=figure_status,
            watermark=watermark,
        ),
        "graph_variation_vs_step": plot_metric_vs_step(
            paired,
            output,
            key="graph_variation_R",
            ylabel="physical-Frobenius graph variation R (dimensionless)",
            stem="graph-variation-vs-step",
            title="Matched historical Adam continuation field variation",
            status_label=figure_status,
            watermark=watermark,
        ),
        "fit_vs_graph_variation": plot_tradeoff(
            paired,
            output,
            status_label=figure_status,
            watermark=watermark,
        ),
    }
    last = paired[-1]
    ratio = last["paired_effect"]["graph_variation_ratio_raw6_s_over_raw6"]
    if prefix_truncated:
        effect_scope = (
            "This is the same-continuation-local-step observation at an explicitly truncated "
            f"prefix through local step {through_step}; later saved local evaluations through "
            f"common local step {latest_common} are excluded."
        )
    elif terminal_count < len(METHODS):
        effect_scope = (
            "This is the latest common same-continuation-local-step observation while at least one "
            "run was still ongoing at capture."
        )
    elif full_budget_pair:
        effect_scope = (
            "This is the full 150-local-step-budget terminal same-continuation-local-step observation "
            f"at local step {through_step}, after the shared source step-50 reset."
        )
    else:
        effect_scope = (
            "This is the terminal same-continuation-local-step observation after at least one run "
            "stopped before its declared local budget."
        )
    summary = {
        "schema_version": 1,
        "status": status,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "through_common_continuation_local_step": through_step,
        "latest_common_continuation_local_step_available_at_read_time": latest_common,
        "capture_semantics": {
            "full_latest_common_prefix_captured": not prefix_truncated,
            "explicit_through_step_requested": args.through_step is not None,
            "prefix_truncated_below_latest_common": prefix_truncated,
            "terminal_summaries_present": terminal_count,
            "both_runs_completed_full_150_local_step_budget": full_budget_pair,
            "full_declared_budget_pair_captured": (
                full_budget_pair and not prefix_truncated and through_step == 150
            ),
            "figure_status_label": figure_status,
        },
        "scope": (
            "exact paired continuation-local trace and solver-receipt prefixes; no solve, resampling, "
            "smoothing, interpolation, extrapolation, or uninterrupted-trajectory claim"
        ),
        "pair_contract": {
            **pair_contract,
            "validation": (
                "continuations share source step 50, fixture/material inputs, archived runtime sources, and an explicit moment reset; "
                "they retain their own interrupted treatment source checkpoint and smoothness weight"
            ),
        },
        "source_snapshots": snapshots,
        "run_termination": termination,
        "metric_definitions": {
            "data_objective_mm2": (
                "uniform Cartesian-component displacement MSE multiplied by 1e6"
            ),
            "fit_rms_mm": (
                "Euclidean displacement residual RMS over target vertices in millimetres"
            ),
            "graph_variation_R": (
                "dimensionless finite-volume same-MuscleId physical symmetric-tensor "
                "Frobenius energy stored as smoothness by the runner"
            ),
            "field_variation_ratio": "Raw6-S graph_variation_R divided by Raw6 R at the same evaluated step",
        },
        "paired_steps": paired,
        "captured_common_step": last,
        "common_step_effect": {
            "graph_variation_ratio_raw6_s_over_raw6": ratio,
            "relative_graph_variation_reduction": None
            if ratio is None
            else 1.0 - ratio,
            "fit_rms_delta_mm_raw6_s_minus_raw6": last["paired_effect"][
                "raw6_s_minus_raw6_fit_rms_mm"
            ],
            "data_objective_delta_mm2_raw6_s_minus_raw6": last["paired_effect"][
                "raw6_s_minus_raw6_data_objective_mm2"
            ],
            "interpretation": (
                f"{effect_scope} A lower Raw6-S R at this step indicates a regularizer "
                "effect on the saved field. This artifact does not compare the runs' "
                "selected-best endpoints and does not establish convergence or the final "
                "fit-variation tradeoff."
            ),
        },
        "selected_best_endpoint_comparison": {
            "included": False,
            "scope": (
                "terminal summaries contribute only exact termination and selected-best "
                "step identifiers; selected-best endpoint metrics belong to the separate "
                "final physics comparison"
            ),
        },
        "figures": figures,
        "source_script": digest(Path(__file__)),
    }
    write_json(output / "summary.json", summary)
    shutil.copy2(Path(__file__), output / Path(__file__).name)


if __name__ == "__main__":
    main()
