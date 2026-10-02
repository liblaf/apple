"""Build the immutable final renderer manifest from the reviewed endpoint inventory.

This is deliberately a CPU-only manifest constructor. It accepts only saved VTU
states. For the matched historical Adam runs, history frames must come from the
checkpoint exporter, which cross-checks each payload against its solver receipt
and omits frames without both successful forward and adjoint solves.
"""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = ROOT / "docs" / "58-final-viewer-inventory.json"
DEFAULT_OUTPUT = ROOT / "docs" / "60-final-diagnosis-render-manifest.json"
CONTINUATION_DISPLAY_NAMES = {
    "historical-adam-raw6": "Raw6",
    "historical-adam-raw6-smooth": "Raw6-S",
}


class ManifestError(RuntimeError):
    """The final viewer cannot be made from the saved endpoint inventory."""


def resolve(value: str, base: Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else base.parent / path
    if not path.is_file():
        raise ManifestError(f"required saved artifact is absent: {path}")
    return path.resolve()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, start=output.parent.resolve())).as_posix()


def summary_history(
    case: dict[str, Any], inventory: Path, output: Path
) -> dict[str, Any]:
    contract = case["history_contract"]
    summary_path = resolve(contract["summary"], inventory)
    directory = resolve(f"{contract['directory']}/summary.json", inventory).parent
    summary = json.loads(summary_path.read_text())
    snapshots = summary.get("snapshots")
    final = summary.get("final")
    status = summary.get("status")
    if (
        not isinstance(snapshots, list)
        or not all(isinstance(step, int) for step in snapshots)
        or not isinstance(final, dict)
        or not isinstance(final.get("step"), int)
        or not isinstance(status, str)
    ):
        raise ManifestError(f"invalid accepted-snapshot summary: {summary_path}")
    trace_path = directory / "trace.csv"
    if not trace_path.is_file():
        raise ManifestError(f"accepted-snapshot trace is absent: {trace_path}")
    trace_steps = {int(row["step"]) for row in csv.DictReader(trace_path.open())}
    frames: list[str] = []
    labels: list[str] = []
    for step in snapshots:
        if step not in trace_steps:
            raise ManifestError(f"snapshot step {step} absent from trace: {trace_path}")
        frame = directory / f"step-{step:04d}.vtu"
        if not frame.is_file():
            raise ManifestError(f"accepted snapshot has no saved VTU: {frame}")
        frames.append(relative(frame, output))
        labels.append(f"accepted saved optimization step {step}")
    endpoint = resolve(case["endpoint_vtu"], inventory)
    if int(final["step"]) not in trace_steps:
        raise ManifestError(f"final step is absent from trace: {summary_path}")
    frames.append(relative(endpoint, output))
    labels.append(
        f"accepted final iteration step {final['step']} · termination {status}"
    )
    return {
        "frames": frames,
        "steps": [*snapshots, int(final["step"])],
        "labels": labels,
        "fps": 2,
    }


def digest_matches(path: Path, record: dict[str, Any], context: str) -> None:
    if (
        record.get("sha256") != sha256(path)
        or record.get("bytes") != path.stat().st_size
    ):
        raise ManifestError(f"digest mismatch for {context}: {path}")


def endpoint_steps(summary: dict[str, Any], *, local: bool) -> tuple[int, int]:
    """Read the frozen continuation schema's explicit best and terminal steps."""
    convergence = summary.get("convergence")
    if not isinstance(convergence, dict):
        raise ManifestError("Adam summary lacks convergence metadata")
    best, last = (
        convergence.get("best_valid_step"),
        convergence.get("last_evaluated_step"),
    )
    if not isinstance(best, int) or not isinstance(last, int):
        raise ManifestError("Adam convergence has invalid best/terminal step fields")
    if not local:
        return best, last

    continuation = summary.get("continuation")
    if not isinstance(continuation, dict):
        raise ManifestError("continuation summary lacks reset metadata")
    if (
        continuation.get("source_global_step") != 50
        or continuation.get("optimizer_moments_at_source")
        != "explicitly reset for both methods"
        or continuation.get("uninterrupted_original_trajectory_claimed") is not False
        or continuation.get("local_step_range") != [0, last]
        or continuation.get("nominal_global_step_range") != [50, 50 + last]
    ):
        raise ManifestError("continuation metadata differs from the frozen contract")
    bootstrap = continuation.get("bootstrap_re_equilibration")
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("source_seed_reused") is not True
        or bootstrap.get("forward", {}).get("success") is not True
        or bootstrap.get("adjoint", {}).get("success") is not True
    ):
        raise ManifestError("continuation bootstrap is not solver-valid")
    return best, last


def adam_history(case: dict[str, Any], inventory: Path, output: Path) -> dict[str, Any]:
    selection = case["endpoint_selection"]
    contract = case["history_contract"]
    summary_path = resolve(selection["summary"], inventory)
    export_manifest_path = resolve(contract["manifest"], inventory)
    summary = json.loads(summary_path.read_text())
    exported = json.loads(export_manifest_path.read_text())
    local_steps = bool(contract.get("continuation_local_steps", False))
    best_step, last_step = endpoint_steps(summary, local=local_steps)
    if exported.get("status") != "completed_run_export":
        raise ManifestError(
            f"Adam checkpoint export is incomplete: {export_manifest_path}"
        )
    run_summary = exported.get("run", {}).get("summary")
    if not isinstance(run_summary, dict):
        raise ManifestError(
            f"checkpoint export lacks run summary digest: {export_manifest_path}"
        )
    digest_matches(summary_path, run_summary, "Adam summary")
    endpoint = resolve(case["endpoint_vtu"], inventory)
    final_best = exported.get("final_best_vtu")
    if not isinstance(final_best, dict):
        raise ManifestError(
            f"checkpoint export lacks final best endpoint: {export_manifest_path}"
        )
    digest_matches(endpoint, final_best, "Adam best endpoint")
    final_verification = exported.get("final_state_verification")
    if (
        not isinstance(final_verification, dict)
        or final_verification.get("status") != "verified_completed_best_state"
    ):
        raise ManifestError("checkpoint export lacks verified completed final state")
    if final_verification.get("summary_best_valid_step") != best_step:
        raise ManifestError("final-state verification best step disagrees with summary")
    for key in ("final_npz", "final_vtu"):
        record = final_verification.get(key)
        if not isinstance(record, dict):
            raise ManifestError(f"final-state verification lacks {key}")
    digest_matches(
        summary_path.parent / "final.npz",
        final_verification["final_npz"],
        "Adam final NPZ",
    )
    digest_matches(
        endpoint, final_verification["final_vtu"], "Adam final VTU verification"
    )
    receipt = final_verification.get("solver_receipt")
    if not isinstance(receipt, dict) or not (
        receipt.get("forward", {}).get("success")
        and receipt.get("adjoint", {}).get("success")
    ):
        raise ManifestError("final-state verification lacks a solver-valid receipt")
    checks = final_verification.get("verification")
    exact = (
        "points_max_abs_error",
        "rest_position_max_abs_error",
        "displacement_max_abs_error",
        "activation_matrix_max_abs_error",
    )
    if (
        not isinstance(checks, dict)
        or not checks.get("activation_mask_exact")
        or any(checks.get(key) != 0.0 for key in exact)
    ):
        raise ManifestError("final-state geometry verification is not exact")
    if checks.get("packed_q_reconstructed_from_final_npz") is not True:
        raise ManifestError(
            "final-state verification lacks reconstructed packed-q proof"
        )
    exporter_source = resolve(str(contract["exporter_source"]), inventory)
    exporter_record = exported.get("exporter")
    if not isinstance(exporter_record, dict):
        raise ManifestError("checkpoint export lacks live exporter digest")
    digest_matches(exporter_source, exporter_record, "checkpoint exporter source")
    if exporter_record.get("sha256") != contract.get("exporter_sha256"):
        raise ManifestError(
            "checkpoint exporter digest disagrees with inventory contract"
        )
    export_dir = export_manifest_path.parent
    frames: list[str] = []
    steps: list[int] = []
    labels: list[str] = []
    for frame in exported.get("frames", []):
        if not isinstance(frame, dict) or not isinstance(frame.get("step"), int):
            raise ManifestError(f"invalid Adam frame record: {export_manifest_path}")
        receipt = frame.get("solver_receipt")
        if not isinstance(receipt, dict) or not (
            receipt.get("forward", {}).get("success")
            and receipt.get("adjoint", {}).get("success")
        ):
            raise ManifestError(f"non-solver-valid Adam frame was exported: {frame}")
        frame_path = export_dir / str(frame.get("file"))
        if not frame_path.is_file():
            raise ManifestError(f"exported Adam frame is absent: {frame_path}")
        output_digest = frame.get("output")
        if not isinstance(output_digest, dict):
            raise ManifestError(f"exported Adam frame lacks digest: {frame_path}")
        digest_matches(frame_path, output_digest, "Adam checkpoint frame")
        frames.append(relative(frame_path, output))
        steps.append(frame["step"])
        step_label = (
            f"saved solver-valid continuation local checkpoint step {frame['step']}"
            if local_steps
            else f"saved solver-valid optimization checkpoint step {frame['step']}"
        )
        labels.append(step_label)
    if not frames:
        raise ManifestError(f"no solver-valid Adam frames: {export_manifest_path}")
    frames.append(relative(endpoint, output))
    steps.append(best_step)
    if local_steps:
        labels.append(
            f"best solver-valid continuation local step {best_step} · terminal local step {last_step}"
        )
    else:
        labels.append(
            f"best solver-valid endpoint step {best_step} · final iteration step {last_step}"
        )
    return {"frames": frames, "steps": steps, "labels": labels, "fps": 2}


def promote_ready_cases(document: dict[str, Any], inventory: Path) -> None:
    """Promote only completed saved inverse endpoints with their declared receipts."""
    for case in document["cases"]:
        if case.get("renderable_now"):
            continue
        if case.get("id") == "raw6-no-skin":
            summary_path = resolve(case["status_source"], inventory)
            endpoint = resolve(case["endpoint_vtu"], inventory)
            summary = json.loads(summary_path.read_text())
            status, final = summary.get("status"), summary.get("final")
            if not isinstance(status, str) or not isinstance(final, dict):
                raise ManifestError(f"invalid Raw6 endpoint summary: {summary_path}")
            case["availability"] = (
                f"saved final.vtu and summary.json; status {status} at step {final.get('step')}"
            )
            case["label"] = f"Raw6 no-skin inverse · saved {status} endpoint"
            case["renderable_now"] = True
            continue
        selection = case.get("endpoint_selection")
        contract = case.get("history_contract")
        if not isinstance(selection, dict) or not isinstance(contract, dict):
            continue
        summary_path = resolve(selection["summary"], inventory)
        export_path = resolve(contract["manifest"], inventory)
        endpoint = summary_path.parent / "final.vtu"
        if not endpoint.is_file():
            raise ManifestError(
                f"Adam summary exists but final endpoint is absent: {endpoint}"
            )
        summary = json.loads(summary_path.read_text())
        local_steps = bool(contract.get("continuation_local_steps", False))
        best, last = endpoint_steps(summary, local=local_steps)
        exported = json.loads(export_path.read_text())
        if exported.get("status") != "completed_run_export":
            raise ManifestError(f"Adam checkpoint export is incomplete: {export_path}")
        case["endpoint_vtu"] = relative(endpoint, inventory)
        if local_steps:
            method = CONTINUATION_DISPLAY_NAMES.get(case.get("id"))
            if method is None:
                raise ManifestError(
                    f"continuation case has no explicit display mapping: {case.get('id')}"
                )
            case["availability"] = (
                f"saved best solver-valid continuation local step {best}; terminal local step {last}; "
                f"status {summary.get('status')}"
            )
            case["label"] = (
                f"{method} · continuation best local {best} · terminal {last}"
            )
        else:
            case["availability"] = (
                f"saved best solver-valid endpoint step {best}; final evaluated iteration step {last}; "
                f"status {summary.get('status')}"
            )
            case["label"] = (
                f"{case['id']} · best solver-valid step {best} · final iteration {last}"
            )
        case["renderable_now"] = True
    document["counts"]["renderable_now"] = sum(
        bool(case.get("renderable_now")) for case in document["cases"]
    )


def case_manifest(
    case: dict[str, Any], inventory: Path, output: Path
) -> dict[str, Any]:
    for key in ("reference_vtu", "endpoint_vtu"):
        resolve(case[key], inventory)
    result = {
        key: value
        for key, value in case.items()
        if key
        not in {
            "category",
            "availability",
            "status_source",
            "renderable_now",
            "history_contract",
            "endpoint_selection",
        }
    }
    contract = case.get("history_contract")
    if contract is not None:
        kind = contract.get("kind")
        if kind == "accepted_saved_vtu_snapshots":
            result["history"] = summary_history(case, inventory, output)
        elif kind == "exported_solver_valid_checkpoints":
            result["history"] = adam_history(case, inventory, output)
        else:
            raise ManifestError(f"unknown history contract: {kind!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--promote-ready",
        action="store_true",
        help="record only completed inverse endpoints whose saved summaries/export receipts validate",
    )
    args = parser.parse_args()
    inventory, output = args.inventory.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    document = json.loads(inventory.read_text())
    if args.promote_ready:
        promote_ready_cases(document, inventory)
        inventory.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != document.get("counts", {}).get(
        "total_cases"
    ):
        raise ManifestError("inventory case count is inconsistent")
    pending = [case.get("id") for case in cases if not case.get("renderable_now")]
    if pending:
        raise ManifestError(
            f"final 27-case viewer awaits: {', '.join(map(str, pending))}"
        )
    if len(cases) != 27:
        raise ManifestError(f"final viewer requires 27 cases, got {len(cases)}")
    manifest = {
        "title": "Face actuation diagnosis · final saved endpoint comparison",
        "inventory": relative(inventory, output),
        "cases": [case_manifest(case, inventory, output) for case in cases],
        "history_interpretation": "Optimization history is a sequence of exact saved solver states, not physical time. Historical Adam best solver-valid endpoints are labeled separately from their final evaluated iterations.",
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
