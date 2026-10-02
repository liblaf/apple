"""Write the final MouthOpen report from terminal runs and verified receipts."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
DOCS = GROUP / "docs"
REPORT = DOCS / "67-mouthopen-results.md"
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    forward: Path = Path("49-forward-contact-off")
    interrupted_fit: Path = Path("55-mouthopen-fit")
    fit: Path = Path("56-mouthopen-fit-reuse")
    analysis: Path = Path("60-mouthopen-fit-analysis-002/analysis.json")
    render: Path = Path("61-mouthopen-fit")
    fixture: Path = Path("30-pruned-fixture")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def check_record(name: str, item: dict[str, Any]) -> tuple[Path, str]:
    path = Path(item["path"])
    actual = sha256(path)
    if actual != item["sha256"]:
        raise ValueError(f"{name} receipt mismatch: {path}")
    return path, actual


def check_records(records: dict[str, Any], label: str) -> list[tuple[str, Path, str]]:
    result = []
    for name, item in records.items():
        path, digest = check_record(f"{label}/{name}", item)
        result.append((name, path, digest))
    return result


def check_source_manifest(path: Path) -> tuple[int, str]:
    manifest = json.loads(path.read_text())
    if not manifest:
        raise ValueError(f"source manifest is empty: {path}")
    for module, item in manifest.items():
        check_record(f"source/{module}", item)
    return len(manifest), sha256(path)


def load_log(stage: str, markers: tuple[str, ...]) -> dict[str, str | None]:
    candidates = []
    for path in sorted((GROUP / "logs").glob(f"{stage}-*.log")):
        text = path.read_text(errors="replace")
        if all(marker in text for marker in markers) and "cherries/end_time" in text:
            candidates.append((path, text))
    if not candidates:
        raise FileNotFoundError(
            f"no completed {stage} Cherries log matched {markers}; report withheld"
        )
    path, text = max(candidates, key=lambda item: item[0].stat().st_mtime_ns)

    def field(name: str) -> str | None:
        pattern = re.compile(rf"\bcherries/{re.escape(name)}\s*:\s*(.*)$")
        values = [
            match.group(1).strip()
            for line in text.splitlines()
            if (match := pattern.search(line))
        ]
        return values[-1] if values else None

    url = field("comet/url")
    if url is None:
        match = re.search(r"https://www\.comet\.com/[^\s]+", text)
        url = match.group(0) if match else None
    return {
        "path": str(path),
        "sha256": sha256(path),
        "command": field("cmd"),
        "url": url,
        "start_time": field("start_time"),
        "end_time": field("end_time"),
        "git_sha": field("git/sha"),
        "name": field("name"),
    }


def md_link(path: str | Path, label: str) -> str:
    resolved = Path(path).resolve()
    relative = Path(__import__("os").path.relpath(resolved, DOCS.resolve())).as_posix()
    return f"[{label}]({relative})"


def f(value: Any, digits: int = 4) -> str:
    if value is None or value == "n/a":
        return "n/a"
    return f"{float(value):.{digits}f}"


def file_receipt_table(rows: list[tuple[str, Path, str]]) -> str:
    lines = ["| Artifact | SHA-256 |", "| --- | --- |"]
    for name, _path, digest in rows:
        lines.append(f"| {name} | `{digest}` |")
    return "\n".join(lines)


def run_table(logs: dict[str, dict[str, str | None]]) -> str:
    lines = [
        "| Stage | Recorded command | Cherries/Comet URL | Run times | Git SHA | Log SHA-256 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, item in logs.items():
        url = (
            f"[Comet]({item['url']})"
            if item["url"]
            else "No Comet URL recorded in this log"
        )
        times = f"{item['start_time']} to {item['end_time']}"
        lines.append(
            f"| {label} | `{item['command'] or 'command not recorded'}` | {url} | `{times}` | "
            f"`{item['git_sha'] or 'n/a'}` | `{item['sha256']}` |"
        )
    return "\n".join(lines)


def main(cfg: Config) -> None:
    forward_dir = cherries.input(cfg.forward)
    old_fit_dir = cherries.input(cfg.interrupted_fit)
    fit_dir = cherries.input(cfg.fit)
    analysis_path = cherries.input(cfg.analysis)
    render_dir = cherries.input(cfg.render)
    fixture_dir = cherries.input(cfg.fixture)

    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite final report: {REPORT}")
    forward_summary = json.loads((forward_dir / "summary.json").read_text())
    old_fit_summary = json.loads((old_fit_dir / "summary.json").read_text())
    fit_summary = json.loads((fit_dir / "summary.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    render_manifest = json.loads((render_dir / "manifest.json").read_text())
    fixture_summary = json.loads((fixture_dir / "summary.json").read_text())
    pose_report = json.loads((GROUP / "data/10-mandible/pose.json").read_text())

    if fit_summary.get("status") in {"running", "initializing", None}:
        raise RuntimeError("fit is not terminal; final report withheld")
    if analysis.get("fit") is None or not analysis["fit"].get(
        "solver_receipts_within_declared_gates"
    ):
        raise ValueError("terminal fit is absent or the CPU solver audit did not pass")
    budget = analysis["fit"]["budget"]
    if int(budget["attempted_updates"]) != 200:
        raise ValueError(f"expected 200 attempted updates; got {budget}")
    if int(budget["optimizer_updates"]) + int(budget["skipped_updates"]) != 200:
        raise ValueError(f"fit update counts do not sum to 200: {budget}")
    if render_manifest.get("fit_status") != fit_summary.get("status"):
        raise ValueError("render manifest does not correspond to the terminal fit")

    forward_path = forward_dir / "final.npz"
    fit_checkpoint = fit_dir / "last.npz"
    forward_sha = sha256(forward_path)
    fit_sha = sha256(fit_checkpoint)
    if forward_sha != forward_summary["final_checkpoint"]["sha256"]:
        raise ValueError("forward checkpoint receipt mismatch")
    if fit_sha != fit_summary["final_checkpoint"]["sha256"]:
        raise ValueError("fit checkpoint receipt mismatch")
    if analysis["forward"]["checkpoint"]["sha256"] != forward_sha:
        raise ValueError("analysis audited a different forward checkpoint")
    if analysis["fit"]["checkpoint"]["sha256"] != fit_sha:
        raise ValueError("analysis audited a different fit checkpoint")

    for name, item in render_manifest.get("inputs", {}).items():
        check_record(f"render input {name}", item)
    for item in render_manifest.get("sources", []):
        check_record("render source", item)
    for key in ("figure", "preview"):
        check_record(f"render {key}", render_manifest[key])

    check_records(forward_summary["inputs"], "forward")
    check_records(fit_summary["inputs"], "fit")
    forward_source_path = forward_dir / "source-manifest.json"
    fit_source_path = fit_dir / "source-manifest.json"
    forward_source_count, _forward_source_sha = check_source_manifest(
        forward_source_path
    )
    fit_source_count, _fit_source_sha = check_source_manifest(fit_source_path)
    if not analysis["forward"]["source_snapshot"]["all_match"]:
        raise ValueError("forward source snapshot failed its audit")
    if not analysis["fit"]["source_snapshot"]["all_match"]:
        raise ValueError("fit source snapshot failed its audit")

    gradient_balance_path = fit_dir / "gradient-balance.json"
    gradient_components_path = fit_dir / "gradient-components.npz"
    gradient_balance = json.loads(gradient_balance_path.read_text())
    if gradient_balance["checkpoint"]["sha256"] != fit_sha:
        raise ValueError("gradient-balance diagnostic references another checkpoint")
    if gradient_balance["components"]["sha256"] != sha256(gradient_components_path):
        raise ValueError("gradient-component receipt mismatch")
    ratio = float(analysis["fit"]["gradient_ratio"]["smoothness_to_l2"])
    np.testing.assert_allclose(
        ratio,
        float(gradient_balance["smoothness_to_l2_gradient_ratio"]),
        rtol=2e-10,
    )

    receipts_path = fit_dir / "solver-receipts.jsonl"
    solver_receipts = [
        json.loads(line) for line in receipts_path.read_text().splitlines() if line
    ]
    if not solver_receipts:
        raise ValueError("fit contains no saved primal/adjoint solver receipts")
    primal = [
        float(item["forward"]["physical_residual"]["absolute"])
        for item in solver_receipts
    ]
    adjoint = [
        float(item["adjoint"]["relative_residual"])
        for item in solver_receipts
        if not item["adjoint"].get("zero_rhs", False)
    ]
    if not adjoint:
        raise ValueError("fit has no nonzero-RHS adjoint receipts")

    forward_metrics = analysis["forward"]["recomputed"]
    fit_metrics = analysis["fit"]["final_metrics"]
    if analysis["fit"]["roughness"]["identity_verified"] is not True:
        raise ValueError("spectral roughness identity was not verified")
    roughness = analysis["fit"]["roughness"]
    pose = pose_report["full_rigid_chin_fit"]
    fit_config = fit_summary["config"]
    material = fit_summary["material_spec"]
    fit_rest_volume = analysis["fit"]["geometry"]["inverted_rest_volume_fraction"]
    forward_rest_volume = forward_metrics["inverted_rest_volume_fraction"]
    forward_intersects = bool(analysis["forward"]["summary_final"]["has_intersections"])
    fit_intersects = analysis["fit"]["geometry"]["has_intersections"]
    displacement_drift_um = (
        float(gradient_balance["displacement_change_from_saved_m"]) * 1e6
    )
    if forward_intersects:
        intersection_sentence = "The complete FEM boundary audit detected self-intersections at the zero-activation forward endpoint."
    else:
        intersection_sentence = "The complete FEM boundary audit reported no self-intersections at the zero-activation forward endpoint."
    if fit_intersects is None:
        intersection_sentence += " A separate complete-boundary intersection result for the fitted endpoint was not recorded."
    else:
        intersection_sentence += " The fitted endpoint " + (
            "also has detected self-intersections."
            if fit_intersects
            else "has no detected self-intersections."
        )

    logs = {
        "49 zero-activation continuation": load_log(
            "49", ("49-forward-contact-off.py",)
        ),
        "56 activation fit": load_log("56", ("56-mouthopen-fit-reuse",)),
        "60 CPU audit": load_log("60", ("56-mouthopen-fit-reuse",)),
        "61 final renderer": load_log(
            "61", ("56-mouthopen-fit-reuse", "61-mouthopen-fit")
        ),
    }
    direction_receipt = render_manifest.get("direction_review")
    if not direction_receipt:
        raise ValueError("final render is missing its sampled direction view")
    for key in ("figure", "preview", "panel"):
        check_record(f"sampled direction {key}", direction_receipt[key])
    render_receipt_rows = [
        ("49 final checkpoint", forward_path, forward_sha),
        ("56 final checkpoint", fit_checkpoint, fit_sha),
        (
            "gradient-balance diagnostic",
            gradient_balance_path,
            sha256(gradient_balance_path),
        ),
        ("independent CPU analysis", analysis_path, sha256(analysis_path)),
        (
            "comparison figure",
            Path(render_manifest["figure"]["path"]),
            render_manifest["figure"]["sha256"],
        ),
        (
            "sampled direction preview",
            Path(direction_receipt["preview"]["path"]),
            direction_receipt["preview"]["sha256"],
        ),
    ]

    figure_preview = Path(render_manifest["preview"]["path"])
    figure_full = Path(render_manifest["figure"]["path"])
    direction_preview = Path(direction_receipt["preview"]["path"])
    old_status = old_fit_summary.get("status", "unknown")
    old_budget = {
        "attempts": old_fit_summary.get("attempted_updates", "n/a"),
        "updates": old_fit_summary.get("optimizer_updates", "n/a"),
        "skips": old_fit_summary.get("skipped_updates", "n/a"),
    }
    if int(fixture_summary["counts"]["removed_allfixed_cells"]) != 2249:
        raise ValueError(
            "pruned fixture no longer matches the documented cell deletion"
        )
    if not analysis["fit"]["initial_zero_activation_and_parent_seed_verified"]:
        raise ValueError("fit initialization was not verified")

    lines = [
        "# MouthOpen full-pose activation fit",
        "",
        (
            f"The MouthOpen fit used full-S PSD6 activation with smoothness weight `η={fit_config['smooth_weight']:.2g}` "
            "(10x the earlier `7.2e-7` baseline). It completed the declared 200-attempt budget with "
            f"{budget['optimizer_updates']} accepted updates. Reference-area-weighted position RMS changed from "
            f"**{f(forward_metrics['fit_rms_mm'], 3)} mm** to **{f(fit_metrics['fit_rms_mm'], 3)} mm**; "
            f"normal-angle RMS changed from **{f(forward_metrics['normal_angle_rms_deg'], 3)}°** to "
            f"**{f(fit_metrics['normal_angle_rms_deg'], 3)}°**. The run budget is not an optimizer-convergence claim."
        ),
        "",
        f"![Neutral, transferred target, contact-off continuation, fitted shape, and fitted activation]({md_link(figure_preview, 'preview').split('](')[1][:-1]})",
        "",
        f"[Full-resolution figure]({md_link(figure_full, 'figure').split('](')[1][:-1]}) · "
        f"[Sampled principal-direction view]({md_link(direction_preview, 'sampled direction view').split('](')[1][:-1]}) · "
        f"{md_link(analysis_path, 'Independent CPU analysis')} · "
        f"[Render manifest](../data/61-mouthopen-fit/manifest.json) · "
        f"[Gradient-balance diagnostic](../data/56-mouthopen-fit-reuse/gradient-balance.json) · "
        f"[Zero-activation forward report](49-contact-off-forward-results.md)",
        "",
        "## Fit result",
        "",
        "| Metric | Zero activation, full jaw | Fitted activation |",
        "| --- | ---: | ---: |",
        f"| Position RMS (reference-area weighted) | {f(forward_metrics['fit_rms_mm'], 6)} mm | {f(fit_metrics['fit_rms_mm'], 6)} mm |",
        f"| Surface-normal angle RMS (reference-area weighted) | {f(forward_metrics['normal_angle_rms_deg'], 6)}° | {f(fit_metrics['normal_angle_rms_deg'], 6)}° |",
        f"| Inverted tetrahedra | {forward_metrics['inverted_cells']} | {fit_metrics['inverted_all_cells']} |",
        f"| Minimum J = det(F) | {f(forward_metrics['minimum_J'], 6)} | {f(fit_metrics['detF_min'], 6)} |",
        f"| Inverted rest-volume fraction | {f(forward_rest_volume, 8)} | {f(fit_rest_volume, 8)} |",
        "",
        (
            f"Stage 56 made **{budget['attempted_updates']} attempts**, accepted **{budget['optimizer_updates']}** "
            f"and skipped **{budget['skipped_updates']}**; status: `{fit_summary['status']}`. Stage 55 was interrupted "
            f"after {old_budget['attempts']} attempts ({old_budget['updates']} accepted, {old_budget['skips']} skipped; "
            f"status `{old_status}`). Stage 56 restarted from the same verified zero-S, full-jaw endpoint with fresh Adam "
            "and the reuse-only Newton search-shift policy."
        ),
        "",
        f"The full-S smoothness-to-L2 gradient ratio is **{ratio:.6g}**, measured with the dual effective-volume-weighted norm over the symmetric activation tensors: `η ||∇S R|| / ||∇S L2||`. The normal-loss gradient is excluded. The endpoint diagnostic's independent forward solve changed saved displacement by at most **{displacement_drift_um:.3f} μm** componentwise, so this ratio uses the saved activation tensor and that diagnostic displacement.",
        "",
        f"The graph roughness is `{roughness['direct_frobenius']:.6g}` and decomposes exactly into eigenvalue-amplitude roughness `{roughness['eigenvalue_amplitude']:.6g}` plus orientation roughness `{roughness['orientation']:.6g}`. The audit verified this identity for the saved PSD tensor field.",
        "",
        "Visual inspection of the comparison and sampled direction preview shows coherent patches alongside local directional variation around the mouth and chin. This MouthOpen trial has not produced a uniformly ordered activation field. There is no MouthOpen 1x control in this trial, so it does not isolate the effect of the 10x smoothness coefficient. The different target and full PSD6 parameterization also prevent a controlled quantitative comparison with the earlier fixed-axis Smile sweep.",
        "",
        "## Numerical and geometry diagnostics",
        "",
        (
            f"The {len(solver_receipts)} saved successful primal/adjoint evaluations had physical free-force residuals "
            f"from `{min(primal):.3g}` to `{max(primal):.3g}` (absolute tolerance `{fit_config['force_atol']:.1e}`) "
            f"and nonzero-RHS relative adjoint residuals from `{min(adjoint):.3g}` to `{max(adjoint):.3g}` "
            f"(tolerance `{fit_config['adjoint_rtol']:.1e}`). Skipped proposals have no successful solve receipt."
        ),
        "",
        (
            f"{intersection_sentence} The zero-activation forward state has {forward_metrics['inverted_cells']} inverted cells "
            f"(minimum `J={forward_metrics['minimum_J']:.4g}`, inverted rest-volume fraction `{forward_rest_volume:.3g}`); "
            f"the fitted state has {fit_metrics['inverted_all_cells']} (minimum `J={fit_metrics['detF_min']:.4g}`, "
            f"fraction `{fit_rest_volume:.3g}`), including {fit_metrics['inverted_active_cells']} active and "
            f"{fit_metrics['inverted_inactive_cells']} inactive cells. This exploratory contact-off fit used a derived mesh with "
            f"{fixture_summary['counts']['removed_allfixed_cells']:,} tetrahedra removed because all four vertices were fixed. "
            "There were no contact forces, separate bone obstacles, or containment check, so the result does not establish "
            "mechanical validity."
        ),
        "",
        "## Pose, mesh, and material assumptions",
        "",
        (
            f"The target is a transferred MouthOpen blendshape. The jaw pose is an area-weighted rigid fit to a 27-vertex chin patch: "
            f"{pose['fit_rotation_degrees']:.4f}° rotation, {pose['translation_norm_m'] * 1000:.4f} mm translation norm, "
            f"and {pose['weighted_rms_after_m'] * 1000:.4f} mm chin RMS. This is a geometric seed, not measured bone motion. "
            "The surviving original `IsFixed` constraints and material model were retained."
        ),
        "",
        (
            "Activation started from exactly zero S on the verified full-jaw endpoint, with fresh Adam and PSD6 projection. "
            f"The material model is `{material['muscle_model']}`; skin membrane energy is `{material['skin_energy']}`, "
            f"muscle E is {material['muscle_E_MPa']} MPa, fat E is "
            f"{material['fat_E_MPa']} MPa, and aponeurosis E is {material['aponeurosis_E_MPa']} MPa (all with the recorded nu values). "
            "The skin membrane energy was disabled."
        ),
        "",
        "## Reproducibility and receipts",
        "",
        "The first CPU audit stopped on a bitwise tensor-reconstruction comparison whose maximum difference was `4.44e-16`. The corrected audit allows a few floating-point ULPs for that reconstruction and retains the exact recorded solver gates. The numerical fit was not repeated; the failed pipeline and [postprocessing recovery receipt](../data/69-finalization/recovery.json) are preserved.",
        "",
        f"Working directory: `{GROUP}`. Commands and run metadata below are read from completed Cherries logs; Comet links are included only when the log records a URL. The relevant logs and their SHA-256 digests are listed in the run table.",
        "",
        run_table(logs),
        "",
        f"The independent audit verified {forward_source_count} forward and {fit_source_count} fit source-module receipts; those entries may reference duplicate files. Exact declared-input and frozen-source hashes remain in the stage summaries and source manifests.",
        "",
        "[Stage 49 source manifest](../data/49-forward-contact-off/source-manifest.json) · "
        "[Stage 56 source manifest](../data/56-mouthopen-fit-reuse/source-manifest.json) · "
        "[Stage 56 summary and declared-input receipts](../data/56-mouthopen-fit-reuse/summary.json)",
        "",
        file_receipt_table(render_receipt_rows),
        "",
        "The forward run, fit analysis, gradient diagnostic, and renderer manifests retain the full inputs, solver receipts, source snapshots, and image hashes.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    cherries.log_output(REPORT)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
