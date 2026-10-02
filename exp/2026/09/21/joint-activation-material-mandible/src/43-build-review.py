"""Build a concise mobile review page with detailed scientific image galleries."""

from __future__ import annotations

import html
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pydantic_settings as ps
from joint_common import GROUP, ProfileJoint, sha256, write_json
from pydantic import Field

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = GROUP / "data/review-site"
    neutral_run_dirs: list[Path] = Field(default_factory=list)
    control_run_dir: Path | None = None
    joint_run_dir: Path | None = None
    convergence_plot_dirs: list[Path] = Field(default_factory=list)
    optimization_plot_dirs: list[Path] = Field(default_factory=list)
    neutral_visual_dir: Path = GROUP / "data/neutral-state-visuals"
    neutral_contact_visual_dir: Path = GROUP / "data/contact-neutral-visuals"
    neutral_lineage_visual_dir: Path = GROUP / "data/neutral-lineage-visuals"
    contact_visual_dir: Path = GROUP / "data/contact-visuals"
    preparation_visual_dir: Path = GROUP / "data/preparation-visuals"
    source_bone_audit_dir: Path = GROUP / "data/source-bone-contact-audit"
    jaw_visual_dir: Path = GROUP / "data/jaw-domain-visuals-001"
    contact_validation_path: Path = GROUP / "data/contact-validation/summary.json"
    newton_face_validation_path: Path = (
        GROUP / "data/face-gradient-validation-contact-newton/summary.json"
    )
    collision_revision_path: Path = GROUP / "data/collision-model-revision.json"
    collision_off_curvature_path: Path = (
        GROUP / "data/collision-off-curvature-diagnostic-001/summary.json"
    )
    prestress_curvature_path: Path = (
        GROUP / "data/collision-off-prestress-curvature-ablation-001/summary.json"
    )
    full_skull_forward_attempt_paths: list[Path] = Field(
        default_factory=lambda: [
            GROUP / "data/full-skull-forward-benchmark-contended-001/summary.json",
            GROUP / "data/full-skull-forward-first-attempt-contended-002/summary.json",
        ]
    )
    simple_forward_dir: Path = GROUP / "data/simple-skin-forward-011"
    simple_forward_visual_dir: Path = GROUP / "data/simple-skin-forward-visuals-011"
    simple_skin_input_dir: Path = GROUP / "data/simple-skin-forward-inputs-001"
    simple_skin_field_manifest_path: Path = (
        GROUP / "data/simple-skin-forward-nu049-inputs-001/skin-field-manifest.json"
    )
    simple_skin_anchor_dir: Path = GROUP / "data/simple-skin-anchor-review-001"
    simple_forward_terminal_audit_path: Path = (
        GROUP / "data/simple-forward-terminal-audit-011/summary.json"
    )
    current_neutral_path: Path = GROUP / "data/current-neutral.json"
    eye_neutral_run_dir: Path = GROUP / "data/eye-neutral-forward-002"
    eye_neutral_review_dir: Path = GROUP / "data/eye-neutral-forward-review-006"
    expression_fit_dir: Path = GROUP / "data/expression-fitting-008"
    expression_fit_visual_dir: Path | None = None
    coupled_continuation_visual_dir: Path = (
        GROUP / "data/coupled-continuation-visuals-002"
    )
    archive_review: bool = True
    skin_field_visual_dir: Path = GROUP / "data/prescribed-skin-field-visuals-001"
    full_skull_visual_dir: Path = GROUP / "data/full-skull-geometry-review-003"
    full_skull_passive_path: Path = (
        GROUP / "data/full-skull-passive-equilibrium-001/summary.json"
    )
    full_skull_passive_interruption_path: Path = (
        GROUP / "data/full-skull-passive-equilibrium-001/interruption.json"
    )
    full_skull_rigid_separation_path: Path = (
        GROUP / "data/full-skull-rigid-separation-001/summary.json"
    )
    full_skull_operation_benchmark_path: Path = (
        GROUP
        / "data/full-skull-model-operation-benchmark-valid-contended-001/summary.json"
    )
    legacy_collision_benchmark_path: Path = (
        GROUP / "data/collision-benchmark-legacy-contended-002/summary.json"
    )
    full_skull_microbenchmark_path: Path = (
        GROUP
        / "data/full-skull-contact-microbenchmark-invalid-candidate-001/summary.json"
    )


CSS = """
:root{color-scheme:light dark;--bg:#f6f7f4;--fg:#1d303a;--card:#fff;--line:#d7dfdf;--muted:#586c75;--accent:#087d81}
@media(prefers-color-scheme:dark){:root{--bg:#121c21;--fg:#e1ebed;--card:#1b2a30;--line:#36484e;--muted:#b1c1c7;--accent:#83d9d3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}
main{max-width:1020px;margin:auto;padding:24px max(18px,env(safe-area-inset-left)) 64px}
h1{font-size:clamp(1.8rem,6vw,2.8rem);line-height:1.15;letter-spacing:-.03em}h2{font-size:1.3rem}
a{color:var(--accent);overflow-wrap:anywhere}p{margin:.8rem 0}small,.muted{color:var(--muted)}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}
.card,details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px}
.card strong{display:block}.card span{display:block;color:var(--muted);font-size:.88rem;margin-top:6px}
details{margin:16px 0}summary{cursor:pointer;font-weight:650;font-size:1.08rem}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:16px}
figure{margin:0}img{width:100%;height:auto;display:block;background:#fff;border-radius:8px}figcaption{font-size:.86rem;padding:.6rem 0;color:var(--muted)}
.callout{border-left:3px solid var(--accent);padding-left:15px}.table-scroll{overflow:auto}
table{border-collapse:collapse;font-size:.9rem;min-width:600px;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid var(--line)}
footer{margin-top:32px;border-top:1px solid var(--line);padding-top:16px;font-size:.85rem;color:var(--muted)}
@media(max-width:600px){.gallery{grid-template-columns:1fr}.status{grid-template-columns:1fr 1fr}.card{padding:11px}}
"""


def load_receipt(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_live_jsonl(path: Path) -> list[dict]:
    """Read complete JSONL rows while permitting one concurrently written tail."""
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
        if not isinstance(value, dict):
            message = f"JSONL row {index + 1} in {path} is not an object"
            raise TypeError(message)
        rows.append(value)
    return rows


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915 - compose the evidence page.
    output = cfg.output_dir
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    neutral = []
    for run in cfg.neutral_run_dirs:
        summary = load_receipt(run / "summary.json")
        trace = load_receipt(run / "trace.json")
        protocol = load_receipt(run / "protocol.json")
        neutral.append(
            {
                "run": str(run.resolve()),
                "summary": summary,
                "trace": trace,
                "protocol": protocol,
            }
        )
    control = (
        load_receipt(cfg.control_run_dir / "summary.json")
        if cfg.control_run_dir is not None
        else None
    )
    joint = (
        load_receipt(cfg.joint_run_dir / "summary.json")
        if cfg.joint_run_dir is not None
        else None
    )
    control_trace = (
        load_receipt(cfg.control_run_dir / "trace.json")
        if cfg.control_run_dir is not None
        else None
    )
    joint_trace = (
        load_receipt(cfg.joint_run_dir / "trace.json")
        if cfg.joint_run_dir is not None
        else None
    )
    contact = load_receipt(cfg.contact_validation_path)
    full_face = load_receipt(
        GROUP / "data/face-gradient-validation-contact/summary.json"
    )
    newton_full_face = load_receipt(cfg.newton_face_validation_path)
    collision_revision = load_receipt(cfg.collision_revision_path)
    collision_off_curvature = load_receipt(cfg.collision_off_curvature_path)
    prestress_curvature = load_receipt(cfg.prestress_curvature_path)
    full_skull_geometry_review = load_receipt(
        cfg.full_skull_visual_dir / "summary.json"
    )
    full_skull_microbenchmark = load_receipt(cfg.full_skull_microbenchmark_path)
    full_skull_forward_attempts = [
        load_receipt(path)
        for path in cfg.full_skull_forward_attempt_paths
        if path.is_file()
    ]
    full_skull_operation_benchmark = load_receipt(
        cfg.full_skull_operation_benchmark_path
    )
    legacy_collision_benchmark = load_receipt(cfg.legacy_collision_benchmark_path)
    full_skull_passive = load_receipt(cfg.full_skull_passive_path)
    full_skull_passive_interruption = load_receipt(
        cfg.full_skull_passive_interruption_path
    )
    full_skull_rigid_separation = load_receipt(cfg.full_skull_rigid_separation_path)
    simple_forward = load_receipt(cfg.simple_forward_dir / "summary.json")
    eye_neutral = load_receipt(cfg.eye_neutral_run_dir / "summary.json")
    eye_neutral_review = load_receipt(cfg.eye_neutral_review_dir / "summary.json")
    eye_neutral_complete = bool(
        eye_neutral
        and eye_neutral["success"]
        and eye_neutral_review
        and eye_neutral_review["success"]
    )
    if eye_neutral_complete:
        assert eye_neutral_review["run"]["summary"]["sha256"] == sha256(
            cfg.eye_neutral_run_dir / "summary.json"
        )
    simple_forward_protocol = load_receipt(cfg.simple_forward_dir / "protocol.json")
    simple_forward_heartbeat = load_receipt(cfg.simple_forward_dir / "heartbeat.json")
    simple_forward_trace = load_live_jsonl(cfg.simple_forward_dir / "trace.jsonl")
    simple_forward_interruption = load_receipt(
        cfg.simple_forward_dir / "external-interruption.json"
    )
    simple_forward_terminal_audit = load_receipt(cfg.simple_forward_terminal_audit_path)
    current_neutral = load_receipt(cfg.current_neutral_path)
    frozen_neutral_manifest = None
    frozen_neutral_validation = None
    frozen_neutral_dir = None
    if current_neutral is not None:
        assert current_neutral["schema"] == "joint-current-neutral-v1"
        assert current_neutral["baseline_stress_trainable_scalars"] == 0
        assert current_neutral["target_policy"] == "preserve_expression_displacements"
        frozen_neutral_dir = Path(current_neutral["directory"])
        manifest_path = frozen_neutral_dir / "manifest.json"
        validation_path = frozen_neutral_dir / "validation.json"
        assert sha256(manifest_path) == current_neutral["manifest_sha256"]
        frozen_neutral_manifest = load_receipt(manifest_path)
        frozen_neutral_validation = load_receipt(validation_path)
        assert frozen_neutral_manifest["schema"] == "joint-frozen-neutral-v1"
        assert frozen_neutral_manifest["success"] is True
        assert (
            frozen_neutral_manifest["target_policy"] == current_neutral["target_policy"]
        )
        assert frozen_neutral_manifest["full_joint_optimization_ready"] is False
        assert frozen_neutral_validation["success"] is True
        assert (
            frozen_neutral_validation["force_norm_N"]
            <= frozen_neutral_validation["force_threshold_N"]
        )
        assert frozen_neutral_validation["pncg_success_without_state_change"] is True
        assert frozen_neutral_validation["soft_bone_intersections"] is False
        assert (
            frozen_neutral_validation["all_transferred_target_residuals_exactly_zero"]
            is True
        )
        assert (
            frozen_neutral_validation["baseline_stress_trainable_scalars"]
            == current_neutral["baseline_stress_trainable_scalars"]
        )
        for receipt in frozen_neutral_manifest["artifacts"].values():
            artifact = Path(receipt["path"])
            assert artifact.is_file()
            assert sha256(artifact) == receipt["sha256"]
    simple_forward_visuals = load_receipt(
        cfg.simple_forward_visual_dir / "summary.json"
    )
    expression_fit_status = load_receipt(cfg.expression_fit_dir / "status.json")
    expression_fit_visuals = (
        load_receipt(cfg.expression_fit_visual_dir / "summary.json")
        if cfg.expression_fit_visual_dir is not None
        else None
    )
    simple_forward_visual_frame = (
        simple_forward_visuals["step_frame"]["current"]
        if isinstance(simple_forward_visuals, dict)
        else cfg.simple_forward_visual_dir.name
    )
    simple_skin_field = load_receipt(cfg.simple_skin_field_manifest_path)
    simple_forward_latest = simple_forward_trace[-1] if simple_forward_trace else None
    simple_forward_latest_exact_force = next(
        (
            row
            for row in reversed(simple_forward_trace)
            if "accepted_state_free_force_norm" in row
        ),
        None,
    )
    simple_forward_run_id = cfg.simple_forward_dir.name.removeprefix(
        "simple-skin-forward-"
    )
    simple_forward_run_label = f"Run {simple_forward_run_id}"
    simple_forward_heartbeat_age_seconds = (
        max(0.0, time.time() - float(simple_forward_heartbeat["unix_time"]))
        if simple_forward_heartbeat
        else None
    )
    simple_forward_heartbeat_is_fresh = (
        simple_forward_heartbeat_age_seconds is not None
        and simple_forward_heartbeat_age_seconds <= 60
    )
    simple_forward_contact_formula = None
    simple_forward_ccd_clearance = None
    simple_forward_armijo = None
    simple_forward_termination_force = None
    simple_forward_solver_detail = None
    if simple_forward_protocol:
        poisson = simple_forward_protocol["mechanics"]["poisson_ratios"]
        assert all(poisson[name] == 0.49 for name in ("fat", "aponeurosis", "muscle"))
        assert poisson["skin_range"] == [0.49, 0.49]
        assert (
            simple_forward_protocol["solver"]["inversion_policy"]
            == "diagnostic only; user permits inverted cells"
        )
        contact_protocol = simple_forward_protocol["mechanics"]["contact"]
        assert contact_protocol["collision_set_type"] == "IPC"
        ccd_cap = int(contact_protocol["ccd_max_iterations"])
        simple_forward_contact_formula = (
            f"standard area-weighted IPC with a {ccd_cap:,}-iteration CCD cap"
        )
        if "ccd_min_distance_m" in contact_protocol:
            ccd_clearance_nm = contact_protocol["ccd_min_distance_m"] * 1e9
            barrier_dmin_nm = contact_protocol["barrier_dmin_m"] * 1e9
            simple_forward_ccd_clearance = (
                f"{ccd_clearance_nm:g} nm swept-path CCD clearance; "
                f"barrier dmin {barrier_dmin_nm:g} nm. The clearance must be inactive "
                "in the reported final equilibrium and is excluded from accepted-state "
                "energy and gradient; the terminal gate uses the unchanged exact "
                "unprojected free-force norm."
            )
        solver_protocol = simple_forward_protocol["solver"]
        armijo = float(solver_protocol.get("line_search_armijo", 1.0e-4))
        armijo_source = (
            "declared by this protocol"
            if "line_search_armijo" in solver_protocol
            else "legacy default; absent from this protocol"
        )
        simple_forward_armijo = f"Armijo c = {armijo:g} ({armijo_source})"
        if termination_force := solver_protocol.get("termination_force"):
            simple_forward_termination_force = (
                f"{termination_force} (declared by this protocol)"
            )
        restart_checkpoint = simple_forward_protocol["inputs"].get("restart_checkpoint")
        restart_text = (
            f"restart from {Path(restart_checkpoint).parent.name}/"
            f"{Path(restart_checkpoint).name}; "
            if restart_checkpoint
            else "reference start; "
        )
        simple_forward_solver_detail = (
            f"{restart_text}Hessian damping "
            f"{solver_protocol.get('hessian_damping_initial', 0):g}; "
            f"PNCG restart every "
            f"{solver_protocol.get('pncg_restart_interval_steps', 0):,} steps; "
            f"{simple_forward_armijo}; "
            f"step infinity-norm cap "
            f"{solver_protocol['max_step_norm_m'] * 1e6:g} µm; fixed force threshold "
            f"{solver_protocol['force_threshold_override'] * 1e6:.7g} N"
        )
    if simple_forward_terminal_audit:
        assert simple_forward, "terminal audit requires a run summary"
        assert simple_forward_terminal_audit["success"] is True
        assert simple_forward_terminal_audit["summary_sha256"] == sha256(
            cfg.simple_forward_dir / "summary.json"
        )
        assert (
            simple_forward_terminal_audit["checkpoint"]["sha256"]
            == simple_forward["checkpoint"]["sha256"]
        )
    full_skull_pending = bool(
        collision_revision and not collision_revision["full_skull_ready"]
    )
    source_bone = load_receipt(cfg.source_bone_audit_dir / "summary.json")
    prepared_manifest = load_receipt(GROUP / "data/prepared/manifest.json")
    assert isinstance(prepared_manifest, dict)
    legacy_shared_count = (
        len(neutral[-1]["trace"][-1]["shared"])
        if neutral and neutral[-1]["trace"]
        else 20
    )
    expression_count = len(prepared_manifest["cohort"]["training"])
    activation_parameter_count = (
        expression_count * prepared_manifest["fixture"]["active_cells"] * 6
    )
    jaw_parameter_count = expression_count * 6
    skin_stiffness_parameter_count = 1
    if expression_fit_status is not None:
        expression_count = expression_fit_status["expression_count"]
        activation_parameter_count = (
            expression_count
            * expression_fit_status["activation_coordinates_per_expression"]
        )
        jaw_parameter_count = expression_count * expression_fit_status.get(
            "mandible_dofs_per_expression", 6
        )
        skin_stiffness_parameter_count = 0
    parameter_count = (
        activation_parameter_count
        + jaw_parameter_count
        + skin_stiffness_parameter_count
    )
    fem_contact_audit = load_receipt(
        GROUP / "data/fem-contact-surface-audit/summary.json"
    )
    neutral_complete = bool(neutral) and bool(
        neutral[-1]["summary"]
        and neutral[-1]["summary"].get("preparation_complete") is True
        and neutral[-1]["trace"]
        and abs(neutral[-1]["trace"][-1]["material"]["skin_target_n_per_m"] - 80.6)
        < 1e-10
    )
    neutral_status = "Converged" if neutral_complete else "In progress; not complete"
    if neutral and (latest_summary := neutral[-1]["summary"]) and not neutral_complete:
        if latest_summary.get("preparation_complete"):
            neutral_status = "Stage converged; continuation required"
        elif latest_summary.get("optimizer_converged"):
            neutral_status = "Stationary; neutral shape limit failed"
        else:
            neutral_status = "Stopped; not converged"
    if full_skull_pending:
        neutral_complete = False
        neutral_status = "Legacy run stopped; collider changing"
    if frozen_neutral_manifest and frozen_neutral_validation:
        neutral_status = "Adopted and force-converged"
    copied = {}

    def gallery(
        title: str, description: str, paths: list[Path], *, opened: bool = False
    ):
        nonlocal body
        if not paths:
            return
        body += f'<details{" open" if opened else ""}><summary>{html.escape(title)}</summary><p>{html.escape(description)}</p><div class="gallery">'
        for path in paths:
            assert path.is_file(), path
            name = f"{path.parent.name}--{path.name}"
            asset = assets / name
            source_stat = path.stat()
            unchanged = asset.is_file() and (
                asset.stat().st_size == source_stat.st_size
                and asset.stat().st_mtime_ns == source_stat.st_mtime_ns
            )
            if not unchanged:
                asset_temporary = assets / (name + ".tmp")
                shutil.copy2(path, asset_temporary)
                asset_temporary.replace(asset)
            caption = path.stem.replace("-", " ").replace("_", " ")
            figure_receipt = load_receipt(path.parent / "summary.json")
            if isinstance(figure_receipt, dict):
                for item in figure_receipt.get("assets", []):
                    if isinstance(item, dict) and item.get("filename") == path.name:
                        caption = item["caption"]
                        break
            href = "assets/" + name
            body += f'<figure><a href="{html.escape(href)}"><img loading="lazy" src="{html.escape(href)}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
            copied[name] = {"source": str(path.resolve()), "sha256": sha256(path)}
        body += "</div></details>"

    updated = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%b %d, %H:%M CST")
    body = f"<small>JOINT INVERSE EXPERIMENT · {html.escape(updated)}</small>"
    if expression_fit_status is not None:
        fit_phase = str(expression_fit_status.get("phase", "starting"))
        current_expression = html.escape(
            str(expression_fit_status.get("current_expression", "not selected"))
        )
        calibration_completed = int(
            expression_fit_status.get("calibration_completed_expressions", 0)
        )
        expression_count = int(expression_fit_status["expression_count"])
        pose_first = bool(expression_fit_status.get("pose_first", False))
        pose_collision = bool(expression_fit_status.get("pose_collision", True))
        if pose_first:
            body += "<h1>Pose first.<br>Then joint fitting.</h1>"
            body += (
                '<p class="callout"><strong>Each expression begins pose-only with '
                "zero muscle activation.</strong> A pose-stage stationarity result "
                "unlocks joint fitting, but is not joint convergence. "
                f"Current expression: {current_expression}. "
                '<a href="#fit-progress">View live stage progress.</a></p>'
            )
            if not pose_collision:
                body += (
                    '<p class="callout"><strong>Pose-only: collision off (initialization).</strong> '
                    "The joint stage restores full-skull and fixed-eye collision."
                    "</p>"
                )
        else:
            body += "<h1>Expression fitting.<br>Calibrating fields.</h1>"
        if not pose_first and fit_phase == "calibrating_strong_smoothness":
            body += (
                '<p class="callout"><strong>Strong-smoothness calibration is running; '
                "no expression fit has been accepted yet.</strong> "
                f"Current target: {current_expression}. Calibration: "
                f"{calibration_completed} / {expression_count}. "
                '<a href="#fit-progress">View live fit progress.</a></p>'
            )
        elif not pose_first:
            body += (
                '<p class="callout"><a href="#fit-progress">View live fit progress.</a> '
                "Saved checkpoints and per-expression state are reported below.</p>"
            )
    else:
        body += (
            "<h1>Fixed eyes.<br>Neutral converged.</h1>"
            if eye_neutral_complete
            else "<h1>Adopted neutral.<br>Then joint trends.</h1>"
            if frozen_neutral_manifest
            else "<h1>Simple forward.<br>Skin tension + full skull.</h1>"
        )
    if eye_neutral_complete:
        eye_difference = eye_neutral_review["difference_from_previous_neutral"]
        body += (
            '<section id="eye-neutral"><p class="callout"><strong>'
            "Latest forward: both eyes are fixed rigid obstacles.</strong> "
            "Full source skull, original heterogeneous skin stiffness and stress, "
            'nu = 0.49, zero activation, fixed mandible.</p><div class="status">'
            f'<div class="card"><strong>PNCG converged</strong><span>{eye_neutral["accepted_steps"]:,} steps · {eye_neutral["wall_seconds"] / 60:.1f} minutes</span></div>'
            f'<div class="card"><strong>{eye_neutral["final_free_force_norm"] * 1e9:.3f} mN residual</strong><span>Target ≤ {eye_neutral["force_threshold"] * 1e9:.3f} mN</span></div>'
            f'<div class="card"><strong>{eye_neutral["metrics"]["inverted_tetrahedra"]} inverted cells</strong><span>No eye or skull intersections</span></div>'
            f'<div class="card"><strong>{eye_neutral["contact"]["minimum_active_distance_m"] * 1e6:.2f} μm minimum gap</strong><span>Active IPC pairs; fixed eyes verified</span></div></div>'
            f"<p>The observed face changed by <strong>{eye_difference['observation_surface_unweighted_rms_mm']:.3f} mm RMS / {eye_difference['observation_surface_max_mm']:.3f} mm maximum</strong> "
            "from the previous neutral. Independent checks found no tissue nodes inside either eye. "
            "The original constitutive reference and material fields are preserved.</p>"
            '<p><a href="https://www.comet.com/liblaf/apple/56f927e06a3a49fab57ac3af120202dc">Recorded forward run</a> · '
            '<a href="#adopted-neutral">Previous neutral</a> · '
            '<a href="/full/">Experiment design</a></p>'
        )
        gallery(
            "Eye-inclusive neutral: shape, contact and convergence",
            "True-scale comparisons, changes around the eyes, complete rigid obstacles, and the final accepted-state force. Tap any figure for full resolution.",
            sorted(cfg.eye_neutral_review_dir.glob("*.png")),
            opened=True,
        )
        body += (
            "<details><summary>Initialization and collision numerics</summary>"
            "<p>The previous loaded face had 687 soft-eye triangle intersections. "
            "A local tissue repair cleared them without moving rigid anatomy or rebasing the FEM reference. "
            "Its seven inverted cells resolved during PNCG. The first forward attempt stopped at a 7.90 nm gap, "
            "below the 10 nm collision buffer. The successful rerun uses 0.1 nm CCD tolerance, "
            "a conservative step margin, and a trial clearance check; the physical barrier and materials are unchanged.</p>"
            "<p>This is a converged neutral forward. The final joint inverse optimization remains pending.</p>"
            "</details></section>"
        )
    if expression_fit_status is not None:
        assert (
            expression_fit_status["schema"]
            == "joint-fixed-material-expression-fitting-v1"
        )
        expression_rows = expression_fit_status["expressions"]
        counts = {
            key: sum(row["status"] == key for row in expression_rows.values())
            for key in ("queued", "fitting", "converged", "line_search_failed")
        }
        counts["fitting"] += sum(
            row["status"] == "initialized" for row in expression_rows.values()
        )
        counts["line_search_failed"] += sum(
            row["status"] in {"stationary_primal_unresolved", "descent_unresolved"}
            for row in expression_rows.values()
        )
        needs_recovery = sum(
            row["status"] == "requires_contact_recovery"
            for row in expression_rows.values()
        )
        activation_total = (
            expression_fit_status["expression_count"]
            * expression_fit_status["activation_coordinates_per_expression"]
        )
        phase = str(expression_fit_status.get("phase", "starting"))
        running = bool(expression_fit_status["running"])
        pose_first = bool(expression_fit_status.get("pose_first", False))
        pose_collision = bool(expression_fit_status.get("pose_collision", True))
        state_text = (
            f"currently {phase.replace('_', ' ')}"
            if running
            else f"stopped at {phase.replace('_', ' ')}"
        )
        smoothness_text = (
            f"{expression_fit_status['smoothness_weight']:.4g} smoothness weight"
            if "smoothness_weight" in expression_fit_status
            else "smoothness calibration pending"
        )
        jaw_description = (
            "one mandible hinge angle (0-40° opening from neutral; translation fixed)"
            if expression_fit_status.get("mandible_dofs_per_expression") == 1
            else "six mandible coordinates"
        )
        stage_text = (
            "Each expression starts pose-only at zero activation, then enters the joint stage only after pose stationarity. Pose convergence is not joint convergence."
            if pose_first
            else "Each expression has its own dense six-component symmetric active-stress field and "
            + jaw_description
            + "."
        )
        body += f'<section id="fit-progress"><p class="callout"><strong>All 36 fixed-material expression fits: {html.escape(state_text)}.</strong> {stage_text} Materials, baseline stress, skull, and eyes are fixed.</p><div class="status">'
        body += f'<div class="card"><strong>{counts["fitting"]} fitting</strong><span>{counts["queued"]} queued · {counts["converged"]} converged</span></div>'
        body += f'<div class="card"><strong>{activation_total:,} activation coordinates</strong><span>{expression_fit_status["activation_coordinates_per_expression"]:,} per expression</span></div>'
        body += f'<div class="card"><strong>{html.escape(smoothness_text)}</strong><span>5 mm graph prior; calibration recorded separately</span></div>'
        body += '<div class="card"><strong>Fixed material stage</strong><span>Convergence tracked per expression</span></div></div>'
        if pose_first:
            pose_only = sum(
                row.get("fit_stage") == "pose_only" for row in expression_rows.values()
            )
            joint = sum(
                row.get("fit_stage") == "joint" for row in expression_rows.values()
            )
            pose_converged = sum(
                bool(row.get("pose_converged", False))
                for row in expression_rows.values()
            )
            body += f"<p><strong>Stage ledger:</strong> {pose_only} pose-only · {joint} joint · {pose_converged} pose-stationary / {expression_fit_status['expression_count']}. A pose-stationary result only authorizes the joint stage.</p>"
            if not pose_collision:
                body += "<p><strong>Collision policy:</strong> pose-only is collision off for initialization; every joint solve uses the full skull and fixed eyes.</p>"
        if expression_fit_status.get("coupled_predictor"):
            body += (
                "<p><strong>Coupled predictor:</strong> internal pose, tissue, and stress "
                "seeds are used while full skull and fixed-eye contact remain on; every "
                "accepted state requires final strict PNCG. Each expression fits pose first, "
                "then joint activation and pose.</p>"
            )
        if expression_fit_status.get("schedule") == "sequential":
            current_name = expression_fit_status.get("current_expression")
            if current_name is not None:
                body += f"<p><strong>Fitting one expression at a time: {html.escape(current_name)}.</strong> The next expression starts after convergence or an explicit fitting failure. A computation budget stops the run without declaring convergence.</p>"
        saved_rows = {
            name: row for name, row in expression_rows.items() if "fit_rms_mm" in row
        }
        if saved_rows:
            if not counts["converged"]:
                body += "<p><strong>No expression fit has converged yet.</strong> The images show saved optimizer states; the forward equilibrium can converge while the expression fit is still close to neutral.</p>"
            body += '<div class="table-scroll"><table><tr><th>Expression</th><th>Stage</th><th>Pose updates</th><th>Joint updates</th><th>Pose stationary</th><th>Initial RMS</th><th>Current RMS</th><th>RMS reduction</th>'
            hinge = expression_fit_status.get("mandible_dofs_per_expression") == 1
            if hinge:
                body += "<th>Opening angle</th>"
            body += "</tr>"
            for name, row in saved_rows.items():
                initial = row.get("initial_fit_rms_mm", row["target_motion_rms_mm"])
                current = row["fit_rms_mm"]
                reduction = 100 * (1 - current / initial)
                stage = html.escape(str(row.get("fit_stage", "joint")))
                pose_steps = int(row.get("pose_accepted_steps", 0))
                joint_steps = int(
                    row.get("joint_accepted_steps", row["accepted_steps"])
                )
                pose_ok = "yes" if row.get("pose_converged", False) else "no"
                body += f"<tr><td>{html.escape(name)}</td><td>{stage}</td><td>{pose_steps}</td><td>{joint_steps}</td><td>{pose_ok}</td><td>{initial:.3f} mm</td><td>{current:.3f} mm</td><td>{reduction:.2f}%</td>"
                if hinge:
                    body += f"<td>{row['mandible_angle_deg']:.3f}°</td>"
                body += "</tr>"
            body += "</table></div><p>Numbers follow the latest saved checkpoints. Each figure records its own saved update; figures refresh less often.</p>"
        if phase == "calibrating_strong_smoothness":
            calibration_completed = int(
                expression_fit_status.get("calibration_completed_expressions", 0)
            )
            current_expression = html.escape(
                str(expression_fit_status.get("current_expression", "not selected"))
            )
            body += (
                f"<p><strong>Calibration target:</strong> {current_expression}; "
                f"{calibration_completed} / {expression_fit_status['expression_count']} "
                "target adjoints completed. This calibrates the common strong "
                "smoothness weight; it is not an accepted expression fit.</p>"
            )
        if expression_fit_status.get("inner_trace_directory"):
            inner_rows = load_live_jsonl(
                Path(expression_fit_status["inner_trace_directory"]) / "trace.jsonl"
            )
            if inner_rows:
                latest_inner = inner_rows[-1]
                body += f"<p>Last forward update: PNCG step {latest_inner['step']:,}, residual {latest_inner['force_code'] * 1e12:.3f} μN. Target ≤ {expression_fit_status['forward_force_threshold_code'] * 1e12:.3f} μN.</p>"
        if counts["line_search_failed"]:
            body += f"<p><strong>{counts['line_search_failed']} expression(s) require attention.</strong> Their last accepted checkpoints are retained and shown separately when available.</p>"
        if needs_recovery or phase == "requires_contact_recovery":
            body += "<p><strong>Contact recovery is required.</strong> The run stopped at its recovery gate; this state is neither fitted nor converged.</p>"
        if expression_fit_visuals is None:
            body += "<p>Accepted-state visualizations will appear here as checkpoints are rendered. Queued expressions have no fit result yet.</p>"
        else:
            visual_dir = cfg.expression_fit_visual_dir
            assert visual_dir is not None
            gallery(
                "Fixed-material expression fits: accepted checkpoints",
                "Target and predicted original-expression increments, residuals, and objective trends. Missing expressions are queued or unavailable, not implicit failures.",
                sorted(visual_dir.glob("*.png")),
                opened=True,
            )
        coupled_visuals = load_receipt(
            cfg.coupled_continuation_visual_dir / "summary.json"
        )
        if coupled_visuals is not None:
            assert coupled_visuals["success"]
            gallery(
                "Coupled continuation: full-contact method validation",
                "A prescribed jaw and stress change with strict PNCG correction. These internal substeps validate the solver method; they are not accepted expression-fitting iterations.",
                sorted(cfg.coupled_continuation_visual_dir.glob("*.png")),
                opened=False,
            )
        body += "</section>"
    if frozen_neutral_manifest and frozen_neutral_validation:
        force = frozen_neutral_validation["force_norm_N"]
        threshold = frozen_neutral_validation["force_threshold_N"]
        stiffness_force = frozen_neutral_validation[
            "one_percent_stiffness_change_force_at_frozen_shape_N"
        ]
        body += (
            '<section id="adopted-neutral"><p class="callout"><strong>'
            "Previous adopted neutral, without eye contact.</strong> The converged prescribed-skin, "
            "full-source-skull shape is frozen as the observation neutral. "
            "The original FEM/material reference remains active; this is not a "
            "stress-free mesh rebase. Baseline stress is no longer fitted. "
            f"Zero-increment force is {force:.6g} N, below {threshold:.6g} N; "
            "full-skull contact and transferred targets passed. "
            f"A +1% skin-stiffness probe gives {stiffness_force:.6g} N at the "
            "frozen geometry, so altered stiffness requires a fresh equilibrium "
            "and must not be labeled an exact neutral state. "
            '<a href="#simple-forward">See the converged forward gallery.</a> '
            '<a href="/full/">Technical design.</a></p></section>'
        )
    if cfg.skin_field_visual_dir.is_dir():
        body += '<section id="skin-fields">'
        gallery(
            "Skin stiffness and prestress: prescribed inputs",
            "Exact fields used by the latest simple forward. E = 127.4-257.9 kPa; isotropic prestress resultant N0 = 30.4-80.8 N/m. At the assumed 1 mm thickness, N0/h = 30.4-80.8 kPa. Tap a figure to inspect its full resolution. These are literature-informed priors, not measured subject fields or optimized results.",
            sorted(cfg.skin_field_visual_dir.glob("*.png")),
            opened=True,
        )
        body += '<details><summary>What supports the material values?</summary><div class="table-scroll"><table><tr><th>Material</th><th>Current E</th><th>Evidence and transfer limit</th></tr><tr><td>Fat</td><td>11.2 kPa</td><td><a href="https://doi.org/10.5114/ada.2018.79778">Paluch et al.</a>: deep medial cheek fat SWE, 89 women, 11.2 ± 6.9 kPa. Age-dependent apparent stiffness; not a static SNH calibration.</td></tr><tr><td>Muscle</td><td>12 kPa</td><td><a href="https://doi.org/10.4236/jbise.2019.1211037">Ternifi et al.</a>: relaxed zygomaticus major SWE, 15 volunteers. Two probes gave 12.0 ± 4.3 and 18.3 ± 3.7 kPa; current value uses the first probe.</td></tr><tr><td>Aponeurosis</td><td>1.693 MPa</td><td><a href="https://doi.org/10.1093/asjof/ojaf126">Tereshenko et al.</a>: cervical SMAS/platysma tensile tests, seven patients, 1.693 ± 0.543 MPa. A neck-tissue proxy, not facial-aponeurosis calibration.</td></tr><tr><td>Skin</td><td>127.4-257.9 kPa</td><td><a href="https://doi.org/10.1016/j.jmbbm.2013.03.004">Flynn et al.</a>: six regional inverse Ogden-QLV fits on one volunteer, converted to a small-strain modulus and mean isotropic prestress.</td></tr></table></div><p>The active diagnostic sets Poisson ratio nu = 0.49 for fat, aponeurosis, muscle and skin. The earlier field illustrations remain valid for E and N0, which are unchanged, but their older metadata does not describe this nu-only rerun. The 1 mm skin thickness, isotropic prestress, bilateral manual placement, and 20 mm Gaussian blend are modeling assumptions. The source skin model used 1.5 mm thickness and omitted underlying tissue/bone attachments, which can inflate its fitted stiffness and tension. Nose, lips, eyelids and scalp are extrapolated. None of the studies calibrates this subject or the present coupled SNH model.</p></details></section>'
    if expression_fit_status is None:
        body += f"<p>Tuesday's result comes from the simultaneous <strong>{parameter_count:,}-variable</strong> optimization: {activation_parameter_count:,} activation, {jaw_parameter_count:,} jaw, and one global skin-stiffness scalar. Baseline stress has zero trainable scalars. Joint-runner integration, derivative/control checks, and the final trajectory remain pending; only the final joint stage may remain unconverged.</p>"
    else:
        body += f"<p>The current fixed-material stage has <strong>{activation_total + jaw_parameter_count:,} variables</strong>: {activation_total:,} active-stress coordinates and {jaw_parameter_count} mandible coordinates across {expression_count} expressions. Material and baseline-stress variables are fixed. The later simultaneous material stage and final trajectory remain pending.</p>"
    if simple_skin_field:
        historical_title = (
            "Previous diagnostic (without eye contact)"
            if eye_neutral_complete
            else "Current diagnostic"
        )
        selected_description = (
            "is the preserved earlier forward"
            if eye_neutral_complete
            else "is the selected diagnostic"
        )
        body += f'<details id="simple-forward"><summary>{historical_title}: prescribed skin forward</summary><p><strong>{html.escape(simple_forward_run_label)} {selected_description}.</strong> Complete source skull contact using {html.escape(simple_forward_contact_formula or "an undeclared contact formula")}; passive bulk with zero baseline stress; no activation; fixed neutral jaw; PNCG only; nu = 0.49 for every tissue. The admitted clearance repair is rebased as the FEM reference, so bulk tissue starts without repair-induced strain.</p><p><strong>Solver state:</strong> {html.escape(simple_forward_solver_detail or "not declared")}.</p>'
        if simple_forward_ccd_clearance:
            body += f"<p><strong>CCD clearance:</strong> {html.escape(simple_forward_ccd_clearance)}</p>"
        if simple_forward_termination_force:
            body += f"<p><strong>Termination force:</strong> {html.escape(simple_forward_termination_force)}. Optimizer success and the independent terminal force gate therefore evaluate the same accepted state.</p>"
        body += "<p><strong>Inversion policy:</strong> inverted cells are recorded as diagnostics and are not a convergence gate for this user-requested run. A converged force residual, terminal receipt, and explicit geometry report are still required before interpreting the result.</p>"
        if simple_forward:
            if simple_forward["success"]:
                status = "Converged"
            elif simple_forward.get("status") == "interrupted":
                status = "Stopped before convergence"
            else:
                status = "Not converged"
            body += f"<p><strong>{status}</strong> after {simple_forward['accepted_steps']:,} accepted steps and {simple_forward['wall_seconds'] / 60:.1f} minutes. Free-force norm: {simple_forward['initial_free_force_norm'] * 1e6:.4g} → {simple_forward['final_free_force_norm'] * 1e6:.4g} N; threshold {simple_forward['force_threshold'] * 1e6:.4g} N.</p>"
            if "metrics" in simple_forward:
                metrics = simple_forward["metrics"]
                gap = simple_forward["contact"]["minimum_active_distance_m"]
                gap_text = (
                    f"{gap * 1e6:.3g} µm" if gap is not None else "no active pairs"
                )
                body += f"<p>Surface RMS motion {metrics['surface_motion_rms_mm']:.3g} mm; physical det(F) {metrics['detF_min']:.4g} to {metrics['detF_max']:.4g}; {metrics['inverted_tetrahedra']} inverted tets; minimum active contact gap {gap_text}.</p>"
            if simple_forward_terminal_audit:
                audit_gap = (
                    simple_forward_terminal_audit["contact"][
                        "minimum_active_distance_m"
                    ]
                    * 1e6
                )
                body += f"<p><strong>Independent CPU endpoint audit passed.</strong> The hash-bound terminal checkpoint retained all 1,146,517 tetrahedra with zero inversions, no soft-bone intersections, zero negative contact weights, and {audit_gap:.4g} µm minimum active gap. This validates the reported endpoint geometry and contact state; it is not a mechanical-stability proof.</p>"
        elif simple_forward_interruption:
            body += f"<p><strong>{html.escape(simple_forward_run_label)} was deliberately interrupted without a terminal convergence receipt.</strong> Its saved checkpoints remain diagnostic evidence, and the interruption record does not indicate solver convergence.</p>"
        elif (
            simple_forward_heartbeat
            and simple_forward_protocol
            and simple_forward_heartbeat_is_fresh
        ):
            heartbeat_step = int(simple_forward_heartbeat["accepted_steps"])
            latest_step = (
                int(simple_forward_latest["step"]) if simple_forward_latest else 0
            )
            accepted_steps = max(heartbeat_step, latest_step)
            operation = html.escape(str(simple_forward_heartbeat["current_operation"]))
            wall_minutes = simple_forward_heartbeat["wall_elapsed_seconds"] / 60
            body += f"<p><strong>Running; no terminal convergence receipt yet.</strong> {accepted_steps:,} accepted steps at the latest heartbeat after {wall_minutes:.1f} minutes; current operation: {operation}."
            if simple_forward_latest_exact_force:
                force_step = simple_forward_latest_exact_force["step"]
                force = (
                    simple_forward_latest_exact_force["accepted_state_free_force_norm"]
                    * 1e6
                )
                body += f" Latest independently evaluated accepted-state force: {force:.4g} N at step {force_step:,}."
            body += " The live trace is progress evidence, not convergence.</p>"
        elif simple_forward_heartbeat and simple_forward_protocol:
            heartbeat_step = int(simple_forward_heartbeat["accepted_steps"])
            latest_step = (
                int(simple_forward_latest["step"]) if simple_forward_latest else 0
            )
            accepted_steps = max(heartbeat_step, latest_step)
            heartbeat_age = simple_forward_heartbeat_age_seconds or 0.0
            body += f"<p><strong>Status unknown; awaiting a fresh update.</strong> The latest heartbeat is {heartbeat_age:.0f} seconds old and recorded {accepted_steps:,} accepted steps. It is too stale to label the run as active.</p>"
        else:
            body += "<p><strong>Status unknown; awaiting an update.</strong> No terminal receipt or fresh heartbeat is available. The figures below identify their saved checkpoint.</p>"
        if simple_forward and not simple_forward["success"]:
            if simple_forward.get("status") == "interrupted":
                body += "<p>This run has a terminal interruption receipt and is not a converged equilibrium. Its inversion count remains a reported diagnostic under the declared run policy.</p>"
            else:
                body += "<p>This run has a terminal failure receipt and is not a converged equilibrium. Its inversion count remains a reported diagnostic under the declared run policy.</p>"
        body += "<p>Historical status: run 001 developed inverted fat tetrahedra; run 002 kept positive determinant with its volume step bound but stopped after strict line-search exhaustion. Run 003 was deliberately stopped at step 1,292 after its IMPROVED_MAX_APPROX collision set produced negative weights, nondeterministic fixed-state energy, and negative contact energy. That is a contact-formula failure independent of the authorized volume inversions. Run 004 used the validated standard area-weighted IPC formula, then stalled inside native CCD after trace step 234; it was deliberately terminated and checkpoint 200 was preserved for run 005. Runs 005 and 006 ended as deliberate continuation segments at local steps 2,739 and 3,343; both retained positive cell determinants and numerically valid contact but remained above the fixed force target. Run 007 was a discarded branch from run 006: its aggressive 0.5 mm step cap produced femtometre-scale active gaps and another native CCD stall at local step 331. Run 008 restarted directly from run 006 and ended cleanly at local step 1,466 with 0.1028 N exact force, 17.25 µm minimum active gap, and no inverted cells; it remained above the force target. Run 009 added the validated 10 nm swept-path CCD clearance and stopped cleanly at local step 8,784 with 0.0008259 N exact force, 17.07 µm minimum active gap, and no inverted cells. Run 010 used stricter Armijo c = 0.25 and reached the upstream optimizer's primary-success state at local step 2,463, but the independent accepted-state force gate rejected it: 0.0001911 N exceeded the unchanged 0.0001519 N target. Run 010 is therefore not converged. Run 011 resumes its terminal checkpoint with the exact free gradient recomputed at every accepted state for termination. The full-source IPC contact gradient finite-difference check passed with 1.15e-6 maximum relative error. None of runs 001-010 converged, and volume inversions were not the stop reason for runs 004-010.</p>"
        body += '<p>Skin properties use a smooth, bilateral transfer of six regional Flynn (2013) inverse fits, with a 1 mm membrane and isotropic mean prestress. These are sparse literature priors, not a measured subject map. Nose, lips, eyelids and scalp use extrapolation. <a href="https://doi.org/10.1016/j.jmbbm.2013.03.004">Primary source</a>.</p><p>This standalone forward does not establish completion of neutral preparation or the final joint optimization.</p></details>'
    if full_skull_pending:
        body += '<p class="callout"><strong>Complete source skull.</strong> The partial FEM bone collider is superseded and its optimization is stopped. The complete-source adapter is implemented. The repaired seed clears both bones with no inverted tetrahedra; equilibrium and full jaw-domain validation remain incomplete. Old optimization galleries below are legacy diagnostics.</p>'
    checked_face = newton_full_face or full_face
    contact_description = (
        f"Cranium + moving mandible; largest derivative error "
        f"{100 * checked_face['maximum_relative_error']:.2f}%. "
        "Anatomical coverage is audited below."
        if checked_face and checked_face.get("success")
        else "Declared FEM cranium + moving mandible; source-bone coverage mapped below."
    )
    cards = [
        (
            "Adopted neutral",
            neutral_status,
            "Frozen Run011 equilibrium; baseline stress is not fitted.",
        ),
        (
            "Bone contact",
            (
                "Full-skull equilibrium passed"
                if frozen_neutral_manifest
                else "Full-skull validation in progress"
                if full_skull_pending
                else "Full-face checks passed"
                if checked_face and checked_face["success"]
                else "Synthetic checks passed"
                if contact and contact["success"]
                else "Validation in progress"
            ),
            "Complete source bones passed frozen-neutral IPC/CCD validation; expression/jaw derivatives remain pending."
            if frozen_neutral_manifest
            else "Full registered source surfaces; legacy derivative checks do not validate this replacement."
            if full_skull_pending
            else contact_description,
        ),
        (
            "Activation / jaw preparation",
            "Converged"
            if control and control.get("preparation_complete")
            else "Not complete",
            "Full-source adapter and derivative/control checks are pending.",
        ),
        (
            "Final joint optimization",
            "Not started",
            "Activation, jaw, and one skin multiplier; no trainable baseline stress.",
        ),
    ]
    body += '<div class="status">'
    for title, status, description in cards:
        body += (
            f'<div class="card"><small>{html.escape(title)}</small>'
            f"<strong>{html.escape(status)}</strong><span>{html.escape(description)}</span></div>"
        )
    body += "</div>"
    if full_skull_forward_attempts:
        body += '<p class="callout"><strong>Full equilibrium timing: no valid ratio.</strong> Both cold solves from the repaired seed failed linear convergence before any adjoint or warmed repeat. Collision off stopped after 243 s; complete-source contact after 76 s. These are failure durations, not solve-speed measurements. Passive zero-baseline contact equilibration is a separate preparation experiment.</p>'
    if collision_off_curvature:
        assert collision_off_curvature["success"]
        witness = collision_off_curvature["lanczos"]["explicit_witness"]
        assert witness["verified_negative_curvature"]
        body += '<details><summary>Why collision-off failed: confirmed solver mismatch</summary><p>The exact Hessian has a verified negative-curvature direction at the initial loaded, repaired state. Ordinary CG requires positive definiteness, which this state violates. All diagonal entries are positive, so the diagonal preconditioner does not remove the negative coupled mode. This diagnoses the starting state; the seventh Newton iterate that exhausted CG was not saved.</p><p>Collision off retained fitted baseline stresses and the repair displacement. The unloaded reference is balanced to numerical precision. Force norms below are in newtons; they are vector norms, not additive contributions.</p><div class="table-scroll"><table><tr><th>Configuration</th><th>No baseline stress, reference stiffness</th><th>Fitted baseline and stiffness</th></tr>'
        forces = collision_off_curvature["forces"]
        for label, suffix in (
            ("Original reference", "reference_zero_displacement"),
            ("Repaired displacement", "repaired_candidate"),
        ):
            passive = 1e6 * forces[f"passive_{suffix}"]["free_force_norm"]
            loaded = 1e6 * forces[f"loaded_{suffix}"]["free_force_norm"]
            body += f"<tr><td>{label}</td><td>{passive:.4g} N</td><td>{loaded:.4g} N</td></tr>"
        body += "</table></div><p>The saved negative-curvature witness is reproducible; no converged equilibrium or final-launch readiness is claimed.</p></details>"
    if prestress_curvature:
        assert prestress_curvature["success"]
        assert prestress_curvature["scope"]["fixed_skin_stiffness_multiplier"] == 3
        body += '<details open><summary>Is prestress the cause?</summary><p>The measured negative mode remains negative with all baseline stresses removed. The imposed repair displacement, evaluated against the original FEM rest geometry, is sufficient for this instability. Prestress makes this mode only slightly more negative. Geometry and skin stiffness are held fixed within each row.</p><div class="table-scroll"><table><tr><th>Configuration</th><th>Zero baseline</th><th>Fitted baseline</th></tr>'
        cases = prestress_curvature["curvature_cases"]
        for label, prefix in (
            ("Original reference", "reference"),
            ("Repaired displacement", "repaired"),
        ):
            unloaded = cases[f"{prefix}_zero_baseline"]["quadratic_v_h_v"]
            loaded = cases[f"{prefix}_fitted_baseline"]["quadratic_v_h_v"]
            body += f"<tr><td>{label}</td><td>{unloaded:+.4f}</td><td>{loaded:+.4f}</td></tr>"
        body += "</table></div><p>Entries are vᵀHv for the same saved direction and normalization. Negative values certify a negative-curvature direction; positive values for this one direction do not prove the whole Hessian positive definite.</p><p><strong>How prestress was fitted:</strong> at zero expression activation and neutral jaw pose, prescribe a staged skin tension (currently 20.15 N/m), then fit smooth signed stress fields in fat, aponeurosis and muscle plus skin stiffness. Each outer BFGS trial solves mechanical equilibrium, and implicit gradients minimize neutral surface and muscle-centroid motion plus material priors and strong spatial smoothness. This uses 78 bulk coefficients and one free skin-stiffness parameter. The old partial-skull checkpoint reached mechanical equilibrium at its own deformation, but the outer fit was unfinished at update 111. It is not a calibrated full-skull prestress solution.</p></details>"
    if full_skull_passive:
        passive_status = (
            "converged at zero baseline stress"
            if full_skull_passive["success"]
            else "did not pass its convergence and geometry gates"
        )
        body += f"<p>Passive full-skull initialization {passive_status}. Nonzero prestress preparation and the full-source jaw domain remain incomplete.</p>"
    elif full_skull_passive_interruption:
        body += "<p>Passive initialization was interrupted after about 15 minutes without an intermediate numerical receipt. Its convergence is unknown; nonzero prestress preparation and the full-source jaw domain remain incomplete.</p>"
    if full_skull_operation_benchmark:
        timings = full_skull_operation_benchmark["warmed_seconds"]
        ratios = full_skull_operation_benchmark[
            "complete_source_over_off_median_ratios"
        ]
        assert full_skull_operation_benchmark["scope"]["candidate_volume_valid"]
        assert full_skull_operation_benchmark["scope"][
            "soft_bone_initialization_admitted"
        ]
        body += '<details open><summary>Full-skull collision: measured operation costs</summary><p>Complete FEM model, including GPU/CPU transfers; shared GPU; five warmed repeats. Both arms use the repaired, non-inverted initialization. This measures operation overhead at one fixed state, not equilibrium or optimization throughput.</p><div class="table-scroll"><table><tr><th>Operation</th><th>Off</th><th>Full skull</th><th>Ratio</th></tr>'
        for key, label in (
            ("full_model_energy", "Energy"),
            ("full_model_gradient", "Force"),
            ("full_model_first_hvp", "First Hessian product"),
            ("full_model_cached_hvp", "Cached Hessian product"),
        ):
            off = timings["off"][key]["median"] * 1000
            on = timings["complete_source_contact"][key]["median"] * 1000
            body += f"<tr><td>{label}</td><td>{off:.2f} ms</td><td>{on:.2f} ms</td><td>{ratios[key]:.2f}&times;</td></tr>"
        body += "</table></div><p>These operation timings remain separate from the failed cold equilibrium attempts described above.</p></details>"
    if legacy_collision_benchmark:
        assert (
            legacy_collision_benchmark["schema"]
            == "joint-collision-performance-benchmark-v1"
        )
        assert (
            legacy_collision_benchmark["scope"]["measured_contact"]
            == "legacy_partial_fem_ipc"
        )
        arms = legacy_collision_benchmark["warmed"]
        ratios = legacy_collision_benchmark["legacy_contact_over_off_median_ratios"]
        body += '<details><summary>Earlier solver performance: partial-FEM collider</summary><p>RTX 4090, shared GPU; one warmup and five alternating repeats per arm. Same seed, materials and tolerances, with a 1% skin stress perturbation. These timings apply only to the superseded partial-FEM collider.</p><div class="table-scroll"><table><tr><th>Operation</th><th>Collision off</th><th>Collision on</th><th>On / off</th></tr>'
        for key, label in (
            ("forward_wall_seconds", "Forward equilibrium"),
            ("adjoint_autograd_wall_seconds", "Backward / adjoint"),
        ):
            off = arms["off"][key]
            on = arms["legacy_partial_fem_ipc"][key]
            body += f"<tr><td>{label}</td><td>{off['median']:.2f} s</td><td>{on['median']:.2f} s</td><td>{ratios[key]:.2f}&times;</td></tr>"
        body += "</table></div><p>Contact used 2 Newton steps versus 5 without contact. The forward speed difference reflects this particular solve; it is not a general contact speedup. Backward took 26% longer.</p></details>"
    if full_skull_microbenchmark:
        assert full_skull_microbenchmark["scope"]["contact_operations_only"] is True
        assert full_skull_microbenchmark["scope"]["production_admission"] is False
        timings = full_skull_microbenchmark["warmed_contact_operations_seconds"]
        body += '<details><summary>Earlier full-skull CPU diagnostic</summary><p>CPU contact operations, one warmup and five repeats. This candidate has 13 inverted tetrahedra and is not admitted for equilibrium. These measurements do not establish the full solver slowdown.</p><div class="table-scroll"><table><tr><th>Operation</th><th>Median</th><th>Range</th></tr>'
        for key, label in (
            ("state_at", "Rebuild contact candidates"),
            ("energy", "Contact energy"),
            ("gradient", "Contact force"),
            (
                "first_hvp_assembly_and_matvec",
                "First Hessian product including assembly",
            ),
            ("cached_hvp_matvec", "Cached Hessian product"),
            ("ccd_max_step_size", "Collision-safe step bound"),
        ):
            values = timings[key]
            body += f"<tr><td>{label}</td><td>{1000 * values['median']:.2f} ms</td><td>{1000 * values['minimum']:.2f} to {1000 * values['maximum']:.2f} ms</td></tr>"
        body += "</table></div></details>"
    spatial_validation = None
    if legacy_shared_count == 80:
        assert neutral[-1]["protocol"]["shared_basis"] == "spatial80"
        evidence = neutral[-1]["protocol"]["spatial_validation"][
            "full_face_directional"
        ]
        path = Path(evidence["path"])
        assert sha256(path) == evidence["sha256"]
        spatial_validation = load_receipt(path)
        assert spatial_validation == evidence["receipt"]
        assert spatial_validation["success"] is True
        error = 100 * spatial_validation["maximum_mechanical_relative_error"]
        qualifier = "Legacy partial-FEM model: " if full_skull_pending else ""
        body += f"<p><strong>Spatial baseline:</strong> 80 shared coefficients; strong smoothness fixed. {qualifier}full-face deformation-loss derivatives passed with {error:.2f}% maximum error. This remains a preparation model under evaluation.</p>"
    if neutral and neutral[-1]["trace"]:
        row = neutral[-1]["trace"][-1]
        conv = row["convergence"]
        body += (
            f"<p><strong>Latest neutral preparation:</strong> "
            f"{100 * row['material']['skin_target_n_per_m'] / 80.6:.0f}% prestress continuation; "
            f"update {row['update']}; "
            f"projected gradient {row['projected_gradient_inf']:.3g} "
            f"(required ≤ {conv['projected_gradient_inf_tolerance']:.3g}); "
            f"surface drift {row['metrics']['surface_motion_rms_mm']:.3f} mm. "
            "This is preparation progress.</p>"
        )
    body += '<p class="callout"><strong>Audit correction:</strong> the earlier neutral “new contacts” were shared FEM vertices/edges. The corrected test finds no separate soft tissue/bone intersections in those saved states. Their short, contact-free runs remain preliminary and are not converged preparation.</p>'
    body += '<p><a href="/">Approved plan</a> · <a href="/full/">Technical design</a> · <a href="status.json">Evidence manifest</a></p>'
    gallery(
        "Simple forward: deformation and convergence",
        f"Saved {simple_forward_visual_frame}, while {simple_forward_run_label} is reported separately above. The images use actual-scale deformation, full-source soft-bone contact with standard area-weighted IPC, and nu = 0.49 for every tissue. Bulk baseline stress and activation are zero. Inversions are diagnostics under the declared policy; they are not the convergence gate.",
        sorted(cfg.simple_forward_visual_dir.glob("*.png")),
        opened=True,
    )
    gallery(
        "Prescribed heterogeneous skin fields",
        "Smooth convex interpolation of six regional literature fits; bilateral transfer is a modeling assumption. Labels identify derived moduli and stress resultants, not direct measurements.",
        sorted(cfg.simple_skin_input_dir.glob("*.png"))
        + sorted(cfg.simple_skin_anchor_dir.glob("*.png")),
        opened=False,
    )
    gallery(
        "Complete skull and initialization audit",
        "Every source cranium and mandible triangle is retained. The original reference intersects the bones. Candidate 001 clears the surfaces but inverts 13 tetrahedra; repaired candidate 002 clears both bones with physical det(F) between 0.2501 and 1.9999. Fixed nodes and observed outer facial nodes are unchanged. These are initialization diagnostics, not equilibrium or final optimization trends. The source bones themselves still intersect and require separate jaw-domain work.",
        sorted(cfg.full_skull_visual_dir.glob("*.png")),
        opened=True,
    )
    preparation = cfg.preparation_visual_dir
    gallery(
        "Contact surfaces and attachment assumptions",
        "Pure soft tissue contacts the cranium and mandible. Magenta mixed-label transition faces are treated as bonded junctions. The supplied labels do not independently establish attachment anatomy.",
        sorted(preparation.glob("05-*.png")),
        opened=False,
    )
    gallery(
        "Contact forces and gaps",
        "Reference-state IPC forces and geometry. Contact stiffness is a numerical parameter. Collision-pair counts are algorithm representatives, not contact area or a convergence metric.",
        sorted(cfg.contact_visual_dir.glob("*.png")),
    )
    gallery(
        "Frozen neutral contact forces and gaps",
        "Contact evaluated at the immutable contact-enabled neutral checkpoint; see figure labels for the checkpoint update.",
        sorted(cfg.neutral_contact_visual_dir.glob("*.png")),
    )
    gallery(
        "Complete source bones: compatibility audit",
        "The complete registered source bones intersected the original soft-tissue boundary. These historical maps expose that mismatch. The active diagnostic instead uses the admitted, rebased collision-free initialization while retaining every source-bone triangle. Green marks bonded coincidences, amber nearby unconfined intersections, and red intersections farther from bonded geometry.",
        sorted(cfg.source_bone_audit_dir.glob("*.png")),
    )
    gallery(
        "Jaw proposals: rigid bone collision",
        "A separate CCD guard checks the pure FEM mandible against the cranium along the solver's straight boundary-vertex path. The broad search box is only a proposal bound; rejected motions cannot become accepted expression states.",
        sorted(cfg.jaw_visual_dir.glob("*.png")),
    )
    gallery(
        "Frozen neutral shape",
        "Deformation overlays and internal sections from the immutable contact-enabled preparation checkpoint.",
        sorted(cfg.neutral_visual_dir.glob("*.png")),
    )
    gallery(
        "Selected neutral preparation lineage",
        "One parent-hash-proven branch through the frozen checkpoint. Cumulative accepted updates exclude abandoned branch tails; recorded wall time is shown separately.",
        sorted(cfg.neutral_lineage_visual_dir.glob("*.png")),
        opened=True,
    )
    for directory in cfg.convergence_plot_dirs:
        gallery(
            f"Preparation convergence: {directory.name}",
            "Stationarity, objective components, neutral drift and physical material trajectories. The convergence label is read from the saved checkpoint.",
            sorted(directory.glob("*.png")),
            opened=True,
        )
    for directory in cfg.optimization_plot_dirs:
        gallery(
            f"Optimization trajectory: {directory.name}",
            "Objective terms, per-expression fit, activation fields, mandible poses and physical validity. Stage labels distinguish converged control preparation from final joint trends.",
            sorted(directory.glob("*.png")),
            opened=True,
        )
    gallery(
        "Anatomy and internal tissue sections",
        "Full FEM anatomy, muscle and aponeurosis views. These show model structure; they do not establish anatomical accuracy.",
        sorted(preparation.glob("01-*.png"))
        + sorted(preparation.glob("03-*.png"))
        + sorted(preparation.glob("04-*.png")),
    )
    gallery(
        "Preliminary neutral shapes",
        "Original 1% and 10% stress-proxy pilots, before converged contact-enabled preparation. Shared camera and displacement color limits; overlays use the labeled magnification.",
        sorted(preparation.glob("10-*.png")) + sorted(preparation.glob("11-*.png")),
    )
    gallery(
        "Oral topology and source-model defects",
        "FEM shared-edge contacts are separated from the 17 nonadjacent intersections in the separately registered source lips.",
        sorted(preparation.glob("29-*.png")) + sorted(preparation.glob("30-*.png")),
    )
    body += "<footer>Private tailnet review · tap any figure to inspect the original · experimental model, anatomical validation incomplete.</footer>"
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        "<title>Joint inverse — preparation and final optimization</title><style>"
        + CSS
        + "</style></head><body><main>"
        + body
        + "</main></body></html>"
    )
    temporary = output / "index.html.tmp"
    temporary.write_text(document)
    temporary.replace(output / "index.html")
    write_json(
        output / "status.json",
        {
            "schema": "joint-mobile-review-v1",
            "updated": updated,
            "neutral_preparation_complete": neutral_complete,
            "neutral_runs": neutral,
            "contact_validation": contact,
            "full_face_contact_validation": full_face,
            "newton_full_face_contact_validation": newton_full_face,
            "spatial_full_face_validation": spatial_validation,
            "source_bone_contact_audit": source_bone,
            "fem_contact_surface_audit": fem_contact_audit,
            "control": control,
            "control_latest": control_trace[-1] if control_trace else None,
            "joint": joint,
            "joint_latest": joint_trace[-1] if joint_trace else None,
            "assets": copied,
            "final_joint_is_deliverable": False,
            "required_deliverable": "final simultaneous joint optimization trends",
            "collision_model_revision": collision_revision,
            "collision_off_curvature": collision_off_curvature,
            "prestress_curvature": prestress_curvature,
            "simple_forward": simple_forward,
            "eye_neutral_forward": eye_neutral,
            "eye_neutral_review": eye_neutral_review,
            "eye_neutral_forward_complete": eye_neutral_complete,
            "expression_fit_status": expression_fit_status,
            "expression_fit_visuals": expression_fit_visuals,
            "current_neutral": current_neutral,
            "frozen_neutral_manifest": frozen_neutral_manifest,
            "frozen_neutral_validation": frozen_neutral_validation,
            "simple_forward_interruption": simple_forward_interruption,
            "simple_forward_terminal_audit": simple_forward_terminal_audit,
            "simple_forward_protocol": simple_forward_protocol,
            "simple_forward_heartbeat": simple_forward_heartbeat,
            "simple_forward_heartbeat_age_seconds": simple_forward_heartbeat_age_seconds,
            "simple_forward_heartbeat_is_fresh": simple_forward_heartbeat_is_fresh,
            "simple_forward_run_label": simple_forward_run_label,
            "simple_forward_contact_formula": simple_forward_contact_formula,
            "simple_forward_ccd_clearance": simple_forward_ccd_clearance,
            "simple_forward_armijo": simple_forward_armijo,
            "simple_forward_termination_force": simple_forward_termination_force,
            "simple_forward_solver_detail": simple_forward_solver_detail,
            "simple_forward_visuals": simple_forward_visuals,
            "simple_forward_visual_frame": simple_forward_visual_frame,
            "simple_forward_latest": simple_forward_latest,
            "simple_forward_latest_exact_force": simple_forward_latest_exact_force,
            "simple_skin_field": simple_skin_field,
            "prescribed_skin_field_visuals": load_receipt(
                cfg.skin_field_visual_dir / "summary.json"
            ),
            "full_skull_geometry_review": full_skull_geometry_review,
            "full_skull_contact_microbenchmark": full_skull_microbenchmark,
            "full_skull_model_operation_benchmark": full_skull_operation_benchmark,
            "full_skull_forward_attempts": full_skull_forward_attempts,
            "full_skull_passive_equilibrium": full_skull_passive,
            "full_skull_passive_interruption": full_skull_passive_interruption,
            "full_skull_rigid_separation": full_skull_rigid_separation,
            "legacy_collision_benchmark": legacy_collision_benchmark,
            "parameter_count": parameter_count,
            "parameter_counts": {
                "activation": activation_parameter_count,
                "jaw": jaw_parameter_count,
                "global_skin_stiffness": skin_stiffness_parameter_count,
                "baseline_stress": 0,
            },
            "legacy_shared_coefficient_count": legacy_shared_count,
            "validation_sources": {
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in (
                    ("contact", cfg.contact_validation_path),
                    ("newton_full_face", cfg.newton_face_validation_path),
                )
                if path.is_file()
            },
        },
    )
    LOG.info("Built mobile review with %d images", len(copied))
    if cfg.archive_review:
        cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
