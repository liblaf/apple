# ruff: noqa: EM101, TRY003, PT018
"""Fit every expression with fixed materials, dense tensor stress, and jaw pose."""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import install_expression_runtime
from joint_expression_inputs import EyeExpressionInputs
from joint_fields import (
    VolumeGraph,
    activation_regularizers,
    activation_stresses_mpa,
    project_activation_,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
REFERENCE_MPA = 0.012328767123287673
CAP = 10.0
HINGE_SCALE_RAD = math.pi / 18
HINGE_MIN = 0.0
HINGE_MAX = 4.0


def hinge_pose(jaw: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Map one angle (in 10-degree units) to the backend's rigid pose."""
    assert jaw.shape == (1,) and axis.shape == (3,)
    return torch.cat((jaw[0] * HINGE_SCALE_RAD * axis, torch.zeros_like(axis)))


class Config(cherries.BaseConfig):
    inputs_dir: Path = GROUP / "data/expression-inputs-002"
    validation: Path = (
        GROUP / "data/expression-scale-gradient-validation-001/summary.json"
    )
    derivative_diagnostic: Path = (
        GROUP / "data/expression-residual-diagnostic-001/summary.json"
    )
    hessian_validation: Path = (
        GROUP / "data/expression-derivative-diagnostic-001/summary.json"
    )
    output_dir: Path = GROUP / "data/expression-fitting-007"
    continue_from: Path | None = None
    calibration_source: Path | None = None
    resume: bool = False
    ipc_threads: int | None = None
    maximum_iterations_per_expression: int = 10000
    wall_cap_seconds: float = 172800.0
    learning_rate: float = 0.003
    magnitude_weight: float = 0.001
    jaw_weight: float = 0.01
    neighbor_rms_budget: float = 0.05
    calibration_scale: float = 0.001
    calibration_strength: float = 3.0
    forward_atol: float = 1.5192003475221146e-10
    adjoint_rtol: float = 1e-7
    max_backtracks: int = 12
    armijo: float = 1e-4
    trial_prescreen: bool = False
    outer_step_policy: Literal["armijo", "full_adam"] = "armijo"
    pose_first: bool = False
    pose_collision: bool = True
    coupled_predictor: bool = False
    pose_max_step_deg: float = 1.0
    pose_stationarity_tolerance: float = 1e-3


class TrialObjectiveRejectedError(RuntimeError):
    """A candidate cannot satisfy the raw Armijo ceiling."""

    def __init__(self, *, stage: str, value: float, ceiling: float) -> None:
        assert stage in {"before_forward", "before_adjoint"}
        assert value > ceiling
        self.receipt = {"stage": stage, "value": value, "ceiling": ceiling}
        super().__init__(
            f"trial objective prescreen at {stage}: {value:.9g} > {ceiling:.9g}"
        )


def atomic_torch(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def append(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def neighbor_rms(q: torch.Tensor, graph: VolumeGraph) -> float:
    delta = q[graph.i] - q[graph.j]
    return float(
        (
            (graph.conductance_m * delta.square().sum(-1)).sum()
            / graph.conductance_m.sum()
        ).sqrt()
    )


@torch.no_grad()
def stationarity(
    q: torch.Tensor,
    jaw: torch.Tensor,
    gq: torch.Tensor,
    gj: torch.Tensor,
    mass: torch.Tensor,
) -> dict:
    # Fixed volume metric avoids declaring convergence merely from tiny cells.
    projected = q - gq / mass[:, None]
    project_activation_(projected, CAP)
    mapping = q - projected
    norms = torch.linalg.vector_norm(mapping, dim=-1)
    jaw_mapping = jaw - (jaw - gj).clamp(HINGE_MIN, HINGE_MAX)
    rms = float((mass * norms.square()).sum().sqrt())
    maximum = float(norms.max())
    jaw_max = float(jaw_mapping.abs().max())
    return {
        "metric": "activation normalized effective volume; jaw Euclidean; unit step",
        "activation_mapping_volume_rms": rms,
        "activation_mapping_max_tet": maximum,
        "jaw_mapping_inf": jaw_max,
        "stationary": rms <= 1e-4 and maximum <= 1e-3 and jaw_max <= 1e-3,
    }


@torch.no_grad()
def projected_gradient_direction(
    q: torch.Tensor,
    jaw: torch.Tensor,
    gq: torch.Tensor,
    gj: torch.Tensor,
    mass: torch.Tensor,
    step_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return a feasible descent direction in the declared stationarity metric.

    Each tetrahedron has one scalar mass for all six coordinates, so spectral
    projection is also the mass-metric projection. Projection optimality gives
    g.d <= -sum(mass * dq**2) - sum(djaw**2) before common positive scaling.
    """
    assert math.isfinite(step_cap) and step_cap > 0
    assert q.shape == gq.shape and jaw.shape == gj.shape
    assert mass.shape == q.shape[:-1] and bool((mass > 0).all())
    for value in (q, jaw, gq, gj, mass):
        assert bool(torch.isfinite(value).all())
    projected = q - gq / mass[:, None]
    assert bool(torch.isfinite(projected).all())
    project_activation_(projected, CAP)
    dq = projected - q
    # An unchanged feasible block needs no eigendecomposition round-trip noise.
    dq[(gq == 0).all(dim=-1)] = 0
    dj = (jaw - gj).clamp(HINGE_MIN, HINGE_MAX) - jaw
    maximum = max(float(dq.abs().max()), float(dj.abs().max()))
    scale = min(1.0, step_cap / maximum) if maximum > 0 else 1.0
    dq, dj = scale * dq, scale * dj
    slope = float((gq * dq).sum() + (gj * dj).sum())
    assert math.isfinite(slope)
    return dq, dj, slope, scale


def pose_only_direction(state: dict, max_step_deg: float) -> tuple[float, float]:
    """Bounded scalar descent with positive secant curvature when available."""
    angle = float(state["jaw_normalized"][0])
    gradient = float(state["gradient_jaw"][0])
    assert math.isfinite(angle) and math.isfinite(gradient)
    assert HINGE_MIN <= angle <= HINGE_MAX and max_step_deg > 0
    inverse_curvature = 1.0
    previous = state.get("pose_previous")
    if previous is not None:
        delta = angle - previous["angle"]
        gradient_delta = gradient - previous["gradient"]
        if delta * gradient_delta > 0:
            inverse_curvature = delta / gradient_delta
    target = min(HINGE_MAX, max(HINGE_MIN, angle - inverse_curvature * gradient))
    cap = max_step_deg / 10
    direction = min(cap, max(-cap, target - angle))
    assert math.isfinite(direction) and math.isfinite(inverse_curvature)
    return direction, inverse_curvature


def compare_continuation_protocol(parent: dict, current: dict) -> dict:
    """Reject changes outside the recorded scheduling/descent-policy revisions."""
    before, after = copy.deepcopy(parent), copy.deepcopy(current)
    old_optimizer = "per-expression projected Adam; transactional Armijo; explicit non-descent momentum restart; round-robin all36"
    previous_optimizer = "per-expression projected Adam; transactional Armijo; explicit non-descent momentum restart; sequential until convergence or explicit line-search failure"
    new_optimizer = "per-expression projected Adam with volume-metric projected-gradient descent safeguard; transactional Armijo; sequential until convergence or explicit failure"
    assert before["optimizer"] in {old_optimizer, previous_optimizer, new_optimizer}
    assert after["optimizer"] == new_optimizer
    if "schedule" in before:
        old_schedule, new_schedule = (
            copy.deepcopy(before["schedule"]),
            copy.deepcopy(after["schedule"]),
        )
        old_terminal = old_schedule.pop("advance_only_after")
        new_terminal = new_schedule.pop("advance_only_after")
        assert old_terminal in (["converged", "line_search_failed"], new_terminal)
        assert old_schedule == new_schedule
    if "descent_policy" in before:
        assert before["descent_policy"] == after["descent_policy"]
    # Older protocols predate the opt-in thread control.  A continuation that
    # keeps the default has no thread choice to compare, so admit that one
    # missing receipt without inventing an effective legacy thread count.
    if (
        "ipctk_threads" not in before
        and "ipctk_threads" in after
        and after["ipctk_threads"]["requested"] is None
    ):
        after.pop("ipctk_threads")
        after["config"].pop("ipc_threads")
    changes = {
        key: {"parent": parent.get(key), "current": current.get(key)}
        for key in parent.keys() | current.keys()
        if parent.get(key) != current.get(key)
    }
    for protocol in (before, after):
        for key in ("optimizer", "schedule", "continuation", "descent_policy"):
            protocol.pop(key, None)
        protocol["implementation_sha256"].pop("93-fit-expressions.py")
        for key in ("output_dir", "continue_from", "calibration_source"):
            protocol["config"].pop(key, None)
    assert before == after, {
        "incompatible_protocol_keys": [
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        ]
    }
    return changes


def verify_continuation_source_change(parent: Path, current: Path) -> dict:
    """Admit orchestration changes only when objective/physical methods match."""
    parent_tree, current_tree = (
        ast.parse(path.read_text()) for path in (parent, current)
    )

    def physical_nodes(tree: ast.Module) -> dict[str, str]:
        nodes = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {
                "main",
                "compare_continuation_protocol",
                "verify_schedule_only_source_change",
                "verify_continuation_source_change",
                "import_continuation",
                "projected_gradient_direction",
            }:
                nodes[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.Assign):
                nodes[ast.unparse(node.targets[0])] = ast.dump(
                    node, include_attributes=False
                )
            elif isinstance(node, ast.ClassDef) and node.name == "Fitter":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name not in {
                        "__init__",
                        "run",
                        "fit_step",
                        "finish_without_update",
                    }:
                        nodes[f"Fitter.{method.name}"] = ast.dump(
                            method, include_attributes=False
                        )
        return nodes

    before, after = physical_nodes(parent_tree), physical_nodes(current_tree)
    assert before == after, {
        "changed_physics_or_optimizer_nodes": [
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        ]
    }
    return {
        "parent_sha256": sha256(parent),
        "current_sha256": sha256(current),
        "unchanged_ast_nodes": sorted(before),
        "change_scope": "recorded scheduler/descent and termination changes; physical methods, objective, hinge mapping, stationarity metric and gradients unchanged",
    }


def import_continuation(cfg: Config, protocol: dict) -> dict:  # noqa: C901, PLR0915
    """Copy a stopped run's accepted state without mutating its source files."""
    assert cfg.continue_from is not None
    source = cfg.continue_from.resolve()
    target = cfg.output_dir.resolve()
    assert source != target and source.is_dir()
    parent = json.loads((source / "protocol.json").read_text())
    changes = compare_continuation_protocol(parent, protocol)
    old_runner = source / "sources/experiment/93-fit-expressions.py"
    assert (
        sha256(old_runner) == parent["implementation_sha256"]["93-fit-expressions.py"]
    )
    source_scope = verify_continuation_source_change(old_runner, Path(__file__))
    status = json.loads((source / "status.json").read_text())
    assert not status["running"], "parent run must be stopped before continuation"
    assert list(status["expressions"]) == protocol["schedule"]["expression_order"]
    calibration = json.loads((source / "calibration.json").read_text())
    assert calibration["success"]
    assert calibration["expression_count"] == len(status["expressions"])
    assert calibration["reference_neighbor_rms"] == cfg.neighbor_rms_budget
    assert calibration["probe_scale"] == cfg.calibration_scale
    assert calibration["strength_factor"] == cfg.calibration_strength
    imports = {}

    def copy_file(relative: Path, destination: Path | None = None) -> None:
        origin = source / relative
        destination = relative if destination is None else destination
        output = target / destination
        digest = sha256(origin)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, output)
        assert sha256(origin) == digest == sha256(output), str(origin)
        imports[str(relative)] = {
            "source_path": str(origin),
            "destination": str(destination),
            "sha256": digest,
        }

    for name in (
        "calibration.json",
        "neutral-numerical-refinement.pt",
        "neutral-numerical-refinement.json",
    ):
        copy_file(Path(name))
    for name in ("calibration-probe.pt", "calibration.jsonl"):
        if (source / name).exists():
            copy_file(Path(name))
    for name in (
        "protocol.json",
        "status.json",
        "provenance.json",
        "summary.json",
        "external-interruption.json",
    ):
        if (source / name).exists():
            copy_file(Path(name), Path("continuation-parent") / name)
    for name, digest in parent["implementation_sha256"].items():
        relative = Path("sources/experiment") / name
        assert sha256(source / relative) == digest, name
        copy_file(relative, Path("continuation-parent/sources") / name)
    accepted = {}
    for name, entry in status["expressions"].items():
        directory = source / "expressions" / name
        if not directory.exists():
            assert entry["accepted_steps"] == 0 and entry["status"] == "queued", entry
            continue
        contents = sorted(directory.iterdir())
        if not contents:
            assert entry["accepted_steps"] == 0 and entry["status"] == "queued", entry
            continue
        for path in contents:
            assert path.is_file() and (
                path.suffix == ".pt"
                or path.name in {"trace.jsonl", "trials.jsonl", "termination.json"}
            ), str(path)
            copy_file(path.relative_to(source))
        state = torch.load(
            target / "expressions" / name / "latest.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert state["jaw_normalized"].shape == (1,) and state[
            "gradient_jaw"
        ].shape == (1,)
        if state["accepted_steps"]:
            assert "optimizer" in state and state["expression"] == name
        trace = [
            json.loads(line)
            for line in (directory / "trace.jsonl").read_text().splitlines()
        ]
        assert trace[-1]["accepted_steps"] == state["accepted_steps"], name
        assert entry["accepted_steps"] == state["accepted_steps"], name
        expression_status = (
            "converged"
            if state.get("inverse_converged", False)
            else (
                entry["status"]
                if entry["status"]
                in {
                    "line_search_failed",
                    "stationary_primal_unresolved",
                    "descent_unresolved",
                }
                else "fitting"
            )
        )
        status["expressions"][name] = {
            **entry,
            **state["metrics"],
            "status": expression_status,
            "accepted_steps": state["accepted_steps"],
        }
        accepted[name] = {
            "accepted_steps": state["accepted_steps"],
            "optimizer_preserved": "optimizer" in state,
            "qualifying_consecutive": state["qualifying_consecutive"],
            "latest_checkpoint_sha256": imports[f"expressions/{name}/latest.pt"][
                "sha256"
            ],
        }
    # Recheck every source after copying to reject a concurrently changing parent.
    for record in imports.values():
        assert sha256(Path(record["source_path"])) == record["sha256"], record[
            "source_path"
        ]
    status["schedule"] = "sequential"
    status["running"] = False
    status["phase"] = "continuation_imported"
    status.pop("round", None)
    write_json(target / "status.json", status)
    receipt = {
        "schema": "expression-schedule-continuation-v1",
        "source_directory": str(source),
        "transition": "continue accepted states and Adam moments under the recorded scheduling/descent policy",
        "parent_protocol_sha256": sha256(source / "protocol.json"),
        "parent_source_sha256": parent["implementation_sha256"],
        "protocol_changes": changes,
        "source_scope": source_scope,
        "imported_files": imports,
        "accepted_expressions": accepted,
        "calibration_reused_exactly": True,
        "parent_files_unchanged": True,
    }
    write_json(target / "continuation.json", receipt)
    return {
        "path": str(target / "continuation.json"),
        "sha256": sha256(target / "continuation.json"),
        "source_directory": str(source),
        "parent_protocol_sha256": receipt["parent_protocol_sha256"],
        "transition": receipt["transition"],
    }


class Fitter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.started = time.perf_counter()
        self.inputs = EyeExpressionInputs.load(cfg.inputs_dir)
        self.physics, _ = self.inputs.build_physics()
        # Install after setting the desired adjoint accuracy, including solver.
        self.physics.runtime.tolerances["adjoint_rtol"] = cfg.adjoint_rtol
        self.physics.runtime.tolerances["atol"] = cfg.forward_atol
        self.runtime = install_expression_runtime(self.physics)
        self.rigid_contact = self.runtime.forward.model.collision
        self.contact_enabled = True
        arrays = self.inputs.arrays
        self.names = self.inputs.expression_names
        assert len(self.names) == 36
        self.neutral = torch.as_tensor(arrays["neutral_displacement_m"])
        self.obs = torch.as_tensor(arrays["observation_node_ids"], dtype=torch.long)
        self.weights = torch.as_tensor(arrays["observation_weight_normalized"])
        self.targets = torch.as_tensor(arrays["target_total_displacement_m"])
        increments = torch.as_tensor(arrays["expression_displacement_m"])
        self.scales = (self.weights[None] * increments.square().sum(-1)).sum(-1)
        assert bool((self.scales > 0).all())
        self.graph = VolumeGraph(
            torch.as_tensor(arrays["graph_i"], dtype=torch.long),
            torch.as_tensor(arrays["graph_j"], dtype=torch.long),
            torch.as_tensor(arrays["graph_conductance_m"]),
            torch.as_tensor(arrays["active_effective_volume_m3"]),
            0.005,
        )
        self.mass = self.graph.effective_cell_volume_m3 / self.graph.tissue_volume_m3
        assert bool((self.mass > 0).all())
        self.hinge_axis = torch.as_tensor(arrays["mandible_frame_world"][:, 0])
        self.hinge_pivot = torch.as_tensor(arrays["mandible_pivot_m"])
        assert self.hinge_pivot.shape == (3,) and bool(
            torch.isfinite(self.hinge_pivot).all()
        )
        assert bool(torch.isfinite(self.hinge_axis).all())
        assert abs(float(torch.linalg.vector_norm(self.hinge_axis)) - 1) < 1e-12
        self.status = {
            "schema": "joint-fixed-material-expression-fitting-v1",
            "stage": "fixed_material_expression_fitting",
            "running": True,
            "inverse_converged": False,
            "final_joint_is_deliverable": False,
            "expression_count": len(self.names),
            "activation_coordinates_per_expression": int(len(self.mass) * 6),
            "mandible_dofs_per_expression": 1,
            "mandible_angle_bounds_deg": [0.0, 40.0],
            "mandible_axis_world": self.hinge_axis.cpu().tolist(),
            "material_parameters_optimized": False,
            "pose_first": cfg.pose_first,
            "pose_collision": cfg.pose_collision,
            "coupled_predictor": cfg.coupled_predictor,
            "joint_collision": True,
            "fit_stages": ["pose_only", "joint"] if cfg.pose_first else ["joint"],
            "forward_force_threshold_code": cfg.forward_atol,
            "expressions": {
                name: {"status": "queued", "accepted_steps": 0} for name in self.names
            },
        }
        self.elapsed_offset = 0.0
        if cfg.resume or cfg.continue_from is not None:
            self.status = json.loads((cfg.output_dir / "status.json").read_text())
            assert self.status["mandible_dofs_per_expression"] == 1
            self.elapsed_offset = self.status["elapsed_seconds"]
            self.status["running"] = True
            self.status.pop("failure", None)
            self.status.pop("error_type", None)
        self.status["schedule"] = "sequential"
        self.status.pop("round", None)
        self.smooth_weight = None
        self.neutral_seed = self.neutral

    def refine_neutral(self):
        path = self.cfg.output_dir / "neutral-numerical-refinement.pt"
        if path.exists():
            self.neutral_seed = torch.load(
                path, map_location=self.neutral.device, weights_only=False
            )["displacement_m"]
            return
        self.publish("refining_neutral_to_expression_force_tolerance")
        neutral_q = torch.zeros((len(self.mass), 6))
        self.neutral_seed = self.solve_from_state(
            neutral_q,
            torch.zeros(1),
            self.neutral,
            torch.zeros(1),
            neutral_q,
            "neutral-numerical-refinement",
        ).detach()
        delta = self.neutral_seed - self.neutral
        receipt = {
            "purpose": "numerical force refinement only; target origin and material state unchanged",
            "maximum_node_change_mm": float(
                torch.linalg.vector_norm(delta, dim=-1).max()
            )
            * 1000,
            "surface_change_rms_mm": float(
                (self.weights * delta[self.obs].square().sum(-1)).sum().sqrt()
            )
            * 1000,
            "forward": copy.deepcopy(self.runtime.last_forward),
        }
        atomic_torch(
            path, cpu_tree({"displacement_m": self.neutral_seed, "receipt": receipt})
        )
        write_json(path.with_suffix(".json"), receipt)

    def publish(self, phase: str, expression: str | None = None):
        self.status.update(
            {
                "phase": phase,
                "current_expression": expression,
                "elapsed_seconds": self.elapsed_offset
                + time.perf_counter()
                - self.started,
            }
        )
        write_json(self.cfg.output_dir / "status.json", self.status)

    def data_loss(self, u: torch.Tensor, index: int):
        error2 = (u[self.obs] - self.targets[index]).square().sum(-1)
        mse = (self.weights * error2).sum()
        return mse / self.scales[index], mse

    def contact_gate(self):
        forward = self.runtime.forward
        collision = forward.model.collision
        receipt = self.runtime.last_forward
        assert receipt["success"], receipt
        if not self.cfg.pose_collision and not self.contact_enabled:
            assert collision is None and receipt["contact"]["enabled"] is False
            return
        assert receipt["contact"]["contact_numerically_valid"], receipt
        gap = receipt["contact"]["minimum_active_distance_m"]
        if gap is not None and gap < collision.min_distance:
            raise ForwardConvergenceError(
                "terminal gap below CCD buffer", receipt=receipt
            )
        positions = (collision.vertices + forward.state.u[collision.indices]).numpy(
            force=True
        )
        if ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH()):
            raise ForwardConvergenceError(
                "terminal soft-rigid intersection", receipt=receipt
            )

    def set_collision_enabled(self, enabled: bool) -> None:  # noqa: FBT001
        """Switch the entire contact energy/derivative model between stages."""
        if enabled == self.contact_enabled:
            return
        self.runtime.warm_adjoints.clear()
        model = self.runtime.forward.model
        self.runtime.forward.state.collision = None
        if enabled:
            model.collision = self.rigid_contact
            self.runtime = install_expression_runtime(self.physics)
        else:
            from joint_collision_off_pose import install_collision_off_pose_runtime

            self.runtime = install_collision_off_pose_runtime(self.physics)
        assert self.physics.runtime is self.runtime
        self.contact_enabled = enabled

    def pose_contact_audit(self, state: dict) -> dict:
        """Audit candidate geometry against full source skull, mandible and eyes."""
        collision = self.rigid_contact
        full = self.physics.full_skull.extend_seed(
            state["displacement_m"],
            hinge_pose(state["jaw_normalized"], self.hinge_axis),
        )
        positions = (collision.vertices + full[collision.indices]).numpy(force=True)
        intersects = bool(
            ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
        )
        if intersects:
            return {"geometric_seed_feasible": False, "soft_rigid_intersections": True}
        contact_state = collision.state_at(full)
        gap = None
        if len(contact_state.collisions):
            distance2 = contact_state.collisions.compute_minimum_distance(
                collision.collision_mesh, positions
            )
            gap = math.sqrt(float(distance2))
        return {
            "geometric_seed_feasible": gap is None
            or (math.isfinite(gap) and gap >= collision.min_distance),
            "soft_rigid_intersections": False,
            "minimum_active_distance_m": gap,
            "minimum_required_distance_m": collision.min_distance,
        }

    def prepare_contact_joint(self, index: int, path: Path, state: dict) -> dict | None:
        """Require a new contact-on equilibrium and adjoint before unlocking stress."""
        name = self.names[index]
        assert not self.cfg.pose_collision and not self.contact_enabled
        assert not bool(torch.count_nonzero(state["activation"]))
        self.publish("checking_contact_handoff", name)
        audit = self.pose_contact_audit(state)
        receipt = {
            "collision_off_checkpoint_sha256": sha256(path),
            "geometry": audit,
            "joint_started": False,
        }
        if not audit["geometric_seed_feasible"]:
            write_json(path.parent / "contact-handoff.json", receipt)
            self.finish_without_update(
                name,
                path,
                state,
                "requires_contact_recovery",
                "collision-off pose needs contact recovery before joint fitting",
                contact_handoff=receipt,
            )
            return None
        self.set_collision_enabled(True)
        self.publish("equilibrating_contact_handoff", name)
        try:
            candidate = self.evaluate_from_state(
                index,
                state["activation"].detach().clone().requires_grad_(),
                state["jaw_normalized"].detach().clone().requires_grad_(),
                state,
            )
        except ForwardConvergenceError as error:
            receipt.update(failure=str(error), forward_failure=error.receipt)
            write_json(path.parent / "contact-handoff.json", receipt)
            self.finish_without_update(
                name,
                path,
                state,
                "requires_contact_recovery",
                "contact-on handoff equilibrium failed; collision-off checkpoint retained",
                contact_handoff=receipt,
            )
            return None
        candidate["metrics"]["initial_fit_rms_mm"] = state["metrics"][
            "initial_fit_rms_mm"
        ]
        receipt.update(
            contact_equilibrium_converged=True,
            collision_on_fit_rms_mm=candidate["metrics"]["fit_rms_mm"],
            collision_off_fit_rms_mm=state["metrics"]["fit_rms_mm"],
            full_gradients_recomputed=True,
        )
        write_json(path.parent / "contact-handoff.json", receipt)
        return {**state, **candidate, "contact_handoff": receipt}

    def solve_from_state(
        self,
        q: torch.Tensor,
        jaw: torch.Tensor,
        seed: torch.Tensor,
        seed_jaw: torch.Tensor,
        seed_q: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Solve one target from an explicitly accepted inverse state."""
        if getattr(self.cfg, "coupled_predictor", False):
            return self.solve(q, jaw, seed, seed_jaw, key, seed_q=seed_q)
        return self.solve(q, jaw, seed, seed_jaw, key)

    def evaluate_from_state(
        self,
        index: int,
        q: torch.Tensor,
        jaw: torch.Tensor,
        state: dict,
        *,
        objective_ceiling: float | None = None,
    ) -> dict:
        """Evaluate a target using only the checkpointed accepted seed fields."""
        kwargs = {}
        if objective_ceiling is not None:
            kwargs["objective_ceiling"] = objective_ceiling
        if getattr(self.cfg, "coupled_predictor", False):
            kwargs["seed_q"] = state["activation"]
        return self.evaluate(
            index, q, jaw, state["displacement_m"], state["jaw_normalized"], **kwargs
        )

    def solve(
        self,
        q: torch.Tensor,
        jaw: torch.Tensor,
        seed: torch.Tensor,
        seed_jaw: torch.Tensor,
        key: str,
        *,
        seed_q: torch.Tensor | None = None,
    ):
        target_stress = activation_stresses_mpa(q, REFERENCE_MPA)
        target_pose = hinge_pose(jaw, self.hinge_axis)
        accepted_pose = hinge_pose(seed_jaw, self.hinge_axis)
        corrected_seed = seed
        coupled_receipt = None
        if getattr(self.cfg, "coupled_predictor", False):
            assert self.cfg.pose_collision
            assert seed_q is not None and seed_q.shape == q.shape
            # The continuation is local to this outer Armijo candidate.  It
            # returns a detached initial geometry; the corrector below keeps
            # the original target tensors for the implicit derivative.
            from joint_coupled_continuation import prepare_expression_seed

            old_stress = activation_stresses_mpa(seed_q.detach(), REFERENCE_MPA)
            corrected_seed, coupled_receipt = prepare_expression_seed(
                self.physics,
                old_stress=old_stress,
                new_stress=target_stress.detach(),
                old_pose=accepted_pose.detach(),
                new_pose=target_pose.detach(),
                seed=seed.detach(),
                axis=self.hinge_axis.detach(),
                pivot=self.hinge_pivot.detach(),
                linear_rtol=self.cfg.adjoint_rtol,
            )
        u = self.physics.solve(
            skin_multiplier=torch.ones(()),
            active_stress=target_stress,
            pose=target_pose,
            seed=corrected_seed.detach(),
            # Legacy solves use the accepted jaw boundary for trial CCD;
            # predictor seeds already contain the target rigid boundary.
            seed_pose=target_pose if coupled_receipt is not None else accepted_pose,
            key=key,
        )
        if coupled_receipt is not None:
            self.runtime.last_forward = {
                **self.runtime.last_forward,
                "coupled_predictor": cpu_tree(coupled_receipt),
            }
        self.contact_gate()
        return u

    def calibrate(self):  # noqa: PLR0915 - keep calibration and its reuse receipt together
        path = self.cfg.output_dir / "calibration.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            assert receipt["success"] and receipt["expression_count"] == len(self.names)
            self.smooth_weight = receipt["strong_weight"]
            return
        if self.cfg.calibration_source is not None:
            source = self.cfg.calibration_source
            receipt = json.loads((source / "calibration.json").read_text())
            protocol = json.loads((source / "protocol.json").read_text())
            assert receipt["success"] and receipt["expression_count"] == len(self.names)
            assert "reference_neighbor_rms" not in receipt
            assert protocol["inputs_manifest_sha256"] == sha256(
                self.cfg.inputs_dir / "manifest.json"
            )
            assert receipt["probe_scale"] == self.cfg.calibration_scale
            assert receipt["strength_factor"] == self.cfg.calibration_strength
            probe = torch.load(
                source / "calibration-probe.pt",
                map_location=self.neutral.device,
                weights_only=False,
            )
            probe_roughness = neighbor_rms(probe["activation"], self.graph)
            assert 0 < probe_roughness < self.cfg.neighbor_rms_budget
            receipt["unscaled_probe_weight"] = receipt["strong_weight"]
            receipt["strong_weight"] *= probe_roughness / self.cfg.neighbor_rms_budget
            receipt.update(
                {
                    "reference_neighbor_rms": self.cfg.neighbor_rms_budget,
                    "reuse_scope": "activation gradients at the same physical zero jaw pose; source jaw-gradient diagnostics describe the former six-coordinate parameterization and are not used for calibration",
                    "probe_neighbor_rms": probe_roughness,
                    "normalization": "smoothness gradient evaluated analytically at the declared neighbor-RMS budget; quadratic homogeneity removes arbitrary feasible-probe amplitude",
                    "source_receipts": {
                        name: {
                            "path": str(source / name),
                            "sha256": sha256(source / name),
                        }
                        for name in (
                            "calibration.json",
                            "calibration-probe.pt",
                            "protocol.json",
                        )
                    },
                }
            )
            self.smooth_weight = receipt["strong_weight"]
            self.status["calibration_completed_expressions"] = len(self.names)
            write_json(path, receipt)
            LOG.info(
                "Reused 36 calibrated adjoints: probe roughness %.9g; reference %.6g; weight %.9g",
                probe_roughness,
                self.cfg.neighbor_rms_budget,
                self.smooth_weight,
            )
            return
        self.publish("calibrating_strong_smoothness")
        points = torch.as_tensor(self.physics.points)
        centers = (points + self.neutral)[self.physics.muscle_tets_t].mean(1)
        q = torch.zeros((len(self.mass), 6))
        # Nonuniform positive diagonal field with a known physical 5 mm length.
        for axis in range(3):
            q[:, axis] = self.cfg.calibration_scale * (
                1 + 0.25 * torch.sin(centers[:, axis] / 0.005)
            )
        q.requires_grad_()
        smooth = activation_regularizers(q, self.graph)["smoothness"]
        gs = torch.autograd.grad(smooth, q)[0]
        smooth_rms = float(gs.square().mean().sqrt())
        assert smooth_rms > 0 and math.isfinite(smooth_rms)
        assert neighbor_rms(q, self.graph) <= self.cfg.neighbor_rms_budget
        jaw = torch.zeros(1, requires_grad=True)
        calibration_seed_q = (
            torch.zeros_like(q)
            if getattr(self.cfg, "coupled_predictor", False)
            else None
        )
        if calibration_seed_q is None:
            u = self.solve(q, jaw, self.neutral_seed, torch.zeros(1), "calibration")
        else:
            u = self.solve(
                q,
                jaw,
                self.neutral_seed,
                torch.zeros(1),
                "calibration",
                seed_q=calibration_seed_q,
            )
        forward_receipt = copy.deepcopy(self.runtime.last_forward)
        atomic_torch(
            self.cfg.output_dir / "calibration-probe.pt",
            cpu_tree(
                {
                    "activation": q,
                    "jaw_normalized": jaw,
                    "displacement_m": u,
                    "smoothness_gradient": gs,
                }
            ),
        )
        rows = []
        for index, name in enumerate(self.names):
            self.publish("calibrating_strong_smoothness", name)
            loss, _ = self.data_loss(u, index)
            gq, gj = torch.autograd.grad(loss, (q, jaw), retain_graph=True)
            rms = float(gq.square().mean().sqrt())
            assert rms > 0 and math.isfinite(rms)
            row = {
                "expression": name,
                "data": float(loss),
                "data_gradient_rms": rms,
                "smooth_gradient_rms": smooth_rms,
                "ratio": rms / smooth_rms,
                "jaw_gradient_inf": float(gj.abs().max()),
                "adjoint": copy.deepcopy(self.runtime.last_adjoint),
            }
            rows.append(row)
            self.status["calibration_completed_expressions"] = len(rows)
            append(self.cfg.output_dir / "calibration.jsonl", row)
            LOG.info(
                "Calibration %d/%d %s ratio %.6g",
                index + 1,
                len(self.names),
                name,
                row["ratio"],
            )
        # Repeat the strongest ratio at tighter adjoint tolerance before freezing.
        worst = max(range(len(rows)), key=lambda index: rows[index]["ratio"])
        from face_physics import SuccessPreferredFallbackSolver

        from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes

        old_solver = self.runtime.solver
        tighter = self.cfg.adjoint_rtol * 0.1
        self.runtime.solver = SuccessPreferredFallbackSolver(
            [
                CupyCG(maxiter=10000, rtol=tighter, atol=0.0),
                CupyMinRes(maxiter=10000, tol=tighter),
            ]
        )
        self.runtime.tolerances["adjoint_rtol"] = tighter
        loss, _ = self.data_loss(u, worst)
        gq = torch.autograd.grad(loss, q)[0]
        tightened_ratio = float(gq.square().mean().sqrt()) / smooth_rms
        relative_change = abs(tightened_ratio / rows[worst]["ratio"] - 1)
        assert relative_change <= 0.05, relative_change
        self.runtime.solver = old_solver
        self.runtime.tolerances["adjoint_rtol"] = self.cfg.adjoint_rtol
        self.runtime.warm_adjoints.clear()
        probe_weight = self.cfg.calibration_strength * max(
            tightened_ratio, *(row["ratio"] for row in rows)
        )
        probe_roughness = neighbor_rms(q.detach(), self.graph)
        self.smooth_weight = (
            probe_weight * probe_roughness / self.cfg.neighbor_rms_budget
        )
        write_json(
            path,
            {
                "success": True,
                "expression_count": len(rows),
                "rows": rows,
                "probe_forward": forward_receipt,
                "probe_scale": self.cfg.calibration_scale,
                "strength_factor": self.cfg.calibration_strength,
                "strong_weight": self.smooth_weight,
                "unscaled_probe_weight": probe_weight,
                "reference_neighbor_rms": self.cfg.neighbor_rms_budget,
                "probe_neighbor_rms": probe_roughness,
                "tightened_expression": self.names[worst],
                "tightened_ratio": tightened_ratio,
                "tightened_ratio_relative_change": relative_change,
                "normalization": "smoothness gradient evaluated analytically at the declared neighbor-RMS budget; each independent expression objective; mean over 36 cancels in ratio",
            },
        )

    def evaluate(
        self,
        index: int,
        q: torch.Tensor,
        jaw: torch.Tensor,
        seed: torch.Tensor,
        seed_jaw: torch.Tensor,
        *,
        objective_ceiling: float | None = None,
        seed_q: torch.Tensor | None = None,
    ):
        name = self.names[index]
        reg = activation_regularizers(q, self.graph)
        smooth = self.smooth_weight * reg["smoothness"]
        magnitude = self.cfg.magnitude_weight * reg["magnitude"]
        # Preserve the former six-coordinate prior restricted to this unit axis.
        pose = self.cfg.jaw_weight * jaw.square().sum() / 6
        prior = smooth + magnitude + pose
        if objective_ceiling is not None and float(prior.detach()) > objective_ceiling:
            raise TrialObjectiveRejectedError(
                stage="before_forward",
                value=float(prior.detach()),
                ceiling=objective_ceiling,
            )
        if getattr(self.cfg, "coupled_predictor", False):
            assert seed_q is not None
            u = self.solve(q, jaw, seed, seed_jaw, name, seed_q=seed_q)
        else:
            # Keep legacy five-positional solve mocks usable in CPU fixtures.
            u = self.solve(q, jaw, seed, seed_jaw, name)
        data, mse = self.data_loss(u, index)
        objective = data + smooth + magnitude + pose
        if (
            objective_ceiling is not None
            and float(objective.detach()) > objective_ceiling
        ):
            raise TrialObjectiveRejectedError(
                stage="before_adjoint",
                value=float(objective.detach()),
                ceiling=objective_ceiling,
            )
        gq, gj = torch.autograd.grad(objective, (q, jaw))
        assert bool(torch.isfinite(gq).all() and torch.isfinite(gj).all())
        # The implicit adjoint uses p = -H_ff^{-1} L_f. This estimates the
        # remaining objective error without changing the accepted objective.
        residual = self.runtime.last_problem.grad(self.runtime.forward.state)
        adjoint = self.runtime.warm_adjoints[name]
        primal_correction = float(torch.dot(adjoint, residual))
        roughness = neighbor_rms(q.detach(), self.graph)
        # The Armijo policy treats this as a trial feasibility constraint.  A
        # diagnostic full Adam step records roughness but deliberately does not
        # replace its single requested proposal with a smaller one.
        if getattr(self.cfg, "outer_step_policy", "armijo") == "armijo":
            assert roughness <= self.cfg.neighbor_rms_budget
        metrics = {
            "objective": float(objective),
            "primal_objective_correction_estimate": primal_correction,
            "residual_corrected_objective_estimate": float(objective)
            + primal_correction,
            "data": float(data),
            "fit_rms_mm": float(mse.sqrt()) * 1000,
            "target_motion_rms_mm": float(self.scales[index].sqrt()) * 1000,
            "smoothness": float(reg["smoothness"]),
            "weighted_smoothness": float(smooth),
            "magnitude": float(reg["magnitude"]),
            "weighted_magnitude": float(magnitude),
            "weighted_jaw_prior": float(pose),
            "neighbor_rms": roughness,
            "activation_tensor_rms_kpa": float(reg["magnitude"].sqrt())
            * REFERENCE_MPA
            * 1000,
            "jaw_pose_rad_m": hinge_pose(jaw.detach(), self.hinge_axis).cpu().tolist(),
            "mandible_angle_deg": float(jaw.detach()[0]) * 10,
            "mandible_angle_rad": float(jaw.detach()[0]) * HINGE_SCALE_RAD,
            "stationarity": stationarity(q, jaw, gq, gj, self.mass),
            "forward": copy.deepcopy(self.runtime.last_forward),
            "coupled_predictor": copy.deepcopy(
                self.runtime.last_forward.get("coupled_predictor")
            ),
            "adjoint": copy.deepcopy(self.runtime.last_adjoint),
            "shape": self.physics.metrics(u.detach(), target_index=index),
            "elapsed_seconds": self.elapsed_offset + time.perf_counter() - self.started,
        }
        return {
            "displacement_m": u.detach().clone(),
            "gradient_q": gq.detach(),
            "gradient_jaw": gj.detach(),
            "metrics": metrics,
        }

    def finish_without_update(
        self,
        name: str,
        path: Path,
        state: dict,
        status: str,
        reason: str,
        **details: Any,
    ) -> None:
        """Record a terminal certificate/failure without inventing an update."""
        termination = {"status": status, "reason": reason, **details}
        write_json(path.parent / "termination.json", termination)
        if status == "converged":
            state = {**state, "inverse_converged": True, "termination": termination}
            atomic_torch(path, cpu_tree(state))
        self.status["expressions"][name] = {
            **state["metrics"],
            "status": status,
            "accepted_steps": state["accepted_steps"],
            "termination": termination,
            "pose_converged": state.get("pose_converged", False),
            "pose_accepted_steps": state.get("pose_accepted_steps", 0),
            "joint_accepted_steps": state.get(
                "joint_accepted_steps", state["accepted_steps"]
            ),
        }
        self.publish("fitting", name)
        LOG.info("%s terminated without an update: %s", name, reason)

    def pose_step(self, index: int, path: Path, state: dict):  # noqa: C901, PLR0912, PLR0915
        """Fit the scalar jaw with identically zero muscle activation."""
        name = self.names[index]
        assert not bool(torch.count_nonzero(state["activation"]))
        jaw0 = state["jaw_normalized"]
        gradient = state["gradient_jaw"]
        assert jaw0.shape == gradient.shape == (1,)
        assert bool(torch.isfinite(gradient).all())
        mapping = float(
            (jaw0 - (jaw0 - gradient).clamp(HINGE_MIN, HINGE_MAX)).abs().max()
        )
        metrics = state["metrics"]
        resolved = abs(metrics["primal_objective_correction_estimate"]) <= (
            1e-6 * max(1.0, abs(metrics["objective"]))
        )
        if mapping <= self.cfg.pose_stationarity_tolerance:
            if not resolved:
                self.finish_without_update(
                    name,
                    path,
                    state,
                    "stationary_primal_unresolved",
                    "pose-only stationarity; primal estimate unresolved",
                    fit_stage="pose_only",
                    jaw_mapping_inf=mapping,
                )
                return
            # Collision-on stages share gradients. An off-to-on handoff must
            # instead solve and differentiate the restored contact model anew.
            certificate = {
                "pose_converged": True,
                "criterion": "first-order projected jaw stationarity with resolved primal estimate; not a global minimum certificate",
                "jaw_mapping_inf": mapping,
                "jaw_mapping_tolerance": self.cfg.pose_stationarity_tolerance,
                "primal_error_resolved": True,
                "muscle_activation_identically_zero": True,
                "accepted_steps": state["accepted_steps"],
                "pose_accepted_steps": state["pose_accepted_steps"],
                "mandible_angle_deg": float(jaw0[0]) * 10,
                "fit_rms_mm": metrics["fit_rms_mm"],
                "collision_enabled": self.cfg.pose_collision,
            }
            final_path = path.parent / "pose-only-final.pt"
            atomic_torch(
                final_path, cpu_tree({**state, "pose_certificate": certificate})
            )
            certificate["checkpoint_sha256"] = sha256(final_path)
            write_json(path.parent / "pose-stage.json", certificate)
            if not self.cfg.pose_collision:
                state = {
                    **state,
                    "pose_converged": True,
                    "pose_certificate": certificate,
                }
                prepared = self.prepare_contact_joint(index, path, state)
                if prepared is None:
                    return
                state = prepared
                metrics = state["metrics"]
            joint = {
                **state,
                "fit_stage": "joint",
                "pose_converged": True,
                "pose_certificate": certificate,
                "joint_accepted_steps": 0,
                "inverse_converged": False,
                "history": [],
                "qualifying_consecutive": 0,
                "metrics": {**metrics, "fit_stage": "joint"},
            }
            joint.pop("optimizer", None)
            joint.pop("pose_previous", None)
            if "contact_handoff" in joint:
                joint["contact_handoff"] = {
                    **joint["contact_handoff"],
                    "joint_started": True,
                }
            atomic_torch(path.parent / "joint-initial.pt", cpu_tree(joint))
            atomic_torch(path, cpu_tree(joint))
            if "contact_handoff" in joint:
                write_json(
                    path.parent / "contact-handoff.json", joint["contact_handoff"]
                )
            self.status["expressions"][name] = {
                **joint["metrics"],
                "status": "initialized",
                "accepted_steps": joint["accepted_steps"],
                "pose_accepted_steps": joint["pose_accepted_steps"],
                "joint_accepted_steps": 0,
                "pose_converged": True,
            }
            self.publish("joint", name)
            LOG.info(
                "%s pose converged at %.6f degrees; unlocking active stress",
                name,
                float(jaw0[0]) * 10,
            )
            return
        step_cap = min(
            self.cfg.pose_max_step_deg,
            state.get("pose_step_cap_deg", self.cfg.pose_max_step_deg),
        )
        direction, inverse_curvature = pose_only_direction(state, step_cap)
        slope = float(gradient[0]) * direction
        assert math.isfinite(slope), (direction, slope)
        if slope >= 0:
            self.finish_without_update(
                name,
                path,
                state,
                "descent_unresolved",
                "pose-only scalar direction has no numerically resolved descent",
                fit_stage="pose_only",
                direction=direction,
                slope=slope,
            )
            return
        q = state["activation"].detach().clone().requires_grad_()
        jaw = jaw0.detach().clone().requires_grad_()
        old_adjoint = self.runtime.warm_adjoints.get(name)
        old_adjoint = None if old_adjoint is None else old_adjoint.detach().clone()
        accepted = None
        trials = []
        for trial in range(self.cfg.max_backtracks):
            fraction = 0.5**trial
            with torch.no_grad():
                jaw.copy_(jaw0 + fraction * direction)
            if old_adjoint is None:
                self.runtime.warm_adjoints.pop(name, None)
            else:
                self.runtime.warm_adjoints[name] = old_adjoint.clone()
            row = {
                "fit_stage": "pose_only",
                "trial": trial,
                "fraction": fraction,
                "accepted": False,
                "direction_method": "bounded_scalar_secant",
                "inverse_curvature": inverse_curvature,
                "proposed_angle_deg": float(jaw[0]) * 10,
            }
            try:
                candidate = self.evaluate_from_state(index, q, jaw, state)
                candidate["metrics"]["fit_stage"] = "pose_only"
                candidate["metrics"]["initial_fit_rms_mm"] = metrics.get(
                    "initial_fit_rms_mm", metrics["fit_rms_mm"]
                )
                new_metrics = candidate["metrics"]
                row.update(
                    {
                        "objective": new_metrics["objective"],
                        "armijo_rhs": metrics["objective"]
                        + self.cfg.armijo * fraction * slope,
                        "corrected_objective_estimate": new_metrics[
                            "residual_corrected_objective_estimate"
                        ],
                        "corrected_armijo_rhs_estimate": metrics[
                            "residual_corrected_objective_estimate"
                        ]
                        + self.cfg.armijo * fraction * slope,
                        "primal_error_estimate_sum": abs(
                            metrics["primal_objective_correction_estimate"]
                        )
                        + abs(new_metrics["primal_objective_correction_estimate"]),
                    }
                )
                row["resolved_descent_margin_estimate"] = (
                    row["armijo_rhs"]
                    - row["objective"]
                    - row["primal_error_estimate_sum"]
                )
                if (
                    row["resolved_descent_margin_estimate"] >= 0
                    and row["corrected_objective_estimate"]
                    <= row["corrected_armijo_rhs_estimate"]
                ):
                    row["accepted"] = True
                    accepted = candidate
                else:
                    row["reason"] = "Armijo decrease or estimated primal noise"
            except ForwardConvergenceError as error:
                row.update(reason=str(error), forward_failure=error.receipt)
            trials.append(row)
            append(
                path.parent / "trials.jsonl",
                {"from_step": state["accepted_steps"], **row},
            )
            if accepted is not None:
                break
        if accepted is None:
            if old_adjoint is None:
                self.runtime.warm_adjoints.pop(name, None)
            else:
                self.runtime.warm_adjoints[name] = old_adjoint
            self.finish_without_update(
                name,
                path,
                state,
                "line_search_failed",
                "pose-only proposals rejected; no transition to joint fitting",
                fit_stage="pose_only",
                trials=trials,
            )
            return
        new_state = {
            **state,
            **accepted,
            "activation": q.detach(),
            "jaw_normalized": jaw.detach(),
            "accepted_steps": state["accepted_steps"] + 1,
            "pose_accepted_steps": state["pose_accepted_steps"] + 1,
            "joint_accepted_steps": 0,
            "inverse_converged": False,
            "pose_previous": {"angle": float(jaw0[0]), "gradient": float(gradient[0])},
            "pose_step_cap_deg": min(
                self.cfg.pose_max_step_deg, 2 * abs(float(jaw[0] - jaw0[0])) * 10
            ),
            "direction_method": "bounded_scalar_secant",
            "accepted_fraction": trials[-1]["fraction"],
        }
        new_state.pop("optimizer", None)
        atomic_torch(path, cpu_tree(new_state))
        row = {
            **accepted["metrics"],
            "status": "fitting",
            "accepted_steps": new_state["accepted_steps"],
            "pose_accepted_steps": new_state["pose_accepted_steps"],
            "joint_accepted_steps": 0,
            "accepted_fraction": trials[-1]["fraction"],
            "direction_method": "bounded_scalar_secant",
        }
        append(path.parent / "trace.jsonl", row)
        self.status["expressions"][name] = row
        self.publish("pose_only", name)
        cherries.log_metrics(
            {
                f"expressions/{name}/pose/{key}": row[key]
                for key in (
                    "objective",
                    "fit_rms_mm",
                    "mandible_angle_deg",
                    "pose_accepted_steps",
                )
            },
            step=new_state["accepted_steps"],
        )
        LOG.info(
            "%s pose-only accepted %d: %.6f degrees, RMS %.6f mm, fraction %.5f",
            name,
            new_state["pose_accepted_steps"],
            row["mandible_angle_deg"],
            row["fit_rms_mm"],
            row["accepted_fraction"],
        )

    def fit_step(self, index: int):  # noqa: C901, PLR0912, PLR0915
        name = self.names[index]
        outer_step_policy = getattr(self.cfg, "outer_step_policy", "armijo")
        assert outer_step_policy in {"armijo", "full_adam"}
        self.publish("fitting", name)
        directory = self.cfg.output_dir / "expressions" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "latest.pt"
        if path.exists():
            state = torch.load(
                path, map_location=self.neutral.device, weights_only=False
            )
            if not self.cfg.pose_collision:
                self.set_collision_enabled(state["fit_stage"] == "joint")
        else:
            if not self.cfg.pose_collision:
                self.set_collision_enabled(False)
            q0 = torch.zeros((len(self.mass), 6), requires_grad=True)
            j0 = torch.zeros(1, requires_grad=True)
            state = {
                "schema": "joint-fixed-material-expression-checkpoint-v1",
                "expression": name,
                "expression_index": index,
                "fit_stage": "pose_only" if self.cfg.pose_first else "joint",
                "pose_accepted_steps": 0,
                "joint_accepted_steps": 0,
                "activation": q0.detach(),
                "jaw_normalized": j0.detach(),
                "accepted_steps": 0,
                "history": [],
                "qualifying_consecutive": 0,
                "outer_step_policy": outer_step_policy,
            }
            if getattr(self.cfg, "coupled_predictor", False):
                initial = self.evaluate(
                    index,
                    q0,
                    j0,
                    self.neutral_seed,
                    torch.zeros(1),
                    seed_q=torch.zeros_like(q0),
                )
            else:
                initial = self.evaluate(
                    index, q0, j0, self.neutral_seed, torch.zeros(1)
                )
            state.update(initial)
            state["metrics"]["fit_stage"] = state["fit_stage"]
            state["metrics"]["initial_fit_rms_mm"] = state["metrics"]["fit_rms_mm"]
            atomic_torch(directory / "initial.pt", cpu_tree(state))
            atomic_torch(path, cpu_tree(state))
            append(directory / "trace.jsonl", {"accepted_steps": 0, **state["metrics"]})
        assert state["jaw_normalized"].shape == (1,)
        assert state["gradient_jaw"].shape == (1,)
        assert state.get("outer_step_policy", "armijo") == outer_step_policy
        self.status["expressions"][name] = {
            "status": "initialized" if state["accepted_steps"] == 0 else "fitting",
            "accepted_steps": state["accepted_steps"],
            "pose_accepted_steps": state.get("pose_accepted_steps", 0),
            "joint_accepted_steps": state.get(
                "joint_accepted_steps", state["accepted_steps"]
            ),
            "pose_converged": state.get("pose_converged", False),
            "outer_step_policy": outer_step_policy,
            **state["metrics"],
        }
        self.publish(state.get("fit_stage", "joint"), name)
        if state.get("fit_stage") == "pose_only":
            self.pose_step(index, path, state)
            return
        if state.get("inverse_converged", False):
            self.status["expressions"][name] = {
                "status": "converged",
                "accepted_steps": state["accepted_steps"],
                **state["metrics"],
            }
            return
        q = torch.nn.Parameter(state["activation"].clone())
        jaw = torch.nn.Parameter(state["jaw_normalized"].clone())
        optimizer = torch.optim.Adam((q, jaw), lr=self.cfg.learning_rate)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        q.grad = state["gradient_q"].clone()
        jaw.grad = state["gradient_jaw"].clone()
        assert bool(torch.isfinite(q.grad).all() and torch.isfinite(jaw.grad).all())
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        project_activation_(q, CAP)
        with torch.no_grad():
            jaw.clamp_(HINGE_MIN, HINGE_MAX)
        dq, dj = (
            q.detach() - state["activation"],
            jaw.detach() - state["jaw_normalized"],
        )
        restarted = False
        direction_method = (
            "full_projected_adam"
            if outer_step_policy == "full_adam"
            else "projected_adam"
        )
        direction_scale = 1.0
        old_adjoint = self.runtime.warm_adjoints.get(name)
        old_adjoint = None if old_adjoint is None else old_adjoint.detach().clone()
        trials = []
        accepted = None
        if outer_step_policy == "full_adam":
            # Full-step policy: exactly one ordinary Adam proposal, followed
            # by the existing PSD and jaw-box projections.  Do not use a slope,
            # roughness budget, prescreen, primal estimate, or backtracking to
            # replace this requested proposal.  Forward failures propagate.
            candidate = self.evaluate_from_state(index, q, jaw, state)
            accepted = candidate
            row = {
                "trial": 0,
                "fraction": 1.0,
                "accepted": True,
                "direction_method": direction_method,
                "direction_scale": direction_scale,
                "outer_step_policy": outer_step_policy,
                "objective": candidate["metrics"]["objective"],
                "primal_correction_change_estimate": (
                    candidate["metrics"]["primal_objective_correction_estimate"]
                    - state["metrics"]["primal_objective_correction_estimate"]
                ),
            }
            trials.append(row)
            append(
                directory / "trials.jsonl",
                {"from_step": state["accepted_steps"], **row},
            )
        else:
            slope = float(
                (state["gradient_q"] * dq).sum() + (state["gradient_jaw"] * dj).sum()
            )
            assert math.isfinite(slope), slope
            if slope >= 0:
                # A fresh coordinatewise Adam step can still be uphill after PSD
                # projection. Use the projection-compatible stationarity metric.
                restarted = True
                direction_method = "volume_metric_projected_gradient"
                dq, dj, slope, direction_scale = projected_gradient_direction(
                    state["activation"],
                    state["jaw_normalized"],
                    state["gradient_q"],
                    state["gradient_jaw"],
                    self.mass,
                    self.cfg.learning_rate,
                )
                if not bool(torch.count_nonzero(dq) or torch.count_nonzero(dj)):
                    check = stationarity(
                        state["activation"],
                        state["jaw_normalized"],
                        state["gradient_q"],
                        state["gradient_jaw"],
                        self.mass,
                    )
                    assert check["stationary"], check
                    resolved = abs(
                        state["metrics"]["primal_objective_correction_estimate"]
                    ) <= (1e-6 * max(1.0, abs(state["metrics"]["objective"])))
                    self.finish_without_update(
                        name,
                        path,
                        state,
                        "converged" if resolved else "stationary_primal_unresolved",
                        "exact projected stationarity with resolved primal estimate"
                        if resolved
                        else "exact projected stationarity; primal estimate unresolved",
                        stationarity=check,
                        primal_error_resolved=resolved,
                        direction_method=direction_method,
                    )
                    return
                if slope >= 0:
                    self.finish_without_update(
                        name,
                        path,
                        state,
                        "descent_unresolved",
                        "nonzero metric-projected direction has no numerically resolved descent",
                        slope=slope,
                        direction_method=direction_method,
                    )
                    return
                # Persist this reset only if the corrected proposal is accepted.
                optimizer = torch.optim.Adam((q, jaw), lr=self.cfg.learning_rate)
            for trial in range(self.cfg.max_backtracks):
                fraction = 0.5**trial
                with torch.no_grad():
                    q.copy_(state["activation"] + fraction * dq)
                    jaw.copy_(state["jaw_normalized"] + fraction * dj)
                if old_adjoint is None:
                    self.runtime.warm_adjoints.pop(name, None)
                else:
                    self.runtime.warm_adjoints[name] = old_adjoint.clone()
                row = {
                    "trial": trial,
                    "fraction": fraction,
                    "accepted": False,
                    "direction_method": direction_method,
                    "direction_scale": direction_scale,
                    "outer_step_policy": outer_step_policy,
                }
                if neighbor_rms(q, self.graph) > self.cfg.neighbor_rms_budget:
                    row["reason"] = "activation neighbor budget"
                else:
                    try:
                        candidate = self.evaluate_from_state(
                            index,
                            q,
                            jaw,
                            state,
                            objective_ceiling=(
                                state["metrics"]["objective"]
                                + self.cfg.armijo * fraction * slope
                                if self.cfg.trial_prescreen
                                else None
                            ),
                        )
                        row["objective"] = candidate["metrics"]["objective"]
                        row["primal_correction_change_estimate"] = (
                            candidate["metrics"]["primal_objective_correction_estimate"]
                            - state["metrics"]["primal_objective_correction_estimate"]
                        )
                        row["armijo_rhs"] = (
                            state["metrics"]["objective"]
                            + self.cfg.armijo * fraction * slope
                        )
                        row["corrected_objective_estimate"] = candidate["metrics"][
                            "residual_corrected_objective_estimate"
                        ]
                        row["corrected_armijo_rhs_estimate"] = (
                            state["metrics"]["residual_corrected_objective_estimate"]
                            + self.cfg.armijo * fraction * slope
                        )
                        # p dot r is a first-order estimate, not a certified bound.
                        # Require actual decrease to exceed both estimated errors.
                        row["primal_error_estimate_sum"] = abs(
                            state["metrics"]["primal_objective_correction_estimate"]
                        ) + abs(
                            candidate["metrics"]["primal_objective_correction_estimate"]
                        )
                        row["resolved_descent_margin_estimate"] = (
                            row["armijo_rhs"]
                            - row["objective"]
                            - row["primal_error_estimate_sum"]
                        )
                        if (
                            row["resolved_descent_margin_estimate"] >= 0
                            and row["corrected_objective_estimate"]
                            <= row["corrected_armijo_rhs_estimate"]
                        ):
                            row["accepted"] = True
                            accepted = candidate
                        else:
                            row["reason"] = "Armijo decrease or estimated primal noise"
                    except TrialObjectiveRejectedError as error:
                        row.update(
                            {
                                "reason": "objective prescreen",
                                "objective_prescreen": error.receipt,
                            }
                        )
                    except ForwardConvergenceError as error:
                        row.update(
                            {"reason": str(error), "forward_failure": error.receipt}
                        )
                trials.append(row)
                append(
                    directory / "trials.jsonl",
                    {"from_step": state["accepted_steps"], **row},
                )
                if accepted is not None:
                    break
        if accepted is None:
            optimizer.load_state_dict(old_optimizer)
            if old_adjoint is None:
                self.runtime.warm_adjoints.pop(name, None)
            else:
                self.runtime.warm_adjoints[name] = old_adjoint
            self.status["expressions"][name] = {
                "status": "line_search_failed",
                "accepted_steps": state["accepted_steps"],
                "reason": "All proposals rejected; accepted checkpoint retained",
                "trials": trials,
                "outer_step_policy": outer_step_policy,
                **state["metrics"],
            }
            return
        history = [*state["history"], accepted["metrics"]["objective"]][-5:]
        stable = len(history) == 5 and (max(history) - min(history)) <= 1e-5 * max(
            map(abs, history)
        )
        primal_resolved = abs(
            accepted["metrics"]["primal_objective_correction_estimate"]
        ) <= 1e-6 * max(1.0, abs(accepted["metrics"]["objective"]))
        accepted["metrics"]["primal_error_resolved_for_convergence"] = primal_resolved
        qualified = (
            stable
            and primal_resolved
            and accepted["metrics"]["stationarity"]["stationary"]
        )
        consecutive = state["qualifying_consecutive"] + 1 if qualified else 0
        accepted["metrics"]["fit_stage"] = "joint"
        accepted["metrics"]["initial_fit_rms_mm"] = state["metrics"].get(
            "initial_fit_rms_mm", state["metrics"]["fit_rms_mm"]
        )
        new_state = {
            "schema": "joint-fixed-material-expression-checkpoint-v1",
            "expression": name,
            "expression_index": index,
            "fit_stage": "joint",
            "pose_accepted_steps": state.get("pose_accepted_steps", 0),
            "joint_accepted_steps": state.get(
                "joint_accepted_steps", state["accepted_steps"]
            )
            + 1,
            "pose_converged": state.get("pose_converged", False),
            "pose_certificate": state.get("pose_certificate"),
            "activation": q.detach(),
            "jaw_normalized": jaw.detach(),
            "accepted_steps": state["accepted_steps"] + 1,
            "history": history,
            "qualifying_consecutive": consecutive,
            "inverse_converged": consecutive >= 5,
            "optimizer": optimizer.state_dict(),
            "outer_step_policy": outer_step_policy,
            "momentum_restarted": restarted,
            "direction_method": direction_method,
            "direction_scale": direction_scale,
            "accepted_fraction": trials[-1]["fraction"],
            **accepted,
        }
        atomic_torch(path, cpu_tree(new_state))
        if new_state["accepted_steps"] % 25 == 0 or new_state["inverse_converged"]:
            atomic_torch(
                directory / f"step-{new_state['accepted_steps']:05d}.pt",
                cpu_tree(new_state),
            )
        row = {
            "accepted_steps": new_state["accepted_steps"],
            "status": "converged" if new_state["inverse_converged"] else "fitting",
            "accepted_fraction": trials[-1]["fraction"],
            "momentum_restarted": restarted,
            "direction_method": direction_method,
            "direction_scale": direction_scale,
            "outer_step_policy": outer_step_policy,
            **accepted["metrics"],
            "fit_stage": "joint",
            "pose_accepted_steps": new_state["pose_accepted_steps"],
            "joint_accepted_steps": new_state["joint_accepted_steps"],
            "pose_converged": new_state["pose_converged"],
        }
        append(directory / "trace.jsonl", row)
        self.status["expressions"][name] = row
        self.publish("joint", name)
        cherries.log_metrics(
            {
                f"expressions/{name}/{key}": row[key]
                for key in (
                    "accepted_steps",
                    "objective",
                    "data",
                    "fit_rms_mm",
                    "weighted_smoothness",
                    "neighbor_rms",
                    "activation_tensor_rms_kpa",
                    "accepted_fraction",
                )
            },
            step=new_state["accepted_steps"],
        )
        LOG.info(
            "%s accepted %d: RMS %.6f mm, data %.8f, fraction %.5f",
            name,
            new_state["accepted_steps"],
            row["fit_rms_mm"],
            row["data"],
            row["accepted_fraction"],
        )

    def run(self):
        self.refine_neutral()
        self.calibrate()
        self.status["smoothness_weight"] = self.smooth_weight
        self.status["schedule"] = "sequential"
        terminal = {
            "converged",
            "line_search_failed",
            "stationary_primal_unresolved",
            "descent_unresolved",
        }
        for index, name in enumerate(self.names):
            self.status["expression_index"] = index
            while self.status["expressions"][name]["status"] not in terminal:
                if (
                    self.status["expressions"][name]["accepted_steps"]
                    >= self.cfg.maximum_iterations_per_expression
                ):
                    self.status["running"] = False
                    self.status["inverse_converged"] = False
                    self.status["iteration_budget"] = {
                        "expression": name,
                        "maximum_iterations_per_expression": self.cfg.maximum_iterations_per_expression,
                        "accepted_steps": self.status["expressions"][name][
                            "accepted_steps"
                        ],
                        "scope": "total accepted steps including continuation; no advance to next expression",
                    }
                    self.publish("expression_iteration_budget_reached", name)
                    return
                if time.perf_counter() - self.started > self.cfg.wall_cap_seconds:
                    self.status["running"] = False
                    self.publish("wall_budget_reached", name)
                    return
                self.fit_step(index)
                if (
                    self.status["expressions"][name]["status"]
                    == "requires_contact_recovery"
                ):
                    self.status["running"] = False
                    self.publish("requires_contact_recovery", name)
                    return
            # Retain one expression's warm adjoint until that expression ends.
            self.runtime.warm_adjoints.pop(name, None)
        statuses = [item["status"] for item in self.status["expressions"].values()]
        self.status["running"] = False
        self.status["inverse_converged"] = all(item == "converged" for item in statuses)
        self.publish(
            "converged" if self.status["inverse_converged"] else "requires_attention"
        )


def main(cfg: Config):  # noqa: PLR0915
    assert cfg.maximum_iterations_per_expression > 0 and cfg.wall_cap_seconds > 0
    assert cfg.calibration_strength >= 3
    assert 0 < cfg.adjoint_rtol <= 1e-7
    assert 0 < cfg.pose_max_step_deg <= 40
    assert 0 < cfg.pose_stationarity_tolerance <= 1e-3
    assert cfg.outer_step_policy in {"armijo", "full_adam"}
    assert not (cfg.outer_step_policy == "full_adam" and cfg.pose_first), (
        "full_adam applies only to joint steps; pose_first has its own line search"
    )
    assert not (cfg.outer_step_policy == "full_adam" and cfg.trial_prescreen), (
        "full_adam has no trial prescreen"
    )
    assert cfg.pose_collision or cfg.pose_first, (
        "collision-off is restricted to pose-only initialization"
    )
    assert not cfg.coupled_predictor or cfg.pose_collision, (
        "coupled predictor requires collision-enabled seeds"
    )
    assert not (cfg.pose_first and cfg.continue_from is not None), (
        "pose-first runs start from zero activation, not prior joint activation checkpoints"
    )
    assert cfg.ipc_threads is None or cfg.ipc_threads > 0
    validation = json.loads(cfg.validation.read_text())
    assert validation["force_tolerance"] == cfg.forward_atol
    assert len(validation["solves"]) == 9
    for receipt in validation["solves"]:
        forward = receipt["forward"]
        assert forward["success"] and forward["grad_norm"] <= cfg.forward_atol
        assert forward["contact"]["contact_numerically_valid"]
        assert all(forward["terminal_gates"].values())
    diagnostic = json.loads(cfg.derivative_diagnostic.read_text())
    assert diagnostic["completed"]
    assert diagnostic["source_summary_sha256"] == sha256(cfg.validation)
    for key in (
        "corrected_gradient_check_passed",
        "corrected_plateau_check_passed",
        "force_parameter_check_passed",
    ):
        assert diagnostic[key], (key, diagnostic)
    hessian = json.loads(cfg.hessian_validation.read_text())
    assert hessian["success"]
    for result in (validation, diagnostic, hessian):
        for source, digest in result["implementation_sha256"].items():
            assert sha256(Path(source)) == digest, source
    input_manifest = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    assert validation["seed"]["checkpoint_sha256"] in {
        record["sha256"] for record in input_manifest["sources"].values()
    }
    pose_bundle = EyeExpressionInputs.load(cfg.inputs_dir)
    pose_inputs = pose_bundle.arrays
    if cfg.ipc_threads is not None:
        ipctk.set_num_threads(cfg.ipc_threads)
    ipc_threads_actual = int(ipctk.get_num_threads())
    protocol = {
        "schema": "joint-fixed-material-expression-protocol-v1",
        "inputs_manifest_sha256": sha256(cfg.inputs_dir / "manifest.json"),
        "validation": {"path": str(cfg.validation), "sha256": sha256(cfg.validation)},
        "derivative_diagnostic": {
            "path": str(cfg.derivative_diagnostic),
            "sha256": sha256(cfg.derivative_diagnostic),
        },
        "hessian_validation": {
            "path": str(cfg.hessian_validation),
            "sha256": sha256(cfg.hessian_validation),
        },
        "raw_finite_tolerance_gradient_check_passed": validation["success"],
        "derivative_validation_scope": "implicit equilibrium derivative via exact Hessian, force-parameter finite differences, and two-scale residual-corrected objective finite differences; not the derivative of finite PNCG iterations",
        "primal_error_policy": (
            "raw and residual-corrected Armijo; raw decrease must exceed sum of absolute first-order primal objective error estimates; estimates are not certified bounds"
            if cfg.outer_step_policy == "armijo"
            else "p dot r is recorded as a first-order primal-error estimate and does not gate the one full projected Adam update"
        ),
        "activation_reference_mpa": REFERENCE_MPA,
        "activation_cap_dimensionless": CAP,
        "skin_multiplier": 1.0,
        "materials_optimized": False,
        "ipctk_threads": {
            "requested": cfg.ipc_threads,
            "actual": ipc_threads_actual,
            "scope": "process-global IPCTK setting before rigid-eye collision construction",
        },
        "objective": f"mean expression normalized area-weighted MSE + strong 5mm graph smoothness + {cfg.magnitude_weight:g} magnitude + ({cfg.jaw_weight:g}/6) squared hinge angle in 10-degree units",
        "normalization": "original expression displacement RMS squared; unchanged expression deltas",
        "mandible_pose": {
            "dofs_per_expression": 1,
            "representation": "rotation vector = normalized_angle * angle_scale_rad * fixed_axis_world; translation identically zero",
            "axis_world": pose_inputs["mandible_frame_world"][:, 0].tolist(),
            "pivot_m": pose_inputs["mandible_pivot_m"].tolist(),
            "axis_definition": "registered mandible landmark 1 toward 9; pivot at their midpoint; positive angle lowers anterior mandible",
            "angle_scale_rad": HINGE_SCALE_RAD,
            "normalized_bounds": [HINGE_MIN, HINGE_MAX],
            "angle_bounds_deg": [0.0, 40.0],
            "zero_definition": "adopted neutral; not independently verified dental occlusion",
            "bound_basis": "exploratory opening range; published maximal-opening mean rotation 36.3 and 39.1 degrees included joint translation; not a subject-specific anatomical bound",
            "bound_reference": "https://pubmed.ncbi.nlm.nih.gov/8765386/",
            "kinematic_limit": "fixed provisional landmark axis; excludes physiological joint translation and other rotations",
            "prior": "jaw_weight/6 * normalized_angle^2; unchanged restriction of former six-coordinate prior",
        },
        "jaw_box_is_anatomical_or_collision_certificate": False,
        "rigid_rigid_collision_enabled": False,
        "soft_soft_collision_enabled": False,
        "solver": "accepted-force PNCG; strict implicit adjoint; collision-off only during explicitly configured pose initialization, otherwise trial CCD and intersection gates",
        "implementation_sha256": {
            name: sha256(GROUP / "src" / name)
            for name in (
                "93-fit-expressions.py",
                "joint_expression_equilibrium.py",
                "joint_expression_inputs.py",
                "joint_fields.py",
                "joint_equilibrium.py",
                "joint_collision_off_pose.py",
                "joint_coupled_seed.py",
                "joint_coupled_predictor.py",
                "joint_coupled_continuation.py",
            )
        },
        "optimizer": (
            "per-expression projected Adam with volume-metric projected-gradient descent safeguard; transactional Armijo; sequential until convergence or explicit failure"
            if cfg.outer_step_policy == "armijo"
            else "per-expression projected Adam; one full projected proposal per joint step with visible forward failure"
        ),
        "descent_policy": {
            "revision": 3,
            "outer_step_policy": cfg.outer_step_policy,
            "non_descent_adam": (
                "volume-metric projected gradient; common scaling caps maximum coordinate change at learning_rate; Adam moments reset only on acceptance"
                if cfg.outer_step_policy == "armijo"
                else "not used: retain ordinary Adam moments and take the full PSD/jaw-projected proposal"
            ),
            "zero_direction": (
                "exact projected stationarity plus resolved primal estimate terminates as converged without fictitious accepted steps; otherwise explicit unresolved failure"
                if cfg.outer_step_policy == "armijo"
                else "not used to reject a full Adam proposal"
            ),
            "ordinary_convergence": "unchanged projected-gradient, primal-accuracy and five-consecutive stable-window checks after accepted updates",
            "nonfinite": "fail visibly; full_adam does not retry a smaller proposal",
        },
        "schedule": {
            "policy": "sequential",
            "expression_order": list(pose_bundle.expression_names),
            "advance_only_after": [
                "converged",
                "line_search_failed",
                "stationary_primal_unresolved",
                "descent_unresolved",
            ],
            "iteration_budget_policy": "stop entire run without convergence or advancing",
        },
        "staging": {
            "pose_first": cfg.pose_first,
            "pose_collision": cfg.pose_collision,
            "coupled_predictor": cfg.coupled_predictor,
            "coupled_predictor_policy": "opt-in detached tangent/CCD continuation seed per exact outer candidate; target forward remains one strict PNCG correction with original autograd stress and pose",
            "joint_collision": True,
            "order_per_expression": ["pose_only", "joint"]
            if cfg.pose_first
            else ["joint"],
            "pose_only": "identically zero muscle activation; same objective and physical model; bounded scalar descent with positive secant inverse curvature and separate angle-step cap",
            "pose_step_cap_deg": cfg.pose_max_step_deg,
            "pose_transition": "projected jaw mapping <= pose_stationarity_tolerance and resolved primal objective error estimate; first-order stationarity, not global minimum; failed line search never unlocks activation",
            "joint_initialization": "zero activation and fitted pose, fresh Adam moments; collision-on pose uses its cached equilibrium/gradient; collision-off pose requires a new contact-on equilibrium/gradient before joint fitting; joint convergence remains separate",
            "collision_off_handoff": "audit complete source skull/mandible/eyes at fitted pose, then require a fresh contact-on equilibrium and adjoint; intersections, inadequate gap or contact solve failure stop the run as requires_contact_recovery without starting joint fitting",
            "budget": "accepted-step and wall budgets include both stages; no budget-triggered transition",
        },
        "config": cfg.model_dump(
            mode="json",
            exclude={"resume", "wall_cap_seconds", "maximum_iterations_per_expression"},
        ),
    }
    if cfg.resume:
        saved = json.loads((cfg.output_dir / "protocol.json").read_text())
        if "continuation" in saved:
            protocol["continuation"] = saved["continuation"]
        assert saved == protocol
    else:
        cfg.output_dir.mkdir(parents=True, exist_ok=False)
        archive_sources(cfg.output_dir)
        if cfg.continue_from is not None:
            protocol["continuation"] = import_continuation(cfg, protocol)
        write_json(cfg.output_dir / "protocol.json", protocol)
    configure_cuda()
    logging.getLogger("joint_expression_equilibrium").setLevel(logging.INFO)
    LOG.setLevel(logging.INFO)
    fitter = Fitter(cfg)
    try:
        fitter.run()
    except BaseException as error:
        fitter.status.update(
            {
                "running": False,
                "failure": str(error),
                "error_type": type(error).__name__,
            }
        )
        fitter.publish("failed")
        raise
    finally:
        write_json(cfg.output_dir / "summary.json", fitter.status)
        cherries.log_output(cfg.output_dir / "summary.json")
        cherries.log_output(cfg.output_dir / "protocol.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
