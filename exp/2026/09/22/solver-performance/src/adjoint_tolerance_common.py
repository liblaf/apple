# ruff: noqa: E402
"""Shared fixed-state construction for the Smile adjoint-tolerance replay.

The helper deliberately separates model initialization from replay.  It can
reconstruct the exact material/boundary inputs for either a fixed-state
implicit-adjoint VJP or a separately authorized forward replay, but never
calls a primal solver itself.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.autograd.function import once_differentiable

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(SOURCE_GROUP / "src")]

from joint_common import sha256
from joint_expression_inputs import EyeExpressionInputs

from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.inverse._diff_forward import _AdjointProblem

SMILE = "Smile"
SMILE_INDEX = 12


def load_runner() -> Any:
    """Load the production fitter module without parsing this script's CLI."""
    path = SOURCE_GROUP / "src/93-fit-expressions.py"
    spec = importlib.util.spec_from_file_location("adjoint_tolerance_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = np.ascontiguousarray(value.numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "tensor_sha256": tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


@dataclass
class FixedStateContext:
    """Fresh contact-on model and one checkpoint's saved physical state.

    ``materials``, ``fixed_values`` and ``full_displacement`` are constructed
    from exactly the saved activation/jaw state.  Building this context does no
    equilibrium solve; callers decide whether to run a fixed-state VJP or a
    future explicitly authorized forward solve.
    """

    runner: Any
    fitter: Any
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    index: int
    q: torch.Tensor
    jaw: torch.Tensor
    materials: dict[str, dict[str, torch.Tensor]]
    fixed_values: torch.Tensor
    full_displacement: torch.Tensor
    fem_node_count: int
    initialization_seconds: float


def build_fixed_state_context(
    *,
    checkpoint_path: Path,
    inputs_dir: Path,
    output_dir: Path,
    forward_atol: float = 1e-8,
    adjoint_rtol: float = 1e-7,
    ipc_threads: int = 8,
) -> FixedStateContext:
    """Rebuild full bone-and-eye physics and checkpoint material inputs.

    This is intentionally the reusable public interface for a future forward
    replay: callers may use ``context.fitter`` with the returned q/jaw and an
    explicit seed.  This helper itself never invokes ``solve``, ``evaluate``,
    or an optimizer.
    """
    started = time.perf_counter()
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    assert checkpoint["expression"] == SMILE
    assert int(checkpoint["expression_index"]) == SMILE_INDEX
    assert checkpoint["activation"].shape[-1] == 6
    assert checkpoint["jaw_normalized"].shape == (1,)
    runner = load_runner()
    cfg = runner.Config(
        _cli_parse_args=False,
        output_dir=output_dir,
        inputs_dir=inputs_dir,
        calibration_source=None,
        forward_atol=forward_atol,
        adjoint_rtol=adjoint_rtol,
        ipc_threads=ipc_threads,
        pose_first=False,
        pose_collision=True,
    )
    fitter = runner.Fitter(cfg)
    # Fitter creates EyeExpressionInputs -> full skull/mandible/eyes.
    assert isinstance(fitter.inputs, EyeExpressionInputs)
    assert fitter.names[SMILE_INDEX] == SMILE
    q = checkpoint["activation"].detach().clone().to(device="cuda").requires_grad_()
    jaw = (
        checkpoint["jaw_normalized"].detach().clone().to(device="cuda").requires_grad_()
    )
    pose = runner.hinge_pose(jaw, fitter.hinge_axis)
    materials = fitter.physics.expression_materials(
        skin_multiplier=torch.ones((), device="cuda", dtype=q.dtype),
        active_stress=runner.activation_stresses_mpa(q, runner.REFERENCE_MPA),
    )
    fixed_values = fitter.physics.boundary(pose)
    saved_fem = checkpoint["displacement_m"].detach().clone().to(device="cuda")
    full_displacement = fitter.physics.full_skull.extend_seed(saved_fem, pose)
    fem_node_count = fitter.physics.full_skull.geometry.fem_node_count
    assert saved_fem.shape == (fem_node_count, 3)
    assert full_displacement.shape[0] > fem_node_count
    return FixedStateContext(
        runner=runner,
        fitter=fitter,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        index=SMILE_INDEX,
        q=q,
        jaw=jaw,
        materials=materials,
        fixed_values=fixed_values,
        full_displacement=full_displacement.detach(),
        fem_node_count=fem_node_count,
        initialization_seconds=time.perf_counter() - started,
    )


class CountedAdjointProblem:
    """Unshifted owned adjoint operator with an exact HVP/matvec count."""

    def __init__(self, delegate: _AdjointProblem) -> None:
        self.delegate = delegate
        self.hvp_matvec_calls = 0

    @property
    def b(self) -> torch.Tensor:
        return self.delegate.b

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self.hvp_matvec_calls += 1
        return self.delegate.matvec(vector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class FixedStateImplicit(torch.autograd.Function):
    """The owned implicit backward evaluated at a supplied equilibrium state.

    Its forward is an identity over the checkpointed full displacement: no
    primal optimizer, forward problem, warm start, or convergence test runs.
    The backward below follows the owned ``joint_equilibrium._Implicit`` path,
    rebuilding the collision state and applying its unshifted Hessian.
    """

    @staticmethod
    def forward(
        runtime: Any,
        key: str,
        spec: Any,
        fixed: torch.Tensor,
        saved_full_displacement: torch.Tensor,
        *leaves: torch.Tensor,
    ) -> torch.Tensor:
        del runtime, key, spec, fixed, leaves
        return saved_full_displacement

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        runtime, key, spec, fixed, saved_full_displacement, *leaves = inputs
        ctx.runtime, ctx.key, ctx.spec = runtime, key, spec
        ctx.save_for_backward(
            saved_full_displacement.detach().clone(), fixed.detach().clone(), *leaves
        )

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        runtime = ctx.runtime
        model = runtime.forward.model
        output, fixed, *saved = ctx.saved_tensors
        original = model.get_materials()
        original_fixed = model.dof_map.fixed_values
        started = time.perf_counter()
        try:
            leaves = [
                value.detach().clone().requires_grad_(needed)
                for value, needed in zip(saved, ctx.needs_input_grad[5:], strict=True)
            ]
            materials = ctx.spec.unflatten(leaves)
            model.set_materials(materials)
            model.dof_map.fixed_values = fixed
            state = model.State(u=output.detach().clone())
            if model.collision is not None:
                state.collision = model.collision.state_at(state.u)
            problem = _AdjointProblem(
                b=-model.dof_map.to_free_grad(grad_output),
                model=model,
                model_state=state,
            )
            initial = runtime.warm_adjoints.get(ctx.key, torch.zeros_like(problem.b))
            runtime.fixed_state_initial_adjoint = initial.detach().clone()
            if torch.count_nonzero(problem.b) == 0:
                p_free = torch.zeros_like(problem.b)
                residual, relative, result = 0.0, 0.0, "zero right-hand side"
                hvp_calls = 0
            else:
                counted = CountedAdjointProblem(problem)
                solution = runtime.solver.solve(counted, initial)
                assert solution.success, f"Adjoint failed: {solution}"
                p_free = solution.params.detach()
                # Snapshot before residual verification, whose HVP is not a CG
                # matvec.  The operator delegates directly to unshifted hess_prod.
                hvp_calls = counted.hvp_matvec_calls
                residual = float(
                    torch.linalg.vector_norm(problem.matvec(p_free) - problem.b)
                )
                relative = residual / float(torch.linalg.vector_norm(problem.b))
                assert relative <= runtime.tolerances["adjoint_rtol"] * 1.05, (
                    relative,
                    solution,
                )
                result = str(solution.result)
            runtime.warm_adjoints[ctx.key] = p_free.detach().clone()
            p = model.dof_map.to_full_grad(p_free)
            model.mixed_derivative_prod(state, p)
            gradients = [leaf.grad for leaf in leaves]
            fixed_gradient = (grad_output + model.hess_prod(state, p)).flatten()[
                model.dof_map.fixed_indices
            ]
            assert torch.isfinite(fixed_gradient).all()
            for leaf, gradient in zip(leaves, gradients, strict=True):
                if leaf.requires_grad:
                    assert gradient is not None
                    assert torch.isfinite(gradient).all()
            torch.cuda.synchronize()
            runtime.last_adjoint = {
                "success": True,
                "residual": residual,
                "relative_residual": relative,
                "result": result,
                "seconds": time.perf_counter() - started,
                "key": ctx.key,
                "hvp_matvec_calls": hvp_calls,
                "initial_adjoint_sha256": tensor_sha256(initial),
                "initial_adjoint_l2": float(torch.linalg.vector_norm(initial)),
                "operator": "owned_unshifted_model.hess_prod",
            }
        finally:
            model.set_materials(original)
            model.dof_map.fixed_values = original_fixed
        # runtime, key, spec, fixed, saved state, material leaves
        return (None, None, None, fixed_gradient, None, *gradients)


def fixed_state_physical_status(context: FixedStateContext) -> dict[str, Any]:
    """Evaluate the saved physical state once, without an equilibrium solve."""
    model = context.fitter.runtime.forward.model
    original = model.get_materials()
    original_fixed = model.dof_map.fixed_values
    try:
        model.set_materials(context.materials)
        model.dof_map.fixed_values = context.fixed_values.detach().clone()
        state = model.State(u=context.full_displacement.detach().clone())
        if model.collision is not None:
            state.collision = model.collision.state_at(state.u)
        with torch.no_grad():
            force = ForwardProblem(model=model).grad(state)
            force_l2 = float(torch.linalg.vector_norm(force))
            contact = model.collision.diagnostics(state.collision, state.u)
        shape = context.fitter.physics.metrics(
            context.checkpoint["displacement_m"].detach(), target_index=context.index
        )
        return {
            "free_force_l2": force_l2,
            "forward_atol": context.fitter.runtime.tolerances["atol"],
            "force_at_or_below_forward_atol": force_l2
            <= context.fitter.runtime.tolerances["atol"],
            "contact": contact,
            "shape": shape,
        }
    finally:
        model.set_materials(original)
        model.dof_map.fixed_values = original_fixed


def fixed_state_output(context: FixedStateContext, *, key: str) -> torch.Tensor:
    """Create a differentiable saved-state output without calling a primal solve."""
    leaves, spec = pytree.tree_flatten(context.materials)
    return FixedStateImplicit.apply(
        context.fitter.runtime,
        key,
        spec,
        context.fixed_values,
        context.full_displacement,
        *leaves,
    )


def install_adjoint_tolerance(runtime: Any, rtol: float) -> dict[str, Any]:
    """Install the actual historical CuPy solvers at one requested rtol."""
    from face_physics import SuccessPreferredFallbackSolver

    from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes

    assert rtol > 0
    prior = {
        "solver": runtime.solver,
        "adjoint_rtol": runtime.tolerances["adjoint_rtol"],
        "warm_adjoints": copy.deepcopy(runtime.warm_adjoints),
    }
    runtime.solver = SuccessPreferredFallbackSolver(
        [
            CupyCG(maxiter=10_000, rtol=rtol, atol=0.0),
            CupyMinRes(maxiter=10_000, tol=rtol),
        ]
    )
    runtime.tolerances["adjoint_rtol"] = rtol
    runtime.warm_adjoints.clear()
    return prior


def restore_adjoint_tolerance(runtime: Any, prior: dict[str, Any]) -> None:
    runtime.solver = prior["solver"]
    runtime.tolerances["adjoint_rtol"] = prior["adjoint_rtol"]
    runtime.warm_adjoints = prior["warm_adjoints"]


def load_protocol_weights(run_dir: Path) -> dict[str, float]:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    objective = protocol["objective"]
    return {
        "smoothness_weight": float(objective["smoothness_weight"]),
        "magnitude_weight": float(objective["magnitude_weight"]),
        "jaw_weight": float(objective["jaw_weight"]),
        "learning_rate": float(protocol["config"]["learning_rate"]),
    }
