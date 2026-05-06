# ruff: noqa: ANN001, E402
from __future__ import annotations

"""Micro-benchmark for branch-heavy PNCG control flow over Warp kernels.

Warp owns the vector work in every path. `jax-jit` keeps scalar decisions inside
a compiled `lax.while_loop`; `jax-eager` and `torch` use eager Python branches
with `.item()` checks, matching the straightforward porting style we would
likely start from when the solver cannot be compiled end to end.
"""

import argparse
import importlib.util
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax

jax.config.update("jax_enable_x64", val=True)

import jax.numpy as jnp
import warp as wp
import warp.jax_experimental.ffi

Backend = Literal["jax-jit", "jax-eager", "torch"]


@wp.kernel
def _quadratic_terms_kernel(
    x: wp.array(dtype=wp.float64),
    target: wp.array(dtype=wp.float64),
    stiffness: wp.array(dtype=wp.float64),
    direction: wp.array(dtype=wp.float64),
    energy: wp.array(dtype=wp.float64),
    grad: wp.array(dtype=wp.float64),
    hess_diag: wp.array(dtype=wp.float64),
    hess_quad: wp.array(dtype=wp.float64),
) -> None:
    tid = wp.tid()
    dx = x[tid] - target[tid]
    k = stiffness[tid]
    p = direction[tid]
    energy[tid] = type(k)(0.5) * k * dx * dx
    grad[tid] = k * dx
    hess_diag[tid] = k
    hess_quad[tid] = k * p * p


_quadratic_terms_jax = warp.jax_experimental.ffi.jax_kernel(
    _quadratic_terms_kernel, num_outputs=4
)


@dataclass(frozen=True)
class BenchmarkResult:
    backend: Backend
    size: int
    steps: int
    warmup: int
    repeat: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    value: float
    accepted_steps: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a branch-heavy PNCG-like loop where Warp evaluates the "
            "vector objective terms and JAX/PyTorch owns the scalar control flow."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["all", "jax", "jax-jit", "jax-eager", "torch"],
        default="all",
        help="Backend wrapper to benchmark.",
    )
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--max-backtracking-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not wp.is_cuda_available():
        print("CUDA is required for this JAX/Warp vs PyTorch/Warp benchmark.")
        raise SystemExit(2)

    wp.init()

    results: list[BenchmarkResult] = []
    if args.backend in {"all", "jax", "jax-jit"}:
        results.append(
            bench_jax(
                size=args.size,
                steps=args.steps,
                max_backtracking_steps=args.max_backtracking_steps,
                warmup=args.warmup,
                repeat=args.repeat,
                device=args.device,
            )
        )
    if args.backend in {"all", "jax-eager"}:
        results.append(
            bench_jax_eager(
                size=args.size,
                steps=args.steps,
                max_backtracking_steps=args.max_backtracking_steps,
                warmup=args.warmup,
                repeat=args.repeat,
                device=args.device,
            )
        )
    if args.backend in {"all", "torch"}:
        if importlib.util.find_spec("torch") is None:
            print("torch: skipped; install torch to benchmark the PyTorch/Warp path")
        else:
            results.append(
                bench_torch(
                    size=args.size,
                    steps=args.steps,
                    max_backtracking_steps=args.max_backtracking_steps,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    device=args.device,
                )
            )

    if results:
        print_results(results)


def bench_jax(
    *,
    size: int,
    steps: int,
    max_backtracking_steps: int,
    warmup: int,
    repeat: int,
    device: str,
) -> BenchmarkResult:
    jax_device = _jax_device(device)
    x0, target, stiffness = _jax_inputs(size, jax_device)

    @jax.jit
    def target_fun(x):
        return _run_jax_pncg_like(
            x,
            target,
            stiffness,
            steps=steps,
            max_backtracking_steps=max_backtracking_steps,
        )

    def call() -> tuple[float, int]:
        value, accepted_steps = jax.block_until_ready(target_fun(x0))
        return float(value), int(accepted_steps)

    timings, value, accepted_steps = _time_call(call, warmup=warmup, repeat=repeat)
    return _result(
        backend="jax-jit",
        size=size,
        steps=steps,
        warmup=warmup,
        repeat=repeat,
        timings=timings,
        value=value,
        accepted_steps=accepted_steps,
    )


def bench_jax_eager(
    *,
    size: int,
    steps: int,
    max_backtracking_steps: int,
    warmup: int,
    repeat: int,
    device: str,
) -> BenchmarkResult:
    jax_device = _jax_device(device)
    x0, target, stiffness = _jax_inputs(size, jax_device)

    def call() -> tuple[float, int]:
        value, accepted_steps = _run_jax_pncg_like_eager(
            x0,
            target,
            stiffness,
            steps=steps,
            max_backtracking_steps=max_backtracking_steps,
        )
        value = jax.block_until_ready(value)
        return float(value), int(accepted_steps)

    timings, value, accepted_steps = _time_call(call, warmup=warmup, repeat=repeat)
    return _result(
        backend="jax-eager",
        size=size,
        steps=steps,
        warmup=warmup,
        repeat=repeat,
        timings=timings,
        value=value,
        accepted_steps=accepted_steps,
    )


def bench_torch(
    *,
    size: int,
    steps: int,
    max_backtracking_steps: int,
    warmup: int,
    repeat: int,
    device: str,
) -> BenchmarkResult:
    torch = import_torch()
    x0, target, stiffness = _torch_inputs(torch, size=size, device=device)

    def call() -> tuple[float, int]:
        value, accepted_steps = _run_torch_pncg_like(
            torch,
            x0,
            target,
            stiffness,
            steps=steps,
            max_backtracking_steps=max_backtracking_steps,
            device=device,
        )
        torch.cuda.synchronize(device)
        return float(value), int(accepted_steps)

    timings, value, accepted_steps = _time_call(call, warmup=warmup, repeat=repeat)
    return _result(
        backend="torch",
        size=size,
        steps=steps,
        warmup=warmup,
        repeat=repeat,
        timings=timings,
        value=value,
        accepted_steps=accepted_steps,
    )


def _run_jax_pncg_like(
    x0,
    target,
    stiffness,
    *,
    steps: int,
    max_backtracking_steps: int,
):
    zeros = jnp.zeros_like(x0)
    value0, grad0, _, _ = _jax_objective_terms(x0, target, stiffness, zeros)
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        x0,
        grad0,
        zeros,
        value0,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def cond_fun(state):
        i, *_ = state
        return i < steps

    def body_fun(state):
        i, x, prev_grad, prev_direction, _prev_value, accepted_count = state
        value, grad, preconditioner, _ = _jax_objective_terms(
            x, target, stiffness, zeros
        )
        beta = _jax_dai_kou_beta(
            grad=grad,
            prev_grad=prev_grad,
            prev_direction=prev_direction,
            preconditioner=preconditioner,
        )
        beta_is_valid = jnp.isfinite(beta) & (jnp.abs(beta) <= 10.0)
        beta = jnp.where((i == 0) | (~beta_is_valid), jnp.zeros_like(beta), beta)
        steepest_direction = -preconditioner * grad
        direction = steepest_direction + beta * prev_direction
        grad_dot_direction = jnp.vdot(grad, direction)
        is_descent = jnp.isfinite(grad_dot_direction) & (grad_dot_direction < 0.0)
        beta, direction, grad_dot_direction = jax.lax.cond(
            is_descent,
            lambda _: (beta, direction, grad_dot_direction),
            lambda _: (
                jnp.zeros_like(beta),
                steepest_direction,
                jnp.vdot(grad, steepest_direction),
            ),
            operand=None,
        )
        _, _, _, hess_quad = _jax_objective_terms(x, target, stiffness, direction)
        alpha = _jax_initial_alpha(
            grad=grad,
            direction=direction,
            hess_quad=hess_quad,
            overstep=4.0,
        )
        trial_x, trial_value, accepted, alpha = _jax_backtracking_line_search(
            x=x,
            target=target,
            stiffness=stiffness,
            direction=direction,
            value=value,
            grad_dot_direction=grad_dot_direction,
            alpha=alpha,
            max_steps=max_backtracking_steps,
        )
        x = jnp.where(accepted, trial_x, x)
        value = jnp.where(accepted, trial_value, value)
        accepted_count = accepted_count + accepted.astype(accepted_count.dtype)
        return i + 1, x, grad, direction, value, accepted_count

    _i, _x, _grad, _direction, value, accepted_count = jax.lax.while_loop(
        cond_fun, body_fun, initial_state
    )
    return value, accepted_count


def _run_jax_pncg_like_eager(
    x0,
    target,
    stiffness,
    *,
    steps: int,
    max_backtracking_steps: int,
):
    x = x0
    zeros = jnp.zeros_like(x)
    value, prev_grad, _, _ = _jax_objective_terms(x, target, stiffness, zeros)
    prev_direction = zeros
    accepted_count = 0

    for i in range(steps):
        value, grad, preconditioner, _ = _jax_objective_terms(
            x, target, stiffness, zeros
        )
        beta = _jax_dai_kou_beta(
            grad=grad,
            prev_grad=prev_grad,
            prev_direction=prev_direction,
            preconditioner=preconditioner,
        )
        beta_is_valid = jnp.isfinite(beta) & (jnp.abs(beta) <= 10.0)
        if i == 0 or not bool(beta_is_valid.item()):
            beta = jnp.zeros_like(beta)
        steepest_direction = -preconditioner * grad
        direction = steepest_direction + beta * prev_direction
        grad_dot_direction = jnp.vdot(grad, direction)
        is_descent = jnp.isfinite(grad_dot_direction) & (grad_dot_direction < 0.0)
        if not bool(is_descent.item()):
            beta = jnp.zeros_like(beta)
            direction = steepest_direction
            grad_dot_direction = jnp.vdot(grad, direction)
        _, _, _, hess_quad = _jax_objective_terms(x, target, stiffness, direction)
        alpha = _jax_initial_alpha(
            grad=grad,
            direction=direction,
            hess_quad=hess_quad,
            overstep=4.0,
        )
        trial_x, trial_value, accepted, alpha = _jax_backtracking_line_search_eager(
            x=x,
            target=target,
            stiffness=stiffness,
            direction=direction,
            value=value,
            grad_dot_direction=grad_dot_direction,
            alpha=alpha,
            max_steps=max_backtracking_steps,
        )
        if bool(accepted.item()):
            x = trial_x
            value = trial_value
            accepted_count += 1
        prev_grad = grad
        prev_direction = direction
    return value, accepted_count


def _jax_objective_terms(x, target, stiffness, direction):
    energy, grad, hess_diag, hess_quad = _quadratic_terms_jax(
        x,
        target,
        stiffness,
        direction,
        output_dims={
            "energy": x.shape,
            "grad": x.shape,
            "hess_diag": x.shape,
            "hess_quad": x.shape,
        },
        launch_dims=(x.shape[0],),
    )
    hess_diag = jnp.abs(hess_diag)
    hess_diag_mean = jnp.mean(hess_diag, where=hess_diag > 0.0)
    hess_diag = jnp.where(hess_diag > 0.0, hess_diag, hess_diag_mean)
    return jnp.sum(energy), grad, jnp.reciprocal(hess_diag), jnp.sum(hess_quad)


def _jax_backtracking_line_search(
    *,
    x,
    target,
    stiffness,
    direction,
    value,
    grad_dot_direction,
    alpha,
    max_steps: int,
):
    def evaluate(alpha):
        trial_x = x + alpha * direction
        trial_value, _, _, _ = _jax_objective_terms(
            trial_x, target, stiffness, jnp.zeros_like(direction)
        )
        accepted = jnp.isfinite(trial_value) & (
            trial_value <= value + 1.0e-4 * alpha * grad_dot_direction
        )
        return trial_x, trial_value, accepted

    trial_x, trial_value, accepted = evaluate(alpha)
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        alpha,
        trial_x,
        trial_value,
        accepted,
    )

    def cond_fun(state):
        i, alpha, _trial_x, _trial_value, accepted = state
        return (~accepted) & (i < max_steps) & (alpha > 0.0)

    def body_fun(state):
        i, alpha, _trial_x, _trial_value, _accepted = state
        alpha = alpha * 0.5
        trial_x, trial_value, accepted = evaluate(alpha)
        return i + 1, alpha, trial_x, trial_value, accepted

    _i, alpha, trial_x, trial_value, accepted = jax.lax.while_loop(
        cond_fun, body_fun, initial_state
    )
    return trial_x, trial_value, accepted, alpha


def _jax_backtracking_line_search_eager(
    *,
    x,
    target,
    stiffness,
    direction,
    value,
    grad_dot_direction,
    alpha,
    max_steps: int,
):
    def evaluate(alpha):
        trial_x = x + alpha * direction
        trial_value, _, _, _ = _jax_objective_terms(
            trial_x, target, stiffness, jnp.zeros_like(direction)
        )
        accepted = jnp.isfinite(trial_value) & (
            trial_value <= value + 1.0e-4 * alpha * grad_dot_direction
        )
        return trial_x, trial_value, accepted

    trial_x, trial_value, accepted = evaluate(alpha)
    i = 0
    while (not bool(accepted.item())) and i < max_steps and float(alpha.item()) > 0.0:
        alpha = alpha * 0.5
        trial_x, trial_value, accepted = evaluate(alpha)
        i += 1
    return trial_x, trial_value, accepted, alpha


def _run_torch_pncg_like(
    torch: Any,
    x0: Any,
    target: Any,
    stiffness: Any,
    *,
    steps: int,
    max_backtracking_steps: int,
    device: str,
) -> tuple[float, int]:
    x = x0.clone()
    zeros = torch.zeros_like(x)
    scratch = _TorchScratch(torch, size=x.numel(), device=device)
    value, prev_grad, _, _ = _torch_objective_terms(
        torch, x, target, stiffness, zeros, scratch=scratch, device=device
    )
    prev_direction = torch.zeros_like(x)
    accepted_count = 0

    for i in range(steps):
        value, grad, preconditioner, _ = _torch_objective_terms(
            torch, x, target, stiffness, zeros, scratch=scratch, device=device
        )
        beta = _torch_dai_kou_beta(
            torch,
            grad=grad,
            prev_grad=prev_grad,
            prev_direction=prev_direction,
            preconditioner=preconditioner,
        )
        beta_is_valid = torch.isfinite(beta) & (torch.abs(beta) <= 10.0)
        if i == 0 or not bool(beta_is_valid.item()):
            beta = torch.zeros_like(beta)
        steepest_direction = -preconditioner * grad
        direction = steepest_direction + beta * prev_direction
        grad_dot_direction = torch.vdot(grad, direction)
        is_descent = torch.isfinite(grad_dot_direction) & (grad_dot_direction < 0.0)
        if not bool(is_descent.item()):
            beta = torch.zeros_like(beta)
            direction = steepest_direction
            grad_dot_direction = torch.vdot(grad, direction)
        _, _, _, hess_quad = _torch_objective_terms(
            torch, x, target, stiffness, direction, scratch=scratch, device=device
        )
        alpha = _torch_initial_alpha(
            torch, grad=grad, direction=direction, hess_quad=hess_quad, overstep=4.0
        )
        trial_x, trial_value, accepted, alpha = _torch_backtracking_line_search(
            torch,
            x=x,
            target=target,
            stiffness=stiffness,
            direction=direction,
            value=value,
            grad_dot_direction=grad_dot_direction,
            alpha=alpha,
            max_steps=max_backtracking_steps,
            scratch=scratch,
            device=device,
        )
        if bool(accepted.item()):
            x = trial_x
            value = trial_value
            accepted_count += 1
        prev_grad = grad
        prev_direction = direction
    return float(value.item()), accepted_count


class _TorchScratch:
    def __init__(self, torch: Any, *, size: int, device: str) -> None:
        self.energy = torch.empty(size, dtype=torch.float64, device=device)
        self.grad = torch.empty(size, dtype=torch.float64, device=device)
        self.hess_diag = torch.empty(size, dtype=torch.float64, device=device)
        self.hess_quad = torch.empty(size, dtype=torch.float64, device=device)


def _torch_objective_terms(
    torch: Any,
    x: Any,
    target: Any,
    stiffness: Any,
    direction: Any,
    *,
    scratch: _TorchScratch,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    wp_stream = wp.stream_from_torch(x.device)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _quadratic_terms_kernel,
            dim=x.numel(),
            inputs=[
                wp.from_torch(x, dtype=wp.float64),
                wp.from_torch(target, dtype=wp.float64),
                wp.from_torch(stiffness, dtype=wp.float64),
                wp.from_torch(direction, dtype=wp.float64),
            ],
            outputs=[
                wp.from_torch(scratch.energy, dtype=wp.float64),
                wp.from_torch(scratch.grad, dtype=wp.float64),
                wp.from_torch(scratch.hess_diag, dtype=wp.float64),
                wp.from_torch(scratch.hess_quad, dtype=wp.float64),
            ],
            device=device,
        )
    hess_diag = torch.abs(scratch.hess_diag)
    hess_diag_mean = torch.mean(hess_diag[hess_diag > 0.0])
    hess_diag = torch.where(hess_diag > 0.0, hess_diag, hess_diag_mean)
    return (
        torch.sum(scratch.energy),
        scratch.grad.clone(),
        torch.reciprocal(hess_diag),
        torch.sum(scratch.hess_quad),
    )


def _jax_initial_alpha(*, grad, direction, hess_quad, overstep: float):
    alpha = -jnp.vdot(grad, direction) / hess_quad
    alpha = jnp.nan_to_num(alpha, nan=0.0, neginf=0.0, posinf=1.0)
    alpha = jnp.where((alpha > 0.0) & jnp.isfinite(alpha), alpha, 1.0)
    return alpha * overstep


def _torch_initial_alpha(
    torch: Any, *, grad: Any, direction: Any, hess_quad: Any, overstep: float
):
    alpha = -torch.vdot(grad, direction) / hess_quad
    alpha = torch.nan_to_num(alpha, nan=0.0, neginf=0.0, posinf=1.0)
    alpha = torch.where(
        (alpha > 0.0) & torch.isfinite(alpha), alpha, alpha.new_tensor(1.0)
    )
    return alpha * overstep


def _torch_backtracking_line_search(
    torch: Any,
    *,
    x: Any,
    target: Any,
    stiffness: Any,
    direction: Any,
    value: Any,
    grad_dot_direction: Any,
    alpha: Any,
    max_steps: int,
    scratch: _TorchScratch,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    def evaluate(alpha):
        trial_x = x + alpha * direction
        trial_value, _, _, _ = _torch_objective_terms(
            torch,
            trial_x,
            target,
            stiffness,
            torch.zeros_like(direction),
            scratch=scratch,
            device=device,
        )
        accepted = torch.isfinite(trial_value) & (
            trial_value <= value + 1.0e-4 * alpha * grad_dot_direction
        )
        return trial_x, trial_value, accepted

    trial_x, trial_value, accepted = evaluate(alpha)
    i = 0
    while (not bool(accepted.item())) and i < max_steps and float(alpha.item()) > 0.0:
        alpha = alpha * 0.5
        trial_x, trial_value, accepted = evaluate(alpha)
        i += 1
    return trial_x, trial_value, accepted, alpha


def _jax_dai_kou_beta(*, grad, prev_grad, prev_direction, preconditioner):
    y = grad - prev_grad
    y_dot_prev_direction = jnp.vdot(y, prev_direction)
    safe = jnp.where(
        jnp.abs(y_dot_prev_direction) > 1.0e-12,
        y_dot_prev_direction,
        jnp.asarray(1.0, dtype=y_dot_prev_direction.dtype),
    )
    preconditioned_y = preconditioner * y
    beta = jnp.vdot(grad, preconditioned_y) / safe - (
        jnp.vdot(y, preconditioned_y) / safe
    ) * (jnp.vdot(prev_direction, grad) / safe)
    return jnp.where(
        jnp.abs(y_dot_prev_direction) > 1.0e-12,
        beta,
        jnp.asarray(jnp.inf, dtype=y_dot_prev_direction.dtype),
    )


def _torch_dai_kou_beta(
    torch: Any, *, grad: Any, prev_grad: Any, prev_direction: Any, preconditioner: Any
):
    y = grad - prev_grad
    y_dot_prev_direction = torch.vdot(y, prev_direction)
    safe = torch.where(
        torch.abs(y_dot_prev_direction) > 1.0e-12,
        y_dot_prev_direction,
        y_dot_prev_direction.new_tensor(1.0),
    )
    preconditioned_y = preconditioner * y
    beta = torch.vdot(grad, preconditioned_y) / safe - (
        torch.vdot(y, preconditioned_y) / safe
    ) * (torch.vdot(prev_direction, grad) / safe)
    return torch.where(
        torch.abs(y_dot_prev_direction) > 1.0e-12,
        beta,
        y_dot_prev_direction.new_tensor(float("inf")),
    )


def _jax_inputs(size: int, device) -> tuple[Any, Any, Any]:
    indices = jnp.arange(size, dtype=jnp.float64)
    x = jnp.linspace(-0.9, 0.9, size, dtype=jnp.float64)
    target = 0.25 * jnp.sin(indices * 0.001)
    stiffness = 1.0 + jnp.mod(indices, 17.0) / 17.0
    return (
        jax.device_put(x, device),
        jax.device_put(target, device),
        jax.device_put(stiffness, device),
    )


def _torch_inputs(torch: Any, *, size: int, device: str) -> tuple[Any, Any, Any]:
    indices = torch.arange(size, dtype=torch.float64, device=device)
    x = torch.linspace(-0.9, 0.9, size, dtype=torch.float64, device=device)
    target = 0.25 * torch.sin(indices * 0.001)
    stiffness = 1.0 + torch.remainder(indices, 17.0) / 17.0
    return x, target, stiffness


def _time_call(
    call: Callable[[], tuple[float, int]], *, warmup: int, repeat: int
) -> tuple[list[float], float, int]:
    value = 0.0
    accepted_steps = 0
    for _ in range(warmup):
        value, accepted_steps = call()
    timings: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        value, accepted_steps = call()
        timings.append(time.perf_counter() - start)
    return timings, value, accepted_steps


def _result(
    *,
    backend: Backend,
    size: int,
    steps: int,
    warmup: int,
    repeat: int,
    timings: list[float],
    value: float,
    accepted_steps: int,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        size=size,
        steps=steps,
        warmup=warmup,
        repeat=repeat,
        median_ms=statistics.median(timings) * 1_000.0,
        mean_ms=statistics.fmean(timings) * 1_000.0,
        min_ms=min(timings) * 1_000.0,
        max_ms=max(timings) * 1_000.0,
        value=value,
        accepted_steps=accepted_steps,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        "backend    size       steps  accepted  median_ms  mean_ms  min_ms  max_ms  value"
    )
    for result in results:
        print(
            f"{result.backend:<11}"
            f"{result.size:<11}"
            f"{result.steps:<7}"
            f"{result.accepted_steps:<10}"
            f"{result.median_ms:<11.3f}"
            f"{result.mean_ms:<9.3f}"
            f"{result.min_ms:<8.3f}"
            f"{result.max_ms:<8.3f}"
            f"{result.value:.6e}"
        )


def _jax_device(device: str):
    if not device.startswith("cuda"):
        return jax.devices("cpu")[0]
    index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
    return jax.devices("gpu")[index]


def import_torch() -> Any:
    import torch

    return torch


if __name__ == "__main__":
    main()
