# Copyright (c) 2026 liblaf
"""Fail-fast CUDA runtime and Apple kernel acceptance checks."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
from typing import Any

# JAX reads this before its first backend initialization.  The verification
# should coexist with Torch, CuPy, and Warp instead of reserving most GPU RAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _cuda_major(version: str | tuple[int, ...] | list[int]) -> int:
    if isinstance(version, str):
        return int(version.split(".", maxsplit=1)[0])
    return int(version[0])


def _require_cuda_major(
    component: str,
    version: str | tuple[int, ...] | list[int],
    allowed: set[int],
) -> None:
    major = _cuda_major(version)
    assert major in allowed, (
        f"{component} uses CUDA {major}, expected one of {sorted(allowed)}"
    )


def _torch_and_sparse(allowed: set[int]) -> dict[str, Any]:
    import torch

    assert torch.version.cuda is not None, "Torch is not a CUDA build"
    _require_cuda_major("Torch", torch.version.cuda, allowed)
    assert torch.cuda.is_available(), "Torch cannot access a CUDA device"

    architectures = torch.cuda.get_arch_list()
    if allowed == {12}:
        assert "sm_70" in architectures, (
            "Torch was not built with sm_70 support required by NVIDIA V100"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    crow = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    columns = torch.tensor([0, 1, 1], dtype=torch.int64, device=device)
    values = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64, device=device)
    matrix = torch.sparse_csr_tensor(
        crow, columns, values, size=(2, 2), dtype=torch.float64, device=device
    )
    vector = torch.tensor([[4.0], [5.0]], dtype=torch.float64, device=device)
    product = torch.sparse.mm(matrix, vector)
    torch.testing.assert_close(
        product,
        torch.tensor([[3.0], [15.0]], dtype=torch.float64, device=device),
        rtol=0.0,
        atol=0.0,
    )
    torch.cuda.synchronize(device)
    return {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "architectures": architectures,
        "device_name": properties.name,
        "device_capability": [properties.major, properties.minor],
        "device_memory_bytes": properties.total_memory,
        "sparse_csr_fp64": True,
    }


def _cupy(allowed: set[int]) -> dict[str, Any]:
    import cupy as cp

    runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    driver_version = int(cp.cuda.runtime.driverGetVersion())
    runtime_tuple = (runtime_version // 1000, (runtime_version % 1000) // 10)
    _require_cuda_major("CuPy runtime", runtime_tuple, allowed)

    matrix = cp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=cp.float64)
    rhs = cp.asarray([1.0, 2.0], dtype=cp.float64)
    solution = cp.linalg.solve(matrix, rhs)
    residual = float(cp.linalg.norm(matrix @ solution - rhs).get())
    assert residual <= 1.0e-12, f"CuPy FP64 solve residual is {residual}"
    return {
        "version": cp.__version__,
        "runtime": list(runtime_tuple),
        "runtime_encoded": runtime_version,
        "driver_encoded": driver_version,
        "fp64_solve_residual": residual,
    }


def _warp(allowed: set[int]) -> dict[str, Any]:
    import warp as wp

    wp.init()
    toolkit = tuple(int(value) for value in wp.get_cuda_toolkit_version())
    driver = tuple(int(value) for value in wp.get_cuda_driver_version())
    _require_cuda_major("Warp toolkit", toolkit, allowed)
    assert wp.is_cuda_available(), "Warp cannot access a CUDA device"
    device = wp.get_device("cuda:0")
    return {
        "version": wp.__version__,
        "toolkit": list(toolkit),
        "driver": list(driver),
        "device": str(device),
        "device_architecture": getattr(device, "arch", None),
    }


def _jax(allowed: set[int]) -> dict[str, Any]:
    import jax

    plugins: dict[int, str] = {}
    for major in (12, 13):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            plugins[major] = importlib.metadata.version(f"jax-cuda{major}-plugin")
    assert plugins, "No JAX CUDA plugin distribution is installed"
    assert set(plugins) <= allowed, (
        f"JAX CUDA plugin majors {sorted(plugins)} are not within {sorted(allowed)}"
    )

    enable_x64 = True
    jax.config.update("jax_enable_x64", enable_x64)
    import jax.numpy as jnp

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    assert devices, "JAX did not initialize a CUDA GPU backend"
    value = jax.device_put(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64), devices[0])
    result = jnp.vdot(value, value)
    result.block_until_ready()
    assert result.dtype == jnp.float64, f"JAX unexpectedly produced {result.dtype}"
    assert float(result) == 14.0, f"JAX FP64 result is {float(result)}"
    return {
        "version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(devices[0]),
        "device_kind": devices[0].device_kind,
        "plugins": {str(major): version for major, version in plugins.items()},
        "fp64": True,
    }


def _wp_vec3(tensor: Any) -> Any:
    import warp as wp

    dtype = wp.types.vector(3, wp.dtype_from_torch(tensor.dtype))
    return wp.from_torch(tensor, dtype=dtype)


def _active_response(
    potential: Any, displacement: Any, direction: Any
) -> tuple[Any, Any, Any]:
    import torch
    import warp as wp

    energy = torch.zeros(1, dtype=displacement.dtype, device=displacement.device)
    gradient = torch.zeros_like(displacement)
    hessian_product = torch.zeros_like(displacement)
    scalar_dtype = wp.dtype_from_torch(displacement.dtype)
    potential.fun(_wp_vec3(displacement), wp.from_torch(energy, dtype=scalar_dtype))
    potential.grad(_wp_vec3(displacement), _wp_vec3(gradient))
    potential.hess_prod(
        _wp_vec3(displacement),
        _wp_vec3(direction),
        _wp_vec3(hessian_product),
    )
    wp.synchronize()
    return energy[0], gradient, hessian_product


def _active_snh() -> dict[str, Any]:
    import numpy as np
    import pyvista as pv
    import torch
    import warp as wp

    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    torch.set_default_dtype(torch.float64)
    points = np.vstack((np.zeros(3), np.eye(3))).astype(np.float64)
    mesh = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA], dtype=np.uint8),
        points,
    )
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.array([2.3], dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.array([1.1], dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.ones(1, dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.array(
        [[0.10, -0.04, 0.03, 0.02, -0.01, 0.015]], dtype=np.float64
    )

    with wp.ScopedDevice("cpu"), torch.device("cpu"):
        cpu_potential = StableNeoHookeanActive.from_pyvista(mesh)
    with wp.ScopedDevice("cuda:0"), torch.device("cuda:0"):
        gpu_potential = StableNeoHookeanActive.from_pyvista(mesh)

    displacement_cpu = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.08, 0.02, -0.01],
            [0.01, -0.04, 0.03],
            [0.00, -0.02, 0.06],
        ],
        dtype=torch.float64,
    )
    direction_cpu = torch.tensor(
        [
            [0.02, -0.03, 0.01],
            [-0.01, 0.04, -0.02],
            [0.03, 0.01, 0.02],
            [-0.02, 0.02, -0.03],
        ],
        dtype=torch.float64,
    )
    displacement = displacement_cpu.cuda()
    direction = direction_cpu.cuda()

    cpu = _active_response(cpu_potential, displacement_cpu, direction_cpu)
    gpu = _active_response(gpu_potential, displacement, direction)
    for gpu_value, cpu_value in zip(gpu, cpu, strict=True):
        torch.testing.assert_close(
            gpu_value.cpu(), cpu_value, rtol=2.0e-11, atol=2.0e-12
        )

    epsilon = 1.0e-5
    plus = _active_response(
        gpu_potential, displacement + epsilon * direction, direction
    )
    minus = _active_response(
        gpu_potential, displacement - epsilon * direction, direction
    )
    energy_direction_fd = (plus[0] - minus[0]) / (2.0 * epsilon)
    energy_direction = torch.sum(gpu[1] * direction)
    torch.testing.assert_close(
        energy_direction, energy_direction_fd, rtol=2.0e-7, atol=2.0e-8
    )
    hessian_product_fd = (plus[1] - minus[1]) / (2.0 * epsilon)
    torch.testing.assert_close(gpu[2], hessian_product_fd, rtol=2.0e-7, atol=2.0e-8)
    return {
        "device": str(displacement.device),
        "dtype": str(displacement.dtype),
        "energy": float(gpu[0]),
        "gradient_direction_fd_error": float(
            torch.abs(energy_direction - energy_direction_fd)
        ),
        "hessian_product_fd_max_error": float(
            torch.max(torch.abs(gpu[2] - hessian_product_fd))
        ),
        "cpu_parity": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cuda13",
        action="store_true",
        help="accept CUDA 13 components for comparison with the baseline environment",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    allowed = {12, 13} if args.allow_cuda13 else {12}
    report: dict[str, Any] = {
        "status": "running",
        "allowed_cuda_majors": sorted(allowed),
    }
    try:
        report["torch"] = _torch_and_sparse(allowed)
        report["cupy"] = _cupy(allowed)
        report["warp"] = _warp(allowed)
        report["jax"] = _jax(allowed)
        report["apple_active_snh"] = _active_snh()
        report["status"] = "ok"
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        raise
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
