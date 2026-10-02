# ruff: noqa: EM101, TRY003
"""CPU algebra and transaction checks for the coupled equilibrium predictor."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_coupled_predictor import equilibrium_predictor, rotation_sagitta
from joint_equilibrium import ForwardConvergenceError

from liblaf import cherries
from liblaf.apple.inverse._diff_forward import _AdjointProblem


class DenseDofMap:
    def __init__(self, size: int, fixed_index: int, fixed_value: float) -> None:
        self.size = size
        self.fixed_indices = torch.tensor([fixed_index])
        self.free_indices = torch.tensor(
            [index for index in range(size) if index != fixed_index]
        )
        self.fixed_values = torch.tensor([fixed_value])

    def to_free(self, full: torch.Tensor) -> torch.Tensor:
        return full[self.free_indices]

    def to_free_grad(self, full: torch.Tensor) -> torch.Tensor:
        return full[self.free_indices]

    def to_full(self, free: torch.Tensor) -> torch.Tensor:
        full = torch.empty(self.size, dtype=free.dtype, device=free.device)
        full[self.free_indices] = free
        full[self.fixed_indices] = self.fixed_values
        return full

    def to_full_grad(self, free: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(self.size, dtype=free.dtype, device=free.device)
        full[self.free_indices] = free
        return full

    def to_free_hess_diag(self, diagonal: torch.Tensor) -> torch.Tensor:
        return diagonal[self.free_indices]


class DenseWarp:
    def __init__(self, model: DenseModel) -> None:
        self.model = model

    def grad(self, displacement: torch.Tensor, output: torch.Tensor) -> None:
        output.add_(self.model.hessian @ displacement - self.model.load())


class DenseCollision:
    def state_at(self, displacement: torch.Tensor) -> dict[str, float]:
        return {"norm": float(torch.linalg.vector_norm(displacement))}


class DenseModel:
    @dataclasses.dataclass
    class State:
        u: torch.Tensor
        collision: Any = None

    def __init__(
        self, hessian: torch.Tensor, fixed_value: float, material: dict
    ) -> None:
        self.hessian = hessian
        self.dof_map = DenseDofMap(len(hessian), len(hessian) - 1, fixed_value)
        self._materials = material
        self.warp_model = DenseWarp(self)
        self.collision = DenseCollision()

    def load(self) -> torch.Tensor:
        return self._materials["bulk"]["load"]

    def get_materials(self) -> dict:
        return self._materials

    def set_materials(self, materials: dict) -> None:
        self._materials = materials

    def grad(self, state: State) -> torch.Tensor:
        return self.hessian @ state.u - self.load()

    def hess_prod(self, state: State, direction: torch.Tensor) -> torch.Tensor:
        del state
        return self.hessian @ direction

    def hess_diag(self, state: State) -> torch.Tensor:
        del state
        return torch.diagonal(self.hessian)


class DenseSolver:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success

    def solve(self, system: _AdjointProblem, initial: torch.Tensor) -> SimpleNamespace:
        del initial
        columns = []
        for index in range(len(system.b)):
            basis = torch.zeros_like(system.b)
            basis[index] = 1
            columns.append(system.matvec(basis))
        matrix = torch.stack(columns, dim=1)
        return SimpleNamespace(
            success=self.success,
            params=torch.linalg.solve(matrix, system.b),
            result="dense_direct",
        )


def materials(load: list[float]) -> dict:
    return {"bulk": {"load": torch.tensor(load)}}


def equilibrium_for(
    hessian: torch.Tensor, load: torch.Tensor, fixed: float
) -> torch.Tensor:
    free = torch.linalg.solve(hessian[:-1, :-1], load[:-1] - hessian[:-1, -1] * fixed)
    return torch.cat((free, torch.tensor([fixed])))


def make_model() -> tuple[DenseModel, torch.Tensor, dict]:
    hessian = torch.tensor(
        (
            (5.0, 1.0, -0.5, 0.75),
            (1.0, 4.0, 0.4, -0.2),
            (-0.5, 0.4, 3.0, 0.6),
            (0.75, -0.2, 0.6, 2.0),
        )
    )
    old = materials([1.0, -0.5, 0.7, 0.0])
    model = DenseModel(hessian, fixed_value=0.91, material=materials([9.0] * 4))
    old_u = equilibrium_for(hessian, old["bulk"]["load"], fixed=0.2)
    return model, old_u, old


def pose_and_material_rhs_case() -> dict:
    model, old_u, old = make_model()
    new = materials([1.5, 0.1, -0.2, 0.0])
    target = torch.tensor([0.55])
    original_material = model.load().clone()
    original_fixed = model.dof_map.fixed_values.clone()
    predicted, receipt = equilibrium_predictor(
        model=model,
        solver=DenseSolver(),
        displacement=old_u,
        old_materials=old,
        new_materials=new,
        fixed_target=target,
        linear_rtol=1e-7,
    )
    expected = equilibrium_for(model.hessian, new["bulk"]["load"], fixed=0.55)
    torch.testing.assert_close(predicted, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(model.load(), original_material)
    torch.testing.assert_close(model.dof_map.fixed_values, original_fixed)
    assert receipt["old_free_force_norm"] < 1e-13
    assert receipt["parameter_force_change_norm"] > 0
    assert receipt["boundary_force_change_norm"] > 0
    return {
        "receipt": receipt,
        "prediction_error": float((predicted - expected).abs().max()),
    }


def zero_change_identity_case() -> dict:
    model, old_u, old = make_model()
    predicted, receipt = equilibrium_predictor(
        model=model,
        solver=DenseSolver(),
        displacement=old_u,
        old_materials=old,
        new_materials=old,
        fixed_target=torch.tensor([0.2]),
        linear_rtol=1e-7,
    )
    torch.testing.assert_close(predicted, old_u, atol=0, rtol=0)
    assert receipt["linear_result"] == "zero right-hand side"
    return receipt


def residual_correction_case() -> dict:
    model, old_u, old = make_model()
    perturbed = old_u.clone()
    perturbed[0] += 0.07
    predicted, receipt = equilibrium_predictor(
        model=model,
        solver=DenseSolver(),
        displacement=perturbed,
        old_materials=old,
        new_materials=old,
        fixed_target=torch.tensor([0.2]),
        linear_rtol=1e-7,
    )
    torch.testing.assert_close(predicted, old_u, atol=1e-13, rtol=1e-13)
    assert receipt["old_free_force_norm"] > 0
    scaled, scaled_receipt = equilibrium_predictor(
        model=model,
        solver=DenseSolver(),
        displacement=perturbed,
        old_materials=old,
        new_materials=old,
        fixed_target=torch.tensor([0.2]),
        linear_rtol=1e-7,
        residual_scale=0.25,
    )
    expected_scaled = perturbed - 0.25 * (perturbed - old_u)
    torch.testing.assert_close(scaled, expected_scaled, atol=1e-13, rtol=1e-13)
    assert scaled_receipt["residual_scale"] == 0.25
    return {"full_correction": receipt, "quarter_correction": scaled_receipt}


def rejection_cases() -> dict:
    model, old_u, old = make_model()
    try:
        equilibrium_predictor(
            model=model,
            solver=DenseSolver(success=False),
            displacement=old_u,
            old_materials=old,
            new_materials=old,
            fixed_target=torch.tensor([0.3]),
            linear_rtol=1e-7,
        )
    except ForwardConvergenceError as error:
        unresolved = type(error).__name__
    else:
        raise AssertionError("unresolved linear solve was accepted")
    bad = old_u.clone()
    bad[0] = torch.nan
    try:
        equilibrium_predictor(
            model=model,
            solver=DenseSolver(),
            displacement=bad,
            old_materials=old,
            new_materials=old,
            fixed_target=torch.tensor([0.2]),
            linear_rtol=1e-7,
        )
    except AssertionError:
        nonfinite = "AssertionError"
    else:
        raise AssertionError("nonfinite displacement was accepted")
    return {"unresolved_solver": unresolved, "nonfinite_displacement": nonfinite}


def sagitta_cases() -> dict:
    assert rotation_sagitta(0.0, 0.7) == 0.0
    angle = math.pi / 180
    measured = rotation_sagitta(0.1, angle)
    expected = 2 * 0.1 * math.sin(angle / 4) ** 2
    torch.testing.assert_close(torch.tensor(measured), torch.tensor(expected))
    assert measured > 0
    return {"zero_radius": 0.0, "radius_m": 0.1, "one_degree_m": measured}


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/coupled-predictor-check-001"


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    receipt = {
        "schema": "coupled-equilibrium-predictor-cpu-check-v1",
        "success": True,
        "cpu_only": True,
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__)),
            str(
                Path(__file__).with_name("joint_coupled_predictor.py").resolve()
            ): sha256(Path(__file__).with_name("joint_coupled_predictor.py")),
        },
        "pose_and_material_rhs": pose_and_material_rhs_case(),
        "zero_change_identity": zero_change_identity_case(),
        "old_residual_correction": residual_correction_case(),
        "rejections": rejection_cases(),
        "rotation_sagitta": sagitta_cases(),
        "geometry_audit_scope": "algebra/state transaction and sagitta checks only; production full-source IPC audit remains required",
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
