from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[6]
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"

type Full = Float[Tensor, "points dim"]
type Scalar = Float[Tensor, ""]


@dataclass(frozen=True)
class Case:
    name: str
    input: Path
    target: Path
    inverse: Path
    use_smas: bool
    smas_stiffness_ratio: float


CASES = (
    Case(
        name="515k-expression000-smas1",
        input=REPO_ROOT / "exp/2026/05/20/inverse-face/data/10-inverse-face-input.vtu",
        target=REPO_ROOT
        / "exp/2026/05/20/inverse-face/data/10-inverse-face-target.vtu",
        inverse=REPO_ROOT
        / "exp/2026/05/20/inverse-face/data/20-inverse-face-fresh-nu049-smas1-noclamp-super-loose-reg.vtu",
        use_smas=True,
        smas_stiffness_ratio=1.0,
    ),
    Case(
        name="3152k-expression000-smas100",
        input=REPO_ROOT
        / "exp/2026/05/20/inverse-face/data/10-inverse-face-3152k-input.vtu",
        target=REPO_ROOT
        / "exp/2026/05/20/inverse-face/data/10-inverse-face-3152k-target.vtu",
        inverse=REPO_ROOT
        / "exp/2026/05/20/inverse-face/data/20-inverse-face-3152k.vtu",
        use_smas=True,
        smas_stiffness_ratio=1.0e2,
    ),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output(
        "22-surface-metric-penalty-expression000-prestrain10-summary.json"
    )
    output_csv: Path = cherries.output(
        "22-surface-metric-penalty-expression000-prestrain10-cases.csv"
    )
    output_table: Path = cherries.output(
        "22-surface-metric-penalty-expression000-prestrain10-table.md"
    )

    cases: tuple[str, ...] = (
        "515k-expression000-smas1",
        "3152k-expression000-smas100",
    )
    target_point_mask: str = "IsFace"
    prestrain: float = 0.10
    surface_stiffness: float = 1.0
    tensile_prestrain: bool = True

    E: float = 1.0
    nu: float = 0.49
    active_fraction_tol: float = 1.0e-3
    forward_rtol: float = 5.0e-4
    forward_atol: float = 0.0
    forward_max_steps: int = 10000


@attrs.define
class SurfaceMetricPrestrain:
    rest_points: Tensor
    triangles: Tensor
    inv_metric: Tensor
    area: Tensor
    stiffness: float
    prestrain_length_scale: float

    def energy(self, u: Full) -> Scalar:
        c00, c01, c10, c11 = self.metric_tensor(u)
        metric_error_sq = (c00 - 1.0).square() + 2.0 * c01 * c10 + (c11 - 1.0).square()
        density = 0.5 * self.stiffness * metric_error_sq
        return torch.sum(self.area * density)

    def invariants(self, u: Full) -> tuple[Tensor, Tensor]:
        c00, c01, c10, c11 = self.metric_tensor(u)
        i1 = c00 + c11
        det_c = c00 * c11 - c01 * c10
        tiny = torch.finfo(det_c.dtype).tiny
        j = torch.sqrt(torch.clamp(det_c, min=tiny))
        return i1, j

    def metric_tensor(self, u: Full) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tri = self.triangles
        x = self.rest_points + u
        p0 = x[tri[:, 0]]
        p1 = x[tri[:, 1]]
        p2 = x[tri[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        g00 = torch.sum(e1 * e1, dim=1)
        g01 = torch.sum(e1 * e2, dim=1)
        g11 = torch.sum(e2 * e2, dim=1)
        inv = self.inv_metric
        c00 = inv[:, 0, 0] * g00 + inv[:, 0, 1] * g01
        c01 = inv[:, 0, 0] * g01 + inv[:, 0, 1] * g11
        c10 = inv[:, 1, 0] * g00 + inv[:, 1, 1] * g01
        c11 = inv[:, 1, 0] * g01 + inv[:, 1, 1] * g11
        return c00, c01, c10, c11

    def diagnostics(self, displacement: np.ndarray) -> dict[str, np.ndarray]:
        tri = self.triangles.detach().cpu().numpy()
        points = self.rest_points.detach().cpu().numpy()
        deformed = points + displacement
        rest_area = triangle_areas(points, tri)
        deformed_area = triangle_areas(deformed, tri)
        i1, area_stretch = self._invariants_numpy(
            points=points, displacement=displacement
        )
        energy_density = self._energy_density_numpy(
            points=points, displacement=displacement
        )
        metric_error_sq = self._metric_error_sq_numpy(
            points=points, displacement=displacement
        )
        return {
            "SurfacePrestrainRestArea": rest_area,
            "SurfacePrestrainDeformedArea": deformed_area,
            "SurfacePrestrainAreaRelChange": safe_rel_change(deformed_area, rest_area),
            "SurfacePrestrainI1": i1,
            "SurfacePrestrainAreaStretch": area_stretch,
            "SurfaceMetricPenaltyErrorSq": metric_error_sq,
            "SurfacePrestrainEnergyDensity": energy_density,
            "SurfacePrestrainEnergy": energy_density * rest_area,
        }

    def _invariants_numpy(
        self, *, points: np.ndarray, displacement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tri = self.triangles.detach().cpu().numpy()
        inv = self.inv_metric.detach().cpu().numpy()
        deformed = points + displacement
        p0 = deformed[tri[:, 0]]
        p1 = deformed[tri[:, 1]]
        p2 = deformed[tri[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        g00 = np.einsum("ij,ij->i", e1, e1)
        g01 = np.einsum("ij,ij->i", e1, e2)
        g11 = np.einsum("ij,ij->i", e2, e2)
        c00 = inv[:, 0, 0] * g00 + inv[:, 0, 1] * g01
        c01 = inv[:, 0, 0] * g01 + inv[:, 0, 1] * g11
        c10 = inv[:, 1, 0] * g00 + inv[:, 1, 1] * g01
        c11 = inv[:, 1, 0] * g01 + inv[:, 1, 1] * g11
        i1 = c00 + c11
        det_c = c00 * c11 - c01 * c10
        area_stretch = np.sqrt(np.maximum(det_c, np.finfo(det_c.dtype).tiny))
        return i1, area_stretch

    def _energy_density_numpy(
        self, *, points: np.ndarray, displacement: np.ndarray
    ) -> np.ndarray:
        metric_error_sq = self._metric_error_sq_numpy(
            points=points, displacement=displacement
        )
        return 0.5 * self.stiffness * metric_error_sq

    def _metric_error_sq_numpy(
        self, *, points: np.ndarray, displacement: np.ndarray
    ) -> np.ndarray:
        c00, c01, c10, c11 = self._metric_tensor_numpy(
            points=points, displacement=displacement
        )
        return np.maximum((c00 - 1.0) ** 2 + 2.0 * c01 * c10 + (c11 - 1.0) ** 2, 0.0)

    def _metric_tensor_numpy(
        self, *, points: np.ndarray, displacement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tri = self.triangles.detach().cpu().numpy()
        inv = self.inv_metric.detach().cpu().numpy()
        deformed = points + displacement
        p0 = deformed[tri[:, 0]]
        p1 = deformed[tri[:, 1]]
        p2 = deformed[tri[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        g00 = np.einsum("ij,ij->i", e1, e1)
        g01 = np.einsum("ij,ij->i", e1, e2)
        g11 = np.einsum("ij,ij->i", e2, e2)
        c00 = inv[:, 0, 0] * g00 + inv[:, 0, 1] * g01
        c01 = inv[:, 0, 0] * g01 + inv[:, 0, 1] * g11
        c10 = inv[:, 1, 0] * g00 + inv[:, 1, 1] * g01
        c11 = inv[:, 1, 0] * g01 + inv[:, 1, 1] * g11
        return c00, c01, c10, c11


@attrs.define
class SurfacePrestrainModel:
    base: Any
    surface: SurfaceMetricPrestrain

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def dof_map(self) -> Any:
        return self.base.dof_map

    @property
    def n_fixed(self) -> int:
        return self.base.n_fixed

    @property
    def n_free(self) -> int:
        return self.base.n_free

    @property
    def n_full(self) -> int:
        return self.base.n_full

    @property
    def n_points(self) -> int:
        return self.base.n_points

    def get_materials(self) -> dict[str, dict[str, Tensor]]:
        return self.base.get_materials()

    def set_materials(self, materials: dict[str, dict[str, Tensor]]) -> None:
        self.base.set_materials(materials)

    def init(self) -> Any:
        return self.base.init()

    def max_step_size(self, state: Any, p: Full) -> Scalar:
        return self.base.max_step_size(state, p)

    def update(self, state: Any, u: Full) -> None:
        self.base.update(state, u)

    def fun(self, state: Any) -> Scalar:
        return self.base.fun(state) + self.surface.energy(state.u)

    def grad(self, state: Any) -> Full:
        output = self.base.grad(state)
        with torch.enable_grad():
            u = state.u.detach().requires_grad_(True)
            energy = self.surface.energy(u)
            (grad,) = torch.autograd.grad(energy, u)
        return output + grad.detach()

    def hess_diag(self, state: Any) -> Full:
        return self.base.hess_diag(state)

    def hess_prod(self, state: Any, p: Full) -> Full:
        output = self.base.hess_prod(state, p)
        with torch.enable_grad():
            u = state.u.detach().requires_grad_(True)
            p_local = p.detach()
            energy = self.surface.energy(u)
            (grad,) = torch.autograd.grad(energy, u, create_graph=True)
            (hess_prod,) = torch.autograd.grad(torch.sum(grad * p_local), u)
        return output + hess_prod.detach()

    def hess_quad(self, state: Any, p: Full) -> Scalar:
        return self.base.hess_quad(state, p) + torch.sum(
            self.hess_prod_surface(state, p) * p
        )

    def hess_prod_surface(self, state: Any, p: Full) -> Full:
        with torch.enable_grad():
            u = state.u.detach().requires_grad_(True)
            p_local = p.detach()
            energy = self.surface.energy(u)
            (grad,) = torch.autograd.grad(energy, u, create_graph=True)
            (hess_prod,) = torch.autograd.grad(torch.sum(grad * p_local), u)
        return hess_prod.detach()

    def mixed_derivative_prod(self, state: Any, p: Full) -> Full:
        return self.base.mixed_derivative_prod(state, p)


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}"
    raise FileNotFoundError(msg)


def require_array(obj: pv.DataSet, association: str, name: str) -> np.ndarray:
    data = obj.cell_data if association == "cell" else obj.point_data
    if name not in data:
        msg = f"{association}_data[{name!r}] is missing"
        raise KeyError(msg)
    return np.asarray(data[name])


def load_meshes(
    case: Case,
) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid, pv.UnstructuredGrid]:
    for path in (case.input, case.target, case.inverse):
        require_path(path)
        cherries.log_input(path)
    mesh = pv.read(case.input)
    target = pv.read(case.target)
    inverse = pv.read(case.inverse)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if not isinstance(target, pv.UnstructuredGrid):
        target = target.cast_to_unstructured_grid()
    if not isinstance(inverse, pv.UnstructuredGrid):
        inverse = inverse.cast_to_unstructured_grid()
    for label, other in (("target", target), ("inverse", inverse)):
        if mesh.n_points != other.n_points or mesh.n_cells != other.n_cells:
            msg = (
                f"{case.name} {label} topology differs from input: "
                f"points {other.n_points} != {mesh.n_points}, "
                f"cells {other.n_cells} != {mesh.n_cells}"
            )
            raise ValueError(msg)
        if not np.allclose(mesh.points, other.points):
            msg = f"{case.name} {label} rest points differ from input"
            raise ValueError(msg)
    return mesh, target, inverse


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def set_material(
    mesh: pv.UnstructuredGrid,
    *,
    E: float,
    nu: float,
    fraction: np.ndarray,
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_parameters(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.asarray(fraction, dtype=np.float64)


def build_base_forward(mesh: pv.UnstructuredGrid, case: Case, cfg: Config) -> Any:
    from liblaf.apple.forward import ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[BACKGROUND_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(
        mesh,
        E=case.smas_stiffness_ratio * cfg.E,
        nu=cfg.nu,
        fraction=mesh.cell_data[ACTIVE_FRACTION],
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    if case.use_smas:
        set_material(
            mesh,
            E=case.smas_stiffness_ratio * cfg.E,
            nu=cfg.nu,
            fraction=mesh.cell_data[SMAS_STIFFNESS_FRACTION],
        )
        builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="smas"))

    return builder.finalize()


def target_point_ids(
    mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid, cfg: Config
) -> np.ndarray:
    if cfg.target_point_mask in target.point_data:
        mask = np.asarray(target.point_data[cfg.target_point_mask], dtype=bool)
    elif cfg.target_point_mask in mesh.point_data:
        mask = np.asarray(mesh.point_data[cfg.target_point_mask], dtype=bool)
    elif TARGET_SURFACE_MASK in target.point_data:
        mask = np.asarray(target.point_data[TARGET_SURFACE_MASK], dtype=bool)
    else:
        msg = (
            f"case has neither point_data[{cfg.target_point_mask!r}] nor "
            f"point_data[{TARGET_SURFACE_MASK!r}]"
        )
        raise KeyError(msg)
    ids = np.flatnonzero(mask).astype(np.int64)
    if ids.size == 0:
        msg = "target point mask selected no points"
        raise ValueError(msg)
    return ids


def active_cell_ids(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    if "ActivationMask" in mesh.cell_data:
        mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    else:
        mask = (
            np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
            > cfg.active_fraction_tol
        )
    ids = np.flatnonzero(mask).astype(np.int64)
    if ids.size == 0:
        msg = "no active tetrahedra selected"
        raise ValueError(msg)
    return ids


def recovered_activation_inv(inverse: pv.UnstructuredGrid) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION_INV

    if "RecoveredActivationInv" in inverse.cell_data:
        return np.asarray(inverse.cell_data["RecoveredActivationInv"], dtype=np.float64)
    return np.asarray(inverse.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)


def surface_triangles(
    mesh: pv.UnstructuredGrid, mask_name: str
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    triangles = original_ids[faces[:, 1:]]
    if mask_name not in mesh.point_data:
        msg = f"input mesh has no point_data[{mask_name!r}]"
        raise KeyError(msg)
    point_mask = np.asarray(mesh.point_data[mask_name], dtype=bool)
    triangle_mask = np.all(point_mask[triangles], axis=1)
    selected = triangles[triangle_mask]
    if selected.size == 0:
        msg = f"no surface triangles selected by all-vertices {mask_name}"
        raise ValueError(msg)
    return surface, selected.astype(np.int64), triangle_mask


def make_surface_prestrain(
    mesh: pv.UnstructuredGrid,
    triangles: np.ndarray,
    cfg: Config,
) -> SurfaceMetricPrestrain:
    points = np.asarray(mesh.points, dtype=np.float64)
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    g00 = np.einsum("ij,ij->i", e1, e1)
    g01 = np.einsum("ij,ij->i", e1, e2)
    g11 = np.einsum("ij,ij->i", e2, e2)
    det = g00 * g11 - g01 * g01
    if np.any(det <= 0.0):
        msg = "surface triangle metric is singular"
        raise ValueError(msg)
    inv = np.empty((triangles.shape[0], 2, 2), dtype=np.float64)
    inv[:, 0, 0] = g11 / det
    inv[:, 0, 1] = -g01 / det
    inv[:, 1, 0] = -g01 / det
    inv[:, 1, 1] = g00 / det
    area = triangle_areas(points, triangles)
    length_scale = (
        1.0 / (1.0 + cfg.prestrain) if cfg.tensile_prestrain else 1.0 + cfg.prestrain
    )
    inv *= 1.0 / (length_scale * length_scale)
    return SurfaceMetricPrestrain(
        rest_points=torch.as_tensor(
            points, dtype=torch.get_default_dtype(), device=torch.get_default_device()
        ),
        triangles=torch.as_tensor(
            triangles, dtype=torch.long, device=torch.get_default_device()
        ),
        inv_metric=torch.as_tensor(
            inv, dtype=torch.get_default_dtype(), device=torch.get_default_device()
        ),
        area=torch.as_tensor(
            area, dtype=torch.get_default_dtype(), device=torch.get_default_device()
        ),
        stiffness=float(cfg.surface_stiffness),
        prestrain_length_scale=float(length_scale),
    )


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def safe_rel_change(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (
        np.divide(
            value,
            reference,
            out=np.full_like(value, np.nan, dtype=np.float64),
            where=reference != 0.0,
        )
        - 1.0
    )


def point_error_stats(
    displacement: np.ndarray, target: np.ndarray, ids: np.ndarray
) -> dict[str, float]:
    residual = displacement[ids] - target[ids]
    norm = np.linalg.norm(residual, axis=1)
    target_norm = np.linalg.norm(target[ids], axis=1)
    target_rms = float(np.linalg.norm(target[ids]) / math.sqrt(ids.size))
    rms = float(np.linalg.norm(residual) / math.sqrt(ids.size))
    return {
        "error_mean": float(norm.mean()),
        "error_rms": rms,
        "error_max": float(norm.max()),
        "error_rms_fraction_of_target": rms / target_rms if target_rms else math.nan,
        "target_rms": target_rms,
        "target_max": float(target_norm.max()),
    }


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def roughness_metrics(
    displacement: np.ndarray,
    target: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, float]:
    edges = unique_edges(triangles)
    disp_edge = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    err = displacement - target
    err_edge = err[edges[:, 0]] - err[edges[:, 1]]
    disp_edge_norm = np.linalg.norm(disp_edge, axis=1)
    err_edge_norm = np.linalg.norm(err_edge, axis=1)

    n_points = displacement.shape[0]
    neighbor_sum = np.zeros_like(displacement)
    neighbor_count = np.zeros(n_points, dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], displacement[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], displacement[edges[:, 0]])
    np.add.at(neighbor_count, edges[:, 0], 1.0)
    np.add.at(neighbor_count, edges[:, 1], 1.0)
    active = neighbor_count > 0.0
    lap = np.zeros_like(displacement)
    lap[active] = (
        displacement[active] - neighbor_sum[active] / neighbor_count[active, None]
    )
    lap_norm = np.linalg.norm(lap[active], axis=1)
    return {
        "surface/n_triangles": int(triangles.shape[0]),
        "surface/n_edges": int(edges.shape[0]),
        "surface/displacement_edge_rms": float(
            np.linalg.norm(disp_edge_norm) / math.sqrt(edges.shape[0])
        ),
        "surface/error_edge_rms": float(
            np.linalg.norm(err_edge_norm) / math.sqrt(edges.shape[0])
        ),
        "surface/displacement_laplacian_rms": float(
            np.linalg.norm(lap_norm) / math.sqrt(lap_norm.size)
        ),
        "surface/displacement_laplacian_max": float(lap_norm.max()),
    }


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "forward/result": "missing",
            "forward/success": False,
            "forward/steps": math.nan,
            "forward/grad_norm": math.nan,
            "forward/relative_grad_norm": math.nan,
        }
    state = solution.state.convergence_state
    grad_norm = float(state.grad_norm.detach().cpu())
    grad_norm_first = float(state.grad_norm_first.detach().cpu())
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": grad_norm / grad_norm_first
        if grad_norm_first
        else math.nan,
    }


def solve_case(case: Case, cfg: Config) -> dict[str, Any]:
    from liblaf.apple.forward import Forward

    start = time.perf_counter()
    mesh, target, inverse = load_meshes(case)
    target_ids = target_point_ids(mesh, target, cfg)
    active_ids = active_cell_ids(mesh, cfg)
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    previous_displacement = np.asarray(
        inverse.point_data["Displacement"], dtype=np.float64
    )
    activation_inv = recovered_activation_inv(inverse)

    surface, prestrain_triangles, surface_triangle_mask = surface_triangles(
        mesh, cfg.target_point_mask
    )
    surface_energy = make_surface_prestrain(mesh, prestrain_triangles, cfg)
    base_model = build_base_forward(mesh, case, cfg)
    base_materials = base_model.get_materials()
    base_materials["muscle"]["activation_inv"] = torch.as_tensor(
        activation_inv,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    base_model.set_materials(base_materials)
    model = SurfacePrestrainModel(base=base_model, surface=surface_energy)
    forward = Forward(model)
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    with torch.no_grad():
        forward.state.u.copy_(
            torch.as_tensor(
                previous_displacement,
                dtype=forward.state.u.dtype,
                device=forward.state.u.device,
            )
        )
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    prestrain_displacement = forward.state.u.detach().cpu().numpy()

    previous_error = point_error_stats(
        previous_displacement, target_displacement, target_ids
    )
    prestrain_error = point_error_stats(
        prestrain_displacement, target_displacement, target_ids
    )
    previous_roughness = roughness_metrics(
        previous_displacement, target_displacement, prestrain_triangles
    )
    prestrain_roughness = roughness_metrics(
        prestrain_displacement, target_displacement, prestrain_triangles
    )
    surface_diag = surface_energy.diagnostics(prestrain_displacement)
    surface_energy_total = float(np.sum(surface_diag["SurfacePrestrainEnergy"]))
    elapsed_s = time.perf_counter() - start

    row: dict[str, Any] = {
        "case": case.name,
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "target/n_points": int(target_ids.size),
        "activation/n_active_tets": int(active_ids.size),
        "surface/n_prestrain_triangles": int(prestrain_triangles.shape[0]),
        "surface/rest_area": float(surface_energy.area.detach().cpu().sum()),
        "surface/prestrain": float(cfg.prestrain),
        "surface/stiffness": float(cfg.surface_stiffness),
        "surface/tensile_prestrain": bool(cfg.tensile_prestrain),
        "surface/prestrain_length_scale": float(surface_energy.prestrain_length_scale),
        "surface/energy_model": "metric_penalty",
        "surface/energy": surface_energy_total,
        "time/total_s": elapsed_s,
        **{f"previous/target/{k}": v for k, v in previous_error.items()},
        **{f"prestrain/target/{k}": v for k, v in prestrain_error.items()},
        **{f"previous/{k}": v for k, v in previous_roughness.items()},
        **{f"prestrain/{k}": v for k, v in prestrain_roughness.items()},
        **forward_solution_metrics(solution),
    }
    row["delta/target_error_rms"] = (
        row["prestrain/target/error_rms"] - row["previous/target/error_rms"]
    )
    row["delta/target_error_max"] = (
        row["prestrain/target/error_max"] - row["previous/target/error_max"]
    )
    row["delta/surface_displacement_laplacian_rms"] = (
        row["prestrain/surface/displacement_laplacian_rms"]
        - row["previous/surface/displacement_laplacian_rms"]
    )
    row["delta/surface_error_edge_rms"] = (
        row["prestrain/surface/error_edge_rms"] - row["previous/surface/error_edge_rms"]
    )

    output = make_result_mesh(
        mesh=mesh,
        target_displacement=target_displacement,
        previous_displacement=previous_displacement,
        prestrain_displacement=prestrain_displacement,
        activation_inv=activation_inv,
        target_ids=target_ids,
        active_ids=active_ids,
        metrics=row,
    )
    output_path = (
        cfg.output_summary.parent
        / f"22-surface-metric-penalty-expression000-prestrain10-{case.name}.vtu"
    )
    surface_path = (
        cfg.output_summary.parent
        / f"22-surface-metric-penalty-expression000-prestrain10-{case.name}-surface.vtp"
    )
    melon.save(output_path, output)
    save_surface_diagnostic(
        path=surface_path,
        surface=surface,
        surface_triangle_mask=surface_triangle_mask,
        mesh=mesh,
        target_displacement=target_displacement,
        previous_displacement=previous_displacement,
        prestrain_displacement=prestrain_displacement,
        surface_diag=surface_diag,
    )
    cherries.log_output(output_path)
    cherries.log_output(surface_path)
    cherries.log_metrics(
        {
            f"{case.name}/previous/error_rms": row["previous/target/error_rms"],
            f"{case.name}/prestrain/error_rms": row["prestrain/target/error_rms"],
            f"{case.name}/previous/laplacian_rms": row[
                "previous/surface/displacement_laplacian_rms"
            ],
            f"{case.name}/prestrain/laplacian_rms": row[
                "prestrain/surface/displacement_laplacian_rms"
            ],
        }
    )
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", surface_path)
    return row


def make_result_mesh(
    *,
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    previous_displacement: np.ndarray,
    prestrain_displacement: np.ndarray,
    activation_inv: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, Any],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    target_mask = np.zeros(mesh.n_points, dtype=np.int8)
    target_mask[target_ids] = 1
    active_mask = np.zeros(mesh.n_cells, dtype=np.int8)
    active_mask[active_ids] = 1
    result.point_data[TARGET_SURFACE_MASK] = target_mask
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["PreviousInverseDisplacement"] = previous_displacement
    result.point_data["PrestrainDisplacement"] = prestrain_displacement
    result.point_data["Displacement"] = prestrain_displacement
    result.point_data["PreviousMinusTarget"] = (
        previous_displacement - target_displacement
    )
    result.point_data["PrestrainMinusTarget"] = (
        prestrain_displacement - target_displacement
    )
    result.point_data["PrestrainMinusPrevious"] = (
        prestrain_displacement - previous_displacement
    )
    result.point_data["PreviousErrorNorm"] = np.linalg.norm(
        previous_displacement - target_displacement, axis=1
    )
    result.point_data["PrestrainErrorNorm"] = np.linalg.norm(
        prestrain_displacement - target_displacement, axis=1
    )
    result.point_data["PrestrainDeltaNorm"] = np.linalg.norm(
        prestrain_displacement - previous_displacement, axis=1
    )
    result.point_data["DeformedPoint"] = mesh.points + prestrain_displacement
    result.point_data["PreviousInversePoint"] = mesh.points + previous_displacement
    result.point_data["TargetPoint"] = mesh.points + target_displacement
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["RecoveredActivationInv"] = activation_inv
    result.cell_data["InverseActiveMask"] = active_mask
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        result.field_data[name] = np.asarray([value])
    return result


def save_surface_diagnostic(
    *,
    path: Path,
    surface: pv.PolyData,
    surface_triangle_mask: np.ndarray,
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    previous_displacement: np.ndarray,
    prestrain_displacement: np.ndarray,
    surface_diag: dict[str, np.ndarray],
) -> None:
    result = surface.copy(deep=True)
    original_ids = np.asarray(result.point_data["vtkOriginalPointIds"], dtype=np.int64)
    result.point_data["TargetDisplacement"] = target_displacement[original_ids]
    result.point_data["PreviousInverseDisplacement"] = previous_displacement[
        original_ids
    ]
    result.point_data["PrestrainDisplacement"] = prestrain_displacement[original_ids]
    result.point_data["PrestrainMinusPrevious"] = (
        prestrain_displacement[original_ids] - previous_displacement[original_ids]
    )
    result.point_data["PreviousErrorNorm"] = np.linalg.norm(
        previous_displacement[original_ids] - target_displacement[original_ids], axis=1
    )
    result.point_data["PrestrainErrorNorm"] = np.linalg.norm(
        prestrain_displacement[original_ids] - target_displacement[original_ids], axis=1
    )
    result.point_data["PrestrainDeltaNorm"] = np.linalg.norm(
        prestrain_displacement[original_ids] - previous_displacement[original_ids],
        axis=1,
    )
    result.point_data["IsFace"] = np.asarray(mesh.point_data["IsFace"], dtype=np.int8)[
        original_ids
    ]
    result.cell_data["SurfacePrestrainTriangle"] = surface_triangle_mask.astype(np.int8)
    for name, values in surface_diag.items():
        full = np.full(surface.n_cells, np.nan, dtype=np.float64)
        full[surface_triangle_mask] = values
        result.cell_data[name] = full
    melon.save(path, result)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_float(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, int | float):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | forward steps | forward success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    format_float(row["target/n_points"]),
                    format_float(row["surface/n_prestrain_triangles"]),
                    format_float(row["previous/target/error_rms"]),
                    format_float(row["prestrain/target/error_rms"]),
                    format_float(row["delta/target_error_rms"]),
                    format_float(row["previous/surface/displacement_laplacian_rms"]),
                    format_float(row["prestrain/surface/displacement_laplacian_rms"]),
                    format_float(row["delta/surface_displacement_laplacian_rms"]),
                    format_float(row["forward/steps"]),
                    format_float(row["forward/success"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    configure_runtime()
    selected = {case.name: case for case in CASES}
    rows = []
    for name in cfg.cases:
        if name not in selected:
            msg = f"unknown case {name!r}; choose from {sorted(selected)}"
            raise ValueError(msg)
        rows.append(solve_case(selected[name], cfg))
    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
