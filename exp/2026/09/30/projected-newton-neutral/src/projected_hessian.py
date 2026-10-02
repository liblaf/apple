# ruff: noqa: SLF001
"""Per-element PSD projection for the assembled sparse Newton Hessian.

This patches the experiment-local ``AssembledFemHvp`` numeric refresh so each
dense local element Hessian (12x12 bulk tetrahedron, 9x9 membrane triangle) is
symmetrized and eigenvalue-clamped before it is scattered into the global BSR
matrix, and asks IPC Toolkit for its per-contact PSD-projected barrier Hessian.
The result is the standard projected-Newton search matrix (Teran et al. 2005;
IPC, Li et al. 2020).  Energy, force, line search, CCD and the stopping rule
keep the exact physics; only the Newton search direction changes.

With ``mode="none"`` the patched refresh is algebraically the original exact
assembly, which is used to validate the refactor against the exact HVP.
"""

from __future__ import annotations

import time
import weakref
from typing import Any, Literal

import assembled_fem_hvp
import hybrid_first_solver
import ipctk
import torch
import warp as wp
from assembled_fem_hvp import AssembledFemHvp, _wp_u
from hybrid_hessian import HybridHessian, _state_key

Mode = Literal["none", "clamp", "abs"]
SCREEN = 1e-10
EIGH_BATCH = 4096

_IPC_METHOD = {
    "none": ipctk.PSDProjectionMethod.NONE,
    "clamp": ipctk.PSDProjectionMethod.CLAMP,
    "abs": ipctk.PSDProjectionMethod.ABS,
}

STATS: dict[str, Any] = {
    "mode": "none",
    "cell_batch": 0,
    "refreshes": 0,
    "projection_seconds": 0.0,
    "eigh_dtype": torch.float32,
    "last": {},
}


def project_local(local: torch.Tensor, mode: Mode) -> tuple[torch.Tensor, int]:
    """Return the PSD-projected batch and the count of elements changed."""
    local = 0.5 * (local + local.transpose(1, 2))
    if mode == "none":
        return local, 0
    # Element Hessians are singular along rigid modes, so screen with a tiny
    # relative shift: Cholesky success certifies lambda_min > -SCREEN * scale.
    # Only the (usually few) failures pay for a batched eigendecomposition.
    scale = (
        local.diagonal(dim1=1, dim2=2)
        .abs()
        .amax(dim=1)
        .clamp_min(torch.finfo(local.dtype).tiny)
    )
    eye = torch.eye(local.shape[1], device=local.device, dtype=local.dtype)
    _, info = torch.linalg.cholesky_ex(local + (SCREEN * scale)[:, None, None] * eye)
    suspect = torch.nonzero(info != 0).squeeze(1)
    if suspect.numel() == 0:
        return local, 0
    rebuilt = []
    # cuSOLVER's batched eigh workspace grows superlinearly past ~4k matrices.
    for chunk in suspect.split(EIGH_BATCH):
        block = local.index_select(0, chunk)
        block_scale = scale.index_select(0, chunk)[:, None, None]
        # Scaled float32 eigenpairs are ~4.5x faster.  Only the negative part
        # is removed, in float64: H+ = H - V min(L, 0) V^T (``abs``: twice it).
        # The residual indefiniteness is ~1e-6 of the element scale, which is
        # harmless for a search matrix; CG still rejects negative curvature.
        eigenvalues, eigenvectors = torch.linalg.eigh(
            (block / block_scale).to(STATS["eigh_dtype"])
        )
        negative = eigenvalues.clamp_max(0).to(local.dtype)
        vectors = eigenvectors.to(local.dtype)
        correction = (vectors * negative[:, None, :]) @ vectors.transpose(1, 2)
        factor = 2.0 if mode == "abs" else 1.0
        rebuilt.append(block - factor * block_scale * correction)
    return local.index_copy(0, suspect, torch.cat(rebuilt)), int(suspect.numel())


class ProjectedAssembledFemHvp(AssembledFemHvp):
    """``AssembledFemHvp`` whose numeric refresh PSD-projects each element."""

    def __init__(self, model: Any, state: Any, mode: Mode) -> None:
        self.mode = mode
        self._slots_gpu: dict[str, torch.Tensor] = {}
        super().__init__(model, state)  # calls self.setup(state)
        self.metadata["method"] = f"per-element PSD projected ({mode}) FEM sparse BSR"
        self.metadata["exact_physical_fem_hessian"] = mode == "none"

    def setup(self, state: Any) -> None:
        if tuple(state.u.shape) != self._state_shape:
            raise ValueError("state shape differs from assembled FEM topology")
        if state.u.device != self._state_device or state.u.dtype != self._state_dtype:
            raise ValueError("state device or dtype differs from assembled FEM matrix")
        if not self._slots_gpu:
            # Upload scatter slots once, not once per batch per refresh.
            for spec in self._topology.potentials:
                self._slots_gpu[spec.name] = torch.as_tensor(
                    spec.slots.reshape(-1), device=state.u.device, dtype=torch.long
                )
        batch = STATS["cell_batch"]
        assembly_started = time.perf_counter()
        projection_seconds = 0.0
        changed: dict[str, int] = {}
        self._values.zero_()
        u = _wp_u(state.u)
        stream = wp.stream_from_torch(torch.cuda.current_stream(state.u.device))
        with wp.ScopedStream(stream):
            for spec in self._topology.potentials:
                potential = self._potentials[spec.name]
                kernel = self._kernels[spec.name]
                slots = self._slots_gpu[spec.name]
                vertices = spec.vertices_per_cell
                n = 3 * vertices
                changed[spec.name] = 0
                for start in range(0, len(spec.cells), batch):
                    stop = min(start + batch, len(spec.cells))
                    local = torch.empty(
                        (stop - start, n, n), device=state.u.device, dtype=state.u.dtype
                    )
                    wp.launch(
                        kernel,
                        dim=(stop - start, n),
                        inputs=[u, potential.cells, potential.materials, start],
                        outputs=[
                            wp.from_torch(local, dtype=wp.dtype_from_torch(local.dtype))
                        ],
                        device=potential.cells.device,
                    )
                    projection_started = time.perf_counter()
                    local, count = project_local(local, self.mode)
                    projection_seconds += time.perf_counter() - projection_started
                    changed[spec.name] += count
                    blocks = (
                        local.reshape(stop - start, vertices, 3, vertices, 3)
                        .permute(0, 1, 3, 2, 4)
                        .reshape(-1, 3, 3)
                    )
                    self._values.index_add_(
                        0, slots[start * vertices**2 : stop * vertices**2], blocks
                    )
        self.matrix = torch.sparse_bsr_tensor(
            self._crow,
            self._col,
            self._values,
            size=(3 * self.n_points, 3 * self.n_points),
            device=state.u.device,
            dtype=state.u.dtype,
        )
        self.metadata["numeric_setup_seconds"] = time.perf_counter() - assembly_started
        STATS["refreshes"] += 1
        STATS["projection_seconds"] += projection_seconds
        STATS["last"] = {
            "projected_elements": changed,
            "projection_seconds": projection_seconds,
            "numeric_setup_seconds": self.metadata["numeric_setup_seconds"],
        }


class ProjectedHybridHessian(HybridHessian):
    """Newton-search-only ``HybridHessian`` with PSD-projected FEM and IPC terms.

    Adjoint code keeps constructing plain ``HybridHessian``.  The projected
    IPC Hessian is placed in ``state.collision.hess`` only while the free CSR
    is refreshed, then the cache is cleared so exact consumers recompute it.
    """

    def __init__(self, model: Any, mode: Mode) -> None:
        super().__init__(model)
        self.mode = mode
        self.metadata["method"] = f"projected ({mode}) FEM+IPC free CSR search matrix"
        self.metadata["exact_physical_hessian"] = mode == "none"

    def prepare(self, state: Any) -> None:
        self._require_gpu_state(state)
        if self._valid_key == _state_key(state):
            self.metadata["prepare_cache_hits"] += 1
            return
        if self._fem is None:
            self._fem = ProjectedAssembledFemHvp(self._model, state, self.mode)
            self.metadata["fem_constructions"] += 1
        exact_contact = None
        if self._model.collision is not None:
            if state.collision is None:
                state.collision = self._model.collision.state_at(state.u)
            exact_contact = state.collision.hess
            state.collision.hess = None
        try:
            super().prepare(state)
        finally:
            if state.collision is not None:
                state.collision.hess = exact_contact

    def _ensure_contact_hessian(self, state: Any) -> None:
        collision = self._model.collision
        vertices = (collision.vertices + state.u[collision.indices]).numpy(force=True)
        state.collision.hess = collision.potential.hessian(
            collisions=state.collision.collisions,
            mesh=collision.collision_mesh,
            X=vertices,
            project_hessian_to_psd=_IPC_METHOD[self.mode],
        ).tocsr()


def projected_sparse_newton_problem(mode: Mode) -> type:
    """``SparseNewtonProblem`` whose search matrix is ``ProjectedHybridHessian``."""

    class ProjectedSparseNewtonProblem(hybrid_first_solver.SparseNewtonProblem):
        def __init__(self, delegate: Any) -> None:
            self.delegate = delegate
            self.model = delegate.model
            self.hessian = ProjectedHybridHessian(self.model, mode)

    return ProjectedSparseNewtonProblem


def install(
    mode: Mode, *, cell_batch: int = 16384, gpu_topology: bool = True
) -> dict[str, Any]:
    """Route only the hybrid solver's Newton search through the projected matrix.

    ``gpu_topology`` swaps in the bit-identical GPU sparsity build globally.
    Returns the live projection statistics.
    """
    STATS.update(mode=mode, cell_batch=cell_batch, gpu_topology=gpu_topology)
    if gpu_topology:
        assembled_fem_hvp._topology = _topology_gpu
    hybrid_first_solver.SparseNewtonProblem = projected_sparse_newton_problem(mode)
    return STATS


def _topology_gpu(model: Any, potentials: dict[str, Any]) -> tuple[Any, bool]:
    """GPU ``torch.unique`` replacement for ``assembled_fem_hvp._topology``.

    Produces bit-identical keys/CSR/slots (asserted against numpy in
    ``check_projection.py``); the numpy version spends ~5 s in unique and
    searchsorted on the 18M local block keys of this mesh.
    """
    model_id = id(model)
    cached = assembled_fem_hvp._TOPOLOGIES.get(model_id)
    if cached is not None and cached[0]() is model:
        return cached[1], True
    n_points = int(model.n_points)
    device = torch.device("cuda")
    raw = []
    for name, potential in potentials.items():
        cells = torch.as_tensor(assembled_fem_hvp._cpu_cells(potential), device=device)
        if cells.numel() and (int(cells.min()) < 0 or int(cells.max()) >= n_points):
            raise ValueError("potential cell indexes a point outside the model")
        vertices = cells.shape[1]
        local = cells.repeat_interleave(vertices, dim=1) * n_points + cells.repeat(
            1, vertices
        )
        raw.append((name, type(potential).__name__, cells, local))
    keys, inverse = torch.unique(
        torch.cat([item[3].reshape(-1) for item in raw]),
        sorted=True,
        return_inverse=True,
    )
    rows = keys // n_points
    counts = torch.bincount(rows, minlength=n_points)
    crow = torch.cat((counts.new_zeros(1), torch.cumsum(counts, 0)))
    potential_topologies = []
    offset = 0
    for name, kind, cells, local in raw:
        size = local.numel()
        slots = inverse[offset : offset + size].reshape(local.shape)
        offset += size
        potential_topologies.append(
            assembled_fem_hvp._PotentialTopology(
                cells.cpu().numpy(), slots.cpu().numpy(), cells.shape[1], name, kind
            )
        )
    result = assembled_fem_hvp._Topology(
        n_points,
        keys.cpu().numpy(),
        crow.cpu().numpy(),
        (keys % n_points).cpu().numpy(),
        tuple(potential_topologies),
    )

    def cleanup(_reference: weakref.ReferenceType[Any], *, key: int = model_id) -> None:
        assembled_fem_hvp._TOPOLOGIES.pop(key, None)

    assembled_fem_hvp._TOPOLOGIES[model_id] = (weakref.ref(model, cleanup), result)
    return result, False
