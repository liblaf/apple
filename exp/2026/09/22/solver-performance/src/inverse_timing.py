# ruff: noqa: EM102, TRY003
"""Exception-safe hierarchical wall-clock instrumentation for inverse updates.

It wraps existing Python and IPC entry points; it never substitutes numerical
methods.  CUDA synchronization is optional because it changes timing scope and
can serialize otherwise asynchronous work.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TimingNode:
    count: int = 0
    inclusive_seconds: float = 0.0
    children_seconds: float = 0.0
    children: dict[str, TimingNode] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "inclusive_seconds": self.inclusive_seconds,
            "exclusive_seconds": self.inclusive_seconds - self.children_seconds,
            "children": {key: value.report() for key, value in self.children.items()},
        }


class InverseTimer:
    """Nested inclusive/exclusive wall time with explicit stage scopes."""

    def __init__(self, *, cuda_sync: bool = False) -> None:
        self.cuda_sync = cuda_sync
        self.root = TimingNode()
        self._stack: list[TimingNode] = [self.root]
        self._restores: list[tuple[Any, str, Any, bool]] = []
        self.missing: list[str] = []

    def _sync(self) -> None:
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def scope(self, name: str, *, sync: bool | None = None) -> Iterator[None]:
        parent = self._stack[-1]
        node = parent.children.setdefault(name, TimingNode())
        do_sync = self.cuda_sync if sync is None else sync
        if do_sync:
            self._sync()
        started = time.perf_counter()
        node.count += 1
        self._stack.append(node)
        try:
            yield
        finally:
            if do_sync:
                self._sync()
            elapsed = time.perf_counter() - started
            self._stack.pop()
            node.inclusive_seconds += elapsed
            parent.children_seconds += elapsed

    def stage(self, name: str) -> contextlib.AbstractContextManager[None]:
        """Tag caller-owned phases as ``forward/coarse_pncg`` or ``adjoint``."""
        return self.scope(name)

    def patch(
        self, owner: Any, attribute: str, label: str, *, sync: bool | None = None
    ) -> None:
        """Patch one existing callable and retain its exact descriptor to restore."""
        target = (
            owner
            if isinstance(owner, type) or hasattr(owner, "__dict__")
            else type(owner)
        )
        if not hasattr(target, attribute):
            self.missing.append(label)
            return
        original = getattr(target, attribute)
        if not callable(original):
            self.missing.append(label)
            return
        raw = vars(target).get(attribute) if hasattr(target, "__dict__") else None
        is_static = isinstance(raw, staticmethod)

        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.scope(label, sync=sync):
                return original(*args, **kwargs)

        try:
            setattr(target, attribute, staticmethod(timed) if is_static else timed)
        except (AttributeError, TypeError) as error:
            raise RuntimeError(f"cannot patch requested timing hook {label}") from error
        self._restores.append((target, attribute, raw, raw is None))

    def restore(self) -> None:
        while self._restores:
            owner, attribute, raw, inherited = self._restores.pop()
            if inherited:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, raw)

    def report(self) -> dict[str, Any]:
        return {
            "schema": "inverse-hierarchical-timing-v1",
            "clock": "perf_counter wall time",
            "cuda_sync": self.cuda_sync,
            "cuda_sync_limitation": (
                "synchronizes each scoped boundary and therefore measures completed CUDA work but perturbs overlap"
                if self.cuda_sync
                else "asynchronous CUDA work can be charged to a later scope; CPU wall times remain unsynchronized"
            ),
            "tree": self.root.report()["children"],
            "missing_hooks": self.missing,
        }


class InstalledInverseTiming:
    def __init__(self, timer: InverseTimer) -> None:
        self.timer = timer
        self._restores: list[tuple[Any, str, Any]] = []
        self.missing: list[str] = []

    def wrap(
        self, owner: Any, attribute: str, path: str, *, sync: bool | None = None
    ) -> None:
        before = len(self.timer.missing)
        self.timer.patch(owner, attribute, path, sync=sync)
        if len(self.timer.missing) > before:
            self.missing.append(path)

    def uninstall(self) -> None:
        self.timer.restore()

    def report(self) -> dict[str, Any]:
        result = self.timer.report()
        result["missing_hooks"] = self.missing
        return result


def install_inverse_timing(
    model: Any, *, cuda_sync: bool = False, native_ipc: bool = True
) -> InstalledInverseTiming:
    """Instrument a model/collision tree and optional patchable IPC methods.

    Missing Python methods are reported explicitly. Native IPC class patches are
    process-global, so install only around one sequential update and uninstall
    before another task starts.
    """
    installed = InstalledInverseTiming(InverseTimer(cuda_sync=cuda_sync))
    model_type = type(model)
    for method in (
        "fun",
        "grad",
        "hess_diag",
        "hess_prod",
        "hess_quad",
        "mixed_derivative_prod",
        "max_step_size",
        "update",
    ):
        installed.wrap(model_type, method, f"model/{method}")
    warp_model = getattr(model, "warp_model", None)
    if warp_model is None:
        installed.missing.append("warp_model")
    else:
        for method in ("fun", "grad", "hess_diag", "hess_prod"):
            installed.wrap(type(warp_model), method, f"warp_model/{method}")
    collision = getattr(model, "collision", None)
    if collision is None:
        installed.missing.append("collision")
    else:
        for method in (
            "state_at",
            "update",
            "fun",
            "grad",
            "hess_diag",
            "hess_prod",
            "max_step_size",
            "diagnostics",
        ):
            installed.wrap(type(collision), method, f"collision/{method}")
    if native_ipc:
        import ipctk

        for cls, method, path in (
            (ipctk.Candidates, "build", "ipc/candidates_build"),
            (ipctk.NormalCollisions, "build", "ipc/normal_collisions_build"),
            (ipctk.BarrierPotential, "hessian", "ipc/barrier_hessian"),
            (ipctk.Candidates, "compute_collision_free_stepsize", "ipc/ccd"),
        ):
            installed.wrap(cls, method, path, sync=False)
    return installed
