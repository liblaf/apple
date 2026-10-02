# ruff: noqa: EM101, PT017, PT018, TRY003
"""Check that the optional raw-Armijo prescreen skips only impossible trials."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fit_expressions", ROOT / "93-fit-expressions.py"
)
assert spec is not None and spec.loader is not None
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)


def fake_fitter() -> SimpleNamespace:
    graph = fit.VolumeGraph(
        i=torch.tensor([0]),
        j=torch.tensor([1]),
        conductance_m=torch.tensor([1.0]),
        effective_cell_volume_m3=torch.tensor([0.5, 0.5]),
        smooth_length_m=0.005,
    )
    instance = SimpleNamespace(
        names=["Smile"],
        graph=graph,
        smooth_weight=1.0,
        cfg=SimpleNamespace(magnitude_weight=0.001, jaw_weight=0.01),
        solve_calls=0,
    )

    def solve(*_args: object) -> torch.Tensor:
        instance.solve_calls += 1
        return torch.zeros((1, 3))

    instance.solve = solve
    instance.data_loss = lambda _u, _index: (torch.ones(()), torch.ones(()))
    return instance


def main() -> None:
    instance = fake_fitter()
    q = torch.zeros((2, 6), requires_grad=True)
    q.data[0, 0] = 1.0
    jaw = torch.zeros(1, requires_grad=True)
    # The exact non-negative smoothness prior exceeds this ceiling. The
    # equilibrium callback is consequently never reached.
    try:
        fit.Fitter.evaluate(
            instance,
            0,
            q,
            jaw,
            torch.zeros((1, 3)),
            torch.zeros(1),
            objective_ceiling=1e-5,
        )
    except fit.TrialObjectiveRejectedError as error:
        assert error.receipt["stage"] == "before_forward"
        assert error.receipt["value"] > error.receipt["ceiling"]
    else:
        raise AssertionError("impossible raw-Armijo trial was not rejected")
    assert instance.solve_calls == 0

    # A raw objective above the ceiling still retains its forward solve, but
    # stops before autograd and the implicit adjoint.
    q.data.zero_()
    try:
        fit.Fitter.evaluate(
            instance,
            0,
            q,
            jaw,
            torch.zeros((1, 3)),
            torch.zeros(1),
            objective_ceiling=0.5,
        )
    except fit.TrialObjectiveRejectedError as error:
        assert error.receipt["stage"] == "before_adjoint"
    else:
        raise AssertionError("raw-uphill trial was not rejected before adjoint")
    assert instance.solve_calls == 1

    # This is the mathematical condition used by the prescreen: all objective
    # terms are non-negative, so prior > ceiling implies objective > ceiling.
    for prior, data, ceiling in ((0.2, 0.0, 0.1), (0.2, 3.0, 0.1), (0.1, 0.0, 0.1)):
        prescreen = prior > ceiling
        impossible = prior + data > ceiling
        if prescreen:
            assert impossible
        if prior == ceiling and data == 0:
            assert not prescreen
    print(
        "trial prescreen: prior lower bound rejects without solve; boundary preserved"
    )


if __name__ == "__main__":
    main()
