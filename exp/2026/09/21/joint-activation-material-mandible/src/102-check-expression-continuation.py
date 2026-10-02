# ruff: noqa: ANN001
"""CPU-only regression checks for sequential expression continuation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/expression-continuation-check-001"


def load_fitter_module():
    """Load the production fitter without constructing its GPU physics."""
    spec = importlib.util.spec_from_file_location(
        "expression_fitting", Path(__file__).with_name("93-fit-expressions.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_fitter(module, maximum_iterations_per_expression: int):
    """Make only the state that ``Fitter.run`` consumes."""
    fitter = module.Fitter.__new__(module.Fitter)
    fitter.cfg = SimpleNamespace(
        maximum_iterations_per_expression=maximum_iterations_per_expression,
        wall_cap_seconds=60.0,
    )
    fitter.names = ("A", "B")
    fitter.status = {
        "running": True,
        "inverse_converged": False,
        "expressions": {
            name: {"status": "queued", "accepted_steps": 0} for name in fitter.names
        },
    }
    fitter.runtime = SimpleNamespace(warm_adjoints={"A": object(), "B": object()})
    fitter.started = time.perf_counter()
    fitter.elapsed_offset = 0.0
    fitter.smooth_weight = 1.0
    phases = []

    def publish(phase: str, expression: str | None = None) -> None:
        fitter.status["phase"] = phase
        fitter.status["current_expression"] = expression
        phases.append((phase, expression))

    fitter.publish = publish
    return fitter, phases


def sequential_case(module) -> dict:
    fitter, phases = mock_fitter(module, maximum_iterations_per_expression=3)
    calls = []
    preparation = []
    fitter.refine_neutral = lambda: preparation.append("refine_neutral")
    fitter.calibrate = lambda: preparation.append("calibrate")

    def fit_step(index: int) -> None:
        name = fitter.names[index]
        calls.append(name)
        prior = fitter.status["expressions"][name]["accepted_steps"]
        terminal = name == "B" or prior + 1 == 2
        fitter.status["expressions"][name] = {
            "status": "converged" if terminal else "fitting",
            "accepted_steps": prior + 1,
        }

    fitter.fit_step = fit_step
    fitter.run()
    assert preparation == ["refine_neutral", "calibrate"]
    assert calls == ["A", "A", "B"], calls
    assert fitter.status["schedule"] == "sequential"
    assert fitter.status["phase"] == "converged"
    assert fitter.status["inverse_converged"]
    assert fitter.runtime.warm_adjoints == {}
    return {"calls": calls, "phases": phases, "status": fitter.status}


def line_search_failure_case(module) -> dict:
    fitter, phases = mock_fitter(module, maximum_iterations_per_expression=3)
    calls = []
    fitter.refine_neutral = lambda: None
    fitter.calibrate = lambda: None

    def fit_step(index: int) -> None:
        name = fitter.names[index]
        calls.append(name)
        fitter.status["expressions"][name] = {
            "status": "line_search_failed" if name == "A" else "converged",
            "accepted_steps": 0 if name == "A" else 1,
        }

    fitter.fit_step = fit_step
    fitter.run()
    assert calls == ["A", "B"], calls
    assert fitter.status["expressions"]["A"]["status"] == "line_search_failed"
    assert fitter.status["expressions"]["B"]["status"] == "converged"
    assert fitter.status["phase"] == "requires_attention"
    assert not fitter.status["inverse_converged"]
    return {"calls": calls, "phases": phases, "status": fitter.status}


def iteration_budget_case(module) -> dict:
    fitter, phases = mock_fitter(module, maximum_iterations_per_expression=2)
    calls = []
    fitter.refine_neutral = lambda: None
    fitter.calibrate = lambda: None

    def fit_step(index: int) -> None:
        name = fitter.names[index]
        calls.append(name)
        prior = fitter.status["expressions"][name]["accepted_steps"]
        fitter.status["expressions"][name] = {
            "status": "fitting",
            "accepted_steps": prior + 1,
        }

    fitter.fit_step = fit_step
    fitter.run()
    assert calls == ["A", "A"], calls
    assert fitter.status["phase"] == "expression_iteration_budget_reached"
    assert fitter.status["current_expression"] == "A"
    assert fitter.status["iteration_budget"] == {
        "expression": "A",
        "maximum_iterations_per_expression": 2,
        "accepted_steps": 2,
        "scope": "total accepted steps including continuation; no advance to next expression",
    }
    assert fitter.status["expressions"]["B"]["status"] == "queued"
    return {"calls": calls, "phases": phases, "status": fitter.status}


def continuation_budget_case(module) -> dict:
    """Count imported accepted steps before allowing another expression."""
    fitter, phases = mock_fitter(module, maximum_iterations_per_expression=3)
    fitter.status["expressions"]["A"] = {"status": "fitting", "accepted_steps": 2}
    calls = []
    fitter.refine_neutral = lambda: None
    fitter.calibrate = lambda: None

    def fit_step(index: int) -> None:
        name = fitter.names[index]
        calls.append(name)
        fitter.status["expressions"][name] = {
            "status": "fitting",
            "accepted_steps": 3,
        }

    fitter.fit_step = fit_step
    fitter.run()
    assert calls == ["A"], calls
    assert fitter.status["phase"] == "expression_iteration_budget_reached"
    assert fitter.status["iteration_budget"]["accepted_steps"] == 3
    assert fitter.status["expressions"]["B"]["status"] == "queued"
    return {"calls": calls, "phases": phases, "status": fitter.status}


def continuation_case(module, output_dir: Path) -> dict:
    """Exercise the actual importer on a tiny CPU checkpoint tree."""
    source = output_dir / "fixture-parent"
    target = output_dir / "fixture-continuation"
    source.mkdir()
    (source / "sources/experiment").mkdir(parents=True)
    runner = Path(module.__file__)
    archived_runner = source / "sources/experiment/93-fit-expressions.py"
    shutil.copy2(runner, archived_runner)
    runner_digest = sha256(archived_runner)
    parent_protocol = {
        "schema": "fixture-protocol-v1",
        "physics": {"fixed": "unchanged"},
        "implementation_sha256": {"93-fit-expressions.py": runner_digest},
        "optimizer": (
            "per-expression projected Adam; transactional Armijo; explicit "
            "non-descent momentum restart; round-robin all36"
        ),
        "schedule": {
            "expression_order": ["A", "B"],
            "advance_only_after": ["converged", "line_search_failed"],
        },
        "config": {
            "physics_setting": "fixed",
            "output_dir": str(source),
            "continue_from": None,
            "calibration_source": "old-receipt",
        },
    }
    current_protocol = json.loads(json.dumps(parent_protocol))
    current_protocol["optimizer"] = (
        "per-expression projected Adam with volume-metric projected-gradient descent safeguard; "
        "transactional Armijo; sequential until convergence or explicit failure"
    )
    current_protocol["config"].update(
        {
            "output_dir": str(target),
            "continue_from": str(source),
            "calibration_source": "new-receipt",
        }
    )
    write_json(source / "protocol.json", parent_protocol)
    source_status = {
        "running": False,
        "expressions": {
            "A": {"status": "fitting", "accepted_steps": 2},
            "B": {"status": "queued", "accepted_steps": 0},
        },
    }
    write_json(source / "status.json", source_status)
    write_json(
        source / "calibration.json",
        {
            "success": True,
            "expression_count": 2,
            "reference_neighbor_rms": 0.05,
            "probe_scale": 0.001,
            "strength_factor": 3.0,
        },
    )
    torch.save(
        {"displacement_m": torch.tensor([1.0])},
        source / "neutral-numerical-refinement.pt",
    )
    write_json(source / "neutral-numerical-refinement.json", {"fixture": True})
    directory = source / "expressions/A"
    directory.mkdir(parents=True)
    (source / "expressions/B").mkdir()
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(2.0),
                "exp_avg": torch.tensor([0.125]),
                "exp_avg_sq": torch.tensor([0.0625]),
            }
        },
        "param_groups": [
            {
                "lr": 0.003,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "params": [0],
            }
        ],
    }
    state = {
        "expression": "A",
        "activation": torch.tensor([[2.0]]),
        "jaw_normalized": torch.tensor([0.25]),
        "gradient_jaw": torch.tensor([0.5]),
        "accepted_steps": 2,
        "qualifying_consecutive": 1,
        "optimizer": optimizer,
        "metrics": {"objective": 1.5},
    }
    torch.save(state, directory / "latest.pt")
    (directory / "trace.jsonl").write_text('{"accepted_steps": 2}\n')
    (directory / "trials.jsonl").write_text('{"trial": 0}\n')
    before = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    target.mkdir()
    receipt = module.import_continuation(
        SimpleNamespace(
            continue_from=source,
            output_dir=target,
            neighbor_rms_budget=0.05,
            calibration_scale=0.001,
            calibration_strength=3.0,
        ),
        current_protocol,
    )
    after = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    imported = torch.load(
        target / "expressions/A/latest.pt", map_location="cpu", weights_only=False
    )
    assert before == after
    assert receipt["transition"].startswith("continue accepted states and Adam moments")
    assert sha256(target / "expressions/A/latest.pt") == sha256(directory / "latest.pt")
    assert torch.equal(imported["activation"], state["activation"])
    assert torch.equal(
        imported["optimizer"]["state"][0]["exp_avg"], optimizer["state"][0]["exp_avg"]
    )
    assert torch.equal(
        imported["optimizer"]["state"][0]["exp_avg_sq"],
        optimizer["state"][0]["exp_avg_sq"],
    )
    assert (target / "expressions/A/trace.jsonl").read_text() == (
        directory / "trace.jsonl"
    ).read_text()
    assert (target / "expressions/A/trials.jsonl").read_text() == (
        directory / "trials.jsonl"
    ).read_text()
    imported_status = json.loads((target / "status.json").read_text())
    assert imported_status["schedule"] == "sequential"
    assert imported_status["phase"] == "continuation_imported"
    assert imported_status["expressions"]["A"]["accepted_steps"] == 2
    return {
        "receipt": receipt,
        "parent_unchanged": before == after,
        "activation_preserved": True,
        "adam_moment_preserved": True,
        "expression_files_preserved": ["latest.pt", "trace.jsonl", "trials.jsonl"],
    }


def continuation_rejection_cases(module, output_dir: Path) -> dict:
    """Require a fully stopped parent and valid queued empty directories."""
    source = output_dir / "fixture-parent"
    current = json.loads((source / "protocol.json").read_text())
    current["optimizer"] = (
        "per-expression projected Adam with volume-metric projected-gradient descent safeguard; "
        "transactional Armijo; sequential until convergence or explicit failure"
    )
    incompatible = json.loads(json.dumps(current))
    incompatible["physics"]["fixed"] = "changed"
    try:
        module.compare_continuation_protocol(
            json.loads((source / "protocol.json").read_text()), incompatible
        )
    except AssertionError:
        pass
    else:
        message = "physical protocol mutation was accepted"
        raise AssertionError(message)

    def rejected(name: str, alter) -> None:
        parent = output_dir / f"fixture-rejected-{name}-parent"
        target = output_dir / f"fixture-rejected-{name}-target"
        shutil.copytree(source, parent)
        alter(parent)
        target.mkdir()
        config = SimpleNamespace(
            continue_from=parent,
            output_dir=target,
            neighbor_rms_budget=0.05,
            calibration_scale=0.001,
            calibration_strength=3.0,
        )
        try:
            module.import_continuation(config, current)
        except AssertionError:
            return
        message = f"{name} parent was accepted"
        raise AssertionError(message)

    def running(parent: Path) -> None:
        status = json.loads((parent / "status.json").read_text())
        status["running"] = True
        write_json(parent / "status.json", status)
        write_json(parent / "external-interruption.json", {"fixture": True})

    def invalid_empty_directory(parent: Path) -> None:
        status = json.loads((parent / "status.json").read_text())
        status["expressions"]["B"]["status"] = "fitting"
        write_json(parent / "status.json", status)

    rejected("running", running)
    rejected("nonqueued-empty-directory", invalid_empty_directory)
    return {
        "running_parent_rejected_even_with_interruption_receipt": True,
        "nonqueued_empty_expression_directory_rejected": True,
        "physical_protocol_mutation_rejected": True,
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    module = load_fitter_module()
    summary = {
        "schema": "joint-expression-continuation-check-v1",
        "success": True,
        "cpu_only": True,
        "scheduling": {
            "multiple_consecutive_before_next": sequential_case(module),
            "line_search_failure_advances": line_search_failure_case(module),
            "accepted_step_budget_stops_before_next": iteration_budget_case(module),
            "imported_steps_count_toward_budget": continuation_budget_case(module),
        },
        "continuation_import": continuation_case(module, cfg.output_dir),
        "continuation_rejections": continuation_rejection_cases(module, cfg.output_dir),
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
