"""Real force and directional-gradient validation of resolved-work PNCG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_expression_precise import install_precise_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/precise-expression-validation-001"


def main(cfg: Config):
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    n = FrozenNeutral.load(GROUP / "data/frozen-neutral-004")
    p, _ = build_eye_collision_physics(n, GROUP / "data/rigid-eyes-001")
    r = install_precise_expression_runtime(p)
    r.tolerances["atol"] = 1e-12
    s = json.loads((GROUP / "data/eye-neutral-forward-002/summary.json").read_text())
    q = Path(s["checkpoint"]["path"])
    assert sha256(q) == s["checkpoint"]["sha256"]
    with np.load(q) as a:
        u = torch.as_tensor(np.asarray(a["displacement_m"], dtype=np.float64))
    z = torch.zeros(6)
    k = len(p.base.active_t)
    a = (torch.eye(3).expand(k, -1, -1).clone() * 2e-6).requires_grad_()
    pose = torch.tensor([1e-7, -1e-7, 1e-7, 1e-7, 0.0, -1e-7]).requires_grad_()
    d = torch.eye(3).expand(k, -1, -1).clone()
    dp = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.5])
    dp /= torch.linalg.vector_norm(dp)
    ids = torch.as_tensor(n.arrays["observation_node_ids"])
    v = torch.tensor([0.31, -0.27, 0.19])

    def f(x):
        return 1e6 * torch.mean(x[ids] @ v)

    x = p.solve(torch.ones(()), a, pose, u, seed_pose=z, key="precise-base")
    f(x).backward()
    pa = float((a.grad * d).sum())
    pp = float((pose.grad * dp).sum())

    def e(A, P, key):
        return float(
            f(
                p.solve(
                    torch.ones(()), A, P, x.detach(), seed_pose=pose.detach(), key=key
                )
            )
        )

    h = 1e-6
    fa = (
        e(a.detach() + h * d, pose.detach(), "a+")
        - e(a.detach() - h * d, pose.detach(), "a-")
    ) / (2 * h)
    fp = (
        e(a.detach(), pose.detach() + h * dp, "p+")
        - e(a.detach(), pose.detach() - h * dp, "p-")
    ) / (2 * h)
    er = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-12)
    out = {
        "success": er(pa, fa) < 0.05 and er(pp, fp) < 0.05,
        "force": r.last_forward,
        "activation": {"predicted": pa, "fd": fa, "error": er(pa, fa)},
        "pose": {"predicted": pp, "fd": fp, "error": er(pp, fp)},
        "hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(
                Path(__file__).with_name("joint_expression_precise.py").resolve()
            ): sha256(
                Path(__file__).with_name("joint_expression_precise.py").resolve()
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", out)
    cherries.log_output(cfg.output_dir)
    assert out["success"], out


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
