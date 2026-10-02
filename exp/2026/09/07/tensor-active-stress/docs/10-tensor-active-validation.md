# Fiber-free tensor active-stress material validation

## Purpose

This experiment validates a standalone Warp FEM material that replaces the prescribed fiber direction and scalar tension with one cellwise symmetric tensor (Q):

$$
\Psi(F,Q)=\Psi_{\mathrm{stable}}(F;\mu,\lambda_{\mathrm{code}})
+\frac12 Q:(F^T F-I),
\qquad
P(F,Q)=P_{\mathrm{stable}}(F)+FQ.
$$

The implementation keeps the caller's passive Lamé parameters unchanged. It contains no fiber field and no `activation_inv` field. The caller is responsible for mapping its six controls to a symmetric positive-semidefinite, norm-capped (Q). The material deliberately does not conceal a broken projection by repairing (Q) internally.

The callable module is [`src/tensor_active.py`](../src/tensor_active.py). Its public construction and update contract is:

```python
from tensor_active import ACTIVE_STRESS, StableNeoHookeanTensorActive

potential = StableNeoHookeanTensorActive.from_pyvista(mesh, name="muscle")
model.set_materials({
    "muscle": {
        ACTIVE_STRESS: q_tensor,  # shape: (cells, 3, 3)
    }
})
```

`from_pyvista` creates zero tensors when the caller has not supplied them. The complete material-field set is `dhdX`, `dV`, `active_stress`, `lmbda`, and `mu`.

## Exact tangent

For a displacement direction with deformation-gradient increment \(\delta F\), the active tangent is

$$
\delta P_{\mathrm{act}}=\delta F Q,
\qquad
\delta^2\Psi_{\mathrm{act}}=delta F:(\delta FQ).
$$

The implementation supplies this term in all three solver paths: assembled Hessian product, Hessian quadratic, and nodal Hessian diagonal. For nodal reference gradient \(g_a\), the three Cartesian diagonal entries are all \(g_a^TQg_a\). A PSD (Q) therefore contributes a nonnegative material tangent.

## Recorded command

The report-worthy run was launched from this experiment directory with the repository interpreter and the explicit noncommitting profile:

```bash
CHERRIES_NAME='Tensor active stress constitutive validation' \
CHERRIES_TAGS='tensor-active-stress,warp,derivative-audit,fiber-free' \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
.venv/bin/python \
  src/10-validate-tensor-active.py
```

The run exited with status 0. Cherries used Git commit `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` without committing or staging files. The source snapshot records `tensor_active.py` with SHA-256 `0ab41c1b7ed129e0fd81bfab251d1ac5c6c5de6115861b5e2a21bb857cd841ef`. [Comet experiment](https://www.comet.com/liblaf/apple/e6c23ec381c147f69e0c640d31d42bf8)

## Results

The CPU audit used an actual one-tetrahedron `WarpPotentialFem` assembly at a nontrivial, nonsymmetric deformation gradient and a symmetric positive-definite (Q). Its eigenvalues were 0.005122, 0.009918, and 0.019960 MPa.

| Check | Absolute error | Declared limit |
| --- | ---: | ---: |
| Constitutive energy density vs analytic formula | `2.60e-18` MPa | `1e-14` |
| First Piola stress vs analytic formula | `0` MPa | `1e-14` |
| Assembled energy directional derivative | `4.86e-15` MPa·m³ | `1e-10` |
| Assembled Hessian product vs force finite difference | `3.56e-13` MPa·m | `1e-10` |
| Nodal Hessian diagonal vs force finite difference | `2.94e-13` MPa·m | `1e-10` |
| Hessian quadratic vs Hessian-product contraction | `0` MPa·m | `1e-12` |
| Hessian quadratic vs energy second difference | `6.05e-9` MPa·m | `1e-8` |
| Warp material-adjoint derivative vs analytic derivative | `7.59e-19` | `1e-14` |
| Combined spatial/material rotation covariance, energy | `2.82e-18` MPa | `1e-14` |
| Combined spatial/material rotation covariance, Piola stress | `3.47e-18` MPa | `1e-14` |

The energy second-difference error is the largest recorded error and remains below its finite-difference tolerance. The force-based Hessian product and diagonal comparisons avoid second-difference cancellation and agree at approximately (10^{-13}).

The material-adjoint test passed a Torch tensor with shape `(1, 3, 3)`, recorded the assembled force on a Warp tape, propagated one nodal adjoint direction to `active_stress`, and compared the resulting directional derivative with both an analytic expression and a finite difference. The tested material perturbation was symmetric, matching the caller contract.

For (Q=Tff^T), the new material reduces algebraically to the prior rank-one tension law. The actual Warp comparison found energy, Piola-stress, and Hessian-quadratic errors of `9.76e-19`, `4.34e-19`, and `5.42e-20` MPa, respectively.

With (Q=0), the new and unmodified `StableNeoHookean` implementations agreed exactly in energy density, integrated energy, Piola stress, assembled force, Hessian diagonal, Hessian product, and Hessian quadratic. This establishes the actuator-off passive response on the audited state.

The CUDA smoke test compiled the energy, Piola, and force kernels on an NVIDIA GeForce RTX 4090. Its energy density matched the CPU value exactly in float64.

## Evidence

- [Complete summary](../data/10-tensor-active-validation/summary.json)
- [CPU constitutive, assembly, adjoint, covariance, and rank-one audit](../data/10-tensor-active-validation/cpu-audit.json)
- [CUDA smoke receipt](../data/10-tensor-active-validation/cuda-smoke.json)
- [Machine-readable check table](../data/10-tensor-active-validation/checks.csv)
- [Source hashes and noncommitting environment receipt](../data/10-tensor-active-validation/provenance.json)
- [Recorded run log](../logs/10-validate-tensor-active.log)

## Limits

The experiment validates the constitutive implementation and its one-cell FEM assembly. It does not show that a face-scale optimizer preserves symmetry, positive semidefiniteness, or the requested cap; those are contracts of the external control map. It does not establish target-strain reachability, physiological calibration, or convergence of a face equilibrium. Because (Q) is a full tensor material field, a caller that passes a nonsymmetric matrix violates the energy/stress derivative contract rather than receiving an internal fallback.
