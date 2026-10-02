# Chunked tensor-control projection validation

## Purpose

The first three-step face pilot requested approximately 145.75 GiB of CUDA eigensolver workspace while projecting all 288,235 active-cell tensors at once. The control implementation was changed to process at most 1,024 matrices per eigensolver call while retaining the full `(288235, 6)` control tensor.

This audit checks that the chunked implementation in [`src/tensor_controls.py`](../src/tensor_controls.py) is mathematically identical to the original unbatched projection on small inputs, remains correct at the complete face control count, and avoids the oversized allocation. It performs no face equilibrium or inverse step.

The six coordinates are orthonormal for the Frobenius inner product:

```text
(Qxx, Qyy, Qzz, sqrt(2) Qxy, sqrt(2) Qyz, sqrt(2) Qxz)
```

Each symmetric tensor is projected onto (0\preceq Q\preceq Q_{\max}I) by eigenvalue clipping. The test cap is `0.030201342281879196 MPa`.

## Recorded command

The report-worthy run used the explicit noncommitting Cherries profile:

```bash
CHERRIES_NAME='Tensor controls chunked CUDA validation' \
CHERRIES_TAGS='tensor-active-stress,controls,projection,cuda,memory-audit' \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
.venv/bin/python \
  src/10b-validate-tensor-controls.py
```

The process exited with status 0. Cherries recorded Git commit `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` without committing or staging files. The tested `tensor_controls.py` SHA-256 is `fef2b003da6316d8d6b8f36c7a09724a05c1c954493e9377d5e7564d30e7f45e`. [Comet experiment](https://www.comet.com/liblaf/apple/d46d979de2574a098431503e6373f58b)

## Results

On 4,097 deterministic random tensors, which force five chunks, the chunked projection and eigenvalue output matched a single unbatched call exactly on both CPU and CUDA. The projection statistics for the clipped lower and upper eigenvalues also matched exactly. A second projection changed coordinates by at most `6.59e-17 MPa` on CPU and `6.25e-17 MPa` on CUDA, consistent with eigendecomposition roundoff.

The face-sized CUDA test used exactly 288,235 rows and six float64 coordinates per row, split into 282 chunks.

| Face-sized CUDA check | Result |
| --- | ---: |
| Finite projected coordinates | yes |
| Finite projected eigenvalues | yes |
| Minimum output eigenvalue | `-1.90e-17 MPa` |
| Maximum output eigenvalue | `0.030201342281879272 MPa` |
| Sample vs unbatched CPU projection | `2.78e-17 MPa` max error |
| Lower-clipping fraction error | `0` |
| Upper-clipping fraction error | `0` |
| Peak CUDA memory allocated | `584.59 MiB` |
| Peak CUDA memory reserved | `2130 MiB` |

The tiny negative minimum and cap excess are approximately (10^{-16}) MPa and arise from reconstruction/eigendecomposition roundoff. Both are within the declared (10^{-13}) MPa bound check. The measured allocated memory is below the audit's 768 MiB ceiling. CUDA's caching allocator reserved 2.13 GiB during the complete report process, still far below the rejected 145.75 GiB workspace request.

## Evidence

- [Complete summary](../data/10b-tensor-controls-validation/summary.json)
- [Configuration](../data/10b-tensor-controls-validation/config.json)
- [Source hashes and environment receipt](../data/10b-tensor-controls-validation/provenance.json)
- [Recorded run log](../logs/10b-validate-tensor-controls.log)

## Limits

This result validates the projection and eigenvalue helpers in an otherwise empty CUDA process. A face solve has additional resident mesh, solver, and adjoint allocations, so its total peak memory will be higher. The audit establishes that these helpers no longer request workspace proportional to one unbatched 288,235-matrix eigendecomposition; it does not establish face forward convergence or inverse progress.
