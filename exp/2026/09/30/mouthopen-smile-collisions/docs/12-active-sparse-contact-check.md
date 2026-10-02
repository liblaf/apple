# Active-strain sparse contact preflight

## Local integration

[`active_assembled_fem.py`](../src/active_assembled_fem.py) provides an
experiment-local `ActiveAssembledFemHvp`. It reuses the core generic
tetrahedral `_bulk_kernel(potential.hess_prod_func)` for `StableNeoHookean` and
`StableNeoHookeanActive`, while retaining the existing topology, BSR storage,
and `GpuFreeSparseHessian` merge implementation.

`ActiveSparseHessianProblem` subclasses `HessianProblem` and overrides only
`_prepare`: it installs `ActiveAssembledFemHvp` before invoking the unchanged
free-coordinate FEM-plus-IPC sparse merger. The caller must precompute the
IPC contact Hessian with `PSDProjectionMethod.CLAMP` before asking for an HVP.

## GPU preflight

The requested full-mesh validation was deferred while a separate job owned
most GPU memory. This run did not allocate the full assembled matrix.

The 1,144,268-tetrahedron fixture has an upper bound of 18,308,288 FEM BSR
blocks. The BSR values and structure alone can require 1,466,486,248 bytes;
the free scalar CSR values and column indices have a 2,641,232,456-byte upper
bound. Temporary GPU union-plan tensors add to those figures. With about 5 GiB
available, the full benchmark was not safe to start.

The Cherries preflight used `Git(commit=False)` and completed at
[e20d48bd6ee9479ab3c54b2912d71c8d](https://www.comet.com/liblaf/apple/e20d48bd6ee9479ab3c54b2912d71c8d).
Its receipt is
[`data/12-active-sparse-contact-check/summary.json`](../data/12-active-sparse-contact-check/summary.json).

No HVP equivalence, timing, or speed claim is made until the two-state,
three-direction GPU validation can run with adequate free memory.
