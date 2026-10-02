# CUDA 12 environment

This standalone uv project installs Apple from the working tree into
`environments/cuda12/.venv`. It has its own lockfile and preserves the root
CUDA 13 development environment. Use Linux x86-64, Python 3.14, and uv 0.12.17
or newer.

## Install and verify

Run from the Apple repository root:

```bash
uv sync --project environments/cuda12 --frozen
environments/cuda12/.venv/bin/python environments/cuda12/verify.py
environments/cuda12/.venv/bin/python -m pytest -o addopts='' \
  --import-mode=importlib --doctest-modules tests src -q
```

The verification script requires a GPU and fails if a runtime or numerical
check fails. It checks Torch's V100 architecture support (`sm_70`), FP64 sparse
matrix multiplication, a CuPy FP64 solve, Warp's runtime, JAX GPU execution,
and Apple's active stable Neo-Hookean energy, gradient, and Hessian product.
The Apple kernel checks compare CPU and GPU results and finite differences.

For comparison with the root environment:

```bash
.venv/bin/python environments/cuda12/verify.py --allow-cuda13
```

## Package selection

| Component | Locked selection |
| --- | --- |
| PyTorch | `2.12.0+cu126`, official CUDA 12.6 wheel index |
| Warp | `1.14.0+cu12`, NVIDIA release wheel built with CUDA 12.9 |
| CuPy | `cupy-cuda12x==14.1.1` |
| JAX | `jax[cuda12]==0.10.1` |
| Python | 3.14 |

PyTorch's CUDA 12.6 wheel includes `sm_70`; the existing CUDA 13 wheel does
not. The CUDA minor versions above intentionally differ: each package uses
its supplied runtime. CUDA 13 packages and Peach are absent from this
environment.

The NVIDIA driver must support the selected runtime. Warp's CUDA 12 packages
require driver 525 or newer; JAX also documents a Linux CUDA 12 minimum of
525. Run the verification on the rented server to check its complete driver
and library combination. See [Warp 1.14 installation requirements](https://github.com/NVIDIA/warp/blob/v1.14.0/docs/user_guide/installation.rst)
and [JAX installation requirements](https://docs.jax.dev/en/latest/installation.html).

Use the root CUDA 13 environment for RTX 5090. This CUDA 12.6 Torch build
does not contain its `sm_120` architecture.

## Run experiments

Use this environment's interpreter explicitly. For example, from the
repository root:

```bash
DEBUG=1 CHERRIES_NAME="CUDA 12 CLI check" CHERRIES_TAGS="cuda12,smoke" \
  environments/cuda12/.venv/bin/python \
  exp/2026/09/21/joint-activation-material-mandible/src/93-fit-expressions.py --help
```

Keep the experiment's normal input paths, working directory, `CHERRIES_NAME`,
and `CHERRIES_TAGS` settings. When changing to an experiment directory, use an
absolute path to this interpreter. Plain `uv run` at the repository root
selects the root environment instead.
The help command above uses `DEBUG=1` to disable remote Comet recording.

Apple now owns the PNCG and CG/MINRES solver implementation under
`liblaf.apple.solvers`, copied from Peach 0.10.1 with its numerical behavior
preserved. See [solver provenance](../../src/liblaf/apple/solvers/NOTICE.md).
Core forward and inverse code and the current joint/solver-performance
experiments use these modules. Older archived experiments that use obsolete
Peach APIs require their historical environments.

For package consumers, Apple's `cuda12` and `cuda13` extras select the CuPy
distribution only; they do not select a Torch or Warp wheel. Use this complete
environment for the tested CUDA 12 combination. The root dev and test groups
select CUDA 13 and conflict with the `cuda12` extra.

## Validation scope

Validated locally on 2026-09-22 with an RTX 4090 and Python 3.14.6:

- GPU runtime verification passed for both the separate CUDA 12 environment
  and the existing CUDA 13 environment.
- Both environments passed all 36 tests and doctests under `tests` and `src`,
  including forward, implicit adjoint, material derivative, CG, and
  symmetric-indefinite MINRES checks.
- The copied PNCG implementation matched Peach on a deterministic quadratic.
- The CUDA 12 environment passes `uv pip check` with Peach uninstalled.

On 2026-09-22, the runtime verification also passed on a
Tesla V100-PCIE-32GB. A replay of the full-face FP64 frozen-system benchmark
completed all 24 PCG solves and operator checks. Its archived RTX 4090
reference was 1.59–1.87 times faster for PCG and 1.72–2.05 times faster for
warmed Hessian-vector products. This comparison includes host and CUDA
library differences; it does not measure total inverse-fitting time.

A complete inverse experiment and an AutoDL-hosted V100 remain untested.
