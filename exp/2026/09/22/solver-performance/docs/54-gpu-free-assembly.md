# GPU free-space Hessian assembly

Across the two frozen states, cached GPU numeric refresh is 13.0-13.3x faster than CPU SciPy free-CSR rebuild; refresh plus PCG is 3.12-3.80x faster. Maximum matrix and lower-storage relative errors are 3.65e-16 and 3.75e-16.

This frozen-state benchmark compares the previous CPU SciPy free-CSR rebuild with a GPU-resident numeric assembly. Both use the same exact FEM plus IPC Hessian restricted to free DOFs. The GPU path caches static FEM/contact index maps and updates numeric values on GPU; collision Hessian evaluation remains CPU-owned, then its exact sparse contribution is remapped and uploaded. It measures matrix preparation and one PCG solve, not a complete forward or inverse solve.

## Measured cold-to-saved sequence

| State | Old refresh median s | Old PCG median s | Old refresh + solve s | GPU refresh median s | GPU PCG median s | GPU refresh + solve s | GPU minus old s | GPU first setup s | Cold startup-excess amortization cycles | Matrix error | Lower error | GPU persistent MiB | SPD status | SPD residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold_loaded_neutral | 3.1 | 0.788 | 3.89 | 0.239 | 0.782 | 1.02 | -2.86 | 1.87 | 0.568 | 3.65e-16 | 3.75e-16 | 1211.1 | success | 2.56e-13 |
| saved_smile | 3.11 | 1.14 | 4.25 | 0.234 | 1.13 | 1.36 | -2.89 | 0.28 | — | 3.14e-16 | 3.63e-16 | 1211.0 | success | 3.73e-13 |

The sequence is chronological: loaded neutral is constructed first, then saved Smile refreshes the cached representation. The primary comparison is median repeated numeric refresh plus median PCG. `GPU first setup s` separately retains the initial constructor or first changed-state setup before those repeats. Each arm has three refresh and three PCG samples, run sequentially on one GPU.

## What the totals include

`Old refresh + solve s` is the old CPU CSR rebuild/upload median refresh plus median PCG. `GPU refresh + solve s` is the cached GPU median refresh plus median PCG. The first setup is not hidden: it is retained separately. The cold startup-excess amortization uses only initial GPU setup minus its warm numeric refresh, divided by a positive same-state refresh-plus-solve saving. The second panel does not combine inclusive sub-timings. Contact-state construction and one-time process startup remain separate.

The cold GPU-route constructor is 1.87 s; its excess over the warm refresh is 1.63 s. Both arms also share 26.38 s of initial FEM topology/assembly preparation, excluded from these free-CSR timings. Persistent buffers are 562.9 MiB for CPU-assembled free CSR and 1211.1 MiB for the GPU route, an incremental 648.2 MiB. The GPU route stores values and cached static index maps; IPC Hessian evaluation remains CPU-owned, while its sparse contact contribution is remapped and uploaded for the GPU numeric assembly.

The earlier [cuDSS comparison](cudss.html) used the CPU-built combined free CSR. The [Hessian representation comparison](hessian-representations.html) used split FEM/contact sparse operators. This run repeats the CPU baseline on the same GPU; it does not infer a full-method speed-up by combining timings from different runs.

## Accuracy, lower storage, and direct validation

Matrices and diagonal-inclusive lower storage are independently compared with the original matrix-free FEM plus CPU IPC operator. The SPD receipt is an additional direct lower-storage residual check. These independent checks validate the frozen linear operator; they do not validate a full forward trajectory.

Return to cold took 0.262 s. Contact remap took 0.00973 s; contact pattern changed=True, union reused=True, symbolic cache hit=True. Matrix/lower errors were 3.65e-16/3.75e-16; pattern hash `cd65e0563589797c2038cff73261d40430322cbd3dfaf86b5c4b122432b3d92b`.

Symbolic analysis may be reused only when the recorded matrix pattern and descriptor agree. Contact entries inside the cached structure only need new scatter destinations; off-pattern entries rebuild the union. The sparse union, sorting, and large index maps are constructed on CUDA; initial FEM/free-DOF mapping and small contact remapping still use CPU. The unchanged material-model FEM topology preparation is a separate shared startup cost. The GPU assembler is an experiment-local backend; the production forward/inverse defaults have not been changed.

## Reproduction

The receipt records source hashes and archives benchmark sources before execution; it also checks checkpoint and input hashes against the established baseline. Run from this experiment directory with local Cherries logging:

```bash
DEBUG=1 CUDSS_LIBRARY="$PWD/tmp/cudss-runtime/nvidia/cu13/lib/libcudss.so.0" CHERRIES_NAME="GPU free Hessian assembly" CHERRIES_TAGS="solver-performance,gpu-assembly,smile,rtx4090" ${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python -u src/53-benchmark-gpu-free-assembly.py --output-dir data/gpu-free-assembly-002
```

The recorded output path was `data/gpu-free-assembly-002`. Use a new output directory for another run; the harness refuses to overwrite receipts. `DEBUG=1` keeps Cherries local. The cuDSS path refers to the previously staged 0.7 runtime. Cherries completed both states; raw terminal output and source-copy verification are retained under `tmp/`.

## Evidence

Source summary: `exp/2026/09/22/solver-performance/data/gpu-free-assembly-002/summary.json` (SHA-256 `532cd42ea053091dfe8fa5b91009322e288a88a88f5e233037bf2e49b8e1aad1`). Assets: `gpu-assembly.png`, `gpu-assembly.svg`, and [gpu-assembly-evidence.json](gpu-assembly-evidence.json). Results are sequential single-GPU measurements; medians describe the recorded samples and do not estimate concurrent throughput.
