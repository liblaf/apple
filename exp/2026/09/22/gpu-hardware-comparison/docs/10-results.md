# V100 host V100 PCIe versus the archived RTX 4090 benchmark

The archived RTX 4090 was **1.59–1.87 times faster for the measured PCG solves** and **1.72–2.05 times faster for warmed Hessian-vector products** than V100 host's Tesla V100-PCIE-32GB. All 24 V100 PCG solves and all operator-equivalence checks passed. This result does not support switching this workload from the measured 4090 configuration to this V100 configuration for speed. CUDA 12 execution on V100 is now verified.

The existing local `expression-fitting-008` job was left running. No new 4090 measurement was taken: the comparison uses the requested existing `hessian-representations-001` record from the Paratera 4090.

![Comparison of measured HVP and linear-solve times](../data/comparison-001/comparison.png)

## Measured results

Times below are medians. Each state and implementation has 30 warmed HVP measurements and three PCG repetitions. “4090 advantage” is V100 time divided by 4090 time; greater than one means the 4090 was faster. Compilation and first construction are excluded from these steady-state timings.

| State / implementation | 4090 HVP (ms) | V100 HVP (ms) | 4090 HVP advantage | 4090 PCG (s) | V100 PCG (s) | 4090 PCG advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold neutral: Matrix-free FEM + CPU contact | 3.681 | 6.317 | 1.72× | 4.573 | 7.292 | 1.59× |
| Cold neutral: Matrix-free FEM + GPU contact | 2.683 | 5.082 | 1.89× | 3.295 | 5.925 | 1.80× |
| Cold neutral: Cached FEM + GPU contact | 2.521 | 4.709 | 1.87× | 2.938 | 5.496 | 1.87× |
| Cold neutral: Assembled FEM + GPU contact | 0.860 | 1.717 | 2.00× | 1.240 | 2.201 | 1.78× |
| Saved Smile: Matrix-free FEM + CPU contact | 3.665 | 6.306 | 1.72× | 6.466 | 10.565 | 1.63× |
| Saved Smile: Matrix-free FEM + GPU contact | 2.685 | 5.039 | 1.88× | 4.724 | 8.517 | 1.80× |
| Saved Smile: Cached FEM + GPU contact | 2.360 | 4.670 | 1.98× | 4.229 | 7.913 | 1.87× |
| Saved Smile: Assembled FEM + GPU contact | 0.823 | 1.687 | 2.05× | 1.771 | 3.192 | 1.80× |

Assembled FEM remains the fastest tested implementation on both systems. Including its per-state numerical matrix refresh, V100 takes **2.564 s** for cold neutral and **3.553 s** for saved Smile; the 4090 record takes **1.468 s** and **1.997 s**. On V100 this is 2.31× and 2.40× faster than its matrix-free FEM route with GPU contact. First construction is a separate cost and is not included in these sums.

Iteration counts are close: V100 requires 1084–1113 steps in the cold state and 1574–1596 in saved Smile, versus 1077–1120 and 1582–1601 on the 4090. The timing difference is therefore not explained by substantially more solver iterations on V100. The machine-readable comparison also contains time per HVP call within PCG and all sample ranges.

## Matched protocol and validation

The workload is the full facial finite-element model with tissue, skin, active stress, fixed neutral prestress, cranium, mandible, and eyeball contact: **257,009 points and 597,177 free degrees of freedom**, using FP64. It replays two frozen physical states. This is a Newton linear-system benchmark; it does not run an entire nonlinear forward solve, an implicit adjoint, or inverse fitting.

The replay preserves checkpoint and input hashes, the archived physical sources, initial displacement hashes, seed `20260922`, eight IPC threads, scalar diagonal preconditioning, zero initial PCG guess, and relative tolerance `1e-3`. It uses the exact shifts selected in the 4090 record: `0` for cold neutral and `2.8459193252829196e-7` for saved Smile. Those shifts are fixed during the replay; they are not increased to make the V100 result succeed. Both machines pass the same true-residual acceptance threshold of `1.05e-3`.

The V100 maximum checked HVP relative L2 error against its original CPU-contact operator is **4.002e-16**. Its worst PCG true relative residual is **9.9985e-4**. Both states completed successfully, with three PCG repetitions for each of four implementations.

The saved Smile force norm is about `3.6694e-9`, already below the `1e-8` forward force target. Its shifted PCG solve is an artificial residual probe, not an iteration required by a converged forward solve.

The displacement hashes match across machines. Recomputed RHS, diagonal, and contact-CSR hashes differ, so this is not a bitwise-identical matrix comparison. The force norms agree to roughly machine precision and contact nonzero counts agree; this does not establish identical CSR ordering or topology. Each machine independently passes the operator and true-residual checks. Floating-point summation and host/library differences can affect the iteration path.

## Hardware and software scope

| Item | V100 host replay | Archived Paratera reference |
| --- | --- | --- |
| GPU | Tesla V100-PCIE-32GB, physical index 4 | GeForce RTX 4090, 24 GB |
| GPU UUID | `GPU-cc51111a-07c8-2a6c-2462-df7b19ac8e9e` | `GPU-5e189b32-4a18-cd3b-98e4-b9da83b64ea4` |
| NVIDIA driver | 580.178.04 | 580.105.08 |
| Torch | 2.12.0+cu126 | 2.12.0+cu130 |
| Warp | 1.14.0, CUDA toolkit 12.9 | 1.14.0, CUDA toolkit 12.9 |
| Python / NumPy / SciPy / IPC | 3.14.6 / 2.4.6 / 1.17.1 / 1.6.0 | Same recorded versions |
| CPU | Dual Xeon Gold 5220R, affinity `0-23,48-71` | Different host; archived CPU quota was 10 cores |
| IPC threads | 8 | 8 |
| Other CUDA compute jobs at start | None | None in reference receipt |

The V100 was selected by UUID through `CUDA_VISIBLE_DEVICES`; the other eight GPUs on V100 host did not participate. The benchmark's general `nvidia-smi` field lists all physical GPUs, so use the selection receipt above to identify the measured device. During samples with at least 80% GPU utilization, the recorded SM clock was 1327–1380 MHz and memory clock 877 MHz. Observed device-memory use peaked at 3068 MiB, power at 155.39 W, and temperature at 61 °C. These are telemetry observations over this replay, not capacity estimates for a complete inverse experiment.

The CUDA 12 acceptance script also passed Torch sparse FP64, CuPy solve, JAX GPU FP64, and Apple Warp energy/gradient/HVP checks on this V100. See the [verification receipt](../data/comp03-receipts/cuda12-verification.txt).

This compares the two recorded server and software configurations. Different CPUs, driver builds, and Torch CUDA libraries prevent attributing the whole difference to GPU silicon. A V100's nominal FP64 capability alone does not predict this workload's measured time. This result does not establish the performance of V100S, DGXS, A800, H20, or a different AutoDL host, and it does not predict total inverse-fitting duration.

## Reproduction and evidence

The new replay ran on 2026-09-22 from **11:44:58 to 11:48:45 UTC** (19:44:58–19:48:45 Asia/Shanghai). Cherries reported `states_completed: 2` and a successful exit. `DEBUG=1` kept local evidence and disabled remote Comet recording. As in the reference run, shutdown warned about the legacy benchmark's unused `data/simple-skin-forward` asset; the requested results and success receipt were produced.

The staging helper copies the exact archived sources and 190 recursively referenced input/provenance files, plus the two benchmark checkpoints. The complete payload has 405 files and 784,632,318 bytes, all hash-verified on V100 host before execution. An isolated compatibility module in the staged tree maps historical `liblaf.peach` imports to Apple's owned solvers so the old source-hash contracts remain intact. No Peach distribution is installed. The active project and historical source archives were not rewritten for this replay.

The isolated remote tree is `${APPLE_ROOT}`. Package selection comes from the CUDA 12 lockfile. V100 host could not fetch the GitHub-hosted Warp and IPC wheels directly; exact copies were transferred and verified against their lockfile SHA-256 hashes before installation. All other packages were installed with frozen resolution. `uv pip check` passed for all 162 installed distributions. A source snapshot without Git metadata used `SETUPTOOLS_SCM_PRETEND_VERSION=27.dev197+gd56fa1b55.d20260922`.

Recreate the staging payload locally:

```bash
python exp/2026/09/22/gpu-hardware-comparison/src/10-stage-replay.py
```

The staged benchmark invocation on V100 host was equivalent to:

```bash
cd ${APPLE_ROOT}/exp/2026/09/22/solver-performance
CUDA_VISIBLE_DEVICES=GPU-cc51111a-07c8-2a6c-2462-df7b19ac8e9e \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
DEBUG=1 CHERRIES_NAME="V100 host V100 PCIe versus archived RTX 4090" \
CHERRIES_TAGS="gpu-hardware,v100,cuda12,matched,hessian" \
taskset -c 0-23,48-71 \
  ${APPLE_ROOT}/environments/cuda12/.venv/bin/python \
  src/49-benchmark-hessian-representations.py \
  --output-dir ${APPLE_ROOT}/exp/2026/09/22/gpu-hardware-comparison/data/comp03-v100-001
```

Use a new output directory for a fresh run. The staging helper and archived raw record are required: the active original benchmark script does not contain the fixed-reference-shift replay patch.

- [V100 raw summary](../data/comp03-v100-001/summary.json), [protocol](../data/comp03-v100-001/protocol.json), and [source hashes](../data/comp03-v100-001/provenance.json).
- [Archived 4090 raw summary](../data/reference-rtx4090/summary.json), [protocol](../data/reference-rtx4090/protocol.json), and [runtime receipt](../data/reference-rtx4090/runtime.json).
- [Comparison JSON](../data/comparison-001/comparison.json), [PNG](../data/comparison-001/comparison.png), and [SVG](../data/comparison-001/comparison.svg).
- [Replay manifest](../data/replay-manifest.json), [V100 terminal log](../data/comp03-receipts/benchmark-terminal.log), [telemetry](../data/comp03-receipts/telemetry.csv), and [installed runtime](../data/comp03-receipts/runtime.json).
- [Staging source](../src/10-stage-replay.py) and [comparison source](../src/30-compare.py).
