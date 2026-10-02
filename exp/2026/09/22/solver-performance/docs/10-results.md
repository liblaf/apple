# Forward solver performance investigation

The measured bottleneck is the forward equilibrium solve: thousands of PNCG
iterations, with CPU IPC directional curvature and CCD adding substantial cost
per contact iteration. The implicit adjoint is a smaller share of the measured
wall time. A faster GPU alone does not remove the CPU contact work or the
iteration count. The implemented diagonal PNCG-to-Newton hybrid gives a repeated
2.8x forward speedup on the valid collision-off fixture. Contact requires
stricter residual control: a separate endpoint-refinement comparison reaches
matching gradients with a 6.59x reduction in the sum of forward-stage timings.
The existing fitting defaults are preserved; the accelerator is opt-in.

## Evidence and validity

The original contact replay on V100 host took **610.402 s forward** (2,402 PNCG
steps) and **29.210 s adjoint**. It met the original force threshold
1.5192003475221146e-10, had zero inverted tetrahedra, and passed the existing
contact checks. This run used IPCTK's default 56 threads. The corresponding
collision-off one-degree replay took 119.404 s forward, but its endpoint had
one inverted tetrahedron: it is explicitly **not a valid numerical reference**.
These are different mechanical problems and their time ratio is not a solver
speedup.

The first collision-off trial illustrates why accepted-state force alone is
insufficient. Original PNCG and cached PNCG both met that threshold but differed
by 0.213 mm at the worst coordinate, and only the cached endpoint had zero
inversions. The checkpoint, seed, activation and jaw proposal hashes matched.
No stale gradient-cache path was found; nonconvex trajectory/reduction
sensitivity is a possibility, not an established cause. Caching reduced 11,380
gradient requests to 5,690 evaluations but took 164.126 s in this trial. It is
not enabled by default. Exact-curvature PNCG took 118.115 s with zero inversions;
the invalid original endpoint prevents an equivalence-based speedup claim.
Direct diagonal Newton exhausted 50 nonlinear iterations. The first block
prototype exhausted CUDA workspace before taking a step; the later version
factors independent 3-by-3 blocks in batches of 4,096.

The remaining original contact test arms were deliberately stopped after its
completed baseline to investigate threading and revised solvers. Their partial
times are not results. The one-thread V100 host replay was also stopped after the
fixed-state study showed that one thread makes the expensive operations slower.

## Matched quarter-degree collision-off results

The smaller 0.25-degree proposal has a valid original reference. Every arm
below has zero inversions, satisfies the unchanged force threshold, and passes
maximum displacement disagreement <= 1 micrometre and relative activation/jaw
adjoint-gradient disagreement <= 1e-3.

| Method | Forward, run 1 (s) | Forward, repeat (s) | Adjoint, run 1 (s) | Forward speedup, run 1 |
| --- | ---: | ---: | ---: | ---: |
| Original accepted-force PNCG | 42.847 | 42.379 | 17.134 | 1.00x |
| Exact-curvature PNCG with gradient reuse | 36.365 | 36.243 | 17.112 | 1.18x |
| Coarse PNCG + diagonal Newton | **15.299** | **15.210** | 17.037 | **2.80x** |
| Coarse PNCG + vertex-block Newton | 30.723 | not repeated | 17.118 | 1.39x |

The diagonal hybrid takes 257 coarse PNCG steps and two Newton steps. Its
worst displacement disagreement in run 1 is 0.370 micrometres; activation
and jaw gradient relative errors are 1.77e-5 and 2.12e-6. Forward-plus-adjoint
time improves by 1.85x, rather than the forward-only 2.80x. These two repeats
use separate RTX 3090 GPUs on V100 host; each candidate is compared against its
own control on that same GPU. Some portions of the independent runs overlap
on the host. Block setup is included, and its cost outweighs the reduced CG
iteration count on this fixture. The simpler diagonal variant is the best
measured candidate here.

- [Run 1](../data/comp07-quarter-degree-002/summary.json)
- [Independent repeat](../data/comp07-quarter-degree-003/summary.json)

## Contact comparison at the original tolerance

On the RTX 4090 with eight IPC threads, the original control required 2,848
PNCG steps, 484.226 s forward and 16.612 s adjoint. It had zero inversions and
force norm 1.487e-10. The two candidates reuse this completed control through
an independently hash-bound replay; the earlier exact-PNCG arm was stopped
before completion and has no claimed timing result.

| Method | Forward (s) | Coarse PNCG + Newton steps | Maximum displacement difference | Activation-gradient relative difference | Original agreement gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Diagonal hybrid | 67.368 | 289 + 4 | 1.992 micrometres | 0.4008% | **Fail** |
| Vertex-block hybrid | 65.533 | 223 + 3 | 1.992 micrometres | 0.1524% | **Fail** |

Both candidates meet the original force/contact checks and have zero inverted
tetrahedra. Both fail the declared 1-micrometre displacement and 0.1% activation
gradient agreement gates, so their lower runtime is **not an accepted
replacement speedup at this tolerance**. The exact physical adjoint was used
in all arms. Merely reducing forward iterations is insufficient for inverse
fitting. A tighter endpoint refinement is a separate diagnostic; these failed
rows are retained unchanged.

- [4090 original control](../data/paratera-contact-002/results.json)
- [Hybrid comparison, including failed gates](../data/paratera-contact-hybrid-003/summary.json)
- [Bound inputs, sources and original output hashes](../data/paratera-contact-hybrid-003/reference.json)

## Stricter contact accuracy diagnostic

Refining each saved endpoint with exact diagonal Newton to force tolerance
**1e-12**, with the same material, target jaw and activation, makes **all
endpoint agreement gates pass**. No additional jaw increment is applied.
The original and diagonal-hybrid endpoints now differ by only 3.55 nm at the
worst coordinate; activation-gradient relative disagreement is 8.64e-8 and jaw
gradient disagreement is 2.19e-10. This supports remaining equilibrium residual
as the explanation for the earlier disagreement on this fixture; it does not
prove uniqueness or stability over the broader model.

| Route to the refined endpoint | Prior forward (s) | Additional Newton refinement (s) | Sum of forward stages (s) | Relative cost |
| --- | ---: | ---: | ---: | ---: |
| Original PNCG + refinement | 484.226 | 11.115 | 495.341 | 1.00x |
| Diagonal hybrid + refinement | 67.368 | 7.815 | 75.184 | 6.59x faster |
| Vertex-block hybrid + refinement | 65.533 | 4.191 | 69.724 | 7.10x faster |

These are **sums of measured forward stages**, excluding intermediate adjoint
validation calls and runtime reconstruction. They are not measurements of one
uninterrupted full inverse-fitting run. The comparison uses a common tighter
accuracy target, whereas the original-tolerance comparisons above still fail.
The block route has one measurement; the small difference between the two
refined hybrid totals does not establish a general block-preconditioner win.

The refined original and diagonal-hybrid force norms are 5.71e-15 and
6.93e-15; the refined block endpoint has force 7.28e-13. All have zero inversions
and valid contact. Thus, for this contact fixture, numerical acceleration and
stricter equilibrium accuracy can be obtained together. The current
1.5192e-10 force threshold alone is insufficient for the declared cross-solver
activation-gradient agreement test.

- [Stricter accuracy results](../data/paratera-contact-polish-004/summary.json)
- [Endpoint/source bindings](../data/paratera-contact-polish-004/reference.json)

## Fixed-state IPC costs on the RTX 4090

Five synchronized measurements per setting at the same contact state, using a
bounded deterministic direction. These are operation timings, not complete
solve speedups. Settings were tested sequentially; no claim of a universally
optimal thread count follows.

| IPC threads | Gradient (ms) | Approximate directional curvature (ms) | First exact HVP (ms) | Cached exact HVP (ms) | CCD (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 (host default) | 5.675 | 89.552 | 23.805 | 4.706 | 75.800 |
| 1 | 3.153 | 392.535 | 31.745 | 4.368 | 311.577 |
| 2 | 3.240 | 209.241 | 22.556 | 4.438 | 172.929 |
| 4 | 4.003 | 91.354 | 19.849 | 4.368 | 82.626 |
| 8 | 4.908 | 49.606 | 20.434 | 4.547 | 45.551 |

Thread settings preserved the sampled force/Hessian values to approximately
floating-point reduction error (all declared numerical checks passed). The
cached contact Hessian is a CPU SciPy sparse matrix in the current code;
these measurements do not demonstrate a GPU-resident IPC implementation.

## Implemented candidates

- Solve-local accepted-gradient reuse, invalidated by state updates and tensor
  mutation version; material lifetime is limited to a single primal solve.
- Exact directional curvature for PNCG using the existing physical HVP.
- Safeguarded Newton-CG with explicit diagonal shifts, true linear-residual
  checks, negative-curvature rejection, strict Armijo and existing CCD.
- A declared two-stage hybrid: coarse PNCG to 1e-3 of initial force, followed by
  Newton refinement to the unchanged original force threshold. Failed Newton
  solves fail visibly; no hidden alternative solver accepts the state.
- Exact 3-by-3 vertex diagonal blocks obtained by graph-colored HVP probes;
  positive eigenvalue flooring affects only the preconditioner. Shifted
  preconditioners reuse eigenpairs. The physical implicit adjoint is unchanged.
- Opt-in `--ipc-threads` in the eye-neutral and expression-fitting entrypoints,
  with requested/effective values in each new run's protocol. The default
  preserves IPCTK behavior.

VBD's vertex-local nonlinear solves are useful motivation for local blocks,
but this prototype is block-preconditioned Newton, not an implementation of
[VBD](https://graphics.cs.utah.edu/research/projects/vbd/). A full VBD replacement
would require demonstrating convergence for this quasistatic, prestressed
model, including contact and the implicit gradients.
[GIPC](https://arxiv.org/abs/2308.09400) and
[StiffGIPC](https://arxiv.org/abs/2411.06224) motivate GPU contact assembly and
stronger global linear solves; adopting them would be a separate implementation
project rather than a switch in the current CPU IPCTK backend. The recent
[MAS-PNCG preprint](https://arxiv.org/abs/2604.19892) is relevant to multilevel
preconditioning, but its reported speedups are not measurements of this model.

## Validation and artifacts

Eleven CPU contracts passed, including actual Warp bulk/membrane blocks, cache
invalidation, force-based termination, failed trial rejection, shifted inverse
checks, and a nonquadratic hybrid transition. The full-size CUDA block-factor
check handled 257,009 blocks in 0.290 s with 2.337 GB peak allocation; direct and
shifted inverse comparisons passed at 1e-12. This is a factorization test, not a
complete nonlinear benchmark.

- [CPU Cherries/Comet run](https://www.comet.com/liblaf/apple/5fb2fce38e134db5a764d81cdfa3b633)
- [CPU validation](../data/cpu-validation-final.json)
- [Full-size CUDA block validation](../data/vertex-block-batched-eigh-cuda.json)
- [Original collision-off trials](../data/comp07-collision-off-001/summary.json)
- [Completed original contact baseline](../data/comp07-contact-001/results.json)
- [Contact baseline interruption record](../data/comp07-contact-001/completion.json)
- [4090 fixed-state thread measurements](../data/paratera-ipc-threads-001.json)
- [Input transfer manifest](../data/remote-staging-inputs.txt)

Remote experiments use Cherries with `DEBUG=1` because account credentials were
not copied to compute machines. Both machines use the same copied Python 3.14.6,
Torch 2.12.0, Warp 1.14.0 and IPCTK 1.6.0 environment. All 192 transferred input
files were verified by SHA-256. Compilation is warmed before solver timings;
preconditioner construction remains inside timed solves. Each run archives its
actual Python sources, checkpoints, input hashes, outputs and failure receipts.
No existing fitting job or physical input was modified. The existing CPU
continuation suite also passed after adding the optional IPC thread setting;
legacy default continuation remains compatible, while explicit thread changes
remain incompatible with an already recorded protocol.

- [Continuation validation](../data/continuation-validation-001/summary.json)
- [Continuation Cherries/Comet run](https://www.comet.com/liblaf/apple/2558704f2cc0481197c88aec728fd632)

## Reproduce and use

Run from `exp/2026/09/22/solver-performance` using the repository environment:

```bash
CHERRIES_NAME="Matched quarter-degree solver replay" \
CHERRIES_TAGS=performance,solver,matched \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python src/10-benchmark.py \
  --cases collision_off_pose \
  --collision-off-jaw-degrees 0.25 \
  --methods exact_pncg,hybrid_diag,hybrid_block \
  --ipc-threads 8 --wall-seconds 600 \
  --output-dir data/quarter-degree-new
```

The candidate wall limit is 600 seconds; the original control retains its
existing iteration budget. Use a fresh output directory. On remote machines
add `DEBUG=1`; see source archives and stdout logs for the precise recorded
runs. The smaller proposal is explicit and must not be compared against the
one-degree trial as a speedup.

The numerical implementation is opt-in through `accelerate_runtime`:

```python
from accelerated_solvers import accelerate_runtime

physics.runtime = accelerate_runtime(physics.runtime, "hybrid_diag")
```

Import this from the new experiment's `src` alongside the existing joint
experiment helpers. It replaces the primal only and retains the existing
implicit-adjoint interface. The broad expression-fitting default remains
unchanged: these local tests do not validate all jaw proposals, activation
states, expressions, or mechanical stability. The failed one-degree reference
is a concrete reason not to make that broader claim.

The contact experiments were run on the same RTX 4090, using the same copied
runtime and eight IPC threads. After the completed original control, the
candidate-only and accuracy commands were:

```bash
DEBUG=1 CHERRIES_NAME="Matched RTX4090 contact hybrids" \
CHERRIES_TAGS=performance,solver,matched,contact,hybrid,paratera \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/16-replay-candidates.py \
  --reference-dir data/paratera-contact-002 \
  --cases contact_expression --methods hybrid_diag,hybrid_block \
  --ipc-threads 8 --wall-seconds 600 \
  --output-dir data/paratera-contact-hybrid-003 \
  --origin-metadata data/origin-metadata.json

DEBUG=1 CHERRIES_NAME="Contact endpoint stricter residual diagnostic" \
CHERRIES_TAGS=performance,contact,accuracy,polish,paratera \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python src/17-polish-contact-endpoints.py \
  --reference-dir data/paratera-contact-002 \
  --hybrid-dir data/paratera-contact-hybrid-003 \
  --output-dir data/paratera-contact-polish-004 \
  --origin-metadata data/origin-metadata.json
```

Here `python` is the copied task environment's Python 3.14.6. Replay deliberately
rejects a changed physical source tree: use the archived matching sources and
fresh output names when reproducing. The GPU kernel source, frozen mechanics,
and exact implicit adjoint have not been changed by this investigation.
