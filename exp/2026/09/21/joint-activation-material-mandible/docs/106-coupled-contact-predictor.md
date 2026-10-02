# Coupled contact prediction for jaw and active stress

The previous initializer moved the prescribed mandible while holding free tissue fixed. Its CCD gate rejected a 1° proposal before the tissue could respond. The new initializer predicts their coupled displacement, checks the complete motion, then runs the unchanged strict PNCG equilibrium corrector.

## Method

At the current seed, assemble the exact contact-inclusive free Hessian and solve

\[
H_{ff}\,\Delta u_f=-\eta r_f-H_{fc}\,\Delta b-\Delta_q R_f.
\]

Here \(\Delta b\) is the exact hinge boundary displacement, and \(\Delta_q R_f\) is the active-stress force change evaluated at the same geometry. Old contact state, materials and displacement are owned snapshots. Each linear solve must satisfy a checked relative residual of at most 1.05e-7.

Full soft-versus-rigid CCD includes the source skull, source mandible and both fixed eyes. For a fixed hinge, the maximum arc-to-chord deviation is \(2R\sin^2(|\Delta\theta|/4)\). The predictor's CCD separation is conservatively enlarged by this deviation; the physical barrier and its 10 nm buffer are unchanged. The endpoint must also pass intersection and contact checks.

`joint_coupled_continuation.py` constructs an initialization for the full outer proposal using internal parameter substeps when necessary. By default, PNCG relaxes each admitted nonfinal substep to the original force tolerance before the next prediction. These internal equilibria are **not accepted inverse iterations**. Both stress and hinge parameters advance together; rejected steps are recomputed and rechecked. The residual factor \(\eta\) decreases with the attempted parameter fraction so that residual correction also vanishes during backtracking. Exhausting the budget rejects the entire proposal; partial progress is never relabeled as its target.

An earlier variant skipped intermediate relaxation. Benchmark `coupled-continuation-benchmark-001` reached 0.367779 of the requested 1°/12.3288 Pa proposal in three admitted seed substeps, but later predictor attempts took approximately 94 seconds and were rejected. It was interrupted to compare strict internal correction; no final target equilibrium was obtained. This timing suggests a numerical issue worth testing, but does not by itself establish Hessian ill-conditioning. The original log and `external-interruption.json` are retained.

At the exact target, the original contact-on PNCG solver must satisfy the original force threshold, then the original implicit adjoint and outer objective acceptance checks apply. Prediction is detached initialization, not a replacement derivative or new physical energy. This does not guarantee global feasibility, branch uniqueness or faster execution for every expression.

## Full-face single-prediction result

Cherries/Comet: [830b4363780643f3b8744b8587b3a4e1](https://www.comet.com/liblaf/apple/830b4363780643f3b8744b8587b3a4e1). Evidence: `data/coupled-predictor-benchmark-001/{summary.json,provenance.json,predicted.pt,corrected.pt}` and its archived sources. Start: approved eye-inclusive neutral, zero activation.

| Proposed opening | Frozen-tissue CCD fraction | Coupled CCD fraction | Admitted |
| --- | ---: | ---: | --- |
| 1° | 0.039858 | 0.154338 | No |
| 0.5° | 0.079939 | 0.320305 | No |
| 0.25° | 0.160103 | 0.646192 | No |
| 0.125° | 0.320433 | 1 | Yes |

For the admitted 0.125° step, prediction took 18.21 s; the strict corrector took 33.80 s, including 264 PNCG iterations. Final force was 1.47892e-10 against the unchanged 1.51920e-10 threshold. The minimum active contact gap was 17.0231 µm, with no soft–rigid intersections. MouthOpen RMS was 7.54798 mm versus the contact-on neutral's approximately 7.63066 mm. This is a forward test, not a converged inverse fit.

The accepted angle is four times the 0.03125° increments used by the earlier contact-on fit. The approximately 52 s above excludes the three rejected predictor trials and model setup. These are shared-machine measurements, not a controlled equal-angle timing comparison.

## Reproduction and checks

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```sh
CHERRIES_NAME='Full face coupled jaw tissue predictor pilot' CHERRIES_TAGS='joint-inverse,contact,coupled-predictor,pncg' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python src/106-benchmark-coupled-predictor.py
```

CPU check 107 verifies block algebra, scaled residual correction, state restoration and failure handling. CPU check 109 verifies internal continuation, strict internal correction, exact target completion, bounded rejection and unchanged caller inputs. The full-target benchmark with strict internal correction is recorded separately in `data/coupled-continuation-benchmark-002`; its success must be read from its receipt rather than inferred from these tests.

The soft collision boundary contains 34,245 vertices and has zero overlap with either the 7,510 prescribed FEM mandible nodes or the complete fixed-node set. Only the separately appended mandible surface curves during the hinge motion; thus one arc-deviation bound per allowed soft–rigid pair is sufficient. This was checked against the original full-skull geometry and expression-input artifacts.

## Basis

The [IPC technical supplement, §5](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf) identifies the limitations of advancing prescribed collision objects before deformable response and uses augmented-Lagrangian kinematic constraints. Our implementation uses a different initializer: coupled tangent prediction and adaptive continuation, followed by the existing fixed-boundary equilibrium solver. The tangent equation follows implicit differentiation of the equilibrium residual; see [Huang et al., §4](https://arxiv.org/html/2205.13643v3#S4.SS1). Neither paper validates this particular facial implementation; the numerical receipts above supply its current evidence.
