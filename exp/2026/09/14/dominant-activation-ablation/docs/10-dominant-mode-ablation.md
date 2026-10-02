# Shape after keeping only the strongest contraction mode

Removing the other effective activation modes substantially weakens the fitted expression at the saved activation strengths. The corrected, unregularized baseline has an area-weighted target-fit RMS of **1.835697 mm**; the dominant-mode-only equilibrium has **3.351208 mm**. Overall facial motion falls from **4.150593 to 2.102464 mm RMS**, retaining **50.65%** of the baseline motion. The two equilibrium shapes differ by **2.287526 mm RMS**.

This is a forward ablation of an existing fit. No activation magnitude or direction was fitted again. It shows that the omitted modes contribute materially to the saved expression; it does not establish that a single-axis model could not fit the target after its strengths are optimized.

![Target, full fitted activation, strongest contraction mode only](../data/20-comparison/composites/side-context.png)

Left to right: target smile, replayed full fitted activation, and the re-solved strongest-contraction-mode-only shape. The last panel has a smaller mouth opening, less mouth-corner pull and weaker adjacent-cheek deformation. All panels use actual displacement scale 1, identical cameras and lighting, and flat shading on unchanged topology.

![Mouth-corner comparison in the same order](../data/20-comparison/composites/region1-mouth-corner.png)

The [surface displacement-difference map](../data/20-comparison/difference/side-context.png) shows the Euclidean magnitude of the full-minus-dominant displacement at each skin vertex, on the full-activation geometry. The largest difference is **7.932187 mm**, concentrated around the mouth and adjacent cheek. This local maximum differs from the full-face area-weighted RMS above. Forehead and temple changes are small in this view.

## Exact starting result and intervention

The user selected the corrected physical-volume result. The input is the rest-start, unregularized Raw6 baseline at Adam update 200, with physical-volume flag set and SHA256 `21b7e546f04c5566629a1d3634659ec0c5738df215699ba23e24eb7ac856abd7`:

`${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/08/physical-volume-baseline/data/20-baseline/step-0200.npz`.

The mesh has 228,660 vertices, 1,146,517 tetrahedra and 288,235 activation controls. Materials, target, fixed degrees of freedom and tissue fractions are unchanged. The face surface is used for fitting and display; **there is no skin energy and no contact**. Aponeurosis has E = 0.1 MPa, nu = 0.35; fat has E = 0.003 MPa, nu = 0.49; muscle has E = 0.03 MPa, nu = 0.49. The baseline's classical Lame lambda convention is retained.

The corrected muscle law is

$$
W(F,B)=\frac{\mu}{2}(\|FB\|_F^2-3)-\mu(J-1)
       +\frac{\lambda_0}{2}(J-1)^2,\qquad J=\det F,\quad B=A^{-1}.
$$

Let \(Z_0=B_0B_0^T-I\), and let \(z_1\) be its largest eigenvalue with unit reference eigenvector \(n_1\). The retained field is

$$
Z_1=\max(z_1,0)n_1n_1^T,\qquad
B_1=I+\big(\sqrt{1+\max(z_1,0)}-1\big)n_1n_1^T.
$$

This preserves exactly the positive effective mode shown by the existing direction glyphs, including its strength. In B, the transverse eigenvalues become 1; in A, the transverse active stretches also become 1. It does not set transverse B entries or stretches to zero, and it does not constrain the actual transverse deformation F. For the 63 cells with no positive mode, activation becomes identity.

The original B is not positive definite in 2,573 active cells. The corrected energy depends on B through BB^T, so its signs do not affect the forward mechanics. Using the positive square root for the retained effective tensor does not introduce an additional mechanical change beyond deleting the unwanted Z modes. These fitted axes are effective tensor directions; their visual resemblance to muscles is not an anatomical measurement.

## Forward solves and measured shapes

First, replay the original signed B at its saved displacement. The replay changes the face by only **0.000001374 mm RMS**, reproducing the original energy, packing and geometry. Then remove the omitted modes through

$$
Z(t)=(1-t)Z_0+tZ_1,\qquad t=0.25,0.5,0.75,1.
$$

At each stage, take \(B(t)=\sqrt{I+Z(t)}\) and solve equilibrium, starting from the preceding shape. The strongest positive reference mode stays fixed throughout. The stage parameter scales the omitted tensor eigenvalues; it is not a percentage of cells or a fraction of directions.

| Removed fraction of omitted Z modes | Target-fit RMS (mm) | Motion RMS (mm) | Change from saved shape RMS (mm) | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| 0%: full activation replay | 1.835697 | 4.150593 | 0.000001 | 1 |
| 25% | 2.213980 | 3.417505 | 0.793420 | 2 |
| 50% | 2.626707 | 2.872398 | 1.392307 | 5 |
| 75% | 3.008371 | 2.444668 | 1.878094 | 1 |
| 100%: dominant mode only | 3.351208 | 2.102464 | 2.287526 | 0 |

All three RMS columns use the same rest-skin lumped-area weights on the finite IsFace target vertices. Motion is measured relative to the reference shape. The final fit error is **82.56% higher**, and motion is **49.35% lower**, than the replayed baseline.

The frozen right-mouth-corner ROI has uniform vector fit RMS **1.658858 → 6.210673 mm**. Its convention differs from the area-weighted full-face table. The result therefore loses a substantial part of the mouth-corner target motion, rather than merely making a small adjustment to an otherwise retained fit.

Active-muscle-volume-weighted RMS(det F - 1) decreases from **0.015434 to 0.010098**. Final minimum det F is **0.035041**, and the final mesh has zero inverted tetrahedra. The continuation contains transient inversions, so this is not an inversion-free physical trajectory or a proof of geometric validity. The baseline already contained one inverted, almost entirely fat, mixed element.

The frozen 5 mm high-pass normal *target residual* over the three diagnostic ROIs rises from **0.182183 to 0.275553 mm RMS**. This measures fine-scale mismatch with the target, not geometric roughness alone. It does not support a claim that deleting the extra modes makes the visible surface smoother.

## What this establishes

The strongest contraction mode alone, at its original strength, produces much less facial motion and fits the target appreciably worse. Alignment of that mode with the apparent muscle orientation is therefore insufficient to explain the complete fitted deformation. Secondary positive modes and negative effective modes were doing consequential work in this saved solution.

The weighted mean of the per-cell omitted squared tensor-magnitude fractions is **44.7375%**. The ratio of the globally volume-weighted omitted squared norm to the globally weighted full squared norm is **11.6518%**. These are distinct statistics; neither is a fraction of mechanical energy, displacement or fit explained.

This experiment deletes the secondary positive modes and negative modes together. It does not separate their contributions. It also examines one continuation branch, one saved activation, one target and one material model. No inverse convergence or equilibrium uniqueness claim is made. A subsequent fixed-direction, scalar-strength refit would answer a different question: whether the retained directions can recover the target after their strengths are adjusted. That experiment has not been run here.

## Verification and reproducibility

All five equilibrium solves returned success under the original PNCG settings: max 5,000 steps, relative tolerance 5e-4, absolute tolerance 1e-10 and the original 10-step line search. Their PNCG step counts were 2 / 2,155 / 2,582 / 2,393 / 1,869. Reported final free-gradient norms were 7.05e-11 / 2.72e-10 / 2.41e-10 / 2.27e-10 / 2.19e-10. Fixed-degree-of-freedom errors are zero. The retained tensor reconstruction error is at most 1.07e-14.

The imported Apple runtime source hashes match the archived baseline runtime. Execution used the isolated worktree through PYTHONPATH, rather than the editable original checkout. Current interpreter versions were Python 3.14.6, Torch 2.12.0+cu130 and Warp 1.14.0, on an NVIDIA RTX 4090. Computation and checkpoint/diagnostic writing took 133.0 seconds after initialization. The process and Cherries shutdown exited with code 0.

Working directory:

`${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Remove secondary activation modes from corrected face' \
CHERRIES_TAGS='face,activation,direction,forward-ablation,physical-volume' \
.venv/bin/python src/10-forward-ablation.py
```

Use a fresh output directory for a new run; the script rejects nonempty output. The default command above reproduces the original command, whose outputs are retained. No optimizer is used to refit activation in this experiment; PNCG solves only the mechanical equilibrium.

The [forward Comet run](https://www.comet.com/liblaf/apple/c20e4e96f22f4c8d88bbbafa6735691e) recorded the name, command, entrypoint, Git SHA and five sets of scalar metrics. Its summary reports fit RMS range 1.835697–3.351208 mm, motion RMS range 2.102464–4.150593 mm, and inversion count range 0–5. Local logs retain the complete Comet summary. An initial startup attempt failed before model construction because the output directory had not been created; the entrypoint now creates it explicitly. The successful run also emitted an automatic-logging import-order warning, which did not affect the solver or locally saved evidence.

The [raw summary](../data/10-forward/summary.json), [protocol and runtime source snapshots](../data/10-forward/protocol.json), and stage NPZ files retain the original run evidence. The original generic `z_update_frobenius_rms` field defaulted to zero because a previous Z was not supplied; it must not be interpreted as the amount of activation removed. The verification receipt recomputes the actual baseline-relative Z changes. The published entrypoint fixes that diagnostic argument, adds maximum det F and an explicit pure-muscle mask, and includes lint-only changes. The exact source that executed the solves remains archived under `data/10-forward/sources/experiment/`; no equilibrium was recomputed for these diagnostic-only changes.

Rendering completed with code 0 and a [separate Comet record](https://www.comet.com/liblaf/apple/d2cd6d8c2a7e46ceb19eb4e4c669760d). The [render receipt](../data/20-comparison/summary.json) records source/input/output hashes, frozen cameras, scalar units and mode-reconstruction checks. Four exported VTPs retain exactly the skin's 15,299 points and 29,899 triangles. The standalone panels, composites and difference maps were visually inspected. The renderer source has one subsequent lint-only removal of an unused suppression; its executed snapshot is preserved.

The exact rendering command, from the same experiment directory, was:

```bash
env \
  PYTHONDONTWRITEBYTECODE=1 \
  LIBGL_ALWAYS_SOFTWARE=1 \
  PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src:${CHERRIES_ROOT}/src \
  CHERRIES_NAME='Render dominant activation ablation comparison' \
  CHERRIES_TAGS='dominant-activation,ablation,shape-comparison,render' \
  .venv/bin/python src/20-render-comparison.py
```

Its images and exact surface meshes are under `data/20-comparison/`.

The independent, CPU-only [verification receipt](../data/30-verification/summary.json) passed all five checkpoint hash, source-snapshot, tensor projection, packed-control, continuation, boundary-condition and metric checks. Across stages, dominant-mode strength is preserved within 2.85e-14, omitted-mode scaling within 2.67e-15, and BB^T - I reconstruction within 4.98e-14. The final physical determinant range is **0.035041 to 2.286384**. The exact pure-muscle mask also gives zero final inversions. Baseline-relative Z change has a muscle-volume-weighted Frobenius RMS of **0.498535**.

The verifier ran as `.venv/bin/python src/30-verify-ablation.py` in this experiment directory, with the same isolated Apple source selection, and exited with code 0. Its [successful Comet record](https://www.comet.com/liblaf/apple/27dc981abfa241438e8db7ea4096efe0) and `logs/30-verify-ablation.log` retain the summary. Two earlier verifier startup/check attempts exposed a missing output-directory creation and an inappropriate exact-equality comparison between the saved shape and its re-equilibrated replay. The final check uses the reverse-triangle-inequality bound supplied by the independently measured replay displacement. These fixes did not rerun or modify the mechanical solutions.

The three authored entrypoints pass focused Ruff checks and formatting checks. A final formatter pass on the verifier preserves its executed AST exactly; its pre-format execution source is retained under `data/30-verification/sources/`. Runtime helpers were copied unchanged from the existing studies. The saved-state ablation was reviewed independently of its implementation and the final plots were inspected by both the renderer and the parent task.

All new files are isolated in this experiment group. No original-checkout or SMAS-task files were changed; no commit or push was made.
