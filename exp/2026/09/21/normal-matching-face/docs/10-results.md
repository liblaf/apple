# Unrestricted 3D face: positional and normal matching

All four independent neutral-start runs completed 100 Adam updates. At this
fixed budget, adding normal matching reduced normal-angle RMS by about 19.7%,
surface-gradient residual by about 7.55%, and target-relative 5 mm high-pass
residual by about 10.1%. Position RMS was about 0.61% lower. This preserves the
positional fit while improving local shape agreement in this fixture.

The fixed activation-smoothing coefficient reduced tensor variation by only
about 3.2% and barely changed the surface metrics. It was too weak to make a
substantial visible difference here; this is not evidence against stronger
regularization or a tuned weight.

Every update-100 endpoint contains one inverted tetrahedron, and all four
trajectories were still improving. Normal matching caused earlier inversion
in this experiment: update 60 rather than 81, with or without smoothing. These
are finite-budget comparisons, not converged or mechanically valid final fits.
The independent CPU audit passed, and the common inversion-free saved geometry
at update 50 is reported separately.

## Full endpoints at update 100

All errors use the same fixed reference skin support and weights. The high-pass
column is the target-relative normal-component residual over the established
primary facial regions at the 5 mm scale; it is distinct from triangle-normal
matching and activation-tensor variation R. Lower errors and R are preferable;
motion is a diagnostic, not an error score.

| Smoothing | Data loss | Position RMS (mm) | Normal-angle RMS (deg) | Surface-gradient RMS | 5 mm high-pass residual (mm) | Activation R | Motion RMS (mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | L2 | 2.341699 | 9.861165 | 0.232366 | 0.215986 | 3.122739 | 3.480294 |
| Off | L2 + normal | 2.327387 | 7.918071 | 0.214792 | 0.194087 | 3.231395 | 3.408106 |
| On | L2 | 2.343596 | 9.858768 | 0.232320 | 0.215988 | 3.021370 | 3.477487 |
| On | L2 + normal | 2.329388 | 7.918563 | 0.214785 | 0.194229 | 3.126932 | 3.405310 |

Adding normal loss with smoothing off changes position RMS by -0.611%, normal
angle by -19.705%, surface-gradient RMS by -7.563%, and high-pass residual by
-10.139%. With smoothing on, the corresponding changes are -0.606%, -19.680%,
-7.548%, and -10.075%. Activation variation increases by about 3.5% when normal
matching is added; improved surface agreement does not mean smoother activation.

Adding smoothing to L2 reduces R by 3.246%, changes position RMS by +0.081%,
and changes high-pass residual by +0.00093%. Adding it to L2 + normal reduces R
by 3.233%, changes position RMS by +0.086%, and changes high-pass residual by
+0.073%. These surface differences are very small. No replicate or statistical
significance claim is made.

For context, neutral position RMS is 5.095908 mm and normal-angle RMS is
9.070089 degrees. Pure L2 substantially improves position but ends with worse
normal agreement than neutral. The combined objective improves both metrics.
The slightly lower positional error under the combined objective is a measured
finite-iteration result in a nonconvex problem, not a claim that its global
positional optimum is better than the pure positional objective's optimum.

### Physical validity and convergence

| Smoothing | Data loss | First inverted update | Min det(F) at 100 | Inverted tets at 100 | Non-SPD active tensors at 100 | Final physical gradient / initial | Objective change, updates 90 to 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | L2 | 81 | -0.118597 | 1 | 819 | 0.2496 | -7.951% |
| Off | L2 + normal | 60 | -0.317052 | 1 | 802 | 0.2503 | -6.960% |
| On | L2 | 81 | -0.119043 | 1 | 784 | 0.2479 | -7.848% |
| On | L2 + normal | 60 | -0.317111 | 1 | 769 | 0.2484 | -6.871% |

Normal matching improves the surface errors but produces a more negative
minimum physical determinant at the full endpoint. The physical gradient
remains about one quarter of its initial value, and every own objective drops
another 6.9–8.0% over the final ten updates. Thus the histories do not establish
inverse convergence. There is no 3D Hessian stability check.

## Common inversion-free saved checkpoint: update 50

All four saved geometries at this checkpoint have zero inverted tetrahedra.
The latest common inversion-free evaluated trace row is update 59, but not all
four geometries were saved there. Do not confuse its scalar metrics with a
saved mesh comparison. The comparison below and its render use update 50.

| Smoothing | Data loss | Position RMS (mm) | Normal-angle RMS (deg) | 5 mm high-pass residual (mm) | Activation R | Min det(F) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Off | L2 | 3.073070 | 9.416825 | 0.271493 | 1.437917 | 0.265578 |
| Off | L2 + normal | 3.053930 | 8.211032 | 0.263082 | 1.483108 | 0.117389 |
| On | L2 | 3.073850 | 9.416147 | 0.271515 | 1.418460 | 0.265682 |
| On | L2 + normal | 3.054741 | 8.211114 | 0.263135 | 1.463020 | 0.117629 |

At this common saved checkpoint, adding normals lowers normal-angle RMS by
about 12.8%, position RMS by about 0.62%, and high-pass residual by about 3.1%
for either smoothing setting. This shows the local matching benefit before
inversion, though positive determinants alone do not certify stability.

## Figures

All face views use the same cameras, reference scale, lighting and flat
shading. Displacements are not exaggerated; no target volume is synthesized.
Rows are smoothing off/on; columns are target, L2 and L2 + normal. The endpoint
images explicitly label their one inverted tetrahedron.

- [Full face, update 100](../data/30-figures/full-comparison.png)
- [Mouth region, update 100](../data/30-figures/mouth-comparison.png)
- [Position-error maps, update 100](../data/30-figures/position-error-maps.png)
- [All loss, residual and gradient histories](../data/30-figures/loss-and-gradient-histories.png)
- [Full face, inversion-free update 50](../data/31-figures-noninverted/full-comparison.png)
- [Mouth region, inversion-free update 50](../data/31-figures-noninverted/mouth-comparison.png)

## Experiment

The user requested the 2D L2-versus-L2-plus-normal experiment on the human face
with unrestricted activation. The four conditions are smoothness off/on by
L2-only/L2 plus normal. Each starts from the neutral face, independent zero
Raw6 controls, zero displacement and fresh Adam moments. The adjoint initial
guess is also reset to zero between branches. No fitted checkpoint or learned
axis initializes any main run.

The full historical Smile fixture contains 228,660 volume vertices, 1,146,517
tetrahedra and 288,235 active muscle tetrahedra. Independent symmetric tensors
`B=I+sym(q)` provide 6 DoF per active tetrahedron, or 1,729,410 controls. There is
no contraction-only, PSD or spectral projection. Passive materials, skull
constraints, zero skin energy and corrected physical-volume energy are inherited
from the established face setup. Physical volume uses `J=det(F)`; activation
affects the `F B` norm term.

The common positional and normal support has 15,299 skin vertices and 29,899
triangles, excluding three isolated legacy positional points. Target geometry
is defined only on the skin. There is no synthesized target volume deformation.

The [frozen protocol](00-protocol.md) defines

```text
objective = L2 + beta * (L20 / N0) * N + lambda * R.
```

L2 is fixed reference-area-lumped positional component MSE in mm². Its vector
position RMS is `sqrt(3 L2)` mm. N is fixed reference-area-weighted squared chord
distance of oriented unit normals on corresponding deformed and target skin
triangles. It measures direction agreement, not triangle area, stretch or
position. At neutral, `L20=8.656092875221388` and
`N0=0.0243814646293817`. Beta is 0 or .05; the positional coefficient remains 1.

R is the same-muscle, shared-face activation-tensor variation, with the existing
5 mm length scale and harmonic muscle-fraction conductances. Lambda is 0 or
`0.003214147722027223`, the previous conservative Raw6 coefficient, fixed for
both data objectives. Its historical calibration used uniform positional
weights, so this is a legacy scale reference rather than a newly optimized
coefficient. It does not transfer the numerical strength of the 2D penalty or
the learned-axis face penalty.

Every main branch uses 100 Adam updates, constant learning rate .3, epsilon .01,
betas .9/.999. Forward tolerances are relative 5e-4 and absolute 1e-10, with a
5,000-step budget. Adjoint relative tolerance is 5e-4, retaining the established
CG/MinRes solver policy. There is no inverse restart, schedule change, determinant
penalty or coefficient search.

## Completed preflight checks

The [validation record](05-validation.md) reports CPU algebra/autograd checks,
full 3D implicit finite differences for normal-only and combined objectives,
and the exact commands. All passed. The largest full implicit derivative error
was 1.8877% (2% limit); the combined-objective maximum was .01093%.

The final one-update smoke ran all four branches and exited 0. Independent CPU
recalculation verified positional, normal, gradient, tensor variation, signed
volume and regional high-pass metrics, source and fixture hashes, both gates,
area-lumped skin weights, loss normalizations and exact neutral receipts.
Paired initial physical activation-update RMS values agree within 6.60e-7
relative error. GPU solve/adjoint numerical variation means elementwise
gradients are not bitwise equal; the raw maximum relative differences are below
5.2e-5. These are small numerical differences, not reused fitted starts.

The final independent CPU audit also passed
([checks.json](../data/20-verification/checks.json)). It confirms all four
100-update completed statuses, null failures, exact neutral starts and 101
successful forward plus adjoint receipts per branch (404 of each overall).
It recomputes the endpoint and shared-checkpoint metrics from saved geometry,
including signed volume, and verifies all 97 numerical source records and three
fixture receipts. The independently reconstructed reference-area skin weights
and neutral loss normalizations agree with the run. The final trace analysis
is saved in [analysis.json](../data/21-analysis/analysis.json).

## Reproducibility and commands

Working directory:
`exp/2026/09/21/normal-matching-face`.

Final smoke:

```bash
env DEBUG=1 CHERRIES_NAME='3D face normal matching: four-branch smoke' CHERRIES_TAGS='face,raw6,normal-matching,smoke' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py --output 09-smoke --steps 1
```

Main run:

```bash
env CHERRIES_NAME='3D neutral face: unrestricted activation L2 and normal factorial' CHERRIES_TAGS='face,3d,raw6,unrestricted,neutral-start,normal-matching,smoothness,2x2' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py
```

[Main Comet experiment](https://www.comet.com/liblaf/apple/385328050ca44caab70020f8bbf77bc8).
`data/10-comparison/protocol.json` contains the actual runtime versions, fixture
hashes, numerical source snapshots and configuration. The repository includes
preexisting uncommitted research and physics changes, so those snapshots, not
Git HEAD alone, specify this run. No production source was edited for this study.

The main process exited 0 after full Cherries/Comet shutdown. Its final Comet
summary contains 101 observations for each of the six explicitly logged metrics
in each of the four branches. It records start time
`2026-09-21 03:16:51.677135+08:00` and end time
`2026-09-21 04:42:31.601049+08:00`. Other experiment processes shared the GPU
during part of the run, so elapsed times are not performance comparisons.
Comet stores the logged metrics and run metadata; full state arrays and the
complete numerical snapshots are preserved locally in `data/10-comparison`.

Final independent verification:

```bash
env CHERRIES_NAME='3D face normal matching: independent final verification' CHERRIES_TAGS='face,raw6,normal-matching,verification,cpu,final' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/20-verify.py --comparison-dir 10-comparison --output 20-verification
```

[Verification Comet experiment](https://www.comet.com/liblaf/apple/000927c1032b4e0585abebd59839898c).
The process completed with exit 0 after shutdown; the receipt has `passed=true`.

Final trace analysis:

```bash
env CHERRIES_NAME='Face normal-matching factorial: trace analysis' CHERRIES_TAGS='face,normal-loss,smoothing,analysis,convergence' .venv/bin/python src/21-analyze.py
```

[Analysis Comet experiment](https://www.comet.com/liblaf/apple/c5acc45aa7254d8293d886a7ee279498).
The process completed with exit 0 after shutdown. It requires the final CPU
verification receipt and completed runs, distinguishes trace step 59 from saved
geometry step 50, and avoids comparing different total objectives as a common
accuracy score.

Final endpoint figures:

```bash
env CHERRIES_NAME='Normal matching face: primary comparison figures' CHERRIES_TAGS='target-normal,3d,normal-matching,neutral-start,figures' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/30-render.py
```

[Endpoint-figure Comet experiment](https://www.comet.com/liblaf/apple/64d7f47bc774457988eb83a991400541).
The process completed with exit 0 after shutdown. Its metadata resolves the
latest common saved step to 100. The first error-map export had overlapping
horizontal color-bar labels; it is preserved in `data/30-figures-initial-bar`
with `logs/30-render-initial-bar.log`. Final maps use readable vertical bars.

Final inversion-free figures:

```bash
env CHERRIES_NAME='Normal matching face: verified inversion-free update 50 figures' CHERRIES_TAGS='face,raw6,normal-matching,figures,inversion-free,step50' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/30-render.py --shared-step 50 --output-dir exp/2026/09/21/normal-matching-face/data/31-figures-noninverted
```

[Inversion-free-figure Comet experiment](https://www.comet.com/liblaf/apple/9e2a5d1c7b0444db98f43c45cd8d9718).
The final process completed with exit 0 after full shutdown, and metadata
confirms step 50 while retaining all four full-run endpoint steps of 100.
A duplicate launch safely refused to overwrite an earlier completed export;
that earlier set was preserved in `data/31-figures-agent`, then the final render
was repeated with a directly observed completion receipt. No optimization was
rerun. All 20 final PNGs are byte-identical to the visually checked earlier set.

Visual inspection covered full-face and mouth comparisons at both steps, the
endpoint error maps, and all six history panels. The geometric cameras and
scale match; both error-map sets use the same 0–11.037214 mm color range. Each
set contains 20 PNGs and its renderer source snapshot. Final lint checks and
all local report links passed.

The main process emits an import-order notice from Comet and an inherited
PyTorch non-leaf-gradient warning. Required metrics are logged explicitly, and
the numerical code asserts finite control derivatives and successful forward
and adjoint receipts. Preflight verification passed despite these notices.

## Interpretation limits

Report positional fit, target normal agreement, target-relative surface
high-pass residual, activation variation and motion separately. A smaller R
does not establish smoother target-relative surface shape. The normalized own
objectives contain different terms and cannot rank position accuracy.

Physical inversions and nonpositive activation eigenvalues remain possible with
unrestricted Raw6. Full endpoints are retained and labeled even if inverted;
the latest common inversion-free saved checkpoint is a separate comparison.
Small force residuals and positive determinants do not certify mechanical
stability; this experiment does not perform a 3D Hessian stability test. A finite
100-update budget and reduced objective likewise do not certify inverse
convergence; gradient and loss histories must accompany endpoint claims.
