# Selective skin energy and c020 pre-strain inverse findings

<!-- markdownlint-disable MD013 -->

## Meeting conclusion

At comparable target fit, this controlled 2x2 inverse supports both mechanisms:

- Selectively removing skin membrane energy from target-expanding triangles
  (`H0P0 -> H1P0`) reduced the contraction-region dihedral error by `25.790%`
  and the normal-residual Laplacian by `24.408%`. It also reduced the required
  muscle-activation RMS by `33.017%`, but increased the area-ratio error by
  `6.859%`.
- Adding c020 pre-strain to that selective material (`H1P0 -> H1P1`) reduced
  the dihedral error by another `50.070%`, the normal-residual Laplacian by
  `12.191%`, the area-ratio error by `35.663%`, and the activation RMS by
  `14.710%`.
- The combined `H1P1` case, relative to the corrected homogeneous/no-pre-strain
  baseline `H0P0`, retained the same target fidelity while reducing dihedral
  error by `62.947%`, normal-residual Laplacian by `33.623%`, area-ratio error
  by `31.250%`, and activation RMS by `42.870%`.

These results show that selective membrane softening changes the inverse
tradeoff and that c020 can substantially mitigate its faceting in this
target-specific inverse. They do **not** identify a physiological heterogeneous
Young's-modulus field: `H1` is an intentionally extreme, target-derived `E=0`
ablation.

**Decision:** c020 (`0.98` length factor) is already strong enough to
demonstrate the pre-strain smoothing mechanism. Do not run c050 (`0.95`) now.
The next issue is the visible H1 corrugation and late-trajectory conditioning,
not insufficient pre-strain magnitude.

## Controlled 2x2 design

`H` changes only the IsFace skin Young's modulus. `P` changes only the IsFace
skin pre-strain. The homogeneous modulus remains the earlier `0.2 MPa`; it was
not mean-matched to `H1`.

| Case | IsFace Young's modulus | IsFace pre-strain | Role |
| --- | --- | --- | --- |
| `H0P0` | Homogeneous `E=0.2 MPa` | `p000`, exact zero | Corrected baseline, reused by exact identity |
| `H0P1` | Homogeneous `E=0.2 MPa` | `c020` | Pre-strain main effect |
| `H1P0` | `E=0` iff raw `R>1`; otherwise `0.2 MPa` | `p000`, exact zero | Selective-softening main effect |
| `H1P1` | `E=0` iff raw `R>1`; otherwise `0.2 MPa` | `c020` | Combined candidate |

Here, `IsFace` is the anatomical exterior-face marker in the source mesh. The
mechanical skin contains only triangles whose three vertices all have
`IsFace=true`: `15,299` points and `29,899` triangles. It excludes the
artificial cross-section. This physical domain is distinct from
`SmileLossMask`, which selects the `15,302` points used by the inverse fit.

### Raw target/rest-area driver

The canonical IsFace triangles were mapped to the target driver using sorted
`GlobalPointId` triangle keys. No deadband, cap, diffusion, or fitted spatial
filter was used for `H1`:

```text
R = TargetArea / RestArea
E_H1 = 0 MPa       if R > 1
       0.2 MPa     otherwise
```

This gives `16,723 / 29,899` zero-energy triangles, representing
`54.55308228719783%` of IsFace rest area. There are `13,159` strict contraction
triangles (`R<1`) and `17` exactly unchanged triangles (`R=1`).

The c020 field is

```text
rho = 0.98^2 * clip(R, 0.5, 1)
ActivationInv = [rho^(-1/2) - 1, rho^(-1/2) - 1, 0]
```

Thus c020 means a `2%` in-plane length tightening, combined with the clipped raw
target/rest-area ratio. Its uniform natural-area multiplier is `0.9604`. Only
`31` triangles hit the `0.5` floor (`0.08793660414554653%` of rest area), and
the maximum stored skin `ActivationInv` component is `0.44307506364601545`.
The field is stored on every IsFace triangle in `P1`. In `H1P1`, however, it
produces no membrane force where `E=0`.

The audited material distribution is shown below. The native ParaView state is
[15-paraview-material-cases.pvsm](../data/15-paraview-material-cases.pvsm), and
the exact field statistics are in
[10-prepared-material-cases-manifest.json](../data/10-prepared-material-cases-manifest.json).

![Audited H0 and H1 Young's-modulus and c020 fields](../data/15-paraview-material-cases.png)

## Shared whole-anatomy and physics configuration

All four cases use the same anatomy, volumetric materials, target, boundary,
solvers, loss, and muscle parameterization.

| Component | Domain and constitutive setting | Material |
| --- | --- | --- |
| Fat | Stable Neo-Hookean tetrahedra | `E=0.003 MPa`, `nu=0.49` |
| Active muscle | Active Stable Neo-Hookean tetrahedra | `E=0.03 MPa`, `nu=0.49` |
| Aponeurosis | Stable Neo-Hookean tetrahedra | `E=0.1 MPa`, `nu=0.35` |
| Skin | IsFace Koiter membrane, fixed original `RestArea`, thickness `1 mm` | Case-dependent `E`, `nu=0.49` |

The volume materials retain the 3D isotropic conversion

```text
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu     = E / (2 * (1 + nu))
```

The thin skin uses the plane-stress conversion

```text
lambda = E * nu / (1 - nu^2)
mu     = E / (2 * (1 + nu))
```

The tetrahedral mixture fractions are already incorporated in each material's
`dV`; there is no separate tetrahedral `fraction` model field. The mesh has
`228,660` points and `1,146,517` tetrahedra. Muscle activation has six
independent symmetric `ActivationInv` components on each of `288,235` active
muscle tetrahedra, or `1,729,410` inverse parameters.

The loss is point-to-point mean squared displacement error on the `15,302`
`SmileLossMask` points, scaled by `10^6` to report `mm^2`. The prescribed smile
has target-displacement RMS `5.310139062299789 mm` and maximum
`15.39470968697625 mm`.

Every vertex incident to the artificial cross-section is hard-fixed to exact
zero displacement. This adds `6,980` cut vertices to the existing constraints;
the complete model has `33,636` fixed vertices (`100,908` fixed displacement
degrees of freedom). Every analyzed frame read back exact zero on the cut.

The forward equilibrium solver is PNCG with `atol=1e-10`, `rtol=5e-4`, and
`max_steps=5000`. The adjoint is `FallbackSolver(CupyCG, CupyMinRes)` with
`rtol=5e-4`.

## Cheap validation before the formal inverse

The preserved v1 smoke test stopped safely before any inverse solver work. All
three new cases raised `KeyError: 'fraction'` because its preflight incorrectly
required a tetrahedral `fraction` field. The anatomy builder had already folded
the mixture fractions into the material `dV` arrays.

The v2 smoke test replaced that check with exact `dV`, Lamé-field, and weighted
volume identities. It ran one evaluation and zero optimizer updates for each
new case. All three forward and adjoint solves succeeded; material readback was
exact; initial activation and displacement were exact zero; and the cut stayed
exactly fixed. The inverse gradient norms at step 0 were:

| Case | Step-0 inverse gradient norm |
| --- | ---: |
| `H0P1` | `0.12396447142984438` |
| `H1P1` | `0.1877295139008744` |
| `H1P0` | `0.19216930837099713` |

The failed v1 record is preserved in
[tmp/20-selective-skin-prestrain-smoke](../tmp/20-selective-skin-prestrain-smoke/),
and the passing v2 record is preserved in
[tmp/20-selective-skin-prestrain-smoke-v2](../tmp/20-selective-skin-prestrain-smoke-v2/).
The canonical v2 aggregate SHA-256 is
`f87ddb3321be5fedff431e063a29d985d916e5821645719867c47a353bc4be33`.

## Formal inverse protocol

- `H0P0` is the corrected 2026-08-18 baseline and was reused, not rerun.
- The new cases ran sequentially as `H0P1 -> H1P1 -> H1P0`.
- Each case constructed a fresh forward model, exact-zero muscle activation,
  exact-zero displacement seed, and fresh Adam optimizer. No activation,
  displacement, optimizer state, or equilibrium state was transferred.
- Adam used fixed learning rate `0.3`, exactly `40` optimizer updates, and `41`
  evaluated/saved frames (`0..40`) per case. There was no early stop based on a
  best frame.
- All `123` new frames were finite and complete, with successful forward and
  adjoint solves.
- The Cherries run used the debug/local profile: Comet was disabled and Git
  commit/verification were disabled. The run snapshot is
  `.cherries/runs/2026/08/19/human-face-smile-selective-skin-energy-prestrain-inverse/20-inverse-selective-skin-prestrain/2026-08-19T095148-Selective-skin-energy-and-c020-inverse-batch`.

The exact shell argv and `CHERRIES_TAGS` were not persisted by that local run,
so they are deliberately not reconstructed here. The executed entry point is
[20-inverse-selective-skin-prestrain.py](../src/20-inverse-selective-skin-prestrain.py),
and the complete live log is
[20-inverse-selective-skin-prestrain.log](../logs/20-inverse-selective-skin-prestrain.log).
Two early Local-plugin asset-copy errors in that log concern already archived
inputs and a not-yet-created snapshot log; the formal solver continued and all
final identity gates passed.

## Registered metrics and comparisons

- **Fit** is target-point displacement RMS in millimetres.
- **D** is the rest-edge-length-weighted RMS of deformed-minus-target dihedral
  angle on the `18,038` interior edges whose two incident canonical IsFace
  triangles both have strict raw `R<1`.
- **L** is the RMS graph Laplacian of target-normal displacement residual over
  all IsFace vertices, reported in millimetres.
- **Area** is the RestArea-weighted RMS of
  `deformed triangle area / target triangle area - 1`.
- **Activation** is RMS of the six-component symmetric muscle
  `ActivationInv`.
- Quality checks count non-SPD `I+ActivationInv`, inverted tetrahedra, and
  signed folded IsFace triangles. They are warnings, not automatic vetoes.

### Primary: registered baseline fidelity

The primary comparison uses the saved frame nearest to the `H0P0` step-40 fit,
whose target-error fraction is `0.5124062087322062` (`2.7209482247538275 mm`).
A case reaches the cohort only when its absolute normalized gap is at most
`0.01`. Selection uses saved frames only, without interpolation.

| Case | Step | Status | Fit (mm) | D (deg) | L (mm) | Area | Activation | non-SPD | Inverted | Folds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H0P0` | 40 | reached | 2.720948 | 13.327645 | 0.217061 | 0.140860 | 0.0609580 | 0 | 47 | 25 |
| `H0P1` | 40 | **did not reach** | 2.849029 | 5.579129 | 0.181487 | 0.072789 | 0.0548136 | 2 | 31 | 11 |
| `H1P0` | 14 | reached | 2.711934 | 9.890389 | 0.164081 | 0.150522 | 0.0408318 | 1 | 1 | 2 |
| `H1P1` | 12 | reached | 2.708135 | 4.938273 | 0.144078 | 0.096841 | 0.0348253 | 0 | 2 | 2 |

The registered effect sizes below are `(B/A - 1) * 100%`; negative values are
reductions.

| Contrast | Fit | D | L | Area | Activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `H0P0 -> H1P0` | -0.331% | -25.790% | -24.408% | +6.859% | -33.017% |
| `H1P0 -> H1P1` | -0.140% | -50.070% | -12.191% | -35.663% | -14.710% |
| `H0P0 -> H1P1` | -0.471% | -62.947% | -33.623% | -31.250% | -42.870% |

`H0P1` did not reach the registered `H0P0` fidelity within 40 updates. Its
step-40 values are descriptive and must not be used as a primary matched-fit
pre-strain contrast.

### Secondary: common tau

For a four-case check that includes `H0P1`, the secondary threshold is the
worst per-case minimum target-error fraction:
`tau=0.5365261789328208`. Again, the nearest saved frame is used without
interpolation.

| Case | Step | Fit (mm) | D (deg) | L (mm) | Area | Activation | non-SPD | Inverted | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H0P0` | 35 | 2.842036 | 12.638810 | 0.215214 | 0.140237 | 0.0571736 | 0 | 39 | 19 |
| `H0P1` | 40 | 2.849029 | 5.579129 | 0.181487 | 0.072789 | 0.0548136 | 2 | 31 | 11 |
| `H1P0` | 13 | 2.830915 | 9.404478 | 0.165645 | 0.149298 | 0.0387800 | 0 | 2 | 1 |
| `H1P1` | 11 | 2.830054 | 4.786367 | 0.147630 | 0.094721 | 0.0327147 | 0 | 2 | 1 |

| Secondary contrast | Fit | D | L | Area | Activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `H0P0 -> H1P0` | -0.391% | -25.590% | -23.033% | +6.461% | -32.171% |
| `H1P0 -> H1P1` | -0.030% | -49.105% | -10.875% | -36.556% | -15.640% |
| `H0P0 -> H1P1` | -0.422% | -62.130% | -31.403% | -32.456% | -42.780% |
| `H0P0 -> H0P1` | +0.246% | -55.857% | -15.671% | -48.096% | -4.128% |

This secondary cohort agrees with the direction of the registered conclusions,
including a large c020 improvement. It remains secondary because its threshold
was defined after observing the four trajectory minima.

### Terminal frames: trajectory evidence only

| Case | Step | Fit (mm) | D (deg) | L (mm) | Area | Activation | non-SPD | Inverted | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H0P0` | 40 | 2.720948 | 13.327645 | 0.217061 | 0.140860 | 0.0609580 | 0 | 47 | 25 |
| `H0P1` | 40 | 2.849029 | 5.579129 | 0.181487 | 0.072789 | 0.0548136 | 2 | 31 | 11 |
| `H1P1` | 40 | 1.414208 | 7.691628 | 0.161078 | 0.121580 | 0.0628549 | 54 | 50 | 16 |
| `H1P0` | 40 | 1.437911 | 15.571349 | 0.198196 | 0.168318 | 0.0672742 | 60 | 41 | 39 |

The terminal `H1` cases achieve substantially lower pointwise error, but they
are not matched to the baseline and are conditioning-confounded. Continuing the
fixed-budget optimization creates non-SPD activation tensors and more geometric
defects; `H1P0` also becomes rougher. These frames are useful trajectory
evidence, not the headline material comparison.

The full trajectories and Pareto summary are available as
[30-selective-skin-prestrain-trajectories.png](../data/30-selective-skin-prestrain-trajectories.png)
and
[30-selective-skin-prestrain-pareto.png](../data/30-selective-skin-prestrain-pareto.png).

## Native ParaView review

The meeting panels were generated by `/usr/bin/pvbatch`, ParaView `6.1.1`, at
`4000 x 3000`, with shared front, 30-degree, mouth, and eye/cheek views. The
normal-residual panels use one shared `+/-4.127603436 mm` colour scale. Matplotlib
was used only for trajectories and Pareto plots, not for geometry.

The manifest persists the exact absolute argv. Its equivalent repo-relative
form is:

```text
/usr/bin/pvbatch src/35-render-selective-skin-prestrain-paraview.py \
  --analysis data/30-selective-skin-prestrain-analysis.json \
  --input-root data/30-paraview-inputs \
  --output-dir data/35-paraview-results
```

The command above is shown repo-relative for readability; the manifest stores
the corresponding absolute paths. All 12 PNG/PVSM outputs passed existence,
dimension, signature, and SHA-256 gates.

### Primary baseline-fidelity cohort

[Open geometry state](../data/35-paraview-results/35-paraview-baseline-fidelity-geometry.pvsm)
or
[open residual state](../data/35-paraview-results/35-paraview-baseline-fidelity-normal-residual.pvsm).

![Primary baseline-fidelity geometry](../data/35-paraview-results/35-paraview-baseline-fidelity-geometry.png)

![Primary baseline-fidelity target-normal residual](../data/35-paraview-results/35-paraview-baseline-fidelity-normal-residual.png)

### Secondary common-tau cohort

[Open geometry state](../data/35-paraview-results/35-paraview-common-tau-geometry.pvsm)
or
[open residual state](../data/35-paraview-results/35-paraview-common-tau-normal-residual.pvsm).

![Secondary common-tau geometry](../data/35-paraview-results/35-paraview-common-tau-geometry.png)

![Secondary common-tau target-normal residual](../data/35-paraview-results/35-paraview-common-tau-normal-residual.png)

### Terminal cohort

[Open geometry state](../data/35-paraview-results/35-paraview-terminal-geometry.pvsm)
or
[open residual state](../data/35-paraview-results/35-paraview-terminal-normal-residual.pvsm).

![Terminal geometry](../data/35-paraview-results/35-paraview-terminal-geometry.png)

![Terminal target-normal residual](../data/35-paraview-results/35-paraview-terminal-normal-residual.png)

### Visual conclusions

- At primary and common-tau fidelity, `H1P0` lowers the aggregate D and L
  metrics and reaches comparable fit earlier, but ParaView still shows genuine
  localized high-frequency corrugation around the mouth and cheek. It is not an
  unconditional visual smoothness improvement, and its area-ratio metric is
  worse.
- `H1P1` is visibly and metrically better regularized than `H1P0` while
  retaining the matched target fit. Pre-strain mitigates the H1 corrugation but
  does not eliminate it; lip rims and eye contours remain jagged.
- `H0P1` at step 40 is visually the smoothest candidate, consistent with its
  secondary metrics, but it did not reach the registered primary fidelity.
- Every cohort retains structured negative cheek/chin residuals and positive
  mouth-corner bands. Neither mechanism eliminates the spatially structured
  target mismatch.
- The terminal H1 panels visibly show the stronger mouth deformation and later
  roughening that make terminal frames unsuitable as the causal comparison.

The render log contains repeated OpenGL texture-cleanup warnings caused by
session resets. `pvbatch` exited successfully and every final render gate
passed; these are implementation warnings, not altered scientific outputs. See
[36-paraview-render-manifest.json](../data/36-paraview-render-manifest.json) for
all image/state identities.

## Interpretation limits and warnings

- `H1` is a nonphysiological ablation: it removes membrane energy exactly on
  target-expanding triangles. It does not estimate a continuous or transferable
  Young's-modulus distribution, and its mean modulus was intentionally not
  matched to `H0`.
- Both `R` and c020 are derived from the same target used by the inverse. This
  demonstrates mechanism effectiveness for this target; it is not independent
  validation or evidence that the field generalizes to other expressions.
- `I+ActivationInv` is unconstrained. Non-SPD counts, inverted tetrahedra, and
  folded skin triangles occur even in some matched cohorts and become severe in
  the terminal H1 frames. The terminal H1 solutions are therefore not
  physically acceptable final reconstructions.
- The experiment used a fixed 40-update Adam budget and no activation
  regularizer, convergence-based stopping rule, or random repeat. Pointwise fit
  alone can hide spatial differences.
- The hard-fixed cross-section is the user-approved conservative approximation,
  not measured boundary motion or anatomical ground truth.
- The IsFace skin term is an isotropic Koiter membrane. This experiment adds no
  skin bending term or fiber anisotropy.
- In `H1P1`, c020 is mechanically inert on every `E=0` triangle. The combined
  result therefore does not test pre-strain on those expanding triangles.
- The deterministic `H0P0` history is reused by exact artifact identity rather
  than rerun in this batch.
- No case is artifact-free; the remaining lip/eye contour jaggedness and
  structured residual fields should be shown with the scalar metrics.

## Preserved erroneous 3D-Lame cohort

The earlier result that incorrectly applied the 3D Lamé conversion to the skin
has been retained for the group meeting. It is **not** part of this 2x2 cohort.

- Historical result:
  [20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen.vtu](../../../17/human-face-smile-material-heuristic-sweep/data/20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen.vtu),
  SHA-256
  `0596f3dcf378f745d80533ac6bd7c0c3f289846e6320e761ef5e10d899e556d5`.
- The later forward audit records it as `historical-full-3d`, a
  `reused-pinned-control` on the historical full boundary, valid only as the old
  model rather than anatomical thin skin:
  [15-forward-domain-conversion-probe-summary.json](../../../18/human-face-smile-plane-stress-skin/data/15-forward-domain-conversion-probe-summary.json).
- Its previous meeting interpretation is preserved in
  [30-corrected-baseline-screen.md](../../../18/human-face-smile-plane-stress-skin/docs/30-corrected-baseline-screen.md).

## Appendix: provenance and operational record

The canonical machine-readable results are
[20-selective-skin-prestrain-inverse-summary-final.json](../data/20-selective-skin-prestrain-inverse-summary-final.json)
and
[30-selective-skin-prestrain-analysis.json](../data/30-selective-skin-prestrain-analysis.json).

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Material manifest | 34,699 | `e436d7d0a1da519b76d6a495b70a75c5c725cf6de346c298aec720cd9de9701e` |
| Formal canonical aggregate | 387,036 | `cf533bb16f481d75587531dfcd5aa21ed1065ed02539ea3ff0290e94d6cd2de6` |
| Audited analysis | 284,873 | `120b03b02cec7e30dc4ecabb8d3ac8197168a347d405bacd78dceb7f8af2d520` |
| ParaView render manifest | 6,266 | `a1c182340d16c16a94b56672770f5c2319f8bc35aaf3c48f7851fb42c0bc18b8` |
| Executed inverse producer | 79,536 | `deece64950f8bf21984fa0ba970d2e1f0e0f71e23db483919bd59de47052456b` |
| Executed analyzer | 66,557 | `d3225740992d57edfc852026416fe11c1bd4ab94c13c955debb4323c7c280548` |
| Executed ParaView renderer | 11,140 | `3cd737c45f377c6cee0ebc2990a41e9112baf5e379d98f3d20e0f2fbd323e737` |
| ParaView wrapper | 10,645 | `b0be24c3bfd7c118c0160fcaaffbed5f52105e22764505d17c9782899cafb632` |

The baseline identity is pinned in the material manifest: its corrected IsFace
skin SHA-256 is
`4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`,
and its reused inverse history SHA-256 is
`6e29d7b205e7901681942f0d413b091c5e4bce003ec4d789c2d7f69ded430d24`.
The raw area-ratio array SHA-256 is
`da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606`.

During review, the Codex desktop `codex-workspace-diff` helper created a
temporary Git index containing the large untracked VTK outputs and passed them
through Git LFS. This was not caused by Cherries. It produced approximately
`125 GiB` of new/orphan LFS objects and approximately `9 GiB` in
`.git/lfs/tmp` (`9,217,222,656` bytes at the audited checkpoint). The active
temporary diff process was terminated, and local
`.git/info/exclude` shields were added for the generated VTKHDF/VTU/VTP output
patterns. The real `.git/index` remained unchanged; its audited SHA-256 is
`509905510308d752e1a8efed2f7ca1bdea5dd8a306267ee117746e473259f073`,
and no files were staged.

No experiment file or LFS object was deleted or moved, and no LFS cleanup was
run. Any later cleanup remains a separate operation requiring explicit approval.
