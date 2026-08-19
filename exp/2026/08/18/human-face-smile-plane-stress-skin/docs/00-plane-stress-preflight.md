# Plane-stress facial membrane preflight

## Status

**The reviewed preparation, forward probe, zero-step smoke, one approved
40-update corrected inverse, and its strict analysis are complete.** Scripts
`15` and `16` ran 10 fixed-activation forward equilibria and the corresponding
static analysis. Script `10` produced the corrected schema-v3 manifest and
29,899-triangle `IsFace` skin. Script `20` then passed the isolated smoke and
the separately approved 40-update / 41-evaluation formal baseline; script `30`
validated all saved frames and produced the final metric and visual analysis.
See the [forward-probe report](15-forward-domain-conversion-probe.md),
[zero-step smoke report](20-hard-fixed-zero-step-smoke.md), and
[formal baseline report](30-corrected-baseline-screen.md).

The probe confirmed the corrected implementation and showed no visible new
artifact from hard-fixing the artificial cut, but all five setups were strongly
sensitive to the initial displacement branch. The user selected the hard-fixed
cut as a conservative approximation despite its small metric penalty. The live
smoke read back all 6,980 cut vertices at exact zero displacement, with 33,636
fixed vertices / 100,908 fixed DoFs, and both forward and adjoint succeeded.
The smoke aggregate is schema v4, `complete=true`, has no hard failures, and is
confined to `tmp/`. The later formal screen also completed without numerical
hard failures. It reduced target RMS error to `51.24%` of target RMS, but its
surface remained visibly Bumpy; no second inverse is automatically approved.

The constitutive correction itself is clear. If `E` and `nu` are 3D material
constants for a thin membrane under zero transverse stress, skin must use

```text
lambda_2d = E * nu / (1 - nu**2)
mu        = E / (2 * (1 + nu))
```

Volume fat, muscle, and aponeurosis must continue to use the 3D conversion.
For `E=0.2 MPa, nu=0.49`, skin `lambda` changes from `3.288590604` to
`0.128964337 MPa`; `mu=0.067114094 MPa` is unchanged. The old area modulus
was `17.114x` the plane-stress value.

## Verified core implementation

The repository now has a separate plane-stress converter. Koiter remains a
material-agnostic metric membrane and does not perform a hidden conversion or
multiply the modulus by thickness. Its input `Lambda` and `Mu` are effective
in-plane moduli, and its energy weight applies the configured thickness once.

Focused tests cover:

- the old 3D converter and the new plane-stress converter at `nu=0.49`;
- vector-valued conversion and the reduced-lambda identity;
- homogeneous triangle patch energy including thickness and `Fraction`;
- the exact zero-energy state of an isotropically prestrained triangle;
- equal energy for equal elastic strain with and without prestrain, proving
  that `ActivationInv` does not change the original reference-area weight;
- existing Koiter energy, gradient, and Hessian finite-difference checks.

The focused suite passes `9/9` tests.

## Authoritative anatomy-construction provenance

The membrane-domain decision is traced to the upstream Melon construction,
not inferred only from the final geometry:

- `22-register-skin.py` registers the PolyGroup-labelled skin surface to the
  anatomical skin, eye, and gingiva geometry;
- `42-gen-masks.py` defines `IsFace` from exactly ten named groups: chin,
  face, outer eyelids, and outer lips;
- the same script builds `InFaceConvex` by taking the convex hull of the
  `IsFace` points and selecting tetrahedron centers inside it;
- `61-delta-transfer.py` and `62-disp-transfer.py` subsequently transfer the
  expression displacements. They do not redefine the anatomy masks.

Therefore `InFaceConvex` is a simulation-volume crop, not an epidermis label.
Its newly exposed boundary must not automatically receive Koiter energy.
`IsTeeth`, `IsGingiva`, and `IsLip` are proximity masks used to subtract the
oral/lip neighborhood from the initial Cranium/Mandible `IsFixed` mask. They
are not group identities and cannot veto otherwise valid `LipTop` or
`LipBottom` membrane triangles. `IsLip`, which includes inner and outer lip
groups, is the most direct guard against accidentally fixing the lip region;
`IsTeeth` supplies an overlapping teeth-neighborhood guard.
The corrected producer pins the Melon mask source and checks the live
`GroupName`/`GroupId` values against the exact upstream face-group allowlist.

## Blocking finding: the historical membrane domain is not skin

The simulation volume is an `InFaceConvex` subset of a 3.19-million-tet source
head. Historical builders apply Koiter to every triangle returned by
`extract_surface()` on that subset. This closes not only the facial exterior,
but also the artificial boundary introduced by the subset operation.

Triangles were mapped back to the source mesh with `vtkOriginalPointIds` and
classified by exact sorted source-face identity:

| subset-boundary class | triangles | area, m2 |
| --- | ---: | ---: |
| complete extracted boundary | 128,172 | 0.142046989 |
| existed on source-mesh boundary | 115,007 | 0.120724290 |
| introduced by `InFaceConvex` cut | 13,165 | 0.021322699 |
| all-vertex `IsFace` | 29,899 | 0.042879981 |

All `13,165` artificial-cut triangles are movable: none is all-fixed, and none
is `IsFace`. Historical skin VTPs nevertheless assign them `Fraction=1` and
`E=0.2 MPa`. The source boundary also contains fixed cranium/mandible, teeth,
gingiva, mouth and eye cavities, so merely excluding the new cut is not enough.

The current time-limited correction should therefore use **all-vertex
`IsFace` as a facial-ROI membrane**, not claim to model the entire anatomical
epidermis. This domain is one connected component, has no artificial-cut or
all-fixed triangles, and exactly matches the material-driver triangle domain.
A later domain-sensitivity experiment can construct a separately audited
anatomical outer-skin mask.

The prepared mesh has `15,310` `IsFace` points. Eight have a non-finite Smile
target, so the pointwise definition `SmileLossMask = IsFace & finite(Smile)`
contains `15,302` points. The filtered membrane instead selects triangles for
which all three vertices are `IsFace`; it has `29,899` triangles and `15,299`
vertices. The remaining three finite loss points lie at Face/HeadBack or
EyelidTop/EyelidInnerTop group seams, where every adjacent triangle crosses
outside `IsFace`. They remain in the point loss and volume model but receive
no membrane energy. Their contribution is only `1.88e-6` of the target squared
norm, so changing the historical loss mask for exact set equality is neither
necessary nor desirable. The material driver is defined on exactly the same
triangle domain as the filtered membrane.

`IsTeeth` is a 2 mm proximity mask and overlaps 52 of these lip triangles; it
is not an anatomical group identity. Their authoritative `GroupId` values are
only `LipTop` and `LipBottom`, so they remain in the facial membrane. Across
the complete `IsFace` domain, all point group identities belong to the ten
upstream face/eyelid/lip/chin groups and none belongs to teeth, gingiva, bone,
or socket groups.

![Skin-domain audit](../tmp/preflight/skin-domain-overlay.png)

## Artificial-cut volume boundary condition

Removing the artificial cross-section from Koiter and constraining the volume
at that cross-section are separate operations. The former is mandatory: the
cut is not epidermis. The latter approximates support from the omitted head
tissue.

The `13,165` cut triangles touch `6,980` vertices. Only `380` are already fixed
by the Cranium mask; `6,600` are free. None belongs to `IsFace` or
`SmileLossMask`, although the closest cut vertex is about `1.34 mm` from a
loss point. The current model therefore has a mostly traction-free artificial
cut with a small clamped cranium seam.

Automatically hard-fixing the whole cut is not known to be correct: the convex
cut is neither a physical attachment surface nor a symmetry plane, and a zero
Dirichlet boundary can over-constrain facial motion. The ideal alternatives are
a larger volume or a calibrated elastic/Robin support from the omitted tissue.
For this time-limited study, hard-fixed and current-cut boundary conditions were
evaluated as two cheap bracketing models. The hard-fixed case changed facial
displacement by `5.75%--6.36%` of target RMS across the two seeds, slightly
worsened target error and both roughness metrics, but introduced no visible new
artifact in the standardized views. The user subsequently selected hard-fixing
all `6,980` cut-incident vertices as a conservative support approximation. This
is a declared modeling choice, not an inferred anatomical ground truth; a
larger volume or calibrated Robin support remains the more physical follow-up.

## Prestrain reference measure

The historical Koiter implementation used

```text
natural area = original mesh area / det(Ainv)
energy weight = thickness * Fraction * natural area
```

so `p100/p200` changed both the stress-free metric and the amount of membrane
energy assigned to a triangle. That convention was inconsistent with this
project's 3D active-strain tetrahedra, which evaluate `G = F Ainv` while
integrating over the unchanged original tetrahedron volume.

The corrected Koiter implementation now follows the same fixed-reference
convention:

```text
stress-free metric inverse = Ainv * original_metric_inverse * Ainv^T
energy weight = thickness * Fraction * original mesh area
```

Prestrain therefore changes only the natural metric. It does not change the
triangle's reference area, thickness, or represented material amount. A
regression patch test compares equal elastic strains with and without
prestrain and requires equal energies.

The first corrected inverse still uses `p000`, but now to isolate the two
already identified corrections (plane-stress Lamé conversion and audited
`IsFace` membrane domain), not because the reference-measure convention is
unresolved. Historical prestrained results remain preserved and are labeled
as using both the old 3D-Lamé/full-boundary model and the old
activation-dependent energy weight.

## Historical artifacts retained for reporting

The correction is a new experiment lineage and must not overwrite the old
outputs. The reportable historical cohorts are:

- six fresh-zero 40-update material cases under
  `exp/2026/08/17/human-face-smile-material-heuristic-sweep`;
- three fresh-zero 40-update exaggerated cases under
  `exp/2026/08/18/human-face-smile-exaggerated-material-screen`;
- the June 17 legacy and continuation runs, kept as a separate archival
  cohort rather than mixed with the fresh-zero screen;
- `no-skin` controls, whose mechanics are unaffected by either skin fix.

Every old skin row must be labeled
`Historical full-boundary + 3D Lamé (superseded; mechanism-only)` in the group
meeting. The August 18 exaggerated histories are currently ignored/untracked
local artifacts, so cleanup commands that remove ignored files are forbidden
until they are intentionally archived or tracked with Git LFS. Their existing
paths, summaries, plots, and temporal histories remain immutable; corrected
outputs use only this group's separate `data/` namespace.

The three at-risk August 18 history SHA-256 identities are pinned here:

- `e100-p200`:
  `df5f7cfb32f041e7886c391a00fd90075aa2347b0adc8bea9a8e5ceaa56c1ae0`;
- `e005-p000`:
  `2bce68976b55a990784456975a84dd6e77afaeb8a0425d5563cd846caf532141`;
- `e005-p200`:
  `71fa0356b6dcda32e63e6191001b19731270c2c3e443c967deeaec3df60e89f2`.

The aggregate summary is
`d6aae85cda6ae45a876e8139e661d17aec43645112bc90fe3ee6c7cda5cd8a5b`; the
analysis JSON is
`61246eec4796e465a81e8cd9d29dc58237e3867117231e8677d73f5eaa44d423`.

## Why the old `e005` is not the next priority

`e005` is a minimum-scale parameter, not the actual minimum modulus over a
large region. Because `ExpansionWeight.max()=0.709369`, its realized minimum is
`0.0238847 MPa`; the full-surface area-weighted mean changes by only about
`2.6%`. Failure of this case would not demonstrate that Young-modulus
softening is ineffective.

If the corrected baseline remains too far from target, a subsequent strong-E
experiment should use an explicit support and actual modulus, such as uniform
`E=0.003 MPa` on the audited expansion support. It should not reuse the
misleading `e005` label as an extreme-softness claim.

## Recommended sequential experiment

### A. Cheap fixed-activation forward diagnostic — completed

Use the same historical baseline activation to evaluate the domain/conversion
factorial without re-optimizing:

1. full extracted boundary + historical 3D lambda (existing result);
2. full extracted boundary + plane-stress lambda;
3. `IsFace` membrane + historical 3D lambda;
4. `IsFace` membrane + plane-stress lambda;
5. setup 4 with every artificial-cut-incident vertex hard-fixed, used only as
   a boundary-condition bracket.

Replay all five setups from both zero displacement and the historical
displacement: ten equilibrium solves in total. This makes the historical
full-boundary/3D setup a same-code replay rather than an unmatched external
result, while the fifth setup isolates the cut-boundary approximation. The two
seeds are a branch-sensitivity test. These are causal forward probes only;
transferred activation is not called a recovered activation for the new setup.

The completed probe reproduced the historical control, but all five zero/old
setup pairs differed by `19.0%--24.9%` of target RMS. Full results and views are
in [the probe report](15-forward-domain-conversion-probe.md). This branch
sensitivity prevents a single transferred-activation seed from ranking the
setups; it does not itself prove that a fresh-zero inverse is invalid, because
that inverse starts at zero activation and follows successive equilibria.

### B. One expensive corrected inverse — completed

If the forward diagnostic is stable, run only:

```text
skin domain: all-vertex IsFace
conversion: plane stress
cut boundary: all 6,980 artificial-cut-incident vertices fixed to zero
E: homogeneous 0.2 MPa
prestrain: none
activation: fresh zero, per active tet unconstrained 6-DoF
optimizer: Adam, LR 0.3
budget: 40 updates / 41 evaluations
```

This is a fixed-budget screen, not a convergence or modeling-upper-bound
claim. Historical no-skin remains a hash-bound external control because neither
the domain nor conversion change affects it.

The producer and schema-v3 single-case analyzer bind the selected hard-fixed
boundary, exact cut topology/GlobalPointId digests, artifact identities, and
fixed-reference Koiter implementation. After separate approval, the formal
screen and analyzer completed. The corrected terminal target error is
`0.512406` of target RMS, but at matched fidelity its dihedral and
residual-normal roughness are respectively `2.24x` and `1.65x` the no-skin
control. See the [formal baseline report](30-corrected-baseline-screen.md).

### C. Conditional second inverse

The corrected trajectory and views have now been reviewed:

- target fit improves, but dihedral Bumpy grows strongly with optimization;
- extending the same `p000` trajectory is therefore not the next priority;
- the highest-information candidate is a corrected fixed-reference prestrain
  dose, selected only after a cheap `p100/p200` forward and branch probe;
- no second inverse has been approved or started.

## Required acceptance evidence

Preparation passed: the live Koiter input has exactly `29,899` triangles, area
`0.042879981 m2`, one component, zero artificial-cut overlap, zero fixed or
cavity/bone `GroupId` overlap, and exact `GlobalPointId` mapping. Lambda, Mu,
E, Fraction, topology, and source identities survived file readback. The
[manifest](../data/10-corrected-baseline-manifest.json) SHA-256 is
`d999be4fc941253b8daa84dca4a52ab44bd02b3e42ce2af6d151a2e14b64a21a`;
the [filtered skin](../data/10-corrected-baseline/skin-isface-e0200-p000.vtp)
SHA-256 is
`4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`.

The zero-step smoke also passed its integration gate: one finite evaluation,
zero optimizer updates, successful forward and adjoint, no hard failures, and
exact-zero readback on all cut vertices. These checks establish executable
wiring only; they do not satisfy or waive the separate approval required for
the formal 41-frame screen and analyzer.

Inverse execution requires fresh optimizer state, exact zero activation, 41
finite frames, and successful forward/adjoint solves. Small inversion/fold
counts remain warning-only. Scientific comparison uses target RMS plus an
area-weighted face error, target-relative contraction dihedral, residual-normal
high-frequency roughness, activation diagnostics, and standardized front,
three-quarter, mouth, and eye-cheek views.
