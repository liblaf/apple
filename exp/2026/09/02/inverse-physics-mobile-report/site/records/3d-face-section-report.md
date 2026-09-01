# Human-face muscle section: folding receipt and limits of interpretation

## Conclusion

The primary no-skin `lr3` endpoint contains the same constitutive folding route
seen in the local material history: an activation-map orientation reversal
(`DetAinv < 0`) appears first, followed by a deformation orientation reversal
(`DetF < 0`), so the two negative determinants make `DetG = DetF * DetAinv`
positive again.  This is direct post-hoc evidence that this route is present in
the visibly bumpy primary endpoint.  It does **not** establish that it is the
sole cause of visible bumpiness: the historical smoother skin-estimated-plus-
tightening comparator has more whole-active and zygomaticus inversions at its
best endpoint.

No constraint, repair, or simulation rerun was applied for this analysis.  The
figures and receipts are observations of the saved materialized endpoints and
history.  The saved face solves used independent six-DoF activation for each of
288,235 active muscle tetrahedra (1,729,410 activation DoF), with
`activation/range_clamping=false` and `activation/shared=false`.

## What the determinants mean

For each tetrahedron, `F` maps its reference edge matrix to its deformed edge
matrix, so `DetF < 0` is a flipped deformation.  `Ainv` is reconstructed exactly
from the six stored components in the order `[xx, yy, zz, xy, yz, xz]` as
`I + symmetric(ActivationInv)`.  `DetAinv < 0` is an activation-map orientation
reversal.  The elastic part is `G = F Ainv`, hence `DetG = DetF * DetAinv`.
`DoubleInverted` is strictly `DetF < 0 && DetAinv < 0`; it can have positive
`DetG`, so determinant signs must be read together rather than treating every
negative `DetF` as the same material state.

## Exact 31-cell id64 material slab

The slab is deterministic: primary active muscle id64 with `MuscleFraction >=
0.5`, PCA axes from its reference tetra centroids, and full longitudinal PCA
axis 0 with transverse half-widths of 15% of the 5th--95th percentile spans
through the primary minimum-`DetF` tetrahedron.  The same 31 source cell IDs
are used for the comparator.  The static endpoints below are the saved **best**
meshes, not their later final optimizer states: the primary best step is 194
and the comparator best step is 192.

| case | min DetF | min DetAinv | min DetG | F-negative | Ainv-negative | G-negative | F&A double-inverted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary no-skin lr3 | -1.273270 | -0.408833 | 0.424545 | 1 / 3.6489% | 1 / 3.6489% | 0 / 0% | 1 / 3.6489% |
| comparator skin-estimated-plus-tightening lr1 | -0.605360 | -1.345951 | 0.061297 | 1 / 4.4697% | 1 / 4.4697% | 0 / 0% | 1 / 4.4697% |

The percentages are rest-volume fractions; the common slab rest volume is
`2.1318705758442154e-08`.

The per-step receipt has 201 consecutive frames (0--200).  `DetAinv < 0` and
`DetG < 0` first occur at step 91.  `DetF < 0` and the double-inverted state
first occur at step 105 and persist through step 200; `DetG < 0` ends at step
104.  The exact step-194 confirmation is one F-negative, one Ainv-negative,
one double-inverted, zero G-negative cells, each nonzero fraction being
3.648901996430297%.  Its minima are `DetF=-1.2732700831380714`,
`DetAinv=-0.40883252668457987`, and `DetG=0.4245445942264014`.

## Whole active muscle and broader zygomaticus context

`active_muscle` means `ActivationMask`.  The broader zygomaticus selection is
`ActivationMask && MuscleId in {63,64}` with no fraction threshold.

| case | active F-negative cells / rest volume | zygomaticus F-negative cells / rest volume | zygomaticus double-inverted cells | forward failures |
| --- | ---: | ---: | ---: | ---: |
| primary no-skin lr3 | 110 / 0.04161% | 12 / 0.43918% | 4 | 6 |
| comparator skin-estimated-plus-tightening lr1 | 974 / 0.35290% | 41 / 1.58822% | 24 | 2 |

Thus the comparator is only a visual/contextual comparison, not a controlled
causal counterfactual: it is historically smoother in the provided comparison
but has more measured inversions in both wider sets.

All six source inverse runs report `inverse/converged=false`; their materialized
best checkpoints therefore cannot support a claim of converged optimum or
stationarity.  Their forward-failure counts are 155 and 6 for the two no-skin
runs, 2 and 0 for the skin-estimated-plus-tightening run and its continuation,
and 1 and 0 for the skin-no-prestrain run and its continuation.  The primary
had 6 and comparator 2.  Together with the differing skin/prestrain and
optimizer histories, this keeps the result at post-hoc mechanistic evidence
rather than controlled causality.

## Artifacts

- [Static face context](../data/20-face-muscle-section-render/face-context-id64.png)
- [Matched primary/comparator section](../data/20-face-muscle-section-render/matched-section-comparison.png)
- [Primary mechanism view](../data/20-face-muscle-section-render/primary-mechanism.png)
- [Historical bumpy primary surface context (non-causal)](../../../../06/17/human-face-smile-prestrain-v2/figs/20-human-face-smile-no-skin-lr3.png)
- [Historical smoother comparator surface context (non-causal)](../../../../06/17/human-face-smile-prestrain-v2/figs/20-human-face-smile-skin-estimated-plus-tightening-lr1.png)
- [History VTK series](../data/15-face-muscle-section-history/history.vtu.series)
- [History export receipt](../data/15-face-muscle-section-history/receipt.json)
- [Per-step determinant CSV](../data/20-face-muscle-section-history-analysis/trajectory.csv)
- [Per-step onset/persistence receipt](../data/20-face-muscle-section-history-analysis/summary.json)
- [Static endpoint receipt](../data/10-face-muscle-section/summary.json)
- [Exact 201-step 30 FPS mechanism video](../data/25-face-muscle-section-evolution/face-muscle-section-evolution.mp4)
- [Evolution render receipt](../data/25-face-muscle-section-evolution/render-receipt.json)
