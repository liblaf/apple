# Forward Skin Prestrain Sanity Check

> **Historical-model result (superseded as a thin-skin baseline, 2026-08-18).**
> The artifacts are retained unchanged. This sanity check used the historical
> full-boundary/3D-Lamé Koiter model and, for nonzero prestrain, the historical
> activation-dependent energy weight. It remains useful for provenance and
> group-meeting comparison, not as validation of the corrected membrane.

## Purpose

Check the Smile-derived skin shrink prestrain before continuing inverse optimization. The revised goal adds a fourth case with uniform `c = 0.02` tightening on `IsFace` triangles, so this sanity pass compares:

- `skin-estimated-prestrain`
- `skin-estimated-plus-tightening`

Both use the `InFaceConvex` simulation subset, Smile target displacement, fixed `IsFixed` boundary, Koiter skin, and zero volume prestrain.

## Commands

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-smile-prestrain-v2
```

Prepare mesh and static skin prestrain surfaces:

```bash
DEBUG=1 CHERRIES_NAME="Human face Smile prestrain v2 prepare plus" CHERRIES_TAGS="human-face,smile,skin-prestrain,prepare,plus-tightening" uv run python src/10-prepare-human-face.py
```

Estimated prestrain forward:

```bash
DEBUG=1 CHERRIES_NAME="Human face Smile estimated skin prestrain forward v2" CHERRIES_TAGS="human-face,smile,skin-prestrain,forward-check,estimated" uv run python src/15-forward-estimated-skin-prestrain.py
```

Estimated plus tightening forward:

```bash
DEBUG=1 CHERRIES_NAME="Human face Smile estimated plus tightening forward" CHERRIES_TAGS="human-face,smile,skin-prestrain,forward-check,plus-tightening" uv run python src/15-forward-estimated-skin-prestrain.py --setup skin-estimated-plus-tightening --output-mesh data/16-estimated-plus-tightening-forward.vtu --output-target data/16-estimated-plus-tightening-target.vtu --output-skin data/16-estimated-plus-tightening-skin.vtp --output-skin-isface data/16-estimated-plus-tightening-isface-skin.vtp --output-summary data/16-estimated-plus-tightening-forward-summary.json
```

## Outputs

Static prestrain surfaces:

- `data/10-smile-isface-skin-estimated-prestrain.vtp`
- `data/10-smile-isface-skin-estimated-plus-tightening.vtp`

Forward visualization outputs:

- `data/15-estimated-skin-prestrain-forward.vtu`
- `data/15-estimated-skin-prestrain-target.vtu`
- `data/15-estimated-skin-prestrain-skin.vtp`
- `data/15-estimated-skin-prestrain-isface-skin.vtp`
- `data/15-estimated-skin-prestrain-forward-summary.json`
- `data/16-estimated-plus-tightening-forward.vtu`
- `data/16-estimated-plus-tightening-target.vtu`
- `data/16-estimated-plus-tightening-skin.vtp`
- `data/16-estimated-plus-tightening-isface-skin.vtp`
- `data/16-estimated-plus-tightening-forward-summary.json`

## Readback

PyVista readback succeeded for all static `.vtp`, forward `.vtu`, target `.vtu`, and skin `.vtp` files listed above.

Each forward volume has `TargetDisplacement`, `TargetPoint`, `LossMask`, `Displacement`, `DeformedPoint`, `SkinPrestrainDisplacement`, and `DisplacementErrorNorm`.

Each skin `.vtp` has `ActivationInv`, `TargetRestAreaRatio`, `LogTargetRestAreaRatio`, `TargetDerivedActivePrestrainMask`, `ConstantTighteningValue`, `ConstantTighteningInvStretch`, `StressFreeAreaRatio`, and `TotalInvStretch`.

## Metrics

| setup | forward | steps | rel grad | active target-derived tris | constant-tightening tris | max ActivationInv diag | min stress-free area ratio | IsFace area deformed/rest | active area deformed/rest | loss-mask disp RMS | loss-mask residual RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skin-estimated-prestrain | primary_success | 2380 | 0.000493063 | 13159 | 0 | 2.16228 | 0.1 | 0.966795 | 0.925233 | 0.00171279 | 0.0047994 |
| skin-estimated-plus-tightening | primary_success | 3309 | 0.000494278 | 13159 | 29899 | 2.22681 | 0.09604 | 0.932992 | 0.893527 | 0.00249161 | 0.00488744 |

## Notes

- The target-derived active mask is unchanged by the extra tightening: `13,159` contracted `IsFace` triangles.
- The plus-tightening case applies `c = 0.02` to all `29,899` `IsFace` skin triangles, with `a_const = 1 / (1 - 0.02) = 1.0204081632653061`.
- The plus-tightening forward response shrinks the `IsFace` surface more strongly, lowering deformed/rest area from `0.966795` to `0.932992`, but it also slightly increases Smile residual RMS in this zero-muscle forward check.
- Cherries local asset-copy logging emitted warnings after files were written. The artifacts were inspected directly and read back successfully.

## Next Gate

Open the forward `.vtu` and IsFace skin `.vtp` files in ParaView and inspect whether the estimated prestrain and the extra `c = 0.02` tightening look physically reasonable. The inverse cases should remain pending until this visual sanity gate is accepted.
