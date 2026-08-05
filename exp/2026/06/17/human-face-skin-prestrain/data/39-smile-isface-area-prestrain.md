# Smile IsFace Area Prestrain

- Mesh: `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu`
- Target: `Smile`
- Area ratio floor: `0.1`
- Output mesh: `/home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-skin-prestrain/data/39-smile-isface-area-prestrain.vtp`
- Output triangles: `29899`
- Active contracted prestrain triangles: `13159`
- Total target/rest area ratio: `0.995904`

## Cell Fields

- `TargetRestAreaRatio`: target triangle area divided by rest area.
- `TargetRestLengthRatio`: `sqrt(TargetRestAreaRatio)`.
- `EstimatedStressFreeLengthRatio`: length ratio used by the shrink prestrain.
- `EstimatedLengthPrestrain`: positive length shrink, `1 - EstimatedStressFreeLengthRatio`.
- `EstimatedInvLengthFactor`: actual isotropic `A_inv` diagonal factor.
- `EstimatedActivationInvDiag`: stored skin `ActivationInv` diagonal offset.

## Summary

| field | min | q1 | median | q99 | max | mean | rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target/rest area | 0.0454157 | 0.632231 | 1.00336 | 1.72515 | 16.513 | 1.01792 | 1.05528 |
| length prestrain | 0.00% | 0.00% | 0.00% | 20.49% | 68.38% | 1.90% | 4.69% |
| activation inv diag | 0 | 0 | 0 | 0.257657 | 2.16228 | 0.0217335 | 0.0589579 |
