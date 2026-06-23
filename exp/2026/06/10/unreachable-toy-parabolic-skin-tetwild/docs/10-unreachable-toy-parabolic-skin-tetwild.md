# Unreachable Toy Parabolic Skin TetWild

## Purpose

Test a toy target where the free top surface really needs to shrink area:
the fixed bottom is flat, the side walls are fixed, the rest top is a high
parabolic cap, and the target top is a lower parabolic cap. This avoids the
flat-box squash artifact where the target translates top points without
shrinking surface triangle area.

## Geometry And Mesh

- Group: `exp/2026/06/10/unreachable-toy-parabolic-skin-tetwild`
- Mesh: `data/10-meshes/lr001/parabolic-toy-tetwild-lr001.vtu`
- TetWild: `lr=0.01`
- Points: `96,284`
- Tetrahedra: `525,321`
- Active muscle tets: `9,199`
- Target points: `5,836`
- Top area target/rest: `0.962875`
- Muscle box: `(x,z) in [0.35, 0.65] x [0.35, 0.65]`

The top cap uses rim height `0.10`, rest amplitude `0.12`, and target
amplitude `0.02`, so the center target displacement is about `-0.10` while the
rim remains fixed.

## Commands

Prepare mesh:

```bash
CHERRIES_NAME="Toy parabolic TetWild prepare lr001" \
CHERRIES_TAGS="unreachable,toy,parabolic,tetwild,prepare,lr001" \
uv run python src/10-prepare-parabolic-toy-skin-tetwild.py \
  --geometry parabolic \
  --tetwild-lr 0.01 \
  --parabolic-rim-height 0.10 \
  --parabolic-rest-amplitude 0.12 \
  --parabolic-target-amplitude 0.02 \
  --parabolic-grid 32 \
  --output-mesh data/10-meshes/lr001/parabolic-toy-tetwild-lr001.vtu \
  --output-summary data/10-meshes/lr001/summary.json
```

Inverse runs used `DEBUG=1`, `mode=squash`, `case_preset=skin-prestrain`,
`skin_thickness=0.005`, Adam, PNCG forward, and CG/MinRes adjoint fallback.
Each run used `inverse_max_steps=40`, `inverse_loss_min_delta=1e-7`, and
`require_convergence=false` for this exploratory sweep.

## Results

| Run | E MPa | Prestrain | Adam lr | Best step | Best loss | Error RMS | Error/target | Top y mean | Near-muscle y mean | Near target fraction | Top area deformed/rest | Residual lap RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20-squash-lr001/skin-prestrain5` | 0.20 | 0.05 | 0.2 | 40 | 0.000618925 | 0.0430903 | 0.802206 | -0.007958 | -0.025156 | 0.267990 | 0.989361 | 0.000298301 |
| `21-squash-lr001/skin-prestrain5-lr1` | 0.20 | 0.05 | 1.0 | 40 | 0.000630622 | 0.0434956 | 0.809751 | -0.008273 | -0.019779 | 0.210708 | 0.988946 | 0.000295389 |
| `22-squash-lr001/skin-e005-prestrain5-lr02` | 0.05 | 0.05 | 0.2 | 39 | 0.000722733 | 0.0465639 | 0.866874 | -0.004667 | -0.024256 | 0.258398 | 0.993983 | 0.000318170 |
| `23-squash-lr001/skin-e020-prestrain10-lr02` | 0.20 | 0.10 | 0.2 | 40 | 0.000493099 | 0.0384616 | 0.716034 | -0.012169 | -0.030598 | 0.325957 | 0.984436 | 0.000273711 |

## Outputs

- Best current result:
  `data/23-squash-lr001/skin-e020-prestrain10-lr02/20-toy-tetwild-squash-lr001-l2-skin-prestrain10-activation-per_tet.vtu`
- Best current temporal VTKHDF:
  `data/23-squash-lr001/skin-e020-prestrain10-lr02/20-toy-tetwild-squash-lr001-l2-skin-prestrain10-activation-per_tet-steps.vtkhdf`
- All four temporal VTKHDF files opened with `h5py`; each has 41 stored step
  values.

## Analysis

The parabolic target works as intended: the target top area is smaller than the
rest top area. The best run also shrinks the deformed top area more than the
5% variants (`0.984436` vs about `0.989`), while keeping the lowest residual
Laplacian RMS.

For this geometry, `10%` prestrain is better than `5%`: it starts closer to the
lower cap and continues to improve with activation. Lowering skin stiffness to
`0.05 MPa` was worse: it weakened the helpful prestrain response and produced a
larger error and rougher residual.

Adam `lr=1.0` was not useful here. It used much larger activations but ended
worse than `lr=0.2`. The best candidate so far is therefore:

- skin `E=0.20 MPa`
- skin thickness `0.005`
- skin prestrain `10%`
- Adam `lr=0.2`

The best run still stopped at the 40-step exploratory limit, not by the
20-step plateau criterion. A report-worthy continuation should rerun the best
case for more steps and allow the existing plateau rule to stop it.

## Notes

The prepare run was launched without `DEBUG=1`; Cherries created commit
`32c0864b` and then hung during Comet metadata flush, so the process was
interrupted after the mesh and summary were written. The inverse sweeps were
run with `DEBUG=1` and are currently uncommitted local artifacts.
