# Fresh active-strain Smile fit — Adam 0.05, no smoothness

The user requested a new active-strain fit from identity activation after stopping the LR 0.1 trajectory. This run uses the no-skin Smile fixture, unrestricted six-component symmetric controls, L2-only objective, zero smoothness, and a 200-attempt Adam budget. It starts with `S = 0`, `B = I + S = I`, zero displacement seed, and fresh Adam moments. No checkpoint is resumed.

## Constitutive model

```text
S = dimensionless symmetric matrix from six Mandel controls
B = I + S = A_inv
G = F B
J = det(F)
W(F, B) = mu/2 (||G||² - 3) - mu (J - 1) + lambda_code/2 (J - 1)²
```

The volume terms use physical deformation `F`; activation affects only the norm term. Controls are unrestricted. Finite approximate solves and inverted cells are recorded as diagnostics while finite gradients remain usable by Adam.

## Command

From `exp/2026/09/21/stress-activation-loss/`:

```sh
CHERRIES_NAME='Smile fresh L2 unrestricted active strain Adam 0.05 no smoothness' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,active-strain,unrestricted6,adam,lr0p05,no-smoothness,fresh' \
.venv/bin/python src/41-fit-l2-unrestricted.py \
  --activation-model strain \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --learning-rate 0.05 --smooth-weight 0 --steps 200 \
  --output 41-l2-unrestricted-active-strain-lr0p05-nosmooth-001
```

Smoothness calibration is skipped because the smoothness weight is explicitly zero. The archived mechanics reference provides fixture and source provenance; it does not certify the active-strain adapter or approximate fitting trajectory.

## Artifacts and status

- [Run directory](../data/41-l2-unrestricted-active-strain-lr0p05-nosmooth-001/)
- [Console](../tmp/41-fit-l2-unrestricted-active-strain-lr0p05-nosmooth-001.console.log)
- Live fitted shape, loss, and ETA (private preview omitted)
- [Stopped LR 0.1 trajectory](41-l2-unrestricted-active-strain-lr0p1-nosmooth.md)

The fresh run is active in transient unit `apple-smile-fit-20260923.service`, PID **3235943**, with [Comet record](https://www.comet.com/liblaf/apple/2bbacc689f894f6a9c8acb9b78e2f4d2). Startup verification confirmed exactly zero controls, S, and displacement; identity B; fresh Adam; no resume checkpoint; learning rate 0.05; and zero normal/smoothness weights. All 110 numerical sources are unchanged from the previously validated implementation and match the new archived snapshots. The existing 105-test validation applies without rerunning unchanged code. The viewer remains a transient service under `/run` at the same tailnet URL. The stopped LR 0.1 trajectory is not a parent or comparison baseline: this run starts with identity activation and fresh optimizer state.

Live verification at step 2 confirmed LR 0.05, 2 Adam updates, 0 skips, RMS 5.094727183278 mm, a fresh single-segment history, and the current process. Fitted vertices matched the saved checkpoint: True.
