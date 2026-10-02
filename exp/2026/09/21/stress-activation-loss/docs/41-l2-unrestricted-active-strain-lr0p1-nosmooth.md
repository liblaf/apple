# Fresh active-strain Smile fit — Adam 0.1, no smoothness

The user requested a fresh active-strain fit after stopping the learning-rate 0.5 trajectory. This run uses the no-skin Smile fixture, unrestricted six-component symmetric controls, an L2-only objective, zero smoothness, and a 200-attempt Adam budget. It starts from dimensionless zero active strain, so `S = 0`, `B = I + S = I`, with zero displacement seed and fresh Adam moments. It does not resume any prior checkpoint.

## Constitutive model

The native active material uses dimensionless active strain:

```text
S = symmetric matrix from six Mandel controls
B = I + S = A_inv
G = F B
J = det(F)
W(F, B) = mu/2 (||G||² - 3) - mu (J - 1) + lambda_code/2 (J - 1)²
```

The determinant terms use physical deformation `F`; activation only affects the norm term. Controls are unrestricted, with no spectral projection or determinant constraint. Finite approximate solves and inverted cells are diagnostic data; finite gradients remain usable by Adam.

## Command

From `exp/2026/09/21/stress-activation-loss/`:

```sh
CHERRIES_NAME='Smile fresh L2 unrestricted active strain Adam 0.1 no smoothness' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,active-strain,unrestricted6,adam,lr0p1,no-smoothness,fresh' \
.venv/bin/python src/41-fit-l2-unrestricted.py \
  --activation-model strain \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --learning-rate 0.1 --smooth-weight 0 --steps 200 \
  --output 41-l2-unrestricted-active-strain-lr0p1-nosmooth-001
```

Smoothness calibration is skipped because the user explicitly selected zero weight. The archived mechanics reference supplies fixture and source provenance; it does not certify the active-strain adapter or approximate fitting trajectory.

## Artifacts and status

- [Run directory](../data/41-l2-unrestricted-active-strain-lr0p1-nosmooth-001/)
- [Console](../tmp/41-fit-l2-unrestricted-active-strain-lr0p1-nosmooth-001.console.log)
- Live fitted shape, loss, and ETA (private preview omitted)
- [Stopped LR 0.5 trajectory](41-l2-unrestricted-active-strain-lr0p5-nosmooth.md)

The fresh run is active in transient unit `apple-smile-fit-20260923.service`, PID **3215657**, with [Comet record](https://www.comet.com/liblaf/apple/713d4994d8314fcabb8f7bbc2c8635f0). Startup verification confirmed exactly zero S, controls, and displacement; B = I; fresh Adam; no resume checkpoint; learning rate 0.1; and zero normal/smoothness loss weights. All 110 numerical source hashes match the previously validated implementation and this run’s archived snapshots. The preceding 105-test validation therefore applies without rerunning the unchanged code. The viewer uses a transient service under `/run` and the same tailnet URL. This report does not treat the stopped LR 0.5 trajectory as a parent or a comparison baseline: the new run starts from identity activation and fresh optimizer state.

Live verification at step 2 confirmed LR 0.1, 2 Adam updates, 0 skips, RMS 5.093610424150 mm, a single history beginning at zero, and the correct current process. The served fitted vertices exactly matched the saved displacement plus rest positions.

## Interruption

This trajectory was stopped at the user's request and is preserved as evidence only. [interruption.json](../data/41-l2-unrestricted-active-strain-lr0p1-nosmooth-001/interruption.json) records `stopped_by_user_request` at last summary step 66: elapsed time 1490.52 s, fit RMS 2.8198 mm, and objective 0.0151287. The next run starts from identity activation with fresh Adam at learning rate 0.05 and does not resume this checkpoint.
