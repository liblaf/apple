# Fresh active-strain Smile fit — Adam 0.5, no smoothness

The user requested replacing the active-stress fit with active strain, excluding activation from the volume-preserving terms, and restarting with learning rate 0.5. This run uses the same no-skin Smile fixture, unrestricted six-component controls, L2-only objective, zero smoothness, and a 200-attempt Adam budget. It starts from identity activation, zero displacement seed, and fresh Adam moments. The previous stress run is preserved separately through step 51.

## Constitutive model

The muscle potential is the existing native `StableNeoHookeanActive`:

```text
S = symmetric matrix from six Mandel controls (dimensionless)
B = I + S = A_inv
G = F B
J = det(F)
W(F, B) = mu/2 (||G||² - 3) - mu (J - 1) + lambda_code/2 (J - 1)²
```

Both determinant terms use the physical deformation `F`. Neither uses `det(F B)`. Activation acts only in the norm term. The native material six-vector uses ordinary off-diagonal entries, so the adapter converts the Mandel controls before supplying `activation_inv`. Controls are unrestricted: no spectral projection or imposed determinant constraint is added. Physical cell inversions and finite unconverged forward/adjoint solves remain diagnostics; Adam may continue with finite gradients and loss increases.

The bulk material convention remains `lambda_code = lambda_classical + mu`, with tissue Poisson ratio 0.49. Fat and aponeurosis retain passive Stable Neo-Hookean response. There is no membrane, contact, or jaw optimization. Newton-CG starts with the mean absolute physical Hessian diagonal and no zero-shift attempt; the implicit adjoint uses the unshifted physical Hessian.

## Command

From `exp/2026/09/21/stress-activation-loss/`:

```sh
CHERRIES_NAME='Smile fresh L2 unrestricted active strain Adam 0.5 no smoothness' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,active-strain,unrestricted6,adam,lr0p5,no-smoothness,fresh' \
.venv/bin/python src/41-fit-l2-unrestricted.py \
  --activation-model strain \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --learning-rate 0.5 --smooth-weight 0 --steps 200 \
  --output 41-l2-unrestricted-active-strain-lr0p5-nosmooth-001
```

No resume checkpoint is supplied. Smoothness calibration is skipped because its weight is explicitly zero. The historical mechanics reference supplies fixture and source provenance; it is not a validation certificate for the new active-strain adapter or the approximate fitting trajectory. Current validation is recorded below.

## Artifacts and live progress

- [Run directory](../data/41-l2-unrestricted-active-strain-lr0p5-nosmooth-001/)
- [Console](../tmp/41-fit-l2-unrestricted-active-strain-lr0p5-nosmooth-001.console.log)
- Live fitted shape, loss, and ETA (private preview omitted)
- [Preserved previous stress run](41-l2-unrestricted-lr0p05-nosmooth.md)

Checkpoints distinguish dimensionless `S` and `B` from stress tensors. Cross-model optimizer resumes are rejected. The fresh history starts at attempt zero. The fit and viewer use transient user services under `/run`; no persistent service is installed.

## Validation and launch evidence

The existing active-strain material was inspected and required no constitutive modification. A new CPU regression uses non-unit physical and activation determinants to isolate the activated-norm energy, force, Hessian product, and diagonal, and compares all six activation mixed derivatives with finite differences. All six native active-strain tests passed.

The combined verification command `uv run pytest tests src -q --no-cov` passed **105 tests**, including stress compatibility, strain mapping and symmetric gradient norms, model-specific checkpoints, cross-model resume rejection, approximate-solve continuation, physical-volume derivatives, and the live ETA. Ruff checks pass. The validation output is saved in `tmp/active-strain-validation.txt`. The fit launched in transient unit `apple-smile-fit-20260923.service`, PID **3206623**. Startup checkpoint verification confirmed exactly zero controls, S, and displacement, and B = I for all 288,235 active cells. The initialization receipt confirms fresh Adam and no parent checkpoint. All 110 archived numerical source hashes match their live source files. The first Adam update was saved and the live API reports active strain, LR 0.5, zero smoothness, and a new history beginning at zero. Completion of the attempt budget will not certify convergence, and this run changes both the activation coordinates and learning rate; it is not a controlled comparison against the preceding stress trajectory.

The experiment is recorded on [Comet](https://www.comet.com/liblaf/apple/f75b5d63d54241539c2531ac14b56c3a). At the first live check, the run had saved step 1 and was working on the next attempt. Initial RMS was 5.095908027590781 mm; this startup check is not a fit-quality conclusion. The viewer continues to refresh the fitted shape, objective curves, and remaining-time estimate at the same tailnet URL.

A subsequent verification at step 4 confirmed the live fitted vertices exactly equal the saved active-strain displacement plus rest positions. RMS was 5.054685638920208 mm, with zero skipped attempts and LR still 0.5. The served HTML matches the updated viewer source.

## Interruption

This trajectory was stopped at the user's request and is preserved as evidence only. [interruption.json](../data/41-l2-unrestricted-active-strain-lr0p5-nosmooth-001/interruption.json) records `stopped_by_user_request` at saved step 21: elapsed time 699.94 s, fit RMS 3.9140 mm, and objective 0.0291479. The requested restart uses identity activation, fresh Adam at learning rate 0.1, and no resume from this checkpoint.
