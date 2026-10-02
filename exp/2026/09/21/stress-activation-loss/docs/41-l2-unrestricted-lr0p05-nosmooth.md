# Fresh no-skin L2 fit — Adam 0.05, no smoothness

Status: stopped at the user's request to switch to a fresh active-strain run with Adam learning rate 0.5. The last saved state is step 51, with RMS 4.512924518562186 mm and L2 objective 0.038750317274108215. The interruption receipt is saved in the run directory. This run replaced the earlier learning-rate-0.5 stress trajectory; it is not a continuation or a matched learning-rate comparison.

The controls start at zero active stress, the displacement seed is zero, and Adam starts with empty moments at **learning rate 0.05**, betas `(0.9, 0.999)`, epsilon `1e-8`. The budget is **200 attempts**. The only fitted loss is L2 position error: normal weight **0** and smoothness weight **0**. No smoothness pilot or coefficient calibration is performed. Raw smoothness measurements may still be logged as diagnostics; they have zero contribution to the objective and its gradient.

The fixture remains no-skin, fixed-jaw and contact-off, with unrestricted symmetric six-component stress. Finite unconverged solves and inverted cells are accepted by Adam, with their diagnostics retained. Newton-CG continues to start with the mean absolute Hessian diagonal as its shift. The forward backend remains matrix-free. Budget completion alone does not certify convergence.

## Command

From `exp/2026/09/21/stress-activation-loss/`, the transient fit service runs:

```sh
CHERRIES_NAME='Smile fresh L2 unrestricted Adam 0.05 no smoothness' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,unrestricted6,adam,lr0p05,no-smoothness,fresh' \
.venv/bin/python src/41-fit-l2-unrestricted.py \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --learning-rate 0.05 --smooth-weight 0 --steps 200 \
  --output 41-l2-unrestricted-inverse-v2-lr0p05-nosmooth-001
```

There is deliberately no resume checkpoint argument. The explicit fixed-weight receipt records `calibration_status: not_run`. The earlier converged mechanics reference and source differences are recorded without claiming that it certifies the current approximate trajectory.

## Outputs

- [Run data](../data/41-l2-unrestricted-inverse-v2-lr0p05-nosmooth-001/)
- [Console log](../tmp/41-fit-l2-unrestricted-inverse-v2-lr0p05-nosmooth-001.console.log)
- Live tailnet page (private preview omitted) with a fresh loss curve and fitted shape
- [Previous trajectory](41-l2-unrestricted-lr0p5.md), preserved through accepted step 15

The fit and viewer use transient user services under `/run`, with no persistent service installation. Sources and protocol are snapshotted in the run directory. Verification and launch details follow below.

## Verification

Before launch, `uv run pytest tests src -q --no-cov` passed **91 tests**. The added CPU orchestration regression verifies that explicit zero smoothness skips the pilot, passes zero stress and zero seed, selects learning rate 0.05 and does not provide a resume checkpoint. Ruff and CLI help checks pass.

The run is recorded on [Comet](https://www.comet.com/liblaf/apple/e66dbfaeb8f142ef96ef6caea075c187). The transient fit unit is `apple-smile-fit-20260923.service`, initially PID `3160064`. No previous Adam moments or accepted displacement checkpoint are loaded.

Startup verification confirmed that `initial-state.npz` has step 0 and exactly zero controls, stress tensors and displacement. Initialization records fresh Adam and no resume checkpoint. At the first inspection, step 3 had three optimizer updates, no skips, learning rate 0.05 and RMS **5.094578435887956 mm** (initial RMS **5.095908027590781 mm**). The objective equals its L2 contribution exactly; the smoothness contribution is zero. The live API and both chart functions were checked to contain only the fresh history starting at zero, with no previous trajectory segments.
