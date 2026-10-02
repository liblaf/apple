# No-skin L2 unrestricted active-stress fit

Status: interrupted at the user request to restart from zero stress with Adam learning rate 0.5. [Live Comet run](https://www.comet.com/liblaf/apple/4e85c8591c6142b48f6c1247acb69e7e).

## Setup

Smile target; fixed jaw and no contact; no skin energy. Unrestricted symmetric stress uses six Mandel coordinates per active tetrahedron. The objective is L2 position error plus within-muscle stress smoothness. The planned fit has 200 Adam attempts from zero stress, learning rate 0.05, epsilon 1e-8, with finite approximate forward/adjoint results allowed. Unusable numerical proposals roll back and reduce the learning rate; budget completion is not a convergence certificate.

## Preflight evidence

- Neutral position error: 5.095908027590781 mm RMS; zero stress; all determinants positive. The neutral forward and adjoint passed.
- The first strict derivative-validation attempt exhausted 100 Newton steps. The PCG cap of 1,000 forced shifted retries; this is retained in the Cherries log and failed run at [Comet](https://www.comet.com/liblaf/apple/2a5cddd25ea24edc9a359d8818518bbd).
- The [linear-budget probe](../data/08-newton-linear-budget-v2/probe.json) isolated this budget effect: the unshifted linear solve used 1,368 iterations. At the same fixed `q=0.04 I` state, one Newton step left force norms 4.4719678429275175e-8 with the 1,000 cap and 1.7102829561986972e-10 with the 10,000 cap. [Comet probe](https://www.comet.com/liblaf/apple/18fe51405e2849d4b33a9824c3e131fe).
- Full-face mechanics derivatives then passed all eight checks with the adequate PCG budget: maximum relative error 0.00018579167688726207 (0.018579%). [Receipt](../data/10-validation-inverse-v2-002/checks.json), [Comet validation](https://www.comet.com/liblaf/apple/e4e73ae592a74756a6f320b65c275c96).
- After that validation, the user requested a positive initial Newton shift equal to the mean absolute Hessian diagonal, skipping zero shift. The fit will record the changed source/configuration separately from the converged derivative reference; it will not claim the old reference validates convergence of approximate fitting states.

## Calibration

The fresh 8-attempt unregularized pilot completed without skips. Its RMS error changed from 5.095908 to 5.083027 mm; the final minimum determinant was 0.951082. Eight noninitial evaluations used finite, unconverged forward states.

The [calibration receipt](../data/41-l2-unrestricted-inverse-v2/calibration.json) freezes smoothness at **4.2207642736316316e-5**. At its saved recalculated state, the L2 dual effective-volume gradient norm was 0.021104288788860465 and the unweighted smoothness norm was 50.001107431431855. Their weighted ratio is exactly 0.1 by construction. The calibration forward state is approximate (`solver_valid=false`); this is not a claim of equilibrium-certified calibration or a promised terminal ratio.

## Results

The learning-rate-0.05 fit was interrupted after 21 completed attempts. Last saved RMS error: 4.987366950 mm. The checkpoint and source archive are retained. See [the replacement learning-rate-0.5 run](41-l2-unrestricted-lr0p5.md).

## Run command and provenance

Working directory: `exp/2026/09/21/stress-activation-loss/`.

```sh
CHERRIES_NAME='Smile L2 unrestricted stress with mean-abs Newton shift' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,unrestricted6,adam,mean-abs-shift' \
uv run python src/41-fit-l2-unrestricted.py \
  --mechanics-reference 10-validation-inverse-v2-002
```

The Cherries profile enables Comet and disables automatic Git commits. Repository HEAD is `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with existing uncommitted work; the [source archive](../data/41-l2-unrestricted-inverse-v2/sources.json) is the exact run provenance. The console stream is saved at `tmp/41-fit-l2-unrestricted-inverse-v2.console.log` and the Cherries log at `logs/41-fit-l2-unrestricted.log`.

Before launch, `uv run pytest tests src -q --no-cov` passed 87 tests, including the positive-first-shift regression and real four-tetrahedron CUDA implicit derivative check. The live pilot receipts also confirm zero zero-shift trials.
