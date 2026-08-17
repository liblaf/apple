# Static material admissibility A2 boundary refinement

Stage: `formal`.

A2 decision: **A2-fail**; safe_low = `None`. at least one new E=0.75 prestrain point is not robustly admissible; the discrete Stage-B rectangle is not selected

| candidate | R0 / R1 class | fidelity R0 / R1 | Δ fidelity | Δu / target | worst inv | worst detF q001 / min | worst folds | worst area q001 / q999 | stable | robust pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| e075-p050 | admissible / admissible | 0.945626 / 0.945671 | 4.54361e-05 | 0.000170302 | 0 | 0.93826 / 0.41822 | 0 | 0.83295 / 1.0042 | yes | yes |
| e075-p075 | admissible / admissible | 0.92405 / 0.924087 | 3.72465e-05 | 0.000144093 | 0 | 0.90773 / 0.32219 | 0 | 0.76004 / 1.0059 | yes | yes |
| e075-p100 | physical-inadmissible / admissible | 0.906723 / 0.906736 | 1.28872e-05 | 0.000497018 | 3 | 0.87805 / -0.93602 | 3 | 0.69322 / 1.0074 | no | no |

Robust pass requires two successful finite forwards, all physical gates in both branches, matching classifications, absolute fidelity difference at most 0.001, and loss-ROI displacement disagreement at most 1% of target RMS.

The A2 decision is limited to p={0.5,0.75,1.0}; it does not assert continuous admissibility between sampled gains. R1 is a fixed cyclic shift so p=0.75 occupies a different order position than in R0.
