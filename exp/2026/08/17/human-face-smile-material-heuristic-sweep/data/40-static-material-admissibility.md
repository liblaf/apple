# Static material admissibility sweep

Stage: `formal`.

A1 decision: **A1-fail**; safe_low = `None`. neither the E=0.25 nor E=0.5 row robustly passes

| candidate | R0 / R1 class | fidelity R0 / R1 | Δ fidelity | Δu / target | worst inv | worst detF q001 / min | worst folds | worst area q001 / q999 | stable | robust pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| e025-p050 | admissible / admissible | 0.944266 / 0.944232 | 3.42207e-05 | 0.000236484 | 0 | 0.93867 / 0.43089 | 0 | 0.83285 / 1.0045 | yes | yes |
| e025-p075 | admissible / admissible | 0.921939 / 0.921919 | 1.96045e-05 | 0.000127079 | 0 | 0.90836 / 0.4982 | 0 | 0.75988 / 1.0061 | yes | yes |
| e025-p100 | admissible / physical-inadmissible | 0.903967 / 0.903932 | 3.57286e-05 | 0.000322464 | 2 | 0.87905 / -0.94183 | 2 | 0.69299 / 1.0077 | no | no |
| e050-p050 | admissible / admissible | 0.945085 / 0.945103 | 1.86741e-05 | 7.52994e-05 | 0 | 0.93842 / 0.41671 | 0 | 0.83291 / 1.0043 | yes | yes |
| e050-p075 | physical-inadmissible / physical-inadmissible | 0.923199 / 0.923228 | 2.92385e-05 | 0.000208139 | 1 | 0.90824 / -0.17196 | 1 | 0.75999 / 1.006 | no | no |
| e050-p100 | admissible / admissible | 0.905674 / 0.905646 | 2.84505e-05 | 0.000592424 | 0 | 0.87861 / 0.36903 | 0 | 0.69312 / 1.0074 | yes | yes |
| e100-p050 | admissible / admissible | 0.945991 / 0.945981 | 9.39433e-06 | 6.83414e-05 | 0 | 0.9383 / 0.41191 | 0 | 0.83297 / 1.0041 | yes | yes |
| e100-p075 | admissible / admissible | 0.924646 / 0.924567 | 7.90181e-05 | 0.000322833 | 0 | 0.90786 / 0.28313 | 0 | 0.76007 / 1.0059 | yes | yes |
| e100-p100 | admissible / admissible | 0.907513 / 0.907505 | 8.13441e-06 | 0.000352504 | 0 | 0.87816 / 0.51922 | 0 | 0.69326 / 1.0073 | yes | yes |

Robust pass requires two successful finite forwards, all physical gates in both branches, matching classifications, absolute fidelity difference at most 0.001, and loss-ROI displacement disagreement at most 1% of target RMS.
