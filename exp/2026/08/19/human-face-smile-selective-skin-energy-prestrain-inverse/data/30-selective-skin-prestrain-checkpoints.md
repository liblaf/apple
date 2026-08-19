# Selective skin energy + prestrain inverse checkpoints

All rows are actual saved frames. Terminal is the equal-budget step-40 comparison.
Baseline-fidelity rows farther than 0.01 target-RMS units are explicitly marked `did-not-reach`.

| cohort | case | step | status | error mm | error/target | D deg | L mm | area ratio RMS | activation RMS | non-SPD | inverted | folded |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| terminal | H0P0 | 40 | selected | 2.72095 | 0.512406 | 13.3276 | 0.217061 | 0.14086 | 0.060958 | 0 | 47 | 25 |
| terminal | H0P1 | 40 | selected | 2.84903 | 0.536526 | 5.57913 | 0.181487 | 0.0727888 | 0.0548136 | 2 | 31 | 11 |
| terminal | H1P1 | 40 | selected | 1.41421 | 0.266322 | 7.69163 | 0.161078 | 0.12158 | 0.0628549 | 54 | 50 | 16 |
| terminal | H1P0 | 40 | selected | 1.43791 | 0.270786 | 15.5713 | 0.198196 | 0.168318 | 0.0672742 | 60 | 41 | 39 |
| baseline-fidelity | H0P0 | 40 | reached | 2.72095 | 0.512406 | 13.3276 | 0.217061 | 0.14086 | 0.060958 | 0 | 47 | 25 |
| baseline-fidelity | H0P1 | 40 | did-not-reach | 2.84903 | 0.536526 | 5.57913 | 0.181487 | 0.0727888 | 0.0548136 | 2 | 31 | 11 |
| baseline-fidelity | H1P1 | 12 | reached | 2.70813 | 0.509993 | 4.93827 | 0.144078 | 0.096841 | 0.0348253 | 0 | 2 | 2 |
| baseline-fidelity | H1P0 | 14 | reached | 2.71193 | 0.510709 | 9.89039 | 0.164081 | 0.150522 | 0.0408318 | 1 | 1 | 2 |
| common-tau | H0P0 | 35 | selected | 2.84204 | 0.535209 | 12.6388 | 0.215214 | 0.140237 | 0.0571736 | 0 | 39 | 19 |
| common-tau | H0P1 | 40 | selected | 2.84903 | 0.536526 | 5.57913 | 0.181487 | 0.0727888 | 0.0548136 | 2 | 31 | 11 |
| common-tau | H1P1 | 11 | selected | 2.83005 | 0.532953 | 4.78637 | 0.14763 | 0.0947211 | 0.0327147 | 0 | 2 | 1 |
| common-tau | H1P0 | 13 | selected | 2.83092 | 0.533115 | 9.40448 | 0.165645 | 0.149298 | 0.03878 | 0 | 2 | 1 |

Common tau: `0.536526178933`.

The common-tau cohort is secondary. Tau is the worst of the four per-case minima,
then each case uses its nearest saved checkpoint without interpolation.
