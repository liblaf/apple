# Folding factorial analysis

Practical stationarity: 13/16 pass; 3 fail. The failed cases are retained for explicit comparison (require_stationarity=False).

All effects are descriptive paired differences; interaction rows are difference-in-differences, not causal estimates. Determinant-sign fractions are descriptive frame classifications using DetF<0, DetAinv, and DetG signs (with a zero tolerance). Factorial coefficients use low=-1/high=+1 and 2^|S| * mean(y * product(coded factors)).

| Dimension | Case | Stationarity / tail | Final grad inf / RMS | Failures (forward/inverse/adjoint/trial) | First/last inversion | Recovered by final | Final/peak F- A- G+ rest | Final/peak F- A+ G- rest | Final inverted rest fraction | Peak inverted rest fraction | Trajectory min detF | Final target RMS | Final activation RMS | Midline arc ratio / turning density / x-reversal |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2d | l010-band-per_cell-nu35 | False / False | 1.771e-06 / 1.904e-07 | 0/0/0/0 | 37/2201 | False | 0.065/0.07 | 0.05/0.075 | 0.115 | 0.13 | -75.2 | 0.0009572 | 4.252 | 2.919 / 76.23 / 0.1 |
| 2d | l010-band-per_cell-nu49 | False / True | 4.484e-05 / 6.953e-06 | 0/0/0/0 | 54/1290 | False | 0.005/0.005 | 0/0 | 0.005 | 0.005 | -11.75 | 0.002994 | 0.3696 | 1.446 / 44.73 / 0 |
| 2d | l010-band-shared-nu35 | True / True | 1.102e-08 / 7.168e-09 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.2906 | 0.01642 | 0.5285 | 1.019 / 5.636 / 0 |
| 2d | l010-band-shared-nu49 | True / False | 3.087e-09 / 2.022e-09 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.4343 | 0.005532 | 0.5037 | 1.173 / 20.58 / 0 |
| 2d | l010-full-per_cell-nu35 | True / False | 6.148e-13 / 6.605e-14 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.5849 | 1.49e-10 | 0.2685 | 1.147 / 14.33 / 0 |
| 2d | l010-full-per_cell-nu49 | True / False | 5.062e-13 / 7.638e-14 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.844 | 8.093e-11 | 0.1425 | 1.191 / 43.39 / 0 |
| 2d | l010-full-shared-nu35 | True / False | 5.502e-14 / 3.184e-14 | 0/0/0/0 | 12/60 | True | 0/0 | 0/0.005 | 0 | 0.005 | -0.51 | 0.007264 | 0.5161 | 1 / 0.04549 / 0 |
| 2d | l010-full-shared-nu49 | True / False | 1.316e-14 / 8.273e-15 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.8737 | 0.002968 | 0.4633 | 1.127 / 12.26 / 0 |
| 2d | l100-band-per_cell-nu35 | True / False | 2.02e-09 / 4.835e-10 | 0/0/0/0 | 18/2201 | False | 0.0165/0.0195 | 0.0075/0.0075 | 0.024 | 0.0255 | -9.814 | 6.02e-05 | 0.4789 | 1.439 / 57.88 / 0.05 |
| 2d | l100-band-per_cell-nu49 | False / False | 8.347e-05 / 4.849e-06 | 0/0/0/0 | 21/1231 | False | 0.0065/0.007 | 0.0005/0.0015 | 0.007 | 0.007 | -26.31 | 0.001575 | 0.3373 | 1.516 / 78.98 / 0.04 |
| 2d | l100-band-shared-nu35 | True / False | 4.425e-14 / 2.73e-14 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.08254 | 0.004441 | 5.956 | 1 / 0.04357 / 0 |
| 2d | l100-band-shared-nu49 | True / False | 3.467e-12 / 2.442e-12 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.2644 | 0.003976 | 21.02 | 1 / 0.04971 / 0 |
| 2d | l100-full-per_cell-nu35 | True / True | 8.312e-13 / 1.009e-13 | 0/0/0/0 | 20/1286 | False | 0.001/0.001 | 0/0.0005 | 0.001 | 0.001 | -2.034 | 2.653e-09 | 0.2288 | 1.005 / 4.648 / 0 |
| 2d | l100-full-per_cell-nu49 | True / True | 4.751e-10 / 1.993e-11 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.138 | 7.967e-07 | 0.1385 | 1.029 / 10.14 / 0 |
| 2d | l100-full-shared-nu35 | True / False | 2.066e-11 / 1.195e-11 | 0/0/0/0 | 10/16 | True | 0/0 | 0/0.0005 | 0 | 0.0005 | -0.2238 | 0.0008996 | 4.218 | 1 / 5.862e-07 / 0 |
| 2d | l100-full-shared-nu49 | True / False | 1.346e-08 / 7.866e-09 | 0/0/0/0 | None/None | False | 0/0 | 0/0 | 0 | 0 | 0.974 | 0.0008986 | 15.08 | 1 / 5.974e-07 / 0 |
