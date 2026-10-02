# Extended saved-endpoint surface roughness

These numbers support inspection of the saved 3D surfaces; they do not replace it. High-pass RMS uses the rest-normal displacement field and the same rest-skin operator for every endpoint.

| case | fit RMS (mm) | motion RMS (mm) | projection | full-face displacement HP RMS at 2/5/10 mm (mm) | mouth displacement HP RMS at 2/5/10 mm (mm) | current-mask field variation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical saved no-skin endpoint | 0.6543 | 4.974 | 0.9681 | 0.1329/0.2866/0.4732 | 0.3053/0.6535/1.054 | 278.5 |
| Manual smile elevators c50 no-skin baseline | 4.844 | 0.4901 | 0.0529 | 0.01722/0.03904/0.06563 | 0.04392/0.09882/0.1643 | 0 |
| Manual c50 selected-muscle Lame parameters x10 | 4.19 | 2.102 | 0.2471 | 0.07032/0.1576/0.2643 | 0.1795/0.3994/0.6624 | 0 |
| Manual c50 fat Lame parameters x0.1 | 4.755 | 0.7575 | 0.07567 | 0.02563/0.05887/0.1008 | 0.06477/0.1471/0.2492 | 0 |
| Manual c50 aponeurosis Lame parameters x0.1 | 4.695 | 0.8146 | 0.08842 | 0.02896/0.06657/0.114 | 0.07407/0.1692/0.2871 | 0 |
| Current-fixture Raw6-S no-skin endpoint, step 9 | 3.174 | 2.933 | 0.4717 | 0.09605/0.224/0.4004 | 0.2081/0.4834/0.865 | 250.2 |
| Current-fixture Region5Modes endpoint, step 40 | 4.292 | 1.066 | 0.1672 | 0.02843/0.0778/0.1488 | 0.06069/0.152/0.2821 | 0.02477 |

Field variation is evaluated only on the current fixture's 120,020-cell activation mask and same-MuscleId shared-face graph. The historical endpoint is restricted to that same support; no full historical-mask graph is reported as the current comparison graph.

Raw6-S stopped when its line search stalled at step 9; its saved endpoint has 2 inverted tetrahedra, and its independent reset reached 10,000 steps without success. Region5Modes reached its step-40 budget with 0 inverted tetrahedra; its independent reset succeeded with displacement difference 2.64819e-05 D. These are saved endpoints, not claims of inverse convergence.
