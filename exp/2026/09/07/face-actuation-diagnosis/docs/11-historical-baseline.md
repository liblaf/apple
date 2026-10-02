# Historical no-skin baseline

This directory converts the saved June 17 no-skin endpoint into the current diagnosis artifact schema. It performs no forward solve, control replay, or fresh-rest check. The June optimizer saved its best endpoint at step 194 after a 200-step budget; the run reported six failed forward evaluations and did not report inverse convergence.

The common September rest-surface weights give the saved endpoint an area-weighted fit RMS of 0.654339 mm against a 5.095908 mm target, with 4.974140 mm motion RMS and target projection amplitude 0.968146. The original uniform-vertex RMS was 0.633555 mm. The area-weighted and uniform values measure the same saved displacement with different vertex weights.

The saved state contains 142 inverted tetrahedra (minimum det(F) -7.3945) and 159 active tensors with a nonpositive minimum eigenvalue. These are recorded endpoint properties. They do not invalidate the fit measurement, and this export does not claim the state satisfies the newer geometric checks.

## Semantic differences from the September screen

| Setting | June saved no-skin endpoint | Current diagnosis Raw6 no-skin | Archived September screen |
| --- | ---: | ---: | ---: |
| Active tetrahedra | 288,235 | 120,020 | 120,020 |
| Scalar activation controls | 1,729,410 | 720,120 | 720,120 |
| Muscle regions represented | 103 | 35 | 35 |
| Fixed vertices | 27,036 | 33,636 | 33,636 |
| Skin energy | absent; zero skin triangles | absent (`skin_factor=0`) | corrected zero-prestrain membrane, factor 0.12 |
| Muscle material | E=30 kPa, nu=0.49 | E=24 kPa, nu=0.46 | E=24 kPa, nu=0.46 |
| Fit weighting | uniform observed vertices and Cartesian components | rest-triangle-area weights | rest-triangle-area weights |
| Outer optimizer | Adam, lr 0.3, eps 0.01 | volume-preconditioned L-BFGS direction with Armijo backtracking and 0.1 per-tet trust radius | volume-preconditioned L-BFGS direction with Armijo backtracking |
| Objective scaling | Cartesian MSE multiplied by 1e6 (mm^2) | squared area-weighted fit normalized by target RMS squared | squared area-weighted fit normalized by target RMS squared |
| Forward / adjoint rtol | 5e-4 / 5e-4 | 1e-5 / 1e-7 | 1e-5 / 1e-7 |
| Endpoint acceptance | solver success only | inversion and active-tensor spectra are recorded diagnostics | SPD active tensor and det(F)>=0.2 checks |

The artificial-cut rule adds 6,600 fixed vertices. None is an observed IsFace or IsLip vertex, but the June endpoint moved those vertices by 1.230671 mm RMS and 3.671175 mm maximum. They touch 7,241 historically active tetrahedra. This proves the boundary changed degrees of freedom used by the June state; only a matched solve can measure how much it suppresses the smile.

The current screen retains 35 of 103 historical muscle labels. The 68 excluded labels contain 58.6% of the historical active cells and 32.2% of the saved volume-weighted squared activation offset. Masseter superficial accounts for most of that excluded activation diagnostic, but this is an optimizer-use ranking, not a causal muscle attribution. A counterfactual equilibrium solve is required to decide which excluded labels improve the surface fit.

## Minimal matched comparison

Use the same rest volume, original 27,036-vertex fixed mask, all 288,235 historical active tetrahedra, no skin energy, the June target, and the same material and solver tolerances for both Raw6 and Raw6-S. Initialize both from rest and zero activation, give them the same fixed step budget, and report both the original uniform RMS and the current area-weighted RMS. Raw6-S should differ only by the within-muscle shared-face penalty. Keep inversion, det(F), and active-tensor eigenvalues as reported diagnostics so distorted trials remain inspectable.

The running diagnosis Raw6 case switches to no skin but still changes the active mask, fixed boundary, muscle material, target weighting, optimizer, objective normalization, and solver tolerances. Its partial trajectory therefore cannot isolate per-tetrahedron capacity or be compared directly with the June best endpoint. If it stalls, the next control should be a matched Adam run before attributing the result to the active mask.

The closest exact rerun command is:

```bash
cd exp/2026/06/17/human-face-smile-prestrain-v2
DEBUG=1 CHERRIES_NAME="Human face Smile no-skin lr0.3 baseline loss-mm2" CHERRIES_TAGS="human-face,smile,inverse,no-skin,lr0.3,loss-mm2,baseline200,local" uv run python src/20-inverse-human-face.py --case-set no-skin --inverse-lr 0.3 --inverse-max-steps 200 --mandatory-baseline-steps 200 --segment-steps 8 --time-budget-hours 10 --reserve-minutes 5 --step-time-budget-s 180 --live-plot-dir figs/live --output-summary data/22-no-skin-lr03-summary.json --output-table data/22-no-skin-lr03-table.md
```

That command would target the historical output names. Add a unique `--case-label` and separate aggregate output paths before rerunning so the verified artifacts are not overwritten.

## Complete region coverage

The activation-energy share is computed from the saved offset H=A_inv-I with muscle-fraction-volume weights. It describes where the historical optimizer used controls and does not isolate a muscle's causal effect on the smile.

| Current screen | ID | Muscle label | Cells | H Frobenius RMS | Saved energy share | Tets touching added fixed vertices |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| retained | 28 | Occipitofrontalis epicranius001_Head_muscles_0 | 35,171 | 0.0342853 | 0.1877% | 1,100 |
| retained | 57 | Levator labii superioris001_Head_muscles_0_0 | 1,303 | 0.457395 | 0.3828% | 0 |
| retained | 58 | Levator labii superioris001_Head_muscles_0_1 | 1,182 | 0.535872 | 0.6753% | 0 |
| retained | 61 | Buccinator001_Head_muscles_0_0 | 6,705 | 0.454008 | 2.9783% | 0 |
| retained | 62 | Buccinator001_Head_muscles_0_1 | 3,861 | 0.483094 | 3.4821% | 0 |
| retained | 63 | Zygomaticus major001_Head_muscles_0_0 | 1,279 | 0.970529 | 4.4655% | 0 |
| retained | 64 | Zygomaticus major001_Head_muscles_0_1 | 1,147 | 0.925291 | 3.6614% | 0 |
| retained | 73 | Risorius001_Head_muscles_0_0 | 632 | 0.640403 | 0.7106% | 0 |
| retained | 74 | Risorius001_Head_muscles_0_1 | 687 | 0.554233 | 0.5293% | 0 |
| retained | 93 | Mentalis001_Head_muscles_0_0 | 1,738 | 0.554906 | 1.2863% | 0 |
| retained | 94 | Mentalis001_Head_muscles_0_1 | 1,412 | 0.596365 | 1.4079% | 0 |
| retained | 97 | Orbicularis oculi001_Head_muscles_0_0 | 8,440 | 0.310893 | 3.7905% | 0 |
| retained | 98 | Orbicularis oculi001_Head_muscles_0_1 | 7,853 | 0.277236 | 2.8912% | 0 |
| retained | 99 | Depressor anguli001_Head_muscles_0_0 | 1,623 | 0.647381 | 2.3249% | 0 |
| retained | 100 | Depressor anguli001_Head_muscles_0_1 | 1,848 | 0.558902 | 2.0475% | 0 |
| retained | 101 | Nasalis alarportion001_Head_muscles_0_0 | 339 | 0.718403 | 0.5102% | 0 |
| retained | 102 | Nasalis alarportion001_Head_muscles_0_1 | 427 | 0.606212 | 0.3721% | 0 |
| retained | 108 | Procerus001_Head_muscles_0 | 1,339 | 0.0545325 | 0.0060% | 0 |
| retained | 110 | Corrugator supercilii001_Head_muscles_0_0 | 388 | 0.0853822 | 0.0082% | 0 |
| retained | 111 | Corrugator supercilii001_Head_muscles_0_1 | 404 | 0.08638 | 0.0083% | 0 |
| retained | 116 | Nasalis transverse portion001_Head_muscles_0 | 2,011 | 0.34907 | 0.7322% | 0 |
| retained | 139 | Depressor septi001_Head_muscles_0 | 1,635 | 0.604108 | 1.1462% | 0 |
| retained | 142 | Levator anguli oris001_Head_muscles_0_0 | 932 | 0.596563 | 0.7468% | 0 |
| retained | 143 | Levator anguli oris001_Head_muscles_0_1 | 939 | 0.518287 | 0.5545% | 0 |
| retained | 162 | Depressor labii inferioris001_Head_muscles_0_0 | 1,177 | 0.573715 | 1.2579% | 0 |
| retained | 163 | Depressor labii inferioris001_Head_muscles_0_1 | 1,159 | 0.540956 | 1.2495% | 0 |
| retained | 218 | Zygomaticus minor001_Head_muscles_0_0 | 835 | 0.84402 | 1.5955% | 0 |
| retained | 219 | Zygomaticus minor001_Head_muscles_0_1 | 775 | 0.773567 | 1.4057% | 0 |
| retained | 254 | Orbicularis oris001_Head_muscles_0 | 14,972 | 0.689585 | 23.4830% | 0 |
| retained | 255 | Platysma001_Head_muscles_0_0 | 7,712 | 0.286453 | 1.9137% | 345 |
| retained | 256 | Platysma001_Head_muscles_0_1 | 7,819 | 0.262128 | 1.6430% | 322 |
| retained | 257 | Depressor supercilli001_Head_muscles_0_0 | 472 | 0.0607514 | 0.0026% | 0 |
| retained | 258 | Depressor supercilli001_Head_muscles_0_1 | 464 | 0.0635953 | 0.0029% | 0 |
| retained | 283 | Levator labii superioris alaeque nasi001_Head_muscles_0_0 | 713 | 0.429104 | 0.2170% | 0 |
| retained | 284 | Levator labii superioris alaeque nasi001_Head_muscles_0_1 | 627 | 0.365764 | 0.1456% | 0 |
| excluded | 2 | Omohyoid001_Head_muscles_0_0 | 86 | 0.17914 | 0.0065% | 40 |
| excluded | 3 | Omohyoid001_Head_muscles_0_1 | 74 | 0.17473 | 0.0075% | 38 |
| excluded | 4 | Temporalis002_Head_muscles_0_0 | 13,712 | 0.0468402 | 0.1875% | 339 |
| excluded | 5 | Temporalis002_Head_muscles_0_1 | 12,781 | 0.0379025 | 0.0974% | 533 |
| excluded | 6 | superior rectus001_Head_muscles_0_0 | 253 | 0.00614003 | 0.0000% | 0 |
| excluded | 7 | superior rectus001_Head_muscles_0_1 | 278 | 0.0089493 | 0.0001% | 0 |
| excluded | 8 | Medial rectus002_Head_muscles_0_0 | 879 | 0.0131526 | 0.0003% | 0 |
| excluded | 9 | Medial rectus002_Head_muscles_0_1 | 589 | 0.00733413 | 0.0001% | 0 |
| excluded | 22 | Inferior oblique002_Head_muscles_0_0 | 332 | 0.0180547 | 0.0003% | 0 |
| excluded | 23 | Inferior oblique002_Head_muscles_0_1 | 409 | 0.0247451 | 0.0006% | 0 |
| excluded | 26 | Levator palpebrae superioris001_Head_muscles_0_0 | 672 | 0.0219772 | 0.0006% | 0 |
| excluded | 27 | Levator palpebrae superioris001_Head_muscles_0_1 | 659 | 0.0211884 | 0.0006% | 0 |
| excluded | 41 | Rectus capitis anterior001_Head_muscles_0_0 | 879 | 0.0506411 | 0.0076% | 82 |
| excluded | 42 | Rectus capitis anterior001_Head_muscles_0_1 | 833 | 0.0458258 | 0.0060% | 90 |
| excluded | 59 | Masseter superficial001_Head_muscles_0_0 | 13,044 | 0.422501 | 14.3420% | 0 |
| excluded | 60 | Masseter superficial001_Head_muscles_0_1 | 15,139 | 0.354743 | 13.2183% | 0 |
| excluded | 71 | Medial pterygoid001_Head_muscles_0_0 | 5,018 | 0.0169927 | 0.0069% | 0 |
| excluded | 72 | Medial pterygoid001_Head_muscles_0_1 | 4,449 | 0.0134488 | 0.0040% | 0 |
| excluded | 77 | Temporal fascia001_Head_muscles_0_0 | 20,987 | 0.050423 | 0.3246% | 1,106 |
| excluded | 78 | Temporal fascia001_Head_muscles_0_1 | 22,227 | 0.0582464 | 0.5247% | 1,216 |
| excluded | 79 | Auricularis anterior001_Head_muscles_0_0 | 2,198 | 0.183661 | 0.1579% | 155 |
| excluded | 80 | Auricularis anterior001_Head_muscles_0_1 | 1,949 | 0.122452 | 0.0686% | 126 |
| excluded | 89 | Digastric fibrous loop001_Head_muscles_0_0 | 29 | 0.0493637 | 0.0001% | 0 |
| excluded | 90 | Digastric fibrous loop001_Head_muscles_0_1 | 31 | 0.0356982 | 0.0000% | 0 |
| excluded | 91 | Lateral rectus001_Head_muscles_0_0 | 484 | 0.00362015 | 0.0000% | 0 |
| excluded | 92 | Lateral rectus001_Head_muscles_0_1 | 531 | 0.0502283 | 0.0030% | 0 |
| excluded | 103 | Longus colli002_Head_muscles_0_0 | 631 | 0.0195008 | 0.0005% | 81 |
| excluded | 104 | Longus colli002_Head_muscles_0_1 | 522 | 0.0201717 | 0.0005% | 62 |
| excluded | 112 | Longus capitis001_Head_muscles_0_0 | 1,368 | 0.0242625 | 0.0022% | 84 |
| excluded | 113 | Longus capitis001_Head_muscles_0_1 | 1,146 | 0.0244441 | 0.0023% | 72 |
| excluded | 114 | Genioglossus001_Head_muscles_0_0 | 3,347 | 0.107283 | 0.2540% | 0 |
| excluded | 115 | Genioglossus001_Head_muscles_0_1 | 3,423 | 0.115994 | 0.3152% | 0 |
| excluded | 135 | Inferior rectus001_Head_muscles_0_0 | 495 | 0.0240369 | 0.0004% | 0 |
| excluded | 136 | Inferior rectus001_Head_muscles_0_1 | 422 | 0.0536561 | 0.0024% | 0 |
| excluded | 137 | Sternocleidomastoid001_Head_muscles_0_0 | 841 | 0.0390988 | 0.0059% | 420 |
| excluded | 138 | Sternocleidomastoid001_Head_muscles_0_1 | 819 | 0.0313651 | 0.0039% | 387 |
| excluded | 160 | Stylohyoid001_Head_muscles_0_0 | 885 | 0.0921519 | 0.0258% | 0 |
| excluded | 161 | Stylohyoid001_Head_muscles_0_1 | 910 | 0.0877757 | 0.0235% | 0 |
| excluded | 164 | Common tendinous ring002_Head_muscles_0_0 | 18 | 0.000449221 | 0.0000% | 0 |
| excluded | 165 | Common tendinous ring002_Head_muscles_0_1 | 30 | 0.000969558 | 0.0000% | 0 |
| excluded | 166 | Mylohyoid001_Head_muscles_0_0 | 2,009 | 0.0643602 | 0.0077% | 0 |
| excluded | 167 | Mylohyoid001_Head_muscles_0_1 | 2,045 | 0.0683543 | 0.0086% | 0 |
| excluded | 186 | Masseter deep001_Head_muscles_0_0 | 4,572 | 0.20955 | 1.1611% | 0 |
| excluded | 187 | Masseter deep001_Head_muscles_0_1 | 4,699 | 0.165107 | 0.7741% | 0 |
| excluded | 188 | Lateral pterygoid001_Head_muscles_0_0 | 2,398 | 0.0491795 | 0.0284% | 0 |
| excluded | 189 | Lateral pterygoid001_Head_muscles_0_1 | 2,658 | 0.0388941 | 0.0204% | 0 |
| excluded | 194 | Styloglossus001_Head_muscles_0_0 | 206 | 0.0173629 | 0.0001% | 0 |
| excluded | 195 | Styloglossus001_Head_muscles_0_1 | 228 | 0.0163056 | 0.0001% | 0 |
| excluded | 198 | Rectus capitis lateralis002_Head_muscles_0_0 | 23 | 0.00372119 | 0.0000% | 23 |
| excluded | 199 | Rectus capitis lateralis002_Head_muscles_0_1 | 16 | 0.0037498 | 0.0000% | 16 |
| excluded | 200 | Sternohyoid001_Head_muscles_0_0 | 77 | 0.20563 | 0.0083% | 49 |
| excluded | 201 | Sternohyoid001_Head_muscles_0_1 | 73 | 0.246952 | 0.0109% | 48 |
| excluded | 202 | Hyoglossus001_Head_muscles_0_0 | 468 | 0.0966003 | 0.0149% | 0 |
| excluded | 203 | Hyoglossus001_Head_muscles_0_1 | 485 | 0.10471 | 0.0190% | 0 |
| excluded | 204 | Longus colli001_Head_muscles_0_0 | 93 | 0.0186919 | 0.0001% | 49 |
| excluded | 205 | Longus colli001_Head_muscles_0_1 | 94 | 0.0165553 | 0.0001% | 53 |
| excluded | 234 | Digastiric001_Head_muscles_0_0 | 3,295 | 0.0911977 | 0.1167% | 68 |
| excluded | 235 | Digastiric001_Head_muscles_0_1 | 3,216 | 0.0834881 | 0.0960% | 64 |
| excluded | 236 | Thyrohyoid001_Head_muscles_0_0 | 160 | 0.10972 | 0.0036% | 127 |
| excluded | 237 | Thyrohyoid001_Head_muscles_0_1 | 133 | 0.101289 | 0.0033% | 100 |
| excluded | 259 | Intertransversarii anteriores cervicis001_Head_muscles_0_0 | 18 | 0.00302969 | 0.0000% | 18 |
| excluded | 260 | Intertransversarii anteriores cervicis001_Head_muscles_0_1 | 28 | 0.0065937 | 0.0000% | 28 |
| excluded | 275 | Lateral pterygoid002_Head_muscles_0_0 | 1,786 | 0.0125007 | 0.0013% | 0 |
| excluded | 276 | Lateral pterygoid002_Head_muscles_0_1 | 1,718 | 0.00789493 | 0.0006% | 0 |
| excluded | 285 | Superior oblique002_Head_muscles_0_0 | 410 | 0.0111221 | 0.0001% | 0 |
| excluded | 286 | Superior oblique002_Head_muscles_0_1 | 519 | 0.00797923 | 0.0001% | 0 |
| excluded | 289 | Geniohyoid001_Head_muscles_0_0 | 1,708 | 0.127236 | 0.1444% | 0 |
| excluded | 290 | Geniohyoid001_Head_muscles_0_1 | 1,701 | 0.132397 | 0.1566% | 0 |
