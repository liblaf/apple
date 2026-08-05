# Human Face Smile Prestrain v2 Final Comparison

## Purpose

This experiment checks whether the estimated skin prestrain is reasonable for the Smile target, then compares inverse activation runs after scaling the optimizer loss by `1e6` into `loss_mm2`. The final additional test is the requested no-skin run at Adam `lr=0.3`.

The optimized scalar is `loss_mm2`, the componentwise mean squared displacement error in `mm^2`. The reported `error_rms_mm` is the vector residual norm RMS over target points.

## Commands

Forward prestrain sanity was run first for the estimated skin prestrain and the estimated-plus-constant-tightening skin prestrain. The required inverse baselines were then run with:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-smile-prestrain-v2
DEBUG=1 CHERRIES_NAME="Human face Smile prestrain v2 200-step baselines loss-mm2" CHERRIES_TAGS="human-face,smile,inverse,skin-prestrain,plus-tightening,loss-mm2,baseline200,local" uv run python src/20-inverse-human-face.py --case-set required --inverse-max-steps 200 --mandatory-baseline-steps 200 --segment-steps 8 --time-budget-hours 10 --reserve-minutes 5 --step-time-budget-s 180 --live-plot-dir figs/live
```

The requested no-skin learning-rate comparison was run with:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-smile-prestrain-v2
DEBUG=1 CHERRIES_NAME="Human face Smile no-skin lr0.3 baseline loss-mm2" CHERRIES_TAGS="human-face,smile,inverse,no-skin,lr0.3,loss-mm2,baseline200,local" uv run python src/20-inverse-human-face.py --case-set no-skin --inverse-lr 0.3 --inverse-max-steps 200 --mandatory-baseline-steps 200 --segment-steps 8 --time-budget-hours 10 --reserve-minutes 5 --step-time-budget-s 180 --live-plot-dir figs/live --output-summary data/22-no-skin-lr03-summary.json --output-table data/22-no-skin-lr03-table.md
```

Final comparison plots were generated from `data/23-final-comparison-summary.json`:

```bash
DEBUG=1 CHERRIES_NAME="Human face Smile final loss curves with no-skin lr0.3" CHERRIES_TAGS="human-face,smile,inverse,plots,loss-mm2,no-skin-lr0.3,local" uv run python src/30-plot-loss-curves.py --comparison-summary data/23-final-comparison-summary.json --output-dir figs/23-final-loss-curves
```

## Outputs

Use these for ParaView:

- No-skin `lr=0.3` best mesh: `data/20-human-face-smile-no-skin-lr3.vtu`
- No-skin `lr=0.3` target mesh: `data/20-human-face-smile-no-skin-lr3-target.vtu`
- No-skin `lr=0.3` temporal history: `data/20-human-face-smile-no-skin-lr3-steps.vtkhdf`
- No-skin `lr=0.3` trace: `data/20-human-face-smile-no-skin-lr3-trace.jsonl`
- No-skin `lr=0.3` live plot: `figs/live/20-human-face-smile-no-skin-lr3-live-log-loss.png`
- Final comparison JSON: `data/23-final-comparison-summary.json`
- Final comparison table: `data/23-final-comparison-table.md`
- Final loss comparison plot: `figs/23-final-loss-curves/loss-comparison-log-y.png`

The prestrain sanity meshes are:

- `data/10-smile-isface-skin-estimated-prestrain.vtp`
- `data/10-smile-isface-skin-estimated-plus-tightening.vtp`
- `data/15-estimated-skin-prestrain-forward.vtu`
- `data/16-estimated-plus-tightening-forward.vtu`

## Results

| display | case | group | best step | loss mm2 | RMS mm | max mm | disp lap RMS | residual lap RMS | activation max | forward fails | last forward | history frames | result | history |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus baseline | 20-human-face-smile-skin-estimated-plus-tightening-lr1 | required baseline | 192 | 1.67666 | 2.24276 | 8.37382 | 8.1997e-05 | 0.00010042 | 10.9115 | 2 | primary_success | 201 | 20-human-face-smile-skin-estimated-plus-tightening-lr1.vtu | 20-human-face-smile-skin-estimated-plus-tightening-lr1-steps.vtkhdf |
| skin baseline | 20-human-face-smile-skin-no-prestrain-lr1 | required baseline | 190 | 1.22221 | 1.91484 | 8.68896 | 0.000141059 | 0.000154393 | 4.50374 | 1 | primary_success | 201 | 20-human-face-smile-skin-no-prestrain-lr1.vtu | 20-human-face-smile-skin-no-prestrain-lr1-steps.vtkhdf |
| no skin lr1 baseline | 20-human-face-smile-no-skin-lr1 | required baseline | 5 | 4.39302 | 3.6303 | 11.9839 | 0.000241191 | 0.00025164 | 2.99328 | 155 | max_steps_reached | 201 | 20-human-face-smile-no-skin-lr1.vtu | 20-human-face-smile-no-skin-lr1-steps.vtkhdf |
| plus warm cont | 20-human-face-smile-skin-estimated-plus-tightening-lr2-cont-lr02-warm-from-best | adaptive continuation | 9 | 1.65853 | 2.2306 | 8.32594 | 8.20248e-05 | 0.00010045 | 10.915 | 0 | primary_success | 49 | 20-human-face-smile-skin-estimated-plus-tightening-lr2-cont-lr02-warm-from-best.vtu | 20-human-face-smile-skin-estimated-plus-tightening-lr2-cont-lr02-warm-from-best-steps.vtkhdf |
| skin warm cont | 20-human-face-smile-skin-no-prestrain-lr3-cont-lr03-from-best | adaptive continuation | 1 | 1.22215 | 1.91479 | 8.68927 | 0.000141059 | 0.000154393 | 4.50374 | 0 | primary_success | 81 | 20-human-face-smile-skin-no-prestrain-lr3-cont-lr03-from-best.vtu | 20-human-face-smile-skin-no-prestrain-lr3-cont-lr03-from-best-steps.vtkhdf |
| no skin lr0.3 | 20-human-face-smile-no-skin-lr3 | requested lr comparison | 194 | 0.133797 | 0.633555 | 3.88455 | 0.00021348 | 0.000215499 | 2.27972 | 6 | primary_success | 201 | 20-human-face-smile-no-skin-lr3.vtu | 20-human-face-smile-no-skin-lr3-steps.vtkhdf |

## Analysis

The no-skin `lr=0.3` run is a major scalar-loss improvement over the no-skin `lr=1.0` baseline. It completed the full 200-step mandatory baseline, accepted best step 194, and had only 6 forward max-step misses, versus 155 misses for no-skin `lr=1.0`.

That result should not be read as proof that no skin is visually better. Its residual Laplacian RMS is `0.000215499`, worse than both skin cases (`0.000154393` for no-prestrain skin and `0.00010045` for plus-tightening skin). The no-skin model is likely using the missing surface stiffness to fit target points more freely. It is worth opening in ParaView because the scalar fit is strong, but the keep/discard decision should be based on visual smoothness and residual field quality, not loss alone.

Among skin-enabled cases, the estimated-plus-tightening setup is smoother by residual Laplacian and displacement Laplacian, but fits less tightly than no-prestrain skin. The warm continuation improved plus-tightening slightly (`1.67666 -> 1.65853 loss_mm2`) without changing its roughness much. The no-prestrain continuation did not materially improve its baseline.

## Verification

Readback checks passed for:

- No-skin `lr=0.3` `.vtu` result and target: PyVista loaded both with 228,660 points and 1,146,517 cells.
- No-skin `lr=0.3` `.vtkhdf`: h5py opened `VTKHDF` with 201 history frames.
- No-skin `lr=0.3` trace: 201 JSONL rows, steps 0 through 200, last new best at step 194.
- No-skin `lr=0.3` live PNG: 1120 x 720 RGBA.
- Final comparison JSON/table and final loss plots exist and were regenerated after excluding smoke and aborted cold-start artifacts.

## Recommendation

For prestrain reasonableness, inspect the plus-tightening skin result first because it is the smoothest skin-enabled run and the forward sanity pass showed plausible surface shrink. Then inspect no-skin `lr=0.3` as a diagnostic lower-bound on achievable point-fit loss. If the no-skin result looks visibly bumpy or physically loose, keep the skin-enabled plus-tightening result as the plausible prestrain candidate and treat no-skin `lr=0.3` as evidence that extra regularization or skin tuning is needed rather than evidence to remove skin.
