# 20 Inverse Face Worklog

This file records the live experiment decisions and findings while the inverse
solve is still in progress. The final report remains `20-inverse-face.md`.

## 2026-05-26

- Prepared the lower-resolution face problem from
  `/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/41-expression-515k.vtu`.
- The prepare script selected tetrahedra with `InFaceConvex` and wrote:
  - `data/10-inverse-face-input.vtu`
  - `data/10-inverse-face-target.vtu`
- Prepared problem size:
  - points: `58651`
  - cells: `253876`
  - active tets: `58494`
  - target `IsFace` points: `6787`
  - target max displacement: `0.965678 cm`

## Attempts

### No Smooth Regularization

Command:

```bash
DEBUG=false CHERRIES_NAME='inverse face 515k fresh conservative' CHERRIES_TAGS='inverse-face,inverse,515k,fresh,conservative,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --inverse-lr 0.01 --adam-beta1 0.3 --adam-beta2 0.9 --inverse-max-steps 1000 --inverse-min-steps 80 --stagnation-patience 250 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8
```

Finding:

- Stopped when retuning was requested.
- Best observed max face error was about `0.550 cm` by step `137`.
- Forward and adjoint solves were healthy in the observed trace.
- Activation field was visually/qualitatively too bumpy.

### Smooth Weight 0.01

Command:

```bash
DEBUG=false CHERRIES_NAME='inverse face 515k smooth conservative' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,fresh,conservative,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --inverse-lr 0.01 --adam-beta1 0.3 --adam-beta2 0.9 --activation-smooth-weight 0.01 --inverse-max-steps 1000 --inverse-min-steps 80 --stagnation-patience 250 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8
```

Findings:

- Added face-adjacency smoothness on neighboring active `activation_inv` rows.
- Smooth neighbor pairs: `100783`.
- The run improved the objective and reduced max face error below the
  unregularized attempt.
- Latest useful checkpoint before stopping:
  - best step: `222`
  - best total loss: `0.0033034645853156036`
  - best max face error: `0.4190477217600812 cm`
  - target RMS error: `0.09120532583099183 cm`
- Step `224` jumped to max face error `0.522686083251468 cm`, while the best
  checkpoint remained at step `222`.
- Forward and adjoint solves remained converged in the observed trace.
- User visual assessment: smooth weight `0.01` is still not strong enough;
  resulting activation remains a bit bumpy.

Decision:

- Stop the weak smoothness run.
- Increase smooth regularization for the next run.
- Add PyVista snapshot output for quick visual inspection of the result mesh.

## 2026-05-27 00:10 CST

- Confirmed weak smoothness process was terminated.
- Changed the default smooth regularization weight from `0.01` to `0.05`.
- Added a simple PyVista result snapshot output:
  - left: deformed surface colored by displacement error norm
  - right: clipped volume colored by `RecoveredActivationInvNorm`
- Script checks passed:
  - `uv run ruff format exp/2026/05/20/inverse-face/src/20-inverse-face.py`
  - `uv run ruff check exp/2026/05/20/inverse-face/src/20-inverse-face.py`
  - `uv run python -m py_compile exp/2026/05/20/inverse-face/src/20-inverse-face.py`

### Smooth Weight 0.05

Command:

```bash
DEBUG=false CHERRIES_NAME='inverse face 515k smooth strong fresh' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,strong,fresh,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --inverse-lr 0.01 --adam-beta1 0.3 --adam-beta2 0.9 --activation-smooth-weight 0.05 --inverse-max-steps 1000 --inverse-min-steps 80 --stagnation-patience 250 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8
```

Comet:

- `https://www.comet.com/liblaf/apple/a2345924bf684c87bd447148d48200ac`

Early trace:

- Step `0` saved the zero-activation state.
- Step `0` max face error: `0.9656781194874481 cm`.
- Step `1` max face error: `0.9648507326333788 cm`.
- Smooth weight is logged as `0.05`; smooth neighbor pairs remain `100783`.
- Forward and adjoint solves are healthy in the first two steps.

Progress update:

- Step `25` max face error: `0.9374677073835271 cm`.
- Step `25` target RMS error: `0.3303530451886248 cm`.
- Step `25` smooth loss: `0.007683306652718549`.
- Step `25` regularization contribution: `0.0003841653326359275`.
- The stronger smoothness run is more conservative than `0.01`, but the trace
  is monotone so far and solver residuals remain within tolerance.
- Step `50` max face error: `0.85816844725637 cm`.
- Step `50` target RMS error: `0.2995125404642581 cm`.
- Step `50` smooth loss: `0.015066550113881829`.
- Step `50` regularization contribution: `0.0007533275056940915`.
- The max-error decrease accelerated after the first 30 steps, so continue the
  `0.05` run rather than retuning immediately.
- Step `70` max face error: `0.7999782052867275 cm`.
- Step `70` target RMS error: `0.2667058640029993 cm`.
- Step `70` smooth loss: `0.020845487153975734`.
- The run is still solver-clean and monotone on the recorded best max error.
- Step `100` max face error: `0.679832543987208 cm`.
- Step `100` target RMS error: `0.2013307279602368 cm`.
- Step `100` smooth loss: `0.02769403672004294`.
- Step `103` max face error: `0.6736478494826544 cm`.
- Step `103` target RMS error: `0.1948346175781037 cm`.
- Step `103` smooth loss: `0.028100462597656677`.
- Step `110` max face error: `0.6559143934820989 cm`.
- Step `110` target RMS error: `0.1804349988552772 cm`.
- Step `110` smooth loss: `0.028597397728848867`.
- The run remains healthy: no observed repeated forward or adjoint failures, and
  best max error is still improving.
- Step `122` max face error: `0.6125822252331934 cm`.
- Step `122` target RMS error: `0.16007591882158767 cm`.
- Step `122` smooth loss: `0.0286936851370196`.
- Step `136` max face error: `0.543335157799551 cm`.
- Step `136` target RMS error: `0.14055731320048273 cm`.
- Step `136` smooth loss: `0.028396493491599638`.
- Step `148` max face error: `0.5113292812654995 cm`.
- Step `148` target RMS error: `0.12947377728490744 cm`.
- Step `148` smooth loss: `0.027803695459368064`.
- Step `160` max face error: `0.507341568458475 cm`.
- Step `160` target RMS error: `0.12084560847655962 cm`.
- Step `160` smooth loss: `0.027085250999680077`.
- Step `181` current best remains step `178`, with max face error
  `0.4859717565439491 cm`.
- Step `181` target RMS error: `0.10974966239557434 cm`.
- The worst-point error is improving much more slowly than the RMS error. This
  suggests the `0.05` smoothness weight may be over-regularized for the final
  `<0.2 cm` pointwise target, although the solve is still numerically healthy.
- Generated a quick PyVista snapshot from step `191`:
  `data/20-inverse-face-step191.png`.
- The first activation-volume panel camera was not useful, but the face-error
  panel showed the remaining high error is localized rather than a global
  mismatch.
- A transient Comet status-report timeout occurred during the run; the local
  optimizer continued and later metrics kept logging.
- Step `191` max face error: `0.4704078103377334 cm`.
- Step `191` target RMS error: `0.1086394308040651 cm`.
- Step `218` max face error: `0.45385750722114276 cm`.
- Step `218` target RMS error: `0.1004675211802974 cm`.
- Step `229` max face error: `0.4488005382442577 cm`.
- Step `229` target RMS error: `0.09623715144393247 cm`.
- Step `243` max face error: `0.4360592438204695 cm`.
- Step `243` target RMS error: `0.09266411893705286 cm`.
- Keep the `0.05` run alive because the saved-best max error is still dropping,
  albeit slowly.

## 2026-05-27 00:55 CST

- Added warm-state resume support to the inverse script:
  - checkpoints written after this code change include the best full
    displacement state in addition to active `activation_inv`
  - `--initial-activation-inv` can now warm-start the forward state from a `.npz`
    checkpoint containing `displacement`, or from a `.vtu` with point data
    `Displacement`
- Reason: a checkpoint-only activation continuation was tested and immediately
  gave a worse cold-start max error (`0.8494990393743189 cm`) than the saved
  warm trajectory (`0.4190477217600812 cm` at step `222`). The forward solve
  state matters, so future resume attempts should carry displacement too.
- The default `w=0.05` run is still active on the default output paths.
- Step `209` checkpoint:
  - max face error: `0.4650297822269738 cm`
  - loss: `0.004670744638742033`
  - checkpoint arrays were written by the already-running process, so this
    particular checkpoint does not yet include the new `displacement` field.
- Interpretation: `w=0.05` is smoother and still stable, but remains slower on
  worst-point error than the earlier `w=0.01` run. Continue monitoring while it
  keeps improving; if it plateaus well above `0.2 cm`, the likely next run is a
  moderate smoothness weight between `0.01` and `0.05`.

## 2026-05-27 01:19 CST

- Step `249` checkpoint:
  - max face error: `0.43192858819766905 cm`
  - loss: `0.004149626089253162`
- Latest inspected frame `262`:
  - current max face error: `0.43697081162197693 cm`
  - current RMS face error: `0.09330003435142509 cm`
  - current total loss: `0.004080611529088379`
  - smooth loss: `0.023579587848586884`
  - regularization contribution: `0.0011789793924293444`
- Finding: total loss can continue improving while the worst face point gets
  worse. Since the success criterion is pointwise max error `<0.2 cm`, future
  runs now use significant decrease in max face error for stagnation accounting,
  while still retaining the explicit `loss_tol` stop.
- Script checks passed again after the stagnation change:
  - `uv run ruff format exp/2026/05/20/inverse-face/src/20-inverse-face.py`
  - `uv run ruff check exp/2026/05/20/inverse-face/src/20-inverse-face.py`
  - `uv run python -m py_compile exp/2026/05/20/inverse-face/src/20-inverse-face.py`

## 2026-05-27 01:40 CST

- A new `w=0.02` run is active:

```bash
uv run python src/20-inverse-face.py --inverse-lr 0.01 --adam-beta1 0.3 --adam-beta2 0.9 --activation-smooth-weight 0.02 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 400 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8
```

- Current process:
  - parent `uv` PID: `1233306`
  - Python PID: `1233319`
- Latest checkpoint:
  - step: `129`
  - max face error: `0.5777285020257367 cm`
  - loss: `0.008169087230695284`
  - checkpoint includes both active `activation_inv` and full displacement
    state.
- The run is still improving and solver residuals remain within tolerance, so
  keep it alive.

## 2026-05-27 01:42 CST

- User visual assessment: `w=0.02` is still not smooth enough.
- Stopped the active `w=0.02` process.
- Last inspected `w=0.02` frame:
  - step: `130`
  - max face error: `0.5669971134381349 cm`
  - RMS face error: `0.14639183857847926 cm`
  - loss: `0.008029522873452861`
- Next run: try activation smooth regularization weight `1.0` with otherwise
  similar Adam settings and separate output paths.
- First launch failed before experiment startup because the inherited
  environment had `DEBUG=false`, which Cherries/environs does not accept as a
  boolean. Relaunch with `DEBUG` unset.

### Smooth Weight 1.0, Learning Rate 0.01

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --activation-smooth-weight 1.0 --inverse-lr 0.01 --adam-beta1 0.3 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 400 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1.vtu --output-series data/20-inverse-face-smooth-w1.vtu.series --output-summary data/20-inverse-face-smooth-w1-summary.json --output-snapshot data/20-inverse-face-smooth-w1.png --checkpoint data/20-inverse-face-smooth-w1-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/e699bdbd914748698ae18dfbe857a3d7`

Finding:

- Stopped for learning-rate retune.
- Latest observed step:
  - step: `33`
  - max face error: `0.9326037823106482 cm`
  - RMS face error: `0.3314180725222323 cm`
  - loss: `0.03727923786337006`
- Forward and adjoint solves were clean, but progress was too slow for the
  requested long solve.

Decision:

- Keep `activation_smooth_weight = 1.0`.
- Try more aggressive Adam learning rate `0.03`.

### Smooth Weight 1.0, Learning Rate 0.03

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 lr 0.03' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,lr003,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --activation-smooth-weight 1.0 --inverse-lr 0.03 --adam-beta1 0.3 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 400 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-lr003.vtu --output-series data/20-inverse-face-smooth-w1-lr003.vtu.series --output-summary data/20-inverse-face-smooth-w1-lr003-summary.json --output-snapshot data/20-inverse-face-smooth-w1-lr003.png --checkpoint data/20-inverse-face-smooth-w1-lr003-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/6ee0f0933264480f8bd690dc6b156b59`

Early finding:

- Step `22`:
  - max face error: `0.8377363593962157 cm`
  - RMS face error: `0.30202653983428435 cm`
  - loss: `0.03236507681057172`
- This is much faster than `lr=0.01` at the same smoothness weight.
- Forward and adjoint solves are still converged, so keep the run alive.

Progress:

- Step `64`: best max face error `0.627340912143386 cm`.
- Step `81`: best max face error `0.5882497758981433 cm`.
- Step `104`: best max face error `0.5598718559511515 cm`;
  RMS face error `0.17675473151651566 cm`.
- The run has intermittent max-error jumps, but saved-best max error is still
  improving. Forward and adjoint solves remain converged.
- Step `136`: best max face error `0.519633694285782 cm`; loss
  `0.014286165654857451`.
- Step `155`: current max face error `0.5487865471365609 cm`, RMS face error
  `0.16834712740226268 cm`, loss `0.013456985214079679`; saved best remains
  step `136`.
- Step `166`: new saved best max face error `0.5097637068502818 cm`; RMS face
  error `0.1652534573871176 cm`; loss `0.013147623074015097`. This confirms
  the `0.03` learning rate is still improving the saved-best solution after
  temporary max-error oscillations.
- Step `206`: new saved best max face error `0.5011808774202496 cm`; RMS face
  error `0.15135058283475686 cm`; loss `0.011406617528322168`.
- Between steps `166` and `206`, total/data loss kept trending lower while max
  face error oscillated around the saved best. The near-best frames around steps
  `201` and `204` were a useful sign that the lower-loss basin was not merely
  reducing average error at the expense of the worst face point.
- Step `208`: new saved best max face error `0.4724317814645711 cm`; RMS face
  error `0.15338749054475984 cm`; loss `0.011656140118146287`.
- Stopped the `lr=0.03` run manually after step `256`. Best checkpoint is
  step `208`; later steps lowered some average-loss states but repeatedly
  bounced back above the saved max-error best, with steps `254` and `255`
  increasing total loss and forward work.
- Interpretation: `lr=0.03` is solver-clean and much faster than `lr=0.01`,
  but the max-error objective is noisy under the stronger smoothness term.
  Continue while the stagnation counter is low and saved-best max error may
  still improve in bursts.

### Smooth Weight 1.0, Learning Rate 0.015 Restart

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 lr 0.015 from lr 0.03 best' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,lr0015,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-lr003-checkpoint.npz --activation-smooth-weight 1.0 --inverse-lr 0.015 --adam-beta1 0.2 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 400 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-lr0015-from-lr003.vtu --output-series data/20-inverse-face-smooth-w1-lr0015-from-lr003.vtu.series --output-summary data/20-inverse-face-smooth-w1-lr0015-from-lr003-summary.json --output-snapshot data/20-inverse-face-smooth-w1-lr0015-from-lr003.png --checkpoint data/20-inverse-face-smooth-w1-lr0015-from-lr003-checkpoint.npz
```

Reason:

- Restart from the saved `lr=0.03` best checkpoint instead of the latest
  oscillating state.
- Use a calmer but still aggressive learning rate `0.015` and lower Adam
  momentum `beta1 = 0.2`.

Comet:

- `https://www.comet.com/liblaf/apple/11840aed3d8b4145a05006455a34d53a`

Early finding:

- Restart step `0` reproduces the saved checkpoint with max face error
  `0.4724015530048676 cm` and RMS face error `0.153476432662903 cm`.
- Initial forward solve from the restored displacement required more work
  (`4372` PNCG steps with line search), but subsequent steps returned to the
  usual converged primary path.
- Stopped after step `20`; this branch did not improve the checkpoint and had
  large max-error spikes (`0.6720061407972395 cm` at step `18`,
  `0.6445331927036622 cm` at step `20`).

### Smooth Weight 1.0, Learning Rate 0.01 Restart, No Momentum

Decision:

- Try the same checkpoint with `inverse_lr = 0.01` and `adam_beta1 = 0.0`.
- Goal is to keep Adam but remove first-moment overshoot near the saved best.

Comet:

- `https://www.comet.com/liblaf/apple/1fbe00ab49d043ae943d2d6de128f6cf`

Early finding:

- Restart step `0` max face error: `0.47240108241019935 cm`.
- First few steps lower total/data loss but do not beat max error yet; the
  branch is less violent than the `lr=0.015` restart, so keep it running for
  more steps before judging it.
- Stopped after step `16`. This restart also failed to improve the saved max
  error; average loss decreased, but max error stayed around `0.50 cm` and
  sometimes jumped higher.

### Add Explicit Max-Error Data Term

Change:

- Added `target_max_loss_weight` to the inverse script.
- Data objective is now optionally:
  `mean(residual^2) + target_max_loss_weight * max(point_error)^2`.
- Default remains `0.0`; previous commands reproduce the old behavior unless
  the new option is provided.

Reason:

- The restart experiments show that plain MSE can reduce average/RMS error
  while not improving the worst face point. The success condition is max face
  point error below `0.2 cm`, so the objective needs a direct but simple term
  for that quantity.

### Smooth Weight 1.0, Max Loss 0.05, Learning Rate 0.01

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 0.05 lr 0.01 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss005,lr001,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-lr003-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 0.05 --inverse-lr 0.01 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 400 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max005-lr001-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max005-lr001-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max005-lr001-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max005-lr001-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max005-lr001-beta10-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/86d40717cdfd4b4899048ecd36696884`

Early finding:

- Step `0` max face error: `0.4724015877554274 cm`.
- Step `4` max face error: `0.4600404071455367 cm`.
- Step `12` max face error: `0.4410419911134384 cm`.
- Step `20` max face error: `0.42825452852444074 cm`.
- The explicit max-error data term is improving the saved max-error state,
  unlike the MSE-only restarts. Continue this branch.

Progress:

- Step `42` became the current best with max face error
  `0.3959917249501161 cm`, RMS face error `0.14565615706329824 cm`, and
  total loss `0.01918453185155822`.
- Steps `43` through `64` oscillated around the best instead of running away.
  The closest later point so far is step `63` with max face error
  `0.39874261246893034 cm`.
- Forward solves remain primary-success and adjoint solves remain successful.
  This is still the most promising branch, so continue it before branching to a
  smaller learning rate from the best checkpoint.
- Step `76` improved the best max face error to `0.3922708443300964 cm`.
  The run then oscillated higher, but recovered instead of diverging.
- Step `102` improved the best max face error to `0.3817678045178647 cm`,
  with RMS face error `0.14843170227457803 cm` and total loss
  `0.01939259880094148`. The branch still has slow sawtooth behavior, but the
  envelope continues downward.
- Step `133` improved the best max face error to `0.3702991079420246 cm`.
- Step `148` improved the best max face error to `0.36471768794557097 cm`.
- Step `159` improved the best max face error to `0.3641446954139306 cm`.
  The best-error envelope is still decreasing, although the rate is now slow.

Stopped manually after step `204` because the branch had gone `30` steps
without beating the step `174` best. The final logged current step had max face
error `0.3641532085957258 cm`, but the saved checkpoint remained step `174`
with max face error `0.36270540796442446 cm`.

### Smooth Weight 1.0, Max Loss 0.2, Learning Rate 0.006

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 0.2 lr 0.006 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss02,lr0006,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max005-lr001-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 0.2 --inverse-lr 0.006 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max02-lr0006-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max02-lr0006-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max02-lr0006-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max02-lr0006-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max02-lr0006-beta10-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/b11adf9d874e49a38e1a380730a2dc35`

Early finding:

- Restart step `0` reproduced the checkpoint within solver tolerance with max
  face error `0.3627486224137097 cm`.
- Step `8` improved the best max face error to `0.36143052650335405 cm`.
- Step `13` improved the best max face error to `0.35787610164701644 cm`.
- Step `24` improved the best max face error to `0.3467935415332599 cm`.
- Step `36` improved the best max face error to `0.3438742465588682 cm`.
- The first restored forward solve hit `max_steps_reached`, but subsequent
  forward solves are primary-success. The stronger max-loss term is currently
  more effective than the `0.05` branch.
- Step `48` improved the best max face error to `0.33759747257847417 cm`.
- Step `55` improved the best max face error to `0.33373620330453907 cm`.
- Step `85` improved the best max face error to `0.33039988329312636 cm`,
  with RMS face error `0.13498169481649822 cm` and total loss
  `0.0333365033179537`. The run is oscillatory but still lowering the best
  max-error envelope.
- Step `100` improved the best max face error to `0.3280007547645758 cm`.
- Step `125` improved the best max face error to `0.32789616700122437 cm`.
  The branch was stopped manually after step `131`; it was still stable, but
  the improvement rate after step `100` was very small.

### Smooth Weight 1.0, Max Loss 0.2, Learning Rate 0.03

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 0.2 lr 0.03 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss02,lr003,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max02-lr0006-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 0.2 --inverse-lr 0.03 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max02-lr003-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max02-lr003-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max02-lr003-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max02-lr003-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max02-lr003-beta10-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/e5ed95fd2f354d97973fbd1123c9b6b5`

Finding:

- Restart step `0` reproduced the checkpoint with max face error
  `0.32785139551639375 cm`.
- Steps `1` through `5` worsened the max face error to the `0.42-0.58 cm`
  range. The run was stopped manually because learning rate `0.03` is too hot
  from this checkpoint.

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.003

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 lr 0.003 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,lr0003,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max02-lr0006-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --inverse-lr 0.003 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-lr0003-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max1-lr0003-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-lr0003-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-lr0003-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max1-lr0003-beta10-checkpoint.npz
```

Comet:

- `https://www.comet.com/liblaf/apple/0a33e6be839c4dc685f5e7e6e32d0748`

Early finding:

- Restart step `0` reproduced the checkpoint with max face error
  `0.3278513769658943 cm`.
- Step `4` improved the best max face error to `0.3272717616020751 cm`.
- Step `6` improved the best max face error to `0.32615294919671955 cm`.
- Step `10` improved the best max face error to `0.32259811010795375 cm`.
  This is a better continuation than learning rate `0.03`: it still oscillates,
  but the best max-error envelope is moving down.
- Step `25` improved the best max face error to `0.3197379013413725 cm`.
- Step `41` improved the best max face error to `0.3175350952260043 cm`.
- Step `49` improved the best max face error to `0.3162605665062138 cm`.
- Step `76` improved the best max face error to `0.31327605533504155 cm`.
- Step `89` improved the best max face error to `0.3101532646446417 cm`.
- Step `111` improved the best max face error to `0.3072304717534322 cm`.
- Step `121` improved the best max face error to `0.30702612327045803 cm`.
- Step `130` improved the best max face error to `0.30667410669626644 cm`.
- Step `136` improved the best max face error to `0.30530527942782004 cm`.
- Step `149` improved the best max face error to `0.3048593815806022 cm`.
- Step `156` improved the best max face error to `0.30383821828998797 cm`.
  The branch remains slow but healthy; the `0.03` learning-rate branch was
  faster only in the bad direction, while this run continues lowering the
  max-error envelope.
- Step `172` improved the best max face error to `0.30360598737916444 cm`
  after a 15-step sawtooth.
- Step `175` improved the best max face error to `0.30282201374092765 cm`.
- Step `189` improved the best max face error to `0.30234279804772407 cm`.
- Step `192` improved the best max face error to `0.3008226958170033 cm`.
- Step `203` improved the best max face error to `0.3004037458775499 cm`.
- Step `207` improved the best max face error to `0.2982452948339013 cm`,
  breaking below `0.3 cm`.
- Step `214` improved the best max face error to `0.29779184496091793 cm`.
- The branch was stopped manually after step `239` with `25` no-improve steps.
  The best checkpoint is
  `data/20-inverse-face-smooth-w1-max1-lr0003-beta10-checkpoint.npz`.

### Smooth Weight 1.0, Max Loss 2.0, Learning Rate 0.002

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 2 lr 0.002 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss2,lr0002,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr0003-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 2.0 --inverse-lr 0.002 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max2-lr0002-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max2-lr0002-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max2-lr0002-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max2-lr0002-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max2-lr0002-beta10-checkpoint.npz
```

Early finding:

- Started from the max-loss `1.0`, learning-rate `0.003` best checkpoint at
  step `214`, max face error `0.29779184496091793 cm`.

### Fresh MSE-Only, Smooth 0.1, Activation L2 0.1, Learning Rate 0.03

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k fresh smooth 0.1 l2 0.1 mse only lr 0.03 beta1 0.5' CHERRIES_TAGS='inverse-face,inverse,515k,fresh,smooth,w01,l2,reg01,mseonly,nomax,nohinge,lr003,beta105,beta209,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --activation-smooth-weight 0.1 --activation-l2-weight 0.1 --inverse-lr 0.03 --adam-beta1 0.5 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.vtu --output-series data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.vtu.series --output-summary data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105-summary.json --output-snapshot data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.png --checkpoint data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105-checkpoint.npz
```

Live finding:

- This branch uses only MSE face displacement data loss plus smooth activation
  regularization and activation L2 regularization. Target max loss and hinge
  loss have been removed from the code.
- Comet: `https://www.comet.com/liblaf/apple/392524890bc846eaa955f8ead18f46f0`.
- Fresh start step `0` max face error was `0.965678 cm`.
- Step `40` reached max face error `0.611726 cm`.
- Step `60` reached max face error `0.512016 cm`.
- Step `77` reached max face error `0.491712 cm`.
- Step `100` reached max face error `0.481329 cm`.
- Step `111` reached max face error `0.472677 cm`.
- Step `120` reached max face error `0.468968 cm`.
- Activation RMS is decreasing slowly under L2 `0.1` while the data term still
  improves, so the branch is numerically healthy but still far above the
  required `0.2 cm` max error.

### Fresh MSE-Only Run With Activation Size Regularization

User correction:

- Start from fresh instead of a warm-start checkpoint.
- Use learning rate `0.03`, Adam `beta1 = 0.5`, `beta2 = 0.9`.
- Do not use max-error as a loss term.
- Add a regularization term to keep activation from becoming too large.

Implementation:

- Added `activation_l2_weight` to `20-inverse-face.py`.
- The objective is now configurable as:
  `mse + activation_smooth_weight * smooth + activation_l2_weight * mean(activation_inv^2)`.
- Max error and hinge loss remain logged as metrics, but the fresh run sets
  both corresponding weights to `0`.

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k fresh smooth 0.1 l2 0.001 mse only lr 0.03 beta1 0.5' CHERRIES_TAGS='inverse-face,inverse,515k,fresh,smooth,w01,l2,reg001,mseonly,nomax,nohinge,lr003,beta105,beta209,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --activation-smooth-weight 0.1 --activation-l2-weight 0.001 --target-max-loss-weight 0.0 --target-hinge-loss-weight 0.0 --inverse-lr 0.03 --adam-beta1 0.5 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-fresh-smooth-w01-l2-001-mseonly-lr003-beta105.vtu --output-series data/20-inverse-face-fresh-smooth-w01-l2-001-mseonly-lr003-beta105.vtu.series --output-summary data/20-inverse-face-fresh-smooth-w01-l2-001-mseonly-lr003-beta105-summary.json --output-snapshot data/20-inverse-face-fresh-smooth-w01-l2-001-mseonly-lr003-beta105.png --checkpoint data/20-inverse-face-fresh-smooth-w01-l2-001-mseonly-lr003-beta105-checkpoint.npz
```

### Remove Max and Hinge Loss Code

User correction:

- Remove target max loss and hinge loss from the script, not just from the run
  configuration.
- Try activation L2 weight `0.1`.

Implementation:

- Removed `target_max_loss_weight` and `target_hinge_loss_weight` config fields.
- Removed max-error and hinge-loss objective terms and their loss logs.
- Kept `target/error_max` as an evaluation metric and stopping criterion only.
- `py_compile` passed after the edit.

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k fresh smooth 0.1 l2 0.1 mse only lr 0.03 beta1 0.5' CHERRIES_TAGS='inverse-face,inverse,515k,fresh,smooth,w01,l2,reg01,mseonly,nomax,nohinge,lr003,beta105,beta209,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --activation-smooth-weight 0.1 --activation-l2-weight 0.1 --inverse-lr 0.03 --adam-beta1 0.5 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.vtu --output-series data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.vtu.series --output-summary data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105-summary.json --output-snapshot data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105.png --checkpoint data/20-inverse-face-fresh-smooth-w01-l2-01-mseonly-lr003-beta105-checkpoint.npz
```

### Smooth Weight 1.0, Max Loss 1.0, Hinge 20, Learning Rate 0.001, Continuation 6

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 hinge 20 lr 0.001 beta1 0 cont6' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,hinge20,lr0001,beta10,restart,cont6,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --target-hinge-loss-weight 20.0 --inverse-lr 0.001 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6.vtu --output-series data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6.png --checkpoint data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/4836120b83864b5097d061f382ea7863`.
- Started from the hinge-20, learning-rate `0.0015` best checkpoint, max face
  error `0.2844479074602095 cm`.
- Restart step `0` reproduced max face error `0.2844453579604742 cm`.
- Improvements were slow but useful: step `7` reached `0.2843005256166111 cm`,
  step `13` reached `0.28399609612516774 cm`, step `24` reached
  `0.2834037005979077 cm`, step `47` reached `0.28189051500554324 cm`, and
  step `71` reached `0.2811030148268988 cm`.
- The run was stopped manually at step `100` after 29 non-improving steps and
  several hotter spikes, including a max face error above `0.31 cm`.
- Best checkpoint saved:
  `data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6-checkpoint.npz`.
- Next attempt: continue from this checkpoint with learning rate `0.0007`.

### Smooth Weight 1.0, Max Loss 1.0, Hinge 20, Learning Rate 0.0007, Continuation 7

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 hinge 20 lr 0.0007 beta1 0 cont7' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,hinge20,lr00007,beta10,restart,cont7,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-hinge20-lr0001-beta10-cont6-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --target-hinge-loss-weight 20.0 --inverse-lr 0.0007 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7.vtu --output-series data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7.png --checkpoint data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/feee80ee24714b80bfa3edf7ec6e45aa`.
- Restart step `0` reproduced max face error `0.2811233589935371 cm`.
- Step `2` improved slightly to `0.28109085577149007 cm`; step `3` improved
  to `0.2810175545292794 cm`.
- Steps `7` through `21` jumped into a bad tail-error branch, peaking at
  `0.3168059611947451 cm` and only recovering to `0.29306716562472407 cm`.
- The branch was stopped manually at step `21`.
- Best checkpoint saved:
  `data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7-checkpoint.npz`.
- Next attempt: continue from this checkpoint with learning rate `0.0003`.

### Smooth Weight 1.0, Max Loss 1.0, Hinge 20, Learning Rate 0.0003, Continuation 8

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 hinge 20 lr 0.0003 beta1 0 cont8' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,hinge20,lr00003,beta10,restart,cont8,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-hinge20-lr00007-beta10-cont7-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --target-hinge-loss-weight 20.0 --inverse-lr 0.0003 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8.vtu --output-series data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8.png --checkpoint data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/16f2a6c7ef2249d1b1c80a9e108fdf33`.
- Restart step `0` reproduced max face error `0.28101676055687713 cm`.
- The smaller learning rate was productive: step `2` reached
  `0.28083431778547613 cm`, step `10` reached `0.2805841455920753 cm`,
  step `21` crossed below `0.28 cm` at `0.279931202423381 cm`, and step `33`
  reached `0.27978326750909654 cm`.
- After step `37`, the run jumped again into a bad tail branch above `0.31 cm`;
  it was stopped manually at step `44`.
- Best checkpoint saved:
  `data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8-checkpoint.npz`.
- Tail inspection at the best checkpoint:
  max `0.27978326750909654 cm`, mean `0.1048391165112908 cm`, RMS
  `0.12699578191612115 cm`; `882` face points remain above `0.2 cm`, and no
  face points remain above `0.28 cm`.
- Next attempt: increase hinge weight to `100` with learning rate `0.00015` so
  the objective pushes the full above-tolerance tail more strongly.

### Objective Correction: Remove Max-Error Loss

User correction:

- The explicit max-error data term is cheating and should not be used as a
  loss.
- Smooth weight `1.0` may be too strong; try smooth weight `0.1`.

Decision:

- Stop the max-loss and hinge-heavy branch.
- Resume from the best available warm-start checkpoint, but set
  `target_max_loss_weight = 0` and `target_hinge_loss_weight = 0`.
- Keep max face error as a reported metric only.
- Use objective `mse + 0.1 * activation_smooth`.

### Smooth Weight 0.1, MSE Only, Learning Rate 0.003, Continuation 10

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 0.1 mse only lr 0.003 beta1 0 cont10' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w01,mseonly,nomax,nohinge,lr0003,beta10,restart,cont10,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-hinge20-lr00003-beta10-cont8-checkpoint.npz --activation-smooth-weight 0.1 --target-max-loss-weight 0.0 --target-hinge-loss-weight 0.0 --inverse-lr 0.003 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w01-mseonly-lr0003-beta10-cont10.vtu --output-series data/20-inverse-face-smooth-w01-mseonly-lr0003-beta10-cont10.vtu.series --output-summary data/20-inverse-face-smooth-w01-mseonly-lr0003-beta10-cont10-summary.json --output-snapshot data/20-inverse-face-smooth-w01-mseonly-lr0003-beta10-cont10.png --checkpoint data/20-inverse-face-smooth-w01-mseonly-lr0003-beta10-cont10-checkpoint.npz
```

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.0015, Continuation 2 Closeout

Finding:

- The continuation eventually improved beyond the live notes: best step `188`,
  loss `0.09475242891622601`, max face error
  `0.2859763190750222 cm`.
- The run was stopped manually at step `214` after 26 non-improving steps and
  visible drift away from the best state.
- The summary JSON was not written because the manual interrupt happened during
  shutdown, but the checkpoint was saved at
  `data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2-checkpoint.npz`.

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.003, Continuation 3

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 lr 0.003 beta1 0 cont3' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,lr0003,beta10,restart,cont3,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --inverse-lr 0.003 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3.vtu --output-series data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3.png --checkpoint data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/35f619e6fef546de9c24c7a96bbb7b66`.
- Restart step `0` slightly improved the restored state to max face error
  `0.28588963207942014 cm`, but the restored forward solve reached
  `max_steps_reached`.
- Steps `1` through `12` did not beat the restart state; max face error moved
  mostly in the `0.288 cm` to `0.310 cm` range.
- The branch was stopped because learning rate `0.003` was still too jumpy from
  this checkpoint. The best checkpoint remained step `0` at
  `data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3-checkpoint.npz`.
- Next attempt: try learning rate `0.002` from this best state.

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.002, Continuation 4

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 lr 0.002 beta1 0 cont4' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,lr0002,beta10,restart,cont4,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr0003-beta10-cont3-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --inverse-lr 0.002 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4.vtu --output-series data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4.png --checkpoint data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/ef9f781632d64725858efcaacdc5a1ae`.
- Restart step `0` reproduced the best state at max face error
  `0.2858873425461347 cm`; the restored forward solve again reached
  `max_steps_reached`.
- Steps `1` through `10` did not beat step `0`; the best post-update value was
  step `4` at max face error `0.286840696700205 cm`.
- The run was stopped because learning rate `0.002` still overshot and did not
  improve the checkpoint.

### Tail Hinge Objective

Finding:

- At the current best checkpoint, `931` of `6787` face points still exceed the
  required `0.2 cm` point-error tolerance.
- The max-error term is useful but noisy because only the worst point receives
  the extra max-loss gradient.
- Added an optional `target_hinge_loss_weight` term:
  `mean(max(point_error - max_point_error_cm, 0)^2)`.
- The term is disabled by default and will be used only for the next tail
  reduction probe.

### Smooth Weight 1.0, Max Loss 1.0, Hinge 20, Learning Rate 0.0015, Continuation 5

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 hinge 20 lr 0.0015 beta1 0 cont5' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,hinge20,lr00015,beta10,restart,cont5,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr0002-beta10-cont4-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --target-hinge-loss-weight 20.0 --inverse-lr 0.0015 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5.vtu --output-series data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5.png --checkpoint data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5-checkpoint.npz
```

Finding:

- Comet: `https://www.comet.com/liblaf/apple/b615e11fc5b9450e9ab6e3c081fcfc88`.
- Restart step `0` had max face error `0.2858862032838504 cm`;
  hinge penalty was `0.007042944831528593`.
- The branch improved late: step `24` reached `0.28576362536746375 cm`,
  step `29` reached `0.2854799925450994 cm`, and step `34` reached
  `0.2844479074602095 cm`.
- After step `34`, the run wandered for 38 non-improving steps and was stopped
  manually at step `72`.
- Checkpoint saved:
  `data/20-inverse-face-smooth-w1-max1-hinge20-lr00015-beta10-cont5-checkpoint.npz`.
- Next attempt: continue from this best checkpoint with learning rate `0.001`.
- Comet: `https://www.comet.com/liblaf/apple/4a9d5287cca0411cb993586b89cb54cd`.
- Restart step `0` reproduced the checkpoint as max face error
  `0.29785374454499103 cm`; the first restored forward solve reached
  `max_steps_reached`.
- Step `10` improved the best max face error to `0.29749668917954253 cm`.
- Step `13` improved the best max face error to `0.2972460977923983 cm`.
- Step `19` improved the best max face error to `0.297003376706202 cm`.
- Step `22` improved the best max face error to `0.29667913007873686 cm`.
- Step `26` improved the best max face error to `0.29621725845888425 cm`.
- Step `32` improved the best max face error to `0.29546154311654216 cm`.
- Step `41` improved the best max face error to `0.2948846265062962 cm`.
- Step `47` improved the best max face error to `0.29482957251336617 cm`.
- Step `48` improved the best max face error to `0.2944264277544116 cm`.
- Step `49` improved the best max face error to `0.29396597580728384 cm`.
- Step `50` improved the best max face error to `0.2937005436723448 cm`.
- Step `56` improved the best max face error to `0.29328888937191855 cm`.
- Step `59` improved the best max face error to `0.2931713026203408 cm`.
- Step `66` improved the best max face error to `0.2926574698232431 cm`.
- Step `70` improved the best max face error to `0.2924755055157813 cm`.
- Step `73` improved the best max face error to `0.2920230195402716 cm`.
- Step `100` improved the best max face error to `0.29191637235751283 cm`.
  This arrived after a long 26-step stall, so the branch was stopped manually
  too early but still produced a useful best checkpoint.

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.0015, Continuation 2

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 lr 0.0015 beta1 0 cont2' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,lr00015,beta10,restart,cont2,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr00015-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --inverse-lr 0.0015 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2.vtu --output-series data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2.png --checkpoint data/20-inverse-face-smooth-w1-max1-lr00015-beta10-cont2-checkpoint.npz
```

Early finding:

- Started from the previous low-LR checkpoint at step `100`, max face error
  `0.29191637235751283 cm`.
- Comet: `https://www.comet.com/liblaf/apple/761e422828b24c15be31afef83444e3b`.
- Restart step `0` reproduced the checkpoint as max face error
  `0.2919127096991687 cm`; the first restored forward solve reached
  `max_steps_reached`.
- Step `7` improved the max face error to `0.291895878091469 cm`.
- Step `12` improved the max face error to `0.29184114794964144 cm`.
- Step `31` improved the max face error to `0.2917224559121959 cm` after an
  18-step dry patch.
- Step `40` improved the max face error to `0.2914485524017837 cm`.
- Step `55` improved the max face error to `0.2912445453921606 cm`.
- Step `63` broke below `0.29 cm` with max face error
  `0.28997020361436404 cm`.
- Step `91` improved the max face error to `0.2892645939802489 cm` after a
  23-step dry patch.
- Step `105` improved the max face error to `0.2889045197411773 cm`.
- Step `132` improved the max face error to `0.2883430953128106 cm`.
- Step `141` made a larger drop to `0.2874829621023278 cm`.
- Current live finding: the branch is slow and noisy, but repeated late
  improvements show that learning rate `0.0015` remains productive. A much
  larger learning rate `0.03` was too hot from the nearby checkpoint; a
  moderate restart such as `0.003` is the next candidate only after this
  branch stalls.

### Smooth Weight 1.0, Max Loss 1.0, Learning Rate 0.0015

Command:

```bash
env -u DEBUG CHERRIES_NAME='inverse face 515k smooth weight 1 max loss 1 lr 0.0015 beta1 0' CHERRIES_TAGS='inverse-face,inverse,515k,smooth,w1,maxloss1,lr00015,beta10,restart,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-smooth-w1-max1-lr0003-beta10-checkpoint.npz --activation-smooth-weight 1.0 --target-max-loss-weight 1.0 --inverse-lr 0.0015 --adam-beta1 0.0 --adam-beta2 0.9 --inverse-max-steps 3000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8 --output data/20-inverse-face-smooth-w1-max1-lr00015-beta10.vtu --output-series data/20-inverse-face-smooth-w1-max1-lr00015-beta10.vtu.series --output-summary data/20-inverse-face-smooth-w1-max1-lr00015-beta10-summary.json --output-snapshot data/20-inverse-face-smooth-w1-max1-lr00015-beta10.png --checkpoint data/20-inverse-face-smooth-w1-max1-lr00015-beta10-checkpoint.npz
```

Early finding:

- Started from the max-loss `1.0`, learning-rate `0.003` best checkpoint at
  step `214`, max face error `0.29779184496091793 cm`.
