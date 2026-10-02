# Continue the contact transition with a larger CG budget

Run 52 completed the full MouthOpen endpoint and 28 of 121 transition frames, then reached its declared 3,600-second budget. It preserved 82 accepted checkpoints, all input/source hashes passed, and its last accepted Smile blend is 0.11979701719998453. The interrupted trial is not an accepted state.

Run 75 resumes from that accepted checkpoint. It retains the repaired reference, exact saved activation tensors, prescribed jaw, no-skin materials, bone/eye contact scope, barrier coefficients, CCD settings, PNCG handoff, Newton search, and acceptance gates. It increases only the maximum iterations per CG solve from 1,000 to 3,000. Both values use the same relative linear tolerance 1e-3 and absolute free-force threshold 0.01 N. Each invocation remains bounded by one hour and can resume only from verified provenance.

## Second execution segment

The first run-75 segment stopped at its declared budget after 3,600.624759 s. It recorded 105 accepted checkpoints and 85 of 121 frames; its last accepted transition blend was `beta = 0.7938926261462365`. The independent [54 audit receipt](../data/54-fixed-contact-audit-002/summary.json) reported `verified_incomplete` and passed its fresh-force checks.

The next segment is running from that verified terminal state with the exact same run-75 source and `linear_max_steps = 3000`:

```bash
.venv/bin/python \
  src/75-resume-fixed-activation-contact.py \
  --output 75-fixed-activation-contact-002 \
  --resume exp/2026/09/30/mouthopen-smile-collisions/data/75-fixed-activation-contact
```

No numerical setting changed between the two segments. Completion remains pending.

## Evidence for the search-budget change

The saved transition traces showed 51 CG-budget failures over 19 accepted transition steps, consuming 146 seconds of retry work. The operator-only probe in `data/71-transition-pcg-probe` held a checkpoint, operator, right-hand side and preconditioner fixed. Cap 1,000 exhausted; cap 3,000 reached the existing tolerance in 2,326 unshifted iterations and 2,514 iterations at the tested positive shift. This is a local conditioning test, not a replay of the unsaved original Newton iterate.

The single-increment pilot in `data/72-linear-cap-pilot` used the same saved parent and target as frames 19 to 20. Both versions passed the existing force, boundary, contact and inversion gates. The 3,000-cap pilot used 10 Newton steps and 6,554 Hessian products with no CG retries; the original used 24 Newton steps, 37,910 Hessian products, and 21 cap retries. Measured durations were about 55 and 163 seconds, respectively, under shared GPU load. The PNCG trajectories also differed (140 versus 200 steps), so this is evidence supporting a larger cap, not a clean attribution of every timing difference to that cap.

The accepted displacements differed by 0.0396 mm vector RMS and 0.697 mm maximum. They contained 423 and 422 inverted cells, respectively. This remains an exploratory forward continuation and does not establish mechanical validity. The change permits more work per linear solve; it does not relax convergence or collision checks.

## Command

From this experiment directory, after the independent audit of run 52:

```bash
CHERRIES_NAME="Continue saved activation contact with 3000 CG steps" CHERRIES_TAGS="mouthopen,smile,contact,fixed-activation,repaired-reference,continuation" .venv/bin/python src/75-resume-fixed-activation-contact.py
```

The source imports run 52's contact-search operator and freezes both implementations. Its resume receipt names the prior checkpoint and old/new CG caps. The independent audit verifies inherited frames and all new frames against the same physical model and prescribed tensors. Render the completed movie only after all 121 states pass.
