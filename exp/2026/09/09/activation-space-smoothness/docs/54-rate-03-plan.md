# Learning rate 0.3 follow-up

The user requested a test of learning rate 0.3 after the quarter-rate smoothed arm still developed severe local inversions. This targeted follow-up changes only the Adam rate relative to the original learned-axis smoothed arm. It does not replace either earlier experiment.

## Fixed protocol

- Use the unchanged `src/20-run-case.py`, numerical sources, frozen fixture, materials, target, forward solver, and adjoint solver.
- Start the learned-axis smoothed arm from the original seed 20260909, strength 0.001, zero displacement seed, and fresh Adam state. Retain epsilon 0.01, betas (0.9, 0.999), and lambda_C = 0.0004450069704277614 without retuning.
- Set the scalar Adam learning rate to **0.3**. No magnitude cap, inversion rejection, projection, or outer line search is added. Record inverted tetrahedra and minimum det(F) as diagnostics under the unchanged failure policy.
- First evaluate 128 updates in `data/55-axis-on-lr03-128/`. If measured runtime supports continuation, extend to 256, 384, and at most 512 updates in separate directories, restoring the exact saved controls, displacement solve seed, gradients, Adam moments, and counter. Do not reset Adam or change the rate.
- Run one GPU fitter at a time. The smoothed arm is prioritized because its quarter-rate trajectory prompted this test. A rate-0.3 off/on comparison is not planned within this targeted budget.
- Reallocate part of the original reporting reserve for this requested test. Stop fitting by **12:40 Asia/Shanghai on September 9, 2026 (04:40 UTC)**, leaving 41 minutes before the original 13:21 deadline. Do not start an extension unless recent measured runtime leaves time to complete it before the cutoff. A time stop is administrative and is reported separately from numerical failure; preserve the last exact evaluated checkpoint.

## Evaluation

Compare the rate-0.3 smoothed arm with the original-rate and quarter-rate smoothed arms at actual saved states matched within 0.05 mm in both fit RMS and motion RMS. Exclude initialization, select the lowest mean fit among eligible pairs with the established mismatch tie-break, and do not interpolate or relax tolerance if no match exists.

Report trajectories, endpoints, best-fit states, first inversion with its attained fit and motion, inversion counts, minimum det(F), physical displacement per update, activation change per update, and the frozen local surface score. Use exact saved geometry for matched sections and endpoint views. Smaller motion alone cannot establish improved geometry or optimization stability. Completion is not a stationarity or physical-validity certificate.

Freeze settings in `data/54-rate-03-settings/` before fitting. Retain all previous outputs and source receipts. The execution record and report will state the actual completed budget and any administrative interruption.
