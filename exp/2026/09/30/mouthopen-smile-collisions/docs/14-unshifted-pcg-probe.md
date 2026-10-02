# Unshifted rest IPC PCG probe

The rest, zero-activation Newton system used the same standard IPC,
fixed/fixed-only pair exclusion, and CLAMP-projected contact Hessian as the
corrected contact validation. It used the exact `gpu_contact` HVP, the normal
diagonal preconditioner, `rtol=1e-3`, and a declared maximum of 20,000 steps.
The probe never updated the displacement or ran a Newton solve.

The first receipt used a signed diagonal and does not match the solver; it is
superseded. The corrected run used the exact main preconditioner,
`vector / diagonal.abs()`, with zero shift.

The gradient norm was `6.40042255e-7`, and the physical diagonal minimum was
`-2.34208637e-4`. The final, 150-second bounded run reached all 20,000 PCG
iterations in 130.28 seconds and stopped with `CG iteration budget exhausted`;
it did not attain `rtol=1e-3`. The signed-diagonal immediate rejection was an
artifact of the earlier probe. Under the solver-matching absolute-diagonal
preconditioner, the unshifted system instead exhausts the declared linear
budget.

The final Cherries run used `Git(commit=False)` and completed at
[8372a3934c8144bc80cb3fa16257dd6a](https://www.comet.com/liblaf/apple/8372a3934c8144bc80cb3fa16257dd6a).
The receipt is
[`data/14-unshifted-pcg-probe-003/summary.json`](../data/14-unshifted-pcg-probe-003/summary.json).
