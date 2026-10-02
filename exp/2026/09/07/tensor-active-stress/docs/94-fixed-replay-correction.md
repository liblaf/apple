# Exact replay of the historical learning rate

The six fresh forward/adjoint samples used the original frozen source-90 code,
source inventory, input manifest, and source-91 protocol. All six completed and
passed the saved-state replay checks. No sample was repeated or substituted.

The separate CPU `90-fixed-replay` attempt failed before writing its result
because its diagnostic re-bisected the historical two-times calibration. That
produced learning rate `0.0033136507734980113`, whereas source 73 had already
recorded `0.0033136508893221615`. Their difference, about `1.158e-10`, exceeded an
extra equality check in that diagnostic. Recalibrating the historical rate was
an implementation mistake: source 91 requires replaying the recorded rate.

Source 94 therefore reads and applies the exact source-73 rate to the exact
serialized gradient and cloned step-64 Adam state. It checks the production
Adam implementation against the closed-form update and repeats each installed
Adam update twice, using the same source-91 `1e-10` coordinate tolerance. It also
compares the physical step magnitude with the historical recorded magnitude.
The original source-90 file, failed output, terminal log, sampling manifest,
and pre-run archive remain unchanged.

The aggregation manifest points its fixed-replay dependency at the corrected
source-94 result. Its six sample paths are unchanged. The fresh one-times
calibrations at steps 64 and 256, noise gates, probe rules, optimizer epsilon,
continuation budgets, and all physics and solver tolerances are unchanged.
Source 94 performs no forward or adjoint solve.

The initial CPU diagnostic and the first sample at each anchor were launched
concurrently, whereas source 91 lists the historical diagnosis before the six
samples. This is an execution-order deviation. They were separate processes
reading immutable saved inputs, with no optimizer updates or state handoff.
The corrected historical replay completed after the six samples and before
aggregation and either optimization probe. No samples were added or replaced.
