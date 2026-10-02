# Matched forward-solver performance protocol

This experiment tests numerical acceleration of the existing eye-inclusive
expression model. Constitutive energies, materials, constraints, targets, IPC
configuration, and the accepted-state force threshold remain unchanged.

Two immutable MouthOpen initial checkpoints define the fixtures: the
collision-off pose initializer from expression-fitting-007 with a one-degree
jaw proposal, and the contact-on pose initializer from expression-fitting-006
with a 0.03125-degree proposal. Both retain zero activation. These are different
mechanical problems; no speedup is inferred by comparing one fixture to the
other.

Each method reconstructs its own runtime, uses the same seed and parameter
proposal, warms the energy/gradient/curvature kernels, and times synchronized
forward and implicit-adjoint operations. Original PNCG is the control. The
alternative arms reuse accepted-state gradients, use exact directional
curvature with PNCG, or use safeguarded Newton-CG with scalar/vertex-block
preconditioning. Newton diagonal shifts affect only the search direction;
the terminal residual and implicit adjoint use the exact physical derivatives.
Negative curvature, nonlinear/linear budget exhaustion, invalid contact, or
failed Armijo searches are recorded explicitly.

The physical force threshold is 1.5192003475221146e-10 code units. The adjoint
relative tolerance is 1e-7. Candidate endpoints must satisfy the same force and
contact checks and have no inverted tetrahedra in these selected fixtures.
The comparison also requires maximum displacement disagreement below 1e-6 m
and activation/jaw gradient relative disagreement below 1e-3 against the
original control. Passing these local checks does not certify all expressions
or global mechanical stability.

The user authorized COMP* computation. V100 host was selected after a live
inventory found two idle RTX 3090 GPUs and sufficient disk space. The runtime
is a task-local copy of the local Python environment. Input files and original
JSON manifests retain their bytes; the loader relocation helper maps their
absolute paths explicitly. The transfer manifest binds 192 files, totaling
754,077,515 bytes, by SHA-256. No existing experiment or server job is modified.

Source snapshots, input/checkpoint hashes, GPU occupancy, operation counts,
individual success/failure receipts, output tensors, and comparisons are saved
under each run directory. CPU validation uses Cherries and Comet. Remote
benchmarks use DEBUG=1 for local Cherries evidence because the remote task
environment has no copied account credentials. Compilation is excluded from
solver timings; setup of a numerical preconditioner is included.
