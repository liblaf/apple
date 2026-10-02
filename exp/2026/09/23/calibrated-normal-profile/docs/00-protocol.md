# Calibrated L2 plus target-normal comparison

Repeat the h=0.20 neutral-start activation experiment with four models, activation
smoothness alpha=0/1, and positional L2 alone versus L2 plus target-normal loss:
16 independent runs. Retain the original 100 by 10 mesh, muscle band, materials,
boundary conditions, physical det(F) energy, and 1,200-update Adam schedule
(learning rate 0.03 times 0.99^step). All runs start at q=u=m=v=0, B=I.

The user confirmed 0.02 means position RMS in model units (strip width is 1).
With L2=mean squared vector position error and N=reference-length-weighted mean
of (1-cos(theta)) over corresponding top-edge normals, fix

    c = 0.02^2 / (1-cos(5 degrees)) = 0.105116495...
    J = L2 + c*N + alpha*h^2*R.

R is the original mean squared Frobenius jump between neighboring muscle B
tensors. Both terms contribute 0.0004 at RMS=0.02 and uniform normal error=5
degrees. For varying angles, the calibration is exact for the cosine/chord
aggregate; angular RMS is approximately equivalent near 5 degrees. This is a
user-specified scale, not a fitted or optimized hyperparameter. The retained
runner/history field `beta` means the direct coefficient c in this experiment.

Retain failed proposals and last accepted states, with no solver replacement or
restart. Use the latest saved step common to all four variants within a model
for quantitative comparisons, and show original-style endpoints separately.
Neither the finite budget nor a decayed learning rate establishes convergence.

Validate calibration, direct and implicit derivatives for all activation models,
neutral initialization, and equivalence to the original L2 objectives. Independently
recompute saved position/normal/roughness/det(F)/force metrics and constraints;
inspect displacement Hessians at common states and endpoints, marking failures
and unstable equilibria on figures. Reproduce the eight historical L2 endpoints.

Export a primary 16:9 PNG and vector PDF/SVG, retaining full-domain equal physical
axes, clear target/condition legends, and the requested calibration on the slide.
Keep all sources, logs and experiment output within this group; record source
hashes and Comet provenance through Cherries without automatic Git commits.
