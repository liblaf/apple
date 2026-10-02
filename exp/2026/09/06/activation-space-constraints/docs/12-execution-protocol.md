# Execution protocol frozen before the main matrix

2026-09-06. The existing `10-experiment-plan.md` is preserved as the design draft.

The first completed scope is the forward frequency check, the 33-fit nonlinear
tetrahedral block screen, a regularization-strength frontier, and selected
robustness checks. This is a controlled computational study, not validation of
physiological facial activation. Facial fibers are not yet registered to the
prepared mesh, so a face comparison is conditional on resolving that input.

The 33-fit screen follows the draft's eleven control rows and three targets:
clean, 2% normal-displacement noise, and an added smooth upward displacement
with RMS 0.5 times the clean displacement RMS D. The fixture has 24 x 24 lateral
cells and ten vertical layers, bottom fixed, sides and top free. The known
fiber axis is x. Passive materials and the production StableNeoHookeanActive
energy are unchanged. Penalties are 0.01 when enabled. The natural-shortening
cap is 35%; it is an experimental sensitivity parameter, not a calibrated
physiological bound.

One deliberate change to the draft: noisy targets use cosine-product Fourier
wave numbers 2 and 3 instead of 4 through 6. The 24-cell lateral grid has at
least eight cells per wavelength for the revised noise. The original modes
would have only four to six cells per wavelength. The forward mechanism probe
retains wave numbers 1 and 4 on its 48-cell grid. Physical high-pass lengths
remain 0.03, 0.06, and 0.12 of the unit footprint.

The initial inverse budget is 160 accepted projected L-BFGS iterations. Every
line-search trial starts from the last accepted equilibrium; cases start from
rest. Forward and adjoint failures terminate the affected case and are saved.
Rejected objective trials are retained in a receipt. Endpoints are audited
against a fresh rest-start solve. Projected stationarity is measured explicitly;
budget exhaustion and line-search stalls are reported without claiming an
optimum. Representative comparisons will be extended if stationarity remains
material to the interpretation.

The 12-cell pilot used 5% noise and 25 iterations to test implementation and
cost. It is excluded from the main numerical table. Its three audited
end-to-end implicit gradients agreed with central differences to within
3.4e-5 relative error. Before the main run, the audit is expanded to all five
coordinate families. An autograd state-alias defect in the initial pilot was
fixed locally in the fixture wrapper by giving each solve a fresh detached
state buffer; no library source was changed.

Runs use Cherries with local snapshots, metrics, logs, and Comet recording.
The explicit `ProfileCometNoCommit` disables automatic Git staging and commits.
Main-run data include source hashes, configuration, targets, graph, checkpoints,
endpoint tetra meshes, convergence traces, and numerical diagnostics.
