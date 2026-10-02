# Sparse contact backend capability check

## Purpose

This check attempted to compare `gpu_contact` and `gpu_sparse` on the same
rest-state, zero-activation, active-strain MouthOpen fixture. Both backends
were configured to use the same IPC contact Hessian with
`PSDProjectionMethod.CLAMP`; no solver, material, or contact parameter was
changed.

## Command

```bash
CHERRIES_NAME='Sparse contact active-strain capability check' \
CHERRIES_TAGS='mouthopen,smile,contact,ipc,gpu,hessian,validation' \
uv run python src/11-check-sparse-contact.py
```

The Cherries Profile used `Git(commit=False)`. The completed capability run is
[c1ca9d63913c4a5d9daceabd2b25ad52](https://www.comet.com/liblaf/apple/c1ca9d63913c4a5d9daceabd2b25ad52).

## Result

`gpu_contact` initialized on CUDA with the complete 63,282-vertex,
126,648-triangle FEM boundary. Its standard diagonal evaluation took
0.05766 seconds and peak allocated CUDA memory was 850,975,232 bytes.

`gpu_sparse` could not build the bulk assembled matrix for this required
active-strain material set. Its explicit capability error was:

```text
unsupported potential(s): aponeurosis:StableNeoHookean,
fat:StableNeoHookean, muscle:StableNeoHookeanActive
```

Therefore no three-direction HVP comparison, diagonal comparison, timing
comparison, or sparse-backend speed recommendation is valid for the requested
physical state. `gpu_contact` remains the available exact backend for the
fixed-S contact solves until the core assembled-FEM registry supports these
three potentials.

The machine-readable receipt is
[`data/11-sparse-contact-check/summary.json`](../data/11-sparse-contact-check/summary.json).
