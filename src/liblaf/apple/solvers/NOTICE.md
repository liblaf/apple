# Solver provenance

This directory contains a project-owned copy of the optimizer and linear-solver
modules from `liblaf-peach` version 0.10.1 (Git tag `v0.10.1`, commit
`779c197a1bca75508439ce79013e75ff5863b5a8`). The original source repository is
<https://github.com/liblaf/peach>.

The copied implementation is unchanged except that imports and documentation
references use the `liblaf.apple.solvers` namespace, imports and lint comments
follow Apple's formatting, and package-level lazy
exports omit Peach version metadata, its SciPy optimizer, and its testing
helpers because those modules are not included here.

`liblaf-peach` declares the MIT License. The license text is reproduced in
[`LICENSE`](LICENSE).
