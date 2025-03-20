# Changelog

## [0.0.3](https://github.com/liblaf/apple/compare/v0.0.2...v0.0.3) (2025-03-20)


### ✨ Features

* add gradient descent optimizer and update inverse problem logic ([ae27101](https://github.com/liblaf/apple/commit/ae27101f950d915e315a9659cec79a16b11ebeb3))
* add triangle element support and Koiter shell problem ([e23525c](https://github.com/liblaf/apple/commit/e23525ce50312a3271baba8cccf706f36ca517ee))
* **phace:** add animation script and simulation modules ([468c982](https://github.com/liblaf/apple/commit/468c982da85847ffe63d7deaee6d17ba22edf4f9))
* **physics:** add gravity support and refactor physics problem architecture ([4f50caa](https://github.com/liblaf/apple/commit/4f50caa40dbe7ba810d311b5540b57f10293f721))


### ⬆️ Dependencies

* **deps:** update dependency jaxtyping to &gt;=0.3,&lt;0.4 ([#14](https://github.com/liblaf/apple/issues/14)) ([467d4dd](https://github.com/liblaf/apple/commit/467d4dd7a2996f3d4c6fa377a40a72f2d611a0aa))
* **deps:** update dependency liblaf-cherries to &gt;=0.0.6,&lt;0.0.7 ([#13](https://github.com/liblaf/apple/issues/13)) ([d773f0f](https://github.com/liblaf/apple/commit/d773f0f1aa4052bcba86f2f70d95c839b058efc9))
* **deps:** update dependency liblaf-cherries to &gt;=0.0.8,&lt;0.0.9 ([#15](https://github.com/liblaf/apple/issues/15)) ([67099cd](https://github.com/liblaf/apple/commit/67099cda79614cc8cf2c1036ceefbffeacec10f2))


### ♻ Code Refactoring

* restructure physics problem implementation and optimization ([e98f4c5](https://github.com/liblaf/apple/commit/e98f4c5733f6911e24815e8b1f544c5214623a65))
* simplify physics problem builder pattern ([1aff338](https://github.com/liblaf/apple/commit/1aff33837cac11433ed80cd4b6f3488edf88119f))

## [0.0.2](https://github.com/liblaf/apple/compare/v0.0.1..v0.0.2) - 2025-02-23

### ✨ Features

- refactor physics simulation and add inverse problem solver - ([e7ea071](https://github.com/liblaf/apple/commit/e7ea07113d2fa8ee4adee2ab5c1637cd81ca9aee))

### ⬆️ Dependencies

- **deps:** update dependency beartype to >=0.20,<0.21 (#8) - ([df34ec2](https://github.com/liblaf/apple/commit/df34ec25a774e7c6858cd579747ec972da40926b))
- **deps:** update dependency liblaf-grapes to >=0.1.0,<0.1.1 (#5) - ([60fb1d0](https://github.com/liblaf/apple/commit/60fb1d09e62d7f10e520f40c7c4a22956a041ce9))

### ♻ Code Refactoring

- remove unnecessary JAX configuration log - ([2587b4e](https://github.com/liblaf/apple/commit/2587b4ee542fe3e47eafcefe58c9f4ca7bafd5a8))
- reorganize and enhance abstract classes and utilities - ([5f16903](https://github.com/liblaf/apple/commit/5f169036cd636ab9480e4f29004075e6af8f6939))

### ❤️ New Contributors

- @liblaf made their first contribution
- @renovate[bot] made their first contribution in [#8](https://github.com/liblaf/apple/pull/8)

## [0.0.1](https://github.com/liblaf/apple/compare/v0.0.0..v0.0.1) - 2025-02-16

### ✨ Features

- **opt:** add PNCG minimization algorithm and refactor strain test - ([91208b7](https://github.com/liblaf/apple/commit/91208b7c382933586e8900419195c939b397f5ad))
- add tetrahedral element FEM and rotation-variant SVD - ([ef11295](https://github.com/liblaf/apple/commit/ef11295ba1f85ef18f89063abcd7694de0d0af73))

### ⬆️ Dependencies

- **deps:** update dependency liblaf-grapes to >=0.0.4,<0.0.5 (#4) - ([7db122f](https://github.com/liblaf/apple/commit/7db122f9f585ac6ba3ca6900e41d1ba059e37abd))

### ♻ Code Refactoring

- reorganize apple module structure and optimize physics simulation - ([cd42fa4](https://github.com/liblaf/apple/commit/cd42fa4423b9ac8b8faeddebaad842d0a1eb8ac9))

### 🔧 Continuous Integration

- correct syntax for `if` condition in test workflow - ([b88f52e](https://github.com/liblaf/apple/commit/b88f52e6c02433c5df5148ca578d831493c18b4f))

### ❤️ New Contributors

- @github-actions[bot] made their first contribution in [#3](https://github.com/liblaf/apple/pull/3)
- @renovate[bot] made their first contribution in [#4](https://github.com/liblaf/apple/pull/4)

## [0.0.0] - 2025-01-19

### ✨ Features

- add optimization and problem-solving modules with JAX integration - ([49f3ac4](https://github.com/liblaf/apple/commit/49f3ac4ef1f427f469170c5e9eaaf83ebed4a6c4))

### 👷 Build System

- initialize Python project with essential configurations - ([10719f8](https://github.com/liblaf/apple/commit/10719f8793ac0aa58caaca57dc44e15165640c9c))

### ❤️ New Contributors

- @release-please[bot] made their first contribution in [#1](https://github.com/liblaf/apple/pull/1)
- @liblaf made their first contribution
- @liblaf-bot[bot] made their first contribution
