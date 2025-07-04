# Changelog

## [0.2.1](https://github.com/liblaf/apple/compare/v0.2.0..v0.2.1) - 2025-07-03

### ✨ Features

- **examples:** add collision simulations and improve physics - ([10c6916](https://github.com/liblaf/apple/commit/10c6916acf633d07e5bb01749b381b28d293dfc3))

### ⬆️ Dependencies

- **deps:** update dependency python to 3.13.\* (#30) - ([6aebf70](https://github.com/liblaf/apple/commit/6aebf70ecbbdb92290aa21f317c84e02335659e7))
- **deps:** update dependency python to 3.13.\* (#18) - ([4643411](https://github.com/liblaf/apple/commit/46434111bdfe0626f7a7ea89ac7a8cc2e2e18516))

### ❤️ New Contributors

- [@renovate[bot]](https://github.com/apps/renovate) made their first contribution in [#30](https://github.com/liblaf/apple/pull/30)
- [@liblaf](https://github.com/liblaf) made their first contribution

## [0.2.0](https://github.com/liblaf/apple/compare/v0.1.0..v0.2.0) - 2025-07-02

### 💥 BREAKING CHANGES

- **collision:** add Hessian calculations and improve PNCG - ([bb37387](https://github.com/liblaf/apple/commit/bb37387384334dc6a221344dc4b6424f3b34a30a))

### ✨ Features

- **sim:** add free-fall example and center of mass utils - ([7c1f1d4](https://github.com/liblaf/apple/commit/7c1f1d4a5cdf8a8417aac9525356bdaf9fcc6f8f))

## [0.1.0](https://github.com/liblaf/apple/compare/v0.0.3..v0.1.0) - 2025-07-01

### 💥 BREAKING CHANGES

- **collision:** restructure vertex-face collision system - ([971cbe2](https://github.com/liblaf/apple/commit/971cbe2c8f8f2d1f51421b6eb3aa8fb89eec4485))
- **core:** migrate to Equinox PyTree management - ([e774eeb](https://github.com/liblaf/apple/commit/e774eeb4ea3a027bde3fddbf9b84dab701a17237))
- **dictutils/tree:** add dictionary utilities and improve container fields - ([85ef3f1](https://github.com/liblaf/apple/commit/85ef3f1d0cb39a9f8da1ee1a15b4e0fa397bb330))
- **optim:** rewrite PNCG with improved math utilities - ([6dcada1](https://github.com/liblaf/apple/commit/6dcada1fb8b5da19ea7d6c20822998d58a1bebef))
- **optim:** restructure optimization module - ([11900ed](https://github.com/liblaf/apple/commit/11900edb9ed023841933a0590381448b96034688))
- **sim:** restructure simulation components and state - ([1a38a3c](https://github.com/liblaf/apple/commit/1a38a3c3502d07ce0a361991878352d895950094))
- **sim:** remove physics module and improve implementation checks - ([ad644d5](https://github.com/liblaf/apple/commit/ad644d560a7367f8924b97cd8b7e7ee72245010c))
- **sim:** implement new collision detection system - ([9d1ffc4](https://github.com/liblaf/apple/commit/9d1ffc45bcc7f3f6e937fdbc82e206b5b4fd19a2))
- **sim:** refactor and enhance simulation framework - ([620906a](https://github.com/liblaf/apple/commit/620906ad20361542fc33c7a9a031ba8cc96ea2b7))
- **sim:** overhaul actor system and geometry handling - ([69b7e5b](https://github.com/liblaf/apple/commit/69b7e5b7b40fce9d181636fc79f3d6df890540d2))
- **sim:** restructure core modules and utilities - ([79da255](https://github.com/liblaf/apple/commit/79da255851e0ce09351c22d125a6d726c3b7b60f))
- **sim:** restructure core components and add collision detection - ([b967f7f](https://github.com/liblaf/apple/commit/b967f7fb6705c888db9de989e13330e853f279d9))
- **sim:** migrate inertia to new framework - ([f4d2e8f](https://github.com/liblaf/apple/commit/f4d2e8f552ece5466f35c1961a70326573c42972))
- **sim:** restructure core simulation modules - ([a0e2108](https://github.com/liblaf/apple/commit/a0e21080325a8cff8a1fb63fc2d724eaf9f487a6))
- **sim:** restructure core components into ABCs - ([bdbe382](https://github.com/liblaf/apple/commit/bdbe3824d4a0e57ba96e4bc82d477eb934492259))
- **sim:** reorganize geometry and field systems - ([53fbd66](https://github.com/liblaf/apple/commit/53fbd66b6a55005251b70863dbbaf606fd78a620))
- **sim:** restructure element and geometry classes - ([3539cea](https://github.com/liblaf/apple/commit/3539cea691d29b3e97e2b477b21f40980338c589))
- **sim:** restructure finite element simulation components - ([885fd8c](https://github.com/liblaf/apple/commit/885fd8c730db5b96ea876d58922202e8e9e2b4c2))
- **simulation:** add rigid-soft collision handling and simulation framework - ([aa9ae7a](https://github.com/liblaf/apple/commit/aa9ae7a5385d344b80d505f8c9388c4f950fc594))
- **struct:** replace flax.struct with custom Node system - ([80d15a4](https://github.com/liblaf/apple/commit/80d15a40cef0e9787fffe4a36c85dca629caf5a2))
- **tree:** reorganize pytree utilities - ([314f218](https://github.com/liblaf/apple/commit/314f218ef59f92df88715183d7a63bb09718e878))
- next (#24) - ([7620f38](https://github.com/liblaf/apple/commit/7620f386ff889b1a76bc3b919f603bc325213d81))

### ✨ Features

- **apple:** add AbstractMinimizeProblem and fix_winding utility - ([c661b07](https://github.com/liblaf/apple/commit/c661b07ad722c844e4fc1d52f53aebeec3d4a237))
- **dictutils,tree:** add FrozenDict and ArrayMixin - ([dee4731](https://github.com/liblaf/apple/commit/dee473197b2b4638fb2da7859264ca76331958c3))
- **dynamics:** enhance collision detection with animation and visualization - ([edd25ce](https://github.com/liblaf/apple/commit/edd25ce18d231be5d45c10c9a6c06d6675a67bee))
- **dynamics:** add collision example and update simulation parameters - ([3906dcb](https://github.com/liblaf/apple/commit/3906dcb89ed3bcf0ddf76bc13e6afd5a94a53a65))
- **dynamics:** add gravity simulation with bunny example - ([9134b72](https://github.com/liblaf/apple/commit/9134b72f0a2459b0e0ad1c0b637a4138edf33261))
- **dynamics:** add dynamic simulation capabilities to bunny example - ([24b60bd](https://github.com/liblaf/apple/commit/24b60bd65306e98528694cf17495c877c9a4c35e))
- **jaw-motion:** add muscle activation simulation - ([8a4a95f](https://github.com/liblaf/apple/commit/8a4a95f9d21536405e3d4171e10f30380bb13365))
- **jaw-motion:** add jaw motion simulation experiment - ([213362d](https://github.com/liblaf/apple/commit/213362d324b09e3e920a771453553a4ce0e0bf0b))
- **optim:** add JIT support and Hessian diagonal computation - ([4aea353](https://github.com/liblaf/apple/commit/4aea353aff08ff3663a12e480239e1c3083c6551))
- **physics:** enhance scene optimization and inertia calculations - ([5fa2302](https://github.com/liblaf/apple/commit/5fa2302511317b002e7eaf835706ab9501b4c416))
- **physics:** refactor field system and add dynamics support - ([b9e5e01](https://github.com/liblaf/apple/commit/b9e5e01ed807ff8acd3ac50adae0ea2c76264dd7))
- **physics:** restructure physics module with new domain and field system - ([07b2823](https://github.com/liblaf/apple/commit/07b2823befd1f28c7d1813821d3821c05dd43efe))
- **strain:** add naive implementations of Qs and h3_diag functions - ([a8bf975](https://github.com/liblaf/apple/commit/a8bf975cd90ddfb94148c996818e6c509323428a))
- **strain:** optimize and refactor tetrahedral strain calculations - ([fe2218f](https://github.com/liblaf/apple/commit/fe2218f41e74d767a561da3822dcb7d2c668d82f))
- enhance physics simulation with Geometry class and PhaceStatic energy - ([6968ad7](https://github.com/liblaf/apple/commit/6968ad78265c6f280496a4c658e201e2443c150a))
- add PNCG optimizer and enhance elastic energy models - ([59c1612](https://github.com/liblaf/apple/commit/59c16127759c01e888980465b31f75bf1bdd74db))
- add dynamic simulation capabilities and improve physics modeling - ([07b0d1c](https://github.com/liblaf/apple/commit/07b0d1c18aa5833d9d324d4992573433cf4eb4fe))
- add benchmark tests and material models for FEM analysis - ([888ffd8](https://github.com/liblaf/apple/commit/888ffd899714559008f5a7b6ee1b0fd47d8faa5b))

### ⬆️ Dependencies

- **deps:** update liblaf-melon dependency to v0.2.8 - ([9ecefe1](https://github.com/liblaf/apple/commit/9ecefe192f31b49b1e1e10acba5a3d78ac58d343))
- **deps:** update liblaf-grapes dependency to v0.1.28 - ([68033eb](https://github.com/liblaf/apple/commit/68033ebc72eadf9e265915959c049363d2a4614f))

### ♻ Code Refactoring

- **optim:** simplify timing callback handling and update type hints - ([31a4910](https://github.com/liblaf/apple/commit/31a4910189b5d6549a77218feb4ef3d0f1b36f98))
- **physics:** restructure physics problem abstraction and material models - ([7a4b236](https://github.com/liblaf/apple/commit/7a4b236d24c1e8cce2c738d0552baeb64fb11572))
- **sim:** improve physics examples and core components - ([156445f](https://github.com/liblaf/apple/commit/156445fe566af413f51cebe213f089c7d660d12f))
- **strain:** optimize deformation gradient calculations - ([6be27eb](https://github.com/liblaf/apple/commit/6be27eb6ba496231a43d72978504b88fb549f329))
- refactor dependencies and add Object class for physics simulation - ([b3dbbaf](https://github.com/liblaf/apple/commit/b3dbbaf80f8410765bcdfb3abe41e6698ae6e542))
- reorganize configuration files and update dependencies - ([b4e8d32](https://github.com/liblaf/apple/commit/b4e8d320b29592c945fa690edfe90e65ffac366a))

### 👷 Build System

- **deps:** update and relax dependency requirements - ([a3e2597](https://github.com/liblaf/apple/commit/a3e259745e3a0eb49cf8ba49d7ad2d48498b56b4))
- add Flax as a dependency - ([97d5688](https://github.com/liblaf/apple/commit/97d5688de5e919ef986c10499975b5cc02a61c36))
- update project configuration and dependencies - ([dbf4953](https://github.com/liblaf/apple/commit/dbf49536e976b10b2a129badd9fe091426725e6e))

## [0.0.3](https://github.com/liblaf/apple/compare/v0.0.2..v0.0.3) - 2025-03-23

### ✨ Features

- **phace:** add animation script and simulation modules - ([468c982](https://github.com/liblaf/apple/commit/468c982da85847ffe63d7deaee6d17ba22edf4f9))
- **physics:** add gravity support and refactor physics problem architecture - ([4f50caa](https://github.com/liblaf/apple/commit/4f50caa40dbe7ba810d311b5540b57f10293f721))
- add gradient descent optimizer and update inverse problem logic - ([ae27101](https://github.com/liblaf/apple/commit/ae27101f950d915e315a9659cec79a16b11ebeb3))
- add triangle element support and Koiter shell problem - ([e23525c](https://github.com/liblaf/apple/commit/e23525ce50312a3271baba8cccf706f36ca517ee))

### ⬆️ Dependencies

- **deps:** update dependency liblaf-cherries to >=0.0.8,<0.0.9 (#15) - ([67099cd](https://github.com/liblaf/apple/commit/67099cda79614cc8cf2c1036ceefbffeacec10f2))
- **deps:** update dependency liblaf-cherries to >=0.0.6,<0.0.7 (#13) - ([d773f0f](https://github.com/liblaf/apple/commit/d773f0f1aa4052bcba86f2f70d95c839b058efc9))
- **deps:** update dependency jaxtyping to >=0.3,<0.4 (#14) - ([467d4dd](https://github.com/liblaf/apple/commit/467d4dd7a2996f3d4c6fa377a40a72f2d611a0aa))

### ♻ Code Refactoring

- simplify physics problem builder pattern - ([1aff338](https://github.com/liblaf/apple/commit/1aff33837cac11433ed80cd4b6f3488edf88119f))
- restructure physics problem implementation and optimization - ([e98f4c5](https://github.com/liblaf/apple/commit/e98f4c5733f6911e24815e8b1f544c5214623a65))

## [0.0.2](https://github.com/liblaf/apple/compare/v0.0.1..v0.0.2) - 2025-02-24

### ✨ Features

- refactor physics simulation and add inverse problem solver - ([e7ea071](https://github.com/liblaf/apple/commit/e7ea07113d2fa8ee4adee2ab5c1637cd81ca9aee))

### ⬆️ Dependencies

- **deps:** update dependency beartype to >=0.20,<0.21 (#8) - ([df34ec2](https://github.com/liblaf/apple/commit/df34ec25a774e7c6858cd579747ec972da40926b))
- **deps:** update dependency liblaf-grapes to >=0.1.0,<0.1.1 (#5) - ([60fb1d0](https://github.com/liblaf/apple/commit/60fb1d09e62d7f10e520f40c7c4a22956a041ce9))

### ♻ Code Refactoring

- remove unnecessary JAX configuration log - ([2587b4e](https://github.com/liblaf/apple/commit/2587b4ee542fe3e47eafcefe58c9f4ca7bafd5a8))
- reorganize and enhance abstract classes and utilities - ([5f16903](https://github.com/liblaf/apple/commit/5f169036cd636ab9480e4f29004075e6af8f6939))

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

- [@github-actions[bot]](https://github.com/apps/github-actions) made their first contribution in [#3](https://github.com/liblaf/apple/pull/3)
- [@renovate[bot]](https://github.com/apps/renovate) made their first contribution in [#4](https://github.com/liblaf/apple/pull/4)

## [0.0.0] - 2025-01-19

### ✨ Features

- add optimization and problem-solving modules with JAX integration - ([49f3ac4](https://github.com/liblaf/apple/commit/49f3ac4ef1f427f469170c5e9eaaf83ebed4a6c4))

### 👷 Build System

- initialize Python project with essential configurations - ([10719f8](https://github.com/liblaf/apple/commit/10719f8793ac0aa58caaca57dc44e15165640c9c))

### ❤️ New Contributors

- [@release-please[bot]](https://github.com/apps/release-please) made their first contribution in [#1](https://github.com/liblaf/apple/pull/1)
- [@liblaf](https://github.com/liblaf) made their first contribution
- [@liblaf-bot[bot]](https://github.com/apps/liblaf-bot) made their first contribution
