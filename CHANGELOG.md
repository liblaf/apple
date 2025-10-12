# Changelog

## [0.4.0](https://github.com/liblaf/apple/compare/v0.3.0...v0.4.0) (2025-10-12)


### ⚠ BREAKING CHANGES

* Point data array renamed from 'point-id' to 'point-ids'
* **jax:** add active material models and enhance optimization infrastructure
* **sim:** Existing energy and quadrature imports have been refactored to use private modules. Update imports from `sim.energy.arap` to `sim.energy.elastic._arap` and similar changes for quadrature components.
* **struct:** This removes the PyTree and PyTreeMutable base classes, and the associated replace and tree_at methods. Existing code using these classes will need to migrate to the new @pytree decorator pattern.
* **sim:** 
* **sim:** 
* **state:** 
* **sim:** 
* **struct:** All classes using @struct.pytree or PyTreeMixin need to be updated to use Equinox's Module system. Field declarations now use field() instead of data/static.

### ✨ Features

* Add JAX-based simulation core with math utilities and type definitions ([36d26fa](https://github.com/liblaf/apple/commit/36d26fa006f8099d8bd3dfabe8c49430856d4c40))
* Add JAX-based simulation framework and mathematical documentation ([d63a206](https://github.com/liblaf/apple/commit/d63a206bb219b2134c138b0876b932c79ffc2b5c))
* add Phace energy model and rename point data array ([d03be25](https://github.com/liblaf/apple/commit/d03be256552548f77abae914ae0f11c0e526e519))
* **experiments:** add dynamic and static simulation examples ([262646e](https://github.com/liblaf/apple/commit/262646e280a8e5d6ae199d012b2b7faa4eb6cfd9))
* Implement phace active and passive energy models ([294f454](https://github.com/liblaf/apple/commit/294f454b359c31ae6e33d456989970e2acf0db6d))
* **inverse-simulation:** enhance inverse muscle simulation with Phace energy model ([8cf739b](https://github.com/liblaf/apple/commit/8cf739b3cb527c4ec7c48b1cd485926c817aaf08))
* **inverse:** Add inverse physics solver with Warp and lineax integration ([cc8ebbb](https://github.com/liblaf/apple/commit/cc8ebbbe9bc5d3777f40b2f14aeae6e94d5bf772))
* **jax/optim:** add modular minimization framework with SciPy backend ([72ae94b](https://github.com/liblaf/apple/commit/72ae94b76df2d0d08d826f2359ff17ea11744685))
* **jax/sim/energy:** Add Koiter thin shell energy model ([e8175e3](https://github.com/liblaf/apple/commit/e8175e3f3763907356368f9cc7b23c818daaa5ca))
* **jax:** add active material models and enhance optimization infrastructure ([518ee0b](https://github.com/liblaf/apple/commit/518ee0b6da54cb01803f220d6c7fabef8fcefd62))
* **jax:** add PNCG optimizer and autodiff utilities ([bb5a51e](https://github.com/liblaf/apple/commit/bb5a51ead25358ef2202ce0bdc6d236ab1ea744c))
* **phace:** add close-mouth and wind simulation examples ([eb5bfdb](https://github.com/liblaf/apple/commit/eb5bfdbe7d34ed487043e308c616d479c797a9cc))
* **phace:** add open-mouth jaw animation example ([5dfa5e3](https://github.com/liblaf/apple/commit/5dfa5e38ff1e23b295d1023e662a0a899cdc3774))
* **sim:** Add activation utility functions for JAX energy models ([ae26615](https://github.com/liblaf/apple/commit/ae26615d4a9fbd5d665b2aabcde8dfe5a7d96a20))
* **sim:** add dirichlet conditions, geometry handling, and model building ([0d8388e](https://github.com/liblaf/apple/commit/0d8388e6b9c427689168dc263dbd08307699bb3e))
* **sim:** add Hessian computation support for energy models ([cb1ea7f](https://github.com/liblaf/apple/commit/cb1ea7fedc5293b5474892b7c14a2fc1f4475d06))
* **sim:** add inter-collision example and improvements ([231d509](https://github.com/liblaf/apple/commit/231d5099f3f5dcc1710c8f1c29dd0a938a26aac7))
* **sim:** integrate Warp backend for simulation ([eaff9b9](https://github.com/liblaf/apple/commit/eaff9b94b92184ecd1847c0d8bcc3773d0bb6343))
* **sim:** introduce active ARAP energy model and Dirichlet builder ([af97357](https://github.com/liblaf/apple/commit/af973573ef9f03d86c71c420c42567c08bc13d6c))
* **simulation:** add ball-head collision example with visualization support ([9550bf2](https://github.com/liblaf/apple/commit/9550bf2c07daf9e0c1509ea515581a3a043ebc60))
* **struct:** add mutable tree variants ([5ed48f6](https://github.com/liblaf/apple/commit/5ed48f6447be1092c76c8454297bbbab7ca1bab2))
* **warp/sim/energy:** add active ARAP and passive PHACE energy models ([f56d137](https://github.com/liblaf/apple/commit/f56d137e589aec81e9383534e3047c5e7e9a0a8a))
* **warp:** add matrix decomposition functions and update warp dependency ([6a5b2a9](https://github.com/liblaf/apple/commit/6a5b2a953509a269490be8fc05a12d3ce0fbd6ec))
* **warp:** add simulation energy infrastructure and ARAP model ([24abf6a](https://github.com/liblaf/apple/commit/24abf6a060ce975cad8c4d532d81cc1a83b961b9))


### 🐛 Bug Fixes

* **data:** update md5 hashes and sizes for input and target VTU files ([e1b9558](https://github.com/liblaf/apple/commit/e1b9558293268340ffcc51a7fee8dea71d89fde7))
* **deps:** update and simplify dependencies ([dcbc065](https://github.com/liblaf/apple/commit/dcbc065ed738f67d1301ba6b7e42c897bbfdc257))
* **deps:** update dependency beartype to &gt;=0.22,&lt;0.23 ([#55](https://github.com/liblaf/apple/issues/55)) ([40e7a51](https://github.com/liblaf/apple/commit/40e7a5139ce5302d86da6eddb967d7e2523c0a94))
* **deps:** update dependency jax to &gt;=0.7,&lt;0.8 ([#49](https://github.com/liblaf/apple/issues/49)) ([05e9451](https://github.com/liblaf/apple/commit/05e9451f9d332c01d257a733d0341323f564c02f))
* **jax/math/rotation:** correct sign handling in SVD-based rotation computation ([76eea15](https://github.com/liblaf/apple/commit/76eea158a320b49a059c597a09c3acccf0ad219c))
* **jax:** Correct activation transformation and add relative energy to PNCG ([0cf40cc](https://github.com/liblaf/apple/commit/0cf40cc71944111bffa5ab4a20e2586ee8444386))
* **uv.lock:** update simplejson source URLs and wheel paths ([d5b3ce3](https://github.com/liblaf/apple/commit/d5b3ce31f877ebdd88a08688c9ba8256fb06d647))
* **warp:** correct Hessian operations and add comprehensive tests ([11588aa](https://github.com/liblaf/apple/commit/11588aa9d36eabf535b811b02c8f0360efbc9784))
* **warp:** properly convert numpy arrays with matrix/vector dtypes ([5586617](https://github.com/liblaf/apple/commit/55866172d56befdfe8280ea8a4e5dd30c9362639))


### 📝 Documentation

* **README:** update banner and badge links ([ac14a00](https://github.com/liblaf/apple/commit/ac14a00c495c8c32fa2b7c122f02012becd3716a))
* **readme:** update socialify badge with transparent pattern ([0339381](https://github.com/liblaf/apple/commit/033938180ea6031aaa93b28a7089e92dd8218efd))


### ♻️ Code Refactoring

* **examples:** clean up dynamics examples ([6b29fa8](https://github.com/liblaf/apple/commit/6b29fa8881f4f57cfdd31419ac1fad62577dac1e))
* **jax:** move tree utilities to jax.tree module ([38966d1](https://github.com/liblaf/apple/commit/38966d1dc74667ea8a21b3511ce3989505ff7be5))
* refactor code structure and remove redundant changes ([3380053](https://github.com/liblaf/apple/commit/33800534f5952a7f5eb5aa5331af5feb8b22ba20))
* **sim:** improve simulation components ([46593af](https://github.com/liblaf/apple/commit/46593aff9e80000ba7507f0a3bdf06d97a7bc68f))
* **sim:** restructure core components for performance ([292835f](https://github.com/liblaf/apple/commit/292835f03449c0bd7421d8b3e4fb14ba0d3fb1ca))
* **sim:** simplify actor API and rename DOF/Dirichlet properties ([bc99c4a](https://github.com/liblaf/apple/commit/bc99c4a70388ee993646308c777c2245a3187f1f))
* **sim:** simplify DOFs and attribute access ([e5fee29](https://github.com/liblaf/apple/commit/e5fee299a8dc1b0f364aeeb6b20dd22fc5baac7f))
* **state:** simplify state updates and container implementations ([f007af5](https://github.com/liblaf/apple/commit/f007af5f23fe8bfd64488f3b6d18487651c62f74))
* **struct:** migrate to Equinox PyTree system ([a5e4888](https://github.com/liblaf/apple/commit/a5e488899aed60472e929ce0a6bdc711708a1ccc))
* **struct:** replace equinox-based PyTree with attrs-based implementation ([d25aaba](https://github.com/liblaf/apple/commit/d25aaba52575d0c7f82f33c453b3280e5e9f1a5d))
* Update dependencies and refactor pytree registration ([d685f16](https://github.com/liblaf/apple/commit/d685f16b9f87e65f1575b1fab98282098b4754eb))
* **warp:** improve type flexibility and add utility functions ([c9d66c6](https://github.com/liblaf/apple/commit/c9d66c621026a0556eadde03b8b45c2bec1ac12f))


### ✅ Tests

* **benchmark:** Add volume aggregation benchmarks for Warp and JAX ([4631a1e](https://github.com/liblaf/apple/commit/4631a1e586a62c232df760a0bd37314500f68a55))
* disable JIT in scipy minimizer tests and adjust tolerances ([fdc27be](https://github.com/liblaf/apple/commit/fdc27be23b633c316b4c632c6719d112a42dcda7))
* **optim:** add tests for MinimizerScipy and refactor Objective partial method ([5eb4ac3](https://github.com/liblaf/apple/commit/5eb4ac3c79de2c79ab46f49cac343032a7b7eac1))

## [0.3.0](https://github.com/liblaf/apple/compare/v0.2.1..v0.3.0) - 2025-07-11

### 💥 BREAKING CHANGES

- **jax:** simplify PyTree registration and jit utilities - ([32147c2](https://github.com/liblaf/apple/commit/32147c2b83f8d8ee0b3e6b0d62b29a1c9d236ef3))

### ✨ Features

- **sim:** add energy tracking callback and improve optimizer - ([02b5003](https://github.com/liblaf/apple/commit/02b5003931e1f9cb2e0a346b7daa616c02a98a4a))
- **sim:** enhance free-falling example with energy tracking - ([8895f79](https://github.com/liblaf/apple/commit/8895f79b9446a5bd5df20f366331fd4d623962c4))

### 🐛 Bug Fixes

- **jax:** enable filter by default in jit wrapper - ([dfd7641](https://github.com/liblaf/apple/commit/dfd7641a020e089275de1ba5cc7096c07c769b8d))

### ♻ Code Refactoring

- **examples:** adjust material and optimization parameters - ([67cf83d](https://github.com/liblaf/apple/commit/67cf83d6ddf0547083535d4039c5a6d9e612ce05))

### ❤️ New Contributors

- [@liblaf](https://github.com/liblaf) made their first contribution
- [@liblaf-bot[bot]](https://github.com/apps/liblaf-bot) made their first contribution

## [0.2.1](https://github.com/liblaf/apple/compare/v0.2.0..v0.2.1) - 2025-07-04

### ✨ Features

- **examples:** add collision simulations and improve physics - ([10c6916](https://github.com/liblaf/apple/commit/10c6916acf633d07e5bb01749b381b28d293dfc3))

### ⬆️ Dependencies

- **deps:** update dependency python to 3.13.\* (#30) - ([6aebf70](https://github.com/liblaf/apple/commit/6aebf70ecbbdb92290aa21f317c84e02335659e7))
- **deps:** update dependency python to 3.13.\* (#18) - ([4643411](https://github.com/liblaf/apple/commit/46434111bdfe0626f7a7ea89ac7a8cc2e2e18516))

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
