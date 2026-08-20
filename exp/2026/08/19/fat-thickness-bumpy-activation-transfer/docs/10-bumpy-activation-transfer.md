# Bumpy Muscle Activation Transfer Through Fat Thickness

## 结论

这个实验直接回答：**同一块肌肉产生相同的空间起伏激活时，增厚肌肉上方的脂肪层，是否会减弱传到外表面的起伏？**

在固定材料、固定肌肉层、固定激活场和 deterministic nested mesh 的 cuboid toy 中，top-fat
厚度从 `0.04` 增加到 `0.12` 后：

- 肌肉—脂肪界面的 `k=4` vertical-displacement modal amplitude 只降低 `13.7%`，而外表面同一
  mode 的 amplitude 降低 `57.6%`；
- 主要指标 interface-to-surface modal transmission 从 `1.23286` 降到 `0.60607`，降低
  `50.8%`；
- paired surface response `u_bumpy - u_uniform` 的 RMS、high-pass RMS 和 Laplacian RMS
  分别降低 `56.5%`、`56.8%` 和 `57.4%`；
- `64 x 64` x-z refinement 保持相同的 monotone trend，transmission 的 thin-to-thick
  降幅为 `58.3%`。但各案绝对 amplitude 对横向分辨率仍有 `12%--34%` 的变化，因此数值幅度
  尚未达到 grid convergence。

因此，当前支持的窄结论是：**在这个无 skin、近不可压弹性固体 fat 的 controlled block 中，
较厚的 top-fat 会显著衰减这一个 `k=4` isochoric muscle-activation mode 从肌肉界面传到自由表面的
vertical modal response。** 这不是 anatomical face 的定量结论，也不是对任意 activation
frequency、amplitude 或 fat constitutive model 的普遍结论。

## 为什么原 pressure sweep 不能回答这个问题

原 cuboid thickness sweep 在自由 bottom-interior surface 施加 positive-y pressure，并通过
continuation 把 pressure 提高到 `0.60`；SMAS 只使用固定 active pre-strain。它测量的是压力造成的
总位移场在不同厚度下的 p95-p05 和 Laplacian。这个实验可以说明不同 thickness 对大变形 load
response 的影响，但不能识别“bumpy muscle activation -> bumpy outer surface”这条传递链：

1. 起伏的直接驱动是外加 pressure 和边界约束，不是空间变化的 muscle activation；
2. 指标作用于各案的总位移，smooth/common response 没有通过 paired control 消掉；
3. 三个 thickness 分别 remesh，volumetric discretization 也随 case 改变。

本实验不施加该 pressure。每个 thickness 都在同一张 mesh 上分别求 uniform 和 bumpy activation
的平衡态，并以 `u_bumpy - u_uniform` 为 response；跨 thickness 的 lower slab 则采用完全相同的
structured connectivity。这样测试对象才是 fat 对 muscle-induced spatial mode 的传递。

## Paired experimental design

### Geometry, material, and boundary conditions

模型的 x-z footprint 为 `1 x 1`。bottom-fat 和 muscle thickness 固定为 `0.04` 和 `0.02`，
只增加 muscle 上方的 top-fat：

| case | top-fat thickness | total height | top-fat layers | points | tets | active muscle tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 0.04 | 0.10 | 4 | 26,411 | 138,240 | 27,648 |
| `medium` | 0.08 | 0.14 | 8 | 36,015 | 193,536 | 27,648 |
| `thick` | 0.12 | 0.18 | 12 | 45,619 | 248,832 | 27,648 |

正式网格为 `48 x 48` x-z cells，所有 y layers 的 spacing 为 `0.01`。fat 使用
`E = 0.003 MPa, nu = 0.49`，muscle 使用 `E = 0.03 MPa, nu = 0.49`。skin energy 和 skin
pre-strain 均关闭。bottom 全固定，四个 lateral sides traction-free，top surface free；runner
没有加入 pressure load。

每个 hexahedral grid cell 由确定性的 Kuhn/Freudenthal rule 分成 6 个 tetrahedra。三个 case
共享同一个 x-z lattice；bottom-fat 加 muscle 的 lower slab 坐标和 connectivity 完全相同，
增厚只是在上面追加同 spacing 的 fat layers。正式结果通过以下 invariants：

- `n_active_tets = 27,648` 在三案相同；
- active tet IDs、active centers 的 x-z coordinates、uniform activation 和 bumpy activation
  的 SHA-256 在三案相同；
- shared lower-slab connectivity SHA-256 在三案相同；
- 每个 case 内 uniform/bumpy/direct solves 使用同一 mesh。

这消除了原实验 independent remeshing 和 activation spatial-resolution 不同的主要混杂。全 mesh
的 point/tet 数随新增 fat layers 增加，这是 thickness manipulation 本身带来的自由度变化。

### `ActivationInv` semantics and isochoric active strain

constitutive model 保存的是 `ActivationInv = A^{-1} - I`，不是直接保存 `A`，也不是把
`ActivationInv_x = 0.25` 当作 `25%` displacement。对每个 active muscle tet，令 fibre-direction
strength 为 `s`，runner 构造

```text
A^{-1} = I + ActivationInv
       = diag(1 + s, (1 + s)^(-1/2), (1 + s)^(-1/2)),

A      = diag((1 + s)^(-1), (1 + s)^(1/2), (1 + s)^(1/2)).
```

因此 `det(A) = 1`。uniform control 使用 `s = 0.25`，对应 natural fibre stretch
`lambda_x = 1 / 1.25 = 0.8`，两个 transverse stretches 为 `sqrt(1.25) = 1.11803`。

bumpy condition 使用

```text
s(x, z) = 0.25 + 0.10 p(x, z),
p_raw(x, z) = cos(2 pi k x) cos(2 pi k z),  k = 4,
```

其中 sampled tet-centre pattern 先按 tet volume 去均值，再归一化到 volume-weighted RMS `1`。
所以 modulation 的 volume-weighted mean 为数值零、RMS 恰为 `0.10`；正式离散后的 `s` 范围是
`[0.053407, 0.446593]`。fat cells 的 activation entries 全为零。所有 active strain 的
`max |det(A)-1| <= 4.44e-16`。

### Paired response and primary transmission metric

每个 thickness 先求 uniform equilibrium，再以 activation modulation 的 `alpha = 0.5, 1.0`
continuation 求 bumpy equilibrium。primary response 是

```text
Delta u_y = u_y,bumpy - u_y,uniform.
```

在 muscle/top-fat interface 和 outer top surface 上，分别把去均值后的 `Delta u_y` 投影到同一个
zero-mean、unit-RMS nodal `k=4` pattern。令所得 absolute modal amplitudes 为
`a_interface` 和 `a_top`，主要指标定义为

```text
T_k = a_top / a_interface.
```

单看 `a_top` 会把“界面 source amplitude 自身随全局 stiffness 改变”也算进 fat effect；`T_k`
把实际产生在 muscle-fat interface 的该 mode 作为分母，更直接地度量它穿过 top-fat 后还剩多少。
`T_k` 是这个 finite-deformation equilibrium 中的 modal amplitude ratio，不是 energy transmission
coefficient；`thin` 的 `T_k > 1` 不能解释成能量放大。

另外报告 full paired response 的 RMS、Gaussian high-pass RMS（smoothing length `0.06`）和
finite-difference Laplacian RMS，检查结论是否只来自单一 projection。正式分析使用整个 top grid，
边界节点采用 trapezoidal weights，没有 interior crop。

## Production results (`48 x 48`)

| case | interface mode `a_interface` | top mode `a_top` | transmission `T_k` | paired top RMS | high-pass RMS | Laplacian RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 3.89432e-4 | 4.80115e-4 | 1.232862 | 5.08901e-4 | 4.49848e-4 | 0.618923 |
| `medium` | 3.52984e-4 | 3.31400e-4 | 0.938852 | 3.57021e-4 | 3.14595e-4 | 0.428507 |
| `thick` | 3.35985e-4 | 2.03631e-4 | 0.606071 | 2.21514e-4 | 1.94553e-4 | 0.263520 |
| `thick` vs `thin` | -13.7% | -57.6% | **-50.8%** | -56.5% | -56.8% | -57.4% |

三个 thickness 上，interface amplitude、top amplitude 和 primary transmission 都从
`thin -> medium -> thick` 单调下降。更关键的是，source interface amplitude 只变化 `13.7%`，
但 surface amplitude 和 interface-normalized transmission 分别下降 `57.6%` 和 `50.8%`；因此
结果不能只归因于 thicker case 的 muscle source 变弱。

top surface 上 prescribed mode 的 correlation 为 `0.9434`、`0.9282`、`0.9193`，modal energy
fraction 为 `89.0%`、`86.2%`、`84.5%`。即使 fat 增厚后 amplitude 变小，输出仍主要保持同一个
空间 mode；这支持把 amplitude reduction 解释为该 mode 的 attenuation，而不是 mode 被数值噪声
淹没。

![Interface-to-surface modal transmission](../data/10-bumpy-activation-transfer/10-modal-gain-vs-thickness.png)

下面三张图使用共同 color range，显示同一 paired quantity `u_y,bumpy - u_y,uniform`：

![Thin top-fat induced field](../data/10-bumpy-activation-transfer/10-thin-top-induced-field.png)

![Medium top-fat induced field](../data/10-bumpy-activation-transfer/10-medium-top-induced-field.png)

![Thick top-fat induced field](../data/10-bumpy-activation-transfer/10-thick-top-induced-field.png)

## Native ParaView assets

为便于组会展示，`src/21-build-bumpy-transfer-assets.py` 调用 ParaView `6.1.1` 的 `pvbatch`，从正式
VTU 生成了四组彼此独立的 `1800 x 1350` PNG/PVSM。三张 surface view 使用相同 camera、相同
symmetric color range `[-0.00165, 0.00165]`，并把各案 rest top plane 归一化到 `y=0`。几何只沿
vertical response 做 `x40` warp，图内明确标注该 exaggeration；因此它们用于比较 roughness
transfer，不表示 slab 的真实绝对高度。source view 不做 warp，显示三个 thickness case 共用的
`ActivationInvXModulation`。

![Shared bumpy activation source](../data/21-bumpy-transfer-paraview/source/21-shared-bumpy-activation-source-paraview.png)

![Thin top-fat ParaView response](../data/21-bumpy-transfer-paraview/thin/21-thin-bumpy-minus-uniform-paraview.png)

![Medium top-fat ParaView response](../data/21-bumpy-transfer-paraview/medium/21-medium-bumpy-minus-uniform-paraview.png)

![Thick top-fat ParaView response](../data/21-bumpy-transfer-paraview/thick/21-thick-bumpy-minus-uniform-paraview.png)

每张 PNG 都有同名 `.pvsm`；完整 identity、hash、camera、range 和 renderer receipt 见
[`../data/21-bumpy-transfer-paraview-manifest.json`](../data/21-bumpy-transfer-paraview-manifest.json)。

## Convergence, geometry, and branch evidence

每个 thickness 包含 4 个 forward equilibrium stages：uniform、`alpha=0.5` continuation、
`alpha=1.0` continuation，以及从 zero displacement 直接求完整 bumpy activation 的 independent
branch check。三案共 12 个 stage 全部为 `PRIMARY_SUCCESS`：

- solver 使用 `max_steps = 8000, atol = 1e-10, rtol = 1e-6`；实际步数为 `503--819`，final
  gradient norm 均小于 `1e-10`；
- 所有 stage 均无 inverted tetrahedra，也没有 `detF < 0.2` 的 tetrahedra；全局最小
  `detF = 0.933502`，全局最小 `q0.001(detF) = 0.962030`，远高于预设 gates `0.20/0.40`；
- direct branch 和 continuation branch 的 top-y RMS difference 分别为 `9.46e-9`、`1.73e-8`、
  `3.37e-8`，只占 paired signal RMS 的 `1.86e-5`、`4.84e-5`、`1.52e-4`；
- direct/continuation transmission relative difference 最大为 `2.54e-6`。

因此正式趋势不是 continuation path 选择或明显 invalid deformation 的产物。

## `64 x 64` x-z refinement

保持 y spacing、材料、activation wave、thickness 和所有 solver gates 不变，只把 x-z grid 从
`48 x 48` 提高到 `64 x 64`。refinement 的 machine-readable summary 位于
`tmp/refinement-nx64/10-summary.json`，持久化的两网格 comparison 位于
[`../data/15-grid-refinement-comparison.json`](../data/15-grid-refinement-comparison.json)。

| x-z grid | thin / medium / thick top modal amplitude | thin / medium / thick transmission | top-amplitude attenuation | transmission attenuation |
| --- | --- | --- | ---: | ---: |
| `48 x 48` | 4.80115e-4 / 3.31400e-4 / 2.03631e-4 | 1.232862 / 0.938852 / 0.606071 | 57.59% | 50.84% |
| `64 x 64` | 6.44840e-4 / 4.11615e-4 / 2.28476e-4 | 1.286449 / 0.918242 / 0.536212 | 64.57% | 58.32% |

相对 `48 x 48`，`64 x 64` 的 top modal amplitude 在 thin、medium、thick 上分别变化
`+34.3%`、`+24.2%`、`+12.2%`；transmission 分别变化 `+4.35%`、`-2.20%`、`-11.53%`。
thickness trend 不但保持单调，thin-to-thick attenuation 还更强。refinement 的全部 12 个 solves
成功，`min detF = 0.9242`、`min q0.001(detF) = 0.9593`，branch ratio 最大 `1.244e-4`。

不过绝对 amplitude 的 `12%--34%` resolution change 说明一个 refinement 还不能宣称 grid
convergence。正式结论应强调 effect direction 和两个分辨率下的 attenuation range，不应把
`50.8%` 当作 continuum-limit 常数。

![Grid-refinement sensitivity](../data/15-grid-refinement-sensitivity.png)

## Half-amplitude linearity check

为检查 `u_bumpy - u_uniform` 是否只是当前 RMS `0.10` 下的 nonlinear coincidence，另在正式
`48 x 48` 网格上把 activation modulation RMS 减半到 `0.05`，其余设置不变：

| case | interface amplitude half/full | top amplitude half/full | half-vs-full transmission change |
| --- | ---: | ---: | ---: |
| `thin` | 0.499949 | 0.499904 | -0.0090% |
| `medium` | 0.500029 | 0.499985 | -0.0090% |
| `thick` | 0.500055 | 0.499989 | -0.0133% |

相对理想的 `0.5` scaling，interface 最大偏差 `0.011%`，top 最大偏差 `0.019%`。thin-to-thick
top-amplitude attenuation 在 half/full amplitude 下为 `57.580% / 57.587%`，transmission attenuation
为 `50.842% / 50.840%`。12 个 stages 全部 `PRIMARY_SUCCESS`，`min detF = 0.946875`，最大 branch
ratio 为 `2.99e-4`。因此在 RMS `0.05--0.10` 范围内 response effectively linear，thickness trend
不是特定 modulation amplitude 下的 nonlinear artifact。machine-readable summary 位于
`tmp/half-amplitude/10-summary.json`。

## Limitations and supported claim

1. 这是规则 cuboid 和单一 `k=4` cosine-product mode。没有扫描 spatial frequency、mean
   contraction 或 muscle localization；虽然 `0.05/0.10` modulation amplitude 的 linearity check
   通过，low-frequency/global modes 与更高频 modes 仍可能有不同 attenuation。
2. fat 是 `E = 0.003 MPa, nu = 0.49` 的近不可压弹性**固体**。它没有真实脂肪的
   poroelastic/viscoelastic dynamics、flow、fascia sliding 或 tissue-interface contact，不能把本结果
   表述成 fluid-fat simulation。
3. skin energy 被刻意关闭以隔离 fat transfer。真实 skin stiffness、pre-strain 和异质结构可能
   改变 surface amplitude 与 frequency response。
4. bottom-fixed、lateral-free 的边界条件会影响 global equilibrium，尤其是 thin case 的
   `T_k > 1`。当前没有 boundary-condition sensitivity，且 primary metric 包含边界节点。
5. deterministic nested mesh 消除了 independent remesh 的主要混杂，但当前只有一个 tetrahedral
   split pattern，没有 shifted-grid/remesh replicates 或统计置信区间。x-z refinement 也尚未
   grid-converged。
6. `u_bumpy - u_uniform` 是两个 finite-deformation equilibria 的 paired difference，不是严格的
   infinitesimal linear response。interface source amplitude 仍随 thickness 改变；primary
   transmission 已对此做 modal normalization，但它不是 energy conservation statement。
7. 各案除了 top-fat thickness 外材料和 lower slab 相同，但 total height、总 tetrahedra 数和全局
   compliance 必然随 thickness 改变。这正是本实验的 controlled intervention，不能再把观察到的
   effect 拆成更细的独立机制而不做额外实验。

当前最窄、可复核的表述是：

> 对这个 fixed-material、no-skin cuboid，在相同 `k=4`、RMS `0.10` 的 isochoric muscle
> `ActivationInv_x` modulation 下，把 top-fat 从 `0.04` 增至 `0.12`，使 interface-to-surface
> vertical modal transmission 在 `48 x 48` 与 `64 x 64` 网格上分别降低 `50.8%` 与 `58.3%`。
> 较厚 fat 对该 muscle-induced surface bump mode 具有明确的 attenuation，但 effect magnitude
> 尚未 grid-converged，也不能直接外推到真实面部。

## Reproducibility and outputs

正式 run 的工作目录为：

```text
/home/liblaf/Projects/liblaf/apple/exp/2026/08/19/fat-thickness-bumpy-activation-transfer
```

exact production command 为：

```bash
CHERRIES_NAME='Fat thickness filters bumpy muscle activation' \
CHERRIES_TAGS='fat,thickness,bumpy-activation,isochoric,nested-mesh,production' \
uv run python src/10-run-bumpy-activation-transfer.py
```

headless replay 使用 `matplotlib` 的 `Agg` backend，numerical stages 和 local output 全部完成，
process exit code 为 `0`。Comet page 为
[63b2215ae5fb4cd18f7180dd6a8f5759](https://www.comet.com/liblaf/apple/63b2215ae5fb4cd18f7180dd6a8f5759)，
但 run 结束时 environment details / git patch upload 报 warning，并最终提示 remote logging failed；
因此本报告把本地 JSON/CSV/VTU、Cherries snapshot 和 exit-0 run 作为 authoritative evidence，
不把远端页面视为完整 provenance record。

核心产物：

- [`../data/10-bumpy-activation-transfer-summary.json`](../data/10-bumpy-activation-transfer-summary.json)：
  完整 config、invariants、per-stage diagnostics、case metrics 和 effect sizes；
- [`../data/10-bumpy-activation-transfer/10-bumpy-activation-transfer.csv`](../data/10-bumpy-activation-transfer/10-bumpy-activation-transfer.csv)：
  三案对照表；
- `../data/10-bumpy-activation-transfer/{thin,medium,thick}/`：每案 VTU、top-grid NPZ 和 JSON；
- `../data/10-bumpy-activation-transfer/10-*-vs-thickness.png` 与
  `10-*-top-induced-field.png`：正式 plots；
- [`../data/15-grid-refinement-comparison.json`](../data/15-grid-refinement-comparison.json) 与
  [`../data/15-grid-refinement-sensitivity.png`](../data/15-grid-refinement-sensitivity.png)：持久化的
  `48 x 48` / `64 x 64` sensitivity 对照；
- [`../data/21-bumpy-transfer-paraview-manifest.json`](../data/21-bumpy-transfer-paraview-manifest.json) 与
  `../data/21-bumpy-transfer-paraview/{source,thin,medium,thick}/`：native ParaView PNG/PVSM 及 hash receipt；
- `../logs/10-run-bumpy-activation-transfer.log`：Cherries local log；
- `../../../../../../.cherries/runs/2026/08/19/fat-thickness-bumpy-activation-transfer/`：local Cherries
  snapshots；
- `../tmp/refinement-nx64/10-summary.json`：`64 x 64` refinement summary；
- `../tmp/half-amplitude/10-summary.json`：RMS `0.05` amplitude-linearity summary。

在 headless replay 前，一次数值相同的 formal run 已写完全部 outputs，但在 GUI/Tk cleanup 阶段
以 exit `132` 结束；修正 backend 后的 exact replay 与它的 summary values 达到 machine precision
一致。前一次 outputs 保留在 `tmp/production-exit132/`，只用于重现 cleanup diagnosis，不作为正式
结果目录。
