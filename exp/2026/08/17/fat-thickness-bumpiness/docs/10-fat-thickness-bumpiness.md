# Fat Thickness And Bumpy Artifacts

## 结论

在这个无表皮能量的抛物面 toy 上，增厚脂肪层会稳定地降低肌肉激活传到表面的高频起伏。

- 固定相同激活时，`thick` 相对 `thin` 的 vertical-displacement high-pass RMS 降低
  `18.0%`，归一化 high-pass 降低 `27.4%`，归一化 Laplacian RMS 降低
  `29.9%`。与此同时，整体 vertical-displacement RMS 反而增加 `12.9%`，因此结果不是简单地把所有位移都压小。
- 三个 setup 分别从零激活做 inverse，再在真正的共同网格 target fidelity 下选帧后，
  `thick` 相对 `thin` 的全域 high-pass RMS 降低 `32.2%`，归一化 high-pass
  降低 `35.1%`；muscle footprint 内分别降低 `33.6%` 和 `35.5%`。
- interior crop、muscle footprint 和 first-crossing 复核得到相同的效应方向。图上也能看到
  `thin -> current -> thick` 的高频块状结构逐渐减弱。

这为“较厚的低剪切、近不可压脂肪层能够缓冲肌肉引起的表面高频起伏”提供了当前 toy
内的 mechanism-level evidence，但还不能称为模型的最终上限：matched-fidelity 阈值约为
`0.8525`，锚点 `thick` 没有收敛，三个 setup 的局部 target fidelity 也没有完全匹配。

## Setup

实验采用同一个抛物面 rest/target 构型，只改变外缘高度，从而得到三种脂肪厚度。抛物面
网格参数为 `32`，rest amplitude 为 `0.12`，target amplitude 为 `0.02`；TetWild
使用 `relative_edge_length_fac = 0.02`。肌肉区域固定在
`x,z in [0.35, 0.65]`、`y in [0.04, 0.06]`。

| case | rim height | minimum fat | center fat | points | tets | active muscle tets / activation DoFs | fat volume |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 0.07 | 0.01 | 0.13 | 11,641 | 58,103 | 2,021 / 12,126 | 0.103209 |
| `current` | 0.10 | 0.04 | 0.16 | 13,067 | 65,796 | 1,589 / 9,534 | 0.133166 |
| `thick` | 0.14 | 0.08 | 0.20 | 17,592 | 91,652 | 1,342 / 8,052 | 0.173051 |

体材料参数在三案中保持一致：fat 为 `E = 0.003 MPa, nu = 0.49`，muscle 为
`E = 0.03 MPa, nu = 0.49`，aponeurosis 为 `E = 0.10 MPa, nu = 0.35`。
所有正式 forward/inverse 都关闭 skin energy 和 skin pre-strain，以便突出 bumpy
artifacts。三案的 target/rest 顶面面积比为 `0.962886`、`0.962879`、`0.962859`，
目标几何基本一致。

## 1. Fixed-activation proof of mechanism

三案施加完全相同的每个肌肉 tet 激活
`ActivationInv = (0.25, 0, 0, 0, 0, 0)`，三个 forward solve 均成功。不同 remesh
的表面先插值到相同的 `129 x 129` x-z 网格；high-pass cutoff 为 `8 cycles/unit`，
Laplacian 前的 Gaussian smoothing length 为 `0.04`。

| case | displacement-y RMS | high-pass RMS | high-pass / RMS | Laplacian RMS | Laplacian / RMS | target error / target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 4.7937e-4 | 6.2520e-5 | 0.13042 | 0.05510 | 114.95 | 0.993754 |
| `current` | 4.9998e-4 | 6.0509e-5 | 0.12102 | 0.04768 | 95.37 | 0.993343 |
| `thick` | 5.4133e-4 | 5.1257e-5 | 0.09469 | 0.04359 | 80.53 | 0.993942 |

`thick` 相对 `thin` 的 high-pass RMS、high-pass/RMS、Laplacian RMS 和
Laplacian/RMS 分别变化 `-18.0%`、`-27.4%`、`-20.9%` 和 `-29.9%`。四个核心
roughness 指标的方向一致。单独的 high-frequency power fraction 在 `current` 上不单调，
因此没有把它作为主结论。

![Fixed-activation common-grid fields](../data/30-cross-grid-fields.png)

这部分是 mechanics proof-of-mechanism，不是 target reconstruction：三案的 target-error
fraction 都约为 `0.994`，固定激活只产生了很小的目标位移。

### Grid and smoothing sensitivity

使用同一批正式 forward 结果做了四组补充的本地 sensitivity rerun。下表均为
`thick` 相对 `thin` 的变化；正式设置一并列出作为基准。

| common grid | smoothing length | high-pass RMS | high-pass / RMS | Laplacian / RMS |
| ---: | ---: | ---: | ---: | ---: |
| 129 | 0.02 | -10.6% | -20.8% | -33.4% |
| 129 | **0.04 (formal)** | **-18.0%** | **-27.4%** | **-29.9%** |
| 129 | 0.08 | -10.8% | -21.0% | -3.6% |
| 65 | 0.04 | -13.3% | -23.2% | -30.1% |
| 257 | 0.04 | -20.3% | -29.4% | -29.8% |

因此，增厚后 high-pass 下降的符号对网格分辨率和 smoothing length 是稳定的，数值幅度
则依赖分析尺度。尤其是 smoothing length `0.08` 时，归一化 Laplacian 的区分度明显下降。
这些 sensitivity 输出位于 `tmp/sensitivity/`，不是单独的正式 Comet run。

## 2. Independent inverse solves

每个 thickness setup 都新建 forward model、activation tensor 和 Adam optimizer，并从
全零 `ActivationInv` 独立求解。优化变量是每个 muscle tet 的 unconstrained 6-DoF
activation；不同 setup 之间没有 transfer 或 warm start。inverse 使用纯 L2 data loss，
activation smoothness 和 residual-Laplacian auxiliary loss 均关闭。

共同参数为 learning rate `0.03`、最多 `200` optimizer steps、plateau patience `20`、
minimum loss delta `1e-8`。

| case | temporal frames | native best step | native best error / target | status |
| --- | ---: | ---: | ---: | --- |
| `thin` | 178 | 157 | 0.847771 | converged: 20-step loss plateau |
| `current` | 201 | 200 | 0.834090 | step limit |
| `thick` | 201 | 193 | 0.846540 | step limit |

`current` 和 `thick` 到 `200` steps 仍未触发 plateau，所以 source manifest 正确地记录为
`complete = false`。这不是 solver failure：三案 `status = ok`，`hard_failures = []`，所有
`580` 个 trace record 的 forward/adjoint 均成功，且 `validation/errors = []`。
由于正式 run 开启了 `require_convergence`，stage 40 在写完这些有效产物后以“两个 case
未达到 plateau”的 `RuntimeError` 结束；Comet 上的 exception 应按 fixed-budget
non-convergence 解读，而不是 forward/adjoint failure。

native-mesh best error 不能直接用来比较 bumpiness，因为三案表面采样不同；同时，直接比较
各案最优帧会混入 target fidelity 差异。下一节因此重新读取完整 temporal histories，在同一
网格上重新计算 fidelity 并选帧。

## 3. True common-grid matched fidelity

### Selection rule

分析没有截断到 native best，而是扫描了三条完整 temporal trace：
`thin 178 + current 201 + thick 201 = 580` 帧。每一帧都插值到共同的 `129 x 129`
x-z 网格，再计算 common-grid residual RMS / target RMS。选帧过程为：

1. 在每案完整 trace 上找到 common-grid fidelity 的最小值及其 step；
2. 定义
   `tau_common = max(case-wise minimum) = 0.8525080124`；
3. 只在 `step <= common-grid best step` 的下降轨迹内，选择不高于 `tau_common` 且最接近它的
   frame，作为 primary `closest-from-below`；
4. 另取最早跨过同一阈值的 `first-crossing`，检查 path sensitivity。

| case | common-grid best: step / ratio | primary step | primary common ratio | native ratio at primary | first crossing | activation RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 157 / 0.850868 | 153 | 0.852492 | 0.849388 | 153 | 0.9005 |
| `current` | 200 / 0.834314 | 143 | 0.852194 | 0.851979 | 142 | 0.8912 |
| `thick` | 193 / 0.852508 | 193 | 0.852508 | 0.846540 | 193 | 1.0576 |

primary common-grid ratio 的 spread 为 `0.0003140`，通过预设的 `0.001` absolute gate。
native ratios 只作为诊断保留，不参与阈值或选帧。`tau_common` 的唯一锚点是 `thick`，而
`thick` 是 step-limit case；因此这里的正式解释范围是
**fixed-budget matched trajectory, not a converged upper bound**。

### Matched metrics

下面均为 primary frame。interior mask 排除距四周边界小于
`3 x smoothing length = 0.12` 的点，共 `9,409` 点；muscle-footprint mask 使用
`x,z in [0.35, 0.65]`，共 `1,521` 点。这两组抗边界指标用于排除 Gaussian
`mode=nearest` 和边缘离散的混杂。

| case | global HP RMS | global HP / RMS | global Lap. RMS | global Lap. / RMS | interior HP RMS / norm. | muscle HP RMS / norm. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thin` | 0.001867 | 0.11365 | 1.8915 | 115.16 | 0.002479 / 0.11363 | 0.005141 / 0.10201 |
| `current` | 0.001346 | 0.08627 | 1.4611 | 93.66 | 0.001782 / 0.08602 | 0.003569 / 0.07564 |
| `thick` | 0.001266 | 0.07372 | 1.4755 | 85.89 | 0.001660 / 0.07294 | 0.003413 / 0.06578 |

`thick` 相对 `thin` 的 effect sizes 为：

| mask / metric | raw high-pass RMS | normalized high-pass | Laplacian RMS | normalized Laplacian |
| --- | ---: | ---: | ---: | ---: |
| global | -32.2% | -35.1% | -22.0% | -25.4% |
| interior | -33.0% | -35.8% | — | — |
| muscle footprint | -33.6% | -35.5% | — | — |

high-pass 指标从 `thin -> current -> thick` 单调降低。需要保留一个细节：global raw
Laplacian RMS 在 `thick` 比 `current` 高 `1.0%`，但仍比 `thin` 低 `22.0%`；归一化
Laplacian 则继续下降。

![Matched-fidelity common-grid fields](../data/50-matched-fidelity-fields.png)

只有 `current` 的两种选帧规则不同一个 step（primary `143`，first-crossing `142`）。八个
核心 global/interior/muscle 指标在两种规则下的最大相对变化为 `0.81%`，所有 thickness
effect 的符号保持一致，说明结论对这一离散选帧选择不敏感。

## Validation and remaining limitations

### Validation passed

- 固定激活的三个 forward solve 全部成功。
- inverse 三案都从 fresh zero 开始；optimizer 独立；没有 auxiliary loss，也没有 skin energy。
- 三案 trace 共 `580` 帧，全部 forward/adjoint 成功；VTKHDF temporal frame 中的
  `inverse_step`、`error` 和 `loss` 与 JSON trace 逐帧核对。
- common-grid selected fidelity spread `0.0003140 < 0.001`；interior 和 muscle masks
  的样本数远高于最低门槛。
- `thin` 有 4 个共同网格边界点无法线性插值；它们只为画图做 nearest fill，不进入正式
  metrics。`current` 和 `thick` 没有这类点。
- target/rest 顶面面积比跨案最大差约 `2.7e-5`。muscle 和 aponeurosis volume 的跨案
  range/mean 分别约 `1.31%` 和 `0.70%`。

### Limitations

1. `tau_common = 0.852508` 仍代表约 `85%` 的 target-normalized RMS error。锚点
   `thick@193` 没有收敛，`current` 也在 step limit；继续优化可能改变各案 minima、阈值和
   排序。本结果只能说明固定预算下的 matched trajectory，不能说明每个模型的 converged
   reconstruction upper bound。
2. 三个 thickness geometry 分别 remesh，mesh size、active muscle tet 数和 activation DoFs
   不同（`12,126 -> 9,534 -> 8,052`）。虽然材料、肌肉包围盒和体积接近，这个 activation
   spatial-resolution 变化仍是 fat-thickness effect 的混杂。还需要 fixed-connectivity 或重复
   remesh 才能把效应完全归因于厚度。
3. fat 是 `E = 0.003 MPa, nu = 0.49` 的近不可压弹性**固体**。它能表达低剪切/高体积
   刚度，但没有真实脂肪的流动、poroelastic/viscoelastic 时间效应或组织间滑移，不能把当前
   结果表述成 fluid-fat simulation。
4. common-grid matching 只约束全域 fidelity。muscle footprint 内的 error/target ratios 为
   `thin 0.4827`、`current 0.5233`、`thick 0.4921`；局部 fidelity 并未匹配，尤其
   `current` 更差。这限制了局部 bumpy 数值的严格因果比较。
5. 当前只有一个 toy geometry、一个 target 和一组 fixed activation，没有 remesh replicate
   或统计置信区间。分析尺度也会影响 effect magnitude。
6. 关闭 skin energy 是为了暴露 artifacts；结论不能直接外推到带异质表皮、预应变和真实面部
   解剖结构的模型。

## Next experiment: skin material heuristic

脂肪厚度已经完成 toy-level mechanism check。下一步固定已经 prepared 的 human-face
Smile geometry 和同一个 target，不再同时改变脂肪厚度；在这个 setup 上加入 skin energy，
先摸清手动/低维材料启发式的上限：

1. 由每个 skin triangle 的 target/rest area ratio 构造两个平滑区域场：面积增大处降低
   Young's modulus，使其更容易拉伸；面积缩小处增加 pre-strain，把表皮绷住以抑制褶皱。
2. 先做小规模离散网格，例如 expansion-region `E` scale 与 contraction-region pre-strain
   gain 的组合。材料场要有正值、幅度、空间平滑和区域范围门禁。
3. 每个 material candidate 都从 fresh-zero muscle activation 做独立的 per-tet 6-DoF
   inverse；不能迁移本实验或其他材料 setup 的 activation。
4. 同时报告 target fidelity、global/interior/muscle-footprint roughness、局部 fidelity 和
   形变有效性。在共同网格上做 fixed-budget 与 matched-fidelity 两套比较，避免只看 native
   best。
5. 如果手动/低维 heuristic 无法在保留 target fidelity 的同时降低 bumpy，再把 bounded、
   regularized skin `E`/pre-strain 参数加入 inverse，与 muscle activation 联合求解。先优化少量
   区域参数，再考虑逐 triangle 参数，以减轻不可辨识性。

## Reproducibility

正式产物和对应记录如下。commit 是包含该 stage 产物的 artifact commit。

| stage | artifact commit | Comet run |
| --- | --- | --- |
| mesh preparation | `8e0cf7589b63c620bf740dc98533791277fd5087` | [e0519359206341a0b67cf562c8bd46a1](https://www.comet.com/liblaf/apple/e0519359206341a0b67cf562c8bd46a1) |
| fixed-activation forward | `edc9e732c3fcb063d60de4c0d97121c113257eec` | [9afa9ea6c64f4af89a23c0669b19c0da](https://www.comet.com/liblaf/apple/9afa9ea6c64f4af89a23c0669b19c0da) |
| fixed-activation common-grid analysis | `a4f9076409b889cbb431df85e13226eff5a31c43` | [b40b7bcac2ed41d793e5f3f48ec15c12](https://www.comet.com/liblaf/apple/b40b7bcac2ed41d793e5f3f48ec15c12) |
| independent inverse | `caf71694bbfc8dedb7e2356cf17611e4ae573e56` | [e30cd665c94c4e9db1bedbde74f9f107](https://www.comet.com/liblaf/apple/e30cd665c94c4e9db1bedbde74f9f107) |
| true common-grid matched analysis | `ce5c3d2b189e9fc5470298cfd72036e114737694` | [31530318116346ec9ebb91fb9e922596](https://www.comet.com/liblaf/apple/31530318116346ec9ebb91fb9e922596) |

从实验目录运行的 entrypoint 顺序为：

```bash
uv run python src/10-prepare-fat-thickness-sweep.py
uv run python src/20-forward-fat-thickness-sweep.py
uv run python src/30-analyze-fat-thickness-sweep.py
uv run python src/40-inverse-fat-thickness-sweep.py
uv run python src/50-analyze-matched-fidelity.py
```

prepare run 的 Comet 日志记录了 amend 前的 Git SHA
`9a0f087cbebac43dec13aa6f596c4cdf03c77b1b`；清除无关 `uv.lock` 变更后的最终 artifact
commit 是表中的 `8e0cf758...`。另外，工作区的 `logs/30-analyze-fat-thickness-sweep.log`
后来被 debug smoke 覆写；正式 stage 30 应以 clean-final commit `a4f90764...`、Comet run
`b40b7b...` 和 `data/30-*` 为准。

核心机器可读结果：

- [`../data/10-prepare-manifest.json`](../data/10-prepare-manifest.json)
- [`../data/20-forward-manifest.json`](../data/20-forward-manifest.json)
- [`../data/30-cross-grid-metrics.json`](../data/30-cross-grid-metrics.json)
- [`../data/40-inverse-manifest.json`](../data/40-inverse-manifest.json)
- [`../data/50-matched-fidelity.json`](../data/50-matched-fidelity.json)
