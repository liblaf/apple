# Smile 夸张异质表皮材料实验

> **本构修正（2026-08-18）**：本实验为保持旧 baseline 可比性，给 Koiter
> skin 使用了 3D Lamé 转换。对由 3D `E, nu` 定义、厚度方向为 plane stress 的薄膜，
> 正确的面内系数应为 `lambda = E * nu / (1 - nu^2)`；本实验使用的
> `lambda = E * nu / ((1 + nu) * (1 - 2 * nu))` 将 `nu=0.49` 时的面积模量
> 高估约 `17.1x`。进一步的 source-topology 审计还确认，旧 Koiter 域包含
> `13,165` 个由 `InFaceConvex` 裁切产生的可动人工边界 triangles，以及骨、口腔和
> 眼眶等非 epidermal surfaces。以下结果仍是可复现的高面积刚度/full-boundary
> 膜机制实验，但不再作为物理 thin-skin baseline；所有含 skin 的定量结论与排序
> 等待独立 plane-stress + audited-domain 重跑验证。
> 此外，当时的 Koiter pre-strain 还用 `1 / det(Ainv)` 缩放了 energy
> weight；这与项目内 3D active-strain tets 的固定参考体积约定不一致。
> 旧产物和图表全部保留，作为组会中展示“错误本构下的可复现对照”，不会被覆盖。
> `no-skin`、volume 材料、target、ROI 和几何材料场不受这项修正影响。

## 结论

这轮实验的目的不是标定生理材料，而是把两个机制的作用拉开：扩张区降低 Young's
modulus，收缩区增加 pre-strain，然后观察 inverse physics 的 target fit 和 Bumpy 是否能
同时改善。

结果比较明确：

- 在接近相同的 target fidelity 下，`e005-p200` 相比 `e100-p000` 将收缩区 target-relative
  dihedral RMS 降低 `53.29%`，displacement-Laplacian RMS 降低 `40.98%`。
- `e100-p200` 已经得到几乎同量级的平滑效果；`e005-p000` 的平滑改善很小。因此当前
  screen 中，**pre-strain 是抑制 Bumpy 的主要来源**。
- `e005-p200` 在 40-step terminal state 的 target error、dihedral 和 Laplacian 相比
  baseline 分别改善 `9.28%`、`50.40%` 和 `37.37%`。它也是有表皮方案中 target fit 和
  Laplacian 最好的一个；dihedral 与 prestrain-only 基本相同。
- `no-skin` 的 terminal target error 最低，但表面最粗糙。这说明纯 target L2 会继续用
  高频形变换取拟合，表皮能量确实在约束这类自由度。
- `e005` 只是公式里的 minimum scale。复核全 skin 面积后，只有 `16.638%` 的面积满足
  `ExpansionWeight > 0`，whole-skin area-mean Young's modulus 仍为 `0.19445 MPa`，接近
  baseline 的 `0.2 MPa`。因此本轮还没有真正测到“大面积软表皮”的上限；不能据此断言
  Young's-modulus softening 不重要。

本报告按当前项目约定处理局部 inversion/fold：它们记录为视觉检查 warning，不再作为
trajectory 或 checkpoint 的剔除条件。三张正式图中未见由这些小区域单独造成的明显视觉
artifact；`no-skin` 的大范围粗糙则肉眼可见。

## 1. 材料场与实验设计

### 1.1 Heterogeneous 公式

材料参数按 skin triangle 设置，不是 homogeneous 常数。记第 `i` 个 triangle 的扩张权重为
`w_i = ExpansionWeight_i`，收缩强度为
`c_i = ContractionSeverityLogCapped_i`。本次输入场的实际范围为：

```text
0 <= w_i <= 0.709369275
0 <= c_i <= 0.417465914
```

Young's modulus 使用

```text
E_i = 0.2 MPa * young_min_scale ** w_i
```

isotropic in-plane pre-strain 使用

```text
a_i = exp(0.5 * prestrain_gain * c_i) - 1
ActivationInv_i = (a_i, a_i, 0)
stress-free area ratio_i = 1 / (1 + a_i)^2
```

solver 实际读取的 Lamé 参数由每个 triangle 的 `E_i` 转换：

```text
lambda_i = E_i * nu / ((1 + nu) * (1 - 2 * nu))
mu_i     = E_i / (2 * (1 + nu))
```

`nu`、skin thickness、体材料、target、mesh 和 loss 均保持不变。`e005` 的实际最小
`E` 是 `0.0238847 MPa`，不是 `0.01 MPa`，因为当前 `w_i` 最大值小于 1。

### 1.2 三个新案例和三个复用案例

<!-- markdownlint-disable MD013 -->

| candidate | 含义 | 来源 | Young's modulus 实际范围 | `Ainv` 最大值 | 最小 stress-free area ratio |
| --- | --- | --- | ---: | ---: | ---: |
| `e100-p000` | baseline | 复用 2026-08-17 | `0.2 MPa` | `0` | `1` |
| `e100-p200` | prestrain only | 新跑 2026-08-18 | `0.2 MPa` | `0.518110` | `0.433904` |
| `e005-p000` | softening only | 新跑 2026-08-18 | `0.0238847..0.2 MPa` | `0` | `1` |
| `e005-p200` | combined extreme | 新跑 2026-08-18 | `0.0238847..0.2 MPa` | `0.518110` | `0.433904` |
| `e025-p100` | current moderate | 复用 2026-08-17 | `0.0748078..0.2 MPa` | `0.232116` | `0.658714` |
| `no-skin` | 关闭 skin energy | 复用 2026-08-17 | N/A | N/A | N/A |

<!-- markdownlint-enable MD013 -->

三个旧 history 不是按文件名直接拼入，而是用固定 size 和 SHA-256 绑定 summary、trace 和
VTKHDF history；分析时还检查了 topology 和 41 个时间步。三个新案例各自重新构建 forward
model，从严格全零 muscle activation 独立开始。不同材料 setup 之间没有 activation transfer
或 warm start。

inverse 协议为固定 learning rate `0.3`、40 个 optimizer update，即 step `0..40` 共 41 次
forward/adjoint evaluation。优化变量仍是每个 active muscle tet 独立的 6-DoF unconstrained
activation。三个新案例的 `123` 个 evaluation 全部 forward/adjoint success，轨迹数值有限，
best step 都是 40；停止原因为 `step_limit_smooth_decrease`，不是收敛。

### 1.3 `e005` 的有效覆盖率复核

只看 `E_min` 会高估本轮 softening 的强度。按 `RestArea` 对完整 skin 加权，现场复核结果为：

- 全 skin 中 `w_i > 0` 的面积占比只有 `16.638%`；
- `young_min_scale=0.05` 时，whole-skin area-mean `E = 0.19445 MPa`；
- 在 `w_i > 0` 的 expansion 区内部，area-mean `E = 0.16665 MPa`；
- 全 skin 只有 `0.449%` 的面积达到 `E < 0.05 MPa`；
- 即使把 scale 再降十倍到 `0.005`，沿用同一 `w_i`，expansion mean 仍有
  `0.15043 MPa`。

所以当前限制主要是 **coverage 和 weight 幅度不足**，不只是 minimum scale 不够小。
这也限制了我们对 softening-only 结果的解释。

### 1.4 当前完整材料表

体网格不是把每个 tet 硬分成一种组织，而是在同一个 tet 上分别积分 aponeurosis、fat 和
muscle 三项能量，再乘各自的组织 fraction；三个 fraction 对每个 tet 精确和为 `1`。

<!-- markdownlint-disable MD013 -->

| 组织 | 本构 | `E` (MPa) | `nu` | `lambda` (MPa) | `mu` (MPa) | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fat volume | Stable Neo-Hookean | 0.003 | 0.49 | 0.0493289 | 0.00100671 | `FatFraction` 加权 |
| muscle volume | active Stable Neo-Hookean | 0.030 | 0.49 | 0.493289 | 0.0100671 | `MuscleFraction` 加权；逐 active tet 优化 6-DoF `ActivationInv` |
| aponeurosis volume | Stable Neo-Hookean | 0.10 | 0.35 | 0.0864198 | 0.0370370 | `AponeurosisFraction` 加权 |
| skin baseline | Koiter membrane | 0.20 | 0.49 | 3.28859 | 0.0671141 | thickness `0.001 m`；triangle fraction 为 1 |
| skin `e005` | Koiter membrane | 0.0238847..0.20 | 0.49 | 0.392735..3.28859 | 0.00801499..0.0671141 | 逐 triangle heterogeneous；whole-skin area-mean `E=0.19445 MPa` |

<!-- markdownlint-enable MD013 -->

当前 Stable Neo-Hookean 使用关于 `J-1` 的有限多项式形式，不含 `log(J)` 或独立 inversion
barrier。skin Koiter 项只有 in-plane metric energy，没有 bending term。虽然 mesh 中保留了
`SMASFraction`，当前 forward builder 没有读取它，也没有独立 SMAS 本构；骨性约束只通过
fixed vertices 表示。模型也没有 density、gravity、contact/collision、viscosity 或显式流体
脂肪参数。

表里的 MPa 是本项目的参数命名约定；solver 直接使用 `0.003/0.03/0.10/0.20` 这些数值，
没有乘 `1e6` 转成 Pa，而几何仍以 m 表示。当前没有真实外力时，共同缩放所有弹性能不改变
静力平衡；以后若加入 gravity、density 或带 SI 单位的外力，必须先统一单位。

另一个容易忽略的问题是：skin 的 Koiter 膜直接使用上述 3D Lamé 转换。`nu=0.49` 时
`lambda/mu=49`；若按二维膜的等效泊松比理解，相当于约 `0.9608`，面积变化惩罚很强。
降低 `E` 会等比例降低 `lambda` 和 `mu`，但不会改变这个相对的强保面积性质。

## 2. 执行、清理与复现状态

工作目录为：

```bash
cd exp/2026/08/18/human-face-smile-exaggerated-material-screen
```

本轮最终本地执行的核心命令为：

```bash
DEBUG=1 uv run --frozen python src/10-prepare-exaggerated-materials.py
DEBUG=1 uv run --frozen python src/20-inverse-exaggerated-material-screen.py
DEBUG=1 uv run --frozen python src/30-analyze-exaggerated-material-screen.py
```

`DEBUG=1` 只关闭远端 Comet 记录和自动 Git commit，并保留本地 Cherries snapshot；它不把
screen 改成数值 smoke。`20` 仍完整执行 40 steps，`30` 仍扫描全部 `6 x 41 = 246` 帧。
这里使用 DEBUG，是因为这些 VTKHDF/VTU/VTP 共数 GB；让 Cherries 的 Git 收尾经过 Git LFS
clean filter 会在 `.git/lfs/tmp` 再复制一份大文件，前一次尝试已经造成磁盘压力。

重跑前只清理可再生的 LFS tmp、失败 smoke/partial artifacts 和失效的本地 run snapshot；
2026-08-17 的三个正式 history 以及已经完成的正式数据保留。重跑全程监控
`.git/lfs/tmp`，最终仍为 `0`。正式 `data/` 约 `6.6 GiB`；完成 JSON、history 和图像读回
检查后，本轮约 `6.6 GiB` 的 `.cherries/` DEBUG snapshot 已删除，只保留正式 `data/`。
`.cherries/` 的总占用由约 `7.5 GiB` 降至约 `910 MiB`。该操作删除的是可再生的本地运行
快照，不是科研产物；快照不可直接恢复，但可由保留数据或重新运行生成。

为防止 DEBUG 重跑期间 Git/LFS 再次自动发现大文件，`.git/info/exclude` 中临时加入了：

```gitignore
/exp/2026/08/18/human-face-smile-exaggerated-material-screen/data/*.vtkhdf
/exp/2026/08/18/human-face-smile-exaggerated-material-screen/data/*.vtkhdf.tmp
/exp/2026/08/18/human-face-smile-exaggerated-material-screen/data/*.vtu
/exp/2026/08/18/human-face-smile-exaggerated-material-screen/data/10-exaggerated-materials/*.vtp
```

这是 repo-local、未提交的保护层，不改变实验文件。以后若要正式 stage 这些 binary，必须先
撤掉或调整该 exclude，再单独核对 LFS pointer，不能因为 `git status` 看不到就认为数据不存在。

`20` 的数值阶段和 aggregate artifacts 都完整，但 Cherries shutdown hook 最后尝试读取一个
已经不存在的 `.cherries/runs/.../logs/20-...log`，留下 `FileNotFoundError`。它发生在三个
41-frame history、case summary 和 aggregate summary 写完之后；`30` 又逐一按 hash、拓扑和
时间步读回了这些文件。因此这是本地 run 归档收尾不完整，不是 forward/adjoint 或科研数据
失败。本轮没有可引用的远端 Comet URL。

## 3. 指标与结果

三个主指标都越小越好：

- target error：`RMS(Displacement - TargetDisplacement) / RMS(TargetDisplacement)`，只在
  `SmileLossMask` 上计算；
- contraction dihedral：收缩 ROI 内部边上，deformed 与 target dihedral 差的
  rest-edge-length-weighted RMS；
- displacement Laplacian：`SmileLossMask` surface 上、固定 topology 的 legacy umbrella
  Laplacian RMS。

### 3.1 Terminal fixed-budget checkpoint

<!-- markdownlint-disable MD013 -->

| candidate | step | error/target | error RMS | contraction dihedral | displacement Laplacian | inverted tets | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e100-p000` | 40 | 0.602461 | 3.19915 mm | 8.54438 deg | 0.119084 mm | 29 | 49 |
| `e100-p200` | 40 | 0.583698 | 3.09952 mm | 4.21326 deg | 0.076901 mm | 24 | 120 |
| `e005-p000` | 40 | 0.564042 | 2.99514 mm | 8.84549 deg | 0.118017 mm | 51 | 58 |
| `e005-p200` | 40 | 0.546565 | 2.90233 mm | 4.23814 deg | 0.074585 mm | 23 | 104 |
| `e025-p100` | 40 | 0.551165 | 2.92676 mm | 4.95345 deg | 0.081942 mm | 33 | 80 |
| `no-skin` | 40 | 0.239161 | 1.26998 mm | 9.44490 deg | 0.218168 mm | 69 | 349 |

<!-- markdownlint-enable MD013 -->

Terminal 相对变化进一步说明了分工：

<!-- markdownlint-disable MD013 -->

| comparison | target error | dihedral | Laplacian |
| --- | ---: | ---: | ---: |
| prestrain-only vs baseline | -3.114% | -50.690% | -35.422% |
| softening-only vs baseline | -6.377% | +3.524% | -0.896% |
| combined vs baseline | -9.278% | -50.398% | -37.367% |
| combined vs moderate | -0.835% | -14.441% | -8.978% |
| no-skin vs baseline | -60.303% | +10.539% | +83.206% |

<!-- markdownlint-enable MD013 -->

负数表示改善。`no-skin` 的 target error 大幅下降，但两个粗糙度指标都变差；这不是一个
可接受的最终解，只是用来突出 target-fit/Bumpy trade-off。

### 3.2 Nearest-discrete common-fidelity checkpoint

为了避免把“多优化了几步”误认为“材料更平滑”，分析器取六个 terminal error fraction 的
最大值作为共同目标：

```text
tau = 0.602460923
```

每个案例只选择真实保存的、离 `tau` 最近的 step，不插值 geometry；并列时选较早 step。

<!-- markdownlint-disable MD013 -->

| candidate | step | error/target | error RMS | contraction dihedral | displacement Laplacian | inverted tets | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e100-p000` | 40 | 0.602461 | 3.19915 mm | 8.54438 deg | 0.119084 mm | 29 | 49 |
| `e100-p200` | 33 | 0.603282 | 3.20351 mm | 4.12568 deg | 0.076724 mm | 16 | 103 |
| `e005-p000` | 32 | 0.604040 | 3.20754 mm | 8.08607 deg | 0.112731 mm | 31 | 43 |
| `e005-p200` | 24 | 0.602016 | 3.19679 mm | 3.99108 deg | 0.070279 mm | 10 | 80 |
| `e025-p100` | 25 | 0.604714 | 3.21112 mm | 4.49928 deg | 0.078962 mm | 12 | 62 |
| `no-skin` | 8 | 0.607827 | 3.22765 mm | 5.09401 deg | 0.118130 mm | 1 | 16 |

<!-- markdownlint-enable MD013 -->

选中点的 fidelity 范围为 `0.602016..0.607827`，absolute spread 为 `0.005811`
（约为 `tau` 的 `0.96%`）。因此这只是 nearest-discrete matching，不是严格相同 fidelity。
最大偏差来自 `no-skin@8`；解释接近 1% 的效果时必须谨慎。对主要结论影响较小：

- prestrain-only vs baseline：dihedral `-51.715%`，Laplacian `-35.571%`；
- softening-only vs baseline：dihedral `-5.364%`，Laplacian `-5.335%`；
- combined vs baseline：dihedral `-53.290%`，Laplacian `-40.983%`；
- combined vs moderate：dihedral `-11.295%`，Laplacian `-10.996%`。

## 4. 视觉检查

![六案 target-fit 与 Bumpy trajectory](../data/30-exaggerated-material-screen-trajectories.png)

trajectory 图与数值一致：`no-skin` 最快降低 target error，但继续优化时两个 roughness 指标
迅速上升；有 pre-strain 的三条曲线整体位于更低的 roughness 区间。

![Terminal fixed-budget contact sheet](../data/30-exaggerated-material-screen-terminal-views.png)

Terminal contact sheet 使用统一的 canonical skin、GlobalPointId 映射、parallel projection、
固定相机、固定光照和固定颜色。每案都有 front、30-degree 和 mouth closeup：

- baseline 与 softening-only 在眼下、脸颊和口周仍能看到明显起伏；
- prestrain-only、combined 和 moderate 明显更平滑；
- `no-skin` 虽然嘴部更接近 target，但脸颊和口周出现大范围高频粗糙，肉眼最明显。

![Nearest-discrete matched-fidelity contact sheet](../data/30-exaggerated-material-screen-matched-views.png)

Matched sheet 中，prestrain-only 和 combined 相对 baseline 的平滑差异仍然存在，不依赖它们
在 40 steps 后取得更低 target error。`no-skin@8` 比 terminal 温和很多，也说明它的严重
Bumpy 是沿着优化轨迹逐步增长的。

### Inversion/fold 的处理

本轮把 inversion 和 fold 定义为 `visual-review-only` warning：记录数量、比例和最差局部值，
但不截断 trajectory，也不排除 terminal/matched checkpoint。原因是这些区域占比很小，当前
contact sheet 中看不出与计数一一对应的明显 artifact；计数本身也不是视觉严重程度指标。

这不等于它们在物理上正确。它只表示当前项目的首要验收是 target fit 与肉眼 Bumpy，而不是
建立严格的全局无翻转可行域。若后续出现肉眼可见的裂缝、翻面或 solver/non-finite failure，
仍应升级为 hard failure。材料准备阶段 `p200` 的最大相邻 `Ainv` jump 为 `0.09304`，超过旧
阈值 `0.08`，同样按 warning 记录；candidate readback 和 solver-content hard gate 均通过。

## 5. 当前能支持的机制结论

1. **Pre-strain 能明显压制 Bumpy。** 这一点同时出现在 terminal、matched metrics 和三视图，
   不是只靠某一个指标得出的。
2. **Pure target loss 不会自动选择平滑解。** `no-skin` 能把 target error 降到
   `0.23916`，但 terminal 表面最粗糙。
3. **Combined 是当前有表皮方案的最好折中。** 它比 moderate 的 target error 略低，同时
   matched dihedral/Laplacian 再低约 `11%`。
4. **本轮不能判定大范围 Young's-modulus softening 的上限。** `e005` 的 minimum 很低，
   但 coverage 很小、全 skin mean 几乎没变。softening-only 的弱平滑效果更准确的解释是
   “当前 weight 场作用有限”，不是“降低 Young's modulus 没用”。
5. **六个 setup 的 muscle activations 不能直接迁移。** 每个 setup 都独立 inverse；本轮比较
   的是各自 trajectory，不是把同一 activation 放进不同材料 forward model。

## 6. 局限

- 只有一个 Smile target，没有 replicate，也没有跨表情验证。
- 40 steps 是 fixed budget；三个新案都仍在平滑下降，`inverse/converged=false`。本报告不能
  声称达到材料 setup 的最终拟合上限或 KKT 点。
- `e005/p200` 是刻意夸张的 mechanism screen，不是生理参数估计。
- 三个旧案例来自 2026-08-17。虽然文件 identity、topology、时间步和协议均被绑定检查，
  它们并非本次同进程重跑。
- nearest-discrete matching 保留 `0.005811` fidelity spread，不能解释非常小的相对差异。
- dihedral 只覆盖 contraction ROI；umbrella Laplacian 只在固定 topology 内可比。两者都不是
  完整的视觉感知模型。
- inversion/fold count 只作 warning，不能作为物理有效性的证明，也不能替代肉眼检查。
- DEBUG 本地 snapshot 的 shutdown 归档有一次日志路径异常，因此没有完整远端 Comet 记录；
  科研结论以 `data/20-*`、`data/30-*` 和现场读回检查为准。

## 7. 下一步最小实验

下一步不继续把 `young_min_scale` 从 `0.05` 降到 `0.005`。同一 weight 场下，后者只会把
expansion mean `E` 从 `0.16665` 降到 `0.15043 MPa`，仍不足以测试软表皮上限。

固定 `prestrain_gain=2.0`，只增强 expansion weight 的覆盖强度：

```text
w'_i = min(weight_gain * w_i, 1)
E_i  = 0.2 MPa * 0.05 ** w'_i
```

做三个新案例即可：

| case | 变化 | 预期 expansion area-mean E | 作用 |
| --- | --- | ---: | --- |
| `weight_gain=4` | 扩大现有 softening 权重 | `0.12217 MPa` | 中等覆盖 |
| `weight_gain=8` | 进一步扩大权重 | `0.09793 MPa` | 强覆盖 |
| expansion `E≈0.01 MPa` | 近似均匀软化 | `≈0.01 MPa` | 非生理 upper bound |

沿用本轮 fresh-zero、LR `0.3`、step `0..40`、相同 target、相同 `p200` 和相同 terminal/matched
分析，不增加优化变量，也不做大网格 sweep。这个最小实验直接回答两个问题：

- 如果 `weight_gain=4/8` 随覆盖增强而持续改善 target-fit/Bumpy trade-off，下一步应调
  weight coverage，而不是继续压低 minimum scale；
- 如果只有 expansion `E≈0.01 MPa` upper bound 有明显变化，说明需要重新设计 expansion
  ROI/过渡；如果 upper bound 也没有明显改善，则 Young's-modulus softening 不是当前主要
  瓶颈，可以把精力转回 pre-strain 分布和 inverse objective。
