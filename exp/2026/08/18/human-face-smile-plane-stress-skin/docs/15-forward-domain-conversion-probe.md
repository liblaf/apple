# 固定激活下的表皮域、Lamé 换算与截断边界实验

## 结论

本次经批准的范围只有 cheap forward probe 和静态分析，没有运行新的
inverse。实验固定历史 `e100-p000`、step 40 的 muscle activation，在 5 个
setup 下分别从 zero displacement 和 old historical displacement 求平衡，共
10 次 forward。

主要结论如下：

- 历史 full-boundary + 3D Lamé 能被当前 forward 路径复现，说明数据和实现
  对接正常。
- 5 个 setup 的 zero/old 分支都不稳定；分支差异远大于预设容差，因此本次
  结果不能用来给材料或表皮域作单一、严格的因果排序。
- 换成正确的 plane-stress Lamé 和 `IsFace` membrane，并没有消除 target gap
  或 Bumpy。所有结果仍离 target 很远。
- 将全部 artificial-cut incident points 固定后，face ROI 内有可测但肉眼很小
  的变化，target 与 roughness 指标略差。用户复核后仍明确选择该 hard-fixed
  条件作为对缺失头部组织支撑的保守近似。这个选择是建模决策，不代表它是
  解剖学 ground truth。
- inversion 和 fold 继续作为 warning 报告，不作自动 veto。当前数量很小，
  标准视图中没有明显可见的 inversion/fold artifact。

因此，本次 forward probe 本身不批准 formal inverse。

## 后续状态更新

在本报告完成后，corrected schema-v3 preparation 和 hard-fixed zero-step
forward/adjoint smoke 已完成并通过静态与运行门禁。Smoke 只执行 1 次 evaluation、
0 次 optimizer update；forward/adjoint 均成功，`hard_failures=[]`，6,980 个
cut-incident vertices 的 readback displacement 保持 exact zero，模型共有
33,636 个 fixed vertices / 100,908 个 fixed DoFs。完整证据见
[zero-step smoke report](20-hard-fixed-zero-step-smoke.md)。

在随后获得单独批准后，formal `20` 已完成 40 updates / 41 evaluations，
analyzer `30` 也完成全部 saved-frame 和标准视图分析。Corrected terminal
target error 为 target RMS 的 `51.24%`，但 matched-fidelity Bumpy 明显高于
no-skin。完整结果见
[formal baseline report](30-corrected-baseline-screen.md)。没有自动启动第二个
inverse。

## 实验定义

所有 setup 使用相同的 homogeneous skin 参数：`E=0.2 MPa`、`nu=0.49`、
`thickness=1 mm`、`prestrain=none`。体组织始终使用 3D Lamé 换算，只有
Koiter skin membrane 的换算和作用域发生变化。

3D 与 plane-stress 的区别为：

```text
3D:           lambda = E nu / ((1 + nu) (1 - 2 nu))
plane stress: lambda = E nu / (1 - nu^2)
both:         mu     = E / (2 (1 + nu))
```

在本次参数下，3D `lambda=3.288590604 MPa`，plane-stress
`lambda=0.128964337 MPa`，两者 `mu=0.067114094 MPa`。完整推导和建模审计见
[plane-stress preflight](00-plane-stress-preflight.md)。

<!-- markdownlint-disable MD013 -->

| case | membrane domain | skin conversion | cut boundary |
| --- | --- | --- | --- |
| `full-3d-replay` | 历史完整 extracted boundary，128,172 triangles | 3D | historical `IsFixed` |
| `full-plane-stress` | 同上 | plane stress | historical `IsFixed` |
| `isface-3d` | all-vertex `IsFace`，29,899 triangles，1 component | 3D | historical `IsFixed` |
| `isface-plane-stress` | 同上 | plane stress | historical `IsFixed` |
| `isface-plane-stress-cut-fixed` | 同上 | plane stress | 全部 cut-incident points 固定为零 |

<!-- markdownlint-enable MD013 -->

历史 full boundary 包含 13,165 个由 `InFaceConvex` 截断产生的非表皮
triangles；`IsFace` membrane 不含这些 triangles。截断面涉及 6,980 个
vertices，其中 380 个原本已固定；hard-fixed setup 额外固定 6,600 个，并对
old seed 作一致的零位移投影。zero/old 两个 seed 使用完全相同的历史
activation；本次 `new_inverse_solves=0`。

## 执行与身份

实验在以下目录执行：

```text
exp/2026/08/18/human-face-smile-plane-stress-skin
```

由于 Cherries run snapshot 没有保存原始 shell 命令，下面给出与本次运行一致
的 DEBUG 复现命令；`CHERRIES_NAME` 只影响运行名称，不影响数值结果：

```bash
DEBUG=1 \
CHERRIES_NAME="domain conversion probe" \
uv run python src/15-forward-domain-conversion-probe.py

DEBUG=1 \
CHERRIES_NAME="domain conversion visual analysis" \
uv run python src/16-analyze-domain-conversion-probe.py
```

实际执行脚本与 Cherries `src/` 快照一致：

| script | SHA-256 |
| --- | --- |
| [`15-forward-domain-conversion-probe.py`](../src/15-forward-domain-conversion-probe.py) | `741d3f3db966f8b1e25b389a8734176fb6991a6872e6f8a1a8b875bd3ec5e2f5` |
| [`16-analyze-domain-conversion-probe.py`](../src/16-analyze-domain-conversion-probe.py) | `1553fc066188bd66ab7ea12424f5dfa890fbd7acc7956a6d187f337206cf3d6f` |

15 在约 3 分 57 秒内完成 10 次 forward；16 在约 26 秒内完成静态验证和
两张 contact sheet。DEBUG 模式关闭了 Comet 和 Git commit。

## 产物与验证

15 写出 10 个 `result.vtu` 和 10 个逐 case `forward-summary.json`。16 在生成
报告前逐项检查了固定路径、有限 JSON、aggregate/sidecar 一致性、文件
SHA/size、mesh topology、point/cell 数量、target/mask、10 份完全一致的
activation，以及 hard-fixed mask/value 和零位移条件。

主要产物：

| artifact | SHA-256 |
| --- | --- |
| [15 aggregate JSON](../data/15-forward-domain-conversion-probe-summary.json) | `8d1fb6f7b0a6b4877cb79ea74de804c734c0ed2e040651cfecae49d357ba25e3` |
| [15 table](../data/15-forward-domain-conversion-probe-table.md) | `1915d4be7e8703243578988bfcc12838c0618bbb558060a6af52d74a145470de` |
| [16 analysis JSON](../data/16-forward-domain-conversion-analysis.json) | `c7e7e19456ea2cf29d91771ac377297a93af57cf9d91a4ad5fa8efc596eebdf9` |
| [16 analysis CSV](../data/16-forward-domain-conversion-analysis.csv) | `d21733892f6668b09f8d8eb2168df6ccddb014540853c8bbc6fff121843beabe` |
| [16 compact table](../data/16-forward-domain-conversion-analysis.md) | `6b21d2af25e41afb9ff557a21b05f101950f6d880a7995a14f3d038ed4f52811` |
| [zero-seed contact sheet](../data/16-forward-domain-conversion-zero-views.png) | `09fd025ff3ffc57a0bc588013a5aa373ecd060a128f9c49e9c9448764fbdefeb` |
| [old-seed contact sheet](../data/16-forward-domain-conversion-old-views.png) | `4967425837872f781c7e80eb9c4709c4b64a12ae9c65b5e35b8764c77217b30b` |

两份实验目录日志仍在：
[15 log](../logs/15-forward-domain-conversion-probe.log) 和
[16 log](../logs/16-analyze-domain-conversion-probe.log)。但 Cherries Local
plugin 在输出完成后尝试归档日志时遇到 `FileNotFoundError`，所以两个
`.cherries/runs/...` snapshot 保存了 source 和 data，却没有 `logs/` 副本。
这不影响已验证产物，但意味着 run snapshot 不能单独恢复原始命令或完整日志；
本报告因此明确给出等价 DEBUG 命令和执行脚本 SHA。

## 定量结果

`area error` 使用 target-area-weighted face RMS；`nLap` 是 residual-normal
Laplacian RMS。单位均已换成表中所示。

<!-- markdownlint-disable MD013 -->

| case | seed | error / target | area error, mm | dihedral, deg | nLap, mm | inv. tets | folded face tris |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-3d-replay | zero | 0.727891 | 3.67067 | 7.13801 | 0.243887 | 20 | 0 |
| full-3d-replay | old | 0.602469 | 3.04582 | 8.54440 | 0.227211 | 29 | 0 |
| full-plane-stress | zero | 0.724917 | 3.68299 | 8.86203 | 0.250681 | 21 | 0 |
| full-plane-stress | old | 0.622119 | 3.16264 | 10.15588 | 0.243061 | 22 | 2 |
| isface-3d | zero | 0.704636 | 3.59550 | 7.86174 | 0.230827 | 99 | 0 |
| isface-3d | old | 0.600786 | 3.07116 | 9.46688 | 0.215689 | 28 | 3 |
| isface-plane-stress | zero | 0.709494 | 3.62051 | 9.25293 | 0.238680 | 12 | 1 |
| isface-plane-stress | old | 0.614102 | 3.14896 | 10.70052 | 0.233050 | 23 | 3 |
| isface-plane-stress-cut-fixed | zero | 0.715134 | 3.65536 | 9.70684 | 0.240036 | 22 | 0 |
| isface-plane-stress-cut-fixed | old | 0.614751 | 3.15789 | 11.08296 | 0.234343 | 25 | 4 |

<!-- markdownlint-enable MD013 -->

### 历史 replay

`full-3d-replay/old` 与 pinned 历史 control 的 loss-mask displacement 差异为
`3.97319e-7 m`，即 target RMS 的 `7.48227e-5`；IsFace 差异为
`3.97352e-7 m`，即对应 target RMS 的 `7.48217e-5`。两者均低于 `0.001`
容差，历史 replay gate 通过。

### zero/old 分支敏感性

<!-- markdownlint-disable MD013 -->

| case | displacement delta / target | target-error fraction delta | stable at 0.001 |
| --- | ---: | ---: | --- |
| full-3d-replay | 0.241569 | 0.125423 | no |
| full-plane-stress | 0.190237 | 0.102798 | no |
| isface-3d | 0.221459 | 0.103850 | no |
| isface-plane-stress | 0.249106 | 0.095392 | no |
| isface-plane-stress-cut-fixed | 0.238126 | 0.100383 | no |

<!-- markdownlint-enable MD013 -->

所有 setup 都失败，而且不是阈值附近的小偏差。zero/old 差异达到 target RMS
的 19.0%--24.9%，是容差的 190--249 倍。因此 plane-stress、domain 和
cut-boundary 的数值差异只能作 seed-matched 描述，不能从单一 seed 宣称材料
排序。

### current 与 hard-fixed 截断边界

<!-- markdownlint-disable MD013 -->

| seed | IsFace displacement delta / target | hard-fixed error delta | dihedral delta, deg | nLap delta, mm |
| --- | ---: | ---: | ---: | ---: |
| zero | 0.057467 | +0.005640 | +0.453907 | +0.001356 |
| old | 0.063518 | +0.000649 | +0.382442 | +0.001293 |

<!-- markdownlint-enable MD013 -->

hard-fixed 在 face ROI 内产生约 5.7%--6.4% target RMS 的位移差异，但 target
error 和两个 Bumpy 指标只小幅变差。matched views 中也看不到它带来的明确
改善或明显新增 artifact。需要注意，contact sheet 只渲染 all-vertex
`IsFace` triangles，因此不能直接看到人工截断面本身，只能看到它对面部 ROI
的传递效应。

用户据此决定：未来 corrected setup 将全部 6,980 个 cut-incident points
hard-fix 为零，作为遗漏头部组织支撑的保守近似，并接受上述小幅指标代价。
这不是从实验中识别出的真实边界条件，也不把 hard-fixed 变成 ground truth；
更完整的 volume 或经标定的 Robin support 仍是后续更物理的方案。

## 视觉结果与局限

### Zero seed

![Zero-seed target and five setups](../data/16-forward-domain-conversion-zero-views.png)

### Old seed

![Old-seed target and five setups](../data/16-forward-domain-conversion-old-views.png)

两张图使用相同的 front、30-degree、mouth 和 eye-cheek 视角。共同现象是：

- 嘴角横向展开、张嘴幅度和 cheek lift 都明显不足；
- 唇周、鼻唇沟、脸颊和下眼睑仍有明显 Bumpy；
- old seed 的 target fidelity 普遍更好，但没有消除 Bumpy；
- full/IsFace、3D/plane-stress 和 current/hard-fixed 的组内视觉差别，都明显
  小于 zero/old 分支差别。

全部 10 个结果的 `error/target` 仍为 `0.6008--0.7279`，target-area-weighted
face error 为 `3.046--3.683 mm`。正确的 plane-stress 换算没有解决主要问题；
这也不构成恢复错误 3D 换算的理由。

inverted tetrahedra 为 12--99 个，folded `IsFace` triangles 为 0--4 个。
按照已经确认的判据，这些小数量、肉眼几乎不可见的 inversion/fold 只作
warning，不单独否决结果；target gap 和可见 Bumpy 才是当前高优先级问题。

## 历史结果的状态

本次没有覆盖历史结果。固定 activation 的来源仍保存在：

- [历史 e100-p000 result](../../../17/human-face-smile-material-heuristic-sweep/data/20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen.vtu)，
  SHA-256 `0596f3dcf378f745d80533ac6bd7c0c3f289846e6320e761ef5e10d899e556d5`；
- [历史 e100-p000 summary](../../../17/human-face-smile-material-heuristic-sweep/data/20-human-face-smile-skin-no-prestrain-lr3-material-e100-p000-screen-summary.json)；
- [历史 material-screen 汇总](../../../17/human-face-smile-material-heuristic-sweep/data/30-material-screen-table.md)。

这些旧 skin 结果使用 full extracted boundary、3D Lamé，并在 prestrained case
中使用旧 reference-area convention。它们保持 immutable、可复现、可用于说明
旧机制，但必须标记为
`Historical full-boundary + 3D Lamé (superseded; mechanism-only)`；不能再作为
正确 thin-skin baseline，也不能与 corrected lineage 混写。

## 后续 inverse 状态

Preparation、isolated zero-step smoke、经单独批准的 formal inverse 和 analyzer
均已完成。当前状态为：

1. `20`/`30` 已更新为用户选择的
   `IsFace + plane stress + all-cut-incident hard-fixed` setup；zero-step
   forward/adjoint smoke 已通过，结果隔离在 `tmp/`；
2. corrected `p000` inverse 从 fresh zero 开始并完成 40 updates；target loss
   每步下降，但 dihedral roughness 随 fit 改善持续上升；
3. 不建议直接延长 `p000`。如果讨论后只批准一个新 inverse，应先用 cheap
   forward 比较 corrected fixed-reference `p100/p200`，再选择一个剂量；
4. 当前没有批准或启动第二个材料 case，不使用历史 displacement 或历史
   activation warm-start 冒充 corrected fresh-zero inverse。
