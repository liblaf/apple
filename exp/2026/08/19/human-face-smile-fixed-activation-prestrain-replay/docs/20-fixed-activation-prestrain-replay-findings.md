# c020 固定激活预应变 replay：结果与结论

## 结论

`c020` 对平滑本身有效，而且效果已经足够明显：两条 `alpha=1` 分支的两个主
bumpiness 指标都改善，target fit 只下降约 2.12%；固定视角复核中，front、
30 degree、mouth、eye-cheek 四个视角都能看出平滑，也没有发现新增的明显
artifact。

但是，这次 replay **没有通过 continuation/direct 路径稳定性硬门**。两条
`alpha=1` 分支在 `SmileLossMask` 和 `IsFace` 上的位移差都是约 `0.0983 mm`，
即对应 target RMS 的 `1.8513%`；预注册上限是 `0.1%`，超出约 `18.5` 倍。
因此当前结论是：**c020 的平滑强度够，但不能作为稳定的 inverse physics
证据；不跑 c050，也不启动 inverse，先查清路径敏感性。**

## 实验合同

这是固定 muscle activation 的 forward-only 因果 replay，共 6 次 forward；
没有 inverse、adjoint、backward 或 activation update。每个 case 都使用同一份
corrected p000 step-40 activation，数组 SHA-256 为
`4494f1eca2ce6f14c2e87a184d2227c080fbfa4594e7d6e96ced0c0c35c981de`。

皮肤保持 homogeneous `E=0.2 MPa`、`nu=0.49`、厚度 `1 mm`，仅作用于
all-vertex-`IsFace` 三角形，并使用 plane-stress Lamé conversion 和固定原始
`RestArea`。全部 6,980 个截面 incident vertices 固定为零位移。预应变为：

```text
rho_full  = 0.98^2 * clip(R, 0.5, 1)
rho_alpha = rho_full^alpha
```

所以 `c020` 是长度收紧 2%，uniform natural-area ratio 为 `0.9604`；raw
target/rest area ratio `R` 的 floor 是 `0.5`。floor 实际只截断 31/29,899 个
三角形，占 corrected-skin rest area 的 `0.0879366%`。

主分支按 `alpha=0, 0.25, 0.5, 0.75, 1` continuation，每一步从前一个平衡态
warm start；direct 分支在 `alpha=1` 时直接从 exact baseline displacement
开始。6 次 forward 都成功，固定点位移保持 bit-exact zero。

## 定量结果

这里 `D` 是 contraction-target-relative dihedral RMS，`L` 是 residual-normal
Laplacian RMS，`fit` 是 `SmileLossMask` target-error RMS。下表指标均由 analyzer
从保存的 result mesh 重新计算，而不是直接信任 producer scalar fields。

| case | D deg | D change | L mm | L change | fit mm | fit change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 13.329 | — | 0.2171 | — | 2.7209 | — |
| continuation | 6.387 | -52.08% | 0.1804 | -16.90% | 2.7788 | +2.126% |
| direct | 6.477 | -51.41% | 0.1801 | -17.02% | 2.7787 | +2.121% |

两条分支都通过预注册的 smoothing effect-size 和 fit 门：`D` 约减半，`L`
下降约 17%，fit 恶化低于允许的 5%。continuation 中 `D`、`L` 随 alpha 总体
持续下降，说明响应不是只出现在终点。

`alpha=0` replay 本身是稳定的：相对 exact baseline，`SmileLossMask` 和
`IsFace` 位移 RMS 差分别为 `0.0009250 mm` 和 `0.0009251 mm`，仅为对应 target
RMS 的 `0.01742%`，低于 `0.1%` 硬门；两项 roughness ratio 也都在 1% 容差内。
所以最终 hard-stop 不是 alpha-0 复现失败，而是 full-prestrain 的分支敏感性：

| alpha-1 branch delta | measured | gate | result |
| --- | ---: | ---: | --- |
| `SmileLossMask` RMS | 0.098308 mm / 1.85133% target RMS | 0.1% | fail |
| `IsFace` RMS | 0.098318 mm / 1.85133% target RMS | 0.1% | fail |

两条分支的终点 scalar metrics 很接近，但位移场并未落在预注册的同一分支
容差内；不能用 scalar 相近来绕过这个硬门。

质量计数没有出现明显恶化：inverted tets 从 47 变为 46，folded skin
triangles 从 25 变为 16/18。另一方面，minimum `det(F)` 从 `-1.214` 降到
`-1.283/-1.291`，所以 analyzer 的极值 warning 仍需保留。结合固定视角复核，
这些小区域没有形成新增的肉眼明显 artifact，但这也不能覆盖路径稳定性失败。

## 固定视角复核

Analyzer JSON 在自动生成时仍把 visual review 记为 `pending`。随后对 matched
geometry 和 shared-scale normal-residual 图板做了独立复核：4/4 视角都能看到
baseline 的局部皱褶被压低，continuation/direct 都没有出现新的明显 artifact。
因此视觉 checklist 已通过；最终结论仍由 branch hard gate 否决。

![终点几何固定视角](../data/20-fixed-activation-prestrain-terminal-geometry.png)

![终点法向残差固定视角](../data/20-fixed-activation-prestrain-terminal-normal-residual.png)

![alpha 轨迹](../data/20-fixed-activation-prestrain-alpha-trajectories.png)

![质量轨迹](../data/20-fixed-activation-prestrain-quality-trajectories.png)

## 运行与溯源

工作目录：

```text
/home/liblaf/Projects/liblaf/apple/exp/2026/08/19/human-face-smile-fixed-activation-prestrain-replay
```

实际命令：

```bash
DEBUG=1 \
CHERRIES_NAME='Fixed-activation c020 prestrain replay' \
CHERRIES_TAGS='human-face,skin,prestrain,fixed-activation,'\
'forward,replay,c020,debug' \
uv run --frozen python src/10-fixed-activation-prestrain-replay.py

DEBUG=1 \
MPLBACKEND=Agg \
PYVISTA_OFF_SCREEN=true \
CHERRIES_NAME='Fixed-activation c020 prestrain replay analysis' \
CHERRIES_TAGS='human-face,skin,prestrain,fixed-activation,'\
'forward,replay,c020,analysis,visual,debug' \
uv run --frozen python src/20-analyze-fixed-activation-prestrain-replay.py
```

本地 Cherries snapshots：

```text
.cherries/runs/2026/08/19/human-face-smile-fixed-activation-prestrain-replay/10-fixed-activation-prestrain-replay/2026-08-19T074817-Fixed-activation-c020-prestrain-replay
.cherries/runs/2026/08/19/human-face-smile-fixed-activation-prestrain-replay/20-analyze-fixed-activation-prestrain-replay/2026-08-19T075606-Fixed-activation-c020-prestrain-replay-analysis
```

Snapshot 中的 producer/analyzer source 和关键 JSON 与 live files byte-identical；
producer 还在全部 solve 结束后确认所有输入和 runtime dependencies 的 size/SHA
均未变化。计划文档中的 “has not been executed” 是审批前状态，当前结果以 live
artifacts、[producer log](../logs/10-fixed-activation-prestrain-replay.log) 和
[analyzer log](../logs/20-analyze-fixed-activation-prestrain-replay.log) 为准。

溯源限制：producer Cherries snapshot 并不自包含。它没有复制实际使用的
corrected baseline result/summary/target/skin 和 reviewed runtime sources；dynamic
import 还注册了三份外来的 08/17 默认 result/summary/target assets。因此不能只靠
snapshot assets 重跑。数值结果仍有效，是因为 producer 在运行前后都锁定并重算
了这些 live inputs/runtime files 的 size 和 SHA，且复查全部通过。

关键 SHA-256：

```text
231668ac7963bd7eff14705a94125ad8396d8886e0d34a152a4f51253f32c7f6  src/10-fixed-activation-prestrain-replay.py
9cc505aea0551f5e60e4c1f3947e1deae9c059a3875f725f2afd02ecf05ccdb2  src/20-analyze-fixed-activation-prestrain-replay.py
b75f2ad1b6298621ad34f9f202d2d56e31517caafbe33c69412e9f7651b44c83  data/10-fixed-activation-prestrain-replay-summary.json
8b16d81a20df966b64803c4dfcaf49797fced98130e6a73c5828157234eb9287  data/20-fixed-activation-prestrain-replay-analysis.json
968fd56b048ecaa3e09ecc8f40e1ff7166ff91e7299927d01cfa4bb23597ff9e  data/20-fixed-activation-prestrain-replay-trajectories.csv
887e98c102ae411cf25ae6253022dde48c6fc8058d255add320cfb7a1e4f2d19  data/20-fixed-activation-prestrain-alpha-trajectories.png
1c997e4592ac3756d691dc029334920a27e3700465bfcf89cee31cdf23177aa6  data/20-fixed-activation-prestrain-quality-trajectories.png
627cc0b5b9db3675215c3290e6a955c8c71ee511fb54d64d8884e670d4d7e052  data/20-fixed-activation-prestrain-terminal-geometry.png
94ae4c2f45e9ea0341a80883bb1b10d16fec47e84c14878ae7edc05d46bfebf5  data/20-fixed-activation-prestrain-terminal-normal-residual.png
7b7721182f81a61e3648bf5f3fae4619746fa07df10c4a3cc6f3c697a18b41fe  logs/10-fixed-activation-prestrain-replay.log
055930b77d409a916fdf701b0047e04068ffd4e94b01daabab721a009785fb06  logs/20-analyze-fixed-activation-prestrain-replay.log
```

Producer log 在 Cherries shutdown 时有两条旧 `15-forward-domain-conversion-probe`
asset 路径不存在的 logging warning；它们不属于本实验的输出合同。6 个 case
artifact、aggregate 和 analyzer outputs 均存在且通过 readback。此次没有生成
或运行 c050，也没有运行任何 inverse physics。
