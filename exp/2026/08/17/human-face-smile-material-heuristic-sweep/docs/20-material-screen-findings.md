# Smile Skin Material Heuristic Screen

## 结论

这轮实验没有找到可以直接进入 200-step Stage B 的材料参数，但识别出了当前 inverse setup
的物理可行窗口：
**在现有纯 L2 inverse、learning rate `0.3` 和每个 muscle tet 独立 6-DoF unconstrained
activation 下，target loss 总体下降且 terminal state 最优，但所有 setup 都会很快离开
物理可行域。**

- 六个表皮材料候选和 `no-skin` control 都完成了 `41` 次 forward/adjoint evaluation；
  所有求解均成功，且没有 learning-rate backtracking。数值求解成功不代表形变物理有效。
- 除 `e025-p100` 外，各案只在 step `0..2/3/4` 保持从零开始连续物理有效；第一个无效帧
  出现在 step `3..5`，原因是 tet inversion 和/或 skin triangle fold。
- `e025-p100`（扩张区 softening `0.25`、收缩区 full prestrain）在 formal step 0、muscle
  activation 仍严格为零时就有 `2` 个 inverted tets 和 `2` 个 folded skin triangles。
  同一 solver-content hash 的早期 smoke 曾得到 zero fold/inversion；两次位移场只差
  `2.11e-6 m` RMS、`2.44e-4 m` max，便跨过了局部翻转边界。这说明该材料组合的静态
  平衡态缺乏鲁棒性，而 formal failure 不是 inverse activation update 造成的。
- `no-skin` 的 terminal target error 最低；在 step 0 有效的 setup 中，它最早在 step 3
  失效，step 40 已有 `69`
  个 inverted tets、`349` 个 surface folds 和 `98` 个 non-SPD muscle activations。
  这说明去掉表皮约束能提高 L2 拟合能力，也会暴露更大的非物理自由度；由于这些状态
  已经无效，这一段本身不能量化 Bumpy 改善或恶化。
- 因为 `e025-p100` 没有任何 admissible frame，七案和六个 material-only 子集都无法建立
  physical matched-fidelity comparison；正式 Pareto front 和 Stage B promotion
  list 均为空。

因此，现在不应把任一 terminal/best-loss 状态称为材料改进，也不应直接增加 inverse 步数。
下一阶段需要先解决两个问题：缩小材料参数到静态可行范围，以及在保留 6-DoF 表达能力的同时，
约束 activation inverse 不要在几步内制造局部翻转。

## 1. Setup

实验固定同一个 prepared human-face mesh、`Smile` target 和 `SmileLossMask`，只改变表皮材料：

- expanding triangles：降低 Young's modulus；
- contracting triangles：施加 isotropic in-plane prestrain；
- control：完全关闭 skin energy（`no-skin`）。

材料网格为：

| axis | values |
| --- | --- |
| expanding-region minimum Young's-modulus scale | `1.0`, `0.25` |
| contracting-region prestrain gain | `0.0`, `0.5`, `1.0` |

baseline skin 为 `E = 0.20 MPa, nu = 0.49`。材料场由 target/rest triangle area ratio
构造：signed log-area、`1%` soft deadband、rest-area-weighted q99 cap、`5 mm` finite-volume
implicit heat diffusion。eligible face patch 共 `29,899` 个 triangles，其中
contraction ROI `13,129` 个、expansion ROI `16,770` 个。

每个候选均重新构造 forward model、activation parameter 和 Adam optimizer，从严格全零
activation 独立开始；不同 setup 之间没有 warm start 或 activation transfer。优化变量保持项目定义：
`288,235` 个 active muscle tets，每个 tet 的 symmetric inverse activation tensor 有 `6`
个独立分量，共 `1,729,410` 个 unconstrained DoFs。

screen 使用纯 L2 target loss、固定 learning rate `0.3`、`40` 个 optimizer updates
（step `0..40` 共 `41` 次 evaluation）。没有 activation magnitude/smoothness loss、没有
residual-Laplacian loss，也没有 inversion/fold barrier。

## 2. Fixed-budget inverse result

七案的 best loss 都出现在 terminal step 40。下表只说明优化器确实降低了 target error，
不能作为材料排序，因为这些 terminal state 全部不满足物理门禁。

<!-- markdownlint-disable MD013 -->

| candidate | best RMS error | error / target | activation RMS | inverted tets | skin folds | non-SPD muscle tets | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e100-p000` | 3.1992 mm | 0.60246 | 0.05787 | 29 | 49 | 0 | no |
| `e100-p050` | 3.1136 mm | 0.58635 | 0.05444 | 26 | 58 | 0 | no |
| `e100-p100` | 3.0056 mm | 0.56601 | 0.05219 | 30 | 97 | 0 | no |
| `e025-p000` | 3.0911 mm | 0.58212 | 0.05904 | 38 | 55 | 2 | no |
| `e025-p050` | 2.9929 mm | 0.56361 | 0.05589 | 29 | 69 | 8 | no |
| `e025-p100` | 2.9268 mm | 0.55117 | 0.05376 | 33 | 80 | 10 | no |
| `no-skin` | 1.2700 mm | 0.23916 | 0.06718 | 69 | 349 | 98 | no |

<!-- markdownlint-enable MD013 -->

表面上看，增加 prestrain、降低局部 Young's modulus、关闭 skin energy 都能进一步降低
terminal target error。但与此同时，inversion、fold 和 non-SPD activation 增多；这些 lower-loss
state 已经通过非物理自由度“拟合”target。不能把该趋势解释成 target reconstruction upper bound。

## 3. Full temporal physical-prefix analysis

后处理重新读取七个 VTKHDF history 的全部 `7 x 41 = 287` 帧，并逐帧核对
`inverse_step`、loss、error、mesh correspondence、target、activation 和 solver trace。
物理门禁为：

- zero inverted tets，`det(F) > 0` 且 q0.1% `>= 0.2`；
- zero skin normal reversals，skin area-ratio q0.1% `>= 0.1`、q99.9% `<= 10`；
- 每个 muscle tet 的 `I + ActivationInv` 为 SPD，minimum eigenvalue `>= 1e-6`。

只有从 step 0 到当前帧的全部历史帧都通过，当前帧才属于 physical prefix。这样可以防止
trajectory 在越过 inversion/fold 后又偶然回到某个局部数值门槛内。

<!-- markdownlint-disable MD013 -->

| candidate | last valid step | last valid error / target | first invalid step | first invalid error / target | first failure |
| --- | ---: | ---: | ---: | ---: | --- |
| `e100-p000` | 4 | 0.91832 | 5 | 0.89811 | 3 inversions, 3 folds |
| `e100-p050` | 3 | 0.88705 | 4 | 0.86738 | 1 inversion, 1 fold |
| `e100-p100` | 3 | **0.84830** | 4 | 0.82913 | 1 fold |
| `e025-p000` | 4 | 0.91443 | 5 | 0.89339 | 3 inversions, 2 folds |
| `e025-p050` | 3 | 0.88248 | 4 | 0.86183 | 1 fold |
| `e025-p100` | none | none | **0** | 0.90398 | 2 inversions, 2 folds |
| `no-skin` | 2 | 0.87622 | 3 | 0.82075 | 2 inversions, 2 folds |

<!-- markdownlint-enable MD013 -->

在有 admissible frame 的材料案中，`e100-p100@3` 达到最小的 valid-prefix error fraction
`0.84830`，但它只比下一步的首次 fold 早一个 optimizer update。这个点可以作为后续更安全
优化器的参考 checkpoint，不能称为 matched-fidelity winner。

![Physical validity trajectories](../data/30-material-screen-trajectories.png)

### Why matched fidelity is unavailable

正式 matching 先要求每案有非空 physical prefix，再在各自 admissible trajectory 上定义共同
target-fidelity threshold。`e025-p100` 在 step 0 已无效，因此这个前提不成立：

- primary seven-case matching：unavailable；
- six-case material-only sensitivity：unavailable；
- matched-checkpoint Pareto front：empty；
- direct Stage B long-run candidates：empty。

分析器仍生成了一组忽略物理有效性的 closest-fidelity fallback，只为排查数值趋势。其共同
阈值约 `0.60246`，但 selected fidelity spread 为 `0.02557`，也远高于预设的 `0.001`
gate。图和表均明确标为 `DIAGNOSTIC ONLY`。

![Diagnostic-only material screen](../data/30-material-screen-matched.png)

在这些**无效状态**中，full prestrain 的 contraction target-relative dihedral RMS 比 no-prestrain
低，且 `no-skin` target fit 最强。这只能用来提出后续假设，不能作为“prestrain 消除了 Bumpy”
的证据，因为所比较的 mesh 已经有 folds/inversions，fidelity 也没有真正匹配。

| diagnostic candidate family | p000 dihedral RMS | p100 dihedral RMS |
| --- | ---: | ---: |
| `e100` | 0.14913 rad | 0.08202 rad |
| `e025` | 0.14624 rad | 0.08012 rad |

diagnostic first-crossing Pareto front 只有 `e025-p100`，但它没有 physical prefix，而且七案
fidelity spread 为 `0.025571`。因此这一 front 与上述 2 x 3 趋势均不可推广、不可 promotion；
正式 `matched_checkpoint_pareto.front` 仍为空，factor effects 的解释级别为
`diagnostic_only`。

## 4. What the experiment established

### Supported

1. 面积启发式可以生成有限、连续、可追溯的逐 triangle Young's modulus/prestrain 场；
   材料准备和 solver-consumed fields 的严格 hash/readback 门禁均通过。
2. 在本次 unconstrained、进入无效域的 trajectory 上，观察到 loss/validity 的关联：
   `no-skin` 的 L2 error 降得最快，但在 step 0 有效的 setup 中最早失效。这不是一个已经
   验证的、物理可接受的 trade-off curve。
3. 当前 unconstrained 6-DoF activation + pure L2 + LR `0.3` 的实际可用优化窗口只有约
   `2..4` steps。当前证据不足以批准 200-step；没有 barrier/backtracking 时，优化会在
   physical prefix 之后继续进行，但这次实验没有证明更长路径绝无恢复可能。
4. softening 与 full prestrain 存在危险交互：单独 `e100-p100` 和 `e025-p050` 在 formal
   step 0 可行，而 combined `e025-p100` 对细小 solver/位移差异已经敏感到会发生局部翻转。

### Not supported

1. 不能声称某个材料候选在 matched target fidelity 下更平滑；没有合法的共同 threshold。
2. 不能用 terminal loss、terminal bumpiness 或 diagnostic fallback 选 winner。
3. 不能把 `no-skin` 的 `1.27 mm` RMS error 当成 inverse upper bound，因为该状态包含大量
   inversions、folds 和 non-SPD activation。
4. 不能直接把当前 screen 的 activation 迁移到其他材料 setup，或 warm-start 后续 long run。

## 5. Next experiment

### Priority A: static material admissibility

先在 zero muscle activation 下扫描较细的 prestrain/softening 组合，只做 forward equilibrium
和完整物理门禁。固定同一 5 mm 场和同一 mesh，参数网格为：

- Young's-modulus scale：`0.25`, `0.50`, `1.0`；
- prestrain gain：`0.50`, `0.75`, `1.0`。

根进程顺序启动两个独立 worker 进程。R0 按 Young scale、prestrain 从小到大运行全部
`9` 案，R1 使用完全相反的顺序，共 `18` 次 static forward。这样同时检查进程重启复现性、
nonlinear equilibrium branch sensitivity 和 case-order/隐藏状态效应。`p=0` 不重复 forward，
只把现有 hash-bound `e100-p000`/`e025-p000` 作为基线锚点。

每个参数点只有在两次都 forward success/finite、通过全部物理门禁、pass/fail 分类一致，
并同时满足下面两个 replicate gate 时才算 robust admissible：

- `|fidelity_R0 - fidelity_R1| <= 0.001`；
- `RMS(u_R0 - u_R1) / target_displacement_RMS <= 0.01`。

报告采用两个 replicate 中较差的物理指标，不用平均值掩盖局部 failure。任何一次出现
zero-activation inversion/fold 或两次分类冲突，都直接判为 branch-unstable/ineligible；
额外 single-case 诊断不能把冲突候选“救回”。

预注册选择规则先要求 high endpoint `E=1.0` 的 `p=.5/.75/1` 整行 robust pass；
否则 A1 直接失败。通过后，若 `E=0.25` 整行也 pass，则 `safe_low=0.25`；否则若
`E=0.50` 整行 pass，则 `safe_low=0.50`；否则 A1 失败，不进入动态 inverse，必要时再补
`E=0.75` 一整行。

### Priority B: keep 6-DoF, regularize the inverse

在不改变“每个 muscle tet 求 6 个 unconstrained activation 分量”这一项目定义的前提下，
先对 `e100-p000`、`e100-p100`、A 找到的最高 prestrain 静态安全候选和 `no-skin`
做短程 pilot：

1. 先跑固定 `lr=0.10, steps=20`；若 physical prefix 或 matched spread 仍失败，再跑
   `lr=0.03, steps=60`。更密的状态必须来自更小真实更新，不能插值 mesh；
2. 从 fresh zero 独立运行，比较各案在 physical prefix 内的可达 fidelity；
3. material-only 的每案都必须有真实 threshold crossing bracket，selected fidelity spread
   `<= 0.001`，相对 baseline error `<= 1.05`；`no-skin` 只作 diagnostic control；
4. 只有选定共同 LR 后，才运行最终 `{safe_low,1.0} x {0,.5,1}` material rectangle；
5. 如果降 LR 仍在共同 threshold 前越界，再加 activation magnitude/spatial regularization
   和几何 barrier，且用同协议 unregularized run 作对照。

降低 learning rate 可能推迟或改变越界路径，但不提供物理保证；它应与
regularization/barrier 一起验证。把 activation 改成 SPD 参数化可以作为单独的后续
variant，但不应悄悄替换当前 6-DoF unconstrained 研究问题。

### Stage B decision

当前 **不运行 200-step Stage B**。formal promotion rule 是 material candidate 同时属于
matched-checkpoint Pareto front 且 source-final scientific eligible；`no-skin` 只作
control，不参与 material promotion。这次两者交集为空。下一次 long run 的进入条件是：

- zero-activation material equilibrium 通过所有物理门禁；
- short inverse 的完整 selected prefix 通过门禁；
- 至少两个 setup 能在 `<= 0.001` spread 内达到共同 target fidelity；
- matched state 的 roughness 改善不是由更差的局部 fidelity 或幅值衰减造成。

## Validation and reproducibility

- candidate preparation artifact commit：`c14048e86b6d195525142d8b2d487c1f40595452`
- 7-case 40-step screen artifact commit：`d432fb0d8ba247f62254e9e1b9345b19cbb009c3`
- 287-frame analysis artifact commit：`a2dd55ce6c09bce15929e90f72135da05ce9b652`
- prepare Comet run：[aaa2723505fa44cd9e839845fe51a66a](https://www.comet.com/liblaf/apple/aaa2723505fa44cd9e839845fe51a66a)
- screen Comet run：[2cb3cff85c0c413a89e5ce1257604696](https://www.comet.com/liblaf/apple/2cb3cff85c0c413a89e5ce1257604696)
- analysis Comet run：[a4ce4a7d349c4089865b78ba35beca02](https://www.comet.com/liblaf/apple/a4ce4a7d349c4089865b78ba35beca02)

formal screen 的 `7 x 41` 条 forward/adjoint trace 全部成功，固定 LR 和 fresh-zero contract
通过；分析器逐帧读取 `287` 个 temporal states 并验证 history/trace/result correspondence。
analysis 在写完五个完整输出后，因 `e025-p100: no admissible frame` 按设计返回非零；这是
scientific gate 的失败结果，不是脚本崩溃。

正式入口命令为：

```bash
python3 src/20-inverse-material-sweep.py \
  --candidate-set all-with-no-skin \
  --stage screen \
  --inverse-max-steps 40 \
  --mandatory-baseline-steps 40
python3 src/30-analyze-material-screen.py
```

screen 约用时 `2:11:58`，analysis 约用时 `6:34`。

正式机器可读产物：

- [`../data/10-material-candidates-manifest.json`](../data/10-material-candidates-manifest.json)
- [`../data/20-material-screen-summary.json`](../data/20-material-screen-summary.json)
- [`../data/30-material-screen-analysis.json`](../data/30-material-screen-analysis.json)
- [`../data/30-material-screen-analysis.csv`](../data/30-material-screen-analysis.csv)
- [`../data/30-material-screen-table.md`](../data/30-material-screen-table.md)
- [`../data/30-material-screen-trajectories.png`](../data/30-material-screen-trajectories.png)
- [`../data/30-material-screen-matched.png`](../data/30-material-screen-matched.png)
- [`../logs/30-analyze-material-screen.log`](../logs/30-analyze-material-screen.log)

Cherries Local snapshot 中的 per-case summaries 是外层 final rewrite 前的旧副本；本报告和
后处理均以 Git HEAD 中的 live `data/20-material-screen-summary.json` 及其 sibling artifacts
为 canonical source。aggregate/table、Git LFS objects 和 Comet run 均已验证；这一归档问题
不改变数值结论，但后续 entrypoint 应避免在 final normalization 之前登记 per-case summary。
