# Corrected hard-fixed baseline inverse

## 结论

经批准的单一 corrected baseline inverse 已完成。此次只运行了一个材料设置：

```text
skin domain: all-vertex IsFace
skin Lamé conversion: plane stress
Koiter measure: fixed original reference area
cut boundary: all 6,980 artificial-cut-incident vertices fixed to zero
skin E: homogeneous 0.2 MPa
skin prestrain: none
activation/displacement: fresh exact zero
optimizer: Adam, LR 0.3, 40 updates / 41 evaluations
```

运行和产物审计均通过，但科学结果不理想：

- target RMS error 从 `5.31014 mm` 降到 `2.72095 mm`，即 target RMS 的
  `51.24%`；corrected model 比历史错误 baseline 更接近 target，但距离仍很大；
- 在近似相同的 target fidelity 下，corrected surface 比 no-skin control
  明显更 Bumpy：contraction dihedral RMS 为 `2.24x`，residual-normal
  Laplacian RMS 为 `1.65x`；
- 标准视图中，口周—鼻唇沟、眼下—外侧脸颊及下颌两侧的起伏肉眼可见；
- 47 个 inverted tets 和 25 个 folded `IsFace` triangles 没有形成可见的
  翻面、自交或穿插，继续只作 warning；
- 40 步内 target loss 仍在下降，但 dihedral roughness 随 fit 改善持续上升。
  因此不建议简单延长当前 `p000` trajectory。

本轮不会自动启动第二个 inverse。若时间只允许再做一次，建议先用 cheap
forward 对 corrected `p100/p200` prestrain 做剂量和分支检查，再只选择一个
通过检查的 prestrain case 运行 inverse。单纯延长 `p000` 或重跑旧 `e005`
都不是当前最高信息量的选择。

## 正式执行与数值审计

Producer `20` 在 `DEBUG=1`、`MPLBACKEND=Agg`、
`PYVISTA_OFF_SCREEN=true` 和 `uv run --frozen` 下执行。DEBUG profile 禁用了
Comet 和 Git commit，因此没有修改现有 dirty worktree 或 `uv.lock`。正式运行
用时约 13 分 35 秒。

执行代码身份：

<!-- markdownlint-disable MD013 -->

| component | SHA-256 |
| --- | --- |
| [`20-inverse-plane-stress-screen.py`](../src/20-inverse-plane-stress-screen.py) | `8c5d75ea06d66e60800d1c83c800d365bef01372340f88a650ef44732ea18f4d` |
| inverse runtime bundle | `3086071201576008047a0b86394e4282c8dc2d37bc0c21a8c8bd4edc73932426` |
| [`30-analyze-plane-stress-screen.py`](../src/30-analyze-plane-stress-screen.py) | `c18f74c165d1616a18a2b362e8ceda834238f04940648fa7d50eff96b06745a2` |

<!-- markdownlint-enable MD013 -->

Formal inverse 的 41 条 trace 精确对应 step `0..40`。全部 forward 和 adjoint
均成功，数值有限，learning rate 始终为 `0.3`，loss 每步下降，best 为
step 40。完整 VTKHDF 逐帧读回还确认：

- 33,636 fixed vertices / 100,908 fixed DoFs；
- 所有 6,980 个 cut vertices 在全部 41 帧中 displacement bitwise zero；
- step 0 activation exact zero；
- final result 与 history best frame 逐数组一致；
- terminal `I + ActivationInv` 没有 non-SPD active tet，最小 eigenvalue
  `0.01678`。

| inverse quantity | step 0 | step 40 |
| --- | ---: | ---: |
| loss, mm2 | 9.39919 | 2.46785 |
| target RMS error, mm | 5.31014 | 2.72095 |
| error / target RMS | 1.00000 | 0.512406 |
| activation RMS | 0 | 0.0609580 |
| activation max abs | 0 | 2.00379 |

Authoritative producer outputs are the post-rewrite canonical copies:

- [aggregate final JSON](../data/20-corrected-baseline-screen-summary-final.json),
  SHA-256 `64a030366053b14eed9ad4da322d910146175fe7bb781e2dca8ee976c03c7045`;
- [case final JSON](../data/20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen-summary-final.json),
  SHA-256 `575ebcbd7152a256917c2a11a9bf9bef9046f00f9831e18adc86d41645be1856`.

Cherries Local plugin 在数值产物全部写完后仍因缺失 snapshot log file 报
`FileNotFoundError`。这不影响 live data、history 或 canonical `*-final.json`；
但 snapshot 内同名 raw summary 是 rewrite 前版本，不能作为 authoritative
metadata。此次没有 Comet 记录，也没有 Git commit。

## Analyzer 与比较定义

Analyzer 逐帧重读 corrected、历史错误 skin baseline 和 no-skin history。
执行前修正了两项只影响 analyzer 的严格检查：

1. 历史完整 extracted boundary 有 43 条 nonmanifold edges，但全部位于
   `IsFace` 以外；实际指标域的 29,899 个 `IsFace` triangles 为一个连通分量，
   有 707 条 boundary edges、0 条 nonmanifold edges，因此 adjacency 和
   manifold gate 改为严格作用在真实 metric domain；
2. analyzer 移除了 producer 从未生成的冗余 sidecar provenance 字段要求，
   改为验证实际存在的 mesh path、skin file identity、size 和三类 content hash。
   Mesh 本身仍由 live size/SHA、manifest 和 aggregate 三重绑定。

两次失败的 analyzer 尝试都发生在读取历史帧前，没有留下正式 `30-*` 输出。
最终版本完成 3 cases x 41 frames 的严格 finite、topology、target、trace、
activation、hard-fixed 和 artifact-identity 验证，并生成以下
[analysis JSON](../data/30-corrected-baseline-analysis.json)，SHA-256
`8834c95305ae0d47e476331a7ad086362294f23830eabc2a7f5f91afd55afe6e`。
生成的 JSON 按预注册保留 `visual_review.status=pending`，避免程序自动批准
后续 inverse；本报告记录其后完成的人工与独立目视复核，不改写生成产物。

Primary matching 只使用 corrected 和 no-skin。历史 full-boundary + 3D-Lamé
skin 只作 secondary diagnostic，不参与 tau。共同 tau 为 corrected terminal
`0.512406`；每个 case 从真实保存帧中选与 tau 最近的一帧，不插值：

<!-- markdownlint-disable MD013 -->

| case | checkpoint | step | error / target | area error / target | dihedral, deg | residual-normal Lap, mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| corrected | terminal/matched | 40 | 0.512406 | 0.517323 | 13.3290 | 0.217061 |
| historical old skin | terminal | 40 | 0.602461 | 0.592527 | 8.54438 | 0.227208 |
| no-skin | matched | 11 | 0.520904 | 0.553316 | 5.94089 | 0.131610 |
| no-skin | terminal | 40 | 0.239161 | 0.252179 | 9.44490 | 0.155229 |

<!-- markdownlint-enable MD013 -->

Matched fidelity spread 为 `0.0084975`。它足以作本轮强烈诊断，但不是精确相等
的连续匹配；而且 no-skin 保留历史 boundary，所以这也不是纯 skin-only
causal ablation。

在 matched checkpoint，corrected 相对 no-skin：

- target RMS error 小 `1.63%`，target-area-weighted error 小 `6.50%`；
- contraction target-relative dihedral 大 `124.36%`；
- residual-normal Laplacian 大 `64.93%`；
- displacement Laplacian 为 `0.17840 mm` vs `0.14092 mm`。

因此 corrected skin 在近似相同 target fit 下并没有起到 smoothing 作用。

## Trajectory 与视觉检查

![Corrected and controls trajectory](../data/30-corrected-baseline-trajectories.png)

Corrected trajectory 的关键变化为：

| step | error / target | dihedral, deg | residual-normal Lap, mm |
| ---: | ---: | ---: | ---: |
| 0 | 1.00000 | 2.89090 | 0.242838 |
| 10 | 0.773998 | 6.15593 | 0.207131 |
| 20 | 0.650801 | 9.63885 | 0.211639 |
| 30 | 0.567309 | 11.8169 | 0.214750 |
| 40 | 0.512406 | 13.3290 | 0.217061 |

Residual-normal Laplacian 在 step 8 达到最低 `0.206829 mm`，随后重新上升；
dihedral roughness 则随着 target fit 改善持续大幅增加。这说明当前 objective
正在用越来越强的局部折皱换取 target error，更多 `p000` steps 不能同时解决
两个主要问题。

### Terminal states

![Terminal target and controls](../data/30-corrected-baseline-terminal-views.png)

### Nearest discrete matched-fidelity states

![Matched target and controls](../data/30-corrected-baseline-matched-views.png)

两次独立目视复核一致认为：

- corrected 的嘴角横向展开、唇形、鼻唇沟和 cheek lift 仍明显不足；
- 口周—鼻唇沟、眼下—外侧脸颊和下颌两侧有肉眼可见的 broad Bumpy；
- matched no-skin step 11 明显比 corrected step 40 平滑；
- 47 inversions / 25 folds 没有形成可见的翻面、自交或穿插；
- 没看到明确的 cut-seam 开裂或位移跳变。30-degree 轮廓边缘较锯齿，且外侧
  皱褶延伸到固定边界，可能有 hard-fixed support 的影响，但目前不能把它认定
  为主要 artifact。

## 历史错误结果继续保留

旧实验没有被覆盖或重命名。它们仍用于组会说明“在旧模型下观察到的机制”，
但所有旧 skin 结果必须标记为：

```text
Historical full-boundary + 3D Lamé + activation-dependent area weight
(superseded; mechanism-only)
```

它们同时使用错误的 full extracted boundary、3D Lamé skin conversion，并在
prestrained case 中让 `ActivationInv` 改变能量积分权重。相关图和产物继续保留在：

- [August 17 material screen](../../../17/human-face-smile-material-heuristic-sweep/data/30-material-screen-table.md)；
- [August 18 exaggerated material report](../../human-face-smile-exaggerated-material-screen/docs/10-exaggerated-material-screen.md)。

历史 no-skin mechanics 不受 skin constitutive correction 影响，但它的 boundary
仍与本次 corrected hard-fixed model 不同。

## 下一步讨论建议

不建议直接做：

- 更长的 corrected `p000`：现有 trajectory 已显示 fit 和 Bumpy 的冲突在扩大；
- 旧 `e005` softening-only：它只在很小面积达到低 E，不能代表真正的大面积
  skin softening；
- 恢复 3D Lamé：旧换算物理上仍不适合 thin membrane。

如果只剩一次 inverse，最值得验证的是 corrected prestrain，而不是继续优化
同一个机制。为避免再次把 expensive inverse 当作 preflight，建议先做以下
cheap forward gate：

1. 保持 `IsFace + plane stress + hard-fixed cut + E=0.2 MPa` 不变，只准备
   `p100` 和 `p200` 两个 fixed-reference prestrain 候选；
2. 每个候选先做 zero-muscle-activation static equilibrium，检查 prestress
   是否造成明显 distortion、solver failure 或非有限值；
3. 再固定本次 `p000` terminal muscle activation，从 zero 和本次 terminal
   displacement 两个 seed 各求一次 equilibrium，比较 target、Bumpy 和 branch；
4. 只有当某个剂量在两个 seed 下都显示一致的 Bumpy 改善且没有明显破面时，
   才批准该剂量的一个 fresh-zero 40-step inverse；最终仍按 matched target
   fidelity 比较。

这最多是 6 次 cheap forward，可以在花费下一次 inverse 预算前区分
`p100` 与更激进的 `p200`。目前没有批准或启动这一步。
