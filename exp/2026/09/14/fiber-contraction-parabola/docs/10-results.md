# 修正 Stable Neo-Hookean 后的二维抛物线 L-BFGS 诊断对照

**优化器更正：本报告的四条轨迹使用 L-BFGS，不是 Adam。** 我不应在原有 Adam 实验中替换优化器。现已保留相同修正物理、几何与边界，重新完成纯 Adam 对照；当前组会请使用 [Adam 结果、loss 曲线与停止原因](./70-adam-results.md)。以下数据作为历史 L-BFGS 诊断保留，不能当作 Adam 的表现。

补充：已绘制 loss 曲线，并通过与原轨迹完全一致的重放记录所有线搜索试算。自由激活有被回溯拒绝的 loss 上升步骤，但所有接受状态的 loss 单调下降。见 [loss 与 overshoot 检查](40-loss-overshoot.md)。

日期：2026-09-14。状态：**主库修正已验证，四组诊断运行已完成；四组逆优化均未达到梯度收敛标准，两组自由激活终点均存在负曲率方向。** 本报告比较的是同一求解协议下保存的优化路径及停止状态，不是模型的全局最优拟合能力，也不是四组稳定物理平衡的展示。

## 1. 主库修正

[StableNeoHookeanActive](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py) 保留激活后的范数项，但两个 determinant 项均改为实际形变的 \(J=\det F\)：

$$
B=A^{-1},\qquad
W(F,B)=\frac{\mu}{2}(\|FB\|_F^2-3)-\mu(J-1)+\frac{\lambda}{2}(J-1)^2.
$$

$$
P=\mu FBB^T+[-\mu+\lambda(J-1)]\operatorname{cof}F.
$$

同步修正了一阶导数、Hessian 对角线、Hessian-vector product 和二次型。范数项使用激活后的形函数梯度，体积项使用原始梯度与物理 F。另修正了该材料的 CPU/CUDA 分配与 kernel 启动设备不一致问题，并用直接 kernel 保证材料参数反向传播有效。公共调用接口保持兼容，**有激活时的材料行为有意改变**；旧实验不能仅凭相同类名视为使用了新公式。

[材料回归测试](../../../../../../tests/warp/test_stable_neo_hookean_active.py) 验证了能量公式、恒等激活与被动材料一致、Hessian 有限差分、非零 activation/lambda 梯度和混合导数，以及三维嵌入下的平面应变能量。与 Koiter、FiberSpring、静态求解相关测试一起运行，**29 项通过**；CUDA 材料与能量反向传播 smoke 也通过。Ruff 和 `git diff --check` 通过。没有提交或推送。

## 2. 固定的实验设置

| 项目 | 设置 |
| --- | --- |
| 历史几何与目标 | 复用 [9 月 2 日二维脚本](../../../02/pork-shared-release-study/src/10-run-pork-2d.py) 的网格、边界和 top-node L2 目标函数 |
| 域与离散 | \([0,1]\times[0,0.1]\)，100×10 网格，2,000 个 P1 三角形 |
| 肌肉 | \(y\in[0.04,0.06]\)，400 个三角形，水平纤维 |
| 材料 | 肌肉 E=0.03 MPa，脂肪 E=0.003 MPa，ν=0.49；无皮肤、外力、接触或正则项 |
| 边界 | 底边、左右两侧 x/y 位移均固定；内部及顶边其余节点自由 |
| 目标 | 顶边位移 \(u_T(x)=(0,4hx(1-x))\)；h=0.05 为主实验，h=0.20 为大目标补充 |
| 初值 | 每组独立从 B=I、u=0 开始 |
| 目标函数 | 自由顶边节点的位移向量平方误差均值；优化时除以 h²，表中报告未归一化 RMS |

历史坐标没有经本次核验的物理长度标定，因此位移单位记为 **L**，不写成 mm。能量为单位厚度下的 MPa·L²，节点残差为 MPa·L。

两组唯一的材料控制空间差异是：

$$
\text{x 收缩：}\quad B_e=\begin{pmatrix}1+a_e&0\\0&1\end{pmatrix},\ a_e\ge0;
\qquad
\text{自由组：}\quad B_e=I+\begin{pmatrix}q_{xx,e}&q_{xy,e}\\q_{xy,e}&q_{yy,e}\end{pmatrix}.
$$

x 收缩组每个肌肉单元 1 个变量，共 400 个；自由组 3 个，共 1,200 个。前者的控制矩阵满足 \(A_{xx}=1/(1+a)\le1\)、\(A_{yy}=1\)，禁止主动横向伸长与剪切。**这不是对实际 F 的运动学约束，也不意味着 F=A 是修正能量的无应力状态。** 自由组没有正定、可逆、定向或幅度限制，是原始对称张量的代数对照。两组均不设收缩幅度上限。

Apple 当前的该 FEM 材料面向四面体；二维计算使用 [physics2d.py](../src/physics2d.py) 的独立三角形装配。它由 \(F_3=\mathrm{diag}(F_2,1)\)、\(B_3=\mathrm{diag}(B_2,1)\) 精确降维，范数中的常数变为 2；并未调用历史脚本的旧能量或旧前向求解器。二维能量导数、HVP、隐式梯度分别通过中心差分，最大相对误差为 \(1.99\times10^{-8}\)、\(4.60\times10^{-11}\)、\(3.86\times10^{-9}\)。这是离散实现验证，不是解剖验证。

两组采用相同的 Newton 前向求解、隐式伴随和投影 L-BFGS/Armijo 外层搜索：前向残差阈值 1e-10，逆问题投影梯度阈值 1e-7，最多 300 个接受步骤、1,500 次前向试算，单次控制分量变化上限 0.25。每次试算从最后一个接受的平衡状态开始。前向不收敛的试算明确记录并回溯，不进入动画。

共同的诊断停止条件是：接受状态的最小 J≤1e-6，或控制更新无穷范数≤1e-7。它们表示网格退化或分辨率停滞，**不表示收敛**。内层线搜索另要求试算 J>1e-8。这些是数值规则，不是经标定的物理容许范围；有限 Stable Neo-Hookean 体积罚本身也不是防翻转屏障。

## 3. 主实验：h=0.05

初始拟合 RMS 为 0.03669879 L。

| 最后接受状态 | x 方向收缩 | 自由对称 B |
| --- | ---: | ---: |
| 接受步骤 / 前向试算总数 | 122 / 141 | 81 / 392 |
| 拟合 RMS / L | 0.03305382 | 0.03376733 |
| 相对初始 RMS 降低 | 9.93% | 7.99% |
| 顶边最大向上位移 / L | 0.01553591 | 0.01239040 |
| 顶边最小竖直位移 / L | −0.02822557 | −0.02483562 |
| 顶边平均竖直位移 / L | 0.00251711 | 0.00071956 |
| 实际截面积比 | 1.019463 | 1.002470 |
| 面积加权 RMS(J−1) | 0.220320 | 0.051743 |
| 最小 / 最大 J | 7.48e-7 / 6.4721 | 0.32328 / 1.65257 |
| 最小 / 最大奇异值 σ(B) | 1 / 28.8006 | 1.92e-4 / 4.4036 |
| 肌肉单元 det(B)≤0 的比例 | 0% | 79% |
| 最小代数特征值 eig(B) | 1 | −0.80449 |
| 逆投影梯度无穷范数 | 1.77e-3 | 8.00e-2 |
| 停止原因 | 网格退化诊断 | 控制步长低于分辨率下限 |

![最终形变：浅色脂肪、红色肌肉，虚线为目标](../data/20-figures/deformation-h050.png)

![顶边的竖直与水平位移](../data/20-figures/top-profile-h050.png)

观察到的 x 收缩响应是：中央和局部区域向上隆起，靠近两端出现下陷，肌肉周围有很强的剪切与局部拉伸。方向约束确实限制了控制变量，却没有消除空间变化的幅度、不均匀形变或网格退化。最终最大 Bxx≈28.8，对应最小控制 Axx≈0.0347；不能把这种极端控制解释为合理的肌肉幅度。

自由组的额外自由度没有恢复完整抛物线。它使大量 B 改变定向，且出现接近零的奇异值。\(\sigma_{\min}(B)\ll1\) 对应 A 的巨大伸长，并不是“更强的肌肉收缩”。所有肌肉单元均有 \(Z=BB^T-I\) 的负模态，即存在 σ(B)<1 的方向；这与“B 有负特征值”是两个不同的统计。

**最终 RMS 的微小排序不能证明 x 收缩表达能力更好。** 两组停止步数不同且都未收敛；共同的第 80 个接受步骤上，自由组 RMS=0.03376734，x 收缩组为 0.03420178，自由组更低。自由对称 B 本来就包含 x 收缩子空间；本次只能比较这两条优化路径。

[并排优化动画](../data/20-figures/evolution-h050.mp4) 使用真实保存的接受状态，无平滑或插值。它是逆优化迭代，不是物理时间动画；短序列结束后保持最终状态并明确标注。

## 4. 大目标 h=0.20：作为优化诊断补充

| 最后接受状态 | x 方向收缩 | 自由对称 B |
| --- | ---: | ---: |
| 接受步骤 | 300 | 43 |
| 拟合 RMS / L | 0.14670257 | 0.13457628 |
| 顶边最大向上位移 / L | 0.00043068 | 0.10522054 |
| 顶边水平误差 RMS / L | 0.00047793 | 0.07275120 |
| 实际截面积比 | 1.000019 | 0.974440 |
| 最小 J | 0.998968 | 1.10e-7 |
| 最小 σ(B) | 1 | 5.46e-4 |
| 肌肉单元 det(B)≤0 比例 | 0% | 65.75% |
| 逆投影梯度无穷范数 | 1.35e-4 | 6.46e-4 |
| 停止原因 | 300 步预算用尽 | 网格退化诊断 |

![大目标最终形变](../data/20-figures/deformation-h200.png)

x 收缩组几乎停留在初始形状，但梯度仍高于阈值，不能解释为模型无法产生运动。作为反例，把 h=0.05 的 x 收缩序列第 50 步原样放到 h=0.20 目标下重新计算误差，RMS 已为 0.14388094，优于这里的 300 步终点。该补充组暴露了初值附近的搜索尺度/曲率问题，**不提供 x 收缩能力的上界**。

自由组出现中央高隆起、两侧下陷和明显水平位移。更大的局部峰值并不等于更好的抛物线边界匹配，且其最终网格已接近压扁。[大目标动画](../data/20-figures/evolution-h200.mp4) 保留了这一过程。

## 5. 面积约束为何重要

完整目标边界包围的连续面积为

$$
|\Omega_T|=\int_0^1[0.1+4hx(1-x)]\,dx=0.1+\frac{2h}{3},\qquad
|\Omega_T|/|\Omega_0|=1+\frac{20h}{3}.
$$

因此 h=0.05 和 h=0.20 分别要求约 **33.3% 和 133.3%** 的截面积增加。100 等分的离散顶边目标面积比分别为 1.333300 和 2.333200。精确匹配该完整边界与处处 J=1 不能同时成立。

这说明目标与物理体积罚存在明确冲突，**不是有限 λ 下的不可达证明**。另外，仅有较小的全局面积变化并不代表局部保体积：主实验 x 收缩组总面积只增加 1.95%，局部 J 却从接近 0 到 6.47。最终自由组还有水平位移，因此不能只积分参考 x 上的 uy 来估计其面积。

从力学上，x 收缩控制对应

$$
Z=BB^T-I=(2a+a^2)e_xe_x^T,\quad
W-W_{\rm passive}=\frac{\mu}{2}\operatorname{tr}(FZF^T).
$$

它只增加沿纤维方向的主动项；竖直鼓起或下陷来自平衡、组织间兼容性、边界反力和 J 项的耦合。不能把“激活只沿 x”误解成“位移只能沿 x”。

## 6. 最终状态的独立检查

独立重装配结果和最终网格见 [endpoint-checks.json](../data/30-endpoint-checks-final/endpoint-checks.json)。检查分别记录前向驻点、逆问题梯度、最小代数 Hessian 特征值、固定边界误差，以及由边界多边形和 \(\sum_e |T_e|J_e\) 得到的面积。前向残差通过不等于力学稳定，也不等于逆优化收敛。

| h / 控制 | 前向残差 ∞ 范数 | 自由位移 Hessian 最小代数特征值 | 二阶检查 |
| --- | ---: | ---: | --- |
| 0.05 / x 收缩 | 1.32e-15 | +9.45e-5 | 本离散状态具有正曲率 |
| 0.05 / 自由 B | 9.33e-11 | −7.32e-7 | 存在负曲率，非稳定极小点 |
| 0.20 / x 收缩 | 1.75e-12 | +3.84e-5 | 本离散状态具有正曲率 |
| 0.20 / 自由 B | 5.07e-13 | −1.74e-3 | 存在负曲率，非稳定极小点 |

这是完整自由位移 Hessian 的最小代数特征值，不是主运行中“最靠近零的特征值”。后者在 h=0.20 自由组为正，却遗漏了更负的模态，不能用于稳定性认证。稠密 LAPACK 最小特征对的归一化残差不超过 1.46e-16，矩阵相对对称误差不超过 5.70e-17。

所有终点的固定边界误差为零，两种面积计算差异不超过 8.33e-17。所有保存终点 J>0；但 h=0.05 x 收缩和 h=0.20 自由组各有一个三角形 J<0.1，已接近零面积。**x 方向限制没有保证几何质量；自由组的残差门槛也没有保证稳定平衡。** 后者应作为本次发现的前向分支/停止判据问题报告，不能当作可靠的物理拟合结果。

ParaView 可直接打开 [四个最终 VTU 所在目录](../data/30-endpoint-checks-final)。文件包含真实变形网格、位移、肌肉标记、J 和控制分量。

## 7. 下次组会可直接使用的内容

1. **公式修正和验证**：展示 J 从 det(FB) 改为 det(F)，给出能量、导数测试和实际导入版本。
2. **同一个抛物线的并排动画**：解释 x 收缩仍能产生 y 方向运动，但方向约束不足以阻止局部退化；同时展示自由 B 的谱与定向问题。
3. **误差必须和停止原因一起读**：主目标第 80 步及最终状态分别比较；展示 det(F)、σ(B)、Hessian，而不是宣布哪个控制空间获胜。
4. **下一项实验**：先补齐前向的负曲率检查与稳定分支处理，再用温和、已知的 x 收缩场前向生成目标并从相同初值反求；检查能否恢复运动，分离目标与体积约束的冲突、搜索尺度问题和控制空间限制。之后再单独比较幅度上限/空间正则及网格分辨率。这些是后续计划，本次未执行。

本次没有验证网格收敛，也没有加入接触、皮肤、解剖纤维或生理幅度标定。结论限于该平面应变条带和明确给定的边界条件。

## 8. 复现和运行证据

工作目录：`exp/2026/09/14/fiber-contraction-parabola`。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Fixed-volume active strain x-fiber versus free parabola' \
CHERRIES_TAGS='2d,active-strain,fixed-fiber,parabola,physical-volume,degeneration-diagnostic' \
uv run python src/10-run-inverse.py --output 10-comparison-final
```

输出目录已有结果，脚本会拒绝覆盖；重跑时使用新的 `--output`。完整数据为 [10-comparison-final](../data/10-comparison-final)，每组包含 `history.npz`、`checkpoint.npz`、`trace.csv`、`summary.json`，失败前向试算另记 `rejected-trials.jsonl`。源码快照和 SHA-256 在 [protocol.json](../data/10-comparison-final/protocol.json)；后处理脚本的视觉排版修改不改变这些逆物理数据。

正常 Cherries/Comet 运行已退出，进程退出码 0；它表示运行和记录完成，不表示四组优化收敛。[Comet 运行](https://www.comet.com/liblaf/apple/cdf28fcd8369450bb2e7553f8018c75d) 的完整 `Comet.ml Experiment Summary` 见 [运行日志](../logs/10-final-launch.log)。关键字段摘录：

```text
Name                : Fixed-volume active strain x-fiber versus free parabola
cherries/entrypoint : exp/2026/09/14/fiber-contraction-parabola/src/10-run-inverse.py
cherries/exp_dir    : exp/2026/09/14/fiber-contraction-parabola
cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
max_iterations     : 300
max_evaluations    : 1500
forward_tolerance : 1e-10
gradient_tolerance: 1e-7
```

工作树有本次和此前未提交修改；Git SHA 不是全部运行源码的替代，复现以快照/hash 为准。该运行使用 NumPy/SciPy CPU 双精度与单线程 BLAS，导数检查随机种子为 20260914。Cherries profile 保留 Comet/本地记录，明确关闭自动 Git commit。Comet 曾限流重试；本地数据和最终退出已检查。

开发过程中的 `00-smoke`、`01-guarded-smoke` 仅是小网格验证；`10-comparison` 是外层试算导致前向失败的中断运行，`10-comparison-guarded` 是发现极小 J 下反复回溯后人工中止的诊断运行。它们和日志均保留，不混入正式四组数据。导入历史网格脚本会登记一个未使用的 `10-pork-2d` 输出名，从而在日志中产生缺失资产提示；本次实际输出路径为 `10-comparison-final`。

回归命令（仓库根目录）：

```bash
uv run pytest -q --randomly-seed=3698340657 \
  tests/warp/test_stable_neo_hookean_active.py \
  tests/warp/test_koiter.py tests/warp/test_fiber_spring.py \
  tests/forward/test_fiber_spring.py tests/forward/test_static_simulation.py
```

渲染与独立检查分别由 [20-render-comparison.py](../src/20-render-comparison.py) 和 [30-verify-endpoints.py](../src/30-verify-endpoints.py) 生成。图像清单、数据来源和显示约定见 [manifest.json](../data/20-figures/manifest.json)。

独立检查命令使用相同的单线程与 Comet 环境变量，实际运行 `uv run python src/30-verify-endpoints.py`，当前默认输出为 `30-endpoint-checks-final`，退出码 0。其 [Comet 运行](https://www.comet.com/liblaf/apple/c5a9e3635ad0476f9ea6257a6ad90934) 和完整 [启动日志](../logs/30-endpoint-launch.log) 已保存。第一次 ARPACK 最小代数特征值求解未收敛（19,801 次迭代，0/1 个特征向量）；最终改用约 1,980 阶矩阵的稠密 `scipy.linalg.eigh(subset_by_index=[0,0], driver='evr')`，没有改动逆物理数据。
