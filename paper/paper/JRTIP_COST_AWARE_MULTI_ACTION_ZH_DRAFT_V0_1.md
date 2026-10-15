<!--
初稿状态：V0.1，2026-07-15。
本文档是独立重构稿，不继承 V0.6 的 CAFRU 主张。
方括号中的“待补”内容必须在最终隐藏测试、硬件闭环和引用复核完成后替换。
“开发性快照”只用于确定叙事，不得直接作为投稿数字。
方法简称暂不固定，避免在算法实体和跨数据集证据稳定前过早命名。
-->

# 面向端云协同目标检测的成本感知多动作路由

**Cost-Aware Multi-Action Routing for Edge-Cloud Object Detection**

作者姓名待补

作者单位待补

通讯作者：姓名与电子邮箱待补

## 摘要

资源受限摄像头通常只能运行轻量目标检测器，而持续上传完整图像又会消耗上行带宽、增加云端调用和网络往返时延。现有选择性卸载方法多依据端侧置信度或检测统计量判断是否上传整图，但端侧“不确定”并不等价于云端“能够纠错”，单一整图动作也不能利用局部目标区域的传输优势。本文研究端侧输出、感兴趣区域复核和整图复核构成的多动作端云目标检测问题，并提出一种低开销成本感知路由框架。该框架首先使用端侧检测类别的历史条件收益构造可解释排序先验，然后复用端侧检测器同一次前向传播中的多尺度特征图，通过全局池化、低维投影和预算边界排序损失学习语义残差；在此基础上，独立的动作头估计感兴趣区域相对整图的条件收益差，并联合上传字节、云端计算和网络时延选择端侧输出、局部复核或整图复核。为提高局部动作的可用性，云端检测器采用整图与动态上下文裁剪混合数据训练，局部云端框被映射回原图，并与区域外端侧结果融合。本文进一步给出置信度路由充分性的多动作解释：只有当各动作的条件净收益能够由端侧置信度单调表示时，单一置信度阈值才可能达到最优；当云端纠错能力、类别互补性或局部传输代价具有输入相关性时，学习路由才存在可利用空间。实验将在 COCO、ENA24 和 Oregon Critters 上采用互斥的检测器训练、路由器训练、校准和最终测试划分，并同时报告 Official AP、Deploy AP、上传比例、上传字节、均摊计算量、选择器开销和不同网络条件下的端到端时延。在最终冻结协议下，本文方法相对随机上传和强置信度基线的精度-带宽-时延改进分别为【待补】，结果同时表明学习路由的收益取决于端云模型互补性以及局部云模型对裁剪输入的适应能力。

**关键词：** 端云协同；目标检测；选择性卸载；多动作路由；感兴趣区域；成本感知推理；模型级联

## 1 引言

目标检测是野外监测、安防巡检、无人机感知和智能交通中的基础视觉任务。现代检测器在标准服务器上能够取得较高精度，但其参数规模、内存访问和算子开销限制了在低功耗摄像头上的直接部署。轻量模型能够降低本地延迟和能耗，却通常在小目标、遮挡、复杂背景和分布偏移条件下产生更多漏检或定位错误。边缘智能因此不应只在“全部本地”和“全部上云”之间二选一，而应根据输入内容和当前资源条件决定何时调用更强模型 [1-5]。

模型切分、早退和云边协同推理已经形成较完整的研究脉络。Neurosurgeon 根据移动端和云端开销选择网络切分位置 [6]，BranchyNet 对简单样本提前退出 [7]，相关综述总结了特征传输、早退和动态网络条件下的主要折中 [8]。然而，目标检测的端云协同具有两个不同于普通分类级联的特点。第一，一张图像包含数量不定的目标，端侧低置信并不能说明云端一定能修复当前错误；第二，传输动作不只有整图，局部目标区域可能大幅降低字节数，但裁剪也可能丢失上下文，且端侧完全漏检时不存在可用裁剪中心。

已有检测卸载工作主要研究“哪些图像应交给强模型”。Adaptive Feeding 使用端侧候选框特征在快速和准确检测器之间选择 [9]；DCSB 使用低置信候选、目标数量和目标尺度识别困难图像 [10]；ORIC/MORIC 进一步构造与检测 mAP 上下文相关的单图收益，并从端侧检测输出估计该收益 [11]；OffloaD 将检测失败与网络状态结合，在物理测试床中调度轻量和重量检测器 [12]；实际移动端卸载研究也表明，网络条件和设备能力会改变最优执行位置 [13]。这些方法证明了检测特定路由的必要性，但其主要动作仍是整图本地或整图远端。

另一方面，EdgeDuet 通过高分辨率 tile 增强小目标检测 [20]，AccMPEG 学习宏块编码质量对视觉任务精度的影响 [21]，Selective Cloud Offloading 使用不确定区域和 ROI packing 降低云调用 [22]，PAROD 研究室外场景中的并行 ROI 卸载 [23]，边界框感知协同与单次训练的可配置检测器分别探索特征压缩和带宽自适应 [24,25]。这些工作说明“上传什么”与“是否上传”同等重要，但尚未统一回答以下问题：在强置信度基线下，如何同时估计整图和 ROI 的条件纠错收益，并以真实上传字节、云调用、计算量和时延选择动作？

这一问题不能通过不断增大路由器复杂度自动解决。EdgeBoost 在分类任务上表明，经过校准的简单置信度规则可以接近学习型路由器 [14]。Jitkrittum 等从理论上指出，当下游模型的残余错误近似输入无关时，置信度延迟可能已经接近最优；学习方法更容易在下游模型具有专长、标签噪声或分布偏移时获得优势 [15]。Learning to Cascade 进一步说明，面向级联的置信度应描述强模型能否纠正弱模型，而不只是弱模型本身是否正确 [16]。此外，检测置信度受框尺度和位置条件偏差影响 [18]，而不同类别上的端云互补性与 FrugalML 所揭示的服务互补性具有相似结构 [19]。因此，本文不把“使用学习器”本身作为贡献，而是把可学习空间限定为置信度无法表达的条件纠错收益和动作成本。

本文提出一种面向 Edge、ROI 和 Full 三种动作的成本感知多动作路由框架。其核心思想是将“选择哪些图像”和“选择何种上传动作”解耦：帧排序以类别条件收益为安全锚点，并仅用同一次端侧前向中的低维语义残差修正排序；动作头则学习 ROI 相对 Full 的收益差，并显式奖励局部动作节省的 payload。绝对在线阈值只在独立校准集上确定，最终测试图像逐张决策，不依赖预先知道的测试集 Top-K。

本文的主要贡献如下：

1. 提出检测专用的成本感知多动作问题，将端侧输出、ROI 复核和整图复核置于统一条件净收益框架中，并给出置信度路由何时充分、何时必然丢失动作信息的判定条件。

2. 设计低开销的两阶段路由器。类别条件收益提供稳定、可解释的排序先验，端侧 FPN 的 32 维语义残差只修正预算边界附近的排序；独立动作头预测 ROI 相对 Full 的收益差，从而避免用一个分数同时承担帧选择与动作选择。

3. 构建裁剪感知云端复核和原图坐标融合流程。云端模型使用整图与动态上下文裁剪混合训练，ROI 结果映射回原图后保留区域外端侧框，并通过安全门控避免在局部收益证据不足时破坏整图基线。

4. 建立区分 Official 与 Deploy、Fixed-budget 与 Online、实测与估算时延的完整评测协议，在通用检测集和野外相机数据上同时报告 AP、payload、GFLOPs、选择器开销、随机置信区间、Oracle gap 和网络敏感性，并公开学习路由不占优势的条件边界。

## 2 相关工作

### 2.1 端侧检测与协同推理

COCO 推动了通用目标检测在多类别、多目标和复杂上下文条件下的统一评估 [1]。实时检测器和端到端检测器持续改善精度与推理效率 [2,3]，但边缘部署仍受到内存、量化、数据搬运和硬件算子支持限制 [4]。边缘智能研究因此将模型执行分布到终端、边缘服务器和云端 [5]。现有协同推理大致分为网络切分、早退和任务级路由三类 [6-8]。网络切分需要端云共享模型结构并传输中间特征；本文则保持端侧和云端检测器独立，只传输标准图像或裁剪图像，便于替换检测器并避免中间特征尺寸和隐私接口的耦合。

### 2.2 检测级整图路由

Adaptive Feeding 使用 top-K proposal 的类别、置信度和几何特征训练线性 SVM，在快、慢检测器之间选择 [9]。DCSB 将云端比端侧检测到更多目标定义为困难样本，并使用目标数量、小目标和低置信 proposal 构造轻量判别规则；其 VOC、COCO 子集和 HELMET 实验说明整图选择可以显著减少上传，但其 COCO 设置和指标不能直接替代官方 80 类 COCO val 口径 [10]。ORIC/MORIC 是与本文最接近的学习型整图基线：ORIC 使用代表集近似单图被替换时对全局 mAP 的影响，MORIC 再将收益映射为分位排序并强调高收益样本 [11]。OffloaD 将 introspective、golden 和 latency 分数组合为检测失败指标，并强调更昂贵的云端检测器并不总能提高结果 [12]。上述工作主要优化调用图像数量，而本文进一步区分 ROI 和 Full 的不同传输成本与条件收益。

EdgeBoost 面向图像分类，通过温度缩放和 softmax margin 阈值进行整图选择性卸载 [14]。其结果不能与检测 mAP 直接数值比较，但它构成重要的方法学基线：如果校准后的单一不确定度已经能够排序下游收益，额外路由器可能没有足够收益覆盖自身训练和部署成本。本文因此同时保留 raw confidence、calibrated confidence、fixed-budget confidence 和 online confidence，而不是有意弱化规则基线。

### 2.3 ROI、Tile 与语义编码

EdgeDuet 在端侧处理低分辨率图像，并将包含小目标的高分辨率 tile 并行发送至边缘服务器 [20]。AccMPEG 从编码角度学习每个宏块对下游视觉精度的影响，在不降低精度的条件下减少端到端延迟 [21]。Selective Cloud Offloading 使用 conformal uncertainty 选择检测框，并将多个 ROI packing 为一次云调用 [22]；PAROD 则结合背景建模、区域划分和多服务器并行处理高分辨率室外图像 [23]。Bounding-Box-Aware Collaboration 根据端侧框变化控制卸载并压缩中间特征 [24]，Single-Training Collaborative Object Detectors 通过一次训练支持不同带宽和计算配置 [25]。

RegionFilter 使用区域感知的视频过滤机制降低受限边缘节点上的传输和处理开销 [32]。它与本文都关注区域价值，但 RegionFilter 的核心是视频区域过滤，本文则比较 ROI 与 Full 云端复核的条件净收益。

上述工作共同说明局部传输必须处理三个问题：端侧漏检没有 ROI 来源；多个 ROI 的总面积和调用次数可能超过整图；裁剪会改变目标尺度和上下文分布。本文采用动态单区域、整图回退和 crop-aware 云模型，并把多 ROI packing 与 tile 并行作为后续扩展，而不把任意 ROI 上传天然视为低成本或高收益。

### 2.4 置信度校准、级联与预算路由

现代神经网络置信度通常存在失准，温度缩放可以在不改变分类预测的情况下改善概率校准 [17]。但面向级联的目标与普通 Expected Calibration Error 不同：Learning to Cascade 直接优化快模型和慢模型组合的精度-成本关系 [16]。Jitkrittum 等给出了两模型级联的 Bayes 最优延迟规则，并通过 specialist、噪声和分布偏移实验说明 confidence-only 的失效条件 [15]。目标检测还存在框尺度和位置相关的条件置信偏差，简单全局校准并不保证提高 AP [18]。FrugalML 则从预算约束服务选择角度表明，路由收益来源于模型在类别和输入上的互补性，而不是模型价格或平均精度本身 [19]。

本文将这些认识落实为三项设计：使用类别条件云端收益而不是类别频率作为先验；将语义分支限定为强先验的残差；在校准增益不足时回退先验或置信度策略。该设计不承诺在所有 IID 数据上超过 confidence-only，而是试图在检测器互补且动作成本异质的场景中改善 AP-KB-E2E Pareto。

### 2.5 动态视频分析与结果新鲜度

静态图像路由不能覆盖连续视频中的帧冗余、网络抖动和迟到结果。AWStream 根据精度-带宽剖析选择视频配置 [31]；Reducto 使用低层帧差和动态阈值过滤无价值帧 [26]；NoScope 在固定摄像头查询中组合差分检测器和专用模型 [27]；ApproxDet 根据内容和资源争用选择检测分支 [28]；SmartDet 将网络状态、目标运动和跟踪纳入动态卸载控制 [29]。REACT 采用异步云支持并融合不同时间的端云检测结果 [30]，VATE 进一步结合跟踪和多种端云部署模式 [33]，近期工作还联合优化丢帧和检测任务卸载 [34]。

这些系统表明，学习路由在动态上下文中可能比静态 IID 图像更有价值。但本文的主要算法评估仍以可复现的逐图检测为中心，视频队列和 stale-aware fusion 作为系统层扩展单独报告，避免把静态重放误称为真实视频 mAP。

**表 1 代表性工作的任务、动作空间与本文边界。**

| 工作 | 任务/数据 | 决策信号 | 动作空间 | 是否学习路由 | 与本文的主要差异 |
| --- | --- | --- | --- | --- | --- |
| Adaptive Feeding [9] | VOC、COCO、Caltech 检测 | Top-K proposal 特征 | Fast / Accurate 整图检测 | 线性 SVM | 不建模传输 payload 和 ROI 动作 |
| DCSB [10,36] | VOC、COCO、HELMET 检测 | 数量、小目标、低置信 proposal | Edge / Full | 规则判别器 | 困难度不等同于云端条件收益 |
| ORIC/MORIC [11] | COCO、VOC 检测 | 端侧框与 mAP-context reward | Edge / Full | 轻量收益回归 | 无 ROI/Full 动作比较和实际 codec 成本 |
| OffloaD [12] | 物理端云检测测试床 | 失败指标与网络状态 | 设备/边缘执行频率 | 混合估计与调度 | 面向检测失败和频率调度，不做多粒度图像动作 |
| EdgeBoost [14] | CIFAR-100、ImageNet、Cars 分类 | 校准 softmax margin | Edge / Full | 否 | 分类任务，无框、ROI 和检测 AP |
| EdgeDuet [20] | 无人机小目标检测 | 小目标 tile | Local / Tile | 系统优化 | 关注 tile 并行，不学习 ROI 与 Full 条件净收益 |
| AccMPEG [21] | 视频检测/分割 | 宏块 accuracy gradient | 分区编码质量 | 轻量预测 | 优化编码而非云调用与动作选择 |
| Selective Cloud Offloading [22] | 资源受限目标检测 | Conformal uncertainty | Edge / Packed ROI | 不确定性规则 | 风险控制和 ROI packing，不含学习型 Full/ROI 相对收益 |
| PAROD [23] | 室外高分辨率检测 | 背景与区域结构 | 并行 ROI | 系统规则 | 多边缘并行，不以强 Confidence 基线为主要比较 |
| 本文 | COCO、ENA24、Oregon 检测 | 类别收益先验、FPN 语义残差、动作成本 | Edge / ROI / Full | 低维排序与轻量动作头 | 联合评估 Official/Deploy AP、KB、GFLOPs 和 E2E，并允许安全回退 |

## 3 问题定义与理论动机

### 3.1 多动作端云检测

给定输入图像 \(x_i\) 和标注 \(y_i\)，端侧检测器 \(D_e\) 输出检测集合

\[
B_i^E=D_e(x_i).
\]

云端整图动作使用检测器 \(D_f\)：

\[
B_i^F=D_f(x_i).
\]

ROI 动作首先根据端侧候选生成区域 \(R_i\)，由裁剪感知云端检测器 \(D_r\) 处理，再将预测映射至原图并与区域外端侧预测融合：

\[
B_i^R=\operatorname{Fuse}\left(B_i^E,
\operatorname{MapBack}(D_r(R_i))\right).
\]

动作集合为 \(\mathcal A=\{E,R,F\}\)，分别表示仅端侧、ROI 复核和整图复核。若端侧没有合法候选或 ROI payload 超过约束，\(R\) 动作不可用。

### 3.2 条件收益与系统成本

记 \(q(B,y)\) 为可分解的检测质量代理，动作 \(a\) 相对端侧的条件收益为

\[
\Delta q_{i,a}=q(B_i^a,y_i)-q(B_i^E,y_i),\qquad
\Delta q_{i,E}=0.
\]

系统成本包含上传字节 \(b_{i,a}\)、云端计算 \(g_{i,a}\) 和预计时延 \(\ell_{i,a}(n)\)，其中 \(n\) 表示上行带宽、RTT、队列和 deadline 等网络状态：

\[
C_{i,a}(n)=\lambda_b\frac{b_{i,a}}{b_0}
+\lambda_g\frac{g_{i,a}}{g_0}
+\lambda_\ell\frac{\ell_{i,a}(n)}{\ell_0}.
\]

理想可部署策略根据端侧可观测量 \(z_i\) 选择条件净收益最大的动作：

\[
\pi^*(z_i,n)=
\arg\max_{a\in\mathcal A_i}
\left[
\mathbb E(\Delta q_{i,a}\mid z_i)-C_{i,a}(n)
\right].
\]

这一形式将传统 confidence-only 视为受限特例：其可观测量只有单一端侧不确定度，动作通常只有 \(E\) 和 \(F\)，成本也常被简化为固定云调用次数。

### 3.3 置信度规则的充分条件

设 \(z\) 表示端侧可观测信息，\(c=c(z)\) 为其中的单一置信度标量，\(U_a(z)\) 为动作 \(a\) 的条件期望净收益。允许使用完整 \(z\) 的最优策略价值和只允许使用 \(c\) 的最优策略价值分别为

\[
V_z=\mathbb E_z\left[\max_a U_a(z)\right],\qquad
V_c=\mathbb E_c\left[\max_a\mathbb E[U_a(z)\mid c]\right].
\]

**命题 1（置信度信息约束遗憾）。** 对任意动作集合，均有

\[
\mathcal R_c^*=V_z-V_c\geq 0.
\]

等号成立要求：对于几乎处处的置信度取值 \(c\)，条件分布 \(z\mid c\) 中存在一个共同的最优动作；若同一置信度条件下不同输入需要不同最优动作且具有正概率，则
\(\mathcal R_c^*>0\)。

**证明。** 对每个 \(c\)，最大值函数的凸性给出

\[
\mathbb E[\max_a U_a(z)\mid c]
\geq
\max_a\mathbb E[U_a(z)\mid c].
\]

再对 \(c\) 取期望即可得到 \(V_z\geq V_c\)。当条件分布中的最优动作不能由同一个动作同时实现时，上式严格不等；反之，若存在共同最优动作，则两侧相等。证毕。

更直观地，对任意仅依赖 \(c(x)\) 的确定性策略 \(\pi_c\)，若存在两个输入 \(x_1,x_2\) 满足

\[
c(x_1)=c(x_2),\qquad
\arg\max_a U_a(x_1)\ne\arg\max_a U_a(x_2),
\]

则 \(\pi_c\) 不可能在两个输入上同时达到 Bayes 最优。进一步地，普通单阈值 Full 上传策略不仅要求 \(c\) 对动作充分，还要求 \(U_F(x)-U_E(x)\) 的符号能够由 \(c(x)\) 单调确定；多动作策略还要求 \(U_R(x)-U_F(x)\) 的符号也能由同一标量确定。

该结论是级联延迟理论在本文动作空间中的直接推论 [15]。它说明学习器存在价值的必要前提不是“端侧会犯错”，而是云端纠错收益或 ROI/Full 动作差异包含置信度之外的可预测结构。类别、目标尺度、场景语义和 payload 均可能形成这种结构；若这些结构不存在，安全策略应回退到 confidence-only，而不是强行使用复杂路由器。

### 3.4 检测代理与全局 AP 的边界

COCO AP 由整个数据集上的排序和匹配共同决定，不是逐图可加函数。因此，\(\Delta q_{i,a}\) 只能作为训练和动作分析的代理，不能被称为单图 AP。本文最终结论均由冻结动作产生的完整预测 JSON 通过原生 YOLO.val 和 COCOeval 计算；逐图 gain Oracle 只表示给定预测缓存和代理质量下的 GT-aware 参考，不是全局 mAP 的数学上界。为补充这一不足，实验同时报告严格 Top-K gain Oracle 和计算成本更高的 global-mAP greedy reference，并明确区分二者。

## 4 成本感知多动作路由框架

### 4.1 总体结构

本文框架由端侧检测器、帧收益排序器、ROI/Full 动作头、裁剪感知云端模型、成本门控和结果融合六部分组成。端侧 YOLO26n 或 YOLO26s 完成一次前向后同时产生检测框与 P3/P4/P5 特征图。检测框用于置信度基线、类别收益先验和 ROI 候选构造；特征图仅进行全局平均池化，不增加第二次视觉前向。帧排序器回答“这张图是否值得远端复核”，动作头回答“如果复核，应选择 ROI 还是 Full”。最后，独立校准阈值将连续排序分数转换为逐图在线动作。

**[图 1 待绘制：端侧单次前向、ClassGainPrior、32D semantic residual、ROI/Full action head、cost gate、云端复核与融合的完整数据流。]**

### 4.2 低开销检测统计与动态 ROI

端侧检测统计包括检测数量、框覆盖率、最大/均值/最小置信度、置信度标准差、空检与稀疏指示、平均/最大框面积、贴边框比例、低置信框比例和小目标比例。ROI 候选另外记录候选置信度、相对面积、贴边与小目标指示、ROI payload、Full payload 以及二者比值。

ROI 候选从满足最低置信度和面积范围的端侧框中选择。候选评分优先保留高置信主体，同时对小目标和贴边目标给予轻量补偿。以候选框中心为基准，扩展比例由源框面积的分段函数 \(\gamma(r)\) 决定：

\[
R_i=\operatorname{clip}\left(
\operatorname{expand}(b_i,\gamma(r_i)),x_i
\right).
\]

小目标需要更大上下文，大目标使用较小扩展以控制成本。最终分段阈值和扩展倍率只允许在 detector/selector development 数据上选择，具体数值在表【待补】中报告。若裁剪面积占比过高，或没有合法候选，动作头只能选择 Full。

### 4.3 裁剪感知云端模型

直接把只见过整图的云端模型用于 ROI 会产生输入分布偏移：目标在裁剪后占比变大，背景上下文减少，边界截断和 JPEG 重编码也会改变外观。为此，云端训练数据由完整图像和基于训练标注生成的动态 ROI 混合组成。训练 ROI 使用与部署相同的扩展规则和 JPEG codec；裁剪内可见比例低于阈值的目标不作为有效标签。云端模型仍保留完整图像训练样本，使 Full 动作的精度不会因 ROI 专门化而显著退化。

最终协议同时报告原始云模型、混合 Full+ROI 模型和独立 ROI expert：前两者用于判断混合训练是否破坏 Full 精度，后者用于测量 ROI 动作本身的可用上限。只有当校准集上的 mean ROI gain 大于零且 Full AP 退化不超过预注册容差时，ROI 动作才进入最终测试。

### 4.4 检测收益监督

为避免将“端侧低置信”直接当作“云端有收益”，本文从端侧、Full 和 ROI 的冻结预测构造 teacher。对预测框与 GT 进行类别一致的一对一匹配，定位质量由 IoU 阈值 \(0.50:0.05:0.95\) 上的通过比例表示，匹配贡献再乘以置信度平方根；未匹配且置信度不低于部署阈值的预测受到轻量 FP 惩罚。图像质量代理写为

\[
\tilde q(B,y)=
\frac{1}{\max(1,|y|)}
\sum_{(p,g)\in M}
s_p^{1/2}L_{\mathrm{IoU}}(p,g)
-\eta\sum_{p\in FP}s_p^{1/2}.
\]

由此得到

\[
g_i^F=\tilde q(B_i^F,y_i)-\tilde q(B_i^E,y_i),
\qquad
g_i^R=\tilde q(B_i^R,y_i)-\tilde q(B_i^E,y_i).
\]

帧排序 teacher 使用可用动作收益 \(g_i^{best}=\max(g_i^F,g_i^R)\)，动作 teacher 使用有符号差值 \(m_i=g_i^R-g_i^F\)。检测器训练图像不直接生成同模型 in-sample teacher；正式实验使用 detector holdout 或 detector OOF prediction，避免 selector 学到因检测器训练泄漏造成的虚假收益。

### 4.5 类别条件收益先验

置信度是强信号，但同一置信水平下不同类别的端云纠错能力可能不同。本文首先拟合类别条件收益先验，而不是直接从高维特征学习完整分数。对类别 \(c\)，平滑收益均值为

\[
\mu_c=
\frac{\sum_i w_{ic}g_i^{best}+\beta\bar g}
{\sum_i w_{ic}+\beta},
\]

其中 \(w_{ic}\) 为图像 \(i\) 中类别 \(c\) 的最大端侧置信度，\(\bar g\) 为全局平均收益，\(\beta\) 控制向全局均值收缩。空检图像使用独立的 empty 类别。包含多个类别时，图像先验为各类别 \(\mu_c\) 的置信度加权平均，记为 \(s_i^{prior}\)。

该先验具有三项作用：提供无需深层网络的可部署排序；显式表达端云模型的类别互补性；在语义残差未通过校准门禁时作为安全回退。

### 4.6 预算边界语义残差

端侧 YOLO 前向输出的 P3、P4 和 P5 特征分别进行全局平均池化并拼接：

\[
h_i=[
\operatorname{GAP}(P3_i);
\operatorname{GAP}(P4_i);
\operatorname{GAP}(P5_i)].
\]

当前实现得到 448 维向量。PCA 只在 selector-train 上拟合并降至 32 维，记为 \(z_i\)。语义分支不替换类别先验，而只学习线性残差：

\[
s_i=s_i^{prior}+\alpha
w^\top\frac{z_i-\mu_z}{\sigma_z}.
\]

训练目标与固定预算的 Top-K 问题对齐。对预算集合 \(\mathcal B\)，只采样跨越各预算边界且 teacher 收益有严格差异的样本对 \((i,j)\)，优化加权 pairwise logistic loss：

\[
\mathcal L_{\mathrm{rank}}=
\sum_{(i,j)\in\mathcal P}
\omega_{ij}\log\left(1+\exp(-(s_i-s_j))\right)
+\lambda\|w\|_2^2.
\]

权重 \(\omega_{ij}\) 随 teacher gain gap 增大，并对低预算边界给予更高权重。残差尺度 \(\alpha\)、PCA 维数和正则强度仅在 selector-calibration 上选择。该结构的目的不是增加视觉计算，而是利用端侧模型已经计算的语义表示修正 confidence/class prior 无法区分的样本。

### 4.7 ROI/Full 动作头

动作头输入检测统计、类别先验、ROI 几何、动态扩展比例和 ROI/Full payload 比，将其记为 \(v_i\)，监督信号为 \(m_i=g_i^R-g_i^F\)。当前实现采用轻量 hurdle 模型：第一组深度为 1 的 boosted decision stumps 估计 \(P(m_i>0\mid v_i)\)，两组鲁棒回归 stump 分别估计 ROI 优势和 Full 优势的条件幅度。预测期望差值为

\[
\hat m_i=
\hat p_i\hat m_i^+
-(1-\hat p_i)\hat m_i^-.
\]

这种分解避免大量接近零或符号不平衡的 action margin 使单一回归器退化。动作头与帧排序器独立训练；对训练集采用交叉拟合动作预测构造帧排序 teacher，减少同一动作头在自身训练样本上的乐观偏差。

### 4.8 成本感知动作选择

当图像被帧排序器选中且 ROI 可用时，系统比较 ROI 与 Full：

\[
d_i=
\alpha_a\hat m_i-\tau_a
+\lambda_s\frac{b_i^F-b_i^R}{b_0}.
\]

若 \(d_i>0\)，选择 ROI，否则选择 Full。帧的最终选择分数还减去动作 payload：

\[
\tilde s_i=s_i-\lambda_b\frac{b_{i,a_i}}{b_0}.
\]

这一规则默认 Full-safe：ROI 必须同时获得预测动作优势或足够大的成本节省才能替代 Full。对于要求更严格的安全版本，还可使用校准残差下界 \(\hat m_i-z_\delta\hat\sigma_i>0\) 作为 ROI 许可条件。本文把该安全版本作为消融，不声称它对全局 COCO AP 提供有限样本保证。

### 4.9 Fixed-budget 与 Online 决策

Fixed-budget 协议在给定预算 \(K\) 时选择排序最高的恰好 \(K\) 张图，用于比较不同方法的排序能力。Online 协议只在 selector-calibration 上选择绝对阈值 \(\tau_b\)：

\[
a_i=
\begin{cases}
E,& \tilde s_i<\tau_b,\\
R\text{ or }F,& \tilde s_i\ge\tau_b.
\end{cases}
\]

最终测试逐图独立决策，不读取其他测试图像，也不通过 Top-K 截断修正上传比例。校准目标上传率和最终自然上传率的差值作为 threshold drift 报告。Official 和 Deploy 评估复用完全相同的冻结动作，防止评估置信阈值改变路由行为。

### 4.10 ROI 映射、MAUR 融合与异步执行

ROI 云端框按裁剪偏移和尺度映射回原图。为避免局部动作删除区域外有效端侧目标，融合时保留与 ROI 交叠比例低于阈值的端侧框，仅用云端 ROI 结果替换区域内预测，随后执行类别感知 NMS。本文将这一机制称为区域外端侧结果保留，而不把它单独作为学习器。

视频部署采用采集/端侧检测主线程和上传线程解耦的有界队列。每个云端结果携带帧号、采集时间和返回时间；超过 deadline 的结果不回写历史输出。队列策略包括 drop-new、drop-oldest 和 keep-latest。静态数据只用于重放不同网络条件，完整视频实验另外报告 stale result ratio、timely result ratio 和 dropped upload ratio。

### 4.11 复杂度

检测统计计算为 \(O(N_i)\)，其中 \(N_i\) 为端侧框数量。FPN 全局池化复用现有特征图，PCA 投影和线性残差为固定维矩阵乘法；在 448 维输入和 32 维 PCA 配置下，投影矩阵与排序向量共含 14,368 个乘法系数，另保存中心化和标准化缓冲。稳定态后处理耗时为【待补】ms/image。动作头由有限数量的单层决策 stump 构成，推理复杂度与 stump 数线性相关。论文最终同时报告 selector-only 耗时和包含特征池化的总附加耗时，避免只报告线性头而忽略特征读取。

## 5 实验设计

### 5.1 研究问题

实验围绕以下五个研究问题展开。

- **RQ1：端云互补性是否足以支撑选择性卸载？** 比较 Edge-only、Cloud-only、随机上传、严格 Oracle 和 global-mAP greedy reference，测量在当前检测器组合与动作空间中可利用的真实空间。
- **RQ2：学习路由能否超过强规则基线？** 在完全相同的固定调用预算和在线阈值协议下，比较 Confidence-only、Composite rule、ClassGainPrior、MORIC-style、M2E 和本文方法，而不只与随机上传比较。
- **RQ3：ROI 是否带来可重复的成本优势？** 分离考察 ROI 动作可用率、ROI 相对 Edge 的收益、ROI 相对 Full 的胜率、payload 节省和 crop-aware 云模型的作用。
- **RQ4：收益能否跨数据分布保持？** COCO 用作通用 IID 检测对照，ENA24 和 Oregon Critters 用于考察低目标密度、野外相机背景与类别互补性；负结果和回退率也纳入报告。
- **RQ5：收益是否覆盖路由器本身的系统代价？** 报告新增参数、特征池化、排序、ROI 裁剪编码、云端执行、融合和网络传输的分项耗时，并分析网络和队列条件。

### 5.2 数据集与无泄漏划分

表 2 给出各数据集的用途。所有 selector 超参数、温度、阈值、成本权重和 ROI 扩展规则都只允许在 selector-train 与 selector-calibration 上确定；最终测试在策略冻结后只执行一次。若已有实验曾反复查看最终划分，则该划分只能标记为 development test，投稿前必须另建隐藏测试或使用独立数据集复核。

**表 2 数据集、划分和研究作用。**

| 数据集 | 检测器训练 | Selector-train | Selector-calibration | Final test | 研究作用 |
| --- | ---: | ---: | ---: | ---: | --- |
| COCO 2017 | 官方 train2017 或官方预训练权重 | train2017 中固定且互斥的【待补】张 | 【待补】张 | val2017 5,000 张 | 通用多类别、多目标 IID 对照 |
| ENA24 | detector-fit 4,800 张 | 1,400 张 | 400 张 | 原始 val 1,758 张 | 低目标密度野外检测、ROI 成本分析 |
| Oregon Critters | 按 camera/site 分组的 detector-fit【待补】张 | 【待补】张 | 【待补】张 | 独立 camera/site【待补】张 | 跨相机背景、物种和域差异 |

ENA24 原始训练集包含 7,031 张图像，另留 431 张作为检测器验证，其余按上表用于 selector；原始验证集仅用于最终评估。该数据集平均约 1.1 个标注框/图且不包含空图，因此适合验证局部复核，但不能支持“空帧过滤”结论。Oregon Critters 的随机图像划分可能使同一相机背景同时进入训练和测试，正式实验采用 camera/site group split；已有 70,034/14,792/14,803 随机划分结果仅作诊断。COCO 上不重新调节野外数据集的阈值，以避免把跨数据集测试变成二次校准。

### 5.3 检测器、输入与编码

端侧使用 YOLO26n 或 YOLO26s，云端使用 YOLO26x [2]。每个数据集先独立报告 Edge-only 与 Cloud-only 的原生验证结果；若 Cloud-only 对 Edge-only 的增益过小，则该组合不适合证明复杂路由器的精度价值，但仍可用于分析负结果。主实验输入尺寸为 640，IoU NMS 阈值为 0.7，最大检测数为 300，关闭 end-to-end head，其他设置见复现实验配置。

传输主协议统一使用 JPEG Q80：Full 动作对原图重新编码为 Q80，ROI 动作使用相同 codec 和质量参数，避免“Full 使用原始 JPEG、ROI 使用 Q80”造成不公平。原始文件传输、Q50、Q70 和 Q90 作为编码消融。ROI 的云端输入保持实际裁剪尺寸或按预注册规则 letterbox 到云端输入；所有 payload 均从实际编码字节计数，而不是由像素面积估算。

### 5.4 对比方法

对比方法分为基础模型、规则路由、学习路由和不可部署参考四组。

**基础模型。** Edge-only 始终使用端侧结果；Cloud-only 对所有图像执行云端整图检测。两者给出系统下界和云端模型参考上界，但 Cloud-only 不是选择性路由 Oracle。

**非学习基线。** Random-Full 在每个固定预算独立采样相同数量的图像并上传整图；Random-Action 进一步匹配被比较方法的调用数以及 Full/ROI 动作数，只在 ROI 可用样本间随机分配局部动作。两者均重复 2,000 次并报告均值和 95% 区间。Confidence-only 的预注册不确定度为

\[
u_i^{conf}=\begin{cases}
1,&\text{端侧无 deploy 框},\\
1-\max_{p\in B_i^E,\,s_p\geq 0.25}s_p,&\text{其他情况}.
\end{cases}
\]

Confidence-only Top-K 按 \(u_i^{conf}\) 排序并恰好上传 \(K\) 张整图；Confidence-only online 在 calibration 上确定绝对阈值；Confidence-only cost-aware 使用 \(u_i^{conf}-\lambda_b b_i^F/b_0\) 排序或阈值，以便与成本感知学习器公平比较。最低框置信度和分位置信度版本仅作为敏感性分析。Composite rule 复现低置信、小目标、贴边、稀疏检测和尺度不稳定规则。DCSB-style [10]、Adaptive Feeding-style [9] 和 bandwidth-adaptive DCSB [36] 仅使用各论文允许的端侧信息；不能完全复现的实现标记为 reimplementation，而不宣称等同原作者代码。ClassGainPrior 使用训练集类别条件收益，不使用测试 GT。

**学习基线。** MORIC-style 复现 mAP-context reward 的整图收益学习 [11]；M2E full-only 学习云端整图相对端侧的 teacher gain；M2E Full+ROI 在相同动作缓存上联合选择 ROI 与 Full。本文方法与所有学习基线使用同一 detector prediction cache、同一 teacher split、同一调参预算和同一动作融合实现。历史探索中的 SAFE、CARS、GEC、BAR 和 BCRank 只在附录诊断表报告，不通过挑选其最佳测试点增加主文比较数量。

**不可部署参考。** Strict Top-K gain Oracle 按 GT 已知的最佳 raw gain 排序并严格选择 \(K\) 张，不加入成本惩罚。Oracle rank + learned action 与 learned rank + Oracle action 用于分解排序误差和动作误差。Global-mAP greedy reference 每一步尝试能使完整 COCOeval AP 增量最大的动作，近似求解全局组合，但受贪心路径和候选动作缓存限制。所有 Oracle 表格均标注 GT-aware、not deployable，并禁止与可部署方法混称为“理论绝对上限”。

### 5.5 Official、Deploy、Fixed-budget 与 Online 协议

本文同时使用四个互不混淆的协议。

1. **Official AP。** 输出阈值为 0.001，`iou=0.7`，`max_det=300`，`end2end=false`，通过原生 `YOLO.val` 产生预测并用 COCO API 计算 AP。缓存评估必须先与原生验证结果做 parity 检查，\(\lvert\Delta AP\rvert\leq 0.001\) 才可进入主表。
2. **Deploy AP。** 输出阈值为 0.25，报告 mAP50-95、Precision、Recall 和 F1。它只与相同阈值下的 Edge、Cloud 和 Fused 结果比较，不能与 Official AP 混算增益。
3. **Fixed-budget。** 给定调用预算 \(b\in\{0,5,10,15,25,40,50,65,80,100\}\%\)，所有排序方法严格选择相同的 \(K=\lfloor bN\rfloor\) 张。若方法允许跳过负收益动作，则另标记为 positive-utility policy，不进入严格 Top-K 主表。
4. **Online。** calibration 上选择阈值，final test 中逐图独立执行，不读取测试集大小和其他样本分数。主表同时列目标调用率、自然调用率和 threshold drift。

Official 与 Deploy 复用同一组冻结路由动作；改变检测输出阈值不得重新选择上传样本。Fixed-budget 用于比较排序能力，Online 才对应真实部署。两者均需要报告调用比例与实际 KB/image，因为相同调用次数下 ROI 与 Full 的传输成本不同。

### 5.6 指标

检测指标包括 COCO mAP50-95、mAP50、mAP75、APsmall/APmedium/APlarge，以及 Deploy Precision、Recall 和 F1。野外数据集另外按类别、目标面积、目标数和相机组报告宏平均结果，防止高频类别掩盖尾部退化。

系统指标定义如下：

| 指标 | 定义 |
| --- | --- |
| Upload ratio | 至少调用一次云端的图像比例 |
| Full / ROI ratio | Full 或 ROI 动作占全部图像的比例，两者之和等于 Upload ratio |
| Upload KB/image | 实际传输图像字节总和除以图像数 |
| Response KB/image | 序列化云端框、类别、置信度和协议头的平均返回字节 |
| GFLOPs/image | Edge 固定计算、按动作均摊的 Cloud/ROI 计算和 selector 计算之和 |
| Precision@K | 选中样本中 teacher raw gain 大于零的比例 |
| Mean selected gain | 选中样本的平均 teacher raw gain，仅用于解释选择器 |
| Gain@budget | 固定预算内选中 teacher gain 之和 |
| Random 95% CI | 同预算随机策略 AP、KB 和 gain 的经验区间 |
| Oracle gap recovered | \((AP_{method}-AP_{edge})/(AP_{oracle}-AP_{edge})\)，仅在同预算同动作空间计算 |

估算端到端时延写为

\[
\hat L_i=t_i^E+t_i^{sel}+t_i^{crop/enc}
+\frac{8b_i}{u}
+\mathbb 1(a_i\ne E)(RTT+t_i^C+t_i^{fusion}),
\]

其中 \(b_i\) 以 KB 计、上行 \(u\) 以 Mbps 计，因此传输项单位为 ms。网络剖面为 1/5/10/20 Mbps 和 RTT 20/50/100 ms。固定公式所得结果一律写为 **Estimated E2E**；只有在指定端侧硬件、云 GPU 和真实链路上逐段打点的结果才写为 measured latency。除均值外报告 P50、P95 和 P99。

### 5.7 统计分析与复现门禁

每个随机学习器使用 5 个训练随机种子，主表报告均值和标准差；确定性方法记录一次结果。Random 使用固定种子列表重复 2,000 次。方法差异采用相同图像上的配对 bootstrap，重复 2,000 次，报告 AP 差异和 KB 差异的 95% 区间。超参数选择完成后生成带 SHA256 的冻结清单，记录模型、数据 YAML、图像列表、标注 JSON、codec、head mode、批量、设备和代码提交。以下任一条件失败时，不生成论文主表：

- 原生 `YOLO.val` 与 prediction-cache COCOeval parity 失败；
- detector-fit、selector-train、calibration 和 final test 存在 image 或 camera-group 交集；
- Official 与 Deploy 使用了不同路由动作；
- Full 与 ROI 主比较未使用同一 codec 质量；
- semantic residual 未在 calibration 上超过冻结锚点，却仍绕过回退门控；
- ROI 动作在 calibration 上 mean raw gain 不大于零；
- 估算时延被标记为实测时延。

## 6 实验结果

> **初稿数据说明：** 本节表格已固定比较对象、指标和允许的结论类型。方括号内容必须由冻结 final-test 产物自动填入。开发过程中反复查看过的 ENA24/Oregon/COCO 数字不得直接转为投稿主结果；若无法重建隐藏测试，应明确标记为 development evidence。

### 6.1 端云检测器基线与互补性

**表 3 检测器基线和端云互补性。**

| 数据集 | 模型 | mAP50-95 | mAP50 | mAP75 | APsmall | GFLOPs | 实测推理 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO | YOLO26s Edge | 【待补/官方 parity】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| COCO | YOLO26x Cloud | 【待补/官方 parity】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ENA24 | YOLO26n Edge | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ENA24 | YOLO26x Cloud | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Oregon | YOLO26n Edge | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Oregon | YOLO26x Cloud | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

除平均 AP 差异外，图 2 将每图 teacher gain 按类别、目标尺度、目标数和相机组分桶。如果 Cloud 对 Edge 的条件增益在各桶近似常数，confidence-only 理论上已接近充分，语义路由不应被预期获得大幅优势；若某些类别、尺度或背景组表现出稳定正增益，则这些结构构成学习路由的可利用空间。本文以此分析解释结果，而不从单一总 AP 反推路由器优劣。

**[图 2 待绘制：各数据集按类别、尺度和相机组的 Cloud-Edge gain 分布及置信度相关性。]**

### 6.2 Fixed-budget Official AP

**表 4 Fixed-budget Official AP 主结果。每个预算严格选择相同调用数，主表使用 Q80。**

| 方法 | Budget | mAP50-95 | mAP50 | mAP75 | Upload | Full | ROI | KB/img | GFLOPs/img | Est. E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge-only | 0% | 【待补】 | 【待补】 | 【待补】 | 0 | 0 | 0 | 0 | 【待补】 | 【待补】 |
| Random Full | 25% | 【待补 ± CI】 | 【待补】 | 【待补】 | 25% | 25% | 0 | 【待补】 | 【待补】 | 【待补】 |
| Random Action-matched | 25% | 【待补 ± CI】 | 【待补】 | 【待补】 | 25% | 【匹配】 | 【匹配】 | 【待补】 | 【待补】 | 【待补】 |
| Confidence Top-K | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 25% | 0 | 【待补】 | 【待补】 | 【待补】 |
| Confidence cost-aware | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 25% | 0 | 【待补】 | 【待补】 | 【待补】 |
| Composite rule | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| MORIC-style | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 25% | 0 | 【待补】 | 【待补】 | 【待补】 |
| M2E Full+ROI | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| 本文方法 | 25% | **【待补】** | 【待补】 | 【待补】 | 25% | 【待补】 | 【待补】 | **【待补】** | 【待补】 | **【待补】** |
| Oracle Full+ROI | 25% | 【待补】 | 【待补】 | 【待补】 | 25% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Cloud-only | 100% | 【待补】 | 【待补】 | 【待补】 | 100% | 100% | 0 | 【待补】 | 【待补】 | 【待补】 |

完整 0%-100% 曲线放在图 3 和附录表 A1，主文只展示预注册的 10%、25% 和 40% 三个代表点。结论应根据最终数据从以下两种表述中选择，而不能预先写死：若本文方法在 AP 和成本上同时支配 Confidence，则报告 Pareto improvement；若 AP 置信区间重叠但 KB 显著降低，则表述为“在统计等价或接近的 AP 下减少传输”；若 AP 明显低于 Confidence，则只保留其为成本优先运行点，不能声称全面优越。

**[图 3 待绘制：mAP50-95-Upload ratio、mAP50-95-KB/image 与 mAP50-95-Estimated E2E 三组 Pareto 曲线；Random 95% CI 用阴影显示。]**

### 6.3 Online Deploy 与自然上传率

**表 5 Online 阈值系统的部署结果。阈值只在 calibration 上确定。**

| 方法 | Target rate | Natural upload | Drift | Full | ROI | Deploy mAP50-95 | Precision | Recall | F1 | KB/img | Est. E2E@10Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Confidence online | 10% | 【待补】 | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Confidence cost-aware online | 10% | 【待补】 | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ClassGainPrior online | 10% | 【待补】 | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| M2E online | 10% | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| 本文方法 online | 10% | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | **【待补】** |
| Confidence online | 25% | 【待补】 | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| 本文方法 online | 25% | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | **【待补】** |
| Confidence online | 40% | 【待补】 | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| 本文方法 online | 40% | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | 【待补】 | 【待补】 | 【待补】 | **【待补】** | **【待补】** |

Online 结果回答真实部署中的两个问题：校准阈值能否在未知测试分布上维持调用率，以及 ROI 是否在不依赖 Top-K 的情况下自然产生。若 threshold drift 较大，应按固定阈值如实报告，而不能在 final test 上重新搜索阈值。Official Online 表使用相同动作另列于附录，避免一张表混合两种输出口径。

### 6.4 选择器质量、随机性与 Oracle gap

**表 6 选择器质量与上限分解，示例为 25% 预算。**

| 方法 | Precision@K | Mean selected gain | Gain@budget | AP | Random AP 95% CI | Oracle gap recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 0 |
| Confidence | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 【待补】 |
| ClassGainPrior | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 【待补】 |
| 本文排序 + learned action | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 【待补】 |
| 本文排序 + Oracle action | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 【待补】 |
| Oracle rank + learned action | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 【待补】 |
| Full+ROI Oracle | 100% 定义相关 | 【待补】 | 【待补】 | 【待补】 | 不适用 | 100% |

该分解用于决定误差主要来自帧排序还是 ROI/Full 动作头。如果“本文排序 + Oracle action”只小幅提高，而“Oracle rank + learned action”大幅提高，则瓶颈在帧选择；反之则应优先改进动作监督。Oracle gap recovered 只描述当前 detector prediction、预算和动作缓存下的空间，不代表对未知模型或真实环境的普遍上限。

### 6.5 ROI 动作成立的条件

**表 7 ROI 机制分析。**

| 数据集/云模型 | ROI available | Mean Full gain | Mean ROI gain | ROI beats Edge | ROI beats Full | Full AP | ROI-crop AP | Mean ROI/Full bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 YOLO26x | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Full+ROI mixed fine-tune | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| ROI expert | 【待补】 | 不适用 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

ROI 上传率低不能只归因于动作头。若 ROI teacher 本身大多为负，合理策略就应回退 Full；其原因可能是端侧框漏掉真正困难目标、扩框不足、裁剪上下文丢失、类别映射错误或云模型只适应整图。实验依次比较固定 1.2/1.5/1.8/2.0 倍扩框和动态扩框，并把 ROI raw gain、payload 和 Full 退化同时作为选择标准。扩框越大通常提高上下文但减少成本优势，因此不以 ROI 调用率本身作为优化目标，而以单位字节条件收益为目标。

### 6.6 组件消融

**表 8 方法消融，固定 25% 调用预算。**

| 变体 | mAP50-95 | Full | ROI | KB/img | Est. E2E | 解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Confidence Full-only | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 强单标量锚点 |
| ClassGainPrior Full-only | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 类别互补性 |
| + semantic residual | 【待补】 | 【待补】 | 0 | 【待补】 | 【待补】 | 帧排序残差 |
| + ROI action，无 cost | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 只优化预测收益 |
| + ROI action，有 cost | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 完整方法 |
| 去除 FPN 特征 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 检测统计与类别先验 |
| 去除 ClassGainPrior | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 纯语义排序风险 |
| 固定 ROI 1.8 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 固定上下文 |
| 动态 ROI | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 尺度感知上下文 |
| MAUR off | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | ROI 内替换全部端侧框 |
| MAUR on | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 保留 ROI 外端侧框 |
| 无 calibration fallback | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 检验安全回退 |

此外报告 PCA 维数 8/16/32/64、cost weight、动作门槛和 teacher 质量定义的敏感性。所有消融共享同一 calibration，不允许每个预算在 final test 上选择不同最佳变体。

### 6.7 编码、网络与队列

**表 9 编码与网络敏感性。**

| Codec/网络 | mAP50-95 | KB/img | Mean E2E | P95 | P99 | Timely ratio | Stale ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q50, 10 Mbps/50 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Q70, 10 Mbps/50 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Q80, 10 Mbps/50 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Q90, 10 Mbps/50 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Q80, 1 Mbps/100 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |
| Q80, 20 Mbps/20 ms | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 | 【待补】 |

视频重放采用 5/10/15/30 FPS，队列长度 0/1/2/4，drop-new、drop-oldest、keep-latest 和 value-aware 四种策略，deadline 为 100/200 ms。该实验借鉴动态视频分析系统对内容、带宽和结果新鲜度的联合考虑 [26,28-31,34,35]，但若输入仍为打乱的静态图像，只能称为队列压力模拟；只有保持原视频时间顺序并带有时序标注时才报告 estimated realtime mAP。

### 6.8 跨数据集边界与负结果

COCO 是检验强 confidence baseline 的关键负对照。若在 COCO 上 Confidence-only 与 Cloud-only 的差距很小，或 semantic residual 在 calibration 上低于锚点，本文方法按协议回退，并报告零或有限学习收益。这一结果与置信度级联充分条件一致 [15]，不能被隐藏或通过选择单一预算规避。

ENA24 用于验证目标数较少时 ROI 的传输收益，但较小的 Edge-Cloud AP gap 会限制绝对提升。Oregon Critters 用于考察相机背景和类别互补性；若 random split 上有效而 camera-group split 上失效，说明学习器利用了背景记忆而非可迁移纠错信号。最终论文至少需要一个数据集在独立测试上形成稳定 AP-KB-E2E 改进，同时在其他数据集如实说明适用边界，才足以支持“条件有效”的主张。

### 6.9 定性案例

图 4 展示四类案例：Confidence 与本文方法都上传且云端修正成功；Confidence 未上传但类别/语义残差选中并修正；本文选择 ROI 并以较低 payload 保持 Full 结果；本文失败或不必要上传。每个案例同时显示端侧框、ROI、Full 框、最终融合框、动作分数、实际 KB 和 GT。案例只能从冻结 final test 按预注册规则导出，不能仅挑选视觉上有利的样本。

**[图 4 待导出：成功、规则漏选、ROI 节省和失败四类可视化案例。]**

## 7 讨论

### 7.1 学习路由不是置信度的必然替代品

本文的理论和实验都把 Confidence-only 视为强基线，而不是容易击败的占位方法。端侧置信度同时编码目标存在性、分类难度和部分定位质量；当云端模型只是端侧模型的同分布放大版本时，低置信样本往往也是最值得调用云端的样本。EdgeBoost 在分类任务上观察到校准阈值可以接近学习路由 [14]，级联理论也给出了这一现象的条件 [15]。因此，本文方法的有效性不由“是否使用神经学习器”定义，而由它是否利用置信度之外且在独立数据上可预测的条件收益定义。

### 7.2 ROI 的价值来自动作专门化而非上传比例

ROI 不会天然提高精度。它只能复核端侧已经定位到的区域，无法修复区域外漏检；裁剪还会改变尺度和上下文。提高 ROI 比例本身可能迫使系统执行负收益动作。更合理的目标是提高 \(P(\Delta q_R>0)\)、\(P(\Delta q_R-C_R>\Delta q_F-C_F)\) 以及单位字节收益。crop-aware 训练、动态扩框和区域外结果保留分别作用于输入适配、上下文与融合，它们应先提升 ROI 动作本身，再由动作头选择。若 ROI teacher 不成立，系统应诚实退回 Full。

### 7.3 为什么解耦帧排序和动作选择

一体化三分类头容易把两个不同问题混合：帧排序关注有限预算下哪些样本具有较高远端收益，动作选择关注同一被选中样本上 ROI 与 Full 的相对效用。前者的类不平衡由高收益尾部决定，后者则受 ROI 可用率、payload 和 crop 模型适配影响。本文用排序锚点加语义残差解决前者，用 hurdle action model 解决后者，并通过 Oracle rank/action 分解验证瓶颈。这种结构比直接增大 MLP 更容易解释，也把失败定位到可操作的模块。

### 7.4 与已有工作的关系

相较 DCSB 和 Confidence-only，本文不把端侧困难度直接等同于云端收益 [10]；相较 ORIC/MORIC，本文保留 mAP-context reward 思想，但把动作空间从 Edge/Full 扩展到 Edge/ROI/Full，并显式计入真实 payload [11]；相较 EdgeDuet、AccMPEG 和 Selective Cloud Offloading，本文重点是低开销条件动作路由，而不是 tile 并行、宏块质量或 conformal 风险控制 [20-22]；相较 Reducto、ApproxDet 和 SmartDet，本文主要证据来自逐图检测，尚未声称解决完整时序控制 [26,28,29]。这一边界使论文贡献集中在“检测器互补性驱动的多动作成本路由”，而不是把所有端云视频问题合并为一个系统。

### 7.5 部署选择

部署方应先测量当前模型对的端云条件增益，再决定策略复杂度。若 confidence calibration 后的排序已经接近 Cloud-only，Confidence online 是更稳妥的实现；若类别或场景存在稳定互补性，可启用 ClassGainPrior；只有 semantic residual 在独立 calibration 上通过门禁时才启用完整排序器。类似地，只有 ROI teacher 通过收益门禁时才开放 ROI 动作。该逐级回退机制使系统复杂度随可验证收益增加，而不是默认部署全部组件。

## 8 局限性与有效性威胁

第一，COCO AP 不可逐图分解，本文 teacher gain 和 Strict Top-K Oracle 都是代理。Global-mAP greedy reference 更接近当前缓存下的 AP 上限，但仍非全局最优。第二，ROI 由端侧框触发，因此无法修复端侧完全漏掉且附近无候选的目标；多区域 proposal、tile 或轻量显著性网络可能扩大动作空间，但会增加端侧开销。第三，数据集规模和相机重复背景可能使类别先验或语义残差过拟合，尤其需要 camera-group split 和跨时间验证。

第四，Estimated E2E 依赖带宽、RTT 和云端耗时假设，不能替代真实链路测量；队列重放也不能替代连续视频评估。第五，本文主要使用同一模型家族的不同规模，结论可能不适用于异构架构或云端 specialist。第六，上传裁剪仍可能暴露敏感视觉信息，本文只优化字节和精度，尚未量化隐私泄漏、能耗和云服务货币成本。第七，若 final test 已在探索中被多次查看，统计区间不能消除选择偏差，必须使用新隐藏测试或第二独立数据源。

## 9 结论

本文研究端云目标检测中的 Edge、ROI 和 Full 多动作选择问题。与仅依据端侧不确定度上传整图的方法不同，本文把决策目标定义为各动作的条件纠错收益减去实际传输、计算和时延成本，并以类别条件收益为排序锚点、同次端侧前向中的低维语义表示为残差、独立动作头为 ROI/Full 选择器。裁剪感知云模型和区域外端侧结果保留进一步提高局部动作的可用性。完整协议严格区分 Official 与 Deploy、Fixed-budget 与 Online、可部署方法与 GT-aware Oracle，以及估算与实测时延。

最终实验将检验本文方法能否在强 Confidence-only、MORIC-style 和 M2E 基线上形成稳定的 AP-KB-E2E Pareto 改进。无论学习路由是否在所有数据集获胜，本文的核心结论都应建立在可复核的条件边界上：只有当云端纠错和 ROI/Full 差异包含置信度之外的输入相关结构，并且这一结构能够从独立训练数据迁移到测试环境时，学习型多动作路由才值得其额外复杂度。

## 声明

**资金支持：**【待补】

**利益冲突：** 作者声明不存在与本文工作相关的利益冲突。【投稿前确认】

**数据可用性：** COCO 数据及标注由其官方渠道提供；ENA24 与 Oregon Critters 的可公开范围、下载地址和许可信息为【待补】。论文接收后计划公开数据划分清单、配置和评估脚本；受许可限制的原始图像不重新分发。

**代码可用性：** 匿名评审代码仓库与归档 DOI 为【待补】。归档内容包括数据清单哈希、模型哈希、prediction cache 构建、selector 训练、Official/Deploy 评估、随机置信区间和网络模拟脚本。

**作者贡献：**【按 CRediT taxonomy 待补】

**伦理声明：**【若野外相机数据涉及人员或敏感地点，补充采集许可、匿名化和伦理审批信息；否则说明不适用】

## 参考文献

1. Lin TY, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. In: Computer Vision - ECCV 2014. Springer; 2014:740-755. https://doi.org/10.1007/978-3-319-10602-1_48

2. Jocher G, Qiu J, Liu M, Lyu S, Akyon FC, Kalfaoglu ME. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv preprint arXiv:2606.03748. 2026. https://doi.org/10.48550/arXiv.2606.03748

3. Zhao Y, Lv W, Xu S, et al. DETRs Beat YOLOs on Real-time Object Detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024:16965-16974. https://doi.org/10.1109/CVPR52733.2024.01605

4. Shuvo MMH, Islam SK, Cheng J, Morshed BI. Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review. Proceedings of the IEEE. 2023;111(1):42-91. https://doi.org/10.1109/JPROC.2022.3226481

5. Zhou Z, Chen X, Li E, Zeng L, Luo K, Zhang J. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. Proceedings of the IEEE. 2019;107(8):1738-1762. https://doi.org/10.1109/JPROC.2019.2918951

6. Kang Y, Hauswald J, Gao C, et al. Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. In: Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems. 2017:615-629. https://doi.org/10.1145/3037697.3037698

7. Teerapittayanon S, McDanel B, Kung HT. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. In: 23rd International Conference on Pattern Recognition. 2016:2464-2469. https://doi.org/10.1109/ICPR.2016.7900006

8. Matsubara Y, Levorato M, Restuccia F. Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges. ACM Computing Surveys. 2023;55(5):1-30. https://doi.org/10.1145/3527155

9. Zhou HY, Gao BB, Wu J. Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors. In: Proceedings of the IEEE International Conference on Computer Vision. 2017:3505-3513. https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Adaptive_Feeding_Achieving_ICCV_2017_paper.html

10. Cao Z, Li Z, Chen Y, et al. Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. In: 43rd IEEE International Conference on Distributed Computing Systems. 2023:259-270. https://doi.org/10.1109/ICDCS57875.2023.00062

11. Qiu J, Wang R, Hu B, Guerin R, Lu C. Optimizing Edge Offloading Decisions for Object Detection. In: IEEE/ACM Symposium on Edge Computing. 2024:164-177. https://doi.org/10.1109/SEC62691.2024.00021

12. Gomez C, Patkhullaev D, Hernandez AC. OffloaD: Detection Failure-based Scheduler for Offloading Object Detection. ACM Transactions on Embedded Computing Systems. 2024;23(5):Article 84, 1-26. https://doi.org/10.1145/3677321

13. Yoon G, Kim GY, Yoo H, Kim SC, Kim R. Implementing Practical DNN-Based Object Detection Offloading Decision for Maximizing Detection Performance of Mobile Edge Devices. IEEE Access. 2021;9:140199-140211. https://doi.org/10.1109/ACCESS.2021.3118731

14. Said N, Landsiedel O. EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading. Computer Networks. 2025;269:111437. https://doi.org/10.1016/j.comnet.2025.111437

15. Jitkrittum W, Gupta N, Menon AK, Narasimhan H, Rawat AS, Kumar S. When Does Confidence-Based Cascade Deferral Suffice? In: Advances in Neural Information Processing Systems. 2023;36. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html

16. Enomoto S, Eda T. Learning to Cascade: Confidence Calibration for Improving the Accuracy and Computational Cost of Cascade Inference Systems. Proceedings of the AAAI Conference on Artificial Intelligence. 2021;35(8):7331-7339. https://doi.org/10.1609/aaai.v35i8.16900

17. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR; 2017;70:1321-1330. https://proceedings.mlr.press/v70/guo17a.html

18. Gilg J, Teepe T, Herzog F, Rigoll G. The Box Size Confidence Bias Harms Your Object Detector. In: IEEE/CVF Winter Conference on Applications of Computer Vision. 2023:1471-1480. https://openaccess.thecvf.com/content/WACV2023/html/Gilg_The_Box_Size_Confidence_Bias_Harms_Your_Object_Detector_WACV_2023_paper.html

19. Chen L, Zaharia M, Zou J. FrugalML: How to Use ML Prediction APIs More Accurately and Cheaply. In: Advances in Neural Information Processing Systems. 2020;33:10685-10696. https://proceedings.neurips.cc/paper/2020/hash/789ba2ae4d335e8a2ad283a3f7effced-Abstract.html

20. Wang X, Yang Z, Wu J, Zhao Y, Zhou Z. EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision. In: IEEE INFOCOM 2021. 2021:1-10. https://doi.org/10.1109/INFOCOM42981.2021.9488843

21. Du K, Zhang Q, Arapin A, Wang H, Xia Z, Jiang J. AccMPEG: Optimizing Video Encoding for Video Analytics. Proceedings of Machine Learning and Systems. 2022;4:450-466. https://proceedings.mlsys.org/paper_files/paper/2022/hash/853f7b3615411c82a2ae439ab8c4c96e-Abstract.html

22. Dehghani D, Chen Y, Yu X, Koudas N. Selective Cloud Offloading for Accurate and Efficient Object Detection. In: IEEE International Conference on Data Mining. 2025:1145-1154. https://doi.org/10.1109/ICDM65498.2025.00123

23. Xiang J, Zheng L, Huang L. PAROD: Real-time High-resolution Object Detection in Outdoor Scenes via Parallel Edge Offloading of Regions of Interest. In: International Joint Conference on Neural Networks. 2025:1-9. https://doi.org/10.1109/IJCNN64981.2025.11227878

24. Akamatsu S, Watanabe H. Bounding Box Aware Edge-Cloud Collaborative Method for Multiple Object Detection. In: IEEE 3rd World Conference on Applied Intelligence and Computing. 2024:1155-1159. https://doi.org/10.1109/AIC61668.2024.10731035

25. Assine JS, Santos Filho JCS, Valle E. Single-Training Collaborative Object Detectors Adaptive to Bandwidth and Computation. arXiv preprint arXiv:2105.00591. 2021. https://doi.org/10.48550/arXiv.2105.00591

26. Li Y, Padmanabhan A, Zhao P, Wang Y, Xu GH, Netravali R. Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. In: Proceedings of ACM SIGCOMM 2020. 2020:359-376. https://doi.org/10.1145/3387514.3405874

27. Kang D, Emmons J, Abuzaid F, Bailis P, Zaharia M. NoScope: Optimizing Neural Network Queries over Video at Scale. Proceedings of the VLDB Endowment. 2017;10(11):1586-1597. https://doi.org/10.14778/3137628.3137664

28. Xu R, Zhang CL, Wang P, et al. ApproxDet: Content and Contention-Aware Approximate Object Detection for Mobiles. In: Proceedings of the 18th ACM Conference on Embedded Networked Sensor Systems. 2020:449-462. https://doi.org/10.1145/3384419.3431159

29. Callegaro D, Levorato M, Restuccia F. SmartDet: Context-Aware Dynamic Control of Edge Task Offloading for Mobile Object Detection. In: IEEE 23rd International Symposium on a World of Wireless, Mobile and Multimedia Networks. 2022:357-366. https://doi.org/10.1109/WoWMoM54355.2022.00034

30. Ghosh A, Iyengar S, Lee S, Rathore A, Padmanabhan VN. REACT: Streaming Video Analytics on the Edge with Asynchronous Cloud Support. In: Proceedings of the 8th ACM/IEEE Conference on Internet of Things Design and Implementation. 2023:222-235. https://doi.org/10.1145/3576842.3582385

31. Zhang B, Jin X, Ratnasamy S, Wawrzynek J, Lee EA. AWStream: Adaptive Wide-Area Streaming Analytics. In: Proceedings of ACM SIGCOMM 2018. 2018:236-252. https://doi.org/10.1145/3230543.3230554

32. Chen T, Bu Y, Zeng Y, Xie L, Lu S. RegionFilter: Region-Aware Video Filtering Mechanism on Resource-Constrained Edge Nodes. Computer Networks. 2024;251:110624. https://doi.org/10.1016/j.comnet.2024.110624

33. Maresch M, Nastic S. VATE: Edge-Cloud System for Object Detection in Real-Time Video Streams. In: IEEE International Conference on Fog and Edge Computing. 2024:27-34. https://doi.org/10.1109/ICFEC61590.2024.00017

34. Sohn V, Kim S, Lee HW. Joint Frame Drop and Object Detection Task Offloading for Mobile Devices via RL With Lyapunov Optimization. IEEE Transactions on Mobile Computing. 2025;24:6168-6182. https://doi.org/10.1109/TMC.2025.3539356

35. Xu R, Razavi S, Zheng R. Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques. IEEE Communications Surveys & Tutorials. 2023;25(4):2951-2982. https://doi.org/10.1109/COMST.2023.3323091

36. Cao Z, Cheng Y, Zhou Z, et al. Edge-Cloud Collaborated Object Detection via Bandwidth Adaptive Difficult-Case Discriminator. IEEE Transactions on Mobile Computing. 2025;24(2):1181-1196. https://doi.org/10.1109/TMC.2024.3474743
