# 面向端云协同目标检测的成本感知样本卸载与自适应载荷选择

**Cost-Aware Sample Offloading and Adaptive Payload Selection for Edge-Cloud Object Detection**

作者姓名待补

作者单位待补

通讯作者：姓名与电子邮箱待补

## 摘要

资源受限视觉终端通常只能运行轻量目标检测器，而将全部图像上传至云端会持续占用上行带宽，并引入编码、传输、排队和云端推理开销。现有端云检测方法多集中于判断图像是否为难例，或者单独优化压缩与区域传输，较少在统一预算下联合考虑上传样本、载荷形式和精度风险。本文提出一种成本感知端云目标检测框架，将在线决策解耦为样本卸载、Original/Q90/ROI 载荷选择和预算调度，并通过离线控制器完成配置校准。对于调用量预算和在线阈值模式，控制器在检测精度相对参考配置下降不超过预设容忍量的条件下最小化实际请求字节；对于字节预算，控制器在容量约束内最大化 Official AP。候选配置随后在独立数据上接受配对 bootstrap 风险门禁，门禁失败时冻结回退至参考策略，最终测试数据不参与选择、调参或修复。

本文在 COCO、ENA24 和 Oregon 三个数据集上统一评测多种样本排序信号、三种载荷策略及 Fixed-call、Online、Fixed-byte 三类预算模式。实验重点比较 Official mAP50-95、实际请求 KB/image、上传率和 Original/Q90/ROI 动作比例，并分析置信度卸载、检测收益排序与成本感知配置之间的差异。【待冻结 final 完成后填入：本文方法相对参考配置的平均请求字节降幅、最差 AP 差、门禁通过率及 final-oracle regret。】结果表明，端云检测不能仅依据端侧不确定性决定是否上传；云端修复收益、载荷字节和预算语义需要在同一成本约束下共同评估。

**关键词：** 端云协同；目标检测；选择性卸载；成本感知路由；载荷选择；风险门禁

## 1 引言

低功耗智能摄像头被广泛用于野外监测、安防巡检和无人值守感知。轻量检测器能够在本地提供基本输出，但其模型容量、输入分辨率和硬件算子受限，对小目标、遮挡和域外场景的识别能力通常弱于云端高容量检测器。全量上云可以利用更强模型，却要求持续传输图像，并使结果受到网络往返、服务排队和云端计算的共同影响。因此，端云协同检测的核心不是简单选择端侧或云端，而是在有限资源下决定哪些图像值得上传以及上传何种表示。

现有方法已经证明选择性卸载具有价值。Adaptive Feeding 使用 SVM 在快速与高精度检测器之间选择 [10]；DCSB 及其带宽自适应版本通过难例判别器决定是否上云 [11,12]；OffloaD 根据检测失败风险调度任务 [20]；ORIC/MORIC 进一步指出，目标检测 mAP 是跨图像、跨类别聚合的指标，上传收益不应简化为单图分类正确性 [13]。这些方法主要关注“上传哪些图像”，且多数使用整图载荷。

通信成本还取决于“上传什么”。Original 保留完整图像信息但字节量最高；Q90 JPEG 可以显著减少请求量，却可能损害细粒度纹理；ROI 进一步压缩空间范围，但依赖端侧产生候选框，并可能因上下文缺失而降低云端检测质量。RegionFilter、EdgeDuet 和 AccMPEG 分别从区域过滤、小目标 tile 和任务感知编码角度说明，空间内容与编码质量会改变精度和传输成本 [14-16]。因此，只优化上传图像数量或固定采用 ROI，都不能充分描述端云系统的成本收益。

样本排序、载荷选择和预算调度之间还存在耦合。Fixed-call 固定上传样本数，适合比较排序能力；Online 通过冻结阈值逐样本接纳任务，但实际上传率会随数据分布漂移；Fixed-byte 限制总请求字节，此时更小的载荷还可能使预算容纳更多样本。同一策略在 Fixed-call 中可能因裁剪损失而不利，却在 Fixed-byte 中因释放容量而有效。若不区分预算语义，容易把上传率漂移或载荷压缩造成的收益错误归因于样本学习器。

本文据此提出成本感知样本卸载与自适应载荷选择框架。系统在线执行一个冻结的样本排序器、载荷策略和预算调度器；离线控制器则在独立校准数据上求解“精度容忍约束下的最小通信成本”问题，并在独立门禁数据上检查候选相对参考配置的 AP 与请求字节差异。该设计不把某个排序学习器视为跨域通用解，而是把端侧置信度、类别收益先验和检测收益学习器作为可替换的排序信号。论文的创新重点是成本约束下的样本与载荷联合配置，而不是排序器数量。

本文的主要贡献如下。

1. 提出一种解耦样本卸载、载荷动作和预算调度的端云目标检测架构，在不重新训练端侧和云端检测器的条件下统一支持 Original、Q90 与 ROI。

2. 提出一个校准式成本感知控制器。该控制器针对调用量和字节预算采用不同优化目标，在 AP 容忍约束下选择低请求成本配置，并通过独立配对 bootstrap 门禁冻结候选或回退参考策略。

3. 在三个数据域上使用统一检测缓存、Official AP 和实际载荷字节进行评测，系统比较 Confidence、检测收益排序和多载荷策略在 Fixed-call、Online 与 Fixed-byte 下的精度成本边界。

本文其余部分组织如下。第 2 节回顾选择性卸载、载荷优化和成本敏感路由；第 3 节定义问题并分析置信度卸载的充分条件；第 4 节介绍系统架构与成本感知控制器；第 5 节说明实验协议；第 6 节报告结果；第 7 节讨论系统验证与局限；第 8 节总结全文。

## 2 相关工作

### 2.1 选择性卸载与检测级路由

Neurosurgeon 依据设备、网络和服务器代价选择网络切分位置 [1]，split computing 与早退研究进一步分析了中间特征传输和多出口推理的成本 [2-4]。视频分析系统还通过模型选择、边缘调度和帧过滤减少不必要的推理或传输 [5-9]。本文不切分单个检测器，也不依赖视频冗余，而是让端侧检测器先产生可直接使用的结果，再选择部分图像交给独立云端检测器复核。

Adaptive Feeding 使用 SVM 判断图像应由快速还是高精度检测器处理 [10]。DCSB 使用检测语义特征识别难例，其后续版本增加带宽自适应和区域采样 [11,12]。ORIC/MORIC 通过 contextual reward 估计一张图对整体检测 mAP 的边际贡献 [13]。OffloaD 以检测失败为卸载信号 [20]。EdgeBoost 则在分类任务中校准端侧置信度或 margin 后上传低置信样本 [21]。这些工作说明 Confidence 是强基线，也说明端侧不确定性与云端相对收益并不总是等价。

### 2.2 ROI、压缩与任务感知传输

RegionFilter 使用区域信息减少资源受限节点的视频传输 [14]；EdgeDuet 将低分辨率端侧检测与高分辨率 tile 处理结合，面向小目标场景降低带宽 [15]；AccMPEG 根据图像区域对下游分析精度的影响分配编码质量 [16]。特征压缩和语义编码则通过传输中间表示降低通信 [22,23]。与这些方法不同，本文不改变检测器或训练新的编码器，而是在统一缓存上比较 Original、Q90 和 ROI，并把实际请求字节直接纳入配置选择。

### 2.3 成本敏感路由与风险校准

Learning to defer 将模型与下游专家之间的分配建模为条件风险最小化问题 [17]，后续工作讨论了校准式 defer 和多专家路由 [18,19]。现代神经网络置信度可能与真实风险不一致 [25]，因此仅使用一个未校准阈值并不能保证部署成本或精度。Learn then Test 等研究提供了在独立数据上筛选候选并检查风险的通用思路 [26]。

本文与上述工作的差异在于：研究对象是具有非加性 Official AP 的目标检测；系统同时决定样本与像素载荷；优化变量受到调用量或实际字节预算约束；最终配置必须经过独立门禁并在 final 前冻结。本文不宣称 bootstrap 门禁提供形式化安全保证，而将其作为防止校准点估计过拟合的经验风险控制。

## 3 问题定义与成本感知原理

### 3.1 端云检测与动作空间

给定图像集合 \(\mathcal D=\{(I_i,Y_i)\}_{i=1}^{N}\)，端侧检测器 \(D_e\) 输出检测集合 \(E_i=D_e(I_i)\)。未上传时，系统直接输出 \(E_i\)。若图像被上传，载荷动作属于

\[
\mathcal A=\{\mathrm{Original},\mathrm{Q90},\mathrm{ROI}\}.
\]

云端检测器 \(D_c\) 对载荷执行复核，并经尺度恢复、坐标回映射和框融合得到 \(F_i(a)\)。令 \(c_i(a)\) 表示动作 \(a\) 的实际请求字节，边缘保留动作记为 \(a=\mathrm{Edge}\)，其请求成本为零。

### 3.2 样本价值与载荷成本

理想情况下，动作 \(a\) 对图像 \(i\) 的净价值可写为

\[
V_i(a;\lambda)
=
\mathbb E[\Delta U_i(a)\mid z_i]
-\lambda c_i(a),
\]

其中 \(\Delta U_i(a)\) 表示云端动作相对端侧输出的检测效用增量，\(z_i\) 是在线可观测的端侧检测、图像统计和载荷属性，\(\lambda\) 是预算对应的成本乘子。理想动作满足

\[
a_i^*=\arg\max_{a\in\{\mathrm{Edge}\}\cup\mathcal A}V_i(a;\lambda).
\]

该表达式说明，端侧不确定性、云端修复收益和单位字节收益是三个不同量。已有级联决策理论同样指出，置信度延期只在特定风险结构下充分 [24]。为明确本文采用成本感知决策的原因，给出如下命题。

**命题 1（Confidence 排序的充分条件）。** 设 \(h_i\) 为端侧不确定性，且样本效用可加。若最优云动作相对端侧的期望收益可写为关于 \(h_i\) 的单调不减函数 \(g(h_i)\)，所有样本的最优载荷成本均为同一常数 \(c\)，则在 Fixed-call 预算下按 \(h_i\) 从高到低选择前 \(K\) 个样本是最优的。反之，只要存在两个样本 \(i,j\) 满足 \(h_i>h_j\)，但 \(g_i-\lambda c_i<g_j-\lambda c_j\)，就存在一个使该样本对跨越选择边界的预算 \(K\)，此时 Confidence 次序不是最优的。

证明可由交换论证得到：在前述充分条件下，任意包含较低 \(h_j\) 而未包含较高 \(h_i\) 的可行集合，都可用 \(i\) 替换 \(j\) 且不降低总效用；反例条件则直接使该替换降低净效用。实际检测中，类别、目标尺寸、拥挤程度、端云错误互补性以及 Original/Q90/ROI 字节均会改变 \(g_i\) 或 \(c_i\)，因此不能预设 Confidence 为充分统计量。

命题 1 只用于解释可加代理效用下的排序条件。Official AP 不是逐图可加指标，因而本文不把单图代理收益直接称为 AP 增量。ORIC/MORIC 风格的 contextual reward 可用于分析可学习 headroom，但当前多动作 Action-ORIC 仅作为 oracle 诊断，不进入三数据集主方法。主控制器直接使用统一评估器在校准集合上计算配置级 Official AP，从而避免把未验证的逐图代理写成最终优化目标。

### 3.3 配置与预算模式

样本排序器 \(r\) 根据 \(z_i\) 产生分数 \(s_i^{(r)}\)，载荷策略 \(\pi\) 为已接纳样本选择动作。完整配置定义为

\[
\theta=(r,\pi,m,B,\tau,\lambda),
\]

其中 \(m\) 是预算模式，\(B\) 是预算，\(\tau\) 是动作门限，\(\lambda\) 是 Fixed-byte 调度乘子。

Fixed-call 严格接纳得分最高的 \(K=\lfloor BN\rfloor\) 张图像：

\[
\mathcal U_{\mathrm{call}}
=\operatorname{TopK}(\{s_i^{(r)}\}_{i=1}^{N},K).
\]

Online 使用校准数据上冻结的阈值 \(t_{r,B}\)：

\[
\mathcal U_{\mathrm{online}}=\{i:s_i^{(r)}\ge t_{r,B}\}.
\]

Fixed-byte 约束所有接纳载荷的总请求字节：

\[
\sum_{i\in\mathcal U}c_i(a_i)\le C_B.
\]

三种模式分别衡量严格调用预算、阈值部署漂移和总字节容量，不在同一上传率假设下混合比较。

## 4 成本感知端云协同方法

### 4.1 总体架构

系统包含在线执行和离线校准两个阶段。在线阶段依次执行端侧检测、样本评分、预算接纳、载荷选择、云端复核与结果融合。离线阶段利用冻结检测缓存和独立数据角色完成候选校准、风险门禁和配置冻结。

\[
I_i\rightarrow D_e\rightarrow (E_i,z_i)
\rightarrow r\rightarrow \mathrm{Scheduler}
\rightarrow \pi\rightarrow \{\mathrm{Original,Q90,ROI}\}
\rightarrow D_c\rightarrow \mathrm{Fusion}.
\]

**[图 1：成本感知端云协同架构。图中区分在线样本/载荷决策与离线 selector/gate/final 数据流。]**

未接纳图像立即使用端侧结果。接纳图像按冻结动作发送至云端。离线控制器不修改端侧或云端检测器权重，各候选共享相同预测、载荷和评估器缓存。

### 4.2 样本卸载信号

框架不绑定单一排序器。公共特征由 17D 端侧检测与图像统计量，以及仅在 fit 集拟合的 32D FPN-PCA 表示组成。Confidence 使用空检优先和最低部署框置信度的补值；ClassGainPrior 估计类别条件云端收益；M2E 及其 residual 变体学习端云收益代理；SemanticRank 使用公共语义特征进行成对排序；MORIC-style 和 GEC-style 分别提供 contextual-reward 相关与组合基线。这些排序器回答“哪些图像优先获得云端资源”，但不单独构成本文贡献。

自动配置只允许选择八个确定性排序器。Random 使用 50 个种子评估随机波动，不参与部署选择。Legacy-Frozen-32D 依赖历史检测器、特征或数据划分，只作为兼容性附录，不进入严格三数据集主矩阵。

### 4.3 自适应载荷策略

三种载荷策略共享同一 Original/Q90/ROI 缓存。

**SharedAction。** 使用 fit-only 的共享动作头选择 Original、Q90 或 ROI，作为参考策略。Q90 分支采用类别平衡逻辑回归，ROI 分支采用安全性 GBDT；阈值仅在 selector 上校准。动作头不读取 Ranker 标识或排序分位，确保样本排序与载荷选择可独立比较。

**CostEligible-AllROI。** 当 ROI 可用且请求字节小于 SharedAction 基础动作时切换为 ROI，否则保留基础动作。该方法是简单成本门禁，用于检查复杂动作模型是否具有必要性。

**GBDT-Hierarchical。** 先得到 SharedAction 基础动作，再使用 fit-only GBDT 估计 ROI 是否在降低字节的同时保持可接受质量。训练正例要求 ROI 的逐图质量代理不低于基础动作减 0.02，且 ROI 请求字节更少。输入包括 49D 公共特征、Q90/Original 字节比、ROI/Original 字节比、ROI 面积比及载荷可用性，共 54D。模型采用平衡样本权重、150 次提升迭代、0.05 学习率、最大深度 3、最小叶节点样本数 20 和 \(L_2=1.0\)，随机种子固定为 20260722。阈值按数据集和预算模式校准，并在全部非随机排序器间共享。

若 GBDT-Hierarchical 与 CostEligible-AllROI 产生相同动作，二者按等价结果报告，不以模型复杂度作为贡献。

### 4.4 配置级成本优化

令 \(Q(\theta;\mathcal S)\) 表示配置 \(\theta\) 在集合 \(\mathcal S\) 上的 Official mAP50-95，令

\[
C(\theta;\mathcal S)
=\frac{1}{|\mathcal S|}
\sum_{i\in\mathcal S}\frac{c_i(a_i)}{1024}
\]

表示按全部输入图像均摊的请求 KB/image。每个数据集、预算模式和工作点先在 SharedAction 配置中选择校准 AP 最高者作为参考 \(\theta_{ref}\)。精度容忍量固定为 \(\epsilon=0.001\)。

对于 Fixed-call 和 Online，控制器求解

\[
\min_{\theta\in\Theta}C(\theta;\mathcal D^{sel})
\quad\mathrm{s.t.}\quad
Q(\theta;\mathcal D^{sel})
\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.
\]

对于 Fixed-byte，总请求容量已被预算限制，控制器求解

\[
\max_{\theta\in\Theta}Q(\theta;\mathcal D^{sel})
\quad\mathrm{s.t.}\quad
C(\theta;\mathcal D^{sel})\le C_B,
\quad
Q(\theta;\mathcal D^{sel})
\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon,
\]

Fixed-call 和 Online 在最低成本并列时依次选择 AP 更高、策略更简单和预注册 Ranker 顺序更靠前的配置；Fixed-byte 在最高 AP 并列时依次选择请求字节更低、策略更简单和预注册 Ranker 顺序更靠前的配置。动作策略简单性顺序冻结为 SharedAction、GBDT-Hierarchical、CostEligible-AllROI。

### 4.5 独立风险门禁与冻结回退

临时候选 \(\theta_{prov}\) 与参考配置在独立 gate 集上进行 200 次配对图像级 bootstrap [27]。第 \(b\) 次重复同时对两种配置使用相同图像索引，计算

\[
\Delta Q^{(b)}=Q(\theta_{prov};\mathcal D_b^{gate})
-Q(\theta_{ref};\mathcal D_b^{gate}),
\]

\[
\Delta C^{(b)}=C(\theta_{prov};\mathcal D_b^{gate})
-C(\theta_{ref};\mathcal D_b^{gate}).
\]

本文取 200 个重采样差值的 2.5% 和 97.5% 分位数作为逐工作点经验区间。Fixed-call 和 Online 要求 AP 差区间下界不低于 \(-\epsilon\)，且请求 KB 差区间上界不高于 0；Fixed-byte 只执行 AP 门禁。冻结配置为

\[
\theta_{frozen}=
\begin{cases}
\theta_{prov}, & \text{门禁通过},\\
\theta_{ref}, & \text{门禁失败}.
\end{cases}
\]

门禁失败后不尝试第二候选。配置、阈值和来源哈希写入冻结文件后，final 只评估一次。final oracle 仅用于计算诊断 regret，不得改变冻结配置。

**算法 1 成本感知配置校准与风险门禁**

```text
输入：selector 集 Dsel，gate 集 Dgate，候选集合 Θ，预算模式 m，容忍量 ε
输出：冻结配置 θfrozen

1: 在 SharedAction 候选中选择校准 AP 最高的参考 θref
2: 删除校准 AP < AP(θref)-ε 的候选
3: Fixed-call/Online 在合格候选中最小化请求 KB
4: Fixed-byte 在容量约束内最大化 Official AP
5: 按冻结并列规则得到唯一 θprov
6: 在 Dgate 上执行配对 bootstrap AP 门禁；必要时同时执行成本门禁
7: 门禁通过则冻结 θprov，否则冻结 θref
8: 记录选择、区间、回退状态和来源哈希
9: final 只评估 θfrozen；oracle 仅作诊断
```

### 4.6 在线开销

配置搜索只在离线执行。在线系统只运行一个冻结排序器、一个载荷策略和一次预算判断。Confidence 与线性排序器的开销相对端侧检测可忽略；Current-Controlled-32D 按现有协议计入 0.08 ms/image 和 32 个排序权重；GBDT 动作头开销单独报告，不混入未实测的端到端时延。

## 5 实验设置

### 5.1 数据集与数据角色

本文使用 COCO [28]、ENA24 和 Oregon Critters。COCO 提供 80 类通用场景对照。ENA24 是 22 类目标存在型野生动物检测集，当前划分不含空图，平均约 1.1 个标注框/图，因此不能据此评价空帧过滤。Oregon Critters 是 45 类野生动物相机图像集，原图分辨率和字节量较高，用于检验类别长尾与背景域变化下的成本收益。每个数据集按用途划分为拟合集（fit）、选择集（selector）、门禁集（gate）和最终评测集（final）：fit 用于训练排序器与动作模型，selector 用于配置选择，gate 只用于门禁，final 不参与拟合、选择、阈值校准或回退。

| 数据集 | fit | selector | gate/audit | final | 评测边界 |
|---|---:|---:|---:|---:|---|
| COCO | 9,678 | 571 | 630 | 5,000 | val2017 为固定 diagnostic |
| ENA24 | 1,400 | 约 200 | 约 200 | 1,758 | selector/gate 近重复组隔离 |
| Oregon | 9,923 | 4,961 | 9,923 | 4,962 | gate 为独立 detector-val；final 为 post-hoc exploratory reuse |

ENA24 的 400 张 calibration 图像先计算 16x16 灰度 dHash，将汉明距离不超过 6 的样本连接为近重复分量，再以固定种子将完整分量分配至 selector 和 gate。由于缺少可靠事件、相机和位置元数据，该划分只能称为 near-duplicate-group-disjoint。Oregon 采用冻结的 random-image 主协议；相机背景可能跨 detector/selector/final 角色出现，这一边界在第 7 节讨论。

### 5.2 检测器与载荷

统一矩阵使用冻结 YOLO26n 级端侧检测器与 YOLO26x 级云端检测器。YOLO26 为 2026 年预印本，本文不把检测器结构作为创新 [29]。三个数据集分别沿用其 canonical 权重、预测缓存、类别映射和数据划分，不跨数据集共享归一化参数或动作阈值。

Original、640-Q90 和 ROI 载荷均预先生成并记录实际字节。Q90 将长边缩放至 640 像素，使用 JPEG quality 90 和 4:2:0 色度抽样。ROI 由端侧置信度不低于 0.25 的候选框生成，按候选面积动态扩框，裁剪总面积上限为原图的 65%，并以 Q90 编码；每张图最多触发一次云调用，云端框经坐标回映射后以 IoU 0.5 的 MAUR 规则与端侧结果融合。ROI 不可用时按冻结策略回退至 Q90 或 Original。所有候选共享相同检测缓存、载荷缓存和 Official AP 评估器。论文级实验不重新训练检测器，也不根据 final 修改 ROI 或 Q90 动作。

### 5.3 对比方法与工作点

系统参考包括 Edge-only、Cloud-only Original、Always Q90 和 Always ROI fallback。样本排序基线包括 Random、Confidence、ClassGainPrior、M2E、Confidence-Residual M2E、SemanticRank、MORIC-style、GEC-style 和 Current-Controlled-32D。载荷策略为 SharedAction、CostEligible-AllROI 和 GBDT-Hierarchical。

注册工作点如下：

- Fixed-call：10%、25%、40%；
- Online：nominal 10%、25%、40%；
- Fixed-byte：Original 字节预算的 5%、10%、20%、40%。

主方法是经过 selector 选择和独立 gate 门禁后的成本感知冻结配置。主要对照为 Confidence+SharedAction、最佳 SharedAction reference、未经过 gate 的 provisional 配置、全数据集共享单一配置和 diagnostic final oracle。

### 5.4 评测指标

检测指标以 conf=0.001 的 COCO-style Official mAP50-95 为主，并报告 mAP50 和 mAP75。成本指标包括实际请求 KB/image、响应 KB/image、上传率、Original/Q90/ROI 比例、均摊 GFLOPs、选择器参数量和选择器时延。

Online 同时报告 nominal 与实际上传率。Fixed-byte 同时报告实际字节和接纳样本比例。模型化 E2E 在 1/5/10/20 Mbps 下单独标记为 Estimated E2E，不能称为真实端到端测量。

### 5.5 系统上下界

| 数据集 | 方法 | Official mAP50-95 | 请求 KB/image |
|---|---|---:|---:|
| COCO | Edge-only | 0.407904 | 0.00 |
| COCO | Cloud-only Original | 0.574243 | 159.12 |
| COCO | Always Q90 | 0.569870 | 84.08 |
| COCO | Always ROI fallback | 0.559398 | 71.16 |
| ENA24 | Edge-only | 0.712660 | 0.00 |
| ENA24 | Cloud-only Original | 0.741774 | 433.29 |
| ENA24 | Always Q90 | 0.736860 | 94.34 |
| ENA24 | Always ROI fallback | 0.737294 | 89.91 |
| Oregon | Edge-only | 0.454815 | 0.00 |
| Oregon | Cloud-only Original | 0.499095 | 1595.49 |
| Oregon | Always Q90 | 0.471120 | 97.98 |
| Oregon | Always ROI fallback | 0.459940 | 96.03 |

三数据集的端云精度差和图像字节分布显著不同。特别是 Oregon 的 Original 成本远高于另两个数据集，而 Q90 损失也更明显，因此成本必须基于实际载荷计算，不能仅比较上传比例。

## 6 结果与分析

### 6.1 总体精度成本结果

主结果比较冻结成本感知配置与对应 SharedAction reference。Fixed-call 和 Online 关注 AP 容忍内的请求字节降低；Fixed-byte 关注同字节容量下的 AP 提升。

| 数据集 | 工作点数 | Gate 通过/回退 | 平均 AP 差 | 最差 AP 差 | 请求 KB 相对变化 | 平均 final-oracle regret |
|---|---:|---:|---:|---:|---:|---:|
| 【冻结 final 完成后由汇总脚本生成 COCO、ENA24、Oregon 三行】 | — | — | — | — | — | — |

表中 AP 差和请求 KB 变化均以对应工作点的 SharedAction reference 为基准；regret 定义为同一 24 候选集合中 diagnostic final oracle 的 AP 减去冻结配置 AP。【待冻结 final 后填入数值与一次性结果解释，不从 final 逐点挑选方法。】

**[图 2：三个数据集的 Official AP-请求 KB Pareto 曲线。颜色表示载荷策略，形状表示预算模式。]**

### 6.2 样本排序信号

完整矩阵用于回答学习型检测收益排序是否提供超出 Confidence 的收益。比较保持载荷策略、预算模式和工作点相同，分别报告 AP 排名、请求 KB 排名及 Pareto 支配次数。Random 50 seeds 用于判断观察到的排序收益是否超过随机选择波动；32D/49D、Residual M2E 和组合排序器的完整工作点放入补充材料。正文只报告 Confidence 与冻结控制器实际选择信号的差异，不按学习器逐项展开。【待填：三数据集同预算胜负数及跨域平均排名。】

**[图 3：排序信号在三个数据集和三类预算模式下的 AP 排名热力图。Random 以均值及 95% 区间单独表示。]**

### 6.3 载荷策略与预算语义

Fixed-call 中上传样本数不随载荷改变，ROI 或 Q90 的作用主要表现为单次请求成本与裁剪/压缩损失。Fixed-byte 中更小载荷可以释放容量并接纳更多样本，因此可能产生不同最优策略。Online 的 AP 必须与实际上传率共同解释。正文按数据集和预算模式报告冻结配置的 Original/Q90/ROI 比例、实际上传率及相对请求成本；逐工作点动作计数放入补充材料。【待填：三类预算下最主要的载荷变化及其 AP/KB 结果。】

若 CostEligible-AllROI 与 GBDT-Hierarchical 动作逐图一致，结果按简单成本门禁解释。若 ROI 在 selector 通过而在 gate 失败，则保留失败与回退，不重新降低阈值。

**[图 4：冻结配置的 Original/Q90/ROI 动作组成；Online 同时标注 nominal 与实际上传率。]**

### 6.4 风险门禁与配置迁移

门禁分析分别报告 provisional 候选数、AP 门禁失败数、成本门禁失败数、冻结回退数以及 final 超出 AP 容忍的工作点数。selector 与 gate 的差异用于判断点估计选优能否迁移，final 只用于评估冻结配置。这些统计与总体结果合并报告于表 3，完整置信区间列入补充材料。

200 次 bootstrap 提供逐点经验区间，不构成多工作点同时保证。COCO 与 ENA24 gate 规模较小，Oregon 图像级重采样又未处理事件内相关性，因此区间、点估计和回退状态必须同时解释。

### 6.5 消融实验

消融包括：只使用 Confidence；只改变载荷而固定样本集合；去除成本门禁；去除独立 gate；所有数据集共享一个 Ranker+policy；以及 final oracle。该设计分别检验样本排序、载荷选择、成本约束和风险门禁的贡献。

| 消融 | Fixed-call/Online 平均 AP 差 | Fixed-call/Online 请求 KB 变化 | Fixed-byte 平均 AP 差 | Final 越界工作点 | 作用 |
|---|---:|---:|---:|---:|---|
| 【冻结 final 完成后由汇总脚本生成六项消融】 | — | — | — | — | Confidence、载荷、门禁、跨域固定配置、完整方法与诊断上界 |

## 7 系统验证与局限

### 7.1 原型可实现性

现有 T41 原型已经完成 JPEG/HTTP、有界异步队列和轻量线性决策器。历史组件实验中，近似特征线性决策器平均耗时 0.676 ms；T41 YOLO26s NNA 成功样本平均 585.971 ms，Q70 JPEG 编码平均 82.069 ms。旁路本地检测的异步调度循环约为 6.6-6.8 FPS。这些结果说明选择器计算不是当前主要瓶颈，端侧推理、编码、网络和云服务决定实际吞吐。

上述结果来自历史组件协议，不等同于本文三数据集冻结控制器的完整部署。各组件延迟不能直接相加为同一帧 E2E，板端特征也尚未完成与 Python 的逐项数值等价测试。因此，本文仅把 T41 结果作为可实现性证据。

### 7.2 有效性威胁

第一，COCO val2017 和 ENA24 final 是固定 diagnostic；Oregon final 已被 canonical 实验访问，本文属于 post-hoc exploratory reuse。严格确认性结论仍需要新的隐藏测试或预注册外部数据。

第二，ENA24 只实现近重复组隔离，缺少事件、相机和位置元数据。Oregon 采用 random-image 主协议，同一相机背景可能跨数据角色出现；其图像级 bootstrap 也忽略事件簇相关性。因此 Oregon 结果不能解释为跨相机或跨地点泛化。

第三，排序学习器使用可学习代理目标，而 Official AP 不是逐图可加指标。多动作 Action-ORIC 目前只是 COCO oracle/headroom 诊断，不能作为三数据集已部署方法。

第四，ROI 依赖端侧候选框，无法修复完全漏检；当前动作空间也未覆盖 tile packing、任务感知编码和特征压缩。

第五，Estimated E2E 不是实测。新的成本感知控制器尚未完成冻结异机链路的端到端时延、吞吐和能耗测量，因此本文不声称真实实时性能优势。

第六，端侧和云端使用同系列检测器，错误可能高度相关。更异构或专用的云端检测器可能改变可路由收益及最佳载荷策略。

## 8 结论

本文提出一种面向端云目标检测的成本感知样本卸载与自适应载荷选择框架。该框架将样本排序、Original/Q90/ROI 载荷动作和 Fixed-call/Online/Fixed-byte 预算调度解耦，并在统一检测缓存上通过配置级 Official AP 与实际请求字节进行校准。对于调用量和在线模式，控制器在 AP 容忍约束下最小化请求成本；对于字节预算，控制器在容量约束内最大化检测精度。独立配对 bootstrap 门禁用于冻结候选或回退参考配置，final 不参与选择或修复。

三个数据集的完整结果将量化【待填：平均通信节省、最差 AP 差、门禁回退率和 oracle regret】。本文不把某个排序学习器描述为跨域通用解，而将排序信号视为成本感知系统的可替换组成。核心结论是，端侧不确定性、云端修复收益和载荷成本必须在明确预算语义下共同评估，才能形成可审计的端云检测配置。

## Statements and Declarations

### Funding

待作者补充。

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

待作者按 CRediT 角色补充。

### Data availability

COCO 可从官方渠道获取。ENA24 与 Oregon 的公开性、许可证和下载地址待作者确认。本文生成的角色清单、冻结配置、来源哈希和匿名化汇总计划随论文或补充材料发布。

### Code availability

代码仓库和永久链接待补。复现材料应包含协议文件、缓存清单、选择器、风险门禁、汇总脚本和环境信息。

### Ethics and consent

本文不进行身份识别。涉及人物、私有相机或敏感野生动物位置的数据，在公开前应满足相应知情同意、匿名化、机构伦理和数据许可要求。

## 参考文献

[1] Kang, Y., Hauswald, J., Gao, C., et al.: Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. ASPLOS, 615-629 (2017). https://doi.org/10.1145/3037697.3037698

[2] Matsubara, Y., Levorato, M., Restuccia, F.: Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges. ACM Computing Surveys 55(5), 1-30 (2023). https://doi.org/10.1145/3527155

[3] Zhou, Z., Chen, X., Li, E., et al.: Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. Proceedings of the IEEE 107(8), 1738-1762 (2019). https://doi.org/10.1109/JPROC.2019.2918951

[4] Teerapittayanon, S., McDanel, B., Kung, H.T.: BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. ICPR, 2464-2469 (2016). https://doi.org/10.1109/ICPR.2016.7900006

[5] Xu, R., Razavi, S., Zheng, R.: Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques. IEEE Communications Surveys & Tutorials 25(4), 2951-2982 (2023). https://doi.org/10.1109/COMST.2023.3323091

[6] Ran, X., Chen, H., Zhu, X., Liu, Z., Chen, J.: DeepDecision: A Mobile Deep Learning Framework for Edge Video Analytics. IEEE INFOCOM, 1421-1429 (2018). https://doi.org/10.1109/INFOCOM.2018.8485905

[7] Yi, S., Hao, Z., Zhang, Q., et al.: LAVEA: Latency-Aware Video Analytics on Edge Computing Platform. ACM/IEEE SEC, 1-13 (2017). https://doi.org/10.1145/3132211.3134459

[8] Li, Y., Padmanabhan, A., Zhao, P., et al.: Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics. ACM SIGCOMM, 359-376 (2020). https://doi.org/10.1145/3387514.3405874

[9] Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M.: NoScope: Optimizing Neural Network Queries over Video at Scale. PVLDB 10(11), 1586-1597 (2017). https://doi.org/10.14778/3137628.3137664

[10] Zhou, H.-Y., Gao, B.-B., Wu, J.: Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors. ICCV, 3505-3513 (2017). https://doi.org/10.1109/ICCV.2017.379

[11] Cao, Z., Cheng, Y., Zhou, Z., et al.: Edge-Cloud Collaborated Object Detection via Difficult-Case Discriminator. ICDCS (2023); preprint version: https://arxiv.org/abs/2108.12858

[12] Cao, Z., Cheng, Y., Zhou, Z., et al.: Edge-Cloud Collaborated Object Detection via Bandwidth Adaptive Difficult-Case Discriminator. IEEE Transactions on Mobile Computing 24(2), 1181-1196 (2025). https://doi.org/10.1109/TMC.2024.3474743

[13] Qiu, J., Wang, R., Hu, B., Guerin, R., Lu, C.: Optimizing Edge Offloading Decisions for Object Detection. ACM/IEEE Symposium on Edge Computing (2024). https://arxiv.org/abs/2410.18919

[14] Chen, T., Bu, Y., Zeng, Y., Xie, L., Lu, S.: RegionFilter: Region-Aware Video Filtering Mechanism on Resource-Constrained Edge Nodes. Computer Networks 251, 110624 (2024). https://doi.org/10.1016/j.comnet.2024.110624

[15] Yang, Z., Wang, X., Wu, J., et al.: EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision. IEEE/ACM Transactions on Networking 31(4), 1765-1778 (2023). https://doi.org/10.1109/TNET.2022.3223412

[16] Du, K., Pervaiz, A., Yuan, X., et al.: AccMPEG: Optimizing Video Encoding for Accurate Video Analytics. Proceedings of Machine Learning and Systems 4 (2022). https://proceedings.mlsys.org/paper_files/paper/2022/hash/853f7b3615411c82a2ae439ab8c4c96e-Abstract.html

[17] Mozannar, H., Sontag, D.: Consistent Estimators for Learning to Defer to an Expert. ICML, PMLR 119, 7076-7087 (2020). https://proceedings.mlr.press/v119/mozannar20b.html

[18] Verma, R., Nalisnick, E.: Calibrated Learning to Defer with One-vs-All Classifiers. ICML, PMLR 162 (2022). https://proceedings.mlr.press/v162/verma22c.html

[19] Mao, A., Mohri, M., Zhong, Y.: Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees for Learning to Defer. ICML, PMLR 267, 43035-43066 (2025). https://proceedings.mlr.press/v267/mao25c.html

[20] Gomez, C., Patkhullaev, D., Hernandez, A.C.: OffloaD: Detection Failure-Based Scheduler for Offloading Object Detection. ACM Transactions on Embedded Computing Systems 23(5), 1-26 (2024). https://doi.org/10.1145/3677321

[21] Said, N., Landsiedel, O.: EdgeBoost: Confidence Boosting for Resource Constrained Inference via Selective Offloading. Computer Networks 269, 111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437

[22] Choi, H., Bajic, I.V.: Deep Feature Compression for Collaborative Object Detection. ICIP (2018). https://arxiv.org/abs/1802.03931

[23] Elgamal, T., Shi, S., Gupta, V., Jana, R., Nahrstedt, K.: SiEVE: Semantically Encoded Video Analytics on Edge and Cloud. ICDCS (2020). https://doi.org/10.1109/ICDCS47774.2020.00182

[24] Jitkrittum, W., Gupta, N., Menon, A.K., Narasimhan, H., Rawat, A.S., Kumar, S.: When Does Confidence-Based Cascade Deferral Suffice? NeurIPS (2023). https://arxiv.org/abs/2307.02764

[25] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of Modern Neural Networks. ICML, PMLR 70, 1321-1330 (2017). https://proceedings.mlr.press/v70/guo17a.html

[26] Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I., Lei, L.: Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. The Annals of Applied Statistics 19(2) (2025). https://doi.org/10.1214/24-AOAS1998

[27] Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman & Hall/CRC (1993).

[28] Lin, T.-Y., Maire, M., Belongie, S., et al.: Microsoft COCO: Common Objects in Context. ECCV, 740-755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

[29] Jocher, G., Qiu, J., Liu, M., et al.: Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748
